# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise production MoE token collectives with a CPU Gloo transport."""

import importlib.util
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    forward_context,
    get_execution_context,
    in_acl_graph,
)


def _load_prepare(
    ops: ModuleType, context: SimpleNamespace, prepare_class: str = "AllGatherPrepareAndFinalize"
) -> tuple:
    root = Path(__file__).parents[2] / "xllm/python/models/kimi_k3_moe.py"
    modules = {}
    for name in ("xllm", "xllm.python", "xllm.python.models", "xllm.python.model_executor"):
        modules[name] = ModuleType(name)
        modules[name].__path__ = []
    modules["xllm.python"].distributed = ops
    ctx = ModuleType("xllm.python.model_executor.forward_context")
    ctx.get_forward_context = lambda: context
    ctx.get_execution_context = get_execution_context

    def graph_query() -> bool:
        # Use the owner predicate with this transport test's scoped metadata.
        with forward_context(context):
            return in_acl_graph()

    ctx.in_acl_graph = graph_query
    modules[ctx.__name__] = ctx
    name = "xllm.python.models.kimi_k3_moe"
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(name, root)
        mod = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: mod}):
            spec.loader.exec_module(mod)
        return mod.MoEParallelConfig, getattr(mod, prepare_class)


def _worker(rank: int, rendezvous: str, output: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=30)
    )
    tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    dp_rank = rank // 2
    ops = ModuleType("ops")

    def group(group_name: str) -> object:
        if group_name == "tp":
            return tp_groups[dp_rank]
        if group_name == "dp":
            return dp_groups[rank % 2]
        return dist.group.WORLD

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        pieces = [torch.empty_like(value) for _ in range(world_size)]
        dist.all_gather(pieces, value.contiguous(), group=group(group_name))
        return torch.cat(pieces, dim=dim)

    def all_reduce(value: torch.Tensor, group_name: str = "tp") -> None:
        dist.all_reduce(value, group=group(group_name))

    def reduce_scatter(value: torch.Tensor, world_size: int, group_name: str) -> torch.Tensor:
        reduced = value.clone()
        dist.all_reduce(reduced, group=group(group_name))
        return reduced.chunk(world_size, dim=0)[dist.get_rank(group(group_name))].contiguous()

    ops.all_gather = all_gather
    ops.all_reduce_ = all_reduce
    ops.reduce_scatter = reduce_scatter
    context = SimpleNamespace(acl_graph=None, execution_state=None, metadata=None)
    config_type, prepare_type = _load_prepare(ops, context)
    config = config_type(
        tp_size=1,
        tp_rank=0,
        ep_size=4,
        ep_rank=rank,
        dp_size=2,
        dp_rank=dp_rank,
        input_tp_size=2,
        input_tp_rank=rank % 2,
    )
    cases = [
        ("decode_different_dp", ([2.0], [5.0]), (1, 1), (False, False)),
        ("decode_idle_dp", ([2.0], [77.0]), (1, 0), (False, False)),
        ("prefill_uneven", ([2.0, 3.0, 4.0], [5.0]), (3, 1), (True, True)),
        ("prefill_idle_dp", ([2.0, 3.0, 4.0], [77.0]), (3, 0), (True, False)),
        ("graph_decode", ([2.0], [5.0]), (1, 1), (False, False)),
        ("graph_idle_dp", ([2.0], [77.0]), (1, 0), (False, False)),
        ("graph_batch3", ([2.0, 3.0, 4.0], [5.0, 6.0, 7.0]), (3, 3), (False, False)),
        ("warmup_decode", ([2.0], [5.0]), (1, 1), (False, False)),
        ("warmup_idle_dp", ([2.0], [77.0]), (1, 0), (False, False)),
        ("warmup_batch3", ([2.0, 3.0, 4.0], [5.0, 6.0, 7.0]), (3, 3), (False, False)),
    ]
    report = {}
    try:
        for name, values, counts, phases in cases:
            context.acl_graph = object() if name.startswith("graph_") else None
            context.execution_state = (
                AclGraphExecutionState(persistent_buffers={}) if name.startswith(("graph_", "warmup_")) else None
            )
            hidden = torch.tensor(values[dp_rank]).reshape(-1, 1)
            context.metadata = SimpleNamespace(
                dp_execution_token_counts=counts, is_prefill=phases[dp_rank], is_chunked_prefill=False
            )
            method = prepare_type(config)
            prepared = method.prepare(hidden, hidden.repeat(1, 4))
            if context.acl_graph is not None or context.execution_state is not None:
                assert prepared.hidden_states.shape[0] == 2 * hidden.shape[0]
            # Four distinct linear experts, each with uniform routing weight.
            partial = prepared.hidden_states * (rank + 1) / 4
            actual = method.finalize(partial, True, prepared.padded_hidden_states_shape)
            valid_count = counts[dp_rank]
            expected = hidden[:valid_count] * 2.5
            actual = actual[:valid_count]
            error = (actual - expected).abs()
            report[name] = dict(
                actual=actual.flatten().tolist(),
                expected=expected.flatten().tolist(),
                max_abs=error.max().item() if error.numel() else 0,
            )
            # Stop collectively after the first mismatch to avoid a later hang
            # obscuring an already-proven decode regression.
            failed = torch.tensor([int(not torch.equal(actual, expected))])
            dist.all_reduce(failed)
            if failed.item():
                break
    finally:
        Path(output, f"rank{rank}.json").write_text(json.dumps(report))
        dist.destroy_process_group()


def test_global_dp_tokens_stay_separate(tmp_path: Path) -> None:
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=4, join=True)
    reports = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(4)]
    print(json.dumps(reports, indent=2))
    for rank, report in enumerate(reports):
        for name, result in report.items():
            assert result["max_abs"] == 0, f"rank={rank} case={name}: {result}"
        assert len(report) == 10
