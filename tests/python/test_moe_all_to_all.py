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

"""Verify production all-to-all dispatch and combine over a Gloo transport."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xllm.python.model_executor.forward_context import get_execution_context, in_acl_graph


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=30)
    )
    root = Path(__file__).parents[2] / "xllm/python"
    collectives = _load_module("_moe_all_to_all_collectives", root / "distributed/collectives.py")
    collectives._groups[("moe_ep", "cpu")] = dist.group.WORLD
    packages = {}
    for name in ("xllm", "xllm.python", "xllm.python.models", "xllm.python.model_executor"):
        packages[name] = ModuleType(name)
        packages[name].__path__ = []
    packages["xllm.python"].distributed = collectives
    context = ModuleType("xllm.python.model_executor.forward_context")
    context.get_forward_context = lambda: SimpleNamespace(acl_graph=None, execution_state=None, metadata=None)
    context.in_acl_graph = in_acl_graph
    context.get_execution_context = get_execution_context
    packages[context.__name__] = context
    try:
        with patch.dict(sys.modules, packages):
            dispatchers = _load_module("xllm.python.models.kimi_k3_moe", root / "models/kimi_k3_moe.py")
            types = dispatchers
            config = types.MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=4, ep_rank=rank)
            dispatcher = dispatchers.AllToAllTokenDispatcher(config, num_experts=896, quantized=False)
            for counts in ((3, 1, 0, 2), (1, 0, 0, 0), (0, 0, 0, 0)):
                tokens = counts[rank]
                hidden = (torch.arange(tokens * 8, dtype=torch.float32).reshape(tokens, 8) + rank + 1) / 8
                # No token routes to owner 3, including when it has source tokens.
                ids = torch.tensor([0, 223, 224, 448], dtype=torch.int32).expand(tokens, -1).clone()
                weights = torch.tensor([0.1, 0.2, 0.3, 0.4]).expand(tokens, -1).clone()
                routing = types.MoERoutingResult(ids, weights)
                dispatched = dispatcher.token_dispatch(types.MoETokenDispatchInput(hidden, routing))
                local_ids = torch.repeat_interleave(torch.arange(rank * 224, (rank + 1) * 224), dispatched.group_list)
                if rank == 3:
                    assert dispatched.hidden_states.shape[0] == 0
                expert_output = dispatched.hidden_states * (local_ids.remainder(5) + 1).unsqueeze(-1)
                actual = dispatcher.token_combine(expert_output, dispatched.combine_metadata)
                expected = hidden * (weights * (ids.remainder(5) + 1)).sum(-1, keepdim=True)
                assert actual.shape == expected.shape
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_all_to_all_keeps_uneven_and_empty_token_owners(tmp_path: Path) -> None:
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)
