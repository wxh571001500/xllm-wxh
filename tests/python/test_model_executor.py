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

"""Unit tests for xllm.python.model_executor.executor.

Tests the device-conditional backend dispatch, ModelExecutor construction
validation, and execution routing — using CPU mocks so no GPU/NPU required.
"""

from __future__ import annotations

import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from importlib import import_module
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# conftest.py stands in for xllm.python, whose import would bind the active
# platform's kernel package and reach for operators from the C++ binary.
from xllm.python.attention.backend import (  # noqa: E402
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    normalize_layer_caches,
)
from xllm.python.attention.sfa_dcp_backend import dcp_layer_options  # noqa: E402
from xllm.python.layers.attention import Attention  # noqa: E402
from xllm.python.layers.npu.mega_moe_metadata import MegaMoeMetadata  # noqa: E402
from xllm.python.layers.npu.mega_moe_metadata_builder import (  # noqa: E402
    TokenOwnerMegaMoeMetadataBuilder,
)
from xllm.python.model_executor.executor import (  # noqa: E402
    ModelExecutor,
    _create_attention_backend,
    _resolve_graph_backend,
)
from xllm.python.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    LayerLoadContext,
    forward_context,
    get_execution_context,
    get_forward_context,
    record_layer_event,
    wait_for_layer_load,
)
from xllm.python.model_executor.input_batch import InputBatch  # noqa: E402
from xllm.python.model_executor.runners.decode_acl_graph import (  # noqa: E402
    DecodeAclGraphRunner,
)
from xllm.python.model_executor.runners.decode_cuda_graph import (  # noqa: E402
    DecodeCudaGraphRunner,
    _decode_bucket,
    _decode_graph_buckets,
)
from xllm.python.model_executor.runners.eager import EagerRunner  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubAttentionBackend(AttentionBackend):
    """Minimal backend that records calls for assertion."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self._kv_caches: list[LayerCache] = []
        self._prepared = False

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        self._prepared = True

    def execute(self, q, k, v, layer) -> torch.Tensor:
        return q

    @property
    def num_kv_blocks(self) -> int:
        return 0

    @property
    def page_size(self) -> int:
        return 1


class _PagedStubAttentionBackend(StubAttentionBackend):
    @property
    def page_size(self) -> int:
        return 4


class _MlaStubAttentionBackend(StubAttentionBackend):
    @property
    def is_mla(self) -> bool:
        return True


def _make_attention_layer(
    num_heads=8,
    num_kv_heads=2,
    head_dim=64,
    scale=0.125,
    sliding_window=0,
    layer_id=0,
) -> Attention:
    return Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        sliding_window=sliding_window,
        layer_id=layer_id,
    )


class _FakeModel(nn.Module):
    """Model with configurable number of uniform Attention layers."""

    def __init__(self, num_layers: int = 2, device: str = "cpu", **attn_kwargs):
        super().__init__()
        self.model = nn.Linear(1, 1)  # execution_model placeholder
        self.layers = nn.ModuleList([_make_attention_layer(layer_id=i, **attn_kwargs) for i in range(num_layers)])
        self._param = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, input_ids, positions):
        return input_ids


class _FakeModelHeterogeneous(nn.Module):
    """Model with non-uniform Attention layers (should fail validation)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.attn1 = _make_attention_layer(num_heads=8, layer_id=0)
        self.attn2 = _make_attention_layer(num_heads=4, layer_id=1)
        self._param = nn.Parameter(torch.zeros(1))


class _FakeModelNoAttention(nn.Module):
    """Model without any Attention layers."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self._param = nn.Parameter(torch.zeros(1))


class _FailingLayerSynchronizer:
    def record_event(self, layer_id: int) -> bool:
        return False


def _make_input_batch(
    num_scheduled_tokens: tuple[int, ...],
    *,
    token_capacity: int | None = None,
    num_draft_tokens_per_req: tuple[int, ...] | None = None,
) -> InputBatch:
    num_tokens = sum(num_scheduled_tokens)
    query_start_loc = [0]
    for count in num_scheduled_tokens:
        query_start_loc.append(query_start_loc[-1] + count)
    metadata = SimpleNamespace(
        num_reqs=len(num_scheduled_tokens),
        num_tokens=num_tokens,
        num_scheduled_tokens=list(num_scheduled_tokens),
        num_computed_tokens=[0] * len(num_scheduled_tokens),
        num_draft_tokens=sum(num_draft_tokens_per_req or ()),
        num_draft_tokens_per_req=(None if num_draft_tokens_per_req is None else list(num_draft_tokens_per_req)),
        query_start_loc=query_start_loc,
        is_prefilling=[0] * len(num_scheduled_tokens),
    )
    batch = InputBatch.from_runtime(
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.arange(num_tokens, dtype=torch.int32),
        metadata,
        is_dummy=False,
    )
    if token_capacity is None:
        return batch
    return batch.bind_graph_inputs(
        torch.zeros(token_capacity, dtype=torch.int32),
        torch.zeros(token_capacity, dtype=torch.int32),
        torch.arange(token_capacity) >= num_tokens,
    )


def test_attention_backend_defaults_to_non_mla() -> None:
    assert StubAttentionBackend().is_mla is False


def test_record_layer_event_propagates_record_failure() -> None:
    context = ForwardContext(
        attention_backend=StubAttentionBackend(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        layer_synchronizer=_FailingLayerSynchronizer(),
    )

    with (
        forward_context(context),
        pytest.raises(RuntimeError, match="failed to record layer completion event for layer 3"),
    ):
        record_layer_event(3)


# ---------------------------------------------------------------------------
# Tests: graph backend resolution
# ---------------------------------------------------------------------------


class _HostLoadSynchronizer:
    def __init__(self, count: int, failed_event: int | None = None) -> None:
        self.count = count
        self.failed_event = failed_event
        self.calls: list[int] = []

    def size(self) -> int:
        return self.count

    def synchronize_layer(self, event_index: int) -> bool:
        self.calls.append(event_index)
        return event_index != self.failed_event


def test_host_load_groups_wait_once_and_reset_between_forwards() -> None:
    synchronizer = _HostLoadSynchronizer(3)
    for _ in range(2):
        context = LayerLoadContext(synchronizer, layers_per_event=2)
        # A layerwise partition may start in the middle of an event group.
        for layer_id in (1, 2, 3, 4, 4, 5):
            context.synchronize_layer(layer_id)
        context.synchronize_all_layers()
    assert synchronizer.calls == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("layers_per_event", [0, -1])
def test_host_load_rejects_invalid_group_size(layers_per_event: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        LayerLoadContext(_HostLoadSynchronizer(1), layers_per_event)


@pytest.mark.parametrize("layer_id", [-1, 2])
def test_host_load_rejects_invalid_layer_index(layer_id: int) -> None:
    synchronizer = _HostLoadSynchronizer(1)
    context = LayerLoadContext(synchronizer, layers_per_event=2)
    with pytest.raises(ValueError):
        context.synchronize_layer(layer_id)
    assert synchronizer.calls == []


def test_host_load_failure_does_not_mark_event_complete() -> None:
    synchronizer = _HostLoadSynchronizer(1, failed_event=0)
    context = LayerLoadContext(synchronizer)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Host cache load event 0"):
            context.synchronize_layer(0)
    assert synchronizer.calls == [0, 0]


@pytest.mark.parametrize(
    "module_name,class_name,num_inputs",
    [
        ("layers.attention", "Attention", 3),
        ("models.glm5_next", "Glm5NextMlaAttention", 3),
        ("models.glm5_next", "Glm5NextKdaAttention", 2),
        ("models.glm5_2", "Glm52MLAAttention", 3),
        ("models.deepseek_v32", "DeepseekV3MLAAttention", 5),
        ("models.deepseek_v4", "DeepseekV4Attention", 3),
    ],
)
def test_attention_aborts_before_any_projection_or_cache_access(
    module_name: str, class_name: str, num_inputs: int
) -> None:
    attention_class = getattr(import_module(f"xllm.python.{module_name}"), class_name)
    attention = attention_class.__new__(attention_class)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    # No weights or cache state are initialized: the failed load must abort
    # before the attention body can touch either of them.
    context = ForwardContext(
        StubAttentionBackend(),
        torch.device("cpu"),
        MagicMock(),
        [],
        layer_load_context=LayerLoadContext(_HostLoadSynchronizer(1, failed_event=0)),
    )
    with forward_context(context), pytest.raises(RuntimeError, match="Host cache load event 0"):
        attention(*[torch.zeros(1) for _ in range(num_inputs)])


def test_attention_wait_is_noop_without_host_restore() -> None:
    wait_for_layer_load(0)
    context = ForwardContext(StubAttentionBackend(), torch.device("cpu"), MagicMock(), [])
    with forward_context(context):
        tensor = torch.zeros(1)
        assert _make_attention_layer(layer_id=0)(tensor, tensor, tensor) is tensor


def test_attention_blocks_until_its_host_copy_completes() -> None:
    waiting = [Event(), Event()]
    ready = [Event(), Event()]
    computed = [Event(), Event()]

    class AsyncLoads(_HostLoadSynchronizer):
        def synchronize_layer(self, event_index: int) -> bool:
            self.calls.append(event_index)
            waiting[event_index].set()
            return ready[event_index].wait(timeout=5)

    class Backend(StubAttentionBackend):
        def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
            assert ready[layer.layer_id].is_set()
            computed[layer.layer_id].set()
            return q

    synchronizer = AsyncLoads(2)
    context = ForwardContext(
        Backend(),
        torch.device("cpu"),
        MagicMock(),
        [],
        layer_load_context=LayerLoadContext(synchronizer),
    )

    def forward() -> None:
        with forward_context(context):
            tensor = torch.zeros(1)
            for layer_id in range(2):
                _make_attention_layer(layer_id=layer_id)(tensor, tensor, tensor)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(forward)
        try:
            assert waiting[0].wait(timeout=5)
            assert not computed[0].is_set()
            ready[0].set()
            assert computed[0].wait(timeout=5)
            assert waiting[1].wait(timeout=5)
            assert not computed[1].is_set()
        finally:
            for event in ready:
                event.set()
        future.result(timeout=5)
    assert computed[1].is_set()
    assert synchronizer.calls == [0, 1]


class TestNpuGraphBackendResolution:
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_enable_graph_selects_aclgraph_on_npu(self, _mock_is_npu):
        config = {"enable_graph": True, "python_graph_backend": "off"}
        assert _resolve_graph_backend(config) == "aclgraph"


# ---------------------------------------------------------------------------
# Tests: _create_attention_backend dispatch
# ---------------------------------------------------------------------------


class TestCreateAttentionBackend:
    @pytest.mark.parametrize("is_ascend_a3", [False, True])
    @pytest.mark.parametrize("proposal_width", [0, 8])
    def test_dflash2_xfia_requires_a3(self, is_ascend_a3: bool, proposal_width: int) -> None:
        native_query = MagicMock(return_value=is_ascend_a3)
        runtime = types.ModuleType("xllm_runtime")
        runtime.is_ascend_a3 = native_query
        with (
            patch.dict(sys.modules, {"xllm_runtime": runtime}),
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
            patch(
                "xllm.python.model_executor.executor.current_platform.get_ascend_soc_generation",
                return_value="a2" if is_ascend_a3 else "a3",
            ) as python_soc_query,
            patch("xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend", StubAttentionBackend),
        ):
            backend = _create_attention_backend(
                _make_attention_layer(num_kv_heads=1, head_dim=128),
                torch.device("npu"),
                torch.bfloat16,
                {"dflash2_block_size": proposal_width, "dflash_xfia_enabled": True},
            )
        assert backend.init_kwargs["use_xfia_decode"] == (is_ascend_a3 and proposal_width > 0)
        assert backend.init_kwargs["xfia_query_width"] == max(proposal_width, 1)
        python_soc_query.assert_not_called()
        if proposal_width > 0:
            native_query.assert_called_once_with()
        else:
            native_query.assert_not_called()

    def test_dflash2_xfia_stays_off_without_prelaunch_gate(self) -> None:
        native_query = MagicMock(return_value=True)
        runtime = types.ModuleType("xllm_runtime")
        runtime.is_ascend_a3 = native_query
        with (
            patch.dict(sys.modules, {"xllm_runtime": runtime}),
            patch("xllm.python.model_executor.executor.current_platform.is_npu", return_value=True),
            patch("xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend", StubAttentionBackend),
        ):
            backend = _create_attention_backend(
                _make_attention_layer(num_kv_heads=1, head_dim=128),
                torch.device("npu"),
                torch.bfloat16,
                {"dflash2_block_size": 8, "dflash_xfia_enabled": False},
            )
        assert backend.init_kwargs["use_xfia_decode"] is False
        native_query.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_creates_dsa_backend(self, _mock_is_npu):
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.dsa_attention")
        module.DsaAttentionBackend = StubAttentionBackend
        config = {
            "model_type": "deepseek_v4",
            "compress_ratios": [1, 4, 128],
            "num_hidden_layers": 3,
            "window_size": 128,
            "index_topk": 512,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "qk_rope_head_dim": 64,
        }
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("npu"), torch.bfloat16, config)

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["attn_head_dim"] == 512
        assert backend.init_kwargs["dspark_block_size"] == 0
        assert backend.init_kwargs["dspark_use_native_sas"] is False
        assert backend.init_kwargs["n_layers"] == 3
        assert backend.init_kwargs["compress_ratios"] == [1, 4, 128]

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_deepseek_v4_dspark_passes_native_sas_config(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(head_dim=512)
        module = types.ModuleType("xllm.python.attention.dsa_attention")
        module.DsaAttentionBackend = StubAttentionBackend
        config = {
            "model_type": "deepseek_v4_dspark",
            "compress_ratios": [1, 1, 1],
            "n_layers": 3,
            "window_size": 128,
            "dspark_block_size": 5,
            "dspark_use_native_sas": True,
        }

        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("npu"), torch.bfloat16, config)

        assert backend.init_kwargs["dspark_block_size"] == 5
        assert backend.init_kwargs["dspark_use_native_sas"] is True

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    @pytest.mark.parametrize("num_decoding_tokens", [1, 4])
    def test_npu_device_creates_npu_backend(self, _mock_is_npu: MagicMock, num_decoding_tokens: int) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        backend = _create_attention_backend(
            attn,
            torch.device("npu"),
            torch.float16,
            {"enable_mla": False},
            num_decoding_tokens=num_decoding_tokens,
        )
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 256
        assert backend.init_kwargs["is_mla"] is False
        assert backend.init_kwargs["num_decoding_tokens"] == num_decoding_tokens

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_prefill_cp_uses_npu_backend_with_dcp_group(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = MagicMock()
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.float16,
                {"cp_size": 4, "enable_mla": True},
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["is_mla"] is True
        sfa_module.SfaDcpAttentionBackend.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_dflash2_draft_cp1_falls_through_to_plain_paged_backend(self, _mock_is_npu: MagicMock) -> None:
        """M11.3 admission pin companion: the DFlash2 GQA draft at cp_size == 1
        with a live DCP group is NOT sharded -- it keeps a full replica of its
        own pool, read through the expanded block table -- so it must skip the
        SfaDcp branch entirely (executor's ``is_draft_engine`` fall-through)
        and land on the plain paged backend with no DCP group attached. A
        dense (non-draft) target in the same shape is refused instead."""
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = MagicMock()
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch(
                "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
                StubAttentionBackend,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {"cp_size": 1, "enable_mla": False, "is_draft_engine": True},
            )

        assert isinstance(backend, StubAttentionBackend)
        # No DCP group and no MLA: the draft's pool stays an unsharded
        # full replica.
        assert "dcp_group" not in backend.init_kwargs
        assert backend.init_kwargs["is_mla"] is False
        sfa_module.SfaDcpAttentionBackend.assert_not_called()

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
            pytest.raises(NotImplementedError, match="Dense DCP attention is not supported"),
        ):
            _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {"cp_size": 1, "enable_mla": False, "is_draft_engine": False},
            )

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_uses_dense_dcp_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        dense_module = types.ModuleType("xllm.python.attention.dense_dcp_backend")
        dense_module.DenseDcpAttentionBackend = StubAttentionBackend

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {dense_module.__name__: dense_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "enable_mla": False,
                    "tp_size": 8,
                    "n_kv_heads": 2,
                },
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 128

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_rejects_dcp_outside_kv_replica_group(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 4

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            pytest.raises(ValueError, match="replicated KV-head group"),
        ):
            _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "enable_mla": False,
                    "tp_size": 8,
                    "n_kv_heads": 4,
                },
            )

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_dcp_is_compatible_with_expert_parallelism(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        dense_module = types.ModuleType("xllm.python.attention.dense_dcp_backend")
        dense_module.DenseDcpAttentionBackend = StubAttentionBackend

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {dense_module.__name__: dense_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "tp_size": 8,
                    "n_kv_heads": 2,
                    "ep_size": 8,
                    "enable_eplb": True,
                },
            )

        assert isinstance(backend, StubAttentionBackend)

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_qwen35_dcp_rejects_disaggregated_pd(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=128)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            pytest.raises(NotImplementedError, match="disaggregated P/D"),
        ):
            _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.bfloat16,
                {
                    "model_type": "qwen3_5_moe_text",
                    "cp_size": 1,
                    "tp_size": 8,
                    "n_kv_heads": 2,
                    "enable_disagg_pd": True,
                },
            )

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_decode_cp1_uses_sfa_dcp_backend(self, _mock_is_npu: MagicMock) -> None:
        attn = _make_attention_layer(num_kv_heads=1, head_dim=256)
        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = StubAttentionBackend
        sfa_module.dcp_layer_options = MagicMock(return_value=512)

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            backend = _create_attention_backend(
                attn,
                torch.device("npu"),
                torch.float16,
                {"cp_size": 1, "enable_mla": True},
                max_num_reqs=3,
                num_decoding_tokens=4,
            )

        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["index_topk"] == 512
        assert backend.init_kwargs["max_num_reqs"] == 3
        assert backend.init_kwargs["num_decoding_tokens"] == 4

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=False,
    )
    @patch(
        "xllm.python.model_executor.executor.current_platform.is_cuda",
        return_value=True,
    )
    def test_cuda_device_creates_flashinfer_backend(self, _mock_is_cuda, _mock_is_npu):
        attn = _make_attention_layer()
        module = types.ModuleType("xllm.python.attention.flashinfer")
        module.FlashInferBackend = StubAttentionBackend
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = _create_attention_backend(attn, torch.device("cuda"), torch.float16)
        assert isinstance(backend, StubAttentionBackend)


# ---------------------------------------------------------------------------
# Tests: ModelExecutor construction
# ---------------------------------------------------------------------------


class TestModelExecutorConstruction:
    @pytest.mark.parametrize("graph_backend", ["off", "aclgraph"])
    @pytest.mark.parametrize(
        "explicit_width,config_depth,expected_width",
        [(1, 0, 1), (1, 3, 4), (4, 1, 4)],
    )
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_decoding_width_reaches_backend_and_graph_runner(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
        graph_backend: str,
        explicit_width: int,
        config_depth: int,
        expected_width: int,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        ModelExecutor(
            _FakeModel(num_layers=1),
            {
                "python_graph_backend": graph_backend,
                "num_speculative_tokens": config_depth,
                "max_position_embeddings": 128,
            },
            max_seqs_per_batch=4,
            num_decoding_tokens=explicit_width,
        )

        assert mock_create.call_args.kwargs["num_decoding_tokens"] == expected_width
        if graph_backend == "aclgraph":
            assert mock_graph_runner.call_args.kwargs["num_decoding_tokens"] == expected_width
        else:
            mock_graph_runner.assert_not_called()

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_valid_model_creates_executor(self, _mock_backend):
        model = _FakeModel(num_layers=3)
        config = {"python_graph_backend": "off"}
        executor = ModelExecutor(model, config, max_seqs_per_batch=4)

        assert executor._num_attention_layers == 3
        assert executor.decode_graph_runner is None
        assert executor.inductor_runner is None

    @patch("xllm.python.model_executor.executor.get_execution_metadata_builder_classes")
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_eager_binds_execution_metadata_builders(
        self,
        _mock_backend,
        mock_builder_classes,
    ) -> None:
        builder = MagicMock()
        builder_class = MagicMock()
        builder_class.from_config.return_value = builder
        mock_builder_classes.return_value = (builder_class,)
        config = {
            "enable_mega_moe": True,
            "model_type": "qwen3_5_moe",
            "python_graph_backend": "off",
        }

        executor = ModelExecutor(
            _FakeModel(num_layers=1),
            config,
            max_seqs_per_batch=4,
        )

        assert executor.eager_runner.execution_metadata_builders == (builder,)
        mock_builder_classes.assert_called_once_with("qwen3_5_moe")
        builder_class.from_config.assert_called_once_with(config)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_no_attention_layers_raises(self, _mock_backend):
        model = _FakeModelNoAttention()
        with pytest.raises(ValueError, match="does not contain an Attention layer"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_heterogeneous_attention_raises(self, _mock_backend):
        model = _FakeModelHeterogeneous()
        with pytest.raises(ValueError, match="identical attention configuration"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_glm_next_mixed_layers_build_backend_from_dsa(self, mock_backend: MagicMock) -> None:
        class _DsaAttention(Attention):
            is_glm_next_mla = True

        class _FakeGlmNext(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(1, 1)
                self.kda = _make_attention_layer(num_heads=16, num_kv_heads=16, head_dim=64, scale=0.125, layer_id=0)
                self.dsa = _DsaAttention(
                    num_heads=8,
                    num_kv_heads=1,
                    head_dim=512,
                    scale=0.044,
                    sliding_window=0,
                    layer_id=3,
                )
                self._param = nn.Parameter(torch.zeros(1))

        ModelExecutor(_FakeGlmNext(), {"python_graph_backend": "off"}, max_seqs_per_batch=4)
        first_attention = mock_backend.call_args.args[0]
        assert first_attention.num_heads == 8
        assert first_attention.num_kv_heads == 1
        assert first_attention.head_dim == 512

    @patch(
        "xllm.python.model_executor.executor.current_platform.is_npu",
        return_value=True,
    )
    def test_glm_next_decode_cp1_selects_sfa_dcp_backend_from_dsa_layer(
        self,
        _mock_is_npu: MagicMock,
    ) -> None:
        """M11.3 admission pin's safety half: the only backend selected for a
        glm5_next MLA instance at cp_size == 1 with a live DCP group is
        SfaDcpAttentionBackend.

        This is what makes the admitted cp1 + kv_split_size > 1 DECODE shape
        safe (validate_model_cp's cp1 early return never gates it): the
        backend localizes its own slots in prepare() and never writes through
        the global slot mapping. The pin covers the full selection chain --
        first_attention picked from the DSA layers (``is_glm_next_mla``), the
        DCP-group branch at cp_size == 1, and ``index_topk`` resolved from the
        DSA layer's cfg (2048 for GLM-5.3 Flash) through the real
        ``dcp_layer_options``. It also documents the num_heads convention:
        glm5_next passes the GLOBAL ``cfg.n_heads`` through the Attention base
        (unlike glm5_2's TP-local count), which is benign because nothing on
        the SfaDcp decode path consumes the backend's num_heads (see
        test_sfa_dcp.py's num_heads pins)."""

        class _DsaAttention(Attention):
            is_glm_next_mla = True

        class _KdaAttention(Attention):
            is_glm_next_kda = True

        class _FakeGlmNextDecode(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(1, 1)
                # KDA (linear attention) layers come first in glm5_next with a
                # different head config; the backend must be built from the
                # first DSA layer instead.
                self.kda = _KdaAttention(
                    num_heads=16,
                    num_kv_heads=16,
                    head_dim=64,
                    scale=0.125,
                    sliding_window=0,
                    layer_id=0,
                )
                self.dsa = _DsaAttention(
                    num_heads=64,
                    num_kv_heads=1,
                    head_dim=512,
                    scale=0.044,
                    sliding_window=0,
                    layer_id=3,
                )
                self.dsa.cfg = SimpleNamespace(index_topk=2048)
                self._param = nn.Parameter(torch.zeros(1))

        dcp_group = MagicMock()
        dcp_group.size.return_value = 2
        dcp_group.rank.return_value = 0
        sfa_module = types.ModuleType("xllm.python.attention.sfa_dcp_backend")
        sfa_module.SfaDcpAttentionBackend = StubAttentionBackend
        # The real option resolution, not a stub: index_topk must come from
        # the DSA layer's cfg.
        sfa_module.dcp_layer_options = dcp_layer_options

        with (
            patch(
                "xllm.python.model_executor.executor.distributed.dcp_group",
                return_value=dcp_group,
            ),
            patch.dict(sys.modules, {sfa_module.__name__: sfa_module}),
        ):
            executor = ModelExecutor(
                _FakeGlmNextDecode(),
                {
                    "model_type": "glm5_next",
                    "cp_size": 1,
                    "enable_mla": True,
                    "python_graph_backend": "off",
                },
                max_seqs_per_batch=4,
                num_decoding_tokens=1,
            )

        backend = executor.attention_backend
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["dcp_group"] is dcp_group
        assert backend.init_kwargs["index_topk"] == 2048
        # The GLOBAL head count flows in from the DSA layer's Attention base
        # (glm5_next convention); the SFA merge derives per-rank heads from
        # output shapes instead, so this value is inert on the decode path.
        assert backend.init_kwargs["num_heads"] == 64
        assert backend.init_kwargs["num_kv_heads"] == 1
        assert backend.init_kwargs["head_dim"] == 512
        assert backend.init_kwargs["max_num_reqs"] == 4
        assert backend.init_kwargs["num_decoding_tokens"] == 1

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_glm_next_kda_rejects_adaptive_speculative_decode(self, _mock_backend: MagicMock) -> None:
        class _KdaAttention(Attention):
            is_glm_next_kda = True

        model = _FakeModel(num_layers=0)
        model.kda = _KdaAttention(
            num_heads=16,
            num_kv_heads=16,
            head_dim=64,
            scale=0.125,
            sliding_window=0,
            layer_id=0,
        )

        with pytest.raises(ValueError, match="GLM5 KDA does not support adaptive speculative decode"):
            ModelExecutor(
                model,
                {
                    "python_graph_backend": "off",
                    "runtime_adaptive_speculative_decode_enabled": True,
                },
                max_seqs_per_batch=4,
            )

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_graph_backend_off_variants(self, _mock_backend):
        for off_value in ("off", "", "none", "0"):
            model = _FakeModel(num_layers=1)
            executor = ModelExecutor(model, {"python_graph_backend": off_value}, max_seqs_per_batch=4)
            assert executor.decode_graph_runner is None
            assert executor.inductor_runner is None

    @patch("xllm.python.model_executor.runners.decode_cuda_graph.DecodeCudaGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_data_parallel_cuda_graph_is_supported(self, mock_create, mock_graph_runner):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "dp_size": 2,
                "dp_rank": 1,
                "max_position_embeddings": 128,
                "python_graph_backend": "cudagraphs",
            },
            max_seqs_per_batch=4,
        )

        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            4,
            128,
            2,
            1,
        )

    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_data_parallel_rejects_unsupported_graph_backend(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        with pytest.raises(NotImplementedError, match="supports cudagraphs and aclgraph only"):
            ModelExecutor(
                model,
                {
                    "dp_size": 2,
                    "max_position_embeddings": 128,
                    "python_graph_backend": "inductor",
                },
                max_seqs_per_batch=4,
            )

    @pytest.mark.parametrize("is_spec_draft", [False, True])
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_capacity_respects_decode_batch_limit(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
        is_spec_draft: bool,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=256,
            num_decoding_tokens=4,
            acl_graph_decode_batch_size_limit=16,
            is_spec_draft=is_spec_draft,
        )

        # PR3 passes max_seqs_per_batch (sequences) directly as the runner's
        # max_batch capacity; the decode_batch_size_limit is enforced at
        # runtime by can_execute (bucket > limit falls back to eager), not by
        # folding it into max_batch at construction.
        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            256,
            128,
            1,
            0,
            16,
            num_decoding_tokens=4,
            enable_mega_moe_token_mask=False,
            is_spec_draft=is_spec_draft,
            draft_query_width=1,
        )

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_glm_mega_moe_enables_legacy_token_mask(
        self,
        mock_create,
        mock_graph_runner,
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "enable_mega_moe": True,
                "model_type": "glm_moe_dsa",
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=4,
        )

        assert mock_graph_runner.call_args.kwargs["enable_mega_moe_token_mask"] is True

    @pytest.mark.parametrize(
        ("model_type", "degree", "ep_size", "expected"),
        [
            ("glm5_next", 2, 8, True),
            ("glm5_next_text", 2, 2, True),
            ("glm5_next", 1, 8, False),
            ("glm5_next", 0, 1, False),
            ("qwen3_5_moe", 2, 8, False),
        ],
    )
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_glm_eplv2_uses_typed_metadata_builder_without_megamoe(
        self, mock_create, mock_graph_runner, model_type, degree, ep_size, expected
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(
            model,
            {
                "model_type": model_type,
                "expert_parallel_degree": degree,
                "ep_size": ep_size,
                "enable_mega_moe": False,
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=16,
            acl_graph_decode_batch_size_limit=16,
        )
        assert bool(executor._execution_metadata_builders) is expected
        assert bool(executor.eager_runner.execution_metadata_builders) is expected
        mock_graph_runner.return_value.bind_execution_metadata_builders.assert_called_once_with(
            executor.eager_runner.execution_metadata_builders
        )
        assert mock_graph_runner.call_args.kwargs["enable_mega_moe_token_mask"] is False

    @pytest.mark.parametrize("mode,expected_limit", [("auto", 24), ("alltoall", 0)])
    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_glm_eplv2_graph_uses_model_mc2_capacity(
        self, mock_create, mock_graph_runner, mode: str, expected_limit: int
    ):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        model.model._comm_policy = SimpleNamespace(mc2_capacity=3, mode=mode)
        ModelExecutor(
            model,
            {
                "model_type": "glm5_next",
                "expert_parallel_degree": 2,
                "tp_size": 8,
                "ep_size": 8,
                "num_speculative_tokens": 7,
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=128,
        )
        assert mock_graph_runner.call_args.kwargs["eplv2_graph_token_limit"] == expected_limit
        assert mock_graph_runner.call_args.kwargs["num_decoding_tokens"] == 8

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor.get_execution_metadata_builder_classes")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_qwen_mega_moe_keeps_legacy_mask_with_metadata_builder(
        self,
        mock_create,
        mock_builder_classes,
        mock_graph_runner,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        builder = MagicMock()
        builder.metadata_type = MegaMoeMetadata
        builder.bind_layer_caches = MagicMock()
        builder_class = MagicMock()
        builder_class.from_config.return_value = builder
        mock_builder_classes.return_value = (builder_class,)

        model = _FakeModel(num_layers=1)
        config = {
            "enable_mega_moe": True,
            "model_type": "qwen3_5_moe",
            "max_position_embeddings": 128,
            "python_graph_backend": "aclgraph",
        }
        ModelExecutor(
            model,
            config,
            max_seqs_per_batch=4,
        )

        assert mock_graph_runner.call_args.kwargs["enable_mega_moe_token_mask"] is True
        builders = mock_graph_runner.return_value.bind_execution_metadata_builders.call_args.args[0]
        assert builders == (builder,)
        mock_builder_classes.assert_called_once_with("qwen3_5_moe")
        builder_class.from_config.assert_called_once_with(config)

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_data_parallel_passes_global_token_budget(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "dp_size": 16,
                "dp_rank": 0,
                "enable_mega_moe": True,
                "model_type": "glm_moe_dsa",
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
            },
            max_seqs_per_batch=17,
            num_decoding_tokens=4,
        )

        # PR3 passes max_seqs_per_batch (sequences) directly as the runner's
        # max_batch capacity; the runner derives its local DP capacity with
        # ceil-div internally.
        assert mock_create.call_args.args[4] == 17
        mock_graph_runner.assert_called_once_with(
            model.model,
            mock_create.return_value,
            torch.device("cpu"),
            17,
            128,
            16,
            0,
            None,
            num_decoding_tokens=4,
            enable_mega_moe_token_mask=True,
            is_spec_draft=False,
            draft_query_width=1,
        )

    @patch("xllm.python.model_executor.runners.decode_acl_graph.DecodeAclGraphRunner")
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_acl_graph_forwards_nondefault_draft_query_width(
        self,
        mock_create: MagicMock,
        mock_graph_runner: MagicMock,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)

        ModelExecutor(
            model,
            {
                "max_position_embeddings": 128,
                "python_graph_backend": "aclgraph",
                "draft_query_width": 8,
            },
            max_seqs_per_batch=8,
            is_spec_draft=True,
        )

        assert mock_graph_runner.call_args.kwargs["draft_query_width"] == 8


class TestDecodeCudaGraphDataParallelKeys:
    @staticmethod
    def _runner(dp_rank: int = 0) -> DecodeCudaGraphRunner:
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.max_batch = 16
        runner.dp_size = 2
        runner.dp_rank = dp_rank
        runner._graphs = {}
        return runner

    @staticmethod
    def _metadata(
        token_counts: list[int] | tuple[int, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
        )

    def test_graph_key_uses_global_max_data_parallel_bucket(self):
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)

        first = runner._graph_key(input_ids, self._metadata([3, 1]))
        second = runner._graph_key(input_ids, self._metadata([3, 2]))

        assert first == (4, (4, 4))
        assert second == first

    def test_data_parallel_warmup_uses_local_batch_capacity(self):
        assert _decode_graph_buckets(16, 2) == [1, 2, 4, 8]
        assert _decode_graph_buckets(20, 2) == [1, 2, 4, 8, 16]

    def test_single_rank_graph_key_reuses_padded_bucket(self):
        runner = self._runner()
        runner.dp_size = 1
        runner.dp_rank = 0

        first = runner._graph_key(torch.zeros(3, dtype=torch.int32), self._metadata([3]))
        second = runner._graph_key(torch.zeros(4, dtype=torch.int32), self._metadata([4]))

        assert first == (4, (4,))
        assert second == first

    def test_graph_key_accepts_dummy_execution_row_for_empty_data_parallel_rank(self):
        runner = self._runner(dp_rank=1)
        input_ids = torch.zeros(1, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata([5, 0])) == (
            8,
            (8, 8),
        )

    def test_graph_key_rejects_unbalanced_unwarmed_bucket(self):
        runner = self._runner()
        input_ids = torch.zeros(9, dtype=torch.int32)

        assert runner._graph_key(input_ids, self._metadata([9, 7])) is None

    def test_can_execute_requires_warmed_graph(self):
        runner = self._runner()
        input_ids = torch.zeros(3, dtype=torch.int32)
        metadata = self._metadata([3, 1])
        graph_key = runner._graph_key(input_ids, metadata)

        assert not runner.can_execute(input_ids, metadata)
        runner._graphs[graph_key] = object()
        assert runner.can_execute(input_ids, metadata)

    def test_execute_accepts_base_runner_context_keywords(self):
        runner = self._runner()
        runner._graph_key = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="decode batch exceeds CUDA graph capacity"):
            runner.execute(
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                self._metadata([1, 1]),
                layer_synchronizer=MagicMock(),
                eplb=MagicMock(),
            )

    @pytest.mark.parametrize("token_counts", ([3], [3, -1], [3, 2]))
    def test_graph_key_rejects_invalid_data_parallel_metadata(self, token_counts):
        runner = self._runner(dp_rank=1)
        input_ids = torch.zeros(1, dtype=torch.int32)

        with pytest.raises(RuntimeError):
            runner._graph_key(input_ids, self._metadata(token_counts))


# ---------------------------------------------------------------------------
# Tests: DecodeAclGraphRunner speculative metadata
# ---------------------------------------------------------------------------


class TestDecodeAclGraphSpeculativeMetadata:
    @staticmethod
    def _runner() -> DecodeAclGraphRunner:
        return DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=4,
            max_model_len=8,
        )

    @staticmethod
    def _metadata() -> SimpleNamespace:
        return SimpleNamespace(
            slot_mapping=torch.arange(4, dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
            paged_kv_indices=torch.tensor([10, 10, 20, 21, 20, 21], dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
            q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
            kv_cu_seq_lens=torch.tensor([0, 4, 12], dtype=torch.int32),
            kv_seq_lens_host=torch.tensor([4, 8], dtype=torch.int32),
            kv_seq_lens_host_values=[4, 8],
            block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            kv_seq_lens=torch.tensor([4, 8], dtype=torch.int32),
            q_seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            expanded_decode_metadata=SimpleNamespace(
                enabled=True,
                kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
                block_table=torch.tensor(
                    [[10, 11], [10, 11], [20, 21], [20, 21]],
                    dtype=torch.int32,
                ),
                paged_kv_indptr=torch.tensor([0, 1, 2, 4, 6], dtype=torch.int32),
                paged_kv_indices=torch.tensor([10, 10, 20, 21, 20, 21], dtype=torch.int32),
                paged_kv_last_page_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
                paged_attention_tiling_data=None,
                kv_seq_lens_host=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
                kv_seq_lens_host_values=[3, 4, 7, 8],
            ),
            is_prefill=False,
            is_chunked_prefill=True,
        )

    @staticmethod
    def _decode_metadata(
        batch_size: int,
        *,
        dp_execution_token_counts: tuple[int, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            slot_mapping=torch.arange(batch_size, dtype=torch.int32),
            paged_kv_indptr=torch.arange(batch_size + 1, dtype=torch.int32),
            paged_kv_indices=torch.arange(batch_size, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(batch_size, dtype=torch.int32),
            kv_cu_seq_lens=torch.arange(batch_size + 1, dtype=torch.int32),
            kv_seq_lens_host_values=[1] * batch_size,
            block_table=torch.arange(batch_size, dtype=torch.int32).view(-1, 1),
            kv_seq_lens=torch.ones(batch_size, dtype=torch.int32),
            expanded_decode_metadata=None,
            dp_execution_token_counts=dp_execution_token_counts,
            is_prefill=False,
            is_chunked_prefill=False,
        )

    def test_expanded_metadata_selects_matching_paged_kv_rows(self) -> None:
        runner = self._runner()
        (
            block_table,
            kv_seq_lens,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = runner._decode_metadata(self._metadata())

        assert block_table.tolist() == [
            [10, 11],
            [10, 11],
            [20, 21],
            [20, 21],
        ]
        assert kv_seq_lens.tolist() == [3, 4, 7, 8]
        assert paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    def test_decode_metadata_rebuilds_token_row_paging_metadata(self) -> None:
        metadata = self._metadata()
        metadata.expanded_decode_metadata = None
        metadata.is_chunked_prefill = False
        metadata.slot_mapping = torch.arange(4, dtype=torch.int32)
        metadata.block_table = torch.tensor(
            [[10, 11], [10, 11], [20, 21], [20, 21]],
            dtype=torch.int32,
        )
        metadata.kv_seq_lens = torch.tensor([3, 4, 7, 8], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [3, 4, 7, 8]
        # Mark this as a GENERIC (untyped) MTP verify batch: 4 token rows back
        # 2 logical sequences (width=2). _is_untyped_spec_verify then admits
        # it and _expanded_verify_view synthesizes the token-row expanded
        # view, rebuilding the per-seq paging metadata per row.
        metadata.linear_state_indices = torch.arange(2, dtype=torch.int32)
        metadata.q_cu_seq_lens = None
        metadata.paged_kv_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([10, 20, 21], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([4, 4], dtype=torch.int32)

        (
            _,
            _,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._runner()._decode_metadata(metadata)

        assert paged_kv_indptr.tolist() == [0, 1, 2, 4, 6]
        assert paged_kv_indices.tolist() == [10, 10, 20, 21, 20, 21]
        assert paged_kv_last_page_len.tolist() == [3, 4, 3, 4]

    def test_expanded_chunked_verify_can_use_decode_graph(self) -> None:
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)

        assert runner.can_execute(input_ids, self._metadata())

    def test_execute_accepts_base_runner_context_keywords(self) -> None:
        runner = self._runner()

        with pytest.raises(ValueError, match="decode batch exceeds ACL graph capacity"):
            runner.execute(
                torch.zeros(5, dtype=torch.int32),
                torch.zeros(5, dtype=torch.int32),
                self._decode_metadata(5),
                layer_synchronizer=MagicMock(),
                eplb=MagicMock(),
            )

    def test_decode_batch_limit_gates_dp_global_token_bucket(self) -> None:
        # PR3 folded the decode-batch-limit check into can_execute's DP branch
        # (runtime gate on the GLOBAL token bucket), removing the standalone
        # _decode_batch_sizes helper. A DP step whose global-max token bucket
        # exceeds decode_batch_size_limit falls back to eager; one within the
        # limit is admitted.
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            dp_size=2,
            dp_rank=0,
            decode_batch_size_limit=16,
            num_decoding_tokens=1,
        )
        with patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ):
            admitted = runner.can_execute(
                torch.zeros(8, dtype=torch.int32),
                SimpleNamespace(
                    is_prefill=False,
                    is_chunked_prefill=False,
                    dp_execution_token_counts=(8, 8),
                    dp_is_decode=(1, 1),
                ),
            )
            assert admitted
            rejected = runner.can_execute(
                torch.zeros(8, dtype=torch.int32),
                SimpleNamespace(
                    is_prefill=False,
                    is_chunked_prefill=False,
                    dp_execution_token_counts=(8, 64),
                    dp_is_decode=(1, 1),
                ),
            )
            assert not rejected

    def test_mtp3_batch_eight_uses_32_row_graph_bucket(self) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            decode_batch_size_limit=16,
            num_decoding_tokens=4,
        )
        # 8 logical sequences x 4 verify rows (mtp3) = 32 token rows. The
        # graph admits on the SEQUENCE count (8, bucket 8 <= limit 16) while
        # capturing/replaying at the full 32-row token bucket.
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=True,
            is_spec_verify=True,
            slot_mapping=torch.arange(32, dtype=torch.int32),
            kv_seq_lens=torch.full((8,), 8, dtype=torch.int32),
            block_table=torch.zeros((8, 2), dtype=torch.int32),
            linear_state_indices=torch.arange(8, dtype=torch.int32),
            q_cu_seq_lens=torch.arange(0, 33, 4, dtype=torch.int32),
            expanded_decode_metadata=SimpleNamespace(
                kv_seq_lens=torch.full((32,), 8, dtype=torch.int32),
                block_table=torch.zeros((32, 2), dtype=torch.int32),
                paged_kv_indptr=None,
                paged_kv_indices=None,
                paged_kv_last_page_len=None,
                paged_attention_tiling_data=None,
                kv_seq_lens_host=None,
                kv_seq_lens_host_values=None,
            ),
        )

        with patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ):
            assert runner.can_execute(
                torch.zeros(32, dtype=torch.int32),
                metadata,
            )

    def test_warmup_validates_capacity_without_capturing(self) -> None:
        # PR3 made warmup a capacity-validation no-op: graph capture is
        # deferred to execute's first decode of each (bucket, expanded,
        # embedding) key, so warmup must not allocate/capture on its own.
        runner = self._runner()
        input_ids = torch.arange(4, dtype=torch.int32)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = self._metadata()

        with patch.object(runner, "_allocate_entry") as allocate:
            runner.warmup(input_ids, positions, metadata)
            allocate.assert_not_called()
            runner.warmup(input_ids, positions, metadata)
            allocate.assert_not_called()

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "block_table",
                torch.arange(8, dtype=torch.int32),
                "block_table must be two-dimensional",
            ),
            (
                "kv_seq_lens",
                torch.tensor([3, 4, 7], dtype=torch.int32),
                "kv_seq_lens must contain one value per sequence",
            ),
            (
                "kv_seq_lens_host",
                torch.tensor([3, 4, 7], dtype=torch.int32),
                "kv_seq_lens_host must contain one value per sequence",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 1, 2, 4], dtype=torch.int32),
                "paged_kv_indptr must contain one offset per sequence",
            ),
            (
                "paged_kv_indices",
                torch.tensor([[10, 10], [20, 21]], dtype=torch.int32),
                "paged_kv_indices must be a non-empty flat page list",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 3], dtype=torch.int32),
                "paged_kv_last_page_len must contain one value per sequence",
            ),
        ],
    )
    def test_expanded_metadata_shape_mismatch_fails(
        self,
        field: str,
        value: torch.Tensor,
        message: str,
    ) -> None:
        metadata = self._metadata()
        setattr(metadata.expanded_decode_metadata, field, value)

        with pytest.raises(RuntimeError, match=message):
            self._runner()._decode_metadata(metadata)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "paged_kv_indptr",
                torch.tensor([1, 1, 2, 4, 6], dtype=torch.int32),
                "must start at zero",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 2, 1, 4, 6], dtype=torch.int32),
                "must be monotonic",
            ),
            (
                "paged_kv_indptr",
                torch.tensor([0, 1, 2, 4, 5], dtype=torch.int32),
                "terminal page offset must match page count",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 0, 4], dtype=torch.int32),
                "last-page lengths must be positive",
            ),
            (
                "paged_kv_last_page_len",
                torch.tensor([3, 4, 5, 4], dtype=torch.int32),
                "must not exceed block size",
            ),
        ],
    )
    def test_expanded_paged_metadata_invariant_fails(
        self,
        field: str,
        value: torch.Tensor,
        message: str,
    ) -> None:
        metadata = self._metadata()
        setattr(metadata.expanded_decode_metadata, field, value)

        with pytest.raises(RuntimeError, match=message):
            self._runner()._decode_metadata(metadata)

    def test_expanded_page_count_exceeding_capacity_fails(self) -> None:
        metadata = self._metadata()
        metadata.expanded_decode_metadata.kv_seq_lens_host_values = [3, 4, 7, 9]

        with pytest.raises(RuntimeError, match="exceeds block-table capacity"):
            self._runner()._decode_metadata(metadata)

    @pytest.mark.parametrize(
        ("input_ids", "slot_mapping", "message"),
        [
            (
                torch.arange(3, dtype=torch.int32),
                torch.arange(4, dtype=torch.int32),
                "input_ids must contain one token per sequence",
            ),
            (
                torch.arange(4, dtype=torch.int32),
                torch.arange(3, dtype=torch.int32),
                "slot_mapping must contain one slot per token",
            ),
        ],
    )
    def test_token_layout_mismatch_fails(
        self,
        input_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(RuntimeError, match=message):
            self._runner()._validate_decode_token_layout(
                input_ids,
                torch.arange(4, dtype=torch.int32),
                slot_mapping,
                sequence_count=4,
            )

    def test_replay_returns_static_output_view(self) -> None:
        runner = self._runner()
        refresh_dsa = MagicMock()
        runner.attention_backend.refresh_dsa_metadata_for_graph_replay = refresh_dsa
        batch_size = 3
        padded_batch_size = 4
        static_output = torch.arange(12).reshape(padded_batch_size, 3)
        graph = MagicMock()
        entry = SimpleNamespace(
            graph=graph,
            static_output=static_output,
            static_metadata=SimpleNamespace(),
            graph_tasks=[],
            execution_state=SimpleNamespace(persistent_buffers={}),
            execution_contexts={},
        )
        graph_key = runner._graph_key(
            padded_batch_size,
            is_expanded=False,
            input_embedding=None,
        )
        runner._graphs[graph_key] = entry

        replay_stream = MagicMock()
        update_stream = MagicMock()
        replay_done_event = MagicMock()
        current_stream = MagicMock()
        runner._stream = replay_stream
        runner._update_stream = update_stream
        runner._replay_done_event = replay_done_event
        fake_npu = SimpleNamespace(
            current_stream=MagicMock(return_value=current_stream),
            stream=MagicMock(return_value=nullcontext()),
        )
        metadata = SimpleNamespace(expanded_decode_metadata=None)

        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(runner, "_fill_entry"),
            patch.object(runner, "_update_graph_tasks"),
        ):
            output = runner.execute(
                torch.arange(batch_size, dtype=torch.int32),
                torch.arange(batch_size, dtype=torch.int32),
                metadata,
            )

        assert output.shape == (batch_size, 3)
        assert output.data_ptr() == static_output.data_ptr()
        output[0, 0] = -1
        assert static_output[0, 0].item() == -1
        replay_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(replay_stream)
        graph.replay.assert_called_once_with()
        refresh_dsa.assert_called_once_with(entry.static_metadata)

    def test_graph_token_mask_updates_in_place_for_reused_bucket(self) -> None:
        runner = self._runner()
        runner._enable_mega_moe_token_mask = True
        metadata = self._decode_metadata(4)
        entry = runner._allocate_entry(
            4,
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            metadata,
        )
        mask = entry.static_metadata.mega_moe_token_mask
        assert mask is not None
        data_ptr = mask.data_ptr()

        runner._fill_mega_moe_token_mask(entry, metadata, batch_size=4)
        assert mask.tolist() == [1, 1, 1, 1]

        runner._fill_mega_moe_token_mask(
            entry,
            self._decode_metadata(3),
            batch_size=3,
        )
        assert mask.data_ptr() == data_ptr
        assert mask.tolist() == [1, 1, 1, 0]

    def test_non_mega_moe_graph_does_not_allocate_token_mask(self) -> None:
        runner = self._runner()
        metadata = self._decode_metadata(4)
        entry = runner._allocate_entry(
            4,
            torch.arange(4, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            metadata,
        )

        assert entry.static_metadata.mega_moe_token_mask is None

    def test_mega_moe_buckets_decode_shapes(self) -> None:
        assert _decode_bucket(1) == 1
        assert _decode_bucket(2) == 2
        assert _decode_bucket(4) == 4
        assert _decode_bucket(5) == 8

    def test_mega_moe_dp_keeps_full_sequence_graph_capacity(self) -> None:
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=68,
            max_model_len=8,
            dp_size=16,
            enable_mega_moe_token_mask=True,
        )

        assert runner.max_batch == 68
        assert _decode_bucket(1) == 1
        assert _decode_bucket(5) == 8
        assert _decode_bucket(6) == 8

    def test_graph_token_mask_uses_real_data_parallel_counts(self) -> None:
        # dp_execution_token_counts floors an empty DP rank to 1 (its
        # worker-materialized dummy row), so a rank contributing exactly 1
        # here also covers the empty-rank case: MegaMoe dispatches that row
        # like a real token, and the mask marks it active accordingly.
        runner = DecodeAclGraphRunner(
            nn.Identity(),
            _PagedStubAttentionBackend(),
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            enable_mega_moe_token_mask=True,
        )
        metadata = self._decode_metadata(3, dp_execution_token_counts=(3, 1))
        entry = runner._allocate_entry(
            4,
            torch.arange(3, dtype=torch.int32),
            torch.arange(3, dtype=torch.int32),
            metadata,
        )

        runner._fill_mega_moe_token_mask(entry, metadata, batch_size=3)

        mask = entry.static_metadata.mega_moe_token_mask
        assert mask is not None
        assert mask.tolist() == [1, 1, 1, 0, 1, 0, 0, 0]

    def test_token_owner_metadata_builder_updates_graph_mask_in_place(self) -> None:
        builder = TokenOwnerMegaMoeMetadataBuilder(
            is_token_owner=True,
            dp_size=1,
            dp_rank=0,
            hidden_size=8,
            top_k=2,
            token_limit=4096,
        )
        input_batch = _make_input_batch((3,), token_capacity=4)
        runtime_metadata = SimpleNamespace()
        persistent_metadata = builder.allocate_persistent(input_batch, runtime_metadata)
        active_token_mask = persistent_metadata.active_token_mask
        data_ptr = active_token_mask.data_ptr()

        builder.update_persistent(
            persistent_metadata,
            input_batch,
            runtime_metadata,
        )

        assert active_token_mask.data_ptr() == data_ptr
        assert active_token_mask.tolist() == [1, 1, 1, 0]

    def test_token_non_owner_graph_metadata_contains_shared_dummy_inputs(self) -> None:
        builder = TokenOwnerMegaMoeMetadataBuilder(
            is_token_owner=False,
            dp_size=1,
            dp_rank=0,
            hidden_size=8,
            top_k=2,
            token_limit=4096,
        )
        input_batch = _make_input_batch((1,), token_capacity=4)
        runtime_metadata = SimpleNamespace()
        metadata = builder.allocate_persistent(input_batch, runtime_metadata)

        assert metadata.active_token_mask.tolist() == [1, 0, 0, 0]
        assert metadata.dummy_input is not None
        assert metadata.dummy_input.shape == (4, 8)
        assert metadata.dummy_input.dtype == torch.bfloat16
        assert metadata.dummy_topk_weights is not None
        assert metadata.dummy_topk_weights.shape == (4, 2)
        assert metadata.dummy_topk_weights.dtype == torch.float32
        assert metadata.dummy_topk_ids is not None
        assert metadata.dummy_topk_ids.shape == (4, 2)
        assert metadata.dummy_topk_ids.dtype == torch.int32

        data_ptrs = (
            metadata.dummy_input.data_ptr(),
            metadata.dummy_topk_weights.data_ptr(),
            metadata.dummy_topk_ids.data_ptr(),
        )
        builder.update_persistent(metadata, input_batch, runtime_metadata)
        assert data_ptrs == (
            metadata.dummy_input.data_ptr(),
            metadata.dummy_topk_weights.data_ptr(),
            metadata.dummy_topk_ids.data_ptr(),
        )

    def test_token_non_owner_skips_dummy_inputs_above_mega_moe_limit(self) -> None:
        builder = TokenOwnerMegaMoeMetadataBuilder(
            is_token_owner=False,
            dp_size=1,
            dp_rank=0,
            hidden_size=8,
            top_k=2,
            token_limit=2,
        )

        metadata = builder.build(
            _make_input_batch((3,)),
            SimpleNamespace(),
        )

        assert metadata.dummy_input is None
        assert metadata.dummy_topk_weights is None
        assert metadata.dummy_topk_ids is None

    def test_token_owner_builder_returns_same_metadata_type_for_eager_and_graph(self) -> None:
        builder = TokenOwnerMegaMoeMetadataBuilder(
            is_token_owner=True,
            dp_size=2,
            dp_rank=0,
            hidden_size=8,
            top_k=2,
            token_limit=4096,
        )
        eager_batch = _make_input_batch((2,))
        eager_metadata = builder.build(
            eager_batch,
            SimpleNamespace(dp_execution_token_counts=(2, 3)),
        )
        graph_metadata = builder.allocate_persistent(
            _make_input_batch((2,), token_capacity=4),
            SimpleNamespace(dp_execution_token_counts=(2, 3)),
        )

        assert isinstance(eager_metadata, MegaMoeMetadata)
        assert isinstance(graph_metadata, MegaMoeMetadata)
        assert eager_metadata.active_token_mask.tolist() == [1, 1, 0]

    @pytest.mark.parametrize(
        ("execution_token_counts", "error"),
        (
            ((), "expected 2 DP execution token counts"),
            ((2,), "expected 2 DP execution token counts"),
            ((1, 3), "does not match the local input"),
        ),
    )
    def test_token_owner_builder_rejects_invalid_dp_execution_counts(
        self,
        execution_token_counts: tuple[int, ...],
        error: str,
    ) -> None:
        builder = TokenOwnerMegaMoeMetadataBuilder(
            is_token_owner=True,
            dp_size=2,
            dp_rank=0,
            hidden_size=8,
            top_k=2,
            token_limit=4096,
        )

        with pytest.raises(RuntimeError, match=error):
            builder.build(
                _make_input_batch((2,)),
                SimpleNamespace(dp_execution_token_counts=execution_token_counts),
            )

    def test_token_owner_metadata_builder_uses_one_graph_mask_for_all_layers(self) -> None:
        builder = TokenOwnerMegaMoeMetadataBuilder(
            is_token_owner=True,
            dp_size=1,
            dp_rank=0,
            hidden_size=8,
            top_k=2,
            token_limit=4096,
        )
        input_batch = _make_input_batch((2,), token_capacity=4)
        runtime_metadata = SimpleNamespace()
        metadata = builder.allocate_persistent(input_batch, runtime_metadata)
        active_token_mask = metadata.active_token_mask

        builder.update_persistent(
            metadata,
            input_batch,
            runtime_metadata,
        )

        assert metadata.active_token_mask.data_ptr() == active_token_mask.data_ptr()
        assert metadata.active_token_mask.tolist() == [1, 1, 0, 0]


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.bind_kv_caches
# ---------------------------------------------------------------------------


class TestNormalizeLayerCaches:
    def test_legacy_five_slot_cache_keeps_generic_layout(self):
        tensors = tuple(torch.full((1,), value) for value in range(1, 6))

        cache = normalize_layer_caches([tensors])[0]

        assert cache.key is tensors[0]
        assert cache.value is tensors[1]
        assert cache.index is tensors[2]
        assert cache.conv is tensors[3]
        assert cache.ssm is tensors[4]
        assert cache.swa is None
        assert cache.compress_kv_state is None
        assert cache.compress_score_state is None
        assert cache.compress_index_kv_state is None
        assert cache.compress_index_score_state is None
        assert cache.indexer_scale is None

    def test_deepseek_v4_eleven_slot_cache_maps_all_slots(self):
        tensors = tuple(torch.full((1,), value) for value in range(1, 12))

        cache = normalize_layer_caches([tensors])[0]

        assert (
            cache.key,
            cache.value,
            cache.index,
            cache.conv,
            cache.ssm,
            cache.swa,
            cache.compress_kv_state,
            cache.compress_score_state,
            cache.compress_index_kv_state,
            cache.compress_index_score_state,
            cache.indexer_scale,
        ) == tensors

    def test_empty_deepseek_v4_slots_are_normalized_to_none(self):
        cache = normalize_layer_caches([(torch.ones(1), torch.ones(1), *(torch.empty(0),) * 9)])[0]

        assert cache.key is not None
        assert cache.value is not None
        assert cache.index is None
        assert cache.indexer_scale is None


class TestBindKvCaches:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_correct_count(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv, kv])
        assert len(backend._kv_caches) == 2

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_wrong_count_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        with pytest.raises(ValueError, match="layer count does not match"):
            executor.bind_kv_caches([kv])

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_idempotent(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])
        executor.bind_kv_caches([kv])  # should not raise or re-bind

    @patch("xllm.python.model_executor.executor.get_execution_metadata_builder_classes")
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_forwards_layer_caches_to_cache_aware_builder(
        self,
        mock_create,
        mock_builder_classes,
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        builder = MagicMock()
        builder.metadata_type = MegaMoeMetadata
        builder_class = MagicMock()
        builder_class.from_config.return_value = builder
        mock_builder_classes.return_value = (builder_class,)
        executor = ModelExecutor(
            _FakeModel(num_layers=1),
            {"model_type": "cache_aware", "python_graph_backend": "off"},
            max_seqs_per_batch=4,
        )
        kv = (torch.zeros(1), torch.zeros(1))

        executor.bind_kv_caches([kv])

        bound_caches = builder.bind_layer_caches.call_args.args[0]
        assert len(bound_caches) == 1
        assert bound_caches[0].key is kv[0]
        assert bound_caches[0].value is kv[1]


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.execute routing
# ---------------------------------------------------------------------------


def _make_eager_runner(*, is_mla: bool = True, restores_aux_hidden_under_cp: bool = False) -> EagerRunner:
    runner = object.__new__(EagerRunner)
    backend_type = _MlaStubAttentionBackend if is_mla else StubAttentionBackend
    runner.attention_backend = backend_type()
    runner.cp_size = 4
    runner.cp_rank = 2
    runner.device = torch.device("cpu")
    runner.layer_caches = []
    runner.model = MagicMock(return_value=torch.ones(2))
    # A plain MagicMock() auto-vivifies any attribute lookup (including
    # ``restores_aux_hidden_under_cp``) as a truthy MagicMock; pin it
    # explicitly so ``model_restores_aux_hidden_under_cp`` reflects the
    # caller's intent instead of defaulting to True by mock
    # auto-attribute accident.
    runner.model.restores_aux_hidden_under_cp = restores_aux_hidden_under_cp
    return runner


def test_eager_runner_builds_model_visible_execution_metadata() -> None:
    runner = _make_eager_runner(is_mla=False)
    runner.cp_size = 1
    builder = MagicMock()
    builder.metadata_type = MegaMoeMetadata
    eager_metadata = MegaMoeMetadata(
        active_token_mask=torch.tensor([1, 1, 0], dtype=torch.int8),
    )
    builder.build.return_value = eager_metadata
    runner.bind_execution_metadata_builders((builder,))
    input_batch = _make_input_batch((2,))
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
    )

    def _execute_model(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        assert get_execution_context(MegaMoeMetadata) is eager_metadata
        return input_ids + positions

    runner.model.side_effect = _execute_model
    output = runner.execute(
        input_batch.input_ids,
        input_batch.positions,
        metadata,
        input_batch=input_batch,
    )

    builder.build.assert_called_once_with(input_batch, metadata)
    torch.testing.assert_close(output, input_batch.input_ids + input_batch.positions)


def test_eager_runner_requires_input_batch_for_execution_metadata() -> None:
    runner = _make_eager_runner(is_mla=False)
    runner.cp_size = 1
    runner.bind_execution_metadata_builders((MagicMock(),))
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
    )

    with pytest.raises(RuntimeError, match="require upstream InputBatch"):
        runner.execute(torch.zeros(2), torch.arange(2), metadata)


def test_eager_runner_preserves_qwen_pure_prefill_cp_context_contract() -> None:
    runner = _make_eager_runner(is_mla=False)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=None,
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    build_context.assert_called_once_with(
        [3, 5],
        [3, 5],
        4,
        2,
        torch.device("cpu"),
    )


def test_eager_runner_skips_cp_for_empty_rank() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        is_dummy=True,
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
    )

    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        runner.execute(torch.ones(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()


def test_eager_runner_preserves_qwen_missing_length_fallback() -> None:
    runner = _make_eager_runner(is_mla=False)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=None,
        kv_seq_lens_host=None,
    )

    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()


def test_eager_runner_builds_cp_context_for_chunked_prefill() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=torch.tensor([3, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([11, 13], dtype=torch.int32),
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(8), torch.arange(8), metadata)

    build_context.assert_called_once_with(
        [3, 5],
        [11, 13],
        4,
        2,
        torch.device("cpu"),
    )


def test_eager_runner_rejects_mixed_cp_before_collective() -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=True,
        is_spec_verify=False,
    )

    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(NotImplementedError, match="mixed batches"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert not runner.attention_backend._prepared


def test_eager_runner_rejects_mla_spec_verify_cp_without_aux_restore_before_collective() -> None:
    """Gate B5 (design.md §3.1): a model without the CP aux-restore
    capability -- including one that HAS capture layers configured but
    leaves the aux buffer CP-sharded (glm_moe_dsa merges only ``hidden``),
    and any model type that never wired the attribute -- still must not
    reach a spec-verify CP forward: a draft would read token-misaligned
    rows."""
    runner = _make_eager_runner(restores_aux_hidden_under_cp=False)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=True,
    )

    with (
        patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context,
        pytest.raises(NotImplementedError, match="target-side speculative verification"),
    ):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    build_context.assert_not_called()
    assert not runner.attention_backend._prepared


def test_eager_runner_admits_mla_spec_verify_cp_with_aux_restore_capability() -> None:
    """Gate B5 (design.md §3.1): a model that restores the aux-hidden
    buffer to global row order under CP (e.g. DFlash2's glm5_next target,
    per B6's CP-safety trace, research/b6-spec-verify-v3-cp-safety.md) is
    no longer blanket-rejected for a spec-verify CP forward -- the batch
    proceeds and builds a cp_context normally."""
    runner = _make_eager_runner(restores_aux_hidden_under_cp=True)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_mixed=False,
        is_spec_verify=True,
        q_seq_lens_host=torch.tensor([2], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([2], dtype=torch.int32),
    )

    with patch(
        "xllm.python.model_executor.runners.eager.build_cp_context",
        return_value=object(),
    ) as build_context:
        runner.execute(torch.zeros(2), torch.arange(2), metadata)

    build_context.assert_called_once_with([2], [2], 4, 2, torch.device("cpu"))
    assert runner.attention_backend._prepared


def test_model_restores_aux_hidden_under_cp_reflects_model_attribute() -> None:
    """Unit-level coverage of the ``model_restores_aux_hidden_under_cp``
    property itself, independent of the ``execute`` gating above."""
    runner = _make_eager_runner(restores_aux_hidden_under_cp=False)
    assert runner.model_restores_aux_hidden_under_cp is False

    runner.model.restores_aux_hidden_under_cp = True
    assert runner.model_restores_aux_hidden_under_cp is True

    # A model with no such attribute at all (e.g. a Python model type that
    # never wired the capability, or glm_moe_dsa whose aux rows stay
    # CP-sharded) must not raise and must stay refused.
    del runner.model.restores_aux_hidden_under_cp
    assert runner.model_restores_aux_hidden_under_cp is False


def test_cp_aux_restore_capability_is_an_explicit_model_opt_in() -> None:
    """The capability the CP spec-verify gate keys on is an explicit class
    attribute only the proven models set: glm5_next and deepseek_v4 merge
    the aux-hidden buffer back to global row order at the CP exit, while
    glm_moe_dsa (glm5_2) captures aux rows but merges only ``hidden``,
    leaving them CP-sharded. A reachable glm_moe_dsa + DFlash + CP shape
    must therefore stay refused even though worker_impl auto-fills its
    layers_to_capture (making capture ``enabled``).

    deepseek_v4 imports torch_npu unconditionally at module level, so the
    model-class pins skip on hosts without torch_npu (mirroring
    test_deepseek_v4_mtp_core.py) instead of failing the file's collection.
    """
    pytest.importorskip("torch_npu")
    from xllm.python.models.deepseek_v4 import DeepseekV4Model
    from xllm.python.models.glm5_2 import Glm52Model
    from xllm.python.models.glm5_next import Glm5NextModel

    assert Glm5NextModel.restores_aux_hidden_under_cp is True
    assert DeepseekV4Model.restores_aux_hidden_under_cp is True
    # glm_moe_dsa merges only ``hidden`` under CP; the attribute must stay
    # unset so the eager gate keeps refusing its spec-verify CP forwards.
    assert not hasattr(Glm52Model, "restores_aux_hidden_under_cp")


@pytest.mark.parametrize(
    ("is_mla", "is_chunked_prefill", "is_mixed", "is_spec_verify"),
    [
        (False, True, True, False),
        (False, True, False, True),
        (True, False, False, True),
    ],
)
def test_eager_runner_preserves_non_cp_fallback(
    is_mla: bool,
    is_chunked_prefill: bool,
    is_mixed: bool,
    is_spec_verify: bool,
) -> None:
    runner = _make_eager_runner(is_mla=is_mla)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=is_chunked_prefill,
        is_mixed=is_mixed,
        is_spec_verify=is_spec_verify,
    )

    def execute_model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert get_forward_context().cp_context is None
        return input_ids + positions

    runner.model.side_effect = execute_model
    with patch("xllm.python.model_executor.runners.eager.build_cp_context") as build_context:
        output = runner.execute(torch.ones(1), torch.ones(1), metadata)

    build_context.assert_not_called()
    assert runner.attention_backend._prepared
    runner.model.assert_called_once()
    torch.testing.assert_close(output, torch.full((1,), 2.0))


@pytest.mark.parametrize(
    ("q_seq_lens_host", "kv_seq_lens_host"),
    [
        (None, None),
        (None, torch.tensor([1], dtype=torch.int32)),
        (torch.tensor([1], dtype=torch.int32), None),
    ],
)
def test_eager_runner_rejects_missing_cp_lengths(
    q_seq_lens_host: torch.Tensor | None,
    kv_seq_lens_host: torch.Tensor | None,
) -> None:
    runner = _make_eager_runner()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_seq_lens_host=q_seq_lens_host,
        kv_seq_lens_host=kv_seq_lens_host,
    )

    with pytest.raises(RuntimeError, match="requires host query and KV"):
        runner.execute(torch.zeros(1), torch.zeros(1), metadata)

    assert not runner.attention_backend._prepared


class TestExecuteRouting:
    @pytest.mark.parametrize(
        ("draft_event_index", "expected_calls"),
        [(2, [2]), (None, [0, 1, 2])],
    )
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_spec_draft_waits_for_dedicated_load_event(
        self,
        mock_create: MagicMock,
        draft_event_index: int | None,
        expected_calls: list[int],
    ) -> None:
        mock_create.return_value = StubAttentionBackend()
        executor = ModelExecutor(
            _FakeModel(num_layers=1),
            {},
            max_seqs_per_batch=4,
            is_spec_draft=True,
        )
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        executor.inductor_runner = MagicMock()
        executor.inductor_runner.execute.return_value = torch.ones(1)
        synchronizer = _HostLoadSynchronizer(3)

        executor.execute(
            torch.ones(1),
            torch.zeros(1),
            SimpleNamespace(is_prefill=True, is_chunked_prefill=False),
            layer_load_synchronizer=synchronizer,
            draft_load_event_index=draft_event_index,
        )

        assert synchronizer.calls == expected_calls

    @pytest.mark.parametrize("prefill,chunked", [(True, False), (False, True), (False, False)])
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_eager_waits_at_each_attention(self, mock_create: MagicMock, prefill: bool, chunked: bool) -> None:
        order: list[tuple[str, int]] = []

        class Loads(_HostLoadSynchronizer):
            def synchronize_layer(self, event_index: int) -> bool:
                order.append(("load", event_index))
                return super().synchronize_layer(event_index)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([_make_attention_layer(layer_id=i) for i in range(2)])

            def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                for i, layer in enumerate(self.layers):
                    input_ids = layer(input_ids, input_ids, input_ids)
                    order.append(("attention", i))
                return input_ids

        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=0)
        model.model = Model()
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1)) for _ in range(2)])
        metadata = SimpleNamespace(is_prefill=prefill, is_chunked_prefill=chunked, is_dummy=False)
        synchronizer = Loads(2)
        for _ in range(2):
            executor.execute(torch.ones(1), torch.zeros(1), metadata, layer_load_synchronizer=synchronizer)
        assert order == [("load", 0), ("attention", 0), ("load", 1), ("attention", 1)] * 2

    @pytest.mark.parametrize("failed_event", [None, 1])
    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_graph_waits_before_warmup_on_every_request(self, mock_create: MagicMock, failed_event: int | None) -> None:
        mock_create.return_value = StubAttentionBackend()
        executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
        synchronizer = _HostLoadSynchronizer(2, failed_event)
        graph = MagicMock()
        graph.can_execute.return_value = True
        executor.decode_graph_runner = graph
        for request_index in range(2):

            def warmup(*args: object, request_count: int = request_index + 1) -> None:
                assert synchronizer.calls == [0, 1] * request_count

            graph.warmup.side_effect = warmup
            expectation = (
                pytest.raises(RuntimeError, match="Host cache load event 1")
                if failed_event is not None
                else nullcontext()
            )
            with expectation:
                executor.execute(torch.ones(1), torch.zeros(1), metadata, layer_load_synchronizer=synchronizer)
        assert graph.warmup.call_count == (2 if failed_event is None else 0)
        assert graph.execute.call_count == (2 if failed_event is None else 0)

    @patch("xllm.python.model_executor.executor._create_attention_backend")
    def test_inductor_waits_outside_compiled_execution(self, mock_create: MagicMock) -> None:
        mock_create.return_value = StubAttentionBackend()
        executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        synchronizer = _HostLoadSynchronizer(2)
        executor.inductor_runner = MagicMock()

        def compiled_execute(*args: object, **kwargs: object) -> torch.Tensor:
            assert synchronizer.calls == [0, 1]
            return torch.ones(1)

        executor.inductor_runner.execute.side_effect = compiled_execute
        executor.execute(
            torch.ones(1),
            torch.zeros(1),
            SimpleNamespace(is_prefill=True, is_chunked_prefill=False),
            layer_load_synchronizer=synchronizer,
        )
        executor.inductor_runner.execute.assert_called_once()

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_without_bind_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        metadata = MagicMock(spec=AttentionMetadata)
        with pytest.raises(RuntimeError, match="KV caches are not bound"):
            executor.execute(torch.zeros(1), torch.zeros(1), metadata)

    @pytest.mark.parametrize("dp_size", [1, 2])
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_routes_to_eager_runner(self, mock_create: MagicMock, dp_size: int) -> None:
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(
            model,
            {"enable_graph": False, "python_graph_backend": "off", "dp_size": dp_size},
            max_seqs_per_batch=4,
        )
        assert executor.decode_graph_runner is None

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        metadata = MagicMock(spec=AttentionMetadata)
        metadata.is_prefill = False
        metadata.is_chunked_prefill = False
        executor.eager_runner = MagicMock()
        grad_enabled = None

        def execute(*_args, **_kwargs):
            nonlocal grad_enabled
            grad_enabled = torch.is_grad_enabled()
            return torch.ones(5)

        executor.eager_runner.execute.side_effect = execute

        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.eager_runner.execute.assert_called_once()
        assert grad_enabled is False
        assert torch.equal(result, torch.ones(5))

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_without_builders_skips_input_batch_materialization(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
        executor.bind_kv_caches([(torch.zeros(1), torch.zeros(1))])
        executor.eager_runner = MagicMock()
        executor.eager_runner.execute.return_value = torch.ones(1)
        metadata = MagicMock(spec=AttentionMetadata)
        metadata.is_prefill = False
        metadata.is_chunked_prefill = False

        with patch.object(InputBatch, "from_runtime") as from_runtime:
            executor.execute(
                torch.zeros(1),
                torch.zeros(1),
                metadata,
                input_batch_metadata=MagicMock(),
            )

        from_runtime.assert_not_called()
        assert executor.eager_runner.execute.call_args.kwargs["input_batch"] is None

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_inductor_runner_takes_priority_over_eager(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        executor.inductor_runner = MagicMock()
        executor.inductor_runner.execute.return_value = torch.ones(3)

        metadata = MagicMock(spec=AttentionMetadata)
        metadata.is_prefill = False
        metadata.is_chunked_prefill = False
        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.inductor_runner.execute.assert_called_once()
        assert torch.equal(result, torch.ones(3))

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_acl_graph_warmup_uses_scheduler_inputs(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        input_ids = torch.zeros(4, dtype=torch.int32)
        positions = torch.arange(4, dtype=torch.int32)
        metadata = MagicMock(spec=AttentionMetadata)
        metadata.is_prefill = False
        metadata.is_chunked_prefill = False
        graph_runner = MagicMock()
        graph_runner.can_execute.return_value = True
        graph_runner.execute.return_value = torch.ones(4)
        executor.decode_graph_runner = graph_runner

        result = executor.execute(input_ids, positions, metadata)

        graph_runner.warmup.assert_called_once_with(
            input_ids,
            positions,
            metadata,
            None,
        )
        graph_runner.execute.assert_called_once_with(
            input_ids,
            positions,
            metadata,
            None,
            eplb=None,
            input_batch=None,
        )
        assert torch.equal(result, torch.ones(4))
