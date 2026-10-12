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

"""Host-only coverage for the shared NPU auxiliary stream cache."""

import runpy
from types import ModuleType, SimpleNamespace

import pytest
import torch

from xllm.python import npu_streams
from xllm.python.models import deepseek_v32, glm5_next


def _stub_stream_creation(monkeypatch: pytest.MonkeyPatch) -> list[SimpleNamespace]:
    """Replace ``torch.npu.Stream`` and reset the cache, returning created streams."""
    created: list[SimpleNamespace] = []

    def create_stream(device: torch.device) -> SimpleNamespace:
        stream = SimpleNamespace(device=device)
        created.append(stream)
        return stream

    monkeypatch.setattr(torch, "npu", SimpleNamespace(Stream=create_stream), raising=False)
    monkeypatch.setattr(npu_streams, "_SHARED_EXPERT_STREAMS", {})
    return created


@pytest.mark.parametrize("first_model", [deepseek_v32, glm5_next])
def test_models_share_one_cached_stream_per_device(
    first_model: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _stub_stream_creation(monkeypatch)
    device = torch.device("privateuseone:0")
    other_device = torch.device("privateuseone:1")

    shared = first_model.shared_expert_stream(device)
    other_shared = npu_streams.shared_expert_stream(other_device)
    for model in (deepseek_v32, glm5_next):
        assert model.shared_expert_stream(device) is shared
        assert model.shared_expert_stream(other_device) is other_shared
    assert shared is not other_shared
    assert [stream.device for stream in created] == [device, other_device]


def test_release_drops_only_the_requested_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stream_creation(monkeypatch)
    device = torch.device("privateuseone:0")
    other_device = torch.device("privateuseone:1")

    first = npu_streams.shared_expert_stream(device)
    other = npu_streams.shared_expert_stream(other_device)
    npu_streams.release_shared_expert_stream(device)

    assert set(npu_streams._SHARED_EXPERT_STREAMS) == {(other_device.type, other_device.index)}
    assert npu_streams.shared_expert_stream(other_device) is other
    assert npu_streams.shared_expert_stream(device) is not first


def test_release_of_uncached_device_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_stream_creation(monkeypatch)
    npu_streams.release_shared_expert_stream(torch.device("privateuseone:7"))
    assert npu_streams._SHARED_EXPERT_STREAMS == {}


def test_indexed_and_unindexed_devices_do_not_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _stub_stream_creation(monkeypatch)
    indexed = npu_streams.shared_expert_stream(torch.device("privateuseone:0"))
    unindexed = npu_streams.shared_expert_stream(torch.device("privateuseone"))

    assert indexed is not unindexed
    assert len(created) == 2


def test_import_without_npu_does_not_create_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(torch, "npu", raising=False)
    namespace = runpy.run_path(npu_streams.__file__)
    assert namespace["_SHARED_EXPERT_STREAMS"] == {}
    assert callable(namespace["shared_expert_stream"])


def test_stream_creation_failure_propagates_without_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_stream(device: torch.device) -> None:
        raise RuntimeError("stream creation failed")

    monkeypatch.setattr(torch, "npu", SimpleNamespace(Stream=fail_stream), raising=False)
    monkeypatch.setattr(npu_streams, "_SHARED_EXPERT_STREAMS", {})
    with pytest.raises(RuntimeError, match="stream creation failed"):
        npu_streams.shared_expert_stream(torch.device("privateuseone:0"))
    assert not npu_streams._SHARED_EXPERT_STREAMS
