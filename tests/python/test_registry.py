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

from unittest.mock import Mock

import pytest

from xllm.python import registry


def test_unsupported_model_fails_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import_model = Mock()
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "cuda")
    monkeypatch.setattr(registry, "import_module", import_model)

    with pytest.raises(NotImplementedError, match="qwen3_vl.*cuda"):
        registry.get_model_class("qwen3_vl")

    import_model.assert_not_called()


def test_glm5_next_text_and_vl_registry_entries_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")

    text_cls = registry.get_model_class("glm5_next")
    vl_cls = registry.get_model_class("glm5_next_vl")

    assert text_cls.__name__ == "Glm5NextForCausalLM"
    assert vl_cls.__name__ == "Glm5NextVLModel"


@pytest.mark.parametrize("name", ["deepseek_v4_mtp", "DeepseekV4MtpForCausalLM"])
def test_deepseek_v4_mtp_registry_entries(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")

    model_class = registry.get_model_class(name)

    assert model_class.__name__ == "DeepseekV4MtpForCausalLM"


def test_qwen35_registers_execution_metadata_builder_class() -> None:
    builder_classes = registry.get_execution_metadata_builder_classes("qwen3_5_moe_text")

    assert tuple(builder_class.__name__ for builder_class in builder_classes) == (
        "DenseDcpMetadataBuilder",
        "TokenOwnerMegaMoeMetadataBuilder",
        "Qwen3_5GdnMetadataBuilder",
    )


def test_qwen35_uses_official_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")

    model_class = registry.get_model_class("qwen3_5_moe_text")

    assert model_class.__name__ == "Qwen3_5ForCausalLM"


@pytest.mark.parametrize("model_type", ["qwen3", "deepseek_v32", "deepseek_v4", "glm_moe_dsa"])
def test_model_without_execution_metadata_builder_returns_empty_tuple(model_type: str) -> None:
    assert registry.get_execution_metadata_builder_classes(model_type) == ()
