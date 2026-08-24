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
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    monkeypatch.setattr(registry, "import_module", import_model)

    with pytest.raises(NotImplementedError, match="qwen3_5.*npu"):
        registry.get_model_class("qwen3_5")

    import_model.assert_not_called()


def test_kimi_k3_dspark_is_registered_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_class = type("KimiK3DSparkForCausalLM", (), {})
    model_module = Mock(KimiK3DSparkForCausalLM=model_class)
    import_model = Mock(return_value=model_module)
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    monkeypatch.setattr(registry, "import_module", import_model)

    assert registry.get_model_class("K3DSparkModel") is model_class
    import_model.assert_called_once_with("xllm.python.models.kimi_k3_dspark")


def test_legacy_dspark_is_registered_as_qwen3_gqa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_class = type("Qwen3DSparkForCausalLM", (), {})
    model_module = Mock(Qwen3DSparkForCausalLM=model_class)
    import_model = Mock(return_value=model_module)
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    monkeypatch.setattr(registry, "import_module", import_model)

    assert registry.get_model_class("DSparkDraftModel") is model_class
    import_model.assert_called_once_with("xllm.python.models.qwen3_dspark")
