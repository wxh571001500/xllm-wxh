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

"""Fake-op registration must stay resilient to stale xLLM binaries.

``xllm/python/kernels_npu/_custom_op.py`` is imported from the source tree at
runtime, while the ``xllm_ops`` C++ schemas come from the compiled binary. A
stale binary (newer Python source, older build) used to make the whole model
runtime fail at boot with ``operator 'xllm_ops::...' is not registered``. The
loader now skips missing operators with a warning so every model can still
start; graph capture through a genuinely missing operator keeps failing.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

from xllm.python.kernels_npu import _custom_op


def test_register_fake_missing_operator_with_namespace_warns_and_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale binary (namespace present, schema missing) must warn, not raise."""

    monkeypatch.setattr(_custom_op, "_is_registered", lambda qualname: False)
    monkeypatch.setattr(_custom_op, "_loaded_xllm_ops_runtime", lambda: True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _custom_op.register_fake("xllm_ops::definitely_missing_op_xyz", lambda tensor: tensor)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "xllm_ops::definitely_missing_op_xyz" in message
    assert "not registered" in message


def test_register_fake_without_namespace_skips_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone/test interpreters (no xLLM ops loaded) must not warn."""

    monkeypatch.setattr(_custom_op, "_is_registered", lambda qualname: False)
    monkeypatch.setattr(_custom_op, "_loaded_xllm_ops_runtime", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _custom_op.register_fake("xllm_ops::definitely_missing_op_xyz", lambda tensor: tensor)

    assert caught == []


def test_custom_op_module_imports_without_native_ops() -> None:
    """Module import must not raise even when no xLLM operator is loaded."""

    module = importlib.import_module("xllm.python.kernels_npu._custom_op")
    assert hasattr(module, "register_fake")
