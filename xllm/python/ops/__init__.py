# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backwards-compatible op-dispatch shim.

The coding-main split the old ``xllm.python.ops`` package into the platform
kernel package (``xllm.python.kernels``, e.g. ``kernels_npu``) and
``xllm.python.distributed`` (collectives). Model and layer code that still
imports ``from xllm.python import ops`` is transparently forwarded to the new
homes without re-registering native ops or fakes, so there is exactly one
source of truth per operator.

Attribute access is resolved lazily on first use, by which point the embedded
runtime has called ``xllm.python.initialize_runtime`` and published the active
platform kernel package.
"""

from __future__ import annotations

import importlib
from typing import Any

_MISSING = object()


def __getattr__(name: str) -> Any:
    # Compute / attention / sparse-op kernels published by initialize_runtime().
    kernels = importlib.import_module("xllm.python.kernels")
    value = getattr(kernels, name, _MISSING)
    if value is not _MISSING:
        return value

    # Tensor-parallel collectives.
    distributed = importlib.import_module("xllm.python.distributed")
    value = getattr(distributed, name, _MISSING)
    if value is not _MISSING:
        return value

    # Vision packed attention lives in the platform kernels' ``vision`` submodule
    # but is intentionally not part of the published ``__all__`` surface, so
    # resolve it through the active backend rather than hard-coding a device.
    if name == "encoder_attention":
        vision = importlib.import_module(f"{kernels.__name__}.vision")
        return getattr(vision, name)

    raise AttributeError(f"module 'xllm.python.ops' has no attribute {name!r}")


__all__: list[str] = []
