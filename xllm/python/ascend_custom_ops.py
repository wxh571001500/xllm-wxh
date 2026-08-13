# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Runtime loader for fused Ascend custom operators."""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
from pathlib import Path
import threading
from typing import Any

import torch


_CUSTOM_OP_HANDLES: list[Any] = []
_CUSTOM_OP_LOCK = threading.Lock()


def _prepend_env_path(name: str, path: str) -> None:
    entries = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if path not in entries:
        entries.insert(0, path)
        os.environ[name] = ":".join(entries)


def ensure_ascend_custom_ops(required_ops: tuple[str, ...]) -> None:
    """Load vLLM-Ascend's extension when required fused ops are unavailable."""
    if all(hasattr(torch.ops._C_ascend, name) for name in required_ops):
        return

    with _CUSTOM_OP_LOCK:
        if all(hasattr(torch.ops._C_ascend, name) for name in required_ops):
            return
        try:
            distribution = importlib.metadata.distribution("vllm-ascend")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "Fused Ascend execution requires the vllm-ascend package"
            ) from error

        package_dir = Path(distribution.locate_file("vllm_ascend"))
        vendor_dir = (
            package_dir / "_cann_ops_custom" / "vendors" / "custom_transformer"
        )
        vendor_library = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
        kernels_library = package_dir / "libvllm_ascend_kernels.so"
        extension_paths = sorted(package_dir.glob("vllm_ascend_C.*.so"))
        required_paths = (vendor_library, kernels_library)
        if not extension_paths or any(not path.is_file() for path in required_paths):
            raise RuntimeError(
                "Installed vllm-ascend does not contain the required custom op libraries"
            )

        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", str(vendor_dir))
        _CUSTOM_OP_HANDLES.extend(
            (
                ctypes.CDLL(str(vendor_library), mode=ctypes.RTLD_GLOBAL),
                ctypes.CDLL(str(kernels_library), mode=ctypes.RTLD_GLOBAL),
            )
        )
        torch.ops.load_library(str(extension_paths[0]))

        missing_ops = [
            name for name in required_ops if not hasattr(torch.ops._C_ascend, name)
        ]
        if missing_ops:
            raise RuntimeError(
                "vllm-ascend did not register required fused Ascend ops: "
                f"{missing_ops}"
            )
