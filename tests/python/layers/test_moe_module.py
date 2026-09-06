# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Unit tests for MoE module - Import and Structure Verification.

This test file verifies that the MoE module files are correctly structured
and can be imported. Full integration tests require the C++ framework to
initialize xllm.python runtime.
"""

import importlib.util
import sys
from pathlib import Path


def test_moe_files_exist():
    """Test that all MoE module files exist."""
    moe_path = Path("xllm/python/layers/moe")

    expected_files = [
        "__init__.py",
        "activation.py",
        "communication.py",
        "experts.py",
        "moe.py",
        "prepare_finalize.py",
        "router.py",
        "runner.py",
        "token_dispatcher.py",
        "types.py",
    ]

    for filename in expected_files:
        file_path = moe_path / filename
        assert file_path.exists(), f"Missing MoE file: {filename}"
        assert file_path.stat().st_size > 0, f"Empty MoE file: {filename}"


def test_moe_types_content():
    """Test MoE types module content."""
    types_file = Path("xllm/python/layers/moe/types.py")
    content = types_file.read_text()

    # Verify expected types are defined
    assert "class MoECommType" in content or "MoECommType = " in content
    assert "MoEExpertsConfig" in content
    assert "MoEParallelConfig" in content
    assert "MoERouterConfig" in content

    # Verify MoECommType enum values are defined
    assert "ALL_GATHER" in content
    assert "ALL_TO_ALL" in content
    assert "MC2" in content
    assert "AUTO" in content


def test_moe_activation_module():
    """Test MoE activation module can be loaded independently."""
    spec = importlib.util.spec_from_file_location("moe_activation", "xllm/python/layers/moe/activation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Verify SituAndMul exists
    assert hasattr(module, "SituAndMul")
    assert hasattr(module.SituAndMul, "forward")


def test_dependency_files_exist():
    """Test that required dependency files exist."""
    dependencies = [
        "xllm/python/ascend_custom_ops.py",
        "xllm/python/distributed/__init__.py",
        "xllm/python/distributed/parallel_state.py",
        "xllm/python/distributed/collectives.py",
        "xllm/python/layers/linear.py",
        "xllm/python/ops/collectives.py",
    ]

    for dep_path in dependencies:
        path = Path(dep_path)
        assert path.exists(), f"Missing dependency: {dep_path}"
        assert path.stat().st_size > 0, f"Empty dependency: {dep_path}"


def test_moe_module_structure():
    """Test MoE module structure and exports."""
    # Read __init__.py to verify exports
    init_file = Path("xllm/python/layers/moe/__init__.py")
    content = init_file.read_text()

    # Verify key exports are listed
    expected_exports = [
        "MoE",
        "MoERouter",
        "MoECommType",
        "MoEExpertsConfig",
        "UnquantizedRoutedExperts",
        "GroupedTopKRouter",
        "SituAndMul",
    ]

    for export in expected_exports:
        assert export in content, f"Missing export in __init__.py: {export}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
