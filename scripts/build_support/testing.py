import tempfile


def prepare_test_environment(environment: dict[str, str], cmake_dir: str) -> dict[str, str]:
    """Keep interrupted JIT builds from blocking a later test invocation."""
    test_environment = environment.copy()
    if not test_environment.get("TORCH_EXTENSIONS_DIR"):
        test_environment["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(prefix="test-torch-extensions-", dir=cmake_dir)
    return test_environment
