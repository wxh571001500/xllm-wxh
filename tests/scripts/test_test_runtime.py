import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from setuptools import Distribution

from scripts.build_support import testing


class TestEnvironmentTest(unittest.TestCase):
    def test_runs_get_separate_caches_without_mutating_parent(self) -> None:
        parent = {"PATH": "/bin", "ASCEND_RT_VISIBLE_DEVICES": "0"}
        with tempfile.TemporaryDirectory() as build_dir:
            first = testing.prepare_test_environment(parent, build_dir)
            lock = Path(first["TORCH_EXTENSIONS_DIR"]) / "quant_flash_attn" / "lock"
            lock.parent.mkdir()
            lock.touch()
            second = testing.prepare_test_environment(parent, build_dir)
            self.assertNotEqual(first["TORCH_EXTENSIONS_DIR"], second["TORCH_EXTENSIONS_DIR"])
            self.assertTrue(Path(second["TORCH_EXTENSIONS_DIR"]).is_dir())
            self.assertTrue(lock.exists())
            self.assertEqual(second["ASCEND_RT_VISIBLE_DEVICES"], "0")
        self.assertNotIn("TORCH_EXTENSIONS_DIR", parent)

    def test_explicit_cache_is_preserved(self) -> None:
        parent = {"TORCH_EXTENSIONS_DIR": "/explicit/cache"}
        with tempfile.TemporaryDirectory() as build_dir:
            result = testing.prepare_test_environment(parent, build_dir)
            self.assertEqual(result, parent)
            self.assertIsNot(result, parent)
            self.assertEqual(list(Path(build_dir).iterdir()), [])

    def test_empty_cache_setting_gets_isolated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as build_dir:
            result = testing.prepare_test_environment({"TORCH_EXTENSIONS_DIR": ""}, build_dir)
            self.assertTrue(Path(result["TORCH_EXTENSIONS_DIR"]).is_dir())


class TestRunnerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        setup_path = Path(__file__).resolve().parents[2] / "setup.py"
        spec = importlib.util.spec_from_file_location("xllm_setup_test_runtime", setup_path)
        self.setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.setup_module)

    def test_ctest_phases_share_one_isolated_cache(self) -> None:
        command = self.setup_module.TestUT(Distribution())
        process = mock.Mock()
        process.stdout.readline.return_value = ""
        process.wait.return_value = 0
        with (
            tempfile.TemporaryDirectory() as build_dir,
            mock.patch.dict(os.environ, {"CTEST_PARALLEL": "4"}, clear=True),
            mock.patch.object(self.setup_module.subprocess, "Popen", return_value=process) as popen,
        ):
            self.assertEqual(command.run_ctest(build_dir), 0)
            self.assertEqual(popen.call_count, 1 + len(command.SEQUENTIAL_TESTS))
            environments = [call.kwargs["env"] for call in popen.call_args_list]
            self.assertEqual(len({env["TORCH_EXTENSIONS_DIR"] for env in environments}), 1)
            self.assertTrue(Path(environments[0]["TORCH_EXTENSIONS_DIR"]).is_dir())
            self.assertIn("4", popen.call_args_list[0].args[0])

    def test_ctest_failure_preserves_failure_status_and_cache(self) -> None:
        command = self.setup_module.TestUT(Distribution())
        process = mock.Mock()
        process.stdout.readline.return_value = ""
        process.wait.return_value = 1
        with (
            tempfile.TemporaryDirectory() as build_dir,
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(self.setup_module.subprocess, "Popen", return_value=process) as popen,
        ):
            with self.assertRaises(SystemExit) as failure:
                command.run_ctest(build_dir)
            self.assertEqual(failure.exception.code, 1)
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(Path(popen.call_args.kwargs["env"]["TORCH_EXTENSIONS_DIR"]).is_dir())

    def test_single_target_uses_isolated_cache(self) -> None:
        command = self.setup_module.ExtBuildSingleTest(Distribution())
        command.test_name = "example_test"
        with (
            tempfile.TemporaryDirectory() as build_dir,
            mock.patch.object(self.setup_module, "get_cmake_dir", return_value=build_dir),
            mock.patch.object(self.setup_module.subprocess, "check_call") as check_call,
        ):
            executable = Path(build_dir) / command.test_name
            executable.touch(mode=0o755)
            command.build_cmake_targets(None, [], [], {"PATH": "/bin"}, build_dir, "unused")
            invocation = check_call.call_args
            self.assertEqual(invocation.args[0], [str(executable)])
            self.assertTrue(Path(invocation.kwargs["env"]["TORCH_EXTENSIONS_DIR"]).is_dir())

    def test_single_target_ctest_fallback_uses_isolated_cache(self) -> None:
        command = self.setup_module.ExtBuildSingleTest(Distribution())
        command.test_name = "example_test"
        with (
            tempfile.TemporaryDirectory() as build_dir,
            mock.patch.object(self.setup_module, "get_cmake_dir", return_value=build_dir),
            mock.patch.object(self.setup_module.subprocess, "check_call") as check_call,
        ):
            command.build_cmake_targets(None, [], [], {"PATH": "/bin"}, build_dir, "unused")
            invocation = check_call.call_args
            self.assertEqual(invocation.args[0], ["ctest", "-R", "example_test", "--verbose"])
            self.assertTrue(Path(invocation.kwargs["env"]["TORCH_EXTENSIONS_DIR"]).is_dir())


if __name__ == "__main__":
    unittest.main()
