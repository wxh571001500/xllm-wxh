import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("cmake") and shutil.which("ctest"), "CMake and CTest are required")
class CpuOnlyTestRegistrationTest(unittest.TestCase):
    def test_cpu_only_preserves_cases_without_npu_bootstrap(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            build = source / "build"
            (source / "tests").mkdir()
            (source / "tests/npu_test_environment.cpp").touch()
            (source / "cpu_test.cpp").write_text("TEST(CpuSuite, Basic) {}\n")
            (source / "npu_test.cpp").write_text("TEST(NpuSuite, Basic) {}\n")
            (source / "CMakeLists.txt").write_text(
                f"""cmake_minimum_required(VERSION 3.18)
project(cc_test_contract LANGUAGES CXX)
enable_testing()
include(GoogleTest)
list(APPEND CMAKE_MODULE_PATH "{repo_root / "cmake"}")
include(cc_test)
set(BUILD_TESTING ON)
set(USE_NPU ON)
add_custom_target(all_tests)
add_library(Python::Python INTERFACE IMPORTED)
cc_test(NAME cpu_test CPU_ONLY SRCS cpu_test.cpp ENVIRONMENT "EXAMPLE=value")
cc_test(NAME npu_test SRCS npu_test.cpp)
foreach(target cpu_test npu_test)
  get_target_property(sources ${{target}} SOURCES)
  get_target_property(libraries ${{target}} LINK_LIBRARIES)
  file(WRITE "${{CMAKE_BINARY_DIR}}/${{target}}.sources" "${{sources}}")
  file(WRITE "${{CMAKE_BINARY_DIR}}/${{target}}.libraries" "${{libraries}}")
endforeach()
"""
            )
            subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True, capture_output=True, text=True)
            self.assertNotIn("npu_test_environment.cpp", (build / "cpu_test.sources").read_text())
            self.assertIn("npu_test_environment.cpp", (build / "npu_test.sources").read_text())
            cpu_libraries = (build / "cpu_test.libraries").read_text()
            npu_libraries = (build / "npu_test.libraries").read_text()
            for library in ("ascendcl", "Python::Python", "torch_npu", "torch_python"):
                self.assertNotIn(library, cpu_libraries)
                self.assertIn(library, npu_libraries)
            result = subprocess.run(
                ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                check=True,
                capture_output=True,
                text=True,
            )
            tests = {test["name"]: test for test in json.loads(result.stdout)["tests"]}
            self.assertEqual(set(tests), {"CpuSuite.Basic", "NpuSuite.Basic"})
            properties = {item["name"]: item["value"] for item in tests["CpuSuite.Basic"]["properties"]}
            self.assertEqual(properties["LABELS"], ["cpu"])
            self.assertEqual(properties["ENVIRONMENT"], ["EXAMPLE=value"])

    def test_text_processing_targets_do_not_bootstrap_npu(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        targets = (
            "partial_json_parser_test",
            "qwen25_detector_test",
            "qwen3_coder_detector_test",
            "kimik2_detector_test",
            "deepseekv3_detector_test",
            "glm45_detector_test",
            "glm47_detector_test",
            "deepseekv32_detector_test",
            "chat_template_test",
            "jinja_chat_template_test",
            "deepseek_v32_cpp_template_test",
            "deepseek_v4_cpp_template_test",
            "fast_tokenizer_test",
            "rwkv_tokenizer_test",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            build = source / "build"
            (source / "tests").mkdir()
            (source / "tests/npu_test_environment.cpp").touch()
            (source / "CMakeLists.txt").write_text(
                f"""cmake_minimum_required(VERSION 3.18)
project(text_processing_test_contract LANGUAGES CXX)
enable_testing()
include(GoogleTest)
list(APPEND CMAKE_MODULE_PATH "{repo_root / "cmake"}")
set(BUILD_TESTING ON)
set(USE_NPU ON)
set(XLLM_TESTS_DIR "{repo_root / "tests"}")
add_custom_target(all_tests)
foreach(dependency GTest::gtest GTest::gtest_main glog::glog nlohmann_json::nlohmann_json Python::Python)
  add_library(${{dependency}} INTERFACE IMPORTED)
endforeach()
add_subdirectory("{repo_root / "tests/function_call"}" function_call)
add_subdirectory("{repo_root / "tests/core/framework/chat_template"}" chat_template)
add_subdirectory("{repo_root / "tests/core/framework/tokenizer"}" tokenizer)
foreach(target {" ".join(targets)})
  get_target_property(sources ${{target}} SOURCES)
  file(WRITE "${{CMAKE_BINARY_DIR}}/${{target}}.sources" "${{sources}}")
endforeach()
"""
            )
            subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True, capture_output=True, text=True)
            for target in targets:
                with self.subTest(target=target):
                    self.assertNotIn("npu_test_environment.cpp", (build / f"{target}.sources").read_text())
            result = subprocess.run(
                ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                check=True,
                capture_output=True,
                text=True,
            )
            tests = json.loads(result.stdout)["tests"]
            self.assertTrue(tests)
            for test in tests:
                with self.subTest(test=test["name"]):
                    properties = {item["name"]: item["value"] for item in test["properties"]}
                    self.assertEqual(properties.get("LABELS"), ["cpu"])

    def test_cpu_indices_preserve_serial_rules_and_other_npu_targets(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        cpu_target = "qwen3_5_gated_delta_net_indices_test"
        cpu_layer_targets = (
            "deepseek_v4_eplb_load_utils_test",
            "deepseek_v4_eplb_test",
        )
        npu_targets = (
            "npu_deepseek_v4_indexer_test",
            "npu_linear_w8a8_dynamic_test",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            build = source / "build"
            (source / "tests").mkdir()
            (source / "tests/npu_test_environment.cpp").touch()
            (source / "CMakeLists.txt").write_text(
                f"""cmake_minimum_required(VERSION 3.18)
project(indices_test_contract LANGUAGES CXX)
enable_testing()
include(GoogleTest)
list(APPEND CMAKE_MODULE_PATH "{repo_root / "cmake"}")
set(BUILD_TESTING ON)
set(USE_NPU ON)
set(XLLM_TESTS_DIR "{repo_root / "tests"}")
add_custom_target(all_tests)
foreach(dependency GTest::gtest_main glog::glog Python::Python)
  add_library(${{dependency}} INTERFACE IMPORTED)
endforeach()
add_subdirectory("{repo_root / "xllm/core/layers/npu_torch"}" production_npu_torch)
add_subdirectory("{repo_root / "tests/core/layers/npu_torch"}" npu_torch)
foreach(target {cpu_target} {" ".join(cpu_layer_targets + npu_targets)})
  get_target_property(sources ${{target}} SOURCES)
  file(WRITE "${{CMAKE_BINARY_DIR}}/${{target}}.sources" "${{sources}}")
  get_target_property(links ${{target}} LINK_LIBRARIES)
  file(WRITE "${{CMAKE_BINARY_DIR}}/${{target}}.links" "${{links}}")
endforeach()
if(TARGET qwen3_5_gdn_indices)
  get_target_property(links qwen3_5_gdn_indices LINK_LIBRARIES)
  file(WRITE "${{CMAKE_BINARY_DIR}}/indices_library.links" "${{links}}")
endif()
get_target_property(links npu_torch_layers LINK_LIBRARIES)
file(WRITE "${{CMAKE_BINARY_DIR}}/npu_layers.links" "${{links}}")
"""
            )
            subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True, capture_output=True, text=True)
            self.assertNotIn("npu_test_environment.cpp", (build / f"{cpu_target}.sources").read_text())
            self.assertEqual(
                (build / f"{cpu_target}.links").read_text().split(";"),
                [":qwen3_5_gdn_indices", "glog::glog", "torch", "GTest::gtest_main"],
            )
            self.assertEqual((build / "indices_library.links").read_text().split(";"), ["torch", "glog::glog"])
            self.assertIn(":qwen3_5_gdn_indices", (build / "npu_layers.links").read_text().split(";"))
            for target in cpu_layer_targets:
                with self.subTest(target=target):
                    self.assertNotIn("npu_test_environment.cpp", (build / f"{target}.sources").read_text())
            for target in npu_targets:
                with self.subTest(target=target):
                    self.assertIn("npu_test_environment.cpp", (build / f"{target}.sources").read_text())
            result = subprocess.run(
                ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                check=True,
                capture_output=True,
                text=True,
            )
            tests = json.loads(result.stdout)["tests"]
            indices_tests = [test for test in tests if test["name"].startswith("Qwen3_5GatedDeltaNetIndices")]
            self.assertTrue(indices_tests)
            for test in tests:
                with self.subTest(test=test["name"]):
                    properties = {item["name"]: item["value"] for item in test["properties"]}
                    if test in indices_tests:
                        self.assertEqual(properties.get("LABELS"), ["cpu"])
                        self.assertTrue(properties.get("RUN_SERIAL"))
                    elif test["name"].startswith(("DeepseekV4EplbTest.", "DeepseekV4EplbLoadUtilsTest.")):
                        self.assertEqual(properties.get("LABELS"), ["cpu"])
                        self.assertFalse(properties.get("RUN_SERIAL", False))
                    else:
                        self.assertNotIn("cpu", properties.get("LABELS", []))

    def test_host_metadata_targets_preserve_device_test_bootstrap(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        cpu_targets = (
            "anthropic_protocol_test",
            "api_service_test",
            "completion_json_parser_test",
            "request_id_test",
            "request_admission_test",
            "anthropic_json_test",
            "anthropic_stream_utils_test",
            "usage_json_test",
            "models_service_impl_test",
            "config_json_test",
            "reshard_planner_test",
            "pd_topology_guard_test",
            "json_object_grammar_test",
            "mtp_prefix_cache_test",
            "block_test",
            "request_params_test",
            "dit_request_params_test",
            "kv_cache_estimation_test",
            "npu_cp_plan_test",
            "hf_model_loader_test",
            "prefix_test",
            "linear_state_prefix_cache_test",
            "eplb_policy_test",
            "util_test",
            "profile_graph_warmup_test",
            "cola_utils_test",
            "common_test",
            "context_parallel_topology_test",
            "collective_communicator_policy_test",
            "cp_group_ranks_test",
            "npu_dp_ep_padding_test",
            "npu_cp_capability_test",
            "cache_layout_builder_test",
            "adaptive_speculative_controller_test",
            "speculative_profile_registry_test",
            "mtp_json_object_state_test",
            "eplb_options_test",
            "eplb_utils_test",
            "eplb_aggregator_test",
            "expert_weight_buffer_shm_test",
            "eplb_manager_test",
            "sequence_stop_output_test",
            "rec_request_factory_test",
            "linear_state_restore_test",
            "worker_json_object_overlap_test",
            "concurrent_block_manager_test",
            "stopping_checker_test",
            "sequence_kv_state_test",
            "sequence_generated_tokens_test",
            "sequence_mrope_positions_test",
            "sequence_mtp_bootstrap_test",
            "request_prefix_cache_tokens_test",
            "sample_slot_test",
            "rec_vocab_dict_test",
            "state_dict_utils_test",
            "mapping_npu_test",
            "acl_graph_bucket_policy_test",
            "anthropic_service_test",
            "openai_service_test",
            "mm_service_utils_test",
            "sample_service_impl_test",
            "spawn_worker_protocol_test",
            "push_route_test",
            "kv_transfer_completion_test",
            "llm_request_factory_test",
            "vlm_request_factory_test",
            "dit_media_sources_test",
            "dit_request_test",
            "dit_batch_test",
            "encoder_cache_test",
            "embedding_cache_test",
            "causal_lm_test",
            "model_registry_test",
            "host_kv_transfer_test",
            "kv_cache_store_test",
            "compressed_tensors_test",
            "rotary_embedding_util_test",
            "dsa_topk_share_plan_test",
            "deepseek_v4_cp_split_test",
            "deepseek_v4_eplb_load_utils_test",
            "deepseek_v4_eplb_test",
            "cp_context_builder_test",
            "deepseek_v4_rotary_embedding_test",
            "dflash2_grouped_conv_test",
        )
        device_targets = (
            "mm_embedding_roundtrip_test",
            "basic_host_kv_transfer_test",
            "hierarchy_kv_cache_transfer_test",
            "sampler_filter_mask_test",
            "batch_test",
            "kv_cache_test",
            "scheduler_test",
            "spec_input_builder_test",
            "mtp_async_state_test",
            "mtp_async_input_builder_test",
            "eplb_executor_test",
            "npu_linear_state_lifecycle_test",
            "worker_hierarchy_kv_cache_transfer_test",
            "mtp_host_offload_test",
            "acl_graph_executor_test",
            "worker_service_test",
            "dit_tensor_sources_test",
            "py_causal_lm_test",
            "attention_metadata_builder_test",
            "npu_deepseek_v4_indexer_test",
            "npu_linear_w8a8_dynamic_test",
            "npu_xllm_ops_test",
            "glm5_next_moe_test",
            "mtp_prepare_next_draft_test",
            "fused_gdn_gating_wrapper_test",
        )
        self.assertTrue(set(cpu_targets).isdisjoint(device_targets))
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            build = source / "build"
            (source / "tests").mkdir()
            (source / "tests/npu_test_environment.cpp").touch()
            for relative_source in (
                "sampling/rejection_sampler.cpp",
                "speculative/spec_input_builder.cpp",
                "speculative/mtp_async_state.cpp",
                "speculative/adaptive_speculative_controller.cpp",
                "speculative/speculative_profile_registry.cpp",
            ):
                test_source = source / "xllm/core/framework" / relative_source
                test_source.parent.mkdir(parents=True, exist_ok=True)
                test_source.touch()
            (source / "CMakeLists.txt").write_text(
                f"""cmake_minimum_required(VERSION 3.24)
project(host_metadata_test_contract LANGUAGES CXX)
enable_testing()
include(GoogleTest)
list(APPEND CMAKE_MODULE_PATH "{repo_root / "cmake"}")
set(BUILD_TESTING ON)
set(USE_NPU ON)
set(XLLM_TESTS_DIR "{repo_root / "tests"}")
add_custom_target(all_tests)
add_custom_target(brpc-static)
foreach(dependency GTest::gtest GTest::gtest_main glog::glog Python::Python
    proto::xllm_proto nlohmann_json::nlohmann_json absl::strings absl::time
    absl::synchronization absl::random_random OpenSSL::SSL OpenSSL::Crypto
    leveldb::leveldb protobuf::libprotobuf Folly::folly Boost::serialization
    spdlog::spdlog gflags::gflags pybind11::embed)
  add_library(${{dependency}} INTERFACE IMPORTED)
endforeach()
add_subdirectory("{repo_root / "xllm/core/layers/npu"}" production_npu)
foreach(directory api_service core/common core/distributed_runtime core/framework core/runtime core/util
    core/scheduler models core/layers core/kernels/npu)
  add_subdirectory("{repo_root / "tests"}/${{directory}}" "${{directory}}")
endforeach()
foreach(target {" ".join(cpu_targets + device_targets)})
  get_target_property(sources ${{target}} SOURCES)
  file(WRITE "${{CMAKE_BINARY_DIR}}/${{target}}.sources" "${{sources}}")
endforeach()
"""
            )
            subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True, capture_output=True, text=True)
            for target in cpu_targets:
                with self.subTest(target=target):
                    self.assertNotIn("npu_test_environment.cpp", (build / f"{target}.sources").read_text())
            for target in device_targets:
                with self.subTest(target=target):
                    self.assertIn("npu_test_environment.cpp", (build / f"{target}.sources").read_text())


if __name__ == "__main__":
    unittest.main()
