# vlm_master_test is a standalone manual test binary and has no registered
# CTest target yet.

include(cc_test)

cc_test(
  NAME
    worker_prefetch_session_test
  CPU_ONLY
  SRCS
    worker_prefetch_session_test.cpp
  DEPS
    :worker_prefetch_session
    GTest::gtest_main
)

cc_test(
  NAME
    spawn_worker_protocol_test
  CPU_ONLY
  SRCS
    spawn_worker_protocol_test.cpp
  DEPS
    GTest::gtest_main
)

if(USE_NPU)
  cc_test(
    NAME
      worker_service_test
    SRCS
      worker_service_test.cpp
    DEPS
      :worker_service
      :runtime
      GTest::gtest_main
  )
  target_link_libraries(worker_service_test
                        PRIVATE
                        torch_npu
                        ascendcl
                        hccl
                        c_sec
                        nnopbase)
  target_link_options(worker_service_test PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
    "-Wl,--no-whole-archive")
endif()

cc_test(
  NAME llm_engine_host_transfer_test
  SRCS llm_engine_host_transfer_test.cpp
  DEPS :distributed_runtime :xllm_server :batch :block GTest::gtest_main
)
target_link_libraries(llm_engine_host_transfer_test PRIVATE
  :xtensor :xllm_server :batch :block)
