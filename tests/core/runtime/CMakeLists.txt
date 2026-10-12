include(cc_test)

cc_test(
  NAME
    draft_model_config_test
  CPU_ONLY
  SRCS
    draft_model_config_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/runtime/draft_model_config.cpp
  DEPS
    :util
    :platform
    $<$<BOOL:${USE_NPU}>:config>
    torch
    glog::glog
    GTest::gtest_main
)

cc_test(
  NAME
    forward_output_test
  CPU_ONLY
  SRCS
    forward_output_test.cpp
  DEPS
    torch
    proto::xllm_proto
    GTest::gtest_main
)

cc_test(
  NAME
    acl_graph_bucket_policy_test
  CPU_ONLY
  SRCS
    acl_graph_bucket_policy_test.cpp
  DEPS
    GTest::gtest_main
)

if(USE_NPU)
  cc_test(
    NAME dflash2_prelaunch_lifecycle_test
    SRCS dflash2_prelaunch_lifecycle_test.cpp
    DEPS :runtime :models :batch :block :platform GTest::gtest_main
  )
  target_link_libraries(dflash2_prelaunch_lifecycle_test PRIVATE
    torch_npu torch_python Python::Python ascendcl hccl c_sec nnopbase)
  target_link_options(dflash2_prelaunch_lifecycle_test PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
    "-Wl,--no-whole-archive")

  cc_test(
    NAME
      npu_linear_state_lifecycle_test
    SRCS
      npu_linear_state_lifecycle_test.cpp
    DEPS
      :runtime
      :model_loader
      :batch
      :kv_cache
      :platform
      GTest::gtest_main
  )
  target_link_libraries(npu_linear_state_lifecycle_test
                        PRIVATE
                        torch_npu
                        ascendcl
                        hccl
                        c_sec
                        nnopbase)
  target_link_options(npu_linear_state_lifecycle_test PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
    "-Wl,--no-whole-archive")

  cc_test(
    NAME
      dspark_confidence_head_test
    SRCS
      dspark_confidence_head_test.cpp
    DEPS
      :models
      :state_dict
      :xllm_ops
      glog::glog
      torch
      GTest::gtest_main
  )
  target_link_libraries(dspark_confidence_head_test
                        PRIVATE
                        torch_npu
                        ascendcl
                        hccl
                        c_sec
                        nnopbase
                        atb)
  target_link_options(dspark_confidence_head_test PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
    "-Wl,--no-whole-archive")

  cc_test(
    NAME
      acl_graph_executor_test
    SRCS
      acl_graph_executor_test.cpp
    DEPS
      :xllm_server
      :runtime
      :model_loader
      :batch
      :block
      :model
      :models
      :sampler
      :kv_cache
      GTest::gtest_main
      torch_npu
      atb_customize
  )
  target_link_libraries(acl_graph_executor_test
                        PRIVATE
                        torch_npu
                        ascendcl
                        hccl
                        c_sec
                        nnopbase)
  target_link_options(acl_graph_executor_test PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
    "-Wl,--no-whole-archive")

  cc_test(
    NAME
      acl_graph_task_update_test
    SRCS
      acl_graph_task_update_test.cpp
    DEPS
      :xllm_server
      :runtime
      :model_loader
      :batch
      :block
      :model
      :models
      :sampler
      :kv_cache
      GTest::gtest_main
      torch_npu
      atb_customize
  )
  target_link_libraries(acl_graph_task_update_test
                        PRIVATE
                        torch_npu
                        ascendcl
                        hccl
                        c_sec
                        nnopbase)
  target_link_options(acl_graph_task_update_test PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
    "-Wl,--no-whole-archive")

  # This case captures and replays three ACL graphs on NPU 0. Running it
  # alongside hundreds of torch_npu test processes can stall graph cleanup.
  set_tests_properties(
    AclGraphTaskUpdateTest.StaticMtpPrepareReplaysFiaSpecVerifyWidths
    PROPERTIES RUN_SERIAL TRUE)
endif()

cc_test(
  NAME
    worker_json_object_overlap_test
  CPU_ONLY
  SRCS
    worker_json_object_overlap_test.cpp
  DEPS
    :json_object_grammar
    GTest::gtest_main
)

if(USE_NPU OR USE_MLU)
  cc_test(
    NAME
      worker_hierarchy_kv_cache_transfer_test
    SRCS
      worker_hierarchy_kv_cache_transfer_test.cpp
    DEPS
      :runtime
      :kv_cache
      :platform
      GTest::gtest_main
  )
endif()

if(USE_MLU)
  cc_test(
    NAME
      mlu_graph_executor_test
    SRCS
      mlu_graph_executor_test.cpp
      mlu_graph_planner_test.cpp
      mlu_linear_state_restore_worker_test.cpp
    DEPS
      :runtime
      :model_loader
      :batch
      :kv_cache
      :platform
      GTest::gtest_main
      torch_mlu
  )

  # Forked ranks select separate logical MLUs before creating a real CNCL group.
  cc_test(
    NAME
      mlu_graph_dp_collective_test
    SRCS
      mlu_graph_dp_collective_test.cpp
    DEPS
      :runtime
      :parallel_state
      :model_loader
      :batch
      :kv_cache
      :platform
      :util
      GTest::gtest_main
      torch_mlu
      glog::glog
  )
  set_tests_properties(
    MluGraphDpCollectiveTest.AllRanksCaptureThenReplay
    MluGraphDpCollectiveTest.OneRankRejectsSoAllRunEager
    MluGraphDpCollectiveTest.DifferentPlansMakeAllRunEager
    MluGraphDpCollectiveTest.DifferentEmbeddingPresenceMakesAllRunEager
    MluGraphDpCollectiveTest.PartialCacheHitRecapturesThenReplays
    MluGraphDpCollectiveTest.VlmAllRanksCaptureThenReplay
    MluGraphDpCollectiveTest.VlmOneRankRejectsSoAllRunEager
    MluGraphDpCollectiveTest.VlmPartialCacheHitRecapturesThenReplays
    PROPERTIES RUN_SERIAL TRUE TIMEOUT 150)
endif()

if(USE_NPU OR USE_MLU)
  cc_test(
    NAME
      mtp_host_offload_test
    SRCS
      mtp_host_offload_test.cpp
    DEPS
      :runtime
      :model_loader
      :batch
      :kv_cache
      :platform
      GTest::gtest_main
  )
  if(USE_MLU)
    target_link_libraries(mtp_host_offload_test PRIVATE torch_mlu)
  endif()
endif()

if(USE_CUDA)
  cc_test(
    NAME
      cuda_graph_executor_test
    SRCS
      cuda_graph_executor_test.cpp
    DEPS
      :config
      :runtime
      :cuda_layers
      :batch
      :block
      :kv_cache
      :kernels
      GTest::gtest_main
      torch
  )
endif()

if(USE_DCU)
  cc_test(
    NAME
      dcu_graph_executor_test
    SRCS
      dcu_graph_executor_test.cpp
    DEPS
      :runtime
      :batch
      :block
      :kv_cache
      :kernels
      GTest::gtest_main
      torch
      hip::host
  )
endif()
