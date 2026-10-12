include(cc_test)

add_subdirectory(dit)

if(USE_NPU)
  cc_test(
    NAME
      model_registry_test
    CPU_ONLY
    SRCS
      model_registry_test.cpp
    DEPS
      models
      GTest::gtest_main
  )
  target_link_libraries(model_registry_test
                        PUBLIC $<$<BOOL:${USE_NPU}>:ascendcl>
                               $<$<BOOL:${USE_NPU}>:hccl>
                               $<$<BOOL:${USE_NPU}>:c_sec>
                               $<$<BOOL:${USE_NPU}>:nnopbase>)

  cc_test(
    NAME
      py_causal_lm_test
    SRCS
      py_causal_lm_test.cpp
    DEPS
      models
      GTest::gtest_main
    ENVIRONMENT
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
  )
  target_link_libraries(py_causal_lm_test
                        PUBLIC Python::Python
                               ascendcl
                               hccl
                               c_sec
                               nnopbase
                               model_context
                               npu_layers)
endif()

if(USE_MLU)
  # Uses CPU tensors and forks before any MLU device initialization.
  cc_test(
    NAME
      glm5_next_mlu_test
    SRCS
      glm5_next_mlu_test.cpp
    DEPS
      models
      GTest::gtest_main
  )

  # Forks eight ranks before any MLU device initialization.
  cc_test(
    NAME
      glm5_next_mlu_pcp_regression_test
    SRCS
      glm5_next_mlu_pcp_regression_test.cpp
    DEPS
      models
      :mlu_layers
      :common_layers
      :parallel_state
      :model_context
      :state_dict
      GTest::gtest_main
      torch_mlu
  )

  cc_test(
    NAME
      qwen3_5_mlu_forward_test
    SRCS
      qwen3_5_mlu_forward_test.cpp
    DEPS
      models
      :mlu_layers
      :common_layers
      :parallel_state
      :model_context
      GTest::gtest_main
      torch_mlu
  )
endif()
