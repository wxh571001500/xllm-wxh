include(cc_test)

cc_test(
  NAME
    host_kv_transfer_test
  CPU_ONLY
  SRCS
    host_transfer/transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    push_route_test
  CPU_ONLY
  SRCS
    push_route_test.cpp
  DEPS
    :push_route
    GTest::gtest_main
)

cc_test(
  NAME
    pd_topology_guard_test
  CPU_ONLY
  SRCS
    pd_topology_guard_test.cpp
  DEPS
    :pd_topology_guard
    GTest::gtest_main
)

cc_test(
  NAME
    kv_transfer_completion_test
  CPU_ONLY
  SRCS
    kv_transfer_completion_test.cpp
  DEPS
    :config
    :kv_transfer_failure
    :kv_transfer_completion
    :parallel_state
    :platform
    torch
    GTest::gtest_main
)

# Rescan the config/platform/util cycle for util's ModelConfig dependency.
target_link_libraries(kv_transfer_completion_test PRIVATE
  "$<LINK_GROUP:RESCAN,config,platform,util>")

cc_test(
  NAME
    prefetch_result_test
  CPU_ONLY
  SRCS
    prefetch_result_test.cpp
  DEPS
    :util
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    prefetch_tail_probe_test
  CPU_ONLY
  SRCS
    prefetch_tail_probe_test.cpp
  DEPS
    :util
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    reshard_planner_test
  CPU_ONLY
  SRCS
    reshard_planner_test.cpp
  DEPS
    :reshard_planner
    :kv_cache
    GTest::gtest_main
)

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    mooncake_transfer_engine_test
  SRCS
    mooncake_transfer_engine_test.cpp
  DEPS
    :kv_cache_transfer
    :xllm_server
    GTest::gtest_main
)

# Resolve static link order between xtensor and xllm_server for this test target.
target_link_libraries(mooncake_transfer_engine_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
endif()

# Drives the source->destination KV block mapping for a P kv_split_size that
# differs from the destination's declared width. Needs the batch builder, the
# block manager and the scheduler's group-type predicate, all reachable through
# xllm_server; the source declares kv_split_size = 2 while the settled decode
# declares 1.
# CPU_ONLY like the sibling host-only targets in this directory: the whole TU
# builds BlockManagerImpl/Sequence/InstanceInfo records and calls the width
# planning helpers directly -- no device, no kernel, no RPC -- so it must not
# inherit the NPU bootstrap sources nor drop out of the `cpu` ctest label set.
cc_test(
  NAME
    kv_split_width_mismatch_test
  CPU_ONLY
  SRCS
    kv_split_width_mismatch_test.cpp
  DEPS
    :kv_cache_transfer
    :xllm_server
    GTest::gtest_main
)

target_link_libraries(kv_split_width_mismatch_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")

if(USE_MLU)
cc_test(
  NAME
    compact_load_test
  SRCS
    host_transfer/compact_load_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    compact_offload_test
  SRCS
    host_transfer/compact_offload_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

endif()

cc_test(
  NAME
    kv_cache_store_test
  CPU_ONLY
  SRCS
    kv_cache_store_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    basic_host_kv_transfer_test
  SRCS
    host_transfer/basic_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    hierarchy_kv_cache_transfer_test
  SRCS
    hierarchy_kv_cache_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)

cc_test(
  NAME
    kv_split_host_transfer_test
  SRCS
    kv_split_host_transfer_test.cpp
  DEPS
    :kv_cache_transfer
    GTest::gtest_main
)
endif()
