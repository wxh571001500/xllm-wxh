include(cc_test)

cc_test(
  NAME
    kimi_k3_reasoning_parser_test
  SRCS
    kimi_k3_reasoning_parser_test.cpp
  DEPS
    :reasoning
    GTest::gtest_main
)
