# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

include(cc_test)
get_filename_component(_xllm_build_import_root "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)
cc_test(
  NAME kpool_kernel_test
  SRCS update_test.cpp score_test.cpp expand_test.cpp
  DEPS :mlu_kernels :triton_jit torch glog::glog GTest::gtest_main
  ENVIRONMENT "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)
