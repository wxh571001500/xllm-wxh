// Restores the process's original value of a behavior-selecting environment
// variable on scope exit, including the parent side of a death-test fork
// (the child's fatal failure never runs a destructor; the parent re-enters
// the scope after the fork and restores there).
//
// Shared by the transfer-engine test TUs that pin an environment-variable
// driven behavior (XLLM_CP_INDEX_WRITE_MODE, MC_TCP_PROTO, ...). A plain
// setenv/unsetenv pair would wipe a value an exported test shell relied on,
// and per-TU copies of this guard drift (the death-test fork visibility fix
// once had to be applied per copy). The pre-existing ScopedEnvVar helpers in
// kv_cache_store_test.cpp / mapping_npu_test.cpp / mlu_graph_executor_test.cpp
// predate this header and can converge onto it incrementally.
#ifndef XLLM_TESTS_KV_CACHE_TRANSFER_SCOPED_ENVIRONMENT_VARIABLE_H_
#define XLLM_TESTS_KV_CACHE_TRANSFER_SCOPED_ENVIRONMENT_VARIABLE_H_

#include <cstdlib>
#include <optional>
#include <string>

namespace xllm::tests {

class ScopedEnvironmentVariable final {
 public:
  explicit ScopedEnvironmentVariable(const char* name) : name_(name) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      original_value_ = value;
    }
  }

  ~ScopedEnvironmentVariable() {
    if (original_value_.has_value()) {
      setenv(name_.c_str(), original_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  bool set(const char* value) {
    return setenv(name_.c_str(), value, /*overwrite=*/1) == 0;
  }

 private:
  std::string name_;
  std::optional<std::string> original_value_;
};

}  // namespace xllm::tests

#endif  // XLLM_TESTS_KV_CACHE_TRANSFER_SCOPED_ENVIRONMENT_VARIABLE_H_
