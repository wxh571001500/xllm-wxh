/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"

namespace xllm {

enum class SpeechToTextTask : int8_t {
  TRANSCRIBE = 0,
  TRANSLATE = 1,
};

struct SpeechToTextParams {
  SpeechToTextTask task = SpeechToTextTask::TRANSCRIBE;
  std::string language;
  std::string request_prompt;
  std::string hotwords;
  std::string mime_type;
  size_t audio_bytes = 0;
};

// Model-specific request adaptation; inference stays in the VLM engine.
class SpeechModelInterface {
 public:
  virtual ~SpeechModelInterface() = default;

  virtual bool supports_translation() const = 0;
  virtual bool supports_verbose_json() const = 0;
  virtual bool supports_segment_timestamps() const = 0;
  virtual Status validate_language(const std::string& language) const = 0;
  virtual std::vector<Message> build_messages(
      const SpeechToTextParams& params) const = 0;
};

using SpeechModelFactory =
    std::function<std::shared_ptr<const SpeechModelInterface>()>;

inline std::unordered_map<std::string, SpeechModelFactory>&
speech_model_registry() {
  static std::unordered_map<std::string, SpeechModelFactory> registry;
  return registry;
}

inline void register_speech_model_factory(const std::string& name,
                                          SpeechModelFactory factory) {
  std::unordered_map<std::string, SpeechModelFactory>& registry =
      speech_model_registry();
  if (registry[name] != nullptr) {
    LOG(WARNING) << "speech model factory for " << name
                 << " already registered.";
  } else {
    registry[name] = std::move(factory);
  }
}

inline SpeechModelFactory get_speech_model_factory(const std::string& name) {
  const std::unordered_map<std::string, SpeechModelFactory>& registry =
      speech_model_registry();
  const auto it = registry.find(name);
  return it == registry.end() ? nullptr : it->second;
}

}  // namespace xllm

#define REGISTER_SPEECH_MODEL_WITH_VARNAME(                                 \
    VarName, ModelType, SpeechModelClass)                                   \
  const bool VarName##_speech_model_registered = []() {                     \
    xllm::register_speech_model_factory(                                    \
        #ModelType, []() { return std::make_shared<SpeechModelClass>(); }); \
    return true;                                                            \
  }()

#define REGISTER_SPEECH_MODEL(ModelType, SpeechModelClass) \
  REGISTER_SPEECH_MODEL_WITH_VARNAME(ModelType, ModelType, SpeechModelClass)
