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

#include "core/framework/model/model_args.h"
#include "processors/qwen2_vl_image_processor.h"

namespace xllm {

// Qwen3-Omni image processor: reuses the Qwen2VL patchify pipeline and only
// overrides the resize pixel bounds with a token budget read from the
// ``image_max_tokens_num`` gflag (token count -> pixels via factor²).
class Qwen3OmniImageProcessor final : public Qwen2VLImageProcessor {
 public:
  explicit Qwen3OmniImageProcessor(const ModelArgs& args);
};

}  // namespace xllm
