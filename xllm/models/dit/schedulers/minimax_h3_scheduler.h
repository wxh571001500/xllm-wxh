/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "core/framework/dit_model_loader.h"

namespace xllm {

class MiniMaxH3Scheduler final {
 public:
  explicit MiniMaxH3Scheduler(double shift = 12.0) { set_shift(shift); }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr) << "MiniMax-H3 scheduler loader must not be null";
    const std::string config_path =
        loader->model_weights_path() + "/scheduler_config.json";
    std::ifstream input(config_path);
    if (!input.good()) {
      LOG(WARNING) << "MiniMax-H3 scheduler config not found: " << config_path
                   << ", using shift=" << shift_;
      return;
    }
    nlohmann::json config;
    input >> config;
    if (config.contains("shift")) {
      const double shift = config.at("shift").get<double>();
      set_shift(shift);
    }
  }

  void set_shift(double shift) {
    CHECK_GT(shift, 0.0) << "MiniMax-H3 scheduler shift must be positive";
    shift_ = shift;
  }

  double shift() const { return shift_; }

  void set_timesteps(int64_t num_inference_steps, const torch::Device& device) {
    CHECK_GE(num_inference_steps, 2)
        << "MiniMax-H3 scheduler requires at least two sigma grid points";
    torch::Tensor base =
        torch::linspace(1.0,
                        0.0,
                        num_inference_steps,
                        torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor sigmas = shift_ * base / (1.0 + (shift_ - 1.0) * base);
    sigmas = std::get<0>(torch::unique_consecutive(sigmas));
    sigmas_ = sigmas.to(device);
    timesteps_ =
        (1.0 - sigmas.slice(/*dim=*/0, 0, sigmas.size(0) - 1)).to(device);
    step_index_ = -1;
    begin_index_ = -1;
  }

  void set_sigmas(const torch::Tensor& sigmas, const torch::Device& device) {
    CHECK_GE(sigmas.numel(), 2)
        << "MiniMax-H3 explicit sigmas need >= 2 values";
    torch::Tensor cpu_sigmas =
        sigmas.flatten().to(torch::kCPU).to(torch::kFloat32);
    CHECK_EQ(cpu_sigmas[-1].item<float>(), 0.0f)
        << "MiniMax-H3 explicit sigmas must end at 0";
    sigmas_ = cpu_sigmas.to(device);
    timesteps_ =
        (1.0 - cpu_sigmas.slice(0, 0, cpu_sigmas.size(0) - 1)).to(device);
    step_index_ = -1;
    begin_index_ = -1;
  }

  const torch::Tensor& timesteps() const { return timesteps_; }
  const torch::Tensor& sigmas() const { return sigmas_; }

  void set_begin_index(int64_t begin_index) { begin_index_ = begin_index; }

  int64_t index_for_timestep(const torch::Tensor& timestep) const {
    CHECK(timesteps_.defined()) << "MiniMax-H3 scheduler timesteps are not set";
    torch::Tensor matches =
        (timesteps_ == timestep.to(timesteps_.device())).nonzero();
    CHECK_GT(matches.numel(), 0) << "MiniMax-H3 timestep is not in schedule";
    return matches[0].item<int64_t>();
  }

  torch::Tensor scale_noise(const torch::Tensor& sample,
                            const torch::Tensor& timestep,
                            const torch::Tensor& noise) const {
    torch::Tensor t = timestep.to(sample.device(), sample.scalar_type());
    while (t.dim() < sample.dim()) {
      t = t.unsqueeze(-1);
    }
    return t * sample + (1.0 - t) * noise;
  }

  torch::Tensor step(const torch::Tensor& model_output,
                     const torch::Tensor& timestep,
                     const torch::Tensor& sample) {
    CHECK(timestep.is_floating_point()) << "MiniMax-H3 scheduler.step expects "
                                           "timestep value, not integer index";
    if (step_index_ < 0) {
      step_index_ =
          begin_index_ >= 0 ? begin_index_ : index_for_timestep(timestep);
    }
    CHECK_LT(step_index_ + 1, sigmas_.size(0))
        << "MiniMax-H3 scheduler step out of range";

    torch::Tensor sigma_from_t =
        1.0 - timestep.to(sample.device(), sample.scalar_type());
    while (sigma_from_t.dim() < sample.dim()) {
      sigma_from_t = sigma_from_t.unsqueeze(-1);
    }
    torch::Tensor denoised = sample + sigma_from_t * model_output;

    torch::Dtype compute_dtype = (sample.scalar_type() == torch::kFloat16 ||
                                  sample.scalar_type() == torch::kBFloat16)
                                     ? torch::kFloat32
                                     : sample.scalar_type();
    torch::Tensor sigma =
        sigmas_[step_index_].to(sample.device(), compute_dtype);
    torch::Tensor sigma_next =
        sigmas_[step_index_ + 1].to(sample.device(), compute_dtype);
    torch::Tensor ratio = sigma_next / sigma;
    torch::Tensor prev = ratio * sample.to(compute_dtype) +
                         (1.0 - ratio) * denoised.to(compute_dtype);
    ++step_index_;
    return prev.to(sample.scalar_type());
  }

 private:
  double shift_{12.0};
  torch::Tensor sigmas_;
  torch::Tensor timesteps_;
  int64_t step_index_{-1};
  int64_t begin_index_{-1};
};

}  // namespace xllm
