/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/parallel_state/python_process_group.h"

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {
namespace {

PythonProcessGroupSpec make_group_spec(const std::string& name,
                                       std::vector<int32_t> ranks,
                                       int32_t global_rank) {
  auto rank_it = std::find(ranks.begin(), ranks.end(), global_rank);
  CHECK(rank_it != ranks.end())
      << "Global rank " << global_rank << " is not in Python group " << name;

  PythonProcessGroupSpec spec;
  spec.name = name;
  spec.local_rank = static_cast<int32_t>(rank_it - ranks.begin());
  spec.group_id = "xllm_python_" + name + "_" + std::to_string(ranks.front());
  spec.ranks = std::move(ranks);
  return spec;
}

PythonProcessGroupSpec make_alias_spec(const std::string& name,
                                       const PythonProcessGroupSpec& target) {
  PythonProcessGroupSpec spec;
  spec.name = name;
  spec.ranks = target.ranks;
  spec.local_rank = target.local_rank;
  spec.alias_of = target.name;
  return spec;
}

std::vector<int32_t> contiguous_group_ranks(int32_t global_rank,
                                            int32_t group_size) {
  const int32_t group_start = global_rank / group_size * group_size;
  std::vector<int32_t> ranks;
  ranks.reserve(group_size);
  for (int32_t rank = group_start; rank < group_start + group_size; ++rank) {
    ranks.emplace_back(rank);
  }
  return ranks;
}

std::vector<int32_t> strided_group_ranks(int32_t global_rank,
                                         int32_t world_size,
                                         int32_t group_size) {
  const int32_t stride = world_size / group_size;
  const int32_t group_offset = global_rank % stride;
  std::vector<int32_t> ranks;
  ranks.reserve(group_size);
  for (int32_t group_rank = 0; group_rank < group_size; ++group_rank) {
    ranks.emplace_back(group_offset + group_rank * stride);
  }
  return ranks;
}

const PythonProcessGroupSpec& find_group_spec(
    const std::vector<PythonProcessGroupSpec>& specs,
    const std::string& name) {
  auto spec_it = std::find_if(
      specs.begin(), specs.end(), [&name](const PythonProcessGroupSpec& spec) {
        return spec.name == name;
      });
  CHECK(spec_it != specs.end()) << "Missing Python process group " << name;
  return *spec_it;
}

}  // namespace

std::vector<PythonProcessGroupSpec> build_python_process_group_specs(
    int32_t global_rank,
    int32_t world_size,
    int32_t dp_size,
    int32_t ep_size,
    int32_t cp_size,
    bool enable_encoder_dp) {
  CHECK_GT(world_size, 0);
  CHECK_GE(global_rank, 0);
  CHECK_LT(global_rank, world_size);
  CHECK_GT(dp_size, 0);
  CHECK_GT(ep_size, 0);
  CHECK_GT(cp_size, 0);
  CHECK_EQ(world_size % dp_size, 0);
  CHECK_EQ(world_size % ep_size, 0);
  CHECK_EQ(world_size % (dp_size * cp_size), 0);

  const int32_t tp_size = world_size / dp_size;
  const int32_t attention_tp_size = world_size / (dp_size * cp_size);
  const int32_t moe_tp_size = world_size / ep_size;

  std::vector<PythonProcessGroupSpec> specs;
  specs.reserve(13);
  specs.emplace_back(make_group_spec(
      "world", contiguous_group_ranks(global_rank, world_size), global_rank));
  specs.emplace_back(make_group_spec(
      "tp", contiguous_group_ranks(global_rank, tp_size), global_rank));

  if (cp_size == 1) {
    specs.emplace_back(
        make_alias_spec("attention_tp", find_group_spec(specs, "tp")));
  } else {
    specs.emplace_back(
        make_group_spec("attention_tp",
                        contiguous_group_ranks(global_rank, attention_tp_size),
                        global_rank));
  }

  if (tp_size == 1) {
    specs.emplace_back(make_alias_spec("single", find_group_spec(specs, "tp")));
  } else {
    specs.emplace_back(make_group_spec("single", {global_rank}, global_rank));
  }

  if (dp_size == 1) {
    specs.emplace_back(make_alias_spec("dp", find_group_spec(specs, "single")));
  } else {
    specs.emplace_back(
        make_group_spec("dp",
                        strided_group_ranks(global_rank, world_size, dp_size),
                        global_rank));
  }

  if (cp_size == 1) {
    specs.emplace_back(make_alias_spec("cp", find_group_spec(specs, "single")));
  } else {
    specs.emplace_back(
        make_group_spec("cp",
                        parallel_state::compute_cp_group_ranks(
                            global_rank, world_size, dp_size, cp_size),
                        global_rank));
  }

  if (ep_size == 1) {
    specs.emplace_back(
        make_alias_spec("moe_tp", find_group_spec(specs, "world")));
    specs.emplace_back(
        make_alias_spec("moe_ep", find_group_spec(specs, "single")));
  } else {
    specs.emplace_back(
        make_group_spec("moe_tp",
                        contiguous_group_ranks(global_rank, moe_tp_size),
                        global_rank));
    specs.emplace_back(
        make_group_spec("moe_ep",
                        strided_group_ranks(global_rank, world_size, ep_size),
                        global_rank));
  }

  if (enable_encoder_dp) {
    specs.emplace_back(
        make_group_spec("encoder_dp",
                        contiguous_group_ranks(global_rank, tp_size),
                        global_rank));
  } else {
    specs.emplace_back(
        make_alias_spec("encoder_dp", find_group_spec(specs, "tp")));
  }

  specs.emplace_back(
      make_alias_spec("attn_tp", find_group_spec(specs, "attention_tp")));
  specs.emplace_back(make_alias_spec("ep", find_group_spec(specs, "moe_ep")));
  specs.emplace_back(
      make_alias_spec("embedding", find_group_spec(specs, "tp")));
  specs.emplace_back(make_alias_spec("lm_head", find_group_spec(specs, "tp")));
  return specs;
}

}  // namespace xllm
