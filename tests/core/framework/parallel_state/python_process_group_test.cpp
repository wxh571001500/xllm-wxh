/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/parallel_state/python_process_group.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace xllm {
namespace {

const PythonProcessGroupSpec& find_spec(
    const std::vector<PythonProcessGroupSpec>& specs,
    const std::string& name) {
  auto spec_it = std::find_if(
      specs.begin(), specs.end(), [&name](const PythonProcessGroupSpec& spec) {
        return spec.name == name;
      });
  if (spec_it == specs.end()) {
    ADD_FAILURE() << "Missing process group spec " << name;
    return specs.front();
  }
  return *spec_it;
}

TEST(PythonProcessGroupTest, BuildsOrthogonalLlmGroups) {
  const std::vector<PythonProcessGroupSpec> specs =
      build_python_process_group_specs(/*global_rank=*/5,
                                       /*world_size=*/8,
                                       /*dp_size=*/2,
                                       /*ep_size=*/4,
                                       /*cp_size=*/2,
                                       /*enable_encoder_dp=*/true);

  EXPECT_EQ(find_spec(specs, "world").ranks,
            (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(find_spec(specs, "tp").ranks, (std::vector<int32_t>{4, 5, 6, 7}));
  EXPECT_EQ(find_spec(specs, "attention_tp").ranks,
            (std::vector<int32_t>{4, 5}));
  EXPECT_EQ(find_spec(specs, "dp").ranks, (std::vector<int32_t>{1, 5}));
  EXPECT_EQ(find_spec(specs, "cp").ranks, (std::vector<int32_t>{5, 7}));
  EXPECT_EQ(find_spec(specs, "moe_tp").ranks, (std::vector<int32_t>{4, 5}));
  EXPECT_EQ(find_spec(specs, "moe_ep").ranks,
            (std::vector<int32_t>{1, 3, 5, 7}));
  EXPECT_EQ(find_spec(specs, "moe_ep").local_rank, 2);

  const PythonProcessGroupSpec& encoder_dp = find_spec(specs, "encoder_dp");
  EXPECT_TRUE(encoder_dp.alias_of.empty());
  EXPECT_EQ(encoder_dp.ranks, find_spec(specs, "tp").ranks);
  EXPECT_NE(encoder_dp.group_id, find_spec(specs, "tp").group_id);

  EXPECT_EQ(find_spec(specs, "attn_tp").alias_of, "attention_tp");
  EXPECT_EQ(find_spec(specs, "ep").alias_of, "moe_ep");
  EXPECT_EQ(find_spec(specs, "embedding").alias_of, "tp");
  EXPECT_EQ(find_spec(specs, "lm_head").alias_of, "tp");
}

TEST(PythonProcessGroupTest, UsesAliasesForSingleRankDimensions) {
  const std::vector<PythonProcessGroupSpec> specs =
      build_python_process_group_specs(/*global_rank=*/0,
                                       /*world_size=*/1,
                                       /*dp_size=*/1,
                                       /*ep_size=*/1,
                                       /*cp_size=*/1,
                                       /*enable_encoder_dp=*/false);

  EXPECT_EQ(find_spec(specs, "single").alias_of, "tp");
  EXPECT_EQ(find_spec(specs, "attention_tp").alias_of, "tp");
  EXPECT_EQ(find_spec(specs, "dp").alias_of, "single");
  EXPECT_EQ(find_spec(specs, "cp").alias_of, "single");
  EXPECT_EQ(find_spec(specs, "moe_tp").alias_of, "world");
  EXPECT_EQ(find_spec(specs, "moe_ep").alias_of, "single");
  EXPECT_EQ(find_spec(specs, "encoder_dp").alias_of, "tp");
}

TEST(PythonProcessGroupTest, AssignsUniquePhysicalGroupIds) {
  const std::vector<PythonProcessGroupSpec> specs =
      build_python_process_group_specs(/*global_rank=*/5,
                                       /*world_size=*/8,
                                       /*dp_size=*/2,
                                       /*ep_size=*/4,
                                       /*cp_size=*/2,
                                       /*enable_encoder_dp=*/true);

  std::unordered_set<std::string> group_ids;
  for (const PythonProcessGroupSpec& spec : specs) {
    if (!spec.alias_of.empty()) {
      continue;
    }
    EXPECT_FALSE(spec.group_id.empty());
    EXPECT_TRUE(group_ids.emplace(spec.group_id).second);
  }
}

}  // namespace
}  // namespace xllm
