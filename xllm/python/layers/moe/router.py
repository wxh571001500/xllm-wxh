# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable expert routing strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from xllm.python.layers.moe.types import MoERouterConfig, MoERoutingResult


class MoERouter(ABC):
    """Select routed experts from model-produced router logits."""

    @abstractmethod
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
    ) -> MoERoutingResult:
        raise NotImplementedError


class GroupedTopKRouter(MoERouter):
    """Grouped top-k router with native and fused implementations."""

    def __init__(self, config: MoERouterConfig) -> None:
        if config.top_k <= 0 or config.top_k > config.num_experts:
            raise ValueError("MoE top_k must be within the number of experts")
        if config.use_grouped_topk:
            if config.num_expert_group <= 0:
                raise ValueError("MoE num_expert_group must be positive")
            if config.num_experts % config.num_expert_group != 0:
                raise ValueError("MoE experts must divide evenly into groups")
            if not 0 < config.topk_group <= config.num_expert_group:
                raise ValueError("MoE topk_group must be within expert groups")
        self.config = config

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
    ) -> MoERoutingResult:
        del hidden_states
        if self._can_use_fused_topk(router_logits):
            return self.select_experts_fused(router_logits, correction_bias)
        return self._select_experts_native(router_logits, correction_bias)

    def _can_use_fused_topk(self, router_logits: torch.Tensor) -> bool:
        return router_logits.device.type in ("npu", "privateuseone") and hasattr(
            torch.ops._C_ascend,
            "moe_gating_top_k",
        )

    def select_experts_fused(
        self,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
    ) -> MoERoutingResult:
        if self.config.scoring_func == "softmax":
            norm_type = 0
        elif self.config.scoring_func == "sigmoid":
            norm_type = 1
        else:
            raise ValueError(f"Unsupported MoE router activation: {self.config.scoring_func}")
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
            router_logits,
            k=self.config.top_k,
            k_group=self.config.topk_group if self.config.use_grouped_topk else 1,
            group_count=(self.config.num_expert_group if self.config.use_grouped_topk else 1),
            group_select_mode=1,
            renorm=int(self.config.renormalize),
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=self.config.routed_scaling_factor,
            eps=1e-20,
            bias_opt=(correction_bias.to(router_logits) if correction_bias is not None else None),
        )
        return MoERoutingResult(
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights,
        )

    def _select_experts_native(
        self,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None,
    ) -> MoERoutingResult:
        if self.config.scoring_func == "softmax":
            scores = torch.softmax(router_logits, dim=-1)
        elif self.config.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
        else:
            raise ValueError(f"Unsupported MoE router activation: {self.config.scoring_func}")

        selection = scores
        if correction_bias is not None:
            selection = selection + correction_bias.to(dtype=scores.dtype)
        if self.config.use_grouped_topk:
            selection = self._mask_unselected_groups(selection)

        topk_ids = torch.topk(
            selection,
            self.config.top_k,
            dim=-1,
        ).indices
        topk_weights = scores.gather(-1, topk_ids)
        if self.config.renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-20)
        topk_weights = topk_weights * self.config.routed_scaling_factor
        return MoERoutingResult(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    def _mask_unselected_groups(self, selection: torch.Tensor) -> torch.Tensor:
        num_tokens = selection.shape[0]
        experts_per_group = self.config.num_experts // self.config.num_expert_group
        grouped = selection.view(
            num_tokens,
            self.config.num_expert_group,
            experts_per_group,
        )
        group_score_width = min(2, experts_per_group)
        group_scores = torch.topk(
            grouped,
            k=group_score_width,
            dim=-1,
        ).values.sum(dim=-1)
        selected_groups = torch.topk(
            group_scores,
            k=self.config.topk_group,
            dim=-1,
        ).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(selection)
        return selection.masked_fill(~expert_mask, torch.finfo(selection.dtype).min)


__all__ = ["GroupedTopKRouter", "MoERouter"]
