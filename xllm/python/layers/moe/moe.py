# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic and Kimi K3 model-level MoE compositions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn

from xllm.python import ops
from xllm.python.layers.moe.activation import SituAndMul
from xllm.python.layers.moe.communication import (
    MoECommMethod,
    TensorParallelCommMethod,
)
from xllm.python.layers.moe.experts import (
    FusedQuantizedSituAndMul,
    FusedW4A8RoutedExperts,
    RoutedExperts,
    UnquantizedRoutedExperts,
)
from xllm.python.layers.moe.router import GroupedTopKRouter
from xllm.python.layers.moe.runner import MoERunner
from xllm.python.layers.moe.token_dispatcher import (
    FusedAllGatherTokenDispatcher,
    NativeTokenDispatcher,
)
from xllm.python.layers.moe.types import MoEExpertsConfig, MoERouterConfig

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


class MoE(nn.Module):
    """Reusable model-level MoE with common routing and weight ownership."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        router_config: MoERouterConfig,
        experts: RoutedExperts,
        comm_method: MoECommMethod,
        dtype: torch.dtype,
        device: torch.device,
        shared_experts: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if router_config.num_experts != num_experts:
            raise ValueError("MoE router and expert counts must match")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.gate = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=dtype, device=device)
        )
        self.experts = experts
        self.shared_experts = shared_experts
        self._router = GroupedTopKRouter(router_config)
        self._runner = MoERunner(self._router, comm_method)
        self._loaded_components: set[str] = set()

    def _topk(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing = self._router.select_experts(
            hidden_states=router_logits,
            router_logits=router_logits,
            correction_bias=self.gate.e_score_correction_bias,
        )
        return routing.topk_ids, routing.topk_weights

    def _fused_topk(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing = self._router.select_experts_fused(
            router_logits,
            self.gate.e_score_correction_bias,
        )
        return routing.topk_ids, routing.topk_weights

    def _routed_input_transform(self) -> TensorTransform | None:
        return None

    def _routed_output_transform(self) -> TensorTransform | None:
        return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        output = self._runner.forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=self.gate.e_score_correction_bias,
            experts=self.experts,
            shared_experts=self.shared_experts,
            routed_input_transform=self._routed_input_transform(),
            routed_output_transform=self._routed_output_transform(),
        )
        return output.reshape(original_shape)

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        if name == "gate.weight":
            target = self.gate.weight
            if tensor.shape != target.shape:
                raise ValueError(
                    f"MoE gate expects {target.shape}, got {tensor.shape}"
                )
            target.data.copy_(tensor.to(target))
            self._loaded_components.add(name)
            return True
        if name in ("e_score_correction_bias", "gate.e_score_correction_bias"):
            target = self.gate.e_score_correction_bias
            if tensor.shape != target.shape:
                raise ValueError(
                    f"MoE correction bias expects {target.shape}, "
                    f"got {tensor.shape}"
                )
            target.data.copy_(tensor.to(target))
            self._loaded_components.add("gate.e_score_correction_bias")
            return True
        return False

    def load_weights(
        self,
        state_dict: "StateDict",
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded: set[str] = set()
        for name in ("gate.weight", "gate.e_score_correction_bias"):
            if state_dict.has(name) and self.load_weight(
                name,
                state_dict.get_tensor(name),
            ):
                loaded.add(name)

        shared_state_dict = state_dict.get_dict_with_prefix("shared_experts.")
        if self.shared_experts is not None and shared_state_dict.size() > 0:
            child_loaded = self.shared_experts.load_weights(
                shared_state_dict,
                tp_rank,
                tp_size,
            )
            loaded.update(f"shared_experts.{name}" for name in child_loaded)

        experts_state_dict = state_dict.get_dict_with_prefix("experts.")
        if experts_state_dict.size() > 0:
            child_loaded = self.experts.load_weights(
                experts_state_dict,
                tp_rank,
                tp_size,
            )
            loaded.update(f"experts.{name}" for name in child_loaded)
        return loaded

    def _required_weight_names(self) -> set[str]:
        return {
            "gate.weight",
            "gate.e_score_correction_bias",
        }

    def _weight_modules_to_finalize(
        self,
    ) -> tuple[nn.Module | None, ...]:
        return self.shared_experts, self.experts

    def finish_weight_loading(self) -> None:
        missing = self._required_weight_names().difference(
            self._loaded_components
        )
        if missing:
            raise KeyError(f"MoE weights are missing: {sorted(missing)}")
        for module in self._weight_modules_to_finalize():
            if module is not None and hasattr(module, "finish_weight_loading"):
                module.finish_weight_loading()


class KimiK3MoERunner(MoERunner):
    """Kimi policy for latent routed and shared expert TP reductions."""

    def __init__(
        self,
        router: GroupedTopKRouter,
        comm_method: MoECommMethod,
        tp_size: int,
    ) -> None:
        super().__init__(router, comm_method)
        self._tp_size = tp_size

    def _reduce_routed_results(self) -> bool:
        return True

    def _finalize_shared_expert_output(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._tp_size > 1:
            ops.all_reduce_(hidden_states)
        return hidden_states


class KimiK3MoE(MoE):
    """Kimi latent MoE extending the reusable model-level implementation."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int,
        tp_rank: int,
        routed_expert_down_proj: nn.Module,
        routed_expert_up_proj: nn.Module,
        shared_experts: nn.Module | None = None,
        quantized: bool = False,
    ) -> None:
        num_experts = int(config.num_experts)
        top_k = int(config.num_experts_per_token)
        hidden_size = int(config.hidden_size)
        routed_hidden_size = int(config.routed_expert_hidden_size)
        experts_config = MoEExpertsConfig(
            num_experts=num_experts,
            hidden_size=routed_hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        situ_beta = float(getattr(config, "activation_situ_beta", None) or 1.0)
        situ_linear_beta = getattr(config, "activation_situ_linear_beta", None)
        if quantized:
            experts: RoutedExperts = FusedW4A8RoutedExperts(
                config=experts_config,
                activation=FusedQuantizedSituAndMul(
                    beta=situ_beta,
                    linear_beta=situ_linear_beta,
                ),
                dtype=dtype,
                device=device,
            )
        else:
            experts = UnquantizedRoutedExperts(
                config=experts_config,
                activation=SituAndMul(
                    beta=situ_beta,
                    linear_beta=situ_linear_beta,
                ),
                dtype=dtype,
                device=device,
            )

        router_config = MoERouterConfig(
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=str(config.moe_router_activation_func),
            renormalize=bool(config.moe_renormalize),
            routed_scaling_factor=float(config.routed_scaling_factor),
            use_grouped_topk=bool(getattr(config, "use_grouped_topk", True)),
            num_expert_group=int(getattr(config, "num_expert_group", 1)),
            topk_group=int(getattr(config, "topk_group", 1)),
        )
        if quantized or device.type in ("npu", "privateuseone"):
            token_dispatcher = FusedAllGatherTokenDispatcher(
                num_experts=num_experts,
                top_k=top_k,
                quantized=quantized,
            )
        else:
            token_dispatcher = NativeTokenDispatcher(num_experts)
        comm_method = TensorParallelCommMethod(
            tp_size=tp_size,
            token_dispatcher=token_dispatcher,
        )
        super().__init__(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_config=router_config,
            experts=experts,
            comm_method=comm_method,
            dtype=dtype,
            device=device,
            shared_experts=shared_experts,
        )

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.quantized = quantized
        self.routed_expert_down_proj = routed_expert_down_proj
        self.routed_expert_up_proj = routed_expert_up_proj
        self.routed_expert_norm = (
            nn.RMSNorm(
                routed_hidden_size,
                eps=float(config.rms_norm_eps),
                dtype=dtype,
                device=device,
            )
            if bool(getattr(config, "latent_moe_use_norm", False))
            else None
        )
        self._runner = KimiK3MoERunner(
            self._router,
            comm_method,
            tp_size,
        )

    def _routed_input_transform(self) -> TensorTransform:
        return self.routed_expert_down_proj

    def _routed_output_transform(self) -> TensorTransform:
        return self._transform_routed_output

    def _transform_routed_output(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.routed_expert_norm is not None:
            hidden_states = self.routed_expert_norm(hidden_states)
        result = self.routed_expert_up_proj(hidden_states)
        if isinstance(result, tuple):
            return result[0]
        return result

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        if super().load_weight(name, tensor):
            return True
        for prefix, module in (
            ("routed_expert_down_proj.", self.routed_expert_down_proj),
            ("routed_expert_up_proj.", self.routed_expert_up_proj),
        ):
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if hasattr(module, "load_weight"):
                loaded = bool(module.load_weight(suffix, tensor))
                if loaded:
                    self._loaded_components.add(name)
                return loaded
            if suffix != "weight":
                return False
            target = module.weight
            if tensor.shape != target.shape:
                raise ValueError(
                    f"Kimi K3 weight {name} expects {target.shape}, "
                    f"got {tensor.shape}"
                )
            target.data.copy_(tensor.to(target))
            self._loaded_components.add(name)
            return True
        if name == "routed_expert_norm.weight" and self.routed_expert_norm is not None:
            target = self.routed_expert_norm.weight
            if tensor.shape != target.shape:
                raise ValueError(
                    f"Kimi K3 weight {name} expects {target.shape}, "
                    f"got {tensor.shape}"
                )
            target.data.copy_(tensor.to(target))
            self._loaded_components.add(name)
            return True
        return False

    def load_weights(
        self,
        state_dict: "StateDict",
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded = super().load_weights(state_dict, tp_rank, tp_size)
        direct_names = [
            "routed_expert_down_proj.weight",
            "routed_expert_up_proj.weight",
        ]
        if self.quantized:
            for projection in (
                "routed_expert_down_proj",
                "routed_expert_up_proj",
            ):
                direct_names.extend(
                    f"{projection}.{suffix}"
                    for suffix in ("weight_scale", "weight_offset")
                )
        if self.routed_expert_norm is not None:
            direct_names.append("routed_expert_norm.weight")
        for name in direct_names:
            if state_dict.has(name) and self.load_weight(
                name,
                state_dict.get_tensor(name),
            ):
                loaded.add(name)
        return loaded

    def _required_weight_names(self) -> set[str]:
        required = super()._required_weight_names()
        required.update(
            {
                "routed_expert_down_proj.weight",
                "routed_expert_up_proj.weight",
            }
        )
        if self.quantized:
            for projection in (
                "routed_expert_down_proj",
                "routed_expert_up_proj",
            ):
                required.update(
                    f"{projection}.{suffix}"
                    for suffix in ("weight_scale", "weight_offset")
                )
        if self.routed_expert_norm is not None:
            required.add("routed_expert_norm.weight")
        return required

    def _weight_modules_to_finalize(
        self,
    ) -> tuple[nn.Module | None, ...]:
        return (
            self.routed_expert_down_proj,
            self.routed_expert_up_proj,
            *super()._weight_modules_to_finalize(),
        )


__all__ = ["KimiK3MoE", "KimiK3MoERunner", "MoE"]
