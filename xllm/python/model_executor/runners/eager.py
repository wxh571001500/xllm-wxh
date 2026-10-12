# Copyright 2026 The xLLM Authors.
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

from __future__ import annotations

import torch

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.cp_utils import build_cp_context
from xllm.python.model_executor.forward_context import (
    EplbRuntimeState,
    ForwardContext,
    LayerLoadContext,
    LayerSynchronizer,
    forward_context,
)
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.base import BaseRunner, ModelExecutionOutput


def _per_seq_lens_from_metadata(
    metadata: AttentionMetadata,
    *,
    include_prefix: bool,
) -> tuple[list[int], list[int]] | None:
    """Per-sequence query and KV lengths for the packed prefill batch.

    Read the host-side, non-cumulative lengths without a D2H copy. MLA chunked
    prefill keeps the full KV length so its cached prefix is visible; non-MLA
    CP preserves the existing query-only contract by using the query length for
    both values. Returns None when a required field is absent.
    """
    q_lens = metadata.q_seq_lens_host
    kv_lens = metadata.kv_seq_lens_host if include_prefix else q_lens
    if q_lens is None or kv_lens is None:
        return None
    return q_lens.tolist(), kv_lens.tolist()


class EagerRunner(BaseRunner):
    # Context-Parallel config, set by ModelExecutor when cp_size > 1. CP shards
    # the prefill sequence across these ranks; decode is left on the non-CP path.
    cp_size: int = 1
    cp_rank: int = 0

    @property
    def model_restores_aux_hidden_under_cp(self) -> bool:
        """Whether the bound model restores the aux-hidden buffer to global
        row order under Context Parallelism, which is what a target-side
        speculative-verification draft (e.g. DFlash2) consumes.

        Having capture layers configured is NOT sufficient: a model may
        capture aux rows yet leave them in CP-sharded order at the model
        exit (glm_moe_dsa merges only ``hidden``; glm5_next and
        deepseek_v4 merge the aux buffer too), and a draft reading
        sharded rows silently consumes token-misaligned hidden.

        Necessary, NOT sufficient. This attribute is the model-side half of
        the admission only: whether a given model+algorithm pairing may run
        spec-verify under CP is decided solely by the C++ admission
        (master.cpp's CP gates + the SpeculativeConfig predicates), the one
        place that knows the algorithm. A shape this runner sees has already
        passed that admission; this gate then fails closed on any model
        that never proved the aux-restore capability (explicit opt-in class
        attribute; anything else -- including every model that never wired
        the attribute -- stays refused).
        """
        return bool(getattr(self.model, "restores_aux_hidden_under_cp", False))

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        eplb: EplbRuntimeState | None = None,
        input_batch: InputBatch | None = None,
        layer_load_context: LayerLoadContext | None = None,
    ) -> ModelExecutionOutput:
        cp_context = None
        is_mla = self.attention_backend.is_mla
        is_empty_rank = bool(getattr(metadata, "is_dummy", False))
        is_mla_cp_prefill = (
            not is_empty_rank and self.cp_size > 1 and is_mla and (metadata.is_prefill or metadata.is_chunked_prefill)
        )
        # B6 (research/b6-spec-verify-v3-cp-safety.md) confirmed the
        # model-level aux-hidden-buffer CP restore -- and, for glm5_next's
        # KDA layers specifically, the ephemeral _spec_verify_v3 scratch-pool
        # mechanism DFlash2's target-verify path uses -- are both CP-safe.
        # Gate on the model's explicit CP aux-restore capability rather
        # than capture configuration: only a model that merges the aux
        # buffer back to global row order (glm5_next, deepseek_v4) feeds a
        # draft token-aligned rows; capture configured but sharded
        # (glm_moe_dsa) or absent must both stay refused.
        if is_mla_cp_prefill and metadata.is_spec_verify and not self.model_restores_aux_hidden_under_cp:
            raise NotImplementedError(
                "Python Context-Parallel does not support this target-side speculative verification path"
            )
        if is_mla_cp_prefill and metadata.is_mixed:
            raise NotImplementedError("Python Context-Parallel does not support mixed batches")

        use_cp_context = (
            not is_empty_rank and self.cp_size > 1 and (metadata.is_prefill or (is_mla and metadata.is_chunked_prefill))
        )
        if use_cp_context:
            seq_lens = _per_seq_lens_from_metadata(
                metadata,
                include_prefix=is_mla,
            )
            if seq_lens is None:
                if is_mla:
                    raise RuntimeError("Python Context-Parallel requires host query and KV sequence lengths")
            else:
                q_seq_lens, kv_seq_lens = seq_lens
                cp_context = build_cp_context(
                    q_seq_lens,
                    kv_seq_lens,
                    self.cp_size,
                    self.cp_rank,
                    self.device,
                )

        execution_contexts = {}
        if self.execution_metadata_builders:
            if input_batch is None:
                raise RuntimeError("execution metadata builders require upstream InputBatch metadata")
            for builder in self.execution_metadata_builders:
                metadata_type = builder.metadata_type
                if metadata_type in execution_contexts:
                    raise RuntimeError(f"duplicate execution metadata builder for {metadata_type.__name__}")
                execution_contexts[metadata_type] = builder.build(input_batch, metadata)

        # Admission and context construction must finish before prepare(). A
        # sharded MLA backend enters CP collectives during prepare, so rejecting
        # unsupported batches afterwards could leave peer ranks deadlocked.
        self.attention_backend.prepare(metadata)

        with forward_context(
            ForwardContext(
                self.attention_backend,
                self.device,
                metadata,
                self.layer_caches,
                layer_synchronizer=layer_synchronizer,
                layer_load_context=layer_load_context,
                cp_context=cp_context,
                eplb=eplb,
                execution_contexts=execution_contexts,
            )
        ):
            # Draft-MTP steps carry the target's (or previous draft step's)
            # hidden state as input_embedding; the MTP body fuses it with the
            # token embedding. Regular steps take the 2-arg path.
            if input_embedding is None:
                return self.model(input_ids, positions)
            return self.model(input_ids, positions, input_embedding)
