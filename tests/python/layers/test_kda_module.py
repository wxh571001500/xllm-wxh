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

"""Unit tests for KDA (Key-value Decode Attention) module."""

import pytest
import torch

from xllm.python.layers.kda import (
    KDA_CHUNK_SIZE,
    PAD_SLOT_ID,
    KimiK3KDAMetadata,
)


class TestKDAMetadata:
    """Test KDA metadata structure."""

    def test_kda_metadata_creation(self):
        """Test creating KDA metadata."""
        num_decode_seqs = 4
        num_prefill_seqs = 2
        num_seqs = num_decode_seqs + num_prefill_seqs

        # Create metadata
        query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
        state_indices = torch.arange(num_seqs, dtype=torch.int64)

        metadata = KimiK3KDAMetadata(
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            num_decode_seqs=num_decode_seqs,
            num_prefill_seqs=num_prefill_seqs,
        )

        assert metadata.num_decode_seqs == num_decode_seqs
        assert metadata.num_prefill_seqs == num_prefill_seqs
        assert metadata.query_start_loc.shape == (num_seqs + 1,)
        assert metadata.state_indices.shape == (num_seqs,)

    def test_kda_metadata_with_initial_state(self):
        """Test KDA metadata with initial state flags."""
        num_seqs = 6

        query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
        state_indices = torch.arange(num_seqs, dtype=torch.int64)
        has_initial_state = torch.tensor([True, True, True, True, False, True], dtype=torch.bool)

        metadata = KimiK3KDAMetadata(
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            num_decode_seqs=4,
            num_prefill_seqs=2,
            has_initial_state=has_initial_state,
        )

        assert metadata.has_initial_state is not None
        assert metadata.has_initial_state.shape == (num_seqs,)
        assert metadata.has_initial_state.dtype == torch.bool

    def test_kda_metadata_graph_mode(self):
        """Test KDA metadata for graph mode with padding."""
        num_seqs = 3
        graph_num_tokens = 128  # Padded token count for graph

        query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
        state_indices = torch.arange(num_seqs, dtype=torch.int64)

        metadata = KimiK3KDAMetadata(
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            num_decode_seqs=3,
            num_prefill_seqs=0,
            graph_num_tokens=graph_num_tokens,
        )

        assert metadata.graph_num_tokens == graph_num_tokens


class TestKDAConstants:
    """Test KDA module constants."""

    def test_kda_chunk_size(self):
        """Test KDA chunk size constant."""
        assert KDA_CHUNK_SIZE == 64
        assert isinstance(KDA_CHUNK_SIZE, int)

    def test_pad_slot_id(self):
        """Test padding slot ID constant."""
        assert PAD_SLOT_ID == -1
        assert isinstance(PAD_SLOT_ID, int)


class TestKDAStateManagement:
    """Test KDA state cache management."""

    def test_conv_state_shape(self):
        """Test conv state cache shape requirements."""
        num_slots = 128
        conv_size = 4
        local_proj = 256

        # Conv state: [num_slots, conv_size - 1, 3 * local_proj]
        conv_state = torch.zeros(num_slots, conv_size - 1, 3 * local_proj, dtype=torch.bfloat16)

        assert conv_state.shape == (num_slots, conv_size - 1, 3 * local_proj)
        assert conv_state.dtype == torch.bfloat16

    def test_recurrent_state_shape(self):
        """Test recurrent state cache shape requirements."""
        num_slots = 128
        local_num_heads = 8
        head_dim = 64

        # Recurrent state: [num_slots, local_num_heads, head_dim, head_dim]
        # Layout: [H, V, K] per slot
        recurrent_state = torch.zeros(num_slots, local_num_heads, head_dim, head_dim, dtype=torch.float32)

        assert recurrent_state.shape == (num_slots, local_num_heads, head_dim, head_dim)
        assert recurrent_state.dtype == torch.float32

    def test_state_indices_with_padding(self):
        """Test state indices handling with padding slots."""
        num_seqs = 5
        num_slots = 10

        # Some sequences use valid slots, others use padding
        state_indices = torch.tensor([0, 5, PAD_SLOT_ID, 3, 7], dtype=torch.int64)

        assert state_indices.shape == (num_seqs,)
        assert (state_indices[state_indices != PAD_SLOT_ID] >= 0).all()
        assert (state_indices[state_indices != PAD_SLOT_ID] < num_slots).all()


class TestKDABatchConfiguration:
    """Test KDA batch configuration scenarios."""

    def test_pure_decode_batch(self):
        """Test pure decode batch configuration."""
        num_decode_seqs = 8
        num_prefill_seqs = 0

        query_start_loc = torch.arange(num_decode_seqs + 1, dtype=torch.int32)
        state_indices = torch.arange(num_decode_seqs, dtype=torch.int64)

        metadata = KimiK3KDAMetadata(
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            num_decode_seqs=num_decode_seqs,
            num_prefill_seqs=num_prefill_seqs,
        )

        # In pure decode, each sequence has exactly 1 token
        num_tokens = query_start_loc[-1].item()
        assert num_tokens == num_decode_seqs
        assert metadata.has_initial_state is None  # Not required for pure decode

    def test_mixed_batch(self):
        """Test mixed decode + prefill batch configuration."""
        num_decode_seqs = 4
        num_prefill_seqs = 2
        num_seqs = num_decode_seqs + num_prefill_seqs

        # Decode sequences: 1 token each
        # Prefill sequences: variable tokens
        token_counts = [1, 1, 1, 1, 128, 64]
        query_start_loc = torch.tensor([0] + torch.cumsum(torch.tensor(token_counts), 0).tolist(), dtype=torch.int32)
        state_indices = torch.arange(num_seqs, dtype=torch.int64)
        has_initial_state = torch.tensor([True, True, True, True, False, True], dtype=torch.bool)

        metadata = KimiK3KDAMetadata(
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            num_decode_seqs=num_decode_seqs,
            num_prefill_seqs=num_prefill_seqs,
            has_initial_state=has_initial_state,
        )

        total_tokens = query_start_loc[-1].item()
        assert total_tokens == sum(token_counts)
        assert metadata.has_initial_state is not None

    def test_chunked_prefill_batch(self):
        """Test chunked prefill batch configuration."""
        num_decode_seqs = 0
        num_prefill_seqs = 3
        num_seqs = num_prefill_seqs

        # Each prefill chunk is at most KDA_CHUNK_SIZE tokens
        token_counts = [KDA_CHUNK_SIZE, KDA_CHUNK_SIZE, 32]
        query_start_loc = torch.tensor([0] + torch.cumsum(torch.tensor(token_counts), 0).tolist(), dtype=torch.int32)
        state_indices = torch.arange(num_seqs, dtype=torch.int64)
        # First chunk: no initial state; continuation chunks: have initial state
        has_initial_state = torch.tensor([False, True, False], dtype=torch.bool)

        metadata = KimiK3KDAMetadata(
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            num_decode_seqs=num_decode_seqs,
            num_prefill_seqs=num_prefill_seqs,
            has_initial_state=has_initial_state,
        )

        assert metadata.num_decode_seqs == 0
        assert all(tc <= KDA_CHUNK_SIZE for tc in token_counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
