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

#pragma once

#include "core/common/global_flags.h"
#include "models/vlm/mposition/mposition.h"
#include "processors/multimodal_processor.h"
#include "processors/qwen3_omni_audio_processor.h"
#include "processors/qwen3_omni_image_processor.h"
#include "processors/qwen3_omni_prompt_processor.h"
#include "processors/qwen3_omni_video_processor.h"
#include "qwen3_omni_moe_thinker.h"

namespace xllm::npu::model {

class Qwen3OmniMoe_ForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen3OmniMoe_ForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    thinker_ = register_module(
        "thinker", Qwen3OmniMoe_Thinker_ForConditionalGeneration(context));
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    return thinker_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return thinker_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    thinker_->load_model(std::move(loader));
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    return thinker_->get_input_embeddings(input_ids, input_params);
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    return thinker_->get_multimodal_embeddings(input_params);
  }
  layer::NpuLmHead get_npu_lm_head() { return thinker_->get_npu_lm_head(); }

  void set_npu_lm_head(layer::NpuLmHead& head) {
    thinker_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return thinker_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    thinker_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  Qwen3OmniMoe_Thinker_ForConditionalGeneration thinker_{nullptr};
};
TORCH_MODULE(Qwen3OmniMoe_ForConditionalGeneration);

using Qwen3OmniMoeMultimodalProcessor =
    MultimodalProcessor<Qwen3OmniPromptProcessor,
                        Qwen3OmniImageProcessor,
                        Qwen3OmniVideoProcessor,
                        Qwen3OmniAudioProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(qwen3_omni_moe, Qwen3OmniMoeMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_omni_moe,
                          Qwen3OmniMoe_ForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_omni_moe, xllm::Qwen3OmniMPositionGenerator);

REGISTER_MODEL_ARGS(qwen3_omni_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_omni_moe");

  // feature extractor default config
  LOAD_ARG_OR(mm_audio_truncation, "truncation", false);
  LOAD_ARG_OR(mm_audio_padding_strategy, "padding_strategy", 1);
  LOAD_ARG_OR(mm_audio_max_length, "max_length", -1);
  LOAD_ARG_OR(mm_audio_pad_to_multiple_of, "pad_to_multiple_of", -1);
  LOAD_ARG_OR(mm_audio_do_normalize, "do_normalize", false);
  LOAD_ARG_OR(
      mm_audio_return_token_timestamps, "return_token_timestamps", false);
  LOAD_ARG_OR(mm_audio_return_attention_mask, "return_attention_mask", true);
  LOAD_ARG_OR(
      mm_use_audio_in_video, "use_audio_in_video", FLAGS_use_audio_in_video);
  LOAD_ARG_OR(mm_position_id_per_seconds, "position_id_per_seconds", 13);
  LOAD_ARG_OR(mm_fps, "fps", 1.0);

  // thinker config
  LOAD_ARG_OR(
      vision_start_token_id, "thinker_config.vision_start_token_id", 151652);
  LOAD_ARG_OR(
      vision_end_token_id, "thinker_config.vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_token_id, "thinker_config.vision_token_id", 151654);
  LOAD_ARG_OR(image_token_id, "thinker_config.image_token_id", 151655);
  LOAD_ARG_OR(video_token_id, "thinker_config.video_token_id", 151656);
  LOAD_ARG_OR(audio_token_id, "thinker_config.audio_token_id", 151675);
  LOAD_ARG_OR(
      audio_start_token_id, "thinker_config.audio_start_token_id", 151669);
  LOAD_ARG_OR(audio_end_token_id, "thinker_config.audio_end_token_id", 151670);
  LOAD_ARG_OR(dtype, "thinker_config.dtype", "bfloat16");

  // thinker.text_config
  LOAD_ARG_OR(
      attention_bias, "thinker_config.text_config.attention_bias", false);
  LOAD_ARG_OR(
      attention_dropout, "thinker_config.text_config.attention_dropout", 0.0);
  LOAD_ARG_OR(
      decoder_sparse_step, "thinker_config.text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(bos_token_id, "thinker_config.text_config.bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "thinker_config.text_config.eos_token_id", 151645);
  LOAD_ARG_OR(hidden_act, "thinker_config.text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "thinker_config.text_config.hidden_size", 2048);
  LOAD_ARG_OR(
      intermediate_size, "thinker_config.text_config.intermediate_size", 768);
  LOAD_ARG_OR(max_position_embeddings,
              "thinker_config.text_config.max_position_embeddings",
              65536);
  LOAD_ARG_OR(
      max_window_layers, "thinker_config.text_config.max_window_layers", 28);
  LOAD_ARG_OR(n_heads, "thinker_config.text_config.num_attention_heads", 32);
  LOAD_ARG_OR(n_layers, "thinker_config.text_config.num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "thinker_config.text_config.num_key_value_heads", 4);
  LOAD_ARG_OR(rms_norm_eps, "thinker_config.text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(
      sliding_window, "thinker_config.text_config.sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings,
              "thinker_config.text_config.tie_word_embeddings",
              false);
  LOAD_ARG(rope_scaling_mrope_section,
           "thinker_config.text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(
      initializer_range, "thinker_config.text_config.initializer_range", 0.02);
  LOAD_ARG_OR(use_sliding_window,
              "thinker_config.text_config.use_sliding_window",
              false);
  LOAD_ARG_OR(moe_intermediate_size,
              "thinker_config.text_config.moe_intermediate_size",
              768);
  LOAD_ARG_OR(
      norm_topk_prob, "thinker_config.text_config.norm_topk_prob", true);
  LOAD_ARG_OR(num_experts, "thinker_config.text_config.num_experts", 128);
  LOAD_ARG_OR(
      num_experts_per_tok, "thinker_config.text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR_FUNC(head_dim, "thinker_config.text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(output_router_logits,
              "thinker_config.text_config.output_router_logits",
              false);
  LOAD_ARG_OR(router_aux_loss_coef,
              "thinker_config.text_config.router_aux_loss_coef",
              0.001f);
  LOAD_ARG_OR(mlp_only_layers,
              "thinker_config.text_config.mlp_only_layers",
              std::vector<int>());
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
  LOAD_ARG_OR(rope_scaling_rope_type,
              "thinker_config.text_config.rope_scaling.type",
              "mrope");
  LOAD_ARG_OR(rope_theta, "thinker_config.text_config.rope_theta", 1000000.0f);
  LOAD_ARG_OR(vocab_size, "thinker_config.text_config.vocab_size", 152064);

  if (args->rope_scaling_rope_type() == "default") {
    args->rope_scaling_rope_type() = "mrope";
  }

  // thinker.vision_config
  LOAD_ARG_OR(mm_num_hidden_layers, "thinker_config.vision_config.depth", 27);
  LOAD_ARG_OR(mm_hidden_act,
              "thinker_config.vision_config.hidden_act",
              "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_hidden_size, "thinker_config.vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_intermediate_size,
              "thinker_config.vision_config.intermediate_size",
              4304);
  LOAD_ARG_OR(
      mm_num_attention_heads, "thinker_config.vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "thinker_config.vision_config.in_channels", 3);
  LOAD_ARG_OR(
      mm_projection_dim, "thinker_config.vision_config.out_hidden_size", 2048);
  LOAD_ARG_OR(mm_patch_size, "thinker_config.vision_config.patch_size", 16);
  LOAD_ARG_OR(mm_num_position_embeddings,
              "thinker_config.vision_config.num_position_embeddings",
              2304);
  LOAD_ARG_OR(mm_spatial_merge_size,
              "thinker_config.vision_config.spatial_merge_size",
              2);
  LOAD_ARG(mm_deepstack_visual_indexes,
           "thinker_config.vision_config.deepstack_visual_indexes");
  LOAD_ARG_OR(mm_temporal_patch_size,
              "thinker_config.vision_config.temporal_patch_size",
              2);
  LOAD_ARG_OR(mm_image_size, "thinker_config.vision_config.image_size", 768);
  LOAD_ARG_OR_FUNC(mm_head_dim, "thinker_config.vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  // thinker.audio_config
  LOAD_ARG_OR(mm_audio_num_attention_heads,
              "thinker_config.audio_config.encoder_attention_heads",
              20);
  LOAD_ARG_OR(
      mm_audio_hidden_size, "thinker_config.audio_config.d_model", 1280);
  LOAD_ARG_OR(mm_audio_downsample_hidden_size,
              "thinker_config.audio_config.downsample_hidden_size",
              480);
  LOAD_ARG_OR(
      mm_audio_num_mel_bins, "thinker_config.audio_config.num_mel_bins", 128);
  LOAD_ARG_OR(mm_audio_max_source_positions,
              "thinker_config.audio_config.max_source_positions",
              1500);
  LOAD_ARG_OR(mm_audio_scale_embedding,
              "thinker_config.audio_config.scale_embedding",
              false);
  LOAD_ARG_OR(mm_audio_n_window, "thinker_config.audio_config.n_window", 50);
  LOAD_ARG_OR(mm_audio_n_window_infer,
              "thinker_config.audio_config.n_window_infer",
              800);
  LOAD_ARG_OR(mm_audio_conv_chunksize,
              "thinker_config.audio_config.conv_chunksize",
              500);
  LOAD_ARG_OR(mm_audio_encoder_layers,
              "thinker_config.audio_config.encoder_layers",
              32);
  LOAD_ARG_OR(
      mm_audio_output_dim, "thinker_config.audio_config.output_dim", 2048);
});

}  // namespace xllm::npu::model
