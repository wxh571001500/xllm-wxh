/* Copyright 2025-2026 The xLLM Authors.

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

#include "api_service/speech_to_text_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <butil/base64.h>
#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "core/common/instance_name.h"
#include "core/common/message.h"
#include "core/common/types.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/config/model_config.h"
#include "core/framework/multimodal/mm_codec.h"
#include "core/framework/multimodal/mm_type.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/util/scope_guard.h"
#include "core/util/uuid.h"
#include "models/model_registry.h"
#include "models/speech_model.h"

namespace xllm {

namespace {

constexpr std::string_view kDefaultMimeType = "audio/wav";
constexpr std::string_view kJsonFormat = "json";
constexpr std::string_view kTextFormat = "text";
constexpr std::string_view kVerboseJsonFormat = "verbose_json";

const char* speech_to_text_chunk_object(SpeechToTextTask task) {
  return task == SpeechToTextTask::TRANSLATE ? "translation.chunk"
                                             : "transcription.chunk";
}

const char* speech_to_text_id_prefix(SpeechToTextTask task) {
  return task == SpeechToTextTask::TRANSLATE ? "trsl-" : "trsc-";
}

// Guesses the upload mime type from the part Content-Type or filename
// extension (case-insensitive); "audio/wav" when unknown.
std::string guess_mime_type(const proto::SpeechToTextRequest& request) {
  // Only trust explicit audio/* types; curl often sends octet-stream.
  if (request.file_content_type().rfind("audio/", 0) == 0) {
    return request.file_content_type();
  }
  const std::string& filename = request.filename();
  const size_t dot = filename.rfind('.');
  if (dot == std::string::npos) {
    return std::string(kDefaultMimeType);
  }
  std::string extension = filename.substr(dot + 1);
  std::transform(
      extension.begin(),
      extension.end(),
      extension.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (extension == "mp3" || extension == "mpga" || extension == "mpeg") {
    return "audio/mpeg";
  }
  if (extension == "m4a" || extension == "mp4") {
    return "audio/mp4";
  }
  if (extension == "ogg" || extension == "opus") {
    return "audio/ogg";
  }
  if (extension == "webm") {
    return "audio/webm";
  }
  if (extension == "flac") {
    return "audio/flac";
  }
  if (extension == "aac") {
    return "audio/aac";
  }
  // wav and anything unknown.
  return std::string(kDefaultMimeType);
}

bool send_result_to_client_brpc(const std::shared_ptr<SpeechToTextCall>& call,
                                SpeechToTextTask task,
                                const std::string& response_format,
                                const std::string& language,
                                double duration_s,
                                const RequestOutput& req_output) {
  auto& response = call->response();
  response.set_text(
      req_output.outputs.empty() ? "" : req_output.outputs.front().text);
  if (response_format == kVerboseJsonFormat) {
    response.set_duration(duration_s);
    response.set_language(language);
    // Segment timestamps are not available for the current adapter.
  } else if (response_format == kJsonFormat &&
             task == SpeechToTextTask::TRANSCRIBE) {
    auto* usage = response.mutable_usage();
    usage->set_type("duration");
    usage->set_seconds(static_cast<int32_t>(std::ceil(duration_s)));
  }

  if (response_format == kTextFormat) {
    return call->write_text_and_finish(response.text());
  }
  return call->write_and_finish(response);
}

void fill_stream_chunk(proto::SpeechToTextStreamChunk& chunk,
                       std::string_view id,
                       std::string_view object,
                       uint32_t created,
                       std::string_view model,
                       std::string_view delta_content,
                       const std::string* finish_reason) {
  chunk.set_id(std::string(id));
  chunk.set_object(std::string(object));
  chunk.set_created(created);
  chunk.set_model(std::string(model));
  auto* choice = chunk.add_choices();
  choice->set_index(0);
  choice->mutable_delta();
  if (!delta_content.empty()) {
    choice->mutable_delta()->set_content(std::string(delta_content));
  }
  if (finish_reason != nullptr) {
    choice->set_finish_reason(*finish_reason);
  }
}

std::shared_ptr<const SpeechModelInterface> resolve_speech_adapter(
    const std::string& model_type) {
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration_name(
          model_type, &resolved_name, &error_message)) {
    return nullptr;
  }
  const SpeechModelFactory factory = get_speech_model_factory(resolved_name);
  return factory != nullptr ? factory() : nullptr;
}

void fill_stream_usage_chunk(proto::SpeechToTextStreamChunk& chunk,
                             std::string_view id,
                             std::string_view object,
                             uint32_t created,
                             std::string_view model,
                             const proto::Usage* usage) {
  chunk.set_id(std::string(id));
  chunk.set_object(std::string(object));
  chunk.set_created(created);
  chunk.set_model(std::string(model));
  if (usage != nullptr) {
    *chunk.mutable_usage() = *usage;
  }
}

std::string generate_speech_to_text_request_id(SpeechToTextTask task) {
  thread_local ShortUUID short_uuid;
  return std::string(speech_to_text_id_prefix(task)) +
         InstanceName::name()->get_name_hash() + "-" + short_uuid.random();
}

bool send_delta_to_client_brpc(const std::shared_ptr<SpeechToTextCall>& call,
                               SpeechToTextTask task,
                               bool include_usage,
                               bool continuous_usage,
                               const std::string& request_id,
                               uint32_t created_time,
                               const std::string& model,
                               const RequestOutput& req_output,
                               std::optional<proto::Usage>& stream_usage) {
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto& proto_usage = stream_usage.emplace();
    proto_usage.set_prompt_tokens(usage.num_prompt_tokens);
    proto_usage.set_completion_tokens(usage.num_generated_tokens);
    proto_usage.set_total_tokens(usage.num_total_tokens);
  }

  for (const auto& seq_output : req_output.outputs) {
    if (!seq_output.text.empty()) {
      proto::SpeechToTextStreamChunk chunk;
      fill_stream_chunk(chunk,
                        request_id,
                        speech_to_text_chunk_object(task),
                        created_time,
                        model,
                        seq_output.text,
                        /*finish_reason=*/nullptr);
      if (continuous_usage && stream_usage.has_value()) {
        *chunk.mutable_usage() = stream_usage.value();
      }
      if (!call->write(chunk)) {
        return false;
      }
    }

    if (seq_output.finish_reason.has_value()) {
      proto::SpeechToTextStreamChunk chunk;
      fill_stream_chunk(chunk,
                        request_id,
                        speech_to_text_chunk_object(task),
                        created_time,
                        model,
                        /*delta_content=*/"",
                        &seq_output.finish_reason.value());
      if (continuous_usage && stream_usage.has_value()) {
        *chunk.mutable_usage() = stream_usage.value();
      }
      if (!call->write(chunk)) {
        return false;
      }
    }
  }

  if (req_output.finished || req_output.cancelled) {
    if (include_usage) {
      proto::Usage total_usage;
      if (stream_usage.has_value()) {
        total_usage = stream_usage.value();
      }
      proto::SpeechToTextStreamChunk chunk;
      fill_stream_usage_chunk(chunk,
                              request_id,
                              speech_to_text_chunk_object(task),
                              created_time,
                              model,
                              &total_usage);
      if (!call->write(chunk)) {
        return false;
      }
    }
    return call->finish();
  }
  return true;
}

// Multipart form field parsing.
std::string field_error(std::string_view name, std::string_view reason) {
  return "Field `" + std::string(name) + "` " + std::string(reason);
}

// Parses one form field value; the overload matches the setter's type.
bool parse_form_value(std::string_view name,
                      std::string_view value,
                      bool* out,
                      std::string* error) {
  if (value == "true" || value == "1") {
    *out = true;
    return true;
  }
  if (value == "false" || value == "0") {
    *out = false;
    return true;
  }
  *error = field_error(name, "must be a boolean.");
  return false;
}

bool parse_form_value(std::string_view name,
                      std::string_view value,
                      float* out,
                      std::string* error) {
  float parsed = 0.0f;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc() || result.ptr != value.data() + value.size() ||
      !std::isfinite(parsed)) {
    *error = field_error(name, "must be a finite number.");
    return false;
  }
  *out = parsed;
  return true;
}

bool parse_form_value(std::string_view name,
                      std::string_view value,
                      int64_t* out,
                      std::string* error) {
  int64_t parsed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc() || result.ptr != value.data() + value.size()) {
    *error = field_error(name, "must be an integer.");
    return false;
  }
  *out = parsed;
  return true;
}

bool parse_form_value(std::string_view name,
                      std::string_view value,
                      uint32_t* out,
                      std::string* error) {
  int64_t parsed = 0;
  if (!parse_form_value(name, value, &parsed, error)) {
    return false;
  }
  if (parsed < 0 || parsed > static_cast<int64_t>(UINT32_MAX)) {
    *error = field_error(name, "is out of range.");
    return false;
  }
  *out = static_cast<uint32_t>(parsed);
  return true;
}

// Maps one form field via `setter`.
template <typename T>
bool set_form_field(const api_service::MultipartFormData& form,
                    std::string_view field,
                    void (proto::SpeechToTextRequest::*setter)(T),
                    proto::SpeechToTextRequest& request,
                    std::string* error) {
  const auto value = api_service::find_multipart_field_value(form, field);
  if (!value.has_value()) {
    return true;
  }
  T parsed{};
  if (!parse_form_value(field, value.value(), &parsed, error)) {
    return false;
  }
  (request.*setter)(parsed);
  return true;
}

Status load_speech_audio(const proto::SpeechToTextRequest& request,
                         butil::IOBuf payload,
                         uint64_t max_bytes,
                         std::string& audio_bytes) {
  if (request.file().type() == "binary") {
    const auto& binary = request.file().binary();
    const uint64_t offset = binary.offset();
    const uint64_t length = binary.length();
    if (offset > payload.size() || length > payload.size() - offset) {
      return {StatusCode::INVALID_ARGUMENT,
              "Audio file binary reference is out of range."};
    }
    if (length > max_bytes) {
      return {StatusCode::INVALID_ARGUMENT,
              "Maximum audio file size exceeded."};
    }
    payload.pop_front(static_cast<size_t>(offset));
    payload.copy_to(&audio_bytes, static_cast<size_t>(length));
  } else {
    // Bound the allocation before decoding the inline payload.
    const uint64_t max_base64_bytes = ((max_bytes + 2) / 3) * 4;
    if (request.file().base64().size() > max_base64_bytes) {
      return {StatusCode::INVALID_ARGUMENT,
              "Maximum audio file size exceeded."};
    }
    if (!butil::Base64Decode(butil::StringPiece(request.file().base64()),
                             &audio_bytes)) {
      return {StatusCode::INVALID_ARGUMENT,
              "Failed to decode base64 audio file."};
    }
  }
  if (audio_bytes.empty()) {
    return {StatusCode::INVALID_ARGUMENT, "Audio file must not be empty."};
  }
  if (audio_bytes.size() > max_bytes) {
    return {StatusCode::INVALID_ARGUMENT, "Maximum audio file size exceeded."};
  }
  return {};
}

}  // namespace

bool fill_request_from_multipart_form(
    const api_service::MultipartFormData& form,
    const api_service::MultipartPart& file,
    proto::SpeechToTextRequest& request,
    std::string* error) {
  auto set_string_field =
      [&form, &request](
          std::string_view field,
          void (proto::SpeechToTextRequest::*setter)(const std::string&)) {
        const auto value = api_service::find_multipart_field_value(form, field);
        if (value.has_value()) {
          (request.*setter)(value.value());
        }
      };

  set_string_field("model", &proto::SpeechToTextRequest::set_model);
  set_string_field("language", &proto::SpeechToTextRequest::set_language);
  set_string_field("prompt", &proto::SpeechToTextRequest::set_prompt);
  set_string_field("response_format",
                   &proto::SpeechToTextRequest::set_response_format);
  set_string_field("to_language", &proto::SpeechToTextRequest::set_to_language);
  set_string_field("hotwords", &proto::SpeechToTextRequest::set_hotwords);
  set_string_field("user", &proto::SpeechToTextRequest::set_user);

  // Repeated field: every plain part contributes one value.
  for (const auto& part : form.parts) {
    if (!part.filename.empty() || part.name != "timestamp_granularities[]") {
      continue;
    }
    request.add_timestamp_granularities(part.value.to_string());
  }

  // Typed fields in parse order; the first invalid field wins the error.
  if (!set_form_field(form,
                      "stream",
                      &proto::SpeechToTextRequest::set_stream,
                      request,
                      error) ||
      !set_form_field(form,
                      "stream_include_usage",
                      &proto::SpeechToTextRequest::set_stream_include_usage,
                      request,
                      error) ||
      !set_form_field(
          form,
          "stream_continuous_usage_stats",
          &proto::SpeechToTextRequest::set_stream_continuous_usage_stats,
          request,
          error) ||
      !set_form_field(form,
                      "temperature",
                      &proto::SpeechToTextRequest::set_temperature,
                      request,
                      error) ||
      !set_form_field(form,
                      "top_p",
                      &proto::SpeechToTextRequest::set_top_p,
                      request,
                      error) ||
      // Reserved for upcoming sampling support; parsed but not consumed.
      !set_form_field(form,
                      "min_p",
                      &proto::SpeechToTextRequest::set_min_p,
                      request,
                      error) ||
      !set_form_field(form,
                      "seed",
                      &proto::SpeechToTextRequest::set_seed,
                      request,
                      error) ||
      !set_form_field(form,
                      "frequency_penalty",
                      &proto::SpeechToTextRequest::set_frequency_penalty,
                      request,
                      error) ||
      !set_form_field(form,
                      "presence_penalty",
                      &proto::SpeechToTextRequest::set_presence_penalty,
                      request,
                      error) ||
      !set_form_field(form,
                      "repetition_penalty",
                      &proto::SpeechToTextRequest::set_repetition_penalty,
                      request,
                      error) ||
      !set_form_field(form,
                      "top_k",
                      &proto::SpeechToTextRequest::set_top_k,
                      request,
                      error) ||
      !set_form_field(form,
                      "max_completion_tokens",
                      &proto::SpeechToTextRequest::set_max_completion_tokens,
                      request,
                      error)) {
    return false;
  }

  if (!file.filename.empty()) {
    request.set_filename(file.filename);
  }
  if (!file.content_type.empty()) {
    request.set_file_content_type(file.content_type);
  }
  return true;
}

// SpeechToTextServiceImpl
SpeechToTextServiceImpl::SpeechToTextServiceImpl(
    VLMMaster* master,
    const std::vector<std::string>& models,
    SpeechToTextTask task)
    : APIServiceImpl(models), master_{master}, task_{task} {
  CHECK(master_ != nullptr);
  adapter_ = resolve_speech_adapter(master_->model_type());
}

void SpeechToTextServiceImpl::process_async_impl(
    std::shared_ptr<SpeechToTextCall> call) {
  const auto& rpc_request = call->request();

  std::string model = rpc_request.model();
  if (model.empty()) {
    model = *models_.begin();
  }
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  const std::string response_format = rpc_request.response_format().empty()
                                          ? std::string(kJsonFormat)
                                          : rpc_request.response_format();
  if (response_format != kJsonFormat && response_format != kTextFormat &&
      response_format != kVerboseJsonFormat) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "Currently only support response_format: `text`, `json` or "
        "`verbose_json`.");
    return;
  }

  if (adapter_ == nullptr) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "The current model does not support speech-to-text.");
    return;
  }

  const bool verbose = response_format == kVerboseJsonFormat;
  if (verbose && !adapter_->supports_verbose_json()) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "Currently do not support verbose_json for " + model);
    return;
  }

  if (rpc_request.timestamp_granularities_size() > 0 &&
      !adapter_->supports_segment_timestamps()) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "Timestamp granularities are not supported by the current model.");
    return;
  }
  const bool stream = rpc_request.stream();
  if (verbose && stream) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "`verbose_json` format doesn't support streaming case.");
    return;
  }
  if (!stream && (rpc_request.stream_include_usage() ||
                  rpc_request.stream_continuous_usage_stats())) {
    call->finish_with_error(
        StatusCode::INVALID_ARGUMENT,
        "Stream options can only be defined when `stream=true`.");
    return;
  }
  if (task_ == SpeechToTextTask::TRANSLATE &&
      !adapter_->supports_translation()) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "Task 'translate' is not supported by the "
                            "current model.");
    return;
  }
  const Status language_status =
      adapter_->validate_language(rpc_request.language());
  if (!language_status.ok()) {
    call->finish_with_error(language_status.code(), language_status.message());
    return;
  }
  if (!rpc_request.has_file() || (rpc_request.file().type() != "binary" &&
                                  rpc_request.file().type() != "base64")) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "Expected `file` to be a file-like object.");
    return;
  }

  if (master_->get_rate_limiter()->is_limited()) {
    if (master_->get_rate_limiter()->is_sleeping()) {
      call->finish_with_error(StatusCode::UNAVAILABLE,
                              "Model is currently in sleep state.");
    } else {
      call->finish_with_error(
          StatusCode::RESOURCE_EXHAUSTED,
          "The number of concurrent requests has reached the limit.");
    }
    return;
  }
  ScopeGuard admission_guard(
      [this] { master_->get_rate_limiter()->decrease_one_request(); });

  std::string audio_bytes;
  const uint64_t max_bytes =
      static_cast<uint64_t>(
          ModelConfig::get_instance().audio_max_upload_file_mb()) *
      1024 * 1024;
  const Status audio_status = load_speech_audio(
      rpc_request, call->take_request_iobuf(), max_bytes, audio_bytes);
  if (!audio_status.ok()) {
    call->finish_with_error(audio_status.code(), audio_status.message());
    return;
  }

  // Keep duration measurement local to the speech endpoint. The VLM path
  // receives the original bytes and performs its usual decoding independently.
  double duration_s = 0.0;
  {
    torch::Tensor audio_tensor;
    AudioMetadata metadata;
    FFmpegAudioDecoder decoder;
    if (!decoder.decode(audio_bytes, audio_tensor, metadata)) {
      call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                              "Failed to decode audio file.");
      return;
    }
    duration_s = metadata.duration;
  }
  const int32_t max_duration_s =
      ModelConfig::get_instance().audio_max_decode_duration_s();
  if (max_duration_s > 0 && duration_s > static_cast<double>(max_duration_s)) {
    call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                            "Maximum audio duration exceeded.");
    return;
  }

  SpeechToTextParams speech_params;
  speech_params.task = task_;
  speech_params.language = rpc_request.language();
  speech_params.request_prompt = rpc_request.prompt();
  speech_params.hotwords = rpc_request.hotwords();
  speech_params.mime_type = guess_mime_type(rpc_request);
  speech_params.audio_bytes = audio_bytes.size();
  std::vector<Message> messages = adapter_->build_messages(speech_params);
  RequestParams request_params;
  request_params.request_id = rpc_request.request_id().empty()
                                  ? generate_speech_to_text_request_id(task_)
                                  : rpc_request.request_id();
  request_params.x_request_id = call->get_x_request_id();
  request_params.x_request_time = call->get_x_request_time();
  request_params.streaming = stream;
  if (rpc_request.has_max_completion_tokens() &&
      rpc_request.max_completion_tokens() > 0) {
    request_params.max_tokens = rpc_request.max_completion_tokens();
  }
  if (rpc_request.has_temperature()) {
    request_params.temperature = rpc_request.temperature();
  }
  if (rpc_request.has_top_p()) {
    request_params.top_p = rpc_request.top_p();
  }
  if (rpc_request.has_top_k()) {
    request_params.top_k = rpc_request.top_k() == 0 ? -1 : rpc_request.top_k();
  }
  if (rpc_request.has_frequency_penalty()) {
    request_params.frequency_penalty = rpc_request.frequency_penalty();
  }
  if (rpc_request.has_presence_penalty()) {
    request_params.presence_penalty = rpc_request.presence_penalty();
  }
  if (rpc_request.has_repetition_penalty()) {
    request_params.repetition_penalty = rpc_request.repetition_penalty();
  }

  std::string saved_request_id = request_params.request_id;

  // Transfer the admission slot to the master and submit once for both modes.
  admission_guard.dismiss();
  master_->handle_request(
      std::move(messages),
      std::move(request_params),
      std::move(audio_bytes),
      [call,
       model = std::move(model),
       task = task_,
       stream,
       include_usage = rpc_request.stream_include_usage(),
       continuous_usage = rpc_request.stream_continuous_usage_stats(),
       request_id = std::move(saved_request_id),
       created_time = static_cast<uint32_t>(absl::ToUnixSeconds(absl::Now())),
       response_format,
       language = rpc_request.language(),
       duration_s,
       stream_usage = std::optional<proto::Usage>()](
          const RequestOutput& req_output) mutable -> bool {
        req_output.log_request_status();
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }
        if (req_output.cancelled) {
          return call->finish_with_error(StatusCode::CANCELLED,
                                         "The request was cancelled.");
        }
        if (stream) {
          return send_delta_to_client_brpc(call,
                                           task,
                                           include_usage,
                                           continuous_usage,
                                           request_id,
                                           created_time,
                                           model,
                                           req_output,
                                           stream_usage);
        }
        return send_result_to_client_brpc(
            call, task, response_format, language, duration_s, req_output);
      });
}

}  // namespace xllm
