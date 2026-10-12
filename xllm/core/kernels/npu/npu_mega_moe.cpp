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

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "graph/types.h"

namespace xllm::kernel::npu {
namespace {

void compact_strides(const int64_t* dims, uint64_t ndim, int64_t* strides) {
  int64_t running = 1;
  for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
    strides[i] = running;
    running *= dims[i];
  }
}

bool is_fractal_nz(const torch::Tensor& tensor) {
  return at_npu::native::custom_ops::get_npu_format(tensor) ==
         ACL_FORMAT_FRACTAL_NZ;
}

aclTensor* create_acl_tensor(const int64_t* view_dims,
                             uint64_t view_ndim,
                             aclDataType dtype,
                             const int64_t* strides,
                             aclFormat format,
                             const int64_t* storage_dims,
                             uint64_t storage_ndim,
                             void* data_ptr) {
  static const auto acl_create_tensor =
      aclnn::detail::get_op_api_func<aclnn::detail::AclCreateTensorFn>(
          "aclCreateTensor");
  CHECK(acl_create_tensor != nullptr)
      << "aclCreateTensor is not available in libopapi.";
  aclTensor* tensor = acl_create_tensor(view_dims,
                                        view_ndim,
                                        dtype,
                                        strides,
                                        /*offset=*/0,
                                        format,
                                        storage_dims,
                                        storage_ndim,
                                        data_ptr);
  CHECK(tensor != nullptr) << "aclCreateTensor returned nullptr.";
  return tensor;
}

aclTensor* create_acl_tensor_nd(const int64_t* dims,
                                uint64_t ndim,
                                aclDataType dtype,
                                const int64_t* strides,
                                void* data_ptr) {
  return create_acl_tensor(
      dims, ndim, dtype, strides, ACL_FORMAT_ND, dims, ndim, data_ptr);
}

aclTensorList* create_acl_tensor_list(const std::vector<aclTensor*>& tensors) {
  static const auto acl_create_tensor_list =
      aclnn::detail::get_op_api_func<aclnn::detail::AclCreateTensorListFn>(
          "aclCreateTensorList");
  CHECK(acl_create_tensor_list != nullptr)
      << "aclCreateTensorList is not available in libopapi.";
  std::vector<const aclTensor*> const_tensors(tensors.begin(), tensors.end());
  aclTensorList* list =
      acl_create_tensor_list(const_tensors.data(), const_tensors.size());
  CHECK(list != nullptr) << "aclCreateTensorList returned nullptr.";
  return list;
}

aclTensor* create_packed_weight_view(const torch::Tensor& packed,
                                     int64_t expert_idx,
                                     int64_t local_expert_num) {
  CHECK(packed.dim() == 3) << "MegaMoe packed weight must be 3D.";
  CHECK(local_expert_num > 0) << "MegaMoe local_expert_num must be > 0.";
  CHECK_EQ(packed.nbytes() % static_cast<size_t>(local_expert_num), 0)
      << "MegaMoe packed weight storage is not divisible by local "
         "expert count.";
  CHECK_GT(packed.itemsize(), 0)
      << "MegaMoe packed weight item size must be > 0.";
  CHECK_GE(expert_idx, 0) << "MegaMoe expert_idx must be >= 0.";
  CHECK_LT(expert_idx, local_expert_num)
      << "MegaMoe expert_idx must be < local_expert_num.";
  const int64_t view_dims[2] = {packed.size(1), packed.size(2)};
  int64_t strides[2];
  compact_strides(view_dims, 2, strides);
  const size_t expert_nbytes =
      packed.nbytes() / static_cast<size_t>(local_expert_num);
  CHECK_EQ(expert_nbytes % packed.itemsize(), 0)
      << "MegaMoe packed expert storage is not aligned to the item size.";
  auto* data = static_cast<uint8_t*>(packed.data_ptr()) +
               static_cast<size_t>(expert_idx) * expert_nbytes;
  if (is_fractal_nz(packed)) {
    // Match convert_type(): keep FRACTAL_NZ as a format flag and pass 1-D
    // storage size in elements, letting ACL infer the block layout from
    // format + view.
    const int64_t storage_dims[1] = {
        static_cast<int64_t>(expert_nbytes / packed.itemsize())};
    return create_acl_tensor(view_dims,
                             2,
                             ACL_INT8,
                             strides,
                             ACL_FORMAT_FRACTAL_NZ,
                             storage_dims,
                             1,
                             data);
  }
  return create_acl_tensor_nd(view_dims, 2, ACL_INT8, strides, data);
}

aclTensor* create_packed_scale_view(const torch::Tensor& packed,
                                    int64_t expert_idx,
                                    int64_t local_expert_num) {
  CHECK(local_expert_num > 0) << "MegaMoe local_expert_num must be > 0.";
  CHECK_EQ(packed.nbytes() % static_cast<size_t>(local_expert_num), 0)
      << "MegaMoe packed scale storage is not divisible by local "
         "expert count.";
  const int64_t expert_numel = packed.numel() / local_expert_num;
  const int64_t view_dims[1] = {expert_numel};
  const int64_t strides[1] = {1};
  const size_t expert_nbytes =
      packed.nbytes() / static_cast<size_t>(local_expert_num);
  auto* data = static_cast<uint8_t*>(packed.data_ptr()) +
               static_cast<size_t>(expert_idx) * expert_nbytes;
  // ATB reinterprets encoded int64 scale storage as uint64 for aclnnMegaMoe.
  return create_acl_tensor_nd(view_dims, 1, ACL_UINT64, strides, data);
}

void release_acl_tensors(const std::vector<aclTensor*>& tensors) {
  for (aclTensor* tensor : tensors) {
    aclnn::detail::release(tensor);
  }
}

void launch_aclnn_mega_moe(const torch::Tensor& context,
                           const torch::Tensor& x,
                           const torch::Tensor& topk_ids,
                           const torch::Tensor& topk_weights,
                           std::vector<aclTensor*> weight1_views,
                           std::vector<aclTensor*> weight2_views,
                           std::vector<aclTensor*> scale1_views,
                           std::vector<aclTensor*> scale2_views,
                           const std::optional<torch::Tensor>& x_active_mask,
                           int64_t moe_expert_num,
                           int64_t ep_world_size,
                           int64_t ccl_buffer_size,
                           int64_t max_recv_token_num,
                           int64_t dispatch_quant_mode,
                           int64_t dispatch_quant_out_dtype,
                           int64_t combine_quant_mode,
                           char* comm_alg,
                           int64_t num_max_tokens_per_rank,
                           char* activation,
                           float activation_clamp,
                           torch::Tensor& y,
                           torch::Tensor& expert_token_nums) {
  static const auto get_workspace_size_func_addr =
      aclnn::detail::get_op_api_func_addr("aclnnMegaMoeGetWorkspaceSize");
  static const auto op_api_func_addr =
      aclnn::detail::get_op_api_func_addr("aclnnMegaMoe");
  static const auto init_mem_addr =
      aclnn::detail::get_op_api_func_addr("InitHugeMemThreadLocal");
  static const auto uninit_mem_addr =
      aclnn::detail::get_op_api_func_addr("UnInitHugeMemThreadLocal");
  static const auto release_mem_addr =
      aclnn::detail::get_op_api_func_addr("ReleaseHugeMem");
  CHECK(get_workspace_size_func_addr != nullptr && op_api_func_addr != nullptr)
      << "aclnnMegaMoe is not available in "
      << aclnn::detail::get_op_api_lib_name();

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  auto init_mem_func =
      reinterpret_cast<aclnn::detail::InitHugeMemThreadLocalFn>(init_mem_addr);
  auto uninit_mem_func =
      reinterpret_cast<aclnn::detail::UnInitHugeMemThreadLocalFn>(
          uninit_mem_addr);
  if (init_mem_func) {
    init_mem_func(nullptr, false);
  }

  aclTensor* context_acl = aclnn::detail::convert_type(context);
  aclTensor* x_acl = aclnn::detail::convert_type(x);
  aclTensor* topk_ids_acl = aclnn::detail::convert_type(topk_ids);
  aclTensor* topk_weights_acl = aclnn::detail::convert_type(topk_weights);
  aclTensor* mask_acl = aclnn::detail::convert_type(x_active_mask);
  aclTensor* y_acl = aclnn::detail::convert_type(y);
  aclTensor* expert_token_nums_acl =
      aclnn::detail::convert_type(expert_token_nums);
  aclTensorList* weight1 = create_acl_tensor_list(weight1_views);
  aclTensorList* weight2 = create_acl_tensor_list(weight2_views);
  aclTensorList* weight_scales1 = create_acl_tensor_list(scale1_views);
  aclTensorList* weight_scales2 = create_acl_tensor_list(scale2_views);

  using GetWorkspaceSizeFn = int (*)(const aclTensor*,
                                     const aclTensor*,
                                     const aclTensor*,
                                     const aclTensor*,
                                     const aclTensorList*,
                                     const aclTensorList*,
                                     const aclTensorList*,
                                     const aclTensorList*,
                                     const aclTensorList*,
                                     const aclTensorList*,
                                     const aclTensor*,
                                     int64_t,
                                     int64_t,
                                     int64_t,
                                     int64_t,
                                     int64_t,
                                     int64_t,
                                     int64_t,
                                     const char*,
                                     int64_t,
                                     const char*,
                                     float,
                                     aclTensor*,
                                     aclTensor*,
                                     uint64_t*,
                                     aclOpExecutor**);
  auto get_workspace_size_func =
      reinterpret_cast<GetWorkspaceSizeFn>(get_workspace_size_func_addr);
  const int workspace_status = get_workspace_size_func(context_acl,
                                                       x_acl,
                                                       topk_ids_acl,
                                                       topk_weights_acl,
                                                       weight1,
                                                       weight2,
                                                       weight_scales1,
                                                       weight_scales2,
                                                       /*bias1=*/nullptr,
                                                       /*bias2=*/nullptr,
                                                       mask_acl,
                                                       moe_expert_num,
                                                       ep_world_size,
                                                       ccl_buffer_size,
                                                       max_recv_token_num,
                                                       dispatch_quant_mode,
                                                       dispatch_quant_out_dtype,
                                                       combine_quant_mode,
                                                       comm_alg,
                                                       num_max_tokens_per_rank,
                                                       activation,
                                                       activation_clamp,
                                                       y_acl,
                                                       expert_token_nums_acl,
                                                       &workspace_size,
                                                       &executor);
  CHECK(workspace_status == 0)
      << "call aclnnMegaMoeGetWorkspaceSize failed, detail:"
      << aclGetRecentErrMsg();

  void* workspace_addr = nullptr;
  at::Tensor workspace_tensor;
  if (workspace_size != 0) {
    at::TensorOptions options =
        at::TensorOptions(torch_npu::utils::get_npu_device_type());
    workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)},
                                 options.dtype(at::kByte));
    workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
  }

  auto acl_call = [=]() -> int {
    using OpApiFunc =
        int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
    OpApiFunc op_api_func = reinterpret_cast<OpApiFunc>(op_api_func_addr);
    auto api_ret =
        op_api_func(workspace_addr, workspace_size, executor, acl_stream);
    CHECK(api_ret == 0) << "call aclnnMegaMoe failed, detail:"
                        << aclGetRecentErrMsg();
    release_acl_tensors({context_acl,
                         x_acl,
                         topk_ids_acl,
                         topk_weights_acl,
                         mask_acl,
                         y_acl,
                         expert_token_nums_acl});
    aclnn::detail::release(weight1);
    aclnn::detail::release(weight2);
    aclnn::detail::release(weight_scales1);
    aclnn::detail::release(weight_scales2);
    auto release_mem_func =
        reinterpret_cast<aclnn::detail::ReleaseHugeMemFn>(release_mem_addr);
    if (release_mem_func) {
      release_mem_func(nullptr, false);
    }
    return api_ret;
  };
  at_npu::native::OpCommand cmd;
  cmd.Name("aclnnMegaMoe");
  cmd.SetCustomHandler(acl_call);
  cmd.Run();
  if (uninit_mem_func) {
    uninit_mem_func(nullptr, false);
  }
}

}  // namespace

bool has_mega_moe() {
  static const bool is_available =
      aclnn::detail::get_op_api_func_addr("aclnnMegaMoeGetWorkspaceSize") !=
          nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnMegaMoe") != nullptr;
  return is_available;
}

// To use the MegaMoe operator, you need to download CANN version 9.1.0 and the
// corresponding ops package. Download path:
// https://www.hiascend.com/cann/download?versionId=767&ids=d802%2Ch0501%2Ch0601%2Ch0703
// If you are using CANN 9.0.0 and an A3 environment, you can upgrade only the
// ops package to CANN 9.1.0.
std::tuple<torch::Tensor, torch::Tensor> apply_npu_mega_moe(
    const torch::Tensor& context,
    const torch::Tensor& x,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::TensorList weight1,
    const torch::TensorList weight2,
    int64_t moe_expert_num,
    int64_t ep_world_size,
    int64_t ccl_buffer_size,
    const std::optional<torch::TensorList>& weight_scales1,
    const std::optional<torch::TensorList>& weight_scales2,
    const std::optional<torch::TensorList>& bias1,
    const std::optional<torch::TensorList>& bias2,
    const std::optional<torch::Tensor>& x_active_mask,
    int64_t max_recv_token_num,
    int64_t dispatch_quant_mode,
    int64_t combine_quant_mode,
    const std::string& comm_alg,
    int64_t num_max_tokens_per_rank,
    const std::string& activation,
    float activation_clamp,
    int64_t dispatch_quant_out_dtype,
    int64_t topo_type,
    int64_t rank_num_per_server) {
  TORCH_CHECK(has_mega_moe(), "aclnnMegaMoe is not available in libopapi.");
  TORCH_CHECK(context.defined(), "MegaMoe expects a defined context tensor.");
  TORCH_CHECK(context.dim() == 1, "MegaMoe expects 1D context.");
  TORCH_CHECK(context.scalar_type() == at::kInt,
              "MegaMoe expects int32 context, got ",
              c10::toString(context.scalar_type()));
  TORCH_CHECK(x.dim() == 2, "MegaMoe expects 2D x.");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16,
              "MegaMoe expects bf16 x, got ",
              c10::toString(x.scalar_type()));
  TORCH_CHECK(topk_ids.dim() == 2, "MegaMoe expects 2D topk_ids.");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt,
              "MegaMoe expects int32 topk_ids, got ",
              c10::toString(topk_ids.scalar_type()));
  TORCH_CHECK(topk_weights.dim() == 2, "MegaMoe expects 2D topk_weights.");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
              "MegaMoe expects float32 topk_weights, got ",
              c10::toString(topk_weights.scalar_type()));
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(),
              "MegaMoe topk_ids/topk_weights shape mismatch: ",
              topk_ids.sizes(),
              " vs ",
              topk_weights.sizes());
  TORCH_CHECK(topk_ids.size(0) == x.size(0),
              "MegaMoe x/router token count mismatch: ",
              x.size(0),
              " vs ",
              topk_ids.size(0));
  TORCH_CHECK(!weight1.empty(), "MegaMoe expects non-empty weight1.");
  TORCH_CHECK(!weight2.empty(), "MegaMoe expects non-empty weight2.");
  TORCH_CHECK(weight1.size() == weight2.size(),
              "MegaMoe weight1/weight2 list size mismatch: ",
              weight1.size(),
              " vs ",
              weight2.size());
  TORCH_CHECK(moe_expert_num > 0, "MegaMoe requires moe_expert_num > 0.");
  TORCH_CHECK(ep_world_size > 0, "MegaMoe requires ep_world_size > 0.");
  TORCH_CHECK(moe_expert_num % ep_world_size == 0,
              "MegaMoe moe_expert_num must be divisible by ep_world_size: ",
              moe_expert_num,
              " vs ",
              ep_world_size);
  TORCH_CHECK(ccl_buffer_size > 0, "MegaMoe requires ccl_buffer_size > 0.");
  TORCH_CHECK(activation == "swiglu",
              "MegaMoe verified path requires swiglu activation, got ",
              activation);
  TORCH_CHECK(rank_num_per_server > 0,
              "MegaMoe requires rank_num_per_server > 0.");
  const int64_t local_expert_num = moe_expert_num / ep_world_size;
  const int64_t hidden_size = x.size(1);
  const bool use_w8a8 = dispatch_quant_mode == kMegaMoeDispatchQuantModeDynamic;
  if (!use_w8a8) {
    TORCH_CHECK(static_cast<int64_t>(weight1.size()) == local_expert_num,
                "MegaMoe A16W16 expects one weight pair per local expert: ",
                local_expert_num,
                ", got ",
                weight1.size());
    for (size_t expert = 0; expert < weight1.size(); ++expert) {
      const auto& w1 = weight1[expert];
      const auto& w2 = weight2[expert];
      TORCH_CHECK(w1.dim() == 2 && w2.dim() == 2,
                  "MegaMoe A16W16 expert weights must be 2D at local expert ",
                  expert,
                  ".");
      TORCH_CHECK(w1.scalar_type() == at::kBFloat16 &&
                      w2.scalar_type() == at::kBFloat16,
                  "MegaMoe A16W16 expects bf16 weights at local expert ",
                  expert,
                  ".");
      TORCH_CHECK(w1.size(0) == hidden_size,
                  "MegaMoe W1 hidden dimension mismatch at local expert ",
                  expert,
                  ": expected ",
                  hidden_size,
                  ", got ",
                  w1.size(0));
      TORCH_CHECK(w2.size(1) == hidden_size,
                  "MegaMoe W2 hidden dimension mismatch at local expert ",
                  expert,
                  ": expected ",
                  hidden_size,
                  ", got ",
                  w2.size(1));
      TORCH_CHECK(w1.size(1) == 2 * w2.size(0),
                  "MegaMoe W1/W2 intermediate dimension mismatch at local "
                  "expert ",
                  expert,
                  ": W1 columns ",
                  w1.size(1),
                  ", W2 rows ",
                  w2.size(0));
    }
  } else {
    TORCH_CHECK(weight1.size() == 1 && weight2.size() == 1,
                "MegaMoe W8A8 expects one packed weight tensor for each "
                "projection.");
    const auto& w1 = weight1[0];
    const auto& w2 = weight2[0];
    TORCH_CHECK(w1.dim() == 3 && w2.dim() == 3,
                "MegaMoe W8A8 packed weights must be 3D.");
    TORCH_CHECK(w1.scalar_type() == at::kChar && w2.scalar_type() == at::kChar,
                "MegaMoe W8A8 expects int8 packed weights.");
    TORCH_CHECK(
        w1.size(0) == local_expert_num && w2.size(0) == local_expert_num,
        "MegaMoe W8A8 local expert dimension mismatch.");
    TORCH_CHECK(w1.size(1) == hidden_size && w2.size(2) == hidden_size,
                "MegaMoe W8A8 hidden dimension mismatch.");
    TORCH_CHECK(w1.size(2) == 2 * w2.size(1),
                "MegaMoe W8A8 intermediate dimension mismatch.");
    TORCH_CHECK(weight_scales1.has_value() && weight_scales2.has_value(),
                "MegaMoe W8A8 requires encoded weight scales.");
    TORCH_CHECK(weight_scales1->size() == 1 && weight_scales2->size() == 1,
                "MegaMoe W8A8 expects one packed scale tensor for each "
                "projection.");
    const auto& scale1 = (*weight_scales1)[0];
    const auto& scale2 = (*weight_scales2)[0];
    TORCH_CHECK(
        scale1.scalar_type() == at::kLong && scale2.scalar_type() == at::kLong,
        "MegaMoe W8A8 encoded scales must use int64 storage.");
    TORCH_CHECK(scale1.numel() == local_expert_num * w1.size(2),
                "MegaMoe W8A8 weight1 scale size mismatch.");
    TORCH_CHECK(scale2.numel() == local_expert_num * hidden_size,
                "MegaMoe W8A8 weight2 scale size mismatch.");
    TORCH_CHECK(dispatch_quant_out_dtype == kMegaMoeDtypeInt8,
                "MegaMoe W8A8 dispatch output dtype must be ACL_INT8.");
    TORCH_CHECK(!bias1.has_value() && !bias2.has_value(),
                "MegaMoe W8A8 path does not accept expert bias.");
    if (x_active_mask.has_value() && x_active_mask->defined()) {
      TORCH_CHECK(x_active_mask->dim() == 1,
                  "MegaMoe W8A8 expects 1D x_active_mask.");
      TORCH_CHECK(x_active_mask->numel() == x.size(0),
                  "MegaMoe W8A8 x_active_mask token count mismatch: expected ",
                  x.size(0),
                  ", got ",
                  x_active_mask->numel());
      TORCH_CHECK(x_active_mask->scalar_type() == at::kChar,
                  "MegaMoe W8A8 expects int8 x_active_mask, got ",
                  c10::toString(x_active_mask->scalar_type()));
    }
  }

  auto y = at::empty_like(x);
  auto expert_token_nums =
      at::empty({local_expert_num}, x.options().dtype(at::kInt));

  std::string comm_alg_copy = comm_alg;
  char* comm_alg_ptr = comm_alg_copy.data();
  std::string activation_copy = activation;
  char* activation_ptr = activation_copy.data();

  const int64_t resolved_dispatch_quant_out_dtype =
      dispatch_quant_mode == kMegaMoeDispatchQuantModeNone
          ? kMegaMoeDtypeBFloat16
          : dispatch_quant_out_dtype;

  if (use_w8a8) {
    const auto& packed_w1 = weight1[0];
    const auto& packed_w2 = weight2[0];
    const auto& packed_s1 = (*weight_scales1)[0];
    const auto& packed_s2 = (*weight_scales2)[0];
    std::vector<aclTensor*> w1_views;
    std::vector<aclTensor*> w2_views;
    std::vector<aclTensor*> s1_views;
    std::vector<aclTensor*> s2_views;
    w1_views.reserve(local_expert_num);
    w2_views.reserve(local_expert_num);
    s1_views.reserve(local_expert_num);
    s2_views.reserve(local_expert_num);
    for (int64_t expert = 0; expert < local_expert_num; ++expert) {
      w1_views.push_back(
          create_packed_weight_view(packed_w1, expert, local_expert_num));
      w2_views.push_back(
          create_packed_weight_view(packed_w2, expert, local_expert_num));
      s1_views.push_back(
          create_packed_scale_view(packed_s1, expert, local_expert_num));
      s2_views.push_back(
          create_packed_scale_view(packed_s2, expert, local_expert_num));
    }
    launch_aclnn_mega_moe(context,
                          x,
                          topk_ids,
                          topk_weights,
                          std::move(w1_views),
                          std::move(w2_views),
                          std::move(s1_views),
                          std::move(s2_views),
                          x_active_mask,
                          moe_expert_num,
                          ep_world_size,
                          ccl_buffer_size,
                          max_recv_token_num,
                          dispatch_quant_mode,
                          resolved_dispatch_quant_out_dtype,
                          combine_quant_mode,
                          comm_alg_ptr,
                          num_max_tokens_per_rank,
                          activation_ptr,
                          activation_clamp,
                          y,
                          expert_token_nums);
    return std::make_tuple(y, expert_token_nums);
  }

  EXEC_NPU_CMD(aclnnMegaMoe,
               context,
               x,
               topk_ids,
               topk_weights,
               weight1,
               weight2,
               weight_scales1,  // weight_scales1 (nullptr on A16W16)
               weight_scales2,  // weight_scales2 (nullptr on A16W16)
               bias1,           // bias1 (nullptr on A16W16)
               bias2,           // bias2 (nullptr on A16W16)
               x_active_mask,
               //    no_tensor_list,    // shared_weight1 (no shared expert)
               //    no_tensor_list,    // shared_weight2
               //    no_tensor_list,    // shared_weight_scales1
               //    no_tensor_list,    // shared_weight_scales2
               //    no_tensor_list,    // shared_bias1
               //    no_tensor_list,    // shared_bias2
               moe_expert_num,
               ep_world_size,
               ccl_buffer_size,
               max_recv_token_num,
               dispatch_quant_mode,
               resolved_dispatch_quant_out_dtype,
               combine_quant_mode,
               comm_alg_ptr,
               num_max_tokens_per_rank,
               activation_ptr,
               activation_clamp,
               //    topo_type,
               //    rank_num_per_server,
               //    topkWeightsType,
               y,
               expert_token_nums);

  return std::make_tuple(y, expert_token_nums);
}

}  // namespace xllm::kernel::npu
