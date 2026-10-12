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

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
namespace py = pybind11;
#include <torch/torch.h>

#if defined(USE_NPU)
#include <acl/acl.h>
#endif

#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "api_service/api_service.h"
#include "core/common/global_flags.h"
#include "core/common/instance_name.h"
#include "core/common/metrics.h"
#include "core/common/options.h"
#include "core/common/types.h"
#include "core/distributed_runtime/master.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/help_formatter.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/profile_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/xtensor/global_xtensor.h"
#include "core/framework/xtensor/options.h"
#include "core/framework/xtensor/xtensor_allocator.h"
#include "core/platform/device_name_utils.h"
#include "core/util/net.h"
#include "core/util/utils.h"
#include "core/util/verbose_trace_logger.h"
#include "function_call/function_call_parser.h"
#include "parser/reasoning_parser.h"
#include "server/xllm_server_registry.h"
using namespace xllm;

static std::atomic<uint32_t> signal_received{0};

namespace {

void initialize_configs() {
  BeamSearchConfig::get_instance().initialize();
  DisaggPDConfig::get_instance().initialize();
  DistributedConfig::get_instance().initialize();
  DiTConfig::get_instance().initialize();
  EPLBConfig::get_instance().initialize();
  ExecutionConfig::get_instance().initialize();
  KernelConfig::get_instance().initialize();
  KVCacheConfig::get_instance().initialize();
  KVCacheStoreConfig::get_instance().initialize();
  LoadConfig::get_instance().initialize();
  ModelConfig::get_instance().initialize();
  ParallelConfig::get_instance().initialize();
  ProfileConfig::get_instance().initialize();
  RecConfig::get_instance().initialize();
  SchedulerConfig::get_instance().initialize();
  ServiceConfig::get_instance().initialize();
  SpeculativeConfig::get_instance().initialize();
}

Options create_options(const std::string& instance_name, bool is_local) {
  const ServiceConfig& service_config = ServiceConfig::get_instance();
  const ModelConfig& model_config = ModelConfig::get_instance();
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  const KVCacheStoreConfig& kv_cache_store_config =
      KVCacheStoreConfig::get_instance();
  const BeamSearchConfig& beam_search_config = BeamSearchConfig::get_instance();
  const SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  const ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const EPLBConfig& eplb_config = EPLBConfig::get_instance();
  const DistributedConfig& distributed_config =
      DistributedConfig::get_instance();
  const DisaggPDConfig& disagg_pd_config = DisaggPDConfig::get_instance();
  const SpeculativeConfig& speculative_config =
      SpeculativeConfig::get_instance();
  const ProfileConfig& profile_config = ProfileConfig::get_instance();
  const ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  const KernelConfig& kernel_config = KernelConfig::get_instance();
  const DiTConfig& dit_config = DiTConfig::get_instance();
  const RecConfig& rec_config = RecConfig::get_instance();

#if !defined(USE_NPU)
  CHECK(!speculative_config.enable_mtp_draft_body_tp1())
      << "enable_mtp_draft_body_tp1 is only supported on the NPU backend";
#endif

  Options options;
#if defined(USE_NPU)
  options.npu_kernel_backend(kernel_config.npu_kernel_backend());
  options.enable_flashcomm1(kernel_config.enable_flashcomm1())
      .flashcomm1_min_prefill_tokens(
          kernel_config.flashcomm1_min_prefill_tokens())
      .enable_mmrs_fusion(kernel_config.enable_mmrs_fusion())
      .mmrs_comm_mode(kernel_config.mmrs_comm_mode());
#endif
  options.model_path(model_config.model())
      .model_id(model_config.model_id())
      .task_type(model_config.task())
      .draft_model_path(speculative_config.draft_model())
      .backend(model_config.backend())
      .limit_image_per_prompt(model_config.limit_image_per_prompt())
      .max_encoder_cache_size(model_config.max_encoder_cache_size())
      .max_processor_cache_items(model_config.max_processor_cache_items())
      .block_size(kv_cache_config.block_size())
      .max_cache_size(kv_cache_config.max_cache_size())
      .max_memory_utilization(kv_cache_config.max_memory_utilization())
      .enable_prefix_cache(kv_cache_config.enable_prefix_cache())
      .max_linear_state_cache_slots(
          kv_cache_config.max_linear_state_cache_slots())
      .max_tokens_per_batch(scheduler_config.max_tokens_per_batch())
      .max_seqs_per_batch(scheduler_config.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          scheduler_config.max_tokens_per_chunk_for_prefill())
      .num_speculative_tokens(speculative_config.num_speculative_tokens())
      .speculative_algorithm(speculative_config.speculative_algorithm())
      .draft_sampling_mode(speculative_config.draft_sampling_mode())
      .speculative_suffix_cache_max_depth(
          speculative_config.speculative_suffix_cache_max_depth())
      .speculative_suffix_max_spec_factor(
          speculative_config.speculative_suffix_max_spec_factor())
      .speculative_suffix_max_spec_offset(
          speculative_config.speculative_suffix_max_spec_offset())
      .speculative_suffix_min_token_prob(
          speculative_config.speculative_suffix_min_token_prob())
      .speculative_suffix_max_cached_requests(
          speculative_config.speculative_suffix_max_cached_requests())
      .speculative_suffix_use_tree_spec(
          speculative_config.speculative_suffix_use_tree_spec())
      .enable_mtp_draft_body_tp1(speculative_config.enable_mtp_draft_body_tp1())
      .enable_adaptive_speculative_decode(
          speculative_config.enable_adaptive_speculative_decode())
      .adaptive_speculative_min_gain(
          speculative_config.adaptive_speculative_min_gain())
      .num_request_handling_threads(
          service_config.num_request_handling_threads())
      .communication_backend(parallel_config.communication_backend())
      .enable_eplb(eplb_config.enable_eplb())
      .redundant_experts_num(eplb_config.redundant_experts_num())
      .eplb_update_interval(eplb_config.eplb_update_interval())
      .eplb_min_peak_load_improvement(
          eplb_config.eplb_min_peak_load_improvement())
      .rank_tablefile(eplb_config.rank_tablefile())
      .expert_parallel_degree(eplb_config.expert_parallel_degree())
      .enable_chunked_prefill(scheduler_config.enable_chunked_prefill())
      .master_node_addr(distributed_config.master_node_addr())
      .instance_role(InstanceRole(disagg_pd_config.instance_role()))
      .transfer_listen_port(
          static_cast<uint16_t>(disagg_pd_config.transfer_listen_port()))
      .nnodes(distributed_config.nnodes())
      .node_rank(distributed_config.node_rank())
      .dp_size(parallel_config.dp_size())
      .cp_size(parallel_config.cp_size())
      .ep_size(parallel_config.ep_size())
      .tp_size(static_cast<int32_t>(parallel_config.tp_size()))
      .sp_size(static_cast<int32_t>(parallel_config.sp_size()))
      .cfg_size(static_cast<int32_t>(parallel_config.cfg_size()))
      .vae_size(static_cast<int32_t>(parallel_config.vae_size()))
      .text_encoder_tp_size(
          static_cast<int32_t>(parallel_config.text_encoder_tp_size()))
      .instance_name(instance_name)
      .enable_disagg_pd(disagg_pd_config.enable_disagg_pd())
      .enable_pd_ooc(disagg_pd_config.enable_pd_ooc())
      .enable_schedule_overlap(scheduler_config.enable_schedule_overlap())
      .kv_cache_transfer_mode(disagg_pd_config.kv_cache_transfer_mode())
      .etcd_addr(distributed_config.etcd_addr())
      .etcd_namespace(distributed_config.etcd_namespace())
      .enable_service_routing(distributed_config.enable_service_routing() ||
                              disagg_pd_config.enable_disagg_pd())
      .tool_call_parser(model_config.tool_call_parser())
      .reasoning_parser(model_config.reasoning_parser())
      .priority_strategy(scheduler_config.priority_strategy())
      .enable_online_preempt_offline(
          scheduler_config.enable_online_preempt_offline())
      .host_blocks_factor(kv_cache_store_config.host_blocks_factor())
      .enable_kvcache_store(kv_cache_store_config.enable_kvcache_store())
      .prefetch_timeout(kv_cache_store_config.prefetch_timeout())
      .prefetch_batch_size(kv_cache_store_config.prefetch_batch_size())
      .layers_wise_copy_batchs(kv_cache_store_config.layers_wise_copy_batchs())
      .store_protocol(kv_cache_store_config.store_protocol())
      .store_rdma_devices(kv_cache_store_config.store_rdma_devices())
      .store_master_server_address(
          kv_cache_store_config.store_master_server_address())
      .store_metadata_server(kv_cache_store_config.store_metadata_server())
      .store_local_hostname(kv_cache_store_config.store_local_hostname())
      .enable_multi_stream_parallel(
          parallel_config.enable_multi_stream_parallel())
      .enable_profile_step_time(profile_config.enable_profile_step_time())
      .enable_profile_token_budget(profile_config.enable_profile_token_budget())
      .enable_latency_aware_schedule(
          profile_config.enable_latency_aware_schedule())
      .profile_max_prompt_length(profile_config.profile_max_prompt_length())
      .enable_profile_kv_blocks(profile_config.enable_profile_kv_blocks())
      .disable_ttft_profiling(profile_config.disable_ttft_profiling())
      .enable_forward_interruption(profile_config.enable_forward_interruption())
      .enable_graph(execution_config.enable_graph())
      .enable_graph_mode_decode_no_padding(
          execution_config.enable_graph_mode_decode_no_padding())
      .enable_prefill_piecewise_graph(
          execution_config.enable_prefill_piecewise_graph())
      .max_tokens_for_graph_mode(execution_config.max_tokens_for_graph_mode())
      .max_global_ttft_ms(profile_config.max_global_ttft_ms())
      .max_global_tpot_ms(profile_config.max_global_tpot_ms())
      .max_requests_per_batch(dit_config.max_requests_per_batch())
      .enable_shm(execution_config.enable_shm())
      .input_shm_size(execution_config.input_shm_size())
      .output_shm_size(execution_config.output_shm_size())
      .beam_width(beam_search_config.beam_width())
      .kv_cache_dtype(kv_cache_config.kv_cache_dtype())
      .rec_worker_max_concurrency(
          static_cast<int32_t>(rec_config.rec_worker_max_concurrency()))
      .is_local(is_local);

  return options;
}

}  // namespace

#if defined(USE_NPU)
namespace {
// Initialize Python interpreter and torch_npu runtime early, before any NPU
// tensor allocation. torch_npu (post4+) calls PyGILState_Ensure inside
// empty_with_format(), so Python must be alive before the first NPU op.
// All NPU processes go through this path for consistency — the build system
// links against the pip-installed torch_npu .so directly.
void init_npu_python_runtime() {
  bool we_initialized_python = false;
  if (!Py_IsInitialized()) {
    py::initialize_interpreter(/*init_signal_handlers=*/false);
    we_initialized_python = true;
  }

  // Select the same logical device this process's worker will run on. Multi-
  // process single-card serving lets every process see all TP cards and picks
  // its own via node_rank (see Master ctor). Using .front() here would pin an
  // extra context on logical device 0 for every node_rank != 0 process, piling
  // small allocations onto die0. Mirror master.cpp's get_device_idx instead.
  const auto& distributed_config = DistributedConfig::get_instance();
  const int32_t visible_device_count =
      DeviceNameUtils::parse_devices("auto").size();
  const int32_t device_index =
      DeviceNameUtils::get_device_idx(distributed_config.node_rank(),
                                      distributed_config.nnodes(),
                                      visible_device_count);

  // Register fla_npu's embedded AscendC opapi (libcust_opapi) + OPP vendor
  // dir before aclInit. The KDA forward op (aclnnChunkKdaFwd) and its kernels
  // live in fla_npu's embedded OPP; without this registration the opapi cannot
  // locate the kernel binary and aclnnChunkKdaFwdGetWorkspaceSize returns
  // ACLNN_ERR_INNER_NULLPTR (561103), breaking GLM-5.3-Flash's KDA layers.
  // Guarded so non-KDA builds without fla_npu stay unaffected.
  {
    py::gil_scoped_acquire gil;
    py::exec(
        "try:\n"
        "    import fla_npu\n"
        "    fla_npu.load_ascendc_opapi_libraries()\n"
        "except (ImportError, RuntimeError):\n"
        "    pass\n");
  }

  auto acl_ret = aclInit(nullptr);
  CHECK(acl_ret == ACL_SUCCESS || acl_ret == 500000)
      << "aclInit failed with error " << acl_ret;

  {
    py::gil_scoped_acquire gil;
    py::exec(
        "import os, sys\n"
        "os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'\n"
        "import torch\n"
        "orig = torch._C._get_accelerator\n"
        "try:\n"
        "    torch._C._get_accelerator = lambda: torch.device('cpu')\n"
        "    import torch_npu\n"
        "finally:\n"
        "    torch._C._get_accelerator = orig\n"
        "import torch_npu.npu as _npu_mod\n"
        "try:\n"
        "    torch_npu._C._npu_init()\n"
        "except RuntimeError as e:\n"
        "    if 'already initialized' not in str(e).lower():\n"
        "        raise\n"
        "_npu_mod._initialized = True\n"
        "_npu_mod._original_pid = os.getpid()\n"
        "torch_npu._C._npu_setDevice(" +
        std::to_string(device_index) + ")\n");
  }

  if (we_initialized_python) {
    PyEval_SaveThread();
  }
}

// Hold a flock across the write -> aclInit read of vendors/config.ini so a
// different-backend process cannot overwrite the file between our write and
// CANN's read (review P2 cross-backend race: e.g. python writes, pauses,
// native overwrites, python resumes and reads native's order). CANN resolves
// the aclnnSparseFlashAttention vendor from config.ini at aclInit (verified:
// the order present at aclInit determines the outcome), so covering write ->
// aclInit closes the read-phase race.
//
// Lock discipline: a process whose desired order already matches the file
// takes a shared lock and runs aclInit concurrently with same-backend peers;
// a process that needs to change the order takes a brief exclusive lock,
// publishes atomically, then drops to a shared lock for aclInit. A
// different-backend writer cannot acquire the exclusive lock while same-
// backend readers hold the shared lock, so it cannot interpose during their
// aclInit. The exclusive section never spans aclInit, so same-backend peers
// do not serialize on aclInit (only on the brief write).
//
// Only installed target vendors are reordered (a target vendor whose
// $ASCEND_OPP_PATH/vendors/<name>/op-api/lib is absent is dropped, never
// prepended -- a missing entry would make the next process's
// get_default_custom_lib_path break the whole vendor search loop); any other
// vendor already present is preserved. The publish is atomic (mkstemp + fsync
// + fchmod/fchown to preserve owner/perms + rename, with the POSIX ACL copied
// strictly so the rename does not drop it).
// XLLM_DISABLE_VENDOR_CONFIG_INI_WRITE=1 disables. A failure to even read the
// active order (bad stream / exists unreadable) leaves config.ini unchanged
// and is non-fatal (proceed with the existing config). A publish failure on a
// CONFIRMED different-backend order (e.g. no write permission after the
// O_RDONLY fallback) fail-stops instead: aclInit must never run on a non-our
// order -- correctness first (review #29).

// Build the full load_priority= line: desired order first, then any other
// vendor already present (preserved, order kept), so a registered vendor is
// never dropped. Used by both the match check and the publish path so they
// agree on what "matches" means.
std::string build_priority_line(const std::vector<std::string>& desired_order,
                                const std::vector<std::string>& existing) {
  std::vector<std::string> vendors = desired_order;
  for (const std::string& v : existing) {
    if (v.empty()) {
      continue;
    }
    if (std::find(vendors.begin(), vendors.end(), v) == vendors.end()) {
      vendors.push_back(v);
    }
  }
  return "load_priority=" + absl::StrJoin(vendors, ",");
}

// Copy xattrs from src to the open dst fd. The POSIX access ACL
// (system.posix_acl_access) is copied strictly: if it exists on src but cannot
// be copied, returns false so the caller refuses to rename -- the atomic
// rename would otherwise silently drop the ACL and other accounts would lose
// read access to the shared OPP config. Other xattrs (user.*, security.*, ...)
// are best-effort; a failure there does not block the publish. A filesystem
// without xattr support (ENOTSUP) or a src without the ACL returns true
// (nothing critical to copy).
bool copy_xattrs_to_fd(const std::filesystem::path& src, int dst_fd) {
  static constexpr const char* kAcl = "system.posix_acl_access";
  // Probe the ACL first: getxattr on the ACL name distinguishes "no ACL"
  // (ENODATA/ENOENT, errno set, sz<0), "no xattr support" (ENOTSUP, sz<0), and
  // "ACL present" (sz>0) without enumerating every xattr.
  ssize_t acl_sz = ::getxattr(src.c_str(), kAcl, nullptr, 0);
  if (acl_sz < 0) {
    if (errno == ENODATA || errno == ENOENT || errno == ENOTSUP) {
      return true;  // no ACL on src / no src / FS lacks xattr support
    }
    LOG(WARNING) << "getxattr ACL " << src << " failed ("
                 << std::strerror(errno) << "); config.ini left unchanged";
    return false;  // cannot determine ACL safety -- refuse to rename
  }
  if (acl_sz == 0) {
    return true;  // present but empty -- nothing to copy
  }
  std::vector<char> acl(static_cast<size_t>(acl_sz));
  ssize_t got = ::getxattr(src.c_str(), kAcl, acl.data(), acl.size());
  if (got <= 0) {
    LOG(WARNING) << "read ACL " << src << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  if (::fsetxattr(dst_fd, kAcl, acl.data(), static_cast<size_t>(got), 0) != 0) {
    LOG(WARNING) << "fsetxattr ACL on temp failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;  // would drop the ACL on rename -- refuse
  }
  // Best-effort: copy any OTHER xattrs. A failure here is non-critical (they
  // are not access-control) and does not block the publish.
  ssize_t list_sz = ::listxattr(src.c_str(), nullptr, 0);
  if (list_sz <= 0) {
    return true;  // no other xattrs / ENOTSUP
  }
  std::vector<char> names(static_cast<size_t>(list_sz));
  ssize_t actual = ::listxattr(src.c_str(), names.data(), names.size());
  if (actual <= 0) {
    return true;
  }
  size_t off = 0;
  const size_t end = static_cast<size_t>(actual);
  while (off < end) {
    const char* name = names.data() + off;
    const size_t name_len = std::strlen(name);
    off += name_len + 1;
    if (std::strcmp(name, kAcl) == 0) {
      continue;  // already copied strictly above
    }
    ssize_t val_sz = ::getxattr(src.c_str(), name, nullptr, 0);
    if (val_sz <= 0) {
      continue;
    }
    std::vector<char> val(static_cast<size_t>(val_sz));
    if (::getxattr(src.c_str(), name, val.data(), val.size()) <= 0) {
      continue;
    }
    ::fsetxattr(dst_fd, name, val.data(), val.size(), 0);  // best-effort
  }
  return true;
}

// Atomic publish of config.ini content via mkstemp + rename, replicating the
// original owner/group/permissions (and xattrs/ACL). Returns true on success;
// on any failure the existing config.ini is left untouched.
bool publish_config_ini(const std::filesystem::path& target,
                        const std::string& content) {
  namespace fs = std::filesystem;
  const fs::path dir = target.parent_path();
  std::string tmpl = (dir / "config.ini.xllm.tmp.XXXXXX").string();
  const int fd = ::mkstemp(tmpl.data());
  if (fd < 0) {
    LOG(WARNING) << "mkstemp in " << dir << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  const fs::path tmp = tmpl;
  class PublishGuard {
   public:
    int fd;
    fs::path path;
    bool dismissed = false;
    ~PublishGuard() {
      if (fd >= 0) {
        ::close(fd);
      }
      if (!dismissed) {
        std::error_code ec;
        fs::remove(path, ec);
      }
    }
  } guard{fd, tmp, false};

  size_t off = 0;
  while (off < content.size()) {
    const ssize_t n = ::write(fd, content.data() + off, content.size() - off);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      LOG(WARNING) << "write " << tmp << " failed (" << std::strerror(errno)
                   << "); config.ini left unchanged";
      return false;
    }
    off += static_cast<size_t>(n);
  }
  if (::fsync(fd) != 0) {
    LOG(WARNING) << "fsync " << tmp << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  struct stat st;
  mode_t mode = 0644;
  uid_t uid = static_cast<uid_t>(-1);  // -1 leaves owner unchanged by fchown
  gid_t gid = static_cast<gid_t>(-1);
  if (::stat(target.c_str(), &st) == 0) {
    mode = st.st_mode & 0777;
    uid = st.st_uid;
    gid = st.st_gid;
  } else if (errno != ENOENT) {
    LOG(WARNING) << "stat " << target << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  if (::fchmod(fd, mode) != 0) {
    LOG(WARNING) << "fchmod " << tmp << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  if ((uid != static_cast<uid_t>(-1) || gid != static_cast<gid_t>(-1)) &&
      ::fchown(fd, uid, gid) != 0) {
    LOG(WARNING) << "fchown " << tmp << " failed (" << std::strerror(errno)
                 << "); cannot preserve ownership, config.ini left unchanged";
    return false;
  }
  // Copy xattrs (incl. POSIX ACL) from the original so the rename does not
  // drop access entries that other accounts rely on for read access. A
  // failure to copy an existing ACL refuses the publish (config.ini left
  // unchanged) rather than shipping a file with a silently dropped ACL.
  if (!copy_xattrs_to_fd(target, fd)) {
    return false;
  }
  const int close_ret = ::close(fd);
  guard.fd = -1;
  if (close_ret != 0) {
    LOG(WARNING) << "close " << tmp << " failed (" << std::strerror(errno)
                 << "); config.ini left unchanged";
    return false;
  }
  std::error_code rename_ec;
  fs::rename(tmp, target, rename_ec);
  if (rename_ec) {
    LOG(WARNING) << "rename " << tmp << " -> " << target << " failed ("
                 << rename_ec.message() << "); config.ini left unchanged";
    return false;
  }
  guard.dismissed = true;
  return true;
}

// Parsed view of the current config.ini.
struct VendorConfigRead {
  std::vector<std::string> lines;
  int32_t priority_index = -1;  // index into lines of the active load_priority=
  std::vector<std::string> existing_vendors;  // parsed from that line
  bool bad = false;                           // mid-stream I/O error
  bool exists_unreadable = false;  // file exists but ifstream failed to open
};

VendorConfigRead read_vendor_config(const std::filesystem::path& target) {
  VendorConfigRead r;
  const std::string prefix = "load_priority=";
  std::ifstream ifs(target);
  if (!ifs) {
    std::error_code ec;
    if (std::filesystem::exists(target, ec)) {
      r.exists_unreadable = true;
    }
    return r;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.rfind(prefix, 0) == 0) {
      r.priority_index = static_cast<int32_t>(r.lines.size());
      if (line.size() > prefix.size()) {
        r.existing_vendors = std::vector<std::string>(
            absl::StrSplit(line.substr(prefix.size()), ','));
      }
    }
    r.lines.push_back(line);
  }
  if (ifs.bad()) {
    r.bad = true;
  }
  return r;
}

// Compose the new content (preserve other lines, replace/append the active
// load_priority line) and atomically publish. Returns true if the active line
// already equals the desired priority (no write) OR the publish succeeded; on
// a successful publish *wrote (if non-null) is set true. Returns false only on
// publish failure.
bool ensure_priority_published(const std::filesystem::path& target,
                               const std::vector<std::string>& desired_order,
                               const VendorConfigRead& r,
                               bool* wrote = nullptr) {
  namespace fs = std::filesystem;
  if (wrote != nullptr) {
    *wrote = false;
  }
  const std::string new_priority =
      build_priority_line(desired_order, r.existing_vendors);
  // Match on the NORMALIZED current order, not the raw disk line: a line that
  // differs only cosmetically (trailing comma, duplicates, empty segments)
  // resolves to the same vendor sequence, so it already "matches" and must not
  // trigger a normalization publish whose failure would be misclassified as a
  // different-backend order (review #34/#36).
  const std::string current = build_priority_line({}, r.existing_vendors);
  if (r.priority_index >= 0 && current == new_priority) {
    return true;  // effective order already matches (cosmetics aside)
  }
  std::vector<std::string> lines = r.lines;
  if (r.priority_index >= 0) {
    lines[static_cast<size_t>(r.priority_index)] = new_priority;
  } else {
    lines.push_back(new_priority);
  }
  std::string content;
  for (const std::string& l : lines) {
    content += l;
    content += "\n";
  }
  if (publish_config_ini(target, content)) {
    if (wrote != nullptr) {
      *wrote = true;
    }
    return true;
  }
  return false;
}

// RAII guard that holds a flock across the caller's aclInit (which reads
// config.ini) so a different-backend process cannot clobber the file in
// between. Construct immediately before init_npu_python_runtime(); the lock is
// released on destruction. The caller's aclInit runs under a shared lock
// whose order matches this process, so a different-backend writer (which needs
// an exclusive lock) cannot interpose during the read phase.
class VendorConfigLock {
 public:
  VendorConfigLock() {
    const char* disable = std::getenv("XLLM_DISABLE_VENDOR_CONFIG_INI_WRITE");
    if (disable != nullptr && std::string(disable) == "1") {
      return;  // fd_ stays -1: no lock, no write
    }
    const char* opp_path_env = std::getenv("ASCEND_OPP_PATH");
    if (opp_path_env == nullptr || std::string(opp_path_env).empty()) {
      LOG(WARNING) << "ASCEND_OPP_PATH is not set; skip vendor config lock";
      return;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path config_file =
        fs::canonical(fs::path(opp_path_env) / "vendors" / "config.ini", ec);
    target_ =
        ec ? fs::path(opp_path_env) / "vendors" / "config.ini" : config_file;
    compute_desired();
    if (desired_order_.empty()) {
      // Legacy/no-target-vendor host: none of this backend's target vendors is
      // installed, so we have no order to enforce and must NOT touch the shared
      // config. Be a true no-op: take no lock, write nothing, proceed with the
      // existing config (state_ stays kIdle). This also avoids misclassifying a
      // cosmetic-only normalization write failure as kUnsettled (review #34).
      LOG(WARNING) << "no target vendor for " << backend_label_
                   << " backend is installed under " << target_.parent_path()
                   << "; vendor config hook is a no-op (proceeding with the "
                      "existing config, no lock, no write)";
      return;
    }
    const fs::path lockfile = target_.parent_path() / ".xllm_vendor.flock";
    // Open the lockfile world-read-write (0666) so processes of different UIDs
    // sharing the same OPP can all acquire it. flock works on a read-only fd
    // too, so if an existing root-owned 0644 lockfile denies O_RDWR, fall back
    // to O_RDONLY (review cross-UID EACCES bypass). On any open failure we do
    // NOT write: an unlocked write could clobber the file during another
    // process's aclInit read, bypassing the very mutex this guard provides.
    fd_ = ::open(lockfile.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (fd_ >= 0) {
      ::fchmod(fd_, 0666);  // best-effort: upgrade an older 0644 lockfile
    } else if (errno == EACCES) {
      fd_ = ::open(lockfile.c_str(), O_RDONLY | O_CLOEXEC);
    }
    if (fd_ < 0) {
      LOG(WARNING) << "open lockfile " << lockfile << " failed ("
                   << std::strerror(errno)
                   << "); vendor lock unavailable, skip write (leave "
                      "config.ini as-is)";
      return;  // state_ stays kIdle: non-fatal, proceed with existing config
    }
    // acquire_with_retry sets state_: kSettled (hold SH through aclInit),
    // kIdle (unreadable order or genuine lock/read error -> non-fatal,
    // proceed with existing config), or kUnsettled (persistent cross-backend
    // contention, OR a publish failure on a confirmed different-backend order
    // -> caller fail-stops, never aclInit on a non-our order).
    acquire_with_retry();
  }
  ~VendorConfigLock() {
    if (fd_ >= 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
      fd_ = -1;
    }
  }
  VendorConfigLock(const VendorConfigLock&) = delete;
  VendorConfigLock& operator=(const VendorConfigLock&) = delete;

  // True when the lock is usable but the order could not be settled to this
  // backend's order under persistent cross-backend contention. In that case we
  // must NOT proceed to aclInit on a different-backend order -- the caller
  // fail-stops. Distinct from the "no lock available / no write permission"
  // case (kIdle), which is non-fatal and proceeds with the existing config.
  bool unsettled() const { return state_ == State::kUnsettled; }

 private:
  enum class State : int8_t { kIdle, kSettled, kUnsettled };
  State state_ = State::kIdle;
  int fd_ = -1;
  std::filesystem::path target_;
  std::vector<std::string> desired_order_;
  std::string backend_label_;

  static std::vector<std::string> desired_order_for(bool is_python) {
    // Vendor names are project conventions; xllm_ops renamed "xllm" to
    // "custom_xllm_math" (see CMakeLists.txt / scripts/build_support/utils.py),
    // keep these in sync if a vendor name changes again.
    return is_python ? std::vector<std::string>{"glm_next_transformer",
                                                "custom_transformer",
                                                "custom_xllm_math"}
                     : std::vector<std::string>{"custom_xllm_math",
                                                "custom_transformer",
                                                "glm_next_transformer"};
  }

  void compute_desired() {
    const bool is_python = ModelConfig::is_python_model_impl(
        ModelConfig::get_instance().model_impl());
    desired_order_ = desired_order_for(is_python);
    backend_label_ = is_python ? "python" : "atb";
    // Keep only target vendors whose libcust_opapi.so is actually installed
    // under $ASCEND_OPP_PATH/vendors/<name>/op_api/lib -- the EXACT artifact
    // get_op_api_func_addr dlopens (pytorch_npu_helper.hpp:277-280). Checking
    // only the lib directory is not enough: a vendor whose op_api/lib exists
    // but whose libcust_opapi.so is missing (partial/broken/uninstall residue)
    // resolves to an empty real_path and BREAKs the whole default-vendor search
    // loop, skipping later, actually-usable vendors (review #35). On this host
    // all three are installed, so this is a no-op; on a legacy 'xllm' layout
    // where none of the target .so files exists the feature becomes a safe
    // no-op (see the empty-desired short-circuit in the ctor) rather than
    // poisoning the shared config.
    namespace fs = std::filesystem;
    const fs::path vendors_dir = target_.parent_path();
    std::error_code ec;
    std::vector<std::string> installed;
    installed.reserve(desired_order_.size());
    for (const std::string& v : desired_order_) {
      const fs::path so =
          vendors_dir / v / "op_api" / "lib" / "libcust_opapi.so";
      if (fs::exists(so, ec)) {
        installed.push_back(v);
      } else {
        LOG(WARNING) << "vendor " << v << " has no libcust_opapi.so under "
                     << vendors_dir << "; dropping from " << backend_label_
                     << " load_priority to avoid breaking the vendor search";
      }
    }
    desired_order_ = std::move(installed);
  }

  // Outcome of a publish attempt under LOCK_EX.
  enum class PublishResult : int8_t {
    kPublished,  // wrote our order
    kMatched,    // already matched, no write
    // Read was bad / file exists but unreadable: the active order could not be
    // determined, so we cannot know whether proceeding is safe. Non-fatal
    // (kIdle): proceed with the existing config, matching the "no read / no
    // write permission does not block start" contract.
    kPublishFailedUnreadable,
    // Read was valid, the EFFECTIVE order (deduped, non-empty -- not just a
    // cosmetic text difference like trailing comma / duplicates, which already
    // matched above and never reach here) differs from desired, and the publish
    // failed (mkstemp/write/fsync/fchmod/fchown/xattr/rename -- typically
    // EACCES on a cross-UID O_RDONLY fallback). Proceeding would aclInit on a
    // confirmed different-backend order, violating the "never aclInit on a
    // non-our order
    // -- correctness first" invariant. This is kUnsettled (caller fail-stops),
    // NOT kIdle (review #29; cosmetic-only differences are caught upstream as
    // kMatched and never reach this branch -- review #34/#36).
    kPublishFailedStale,
  };

  // Outcome of a shared-lock match check.
  enum class CheckResult : int8_t {
    kCheckSettled,
    kCheckMismatch,
    kCheckAbort,
  };

  // Caller holds LOCK_EX on fd_. Re-read config and publish our order if it
  // does not already match. Does not release the lock.
  PublishResult publish_under_exclusive() {
    VendorConfigRead w = read_vendor_config(target_);
    if (w.bad || w.exists_unreadable) {
      return PublishResult::kPublishFailedUnreadable;
    }
    bool wrote = false;
    if (!ensure_priority_published(target_, desired_order_, w, &wrote)) {
      // Valid read, active order != desired (else ensure_priority_published
      // would have returned true on match), and the publish failed: proceeding
      // would aclInit on a different-backend order.
      return PublishResult::kPublishFailedStale;
    }
    if (wrote) {
      LOG(INFO) << "wrote " << target_ << " "
                << build_priority_line(desired_order_, w.existing_vendors)
                << " for " << backend_label_ << " backend";
      return PublishResult::kPublished;
    }
    return PublishResult::kMatched;  // already matched, no write
  }

  // Acquire SH, re-read config, and check whether the file matches our desired
  // order. On match, sets state_ = kSettled (hold SH through aclInit) and
  // returns kCheckSettled. On mismatch, releases SH and returns
  // kCheckMismatch (caller retries). On a genuine lock/read error, returns
  // kCheckAbort and leaves state_ = kIdle (non-fatal, proceed with existing
  // config).
  CheckResult check_match_under_shared() {
    if (::flock(fd_, LOCK_SH) != 0) {
      LOG(WARNING) << "flock LOCK_SH failed (" << std::strerror(errno) << ")";
      return CheckResult::kCheckAbort;  // state_ stays kIdle (non-fatal)
    }
    VendorConfigRead r = read_vendor_config(target_);
    if (r.bad) {
      LOG(WARNING) << "read " << target_ << " failed mid-stream";
      ::flock(fd_, LOCK_UN);
      return CheckResult::kCheckAbort;
    }
    if (r.exists_unreadable) {
      LOG(WARNING) << target_ << " exists but cannot be read; skip lock";
      ::flock(fd_, LOCK_UN);
      return CheckResult::kCheckAbort;
    }
    // Compare the NORMALIZED current order (deduped, empty segments dropped)
    // to the normalized target, NOT the raw disk line. A raw line differing
    // only by cosmetics (trailing comma, duplicate entries, leading/trailing
    // whitespace segments) resolves to the SAME vendor sequence in
    // get_default_custom_lib_path/get_op_api_func_addr, so it must count as a
    // match and NOT trigger a (possibly failing) normalization publish that
    // would then be misclassified as kPublishFailedStale -> kUnsettled
    // (review #34/#36). build_priority_line({}, existing) is exactly the
    // deduped/non-empty re-join of the parsed existing vendors.
    const std::string want =
        build_priority_line(desired_order_, r.existing_vendors);
    const std::string current = build_priority_line({}, r.existing_vendors);
    if (r.priority_index >= 0 && current == want) {
      LOG(INFO) << "vendors/config.ini load_priority matches (" << want
                << "); holding shared lock through aclInit for "
                << backend_label_ << " backend";
      state_ = State::kSettled;  // hold SH through aclInit
      return CheckResult::kCheckSettled;
    }
    ::flock(fd_, LOCK_UN);
    return CheckResult::kCheckMismatch;
  }

  // Settle so that, on success, fd_ holds a shared lock while the file's
  // active order equals desired_order_ (state_ == kSettled). A bounded
  // non-blocking probe loop distinguishes the two contention cases:
  //  - Same-backend peer: it publishes OUR order within its brief exclusive
  //    write (sub-ms). A re-read under SH then matches and we run aclInit
  //    concurrently with it -- no block on its aclInit.
  //  - Different-backend peer: it holds SH for its aclInit with the file in
  //    ITS order, which never matches ours across the probe window. We must
  //    NOT aclInit on that order -- correctness first -- so we block-wait for
  //    the exclusive lock (bounded by the peer's aclInit), publish our order,
  //    and re-verify under SH IN THE SAME ROUND. If a peer re-publishes its
  //    order in the EX->SH window, the re-verify mismatches and we retry.
  // A publish failure is classified (review #29): if the active order could
  // not be read (bad/unreadable) it is kIdle (non-fatal, proceed -- we cannot
  // know it is wrong); if the order was read validly and differs from ours
  // but the publish failed (e.g. EACCES), proceeding would aclInit on a
  // confirmed different-backend order, so it is kUnsettled (caller fail-stops)
  // -- never aclInit on a wrong order. Persistent cross-backend contention (4
  // rounds each ending in a mismatched re-verify) also sets state_ =
  // kUnsettled.
  void acquire_with_retry() {
    // Each round covers at most one peer-aclInit block-wait plus an in-round
    // SH re-verify; a handful of rounds bounds persistent cross-backend
    // ping-pong before kUnsettled.
    for (int32_t round = 0; round < 4; ++round) {
      // Phase 1: non-blocking probe window (~200ms). A same-backend peer
      // publishes our order (brief write) so we match on SH and aclInit
      // concurrently; or we grab EX to publish when the file is free.
      PublishResult pr = PublishResult::kMatched;
      bool did_publish = false;
      for (int32_t probe = 0; probe < 200; ++probe) {
        const CheckResult c = check_match_under_shared();
        if (c == CheckResult::kCheckSettled) {
          return;  // hold SH through aclInit
        }
        if (c == CheckResult::kCheckAbort) {
          return;  // state_ stays kIdle (non-fatal)
        }
        // kCheckMismatch: SH released by the helper. Try a NON-blocking
        // exclusive lock so a same-backend peer mid-aclInit (holding SH) does
        // not make us block on its aclInit.
        if (::flock(fd_, LOCK_EX | LOCK_NB) == 0) {
          pr = publish_under_exclusive();
          ::flock(fd_, LOCK_UN);
          did_publish = true;
          break;  // published -> phase 3/4 below
        }
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
          LOG(WARNING) << "flock LOCK_EX|NB failed (" << std::strerror(errno)
                       << ")";
          return;  // state_ stays kIdle (non-fatal)
        }
        // EX unavailable: a peer holds the lock. Back off briefly; a
        // same-backend peer will publish our order within its brief write.
        ::usleep(1000);
      }
      // Phase 2: NB-probe exhausted without publishing -> a different-backend
      // peer holds SH for its aclInit. Block-wait for EX, then publish.
      if (!did_publish) {
        if (::flock(fd_, LOCK_EX) != 0) {
          LOG(WARNING) << "flock LOCK_EX failed (" << std::strerror(errno)
                       << ")";
          return;  // state_ stays kIdle (non-fatal)
        }
        pr = publish_under_exclusive();
        ::flock(fd_, LOCK_UN);
      }
      // Phase 3: classify the publish outcome (review #29). Two failure modes
      // were split in publish_under_exclusive:
      //  - kPublishFailedUnreadable: the active order could not be determined
      //    (bad read / exists unreadable). Non-fatal: proceed with the
      //    existing config (state_ kIdle). We cannot know it is wrong.
      //  - kPublishFailedStale: the active order was read validly, differs
      //    from desired, and we could not rewrite it (e.g. EACCES). Proceeding
      //    would aclInit on a confirmed different-backend order, violating
      //    "never aclInit on a non-our order" -- fail-stop (kUnsettled).
      if (pr == PublishResult::kPublishFailedUnreadable) {
        LOG(WARNING) << "vendor config unreadable; config.ini left as-is, "
                     << "proceeding without a lock";
        return;  // state_ stays kIdle (non-fatal)
      }
      if (pr == PublishResult::kPublishFailedStale) {
        LOG(ERROR) << "config.ini is at a different-backend order and we "
                      "cannot rewrite it (permission/IO); refusing to aclInit "
                      "on a wrong order";
        state_ = State::kUnsettled;
        return;  // caller fail-stops
      }
      // Phase 4: re-verify under SH IN THIS ROUND (not deferred to the next).
      // This verifies the last round's publish too: if our publish held, this
      // matches and we settle; if a peer re-published in the EX->SH window,
      // this mismatches and we retry.
      const CheckResult c = check_match_under_shared();
      if (c == CheckResult::kCheckSettled) {
        return;  // hold SH through aclInit
      }
      if (c == CheckResult::kCheckAbort) {
        return;  // state_ stays kIdle (non-fatal)
      }
      // kCheckMismatch: peer re-published. Retry the whole round (bounded).
    }
    // 4 full rounds, each ending in a mismatched SH re-verify: persistent
    // cross-backend contention. Do NOT aclInit on the peer's order --
    // correctness first. Signal the caller to fail-stop; aclInit must not run.
    LOG(ERROR) << "could not settle vendor config order to " << backend_label_
               << " backend under cross-backend contention; refusing to "
                  "aclInit on a different-backend order";
    state_ = State::kUnsettled;
  }
};
}  // namespace
#endif

// Flush logs/stdio and hard-exit without running global/static destructors.
//
// This binary embeds CPython (pybind11) and intentionally never finalizes it.
// Returning from main (or calling exit()) would destroy global pybind11 type
// objects during static teardown without holding the GIL, aborting the process
// with "pybind11::handle::dec_ref() ... PyGILState_Check() failure" whenever a
// server is stopped (observed on every NPU DeepSeek-V4 eager/MTP run once the
// HTTP server finishes serving). Bypassing atexit/static teardown lets the OS
// reclaim the interpreter; glog/stdio are flushed explicitly first so no log
// records are lost.
[[noreturn]] void exit_without_python_teardown(int exit_code) {
  google::FlushLogFiles(google::GLOG_INFO);
  fflush(nullptr);
  std::_Exit(exit_code);
}

// Signal handlers may only call async-signal-safe functions. In particular,
// neither logging nor flushing is safe here, so exit without any teardown.
[[noreturn]] void exit_immediately_without_python_teardown(int exit_code) {
  std::_Exit(exit_code);
}

void shutdown_handler(int signal) {
  (void)signal;
  exit_immediately_without_python_teardown(1);
}

void validate_config(const std::string& model_type) {
  ModelConfig& model_config = ModelConfig::get_instance();
  LoadConfig& load_config = LoadConfig::get_instance();
  KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  KVCacheStoreConfig& kv_cache_store_config =
      KVCacheStoreConfig::get_instance();
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  DisaggPDConfig& disagg_pd_config = DisaggPDConfig::get_instance();
  SpeculativeConfig& speculative_config = SpeculativeConfig::get_instance();
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();

  ModelArgs model_args;
  if (!model_type.empty()) {
    JsonReader model_config_json;
    const std::filesystem::path model_config_path =
        std::filesystem::path(model_config.model()) / "config.json";
    CHECK(model_config_json.parse(model_config_path.string()))
        << "Failed to parse model config: " << model_config_path;

    std::string resolved_model_type;
    std::string error_message;
    CHECK(resolve_model_registration_name(
        model_type, &resolved_model_type, &error_message))
        << error_message;
    const auto model_args_loader =
        ModelRegistry::get_model_args_loader(resolved_model_type);
    CHECK(model_args_loader != nullptr)
        << "Failed to find model args loader for model type "
        << resolved_model_type;
    CHECK(model_args_loader(model_config_json, &model_args))
        << "Failed to load model args for model type " << resolved_model_type;
  }

  if (kv_cache_store_config.enable_kvcache_store()) {
    CHECK(kv_cache_config.enable_prefix_cache())
        << "KV cache Store requires --enable_prefix_cache=true.";
    CHECK_GT(kv_cache_store_config.host_blocks_factor(), 1.0)
        << "KV cache Store requires --host_blocks_factor > 1.";
  }

  if (model_config.backend().empty()) {
    LOG(FATAL) << "Model is not supported currently, model type: "
               << model_type;
  }
  if (model_config.max_encoder_cache_size() < 0) {
    LOG(FATAL) << "max_encoder_cache_size must be >= 0.";
  }
  if (model_config.max_processor_cache_items() < 0) {
    LOG(FATAL) << "max_processor_cache_items must be >= 0.";
  }
  if (model_config.image_max_tokens_num() < 0) {
    LOG(FATAL) << "image_max_tokens_num must be >= 0.";
  }
  if (model_config.fps_min_frames() < 0) {
    LOG(FATAL) << "fps_min_frames must be >= 0.";
  }
  if (model_config.fps_max_frames() < 0) {
    LOG(FATAL) << "fps_max_frames must be >= 0.";
  }
  if (model_config.fps_max_frames() < model_config.fps_min_frames()) {
    LOG(FATAL) << "fps_max_frames must be greater than or equal to "
                  "fps_min_frames.";
  }
  if (model_config.video_max_token_num() == 0 ||
      model_config.video_max_token_num() < -1) {
    LOG(FATAL) << "video_max_token_num must be -1 or greater than 0.";
  }
#if defined(USE_MLU)
  // Disable enable_schedule_overlap for VLM models on MLU backend
  if (scheduler_config.enable_schedule_overlap() &&
      model_config.backend() == "vlm") {
    LOG(WARNING) << "enable_schedule_overlap is not supported for VLM models "
                    "on MLU backend. "
                 << "Disabling enable_schedule_overlap.";
    scheduler_config.enable_schedule_overlap(false);
  }
  // TODO: support other block sizes in the future
  if (kv_cache_config.block_size() != 16 && kv_cache_config.block_size() != 1 &&
      model_config.backend() != "dit") {
    LOG(FATAL) << "Currently, block_size must be 16 for MLU backend, we will "
                  "support other block sizes in the future.";
  }
  if (disagg_pd_config.enable_disagg_pd()) {
    if (model_config.backend() != "llm") {
      LOG(FATAL) << "MLU disaggregated PD only supports backend=llm.";
    }
    disagg_pd_config.normalize_mlu(kv_cache_config, scheduler_config);
  }
#endif

#if defined(USE_DCU)
  if (disagg_pd_config.enable_disagg_pd()) {
    if (scheduler_config.enable_schedule_overlap()) {
      LOG(WARNING) << "enable_schedule_overlap is not supported for "
                      "disaggregated PD on DCU backend. "
                   << "Disabling enable_schedule_overlap.";
      scheduler_config.enable_schedule_overlap(false);
    }
    if (model_config.backend() != "llm") {
      LOG(FATAL) << "DCU disaggregated PD only supports backend=llm.";
    }
    disagg_pd_config.normalize_dcu(scheduler_config);
  }
#endif

#if defined(USE_NPU)
  if (speculative_config.num_speculative_tokens() > 0 &&
      execution_config.enable_graph_double_buffer()) {
    LOG(WARNING) << "enable_graph_double_buffer is not compatible with "
                    "speculative decoding. "
                 << "Disabling enable_graph_double_buffer.";
    execution_config.enable_graph_double_buffer(false);
  }
  // enable_xtensor / enable_rolling_load imply enable_manual_loader
  if ((kv_cache_config.enable_xtensor() || load_config.enable_rolling_load()) &&
      !load_config.enable_manual_loader()) {
    LOG(WARNING) << "enable_xtensor or enable_rolling_load requires "
                    "enable_manual_loader; forcing enable_manual_loader=true.";
    load_config.enable_manual_loader(true);
  }
  if (load_config.enable_rolling_load() &&
      load_config.rolling_load_num_cached_layers() < 1) {
    LOG(FATAL) << "rolling_load_num_cached_layers must be >= 1.";
  }
  if (load_config.enable_rolling_load() &&
      load_config.rolling_load_num_rolling_slots() < -1) {
    LOG(FATAL) << "rolling_load_num_rolling_slots must be >= -1.";
  }
  if (load_config.enable_rolling_load() &&
      load_config.rolling_load_num_rolling_slots() >= 0 &&
      load_config.rolling_load_num_rolling_slots() >
          load_config.rolling_load_num_cached_layers()) {
    LOG(FATAL) << "rolling_load_num_rolling_slots must be <= "
               << "rolling_load_num_cached_layers.";
  }
#else
  if (kv_cache_config.enable_xtensor()) {
    LOG(FATAL) << "enable_xtensor is only supported on NPU.";
  }
  if (load_config.enable_manual_loader()) {
    LOG(FATAL) << "enable_manual_loader is only supported on NPU.";
  }
  if (load_config.enable_rolling_load()) {
    LOG(FATAL) << "enable_rolling_load is only supported on NPU.";
  }
#endif

  model_config.normalize_cpp_chat_template(model_type);
}

int run() {
  ModelConfig& model_config = ModelConfig::get_instance();
  KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  BeamSearchConfig& beam_search_config = BeamSearchConfig::get_instance();
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  DistributedConfig& distributed_config = DistributedConfig::get_instance();
  ServiceConfig& service_config = ServiceConfig::get_instance();
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();

  // check if model path exists
  if (!std::filesystem::exists(model_config.model())) {
    LOG(FATAL) << "Model path " << model_config.model() << " does not exist.";
  }

  std::filesystem::path model_path =
      std::filesystem::path(model_config.model()).lexically_normal();
  const std::string default_model_name = xllm::util::get_model_name(model_path);
  const std::string model_repository_name =
      xllm::util::get_model_repository_name(model_path);

  if (model_config.model_id().empty()) {
    // use last part of the path as model id
    model_config.model_id(default_model_name);
  }

  if (model_config.backend().empty()) {
    model_config.backend(xllm::util::get_model_backend(model_path));
  }

  const std::string local_ip = net::get_local_ip_addr();
  if (service_config.host().empty()) {
    // set the host to the local IP when the host is empty
    service_config.host(local_ip);
  }

  const std::string master_ip =
      net::extract_ip(distributed_config.master_node_addr());
  const bool is_local = !service_config.host().empty() &&
                        master_ip == net::extract_ip(service_config.host());

  LOG(INFO) << "set worker role to "
            << (is_local ? "local worker" : "remote worker");

  // if max_tokens_per_chunk_for_prefill is not set, set its value to
  // max_tokens_per_batch
  if (scheduler_config.max_tokens_per_chunk_for_prefill() < 0) {
    scheduler_config.max_tokens_per_chunk_for_prefill(
        scheduler_config.max_tokens_per_batch());
  }

// disable block copy kernel on unsupported backends
#if !defined(USE_NPU) && !defined(USE_CUDA) && !defined(USE_MUSA)
  beam_search_config.enable_block_copy_kernel(false);
#endif
  std::string model_type = "";
  if (model_config.backend() != "dit") {
    model_type = xllm::util::get_model_type(model_path, model_config.backend());
    model_config.tool_call_parser(
        function_call::FunctionCallParser::get_parser_auto(
            model_config.tool_call_parser(), model_type));
    model_config.reasoning_parser(ReasoningParser::get_parser_auto(
        model_config.reasoning_parser(), model_type));
  }

  // validate config before creating master
  validate_config(model_type);

  if (distributed_config.node_rank() == 0 &&
      execution_config.random_seed() < 0) {
    execution_config.random_seed(std::random_device{}() % (1 << 30));
  }

  if (distributed_config.node_rank() == 0) {
    config::dump_startup_config();
  }

  // Create Master
  Options options = create_options(
      service_config.host() + ":" + std::to_string(service_config.port()),
      is_local);

  InstanceName::name()->set_name(options.instance_name().value_or(""));

  // master node
  // init XTensor allocator and PhyPagePool for xtensor mode
  if (kv_cache_config.enable_xtensor()) {
    // Parse devices
    const auto devices = DeviceNameUtils::parse_devices("auto");

    // Initialize XTensorAllocator with first device
    auto& allocator = XTensorAllocator::get_instance();
    allocator.init(devices[0]);

    // Setup distributed XTensor service for multi-GPU/multi-node
    if (distributed_config.nnodes() > 1) {
      xtensor::Options xtensor_options;
      xtensor_options.devices(devices)
          .nnodes(distributed_config.nnodes())
          .node_rank(distributed_config.node_rank());
      allocator.setup_multi_node_xtensor_dist(
          xtensor_options,
          distributed_config.xtensor_master_node_addr(),
          parallel_config.dp_size());
    }

    // Initialize PhyPagePool on all workers
    int64_t num_pages =
        allocator.init_phy_page_pools(kv_cache_config.max_memory_utilization(),
                                      kv_cache_config.max_cache_size());
    if (num_pages <= 0) {
      LOG(FATAL) << "Failed to initialize PhyPagePool";
    }
    LOG(INFO) << "XTensor initialized with " << num_pages << " physical pages";
  }

  std::unique_ptr<Master> master =
      create_master(model_config.backend(), options);
  master->run();

  // supported models
  std::vector<std::string> model_names = {model_config.model_id()};
  std::vector<std::string> model_repository_names = {model_repository_name};
  std::string model_version = default_model_name;
  std::vector<std::string> model_versions = {model_version};

  if (distributed_config.node_rank() == 0 || kv_cache_config.enable_xtensor()) {
    auto api_service = std::make_unique<APIService>(
        master.get(), model_names, model_repository_names, model_versions);
    auto xllm_server =
        ServerRegistry::get_instance().register_server("HttpServer");

    // start brpc server
    if (!xllm_server->start(std::move(api_service))) {
      LOG(ERROR) << "Failed to start brpc server on port "
                 << service_config.port();
      exit_without_python_teardown(1);
    }
  } else {
    // No HTTP server on this rank. Stay alive until SIGINT/SIGTERM so
    // the destructor can stop the idle thread instead of hanging on it.
    master->wait();
  }

  // Join serving threads through the master destructor, then hard-exit instead
  // of returning to main: normal exit would run global/static destructors that
  // release pybind11 type objects without the GIL (see
  // exit_without_python_teardown).
  master.reset();
  exit_without_python_teardown(0);
}

int main(int argc, char** argv) {
  // Check for --help flag before parsing other flags
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      HelpFormatter::print_help();
      return 0;
    }
  }

  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 0;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging("xllm");
  initialize_configs();

  const ServiceConfig& service_config = ServiceConfig::get_instance();
  const DistributedConfig& distributed_config =
      DistributedConfig::get_instance();
  const std::string verbose_trace_log_path =
      resolve_verbose_trace_log_path(service_config.verbose_trace_log_path(),
                                     distributed_config.nnodes(),
                                     distributed_config.node_rank());
  VerboseTraceLogger::get_instance().initialize(
      service_config.enable_verbose_trace_log(),
      verbose_trace_log_path,
      service_config.verbose_trace_log_max_size_mb(),
      service_config.verbose_trace_log_max_files());

  // Check if model path is provided
  if (::xllm::ModelConfig::get_instance().model().empty()) {
    HelpFormatter::print_error("--model flag is required");
    return 1;
  }

#if defined(USE_NPU)
  // Hold the vendor config lock across aclInit (inside init_npu_python_runtime)
  // so a different-backend process cannot overwrite config.ini between our
  // write and CANN's read (review P2 cross-backend race). If the lock is
  // usable but the order could not be settled to this backend under persistent
  // cross-backend contention, fail-stop rather than aclInit on a different-
  // backend order (which would mis-select aclnnSparseFlashAttention's vendor).
  // This is distinct from the non-fatal "no lock / no write permission" case,
  // which proceeds with the existing config.
  {
    VendorConfigLock vendor_lock;
    if (vendor_lock.unsettled()) {
      LOG(ERROR) << "refusing to start: vendor config order unsettled under "
                    "cross-backend contention; ensure only one backend is "
                    "launched concurrently on this host";
      return 1;
    }
    init_npu_python_runtime();
  }
#endif

  return run();
}
