# Test execution

Run build and test commands in the container matching the target backend.

## Focused and full runs

```bash
python setup.py --device npu --test-name partial_json_parser_test test
python setup.py --device npu build test
```

`--test-name` takes a CMake executable target, not a gtest case name. It builds
the target and runs its executable. For case-level debugging, use the freshly
built executable with `--gtest_filter`. Focused runs do not replace the required
full regression suite.

Existing CTest concurrency, serial test rules and retry behavior remain
unchanged.

## Torch extension caches

Unless `TORCH_EXTENSIONS_DIR` is explicitly set to a nonempty path, each full
or single-target test invocation creates a unique
`test-torch-extensions-*` directory inside its CMake build directory. All CTest
phases and retries within that invocation share the directory. A later run
does not reuse it, so a lock left by an interrupted JIT build cannot block the
next run.

The runner logs the cache path and retains it for diagnostics. Remove old run
directories only after confirming that their processes have exited. It never
deletes locks or modifies a caller-provided shared cache. An explicit cache
override preserves reuse across runs, but its owner is responsible for stale
locks, compatibility and concurrent access. A fresh default cache can require
one extension build per invocation; the extension is shared within that run.

## CPU-only C++ tests

Use the `CPU_ONLY` option of `cc_test` only after auditing all cases and their
dependencies for device runtime use. Such targets do not receive the shared
NPU/Python bootstrap or its additional runtime libraries, even in an NPU build.
Their existing gtest case names and discovery granularity are unchanged, and
they receive the CTest label `cpu`. Other targets retain the NPU environment.

The partial JSON parser, tool-call detectors, chat templates, tokenizer and
Qwen3.5 GatedDeltaNet prefill-indices tests use this option. The indices cases
use CPU tensors and Meta-device validation; their existing `RUN_SERIAL` rule
remains unchanged. It does not remove transitive library dependencies;
some targets still link Torch through their production libraries, but do not
initialize Python or an NPU context through the shared test environment.
Mixed CPU/NPU executables must be split or remain on the default path; this
option is not a mock NPU backend.

The Qwen3.5 prefill-indices target links the same small indices library used by
the production layer, rather than the complete NPU layer library. Its explicit
dependencies are Torch and glog; CPU and Meta-device cases do not need the NPU
operator stack. This also avoids repeatedly loading that stack in thread-safe
death-test subprocesses. The death-test style, assertions and serial execution
rule remain unchanged.

Audited API protocol/JSON parsing, request identity/admission, model listing,
configuration JSON, KV reshard planning and topology validation targets also
use `CPU_ONLY`. The JSON-object grammar target exercises CPU masks only;
device-side sampler and KV transfer tests retain their NPU bootstrap. Classify
the operations exercised by the entire target, not its directory name.

The block-pool and MTP prefix-cache targets exercise host metadata and stub
worker transfers, without allocating accelerator KV storage. Request-parameter
parsing, CPU media decoding, KV capacity estimation and CPU-configured CP
planning also avoid the shared NPU bootstrap. In contrast, `batch_test` retains
it: pinned host-memory allocation depends on the device backend even when the
resulting tensor is on CPU. CPU tensor placement alone is not sufficient to
classify a target as `CPU_ONLY`.

The host utility, Hugging Face model configuration, prefix-cache metadata,
expert-layout policy and DiT mask targets also use `CPU_ONLY`. The graph-warmup
profile tests exercise planning through a recording engine, not actual device
execution. Device-name parsing in the utility target uses explicit indices,
not automatic device discovery. These classifications apply to the current
cases: adding device discovery, pinned allocation, model execution or backend
calls requires reevaluating the whole target's bootstrap requirements.

The common host utilities, rank topology, communicator-selection policy,
CPU-configured DP/EP padding and model CP-capability checks use `CPU_ONLY` too.
The cache-layout builder, adaptive speculative controller, speculative profile
registry and MTP JSON-object state cases only exercise host state or CPU tensors.
EPLB options, rank utilities, CPU load aggregation, shared-memory buffers and
manager tests use CPU data and a recording policy, without a device executor.
In contrast, scheduler and speculative input/state builders retain bootstrap
for pinned allocations, and the EPLB executor tests create real device streams.
Neither a CPU tensor nor a mock model proves that the whole target is CPU-only.

Sequence stop-output and recommendation request-factory tests use host request
state and fake tokenizers, not model execution. Linear-state restore tests use
preallocated CPU cache tensors; worker JSON-object overlap tests exercise row
mapping and CPU placeholder sanitization. These targets also use `CPU_ONLY`.
NPU linear-state lifecycle, worker cache transfer, MTP host offload and ACL graph
executor tests still need device initialization and retain the default path.

Host-only targets also cover concurrent block ownership, sample-slot output,
stop rules, sequence token/cache bookkeeping, CPU MRoPE positions and bootstrap
embedding storage. Vocabulary constraint tables and CPU expert-weight loading
need no device runtime. Mapping JSON and ACL graph bucket policies exercise
rank arithmetic and capacity decisions, not communication or graph execution.
An explicitly indexed device descriptor alone does not initialize hardware.

Sample-request validation/response serialization and multimodal payload helpers
use host data and fake tokenizers. Worker-launch argument parsing, TP/DP route
arithmetic and transfer-completion futures also need no device bootstrap.
The disabled Anthropic/OpenAI integration clients use HTTP, not a local NPU;
their existing disabled status and requirement for a running server are unchanged.
Worker-service metrics tests still construct real device contexts and retain
bootstrap, even though some other cases in the same target use only CPU tensors.

LLM/VLM request-factory tests use fake tokenizers, templates and multimodal
processors. Encoder and speculative embedding-cache tests store CPU tensors.
DiT media-source collections, CPU media output encoding and single-request
batch views also use `CPU_ONLY`; they do not execute a diffusion model. This
does not apply to DiT tensor-source transfer tests, which include an NPU copy,
or to LLM batch builders that allocate pinned memory.

The CausalLM logits wrapper uses a fake CPU model; model-registration tests
only resolve backend names. Host KV transfer validation uses recording
strategies and fake synchronizers. Store-index tests use CPU caches without
opening a remote Store or registering device memory. These targets use
`CPU_ONLY`, unlike the real basic/hierarchical transfer tests. The Python
CausalLM bridge retains the default bootstrap pending interpreter isolation.

Host layer helpers also use `CPU_ONLY`: compressed-weight validation, DSA
sharing policy, CP split/index planning, CPU rotary embeddings and grouped
convolution. DeepSeek-V4 EPLB helper tests use CPU tensors for masks, slot
staging, scale conversion and load accounting; their P2P cases only construct
transfer plans. This does not apply to device indexer, quantized linear or
kernel-wrapper tests. Linking a library that also implements NPU kernels does
not make these host-only paths require the shared NPU/Python bootstrap.

## Prefix-cache teardown

Prefix caches have no background worker to stop. Their virtual destructor uses
normal member cleanup to release cached block references, without a fixed sleep.
This matters for tests that repeatedly construct multiple device/host cache
groups: an unconditional delay per cache compounds even without any NPU work.
The existing prefix-cache tests also verify that destruction releases the cache's
block references. Timing comparisons should use the same tests and scheduling
settings; avoid wall-clock assertions in shared-machine correctness tests.

## Runner regression tests

```bash
python -m pytest -q tests/scripts/test_test_runtime.py tests/scripts/test_cc_test.py
```
