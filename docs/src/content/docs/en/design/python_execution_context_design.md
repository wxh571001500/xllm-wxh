---
title: "Python Model Execution Metadata Design"
sidebar:
  order: 5
---
<!--
Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## 1. Background

Python models often need execution inputs beyond their regular `forward` arguments, for example:

- dynamic tensors supplied by the scheduler, runtime, or shared metadata;
- external inputs whose contents change before ACL Graph replay while their addresses must remain stable;
- execution information that a model cannot derive from its parameters or normal inputs.

Handling these requirements directly in model layers or runners usually creates two forms of coupling.

First, the model becomes aware of the execution mode. It starts checking whether the current execution is eager or
Graph Mode and chooses temporary or persistent allocation accordingly. The same mathematical computation then gains
multiple control paths, coupling model correctness to executor-side address lifetimes.

Second, the runner becomes aware of feature semantics. Eager and graph runners gradually accumulate model-specific,
operator-specific, and feature-specific input preparation branches.

Several existing models follow exactly this coupled pattern: the runner owns model-specific static tensors or graph
buffers, while the model reads low-level carriers such as public `AttentionMetadata` and
`ForwardContext.layer_caches`. Individual layers then transform state indices, apply padding, build plans, and
interpret execution phases. The dependency chain looks roughly like this:

```text
the runner knows which fixed addresses and special fields a model needs
        ↓
public metadata carries both attention data and unrelated module data
        ↓
the model reads public metadata and interprets eager/Graph, padding, and dummy details
        ↓
each layer prepares the exact operator inputs it needs
```

This approach is direct for a single feature, but the model, runner, and public metadata all grow together. Even when
another framework already has an equivalent mathematical model graph, porting it still requires understanding
xLLM-specific graph lifetimes and historical field conventions.

The execution metadata design introduces a boundary between these responsibilities:

```text
the runner manages execution timing and lifetimes
        ↓
InputBatch carries authoritative upstream execution semantics
        ↓
Execution Metadata Builders prepare model-visible inputs
        ↓
ForwardContext transports typed metadata
        ↓
the model only consumes inputs and performs computation
```

The execution context is not a general-purpose scratch allocator or a registry for graph intermediates. A stable
address requirement alone does not justify putting a tensor in `ForwardContext`. Values produced and consumed
entirely inside the model graph should remain ordinary graph intermediates owned by the model and graph runtime.

## 2. What This PR Changes

This PR does not add another set of Qwen3.5 branches to the runners. It first establishes a general model execution
metadata boundary and then uses Qwen3.5 MegaMoe and MegaGDN to validate that boundary.

| Before | After |
| --- | --- |
| Runners directly own model- or feature-specific tensors | Runners only invoke metadata builders at common lifecycle points |
| Models read and interpret public metadata directly | Builders convert public inputs into module-level typed metadata |
| Every layer repeatedly transforms state indices and builds operator plans | A builder prepares and validates operator-ready inputs before model execution |
| Models select preparation paths based on eager, Graph, padding, or dummy state | Models always consume the same typed metadata contract |
| In-graph intermediates may be promoted to runner-owned persistent resources | In-graph intermediates remain owned by the model graph and graph runtime |
| A feature requires coordinated changes to models, runners, and public metadata | The model registry declares builders without adding model branches to generic runners |

### 2.1 Stable Step-Level Input Semantics

The upstream pipeline now propagates request-scoped batch metadata and materializes it as an `InputBatch` in Python.
The current stable fields include the request count, execution token count, scheduled and computed tokens per
request, query boundaries, and prefill state. The upstream producer that owns each semantic value supplies it
explicitly. Python no longer infers request counts from block-table rows, state-index counts, or tensor shapes.

An `InputBatch` is constructed only when the current model actually registers a builder. Models that have not been
migrated therefore do not gain new validation, object construction, or runtime dependencies. A graph runner only
adds fixed-address inputs, the after-padding capacity, and a padding mask for its graph entry; it does not change the
real request semantics in the batch.

### 2.2 Common Builder Lifecycle

This PR introduces `ExecutionMetadataBuilder` as the module-level adaptation boundary between materialized runtime
inputs and the model graph. The protocol has three lifecycle methods:

- `build()` constructs metadata for an eager forward;
- `allocate_persistent()` allocates stable addresses when a graph entry is created;
- `update_persistent()` updates those addresses in place before replay.

The registry declares which builders a model requires. The executor instantiates and binds them, and runners only
iterate over them at the correct lifecycle point. The generic runner does not know which tensors a module needs, how
they are validated, or which addresses must persist.

### 2.3 Qwen3.5 as the Reference Migration

This PR migrates two module families:

- MegaMoe: the model only reads `MegaMoeMetadata.active_token_mask`. Router top-k values, padded operator inputs,
  and outputs remain in-graph intermediates instead of becoming per-layer external resources.
- MegaGDN: a builder converts public metadata, `InputBatch`, and layer caches into `GdnPrefillMetadata` or
  `GdnDecodeMetadata`. The model no longer reads public `AttentionMetadata` or rebuilds the prefill plan in every
  layer.

The Qwen3.5 `forward` signature is also moved closer to the vLLM model contract. The current implementation still
accepts only `input_ids`; pipeline parallel execution and `inputs_embeds` are not added by this PR.

### 2.4 Explicit Non-Goals

This PR does not claim that the entire Python execution architecture has already been migrated. The following areas
remain unchanged:

- Full Attention continues to use the existing `AttentionBackend` and public `AttentionMetadata`.
- The current `InputBatch` and Qwen3.5 GDN builder do not support speculative execution.
- Other models, including DeepSeek-V4, do not switch to the new MegaMoe builder.
- The original `mega_moe_token_mask` compatibility path remains in the ACL Graph runner.
- The vLLM Ascend operator registry, compilation system, and full runner class hierarchy are not migrated.

The success criterion for this PR is therefore not that every model is already unified. It is that the new
model-visible contract is clear, generic runners gain no Qwen3.5-specific branches, and models that have not been
migrated preserve their existing behavior.

## 3. Core Design Principles

### 3.1 Keep the Context Minimal

The first distinction is between authoritative inputs and derived inputs. If the scheduler, BatchBuilder, or a
speculative worker already owns the precise semantics of a field, that producer must write the value explicitly and
propagate it through the execution pipeline. Python must not reconstruct it from tensor shapes, attention row counts,
or other indirect signals.

Only a purely derived value with no independent business semantics should be computed downstream. For example,
`has_prefill` may be derived from authoritative per-request `is_prefilling` values, but request count must not be
inferred from attention rows. Redundant representations must be checked for consistency and fail closed on conflict.

A context should contain only information that the model cannot derive from its existing inputs and that must be
provided by the execution layer. Before adding a field, check the following in order:

1. If it is a model parameter or normal `forward` input, use the existing interface.
2. If it can be produced in-graph from existing tensors, keep it as an ordinary intermediate.
3. If an appropriate domain carrier already exists, such as `AttentionMetadata` or `LayerCache`, do not duplicate it.
4. Introduce execution metadata only when the value originates outside `forward` and the model genuinely needs it.
   A fixed Graph address is an implementation choice, not a reason to change the model-visible contract.

A useful test is:

> If this context field is removed, can the model still compute the same result from its existing inputs?

If the answer is yes, the field usually does not belong in the context. Expanding the contract only to reuse memory
or avoid temporary allocations turns a performance choice into architecture. Such optimizations should be introduced
separately after profiling proves they are necessary.

### 3.2 Models Know Computation, Not Address Origins

From the model's perspective, an upstream tensor is simply a tensor it can consume. Whether that tensor was allocated
for one eager call or is held at a stable address by a graph entry must not change the mathematical path.

Model layers therefore must not:

- query whether execution is currently in Graph Mode;
- call graph buffer management APIs;
- select different mathematical paths for eager, warmup, capture, or replay;
- infer whether a tensor should be reused across steps or layers.

The model consumes declared inputs, executes operators, and returns results.

### 3.3 Producers Own External Input Lifetimes

The component that creates a tensor must decide:

- its shape, dtype, and device;
- how long its address remains valid;
- when its contents are updated;
- whether it can be shared across layers or steps;
- whether it is shared read-only data or exclusive writable data;
- whether an unsupported configuration is rejected before execution.

For external execution inputs requiring fixed addresses, the Execution Metadata Builder is the sole owner of these
decisions. The model consumes builder output without reverse-engineering address lifetimes. Top-k results, temporary
outputs, and other values produced inside the model remain owned by the model and graph runtime.

### 3.4 Runners Manage Timing, Not Feature Semantics

A runner knows when a graph entry is created and when replay inputs must be prepared. It should decide when to invoke
the lifecycle without understanding the tensors required by a particular model feature.

The runner therefore invokes only the common methods:

- `build()` before an eager forward;
- `allocate_persistent()` when creating a graph entry;
- `update_persistent()` before replay.

Builders decide what to construct, allocate, update, validate, or share. The eager runner builds per-forward metadata,
while the graph runner manages persistent metadata lifetimes.

### 3.5 ForwardContext Only Transports Data

`ForwardContext` is the runtime carrier for one model forward. It stores typed execution metadata in
`execution_contexts: dict[type[object], object]`.

It does not construct or interpret that metadata. A model retrieves the object by type:

```python
metadata = get_execution_context(MyFeatureMetadata)
```

Using the metadata type as the key avoids adding a new feature-specific field to `ForwardContext` for every feature
and allows duplicate registration or type errors to fail early.

## 4. Common Abstraction

`xllm/python/model_executor/execution_context.py` defines the protocol:

```python
class ExecutionMetadataBuilder(Protocol):
    metadata_type: type[object]

    def build(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> object: ...

    def allocate_persistent(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> object: ...

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None: ...
```

`Metadata` describes data semantics visible to the model; it does not promise a particular tensor lifetime. Eager
metadata may reference tensors owned by the current forward, while Graph metadata references stable addresses owned
by a graph entry.

### 4.1 `build()`

The eager runner invokes `build()` for each forward, converting `InputBatch` and materialized public metadata into
typed metadata that the model can consume directly. It may reference upstream tensors or create temporary tensors
sized for the current execution.

The model must not choose an address preparation path by calling `in_acl_graph()` or by checking whether a context is
present. Eager and Graph execution must expose the same metadata type through the same typed-context API.

### 4.2 `allocate_persistent()`

This method runs once when a graph entry is created and allocates stable addresses for external inputs carried by the
context. Capture and all later replays use the same addresses. Intermediates produced inside the model continue to be
managed by the graph runtime.

A builder may use `input_batch` and public metadata to determine capacity, shape, dtype, and device. Dynamic contents
must still be written by `update_persistent()` before each execution. Allocation should establish long-lived storage
for the graph entry and should not embed values that are known only for a particular step.

### 4.3 `update_persistent()`

This method refreshes dynamic contents before replay. It must update existing tensors in place and must not replace an
address captured by the graph. `InputBatch` supplies request-level execution semantics and Graph padding information,
while public `AttentionMetadata` supplies materialized tensors such as block tables and state indices. A builder
selects only the sources required by its module.

Not every tensor requires an update. A builder must distinguish between:

- values that remain constant across steps and are initialized once;
- values that change each step but are shared by all layers and updated once;
- values that differ by layer and require separate addresses managed by the builder.

## 5. End-to-End Execution Flow

### 5.1 Initialization

When registering a model implementation, the model registry also declares its execution metadata builder types. Each
builder uses model configuration to decide whether it should be instantiated for the current execution.

`ModelExecutor` obtains builder classes from the registry, invokes `from_config()`, and binds successful instances to
the eager and graph runners. The executor does not scan model modules or identify feature semantics. When no builder
is registered or enabled, the existing execution path remains unchanged.

### 5.2 Eager Execution

```text
request inputs and public metadata
  -> runner calls builder.build(input_batch, metadata)
  -> runner stores typed metadata in ForwardContext
  -> runner calls attention_backend.prepare()
  -> model reads typed metadata and executes forward
```

Eager execution does not allocate fixed addresses merely to mirror Graph lifetimes. If an operator requires identical
input shapes across ranks, the builder computes the capacity from authoritative step-level inputs and constructs
temporary metadata for that execution. The model only consumes the result.

### 5.3 Graph Mode

When a graph entry is created:

```text
select the graph bucket
  -> runner adds after-padding capacity and a padding mask to InputBatch
  -> runner calls builder.allocate_persistent(input_batch, metadata)
  -> builder allocates stable addresses
  -> warmup and capture use the resulting context
```

Before each replay:

```text
actual InputBatch for the current step
  -> combine its semantics with the graph entry's fixed-address inputs
  -> runner calls builder.update_persistent(persistent_metadata, input_batch, metadata)
  -> builder reads current values from InputBatch or public metadata
  -> builder updates persistent contents in place
  -> graph replays against the original addresses
```

The model sees the same context type and mathematical interface during warmup, capture, and replay. It does not need
to know which execution phase supplied the metadata.

## 6. How the Qwen3.5 Reference Migration Works

### 6.1 MegaMoe

The only dynamic MegaMoe input that originates outside the model and cannot be derived by the model is the active
token mask. The typed metadata therefore remains minimal:

```python
@dataclass(frozen=True, slots=True)
class MegaMoeMetadata:
    active_token_mask: torch.Tensor
```

In eager execution, the builder constructs a mask from execution token counts across DP ranks. In Graph execution,
it allocates a fixed-address mask for the graph bucket and updates it in place from the padding mask before each
replay. The model always reads `MegaMoeMetadata` and does not know whether the storage is temporary or persistent.

Each MoE layer computes its own gate, top-k results, padded operator input, and output. These values are derived from
that layer's hidden states and are consumed in the same graph, so they are not external execution inputs. They do not
belong in metadata and do not require `layer_id`-indexed buffers. Graph capture naturally records the intermediates
created by each layer.

The original generic MegaMoe mask in the ACL Graph runner remains for compatibility with models that have not been
migrated. Qwen3.5 consumes the builder-owned mask, so the old mask is temporarily an unused compatibility resource for
this model. It can be removed separately only after all consumers have migrated and been validated.

### 6.2 MegaGDN

MegaGDN differs from MegaMoe because it has Conv and SSM state that persists across steps, and its prefill and decode
operators consume different inputs. When caches are bound, the builder creates a `GdnStateCache` mapping keyed by
`layer_id`. Each GDN layer uses its own `layer_id` to retrieve the correct cache:

```text
GdnMetadata
  -> state_caches[layer_id]
       -> conv_state
       -> ssm_state
```

For prefill, the builder performs the following work once per forward:

- resolve read and write state indices;
- convert invalid initial-state reads to invalid slots;
- convert Conv slots to SSM checkpoint slots;
- construct `cu_seqlens` and `num_matrices`;
- validate the common capability boundary of prefill and decode before execution.

The result is one shared `GdnPrefillMetadata` object reused by all GDN layers in the forward. Each layer still selects
its own state cache through `layer_id`. Decode uses `GdnDecodeMetadata`; a graph entry owns fixed-address read and
write index tensors, and the builder updates their contents in place before replay.

It is valid for the model to distinguish prefill and decode typed metadata because they represent different operator
semantics. The model must not distinguish whether that metadata came from eager, warmup, capture, or replay.

## 7. Tensor Sources and Ownership

Determine where a value originates before deciding who owns its address.

| Tensor category | Examples | Recommended owner |
| --- | --- | --- |
| Model parameters | Weights, static communication context | Module or model configuration |
| Normal forward inputs | Tokens, positions, hidden states | Explicit `forward` arguments |
| In-graph intermediates | Router top-k results, temporary outputs | Model and graph runtime |
| Existing domain inputs | Attention block table, KV cache | Existing carriers such as `AttentionMetadata` and `LayerCache` |
| External dynamic execution inputs | Active-row mask, dynamic state indices | Execution context |
| Fixed-address Graph copies of those inputs | Static mask, static state indices | Metadata builder, updated in place |

A value that genuinely belongs in the context may use a different storage strategy in eager and Graph execution while
preserving the same model-visible tensor semantics:

```text
Eager: use an upstream tensor directly, or derive a temporary tensor for this execution
Graph: metadata references a fixed address updated by the builder before replay
Model: execute the same mathematical computation without interpreting address lifetime
```

For example, state-index values originate in upstream resource management and cannot be derived by the model. Graph
replay also requires their input addresses to remain stable, so a builder may allocate persistent storage in
`allocate_persistent()` and write current values in `update_persistent()`. Router top-k results and layer outputs, in
contrast, are produced by model computation and remain ordinary graph intermediates even if Graph capture ultimately
gives them stable addresses.

When a context genuinely needs multiple buffers, reuse decisions still need to consider semantic shape, dtype,
device, read/write behavior, and lifetime. Reuse is a second-stage performance optimization; it cannot replace the
initial decision about whether a field belongs in the context at all.

## 8. Module Responsibilities

| Module | Responsibilities | Non-responsibilities |
| --- | --- | --- |
| Model or layer | Read declared inputs and execute mathematical computation | Detect eager/Graph or manage persistent addresses |
| Execution metadata builder | Build eager metadata; allocate, reuse, update, and validate Graph addresses | Execute model mathematics |
| Graph runner | Invoke all builders at the correct lifecycle points | Interpret model- or feature-specific tensor semantics |
| Eager runner | Invoke builders to construct typed metadata for the current forward | Interpret model- or feature-specific tensor semantics |
| `ForwardContext` | Carry typed metadata for the current forward | Construct metadata or decide reuse policy |
| Scheduler and batch construction | Describe what executes in the current step | Know the model's internal buffer layout |

The boundary can be summarized as:

```text
scheduling decides "what executes in this step"
the runner decides "when preparation and execution happen"
the metadata builder decides "what metadata the model reads and how Graph addresses are maintained"
the model decides "what computation to perform with those inputs"
```

## 9. Rules for Integrating New Features

Before adding an execution-context tensor, first demonstrate that it cannot be produced from existing inputs or
carried by an existing domain object. If a new context is genuinely required, follow these rules:

1. Define typed metadata containing only the minimal external inputs the model actually consumes.
2. Register the metadata builder explicitly in the model registry; do not discover features by scanning modules.
3. Read only the model-level configuration required to build that metadata.
4. Use `build()` to construct temporary eager metadata without allocating addresses across forwards.
5. In Graph Mode, allocate stable addresses with `allocate_persistent()` and update contents in place with
   `update_persistent()` before replay.
6. Keep in-graph intermediates under model and graph-runtime ownership even when their addresses become stable.
7. Do not call `in_acl_graph()` or `get_execution_buffer()` from model code to choose an address source.
8. Do not add model-specific, non-attention fields to `AttentionMetadata`.
9. Validate unsupported inputs before model execution and before entering collectives.
10. Preserve the existing runner path when no builder is registered.
11. Do not expand the model/executor semantic interface for a performance optimization unless profiling proves that
    the optimization cannot remain internal to the builder.

## 10. Review Checklist

### 10.1 Is the Model-Visible Contract Pure?

Models and layers should consume only `forward` arguments and typed metadata. They should not read runner state,
detect Graph Mode, or manage external tensor lifetimes. Mathematical phases such as prefill and decode may be visible
to the model; dummy status, padding origin, warmup, capture, replay, and storage persistence should not be interpreted
by model code.

### 10.2 Is the Metadata Minimal?

Every field should answer two questions: why can the model not derive it from existing inputs, and why can it not use
an existing domain carrier? Do not turn top-k results, outputs, or other in-graph intermediates into execution-contract
fields merely because Graph execution requires stable addresses.

### 10.3 Are Inputs Authoritative?

Independent business semantics such as request count, tokens per request, and query boundaries must come from the
upstream producer. A builder may compute purely derived values, but it must not infer request-scoped semantics from
block tables, state indices, or padded tensor shapes. Redundant values must be checked for consistency rather than
silently choosing one representation.

### 10.4 Is the Graph Lifetime Correct?

Each graph entry must call `allocate_persistent()` only once. Later steps must update contents through
`update_persistent()` without replacing storage. Reviews and tests should verify that:

- `data_ptr` remains unchanged across capture and multiple replays;
- every replay fully resets padding regions instead of retaining stale data;
- no new tensor replaces an address already captured in persistent metadata;
- requests hitting the same bucket do not change metadata type or shape.

### 10.5 Does Validation Fail Before Side Effects?

Shape, dtype, device, head geometry, parallel topology, and unsupported execution phases should be rejected during
builder construction or before the runner enters the model. Avoid cases where some ranks enter a collective before
another rank rejects metadata. Also avoid modifying cache during prefill only to discover an incompatible decode
geometry when generating the first token.

### 10.6 Are Compatibility and Performance Costs Explicit?

Models without registered builders must not construct or validate `InputBatch`; their original paths must remain
unchanged. Retaining the old MegaMoe mask is an intentional migration compatibility measure and should not be removed
as an unrelated cleanup in this PR.

For performance, check whether builders introduce D2H transfers, repeated `.to()` or `.contiguous()` calls, repeated
per-layer plan construction, or unnecessary full copies. Clear semantics take precedence over premature buffer reuse,
but all extra costs must remain identifiable. Optimize them inside builders after profiling instead of exposing Graph
details to the model again.

## 11. Why This Is Not a General Hook System

The execution metadata mechanism resembles hooks only in how runners invoke it. It is not an arbitrary callback
system. Builder lifecycle methods are fixed and constrained, and their results must enter `ForwardContext` as typed
metadata.

These constraints provide three benefits:

- runners can orchestrate all features without allowing builders to alter the main execution flow;
- models depend on explicit data contracts rather than hidden side effects;
- Graph Mode's allocate-once and update-in-place semantics can be tested centrally.

## 12. Follow-Up Work

Future work should proceed in the following order:

1. Complete numerical alignment for Qwen3.5 eager prefill, eager decode, and ACL Graph decode, including end-to-end
   Token Owner MegaMoe validation with TP2/DP2/EP4.
2. Add tests for persistent metadata address stability, cross-step content updates, graph-bucket isolation, and dummy
   ranks.
3. Profile MegaMoe padding, mask updates, and GDN metadata construction. Prefer upstream tensor views and copy only
   when fixed addresses require it.
4. When speculative execution is needed, first propagate authoritative request counts and per-request execution
   widths from upstream, then extend `InputBatch` and the relevant builders. Do not infer them from expanded tensor
   rows.
5. Apply the same pattern incrementally to Full Attention, other linear-attention modules, and other models. Keep
   public `AttentionMetadata` and compatibility paths until their consumers have migrated.
6. Remove the legacy mask field and transitional runner logic only after every consumer uses typed metadata and the
   migration has passed regression testing.
7. Finally, port the Qwen3.5 model implementation against the model-visible input contract used by vLLM Ascend. The
   goal is to reuse the mathematical model graph, not to force both frameworks to share identical schedulers,
   runners, or class hierarchies.

The immediate priority is not to keep expanding the builder interface. It is to prove with a real model that, after
the materialized-input boundary, xLLM and vLLM Ascend can expose equivalent model-visible information without making
model code understand xLLM scheduling or Graph lifetimes.

## 13. Long-Term Maintenance Value

Execution metadata provides a reusable extension path: features integrate with eager and Graph execution through
metadata builders instead of adding special cases to models, runners, or public metadata.

The long-term benefits include:

- eager and Graph Mode reuse the same mathematical model code, improving numerical consistency;
- ownership, update frequency, and reuse policy for external inputs have one reviewable and testable home;
- runners remain execution orchestrators instead of becoming collections of model branches;
- new models and features can provide typed metadata without modifying unrelated features;
- a new graph backend needs to implement the common lifecycle without understanding model details again;
- fixed-address and in-place update optimizations can remain internal to builders without changing model semantics.

The goal is not to hide Graph Mode from every layer of the system. It is to confine Graph knowledge to the execution
layers that should own it: runners manage timing, metadata builders manage model inputs and persistent addresses, and
models remain focused on computation.
