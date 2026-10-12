
# Global Multi-Level KV Cache
## Background

Long-context inference repeatedly reads historical KV cache during autoregressive decoding. As model sizes and context windows grow, device memory capacity and bandwidth become major constraints. A device-only cache also makes a cold request recompute a prefix even when the same prefix was produced by an earlier request or another xLLM instance.

xLLM extends the device prefix cache into a three-level hierarchy:

| Tier | Purpose | Lifetime |
|---|---|---|
| Device HBM | Lowest-latency KV used by the current forward pass | Device-local |
| Host cache | Pinned CPU-memory staging and reusable host prefix cache | xLLM process |
| Mooncake Store | Distributed KV objects shared across xLLM processes and restarts | Store cluster |

A request first checks the Host prefix cache. Missing full blocks can be fetched from Mooncake Store into preallocated Host blocks, restored to HBM layer by layer, and reused without recomputing the matched prefix. Completed HBM blocks are asynchronously copied back to Host memory and then written to Mooncake Store.

The current implementation uses `HierarchyBlockManagerPool` to manage block allocation, mounting, and lifetime; `HierarchyKVCacheTransfer` to perform unified Host-to-Device and Device-to-Host transfers; and one `KVCacheStore` in each worker to access Mooncake Store. A single Store client can manage the target model, a speculative draft model, and multiple `BlockType` values from the same model.

## Architecture

The deployment can contain the following components:

- **etcd**: Registers compute instances and synchronizes service metadata.
- **xLLM Service**: Routes requests and manages fused or disaggregated Prefill/Decode instances.
- **HierarchyBlockManagerPool**: Probes the device and Host prefix caches, creates G2H, H2D, and D2H2G plans, and publishes or releases blocks after asynchronous work completes.
- **HierarchyKVCacheTransfer**: Registers target and draft cache domains, creates Host caches, and performs Host-to-Device and Device-to-Host copies.
- **KVCacheStore**: Maps each logical block to one or more Mooncake objects and performs batched reads, writes, and logical-hit aggregation.
- **xLLM Worker**: Owns the device and Host KV caches, `HierarchyKVCacheTransfer`, and `KVCacheStore`, and executes inference.
- **Mooncake Store**: Provides the distributed, process-independent KV object tier.

The service-level architecture is shown below:

![xLLM Global Multi-Level KV Cache](../../assets/globalkvcache_architecture.png)

## Cache Layouts and BlockType

`HierarchyBlockManagerPool` uses `CompositeBlockManager` to combine a request's cache pools. The hierarchy currently supports three layouts:

| Layout | Device/Host `BlockType` | Complete copy unit / write-back boundary |
|---|---|---|
| `FLAT_KV` | `KV` | One complete KV block |
| `FLAT_KV_LINEAR` | `KV`, `LINEAR` | For Qwen3.5/GLM5.3, one checkpoint/copy unit every `max_tokens_per_chunk_for_prefill` completed tokens |
| `SWA_COMPRESSED` | `SWA`, `C4`, `C128` | For DeepSeek-V4, each unit is exactly `1 SWA + 32 C4 + 1 C128`, or 34 transfer entries |

`KV` is the attention cache used by ordinary and Qwen-family models. `SWA`, `C4`, and `C128` are independent pools in the DeepSeek-V4 compressed-cache layout. `C4` and `C128` must be complete, while `SWA` only needs to cover the window immediately before each restorable `C128` checkpoint. On Decode, the `SWA` Host leaf is write-back-only and does not participate in prefix probing or Host restoration.

With linear attention (GDN/KDA), KV and recurrent state use separate Host block pools, each scaled by `host_blocks_factor`. Host transfer for `LINEAR` stores only committed state: it copies only the first three rows of the device Conv history and the first checkpoint row of the current SSM slot. Prefill retains at most two rolling slots, `[restore, live]`. `EMBEDDING` is a per-sequence speculative-decoding embedding slot; it is not part of hierarchical KV transfer and does not create Mooncake Store objects.

`HierarchyKVCacheTransfer` currently supports H2D and D2H2G copies for `LINEAR`, but the Mooncake G2H prefetch unit in `HierarchyBlockManagerPool` is fully implemented only for `FLAT_KV` and `SWA_COMPRESSED`. Consequently, `FLAT_KV_LINEAR` can use Host transfer and eligible checkpoint write-back, but should not rely on the current Store path for cross-process prefix prefetch. Full Store admission and rolling-state lifetime management for `LINEAR` remain incomplete.

## Unified Cache Domains and Store Keys

`HierarchyKVCacheTransfer` supports multiple registered cache domains. Normal inference registers only the target-model domain. Speculative decoding registers the target and draft models with the same transfer instance so they share one Host-transfer layout and one `KVCacheStore`:

| Cache domain | `CacheRole` | Store `key_component` |
|---|---|---|
| Target model | `TARGET` | Fixed to `main` |
| Embedded draft | `DRAFT` | `spec_draft::<algorithm>::embedded` |
| Separate draft model | `DRAFT` | `spec_draft::<algorithm>::<directory-name>::<normalized-path-digest>` |

During initialization, the Store builds an index by `BlockType`. Every cache domain that supports that `BlockType` produces a separate physical object. For example, when both the target and draft models contain `BlockType::KV`, one logical KV block maps to two Store objects. If a `BlockType` exists only in the target model, only the target object is generated.

The current object-key namespace is `xllm-kv-v3`. `model_id` and `key_component` use length-prefixed encoding, while hash fields are appended as 128-bit binary values. Conceptually, a key contains the following fields:

```text
Non-MLA: xllm-kv-v3:<model_id>:<key_component>:<tp_size>:<tp_rank>
                      :<kv_split_size>:<kv_split_rank>:<block_type>
                      :<schema_hash><block_hash>

MLA:     xllm-kv-v3:<model_id>:<key_component>:mla
                      :<kv_split_size>:<kv_split_rank>:<block_type>
                      :<schema_hash><block_hash>
```

- `model_id` is the target-model namespace and is included in both target and draft objects.
- `key_component` separates the target model, speculative algorithm, and draft-model source.
- Non-MLA caches use `tp_size`, `tp_rank`, `kv_split_size`, `kv_split_rank`, and `block_type` to isolate different parallel topologies, ranks, KV shards, and cache types. Here, `tp_size` and `tp_rank` are the effective local TP topology used by the Store client and do not necessarily equal the global process rank.
- MLA KV cache uses a fixed `mla` marker and omits `tp_size` and `tp_rank`; `kv_split_size` and `kv_split_rank` remain part of the object key. With `kv_split_size=1`, KV is not sharded, every rank reads the same object, and only TP rank 0 writes it. With KV split enabled, each KV split rank reads and writes its own Store object.
- `kv_split_size` is the effective number of KV shards, and `kv_split_rank` is the current Worker's shard index. A configured value of `0` inherits `cp_size`; `1` disables KV sharding and fixes `kv_split_rank` to `0`.
- `schema_hash` is derived from the parallel mode, `BlockType`, and each tensor's role, dtype, and per-block shape, excluding the number of Host blocks. Changing only `host_blocks_factor` therefore does not change object keys. Changes to the KV/LINEAR/compressed layout, dtype, role, or per-block shape automatically select a new key space.
- `block_hash` is the 128-bit content hash of the corresponding logical block. For `LINEAR`, it identifies the committed recurrent checkpoint state.

The Store API exposes logical blocks to its caller and expands them into physical cache-domain requests internally. A worker reports a hit only when **all physical objects** for that logical block are read successfully. `PrefetchResult` then applies a logical AND across the local Worker group involved for the current DP rank. The group size is `world_size / dp_size`; which TP/CP workers it contains depends on the backend topology and must not be reduced to TP rank alone. With KV split enabled, different `kv_split_rank` values use different object keys and cannot be treated as the same cache copy. A Store hit is publishable only when every cache domain and every required Worker are complete.

### KV Split and Store

KV split can be enabled together with Mooncake Store. The Store maps every logical block to an independent object key for each `kv_split_size` and `kv_split_rank`, so one KV shard cannot overwrite another or be mistaken for a complete KV hit. During Store restoration on Prefill, a block is mounted into the Host Prefix Cache only when every registered cache domain and every Worker required by the request hit.

For MLA cache, the unsharded configuration still lets only TP rank 0 write the shared object; with sharding enabled, every KV split rank writes its own object. Under speculative decoding, the target and draft models retain separate `key_component` values and each cache domain is further separated by KV split, so a missing draft shard makes the corresponding logical block a miss.

Valid `kv_split_size` values are checked against the backend, model, and parallel topology. In a conventional CP configuration, `K` must be positive and meet the backend constraint, usually by dividing `cp_size`; `0` inherits `cp_size`, and `1` disables KV sharding. Instances that need to reuse the same Store objects must use the same model identity, cache layout, and effective split topology. Different split topologies occupy different key spaces and do not cross-hit.

## Block Lifecycle

For `FLAT_KV` and `SWA_COMPRESSED`, fused instances and the Prefill side of disaggregated PD use the complete Mooncake admission, Host restore, and write-back path. Decode keeps Store enabled for its Host/Mooncake write-back path, while its request-admission path remains Device-prefix-only. `FLAT_KV_LINEAR` currently guarantees only Host-to-Device and Device-to-Host transfer plus eligible checkpoint write-back; the complete Store-admission path in the diagram does not apply to it. With speculative decoding enabled, each logical block operation in the diagram covers every target or draft cache domain that supports its `BlockType`.

Scheduling-side and execution-side responsibilities are separate:

- `HierarchyBlockManagerPool` owns `load_block_transfer_infos_` and `offload_block_pair_queues_`, block ownership, Host destination reservation, and publish/free operations in completion callbacks. It generates and records transfer plans but does not perform physical copies.
- In each Worker, `HierarchyKVCacheTransfer` performs the actual layer-wise Host-to-Device and Device-to-Host copies and Store get/put operations, creates copy streams, events, and `LayerSynchronizer`, and records `batch_id -> HostKVLoadHandle`. D2H results return to the scheduling side through futures, where the BlockManager aggregates them and updates the Host Prefix Cache.

Write-back does not wait until request `deallocate`. After each Forward updates the HBM token cursor, the next `allocate/grow` publishes the copy unit completed by the previous Forward and immediately collects a new D2H2G mapping. Ordinary models collect once per complete KV block; DeepSeek-V4 collects once per `1 SWA + 32 C4 + 1 C128` composite unit; Qwen3.5/GLM5.3 collect once per chunked-prefill stride. `deallocate` only collects the final complete unit that has not yet been queued, then releases the sequence.

Arrows in the following diagram use function names from the implementation. State changes, thread relationships, and data semantics are described in `Note` entries.

```mermaid
sequenceDiagram
    autonumber

    participant Client as Client / xLLM Service
    participant Scheduler as ContinuousScheduler
    participant BlockMgr as HierarchyBlockManagerPool
    participant Engine as LLMEngine
    participant Remote as RemoteWorker
    participant Channel as CommChannel
    participant Service as WorkerService
    participant Result as PrefetchResult
    participant Worker as WorkerImpl / HierarchyKVCacheTransfer
    participant Copy as HostKVTransfer
    participant Store as KVCacheStore / Mooncake Store
    participant Cache as Host Cache / Device HBM

    rect rgb(235, 245, 255)
        Note over Client,Store: Phase 1: request admission and Mooncake prefetch

        Client->>Scheduler: ContinuousScheduler::add_request()
        Scheduler->>BlockMgr: HierarchyBlockManagerPool::prefetch_from_storage()
        BlockMgr->>BlockMgr: CompositeBlockManager::probe_prefix_cache()
        BlockMgr->>BlockMgr: BlockManager::allocate_for_prefetch()
        BlockMgr->>BlockMgr: build_prefetch_request()
        Note right of BlockMgr: Retain Host hits and reserve holes as G2H destinations

        Note over BlockMgr,Store: If Host already covers the prefix, Store RPCs are skipped
        BlockMgr->>Engine: LLMEngine::prefetch_from_storage()
        Engine->>Result: PrefetchResult::PrefetchResult()

        par All local Workers involved in this DP rank
            Engine->>Remote: RemoteWorker::prefetch_from_storage()
            Remote->>Channel: CommChannel::prefetch_from_storage()
            Channel->>Service: WorkerService::PrefetchFromStorage()
            loop Each WorkerPrefetchSession batch
                Service->>Service: WorkerPrefetchSession::run_batch()
                Service->>Worker: WorkerImpl::prefetch_kv_blocks()
                Worker->>Worker: HierarchyKVCacheTransfer::prefetch_kv_blocks()
                Worker->>Store: KVCacheStore::batch_get_with_status()
                Store-->>Worker: std::vector<uint8_t>
                Worker-->>Service: std::vector<uint8_t> logical_hits
                Service-->>Channel: uint8_t prefix_hit_units (brpc stream)
                Channel->>Result: PrefetchResult::record_batch_result()
                Result-->>Channel: PrefetchControl
                Channel-->>Service: PrefetchControl (brpc stream)
            end
            Service-->>Channel: brpc::StreamClose()
            Channel->>Result: PrefetchResult::mark_worker_ended()
        end

        Result-->>BlockMgr: DoneCallback(common_hit_units)
        Note right of Result: ClientStreamReceiver invokes PrefetchResult in the scheduling process;<br/>common_hit_units is the minimum contiguous hit count across participating Workers
        BlockMgr->>BlockMgr: finalize_prefetch()
        Note right of BlockMgr: Publish hit blocks, release miss blocks, and mount Host state
        BlockMgr-->>Scheduler: PrefetchDoneCallback(request)
        Scheduler->>Scheduler: ContinuousScheduler::enqueue_ready_request()
    end

    rect rgb(240, 255, 240)
        Note over Scheduler,Cache: Phase 2: restore Host KV to HBM and execute Forward

        Scheduler->>BlockMgr: HierarchyBlockManagerPool::allocate()
        BlockMgr->>BlockMgr: CompositeBlockManager::allocate_sequence(sequence, num_tokens)
        BlockMgr->>BlockMgr: CompositeBlockManager::allocate_sequence(sequence, host_state, num_tokens)
        BlockMgr->>BlockMgr: collect_load_block_transfer_infos()
        Note right of BlockMgr: H2D mappings are stored in load_block_transfer_infos_

        Scheduler->>BlockMgr: HierarchyBlockManagerPool::transfer_blocks(batches)
        BlockMgr->>Engine: LLMEngine::transfer_kv_blocks(dp_rank, batch_id, infos)
        Engine->>Remote: RemoteWorker::transfer_kv_blocks(batch_id, infos)
        Remote->>Channel: CommChannel::transfer_kv_blocks(batch_id, infos)
        Channel->>Service: WorkerService::TransferBlocks()
        Service->>Worker: WorkerImpl::transfer_kv_blocks(batch_id, infos)
        Worker->>Worker: HierarchyKVCacheTransfer::transfer_kv_blocks()
        Worker->>Copy: BasicHostKVTransfer::prepare_load() / CompactHostKVTransfer::prepare_load()
        Note right of Worker: HostKVLoadHandle is stored in load_handles_[batch_id]
        Worker->>Worker: load_threadpool_->schedule(load_from_host)
        Worker-->>Service: uint32_t info_count
        Service-->>Channel: TransferStatus.success_cnt
        Note over Remote,Service: RemoteWorker::threadpool_ executes the H2D registration RPC<br/>before the subsequently queued step_remote_async()

        par load_threadpool
            Worker->>Worker: HierarchyKVCacheTransfer::load_from_host()
            Worker->>Copy: HostKVTransfer::load(request, handle)
            Copy->>Copy: BasicHostKVTransfer::load_impl() / CompactHostKVTransfer::load_impl()
            Note over Copy,Cache: The H2D copy stream records a LayerSynchronizer ready event per layer range
        and Forward executor
            Scheduler->>Engine: LLMEngine::step(batches)
            Engine->>Remote: RemoteWorker::step_remote_async(input)
            Remote->>Channel: CommChannel::execute_model_async(input, promise)
            Channel->>Service: WorkerService::ExecuteModel()
            Service->>Worker: WorkerImpl::step_async(input)
            Worker->>Worker: WorkerImpl::set_hierarchy_layer_synchronizer()
            Worker->>Worker: HierarchyKVCacheTransfer::set_layer_synchronizer(params)
            opt Worker owns recurrent cache
                Worker->>Worker: WorkerImpl::prepare_linear_state_cache(params)
                Worker->>Worker: ModelInputParams::synchronize_all_layers()
                Note right of Worker: LINEAR restoration waits for all H2D events<br/>before restore_linear_state_slots()
            end
            loop Each model layer
                Worker->>Worker: ModelInputParams::synchronize_layer(layer_idx)
                Note right of Worker: Wait for the layer-range event before<br/>the layer attention reads HBM KV
            end
            Worker-->>Service: optional<ForwardOutput>
            Service-->>Channel: proto::ForwardOutput
            Channel-->>Remote: RawForwardOutput
            Remote-->>Engine: SemiFuture<optional<RawForwardOutput>>
        end

        Note right of Worker: Target and draft mappings share the same batch_id and synchronizer
    end

    rect rgb(255, 245, 235)
        Note over Scheduler,Store: Phase 3: incremental write-back by copy unit during Prefill/Decode

        loop Continue growing after each Forward
            Note over Scheduler,Cache: Forward N has completed and updated the HBM token cursor
            Scheduler->>BlockMgr: HierarchyBlockManagerPool::allocate()
            BlockMgr->>BlockMgr: CompositeBlockManager::allocate_sequence(sequence, num_tokens)
            Note right of BlockMgr: allocate_sequence() calls<br/>cache_full_blocks_for_sequence() before and after growth
            BlockMgr->>BlockMgr: CompositeBlockManager::allocate_sequence(sequence, host_state, num_tokens)
            BlockMgr->>BlockMgr: collect_offload_pairs()
            Note right of BlockMgr: HBM-to-Host pairs for newly complete units enter<br/>offload_block_pair_queues_

            Scheduler->>BlockMgr: HierarchyBlockManagerPool::transfer_blocks(batches)
            BlockMgr->>BlockMgr: transfer_offload_blocks()
            BlockMgr->>Engine: LLMEngine::transfer_kv_blocks(dp_rank, infos)

            loop Every participating local Worker
                Engine->>Remote: RemoteWorker::transfer_kv_blocks(infos)
                Remote-->>Engine: SemiFuture<uint32_t>
            end
            Engine-->>BlockMgr: vector<SemiFuture<uint32_t>>
            BlockMgr->>BlockMgr: folly::collectAll(...).thenValue()

            par RemoteWorker copy_threadpool_ / Worker executor
                Remote->>Channel: CommChannel::transfer_kv_blocks(infos, promise)
                Channel->>Service: WorkerService::TransferBlocks()
                Service->>Worker: WorkerImpl::transfer_kv_blocks(batch_id, infos)
                Note right of Worker: WorkerImpl queues D2H2G on its single-threaded executor<br/>after the preceding Forward
                Worker->>Worker: HierarchyKVCacheTransfer::transfer_kv_blocks()
                Worker->>Worker: HierarchyKVCacheTransfer::offload()
                Worker->>Worker: HierarchyKVCacheTransfer::offload_to_host()
                Worker->>Copy: HostKVTransfer::offload(request)
                Copy->>Copy: BasicHostKVTransfer::offload_impl() / CompactHostKVTransfer::offload_impl()
                Note over Copy,Cache: When HostKVTransfer::offload() succeeds,<br/>Host tensors are safe for CPU and Store access
                Worker->>Store: KVCacheStore::batch_put()
                Store-->>Worker: uint32_t put_count
                Note right of Store: Unsharded MLA skips inside batch_put() on non-zero TP ranks;<br/>other Workers write their own objects or KV shards
                Note right of Worker: BatchPut is best-effort;<br/>partial failure does not change D2H success
                Worker-->>Service: uint32_t block_count
                Service-->>Channel: TransferStatus.success_cnt
                Channel-->>Remote: folly::Promise<uint32_t>::setValue()
            end

            Note right of BlockMgr: Validate the Worker count and each returned block count
            BlockMgr->>BlockMgr: CompositeBlockManager::deallocate(device_blocks)
            BlockMgr->>BlockMgr: finalize_host_blocks(copy_ok, ...)
            Note right of BlockMgr: On copy_ok, call cache_blocks() before release;<br/>both paths eventually deallocate the retained Host references
        end

        opt Sequence completes or is cancelled
            Scheduler->>BlockMgr: HierarchyBlockManagerPool::deallocate()
            BlockMgr->>BlockMgr: CompositeBlockManager::cache_full_blocks_for_sequence()
            BlockMgr->>BlockMgr: collect_offload_pairs()
            BlockMgr->>BlockMgr: CompositeBlockManager::deallocate_for_sequence(sequence, host_state)
            BlockMgr->>BlockMgr: CompositeBlockManager::deallocate_for_sequence(sequence)
            BlockMgr->>BlockMgr: Sequence::reset()
            Note right of BlockMgr: Collect only the final complete unit not already queued;<br/>the queue retains block references required by D2H
            Scheduler->>BlockMgr: HierarchyBlockManagerPool::transfer_blocks()
            BlockMgr->>BlockMgr: transfer_offload_blocks()
            Note over Scheduler,BlockMgr: ContinuousScheduler::prepare_batch() also calls<br/>the no-argument transfer_blocks() for an empty batch
        end
    end
```

## Speculative-Decoding Cache

When speculative decoding and the Host cache are enabled, `SpeculativeWorkerImpl` makes the target and draft workers reuse one `HierarchyKVCacheTransfer`. Both cache domains must share the same producer stream. After registration is finalized, xLLM creates their Host caches, Host-transfer implementation, and `KVCacheStore` together.

The unified cache domains have the following behavior:

- G2H prefetch reads both target and draft objects. If any object is missing, the logical block is treated as a miss so xLLM never restores an incomplete draft state.
- H2D and D2H requests create separate `target_mappings` and `draft_mappings`, but advance under the same `batch_id` and layer synchronizer.
- D2H2G write-back first copies both cache domains into their Host caches, then writes each domain through the same `KVCacheStore` into its own key space.
- Draft keys automatically include the speculative algorithm and draft source. Changing the draft path selects new keys. If weights are replaced in place at the same path, change `model_id` as well.

Normal non-speculative inference still registers only the `main` cache domain and retains the previous single-domain behavior.

## Disaggregated PD

In disaggregated PD, Mooncake Store admission and Host-to-HBM restore run on the **Prefill** instance. The Decode instance allocates destination Device blocks before Prefill starts and only probes its Device Prefix Cache during admission; it does not mount Host aliases, fetch a prefix from Mooncake, or schedule Host-to-Device restoration. Decode still enables Store and Host cache capacity for its write-back path.

```mermaid
sequenceDiagram
    autonumber

    participant Client as Client / xLLM Service
    participant PSched as PREFILL DisaggPDScheduler
    participant PBlock as PREFILL HierarchyBlockManagerPool
    participant PEngine as PREFILL LLMEngine
    participant PRemote as PREFILL RemoteWorker / RPC
    participant PResult as PREFILL PrefetchResult
    participant PWorker as PREFILL WorkerImpl / HierarchyKVCacheTransfer
    participant Store as KVCacheStore / Mooncake Store
    participant PCache as PREFILL Host / HBM
    participant DService as DisaggPDService / Impl
    participant DSched as DECODE DisaggPDScheduler
    participant DBlock as DECODE KVCacheManager
    participant DEngine as DECODE LLMEngine
    participant DRemote as DECODE RemoteWorker / WorkerImpl
    participant KVTransfer as KVCacheTransfer
    participant DHBM as DECODE HBM

    rect rgb(235, 245, 255)
        Note over Client,PCache: Phase 1: PREFILL admission and Mooncake restoration

        Client->>PSched: ContinuousScheduler::add_request()
        PSched->>PBlock: HierarchyBlockManagerPool::prefetch_from_storage()
        PBlock->>PEngine: LLMEngine::prefetch_from_storage()
        PEngine->>PRemote: RemoteWorker::prefetch_from_storage()
        PRemote->>PRemote: CommChannel::prefetch_from_storage()
        PRemote->>PWorker: WorkerService::PrefetchFromStorage()
        PWorker->>PWorker: WorkerImpl::prefetch_kv_blocks()
        PWorker->>PWorker: HierarchyKVCacheTransfer::prefetch_kv_blocks()
        PWorker->>Store: KVCacheStore::batch_get_with_status()
        Store-->>PWorker: std::vector<uint8_t>
        PWorker-->>PRemote: uint8_t prefix_hit_units (brpc stream)
        PRemote->>PResult: PrefetchResult::record_batch_result()
        PRemote->>PResult: PrefetchResult::mark_worker_ended()
        PResult-->>PBlock: DoneCallback(common_hit_units)
        PBlock->>PBlock: finalize_prefetch()
        PBlock-->>PSched: PrefetchDoneCallback(request)
        PSched->>PSched: DisaggPDScheduler::enqueue_ready_request()
        Note right of PSched: The request enters prefill_request_queue_;<br/>the Scheduler does not poll for Store results
    end

    rect rgb(250, 240, 255)
        Note over PSched,DHBM: Phase 2: allocate Decode destinations first

        PSched->>PSched: DisaggPDScheduler::dispatch_requests()
        PSched->>DService: DisaggPDService_Stub::AddNewRequests()
        DService->>DService: DisaggPDService::AddNewRequests()
        DService->>DService: DisaggPDServiceImpl::decode_recv_new_requests()
        DService->>DSched: DisaggPDScheduler::try_allocate()
        DSched->>DBlock: KVCacheManager::try_allocate()
        Note over Store,DBlock: DECODE admission does not fetch a Host/Mooncake prefix or schedule H2D restore<br/>Store remains enabled for DECODE write-back
        DService->>DSched: DisaggPDScheduler::decode_schedule()
        DService-->>PSched: proto::DisaggResponses
        PSched->>PSched: KVCacheState::set_transfer_kv_info()
        PSched->>PSched: KVCacheState::advance_transfer_block_idx()
        PSched->>PSched: KVCacheState::advance_group_transfer_block_idx()
        PSched->>PSched: folly::MPMCQueue::write(request)
        Note right of PSched: The cursor starts after the D-side remote_shared_num
    end

    rect rgb(240, 255, 240)
        Note over PSched,DHBM: Phase 3: Forward, Host write-back, and P-to-D transfer for each PREFILL chunk

        loop Each PREFILL chunk
            PSched->>PBlock: HierarchyBlockManagerPool::allocate()
            PBlock->>PBlock: CompositeBlockManager::allocate_sequence(sequence, num_tokens)
            PBlock->>PBlock: CompositeBlockManager::allocate_sequence(sequence, host_state, num_tokens)
            PBlock->>PBlock: collect_load_block_transfer_infos()
            PBlock->>PBlock: collect_offload_pairs()
            Note right of PBlock: Growth publishes the copy unit completed by the prior round;<br/>the BlockManagerPool retains both H2D and D2H mappings

            PSched->>PBlock: HierarchyBlockManagerPool::transfer_blocks(batches)
            opt H2D mappings exist
                PBlock->>PEngine: LLMEngine::transfer_kv_blocks(dp_rank, batch_id, infos)
                PEngine->>PRemote: RemoteWorker::transfer_kv_blocks(batch_id, infos)
                PRemote->>PRemote: CommChannel::transfer_kv_blocks(batch_id, infos)
                PRemote->>PWorker: WorkerService::TransferBlocks()
                PWorker->>PWorker: WorkerImpl::transfer_kv_blocks(batch_id, infos)
                PWorker->>PWorker: HierarchyKVCacheTransfer::transfer_kv_blocks()
                PWorker->>PWorker: load_threadpool_->schedule(load_from_host)
                Note over PWorker,PCache: HierarchyKVCacheTransfer::load_from_host() calls HostKVTransfer::load()<br/>on load_threadpool_ to perform layer-wise H2D and record events
            end
            opt D2H2G mappings completed by the previous round exist
                PBlock->>PBlock: transfer_offload_blocks()
                PBlock->>PEngine: LLMEngine::transfer_kv_blocks(dp_rank, infos)
                PEngine->>PRemote: RemoteWorker::transfer_kv_blocks(infos)
                PRemote-->>PEngine: SemiFuture<uint32_t>
                PEngine-->>PBlock: vector<SemiFuture<uint32_t>>
                PBlock->>PBlock: folly::collectAll(...).thenValue()
                PRemote->>PRemote: CommChannel::transfer_kv_blocks(infos, promise)
                PRemote->>PWorker: WorkerService::TransferBlocks()
                PWorker->>PWorker: WorkerImpl::transfer_kv_blocks(batch_id, infos)
                PWorker->>PWorker: HierarchyKVCacheTransfer::offload()
                PWorker->>PWorker: HierarchyKVCacheTransfer::offload_to_host()
                Note over PWorker,PCache: HostKVTransfer::offload() performs layer-wise D2H on the Worker
                PWorker->>Store: KVCacheStore::batch_put()
                PWorker-->>PRemote: uint32_t block_count
                PRemote-->>PBlock: folly::Promise<uint32_t>::setValue(block_count)
                PBlock->>PBlock: CompositeBlockManager::deallocate(device_blocks)
                PBlock->>PBlock: finalize_host_blocks(copy_ok, ...)
            end

            PSched->>PEngine: LLMEngine::step(batches)
            PEngine->>PRemote: RemoteWorker::step_remote_async(input)
            PRemote->>PRemote: CommChannel::execute_model_async(input, promise)
            PRemote->>PWorker: WorkerService::ExecuteModel()
            PWorker->>PWorker: WorkerImpl::step_async(input)
            PWorker->>PWorker: WorkerImpl::set_hierarchy_layer_synchronizer()
            PWorker->>PWorker: HierarchyKVCacheTransfer::set_layer_synchronizer(params)
            opt Worker owns recurrent cache
                PWorker->>PWorker: WorkerImpl::prepare_linear_state_cache(params)
                PWorker->>PWorker: ModelInputParams::synchronize_all_layers()
            end
            opt kv_cache_transfer_mode == PUSH
                PWorker->>KVTransfer: KVCacheTransfer::push_kv_blocks_async()
            end
            loop Each model layer
                PWorker->>PWorker: ModelInputParams::synchronize_layer(layer_idx)
                PWorker->>PWorker: ModelInputParams::record_layer(layer_idx, device)
                Note over PWorker,KVTransfer: synchronize_layer() waits for Host load before attention;<br/>record_layer() lets the PUSH thread read completed KV layer by layer
            end
            opt kv_cache_transfer_mode == PUSH
                Note over KVTransfer,DHBM: KVCacheTransfer writes the preallocated DECODE destination blocks
                PWorker->>PWorker: KVTransferCompletion::wait()
            end
            PWorker-->>PRemote: optional<ForwardOutput> / RawForwardOutput
            PRemote-->>PEngine: SemiFuture<optional<RawForwardOutput>>
            PEngine-->>PSched: ForwardOutput
        end

        PSched->>PSched: DisaggPDScheduler::prefill_send_first_generation()
        PSched->>DService: DisaggPDService_Stub::FirstGeneration()
        DService->>DService: DisaggPDService::FirstGeneration()
        DService->>DService: DisaggPDServiceImpl::decode_recv_first_generation()
        DService->>DSched: DisaggPDScheduler::decode_recv_first_generation()
        alt kv_cache_transfer_mode == PULL
            DSched->>DEngine: LLMEngine::pull_kv_blocks()
            DEngine->>DRemote: RemoteWorker::pull_kv_blocks()
            DRemote->>DRemote: CommChannel::pull_kv_blocks()
            DRemote->>DRemote: WorkerService::PullKVCache()
            DRemote->>DRemote: WorkerImpl::pull_kv_blocks_async()
            DRemote->>KVTransfer: KVCacheTransfer::pull_kv_blocks_async()
            KVTransfer-->>DRemote: SemiFuture<bool>
            DRemote-->>DEngine: bool
            DEngine-->>DSched: bool
            Note over DSched,DHBM: Enqueue continues only after pull_kv_blocks() succeeds
        else kv_cache_transfer_mode == PUSH
            Note over DSched,DHBM: The Prefill Worker already wrote KV to its final destination; no PULL runs
        end
        DSched->>DSched: folly::MPMCQueue::write(request)
        DService-->>PSched: proto::Status

        opt FirstGeneration succeeds
            PSched->>PSched: DisaggPDScheduler::cache_prefill_blocks()
        end
        PSched->>PBlock: HierarchyBlockManagerPool::deallocate()
        PBlock->>PBlock: CompositeBlockManager::cache_full_blocks_for_sequence()
        PBlock->>PBlock: collect_offload_pairs()
        Note over PBlock,Store: The chunk loop has already written back incrementally; deallocate() only collects<br/>the final complete copy unit not yet queued, then releases the sequence
        Note over PSched,PBlock: A later ContinuousScheduler::prepare_batch() keeps flushing<br/>even when it produces an empty batch
        PSched->>PBlock: HierarchyBlockManagerPool::transfer_blocks()
        PBlock->>PBlock: transfer_offload_blocks()
    end

    rect rgb(255, 245, 235)
        Note over DService,DHBM: Phase 4: DECODE execution

        DSched->>DEngine: LLMEngine::step(batches)
        Note over DEngine,DHBM: Forward uses the preallocated and populated Device blocks
        DEngine-->>DSched: ForwardOutput
        Note over Client,DSched: ResponseProcessor emits the token stream
    end
```

The scheduler supports both `PUSH` and `PULL` through `kv_cache_transfer_mode`. The current code no longer provides a `kv_cache_transfer_type` option. Enable the global Mooncake Store tier on both Prefill and Decode, using disjoint `store_local_hostname` base-port ranges.

## Deployment

### Prerequisites

- Build and install [xLLM](/en/getting_started/quick_start/).
- Install [xLLM Service](https://github.com/xLLM-AI/xllm-service) when service routing or disaggregated PD is required.
- Build or install the Mooncake Store `mooncake_master` and `mooncake_client` binaries.
- Reserve enough Host memory. `--host_blocks_factor > 1` is required to create Host Cache. Mooncake Store additionally requires `--enable_prefix_cache=true` together with `--host_blocks_factor > 1`.
- For DeepSeek-V4 compressed cache, Host capacity is allocated independently for `SWA`, `C4`, and `C128`. With linear attention, `KV` and `LINEAR` also consume separate Host block pools. Do not estimate Host memory from ordinary KV blocks alone.

Mooncake's etcd-backed high availability backends are enabled by default when building xLLM and the bundled Mooncake binaries:

```bash
MAX_JOBS=32 SKIP_EXPORT=1 \
  python setup.py build --device npu
cmake --build build/cmake.linux-aarch64-cpython-311 \
  --target mooncake_master mooncake_client -j32
```

Ready-to-use HA master, independent Store client, and xLLM argument scripts are available under `scripts/kvcache_store/`.

### Start a Minimal Mooncake Store

The following TCP example uses Mooncake's P2P handshake, so no separate Transfer Engine metadata service is required:

```bash
export MC_STORE_CLUSTER_ID=xllm-mooncake

mooncake_master \
  --rpc_address=0.0.0.0 \
  --rpc_port=50051
```

Start at least one resource-owning Store client:

```bash
mooncake_client \
  --host=0.0.0.0:50053 \
  --port=50052 \
  --global_segment_size=4GB \
  --master_server_address=127.0.0.1:50051 \
  --metadata_server=P2PHANDSHAKE \
  --protocol=tcp
```

### Start a High-Availability Mooncake Store Cluster

First start an etcd cluster reachable by every Mooncake master. Then start one master instance on each master node. All instances use the same etcd endpoints and `cluster_id`, while `rpc_address` must identify the reachable address of that specific instance:

```bash
mooncake_master \
  --enable_ha=true \
  --ha_backend_type=etcd \
  --ha_backend_connstring="10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379" \
  --cluster_id=xllm-mooncake \
  --rpc_address=10.0.1.11 \
  --rpc_port=50051
```

Store clients and xLLM use etcd to discover and follow the current leader instead of binding to one master address:

```bash
export MC_STORE_CLUSTER_ID=xllm-mooncake
MOONCAKE_HA_ENTRY='etcd://10.0.0.1:2379;10.0.0.2:2379;10.0.0.3:2379'

mooncake_client \
  --host=0.0.0.0:50053 \
  --port=50052 \
  --global_segment_size=4GB \
  --master_server_address="${MOONCAKE_HA_ENTRY}" \
  --metadata_server=P2PHANDSHAKE \
  --protocol=tcp

/path/to/xllm \
  --enable_prefix_cache=true \
  --host_blocks_factor=4 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address="${MOONCAKE_HA_ENTRY}" \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:12345
```

The `etcd://` prefix in `store_master_server_address` selects the HA leader-discovery backend. Do not add `http://` to the endpoint list after that prefix. When using a custom `cluster_id`, every Mooncake master, Store client, and xLLM process must use the same `MC_STORE_CLUSTER_ID`.

### Start etcd and xLLM Service

This step is required for service routing and disaggregated PD, but not for a standalone fused xLLM process:

```bash
./etcd \
  --listen-peer-urls=http://0.0.0.0:10999 \
  --listen-client-urls=http://0.0.0.0:10998
```

```bash
./xllm_master_serving \
  --etcd_addr=127.0.0.1:10998 \
  --http_server_port=28888 \
  --rpc_server_port=28889 \
  --tokenizer_path=/path/to/tokenizer_config_dir/
```

### Fused xLLM Example

```bash
/path/to/xllm \
  --model=/path/to/model \
  --model_id=my-model-revision-v1 \
  --enable_prefix_cache=true \
  --host_blocks_factor=4 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address=127.0.0.1:50051 \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:12345 \
  --prefetch_batch_size=8 \
  --prefetch_timeout=30000
```

`store_local_hostname` is a base Transfer Engine endpoint. When no port is configured, it defaults to `127.0.0.1:12345`. Each worker uses `base_port + worker_rank`, so the entire port range must be free and reachable. Prefill and Decode, or multiple xLLM instances on the same node, must use disjoint base-port ranges.

For RDMA, set `--store_protocol=rdma`. Use `--store_rdma_devices=mlx5_0,mlx5_1` to select HCAs for the Store client embedded in each xLLM Worker, or leave it empty for Mooncake auto-discovery. Initialization failures remain RDMA failures and never fall back to TCP. xLLM does not read `DEVICE_NAMES`; the standalone `mooncake_client` uses its own `--device_names` option.

Speculative decoding does not require a separate draft Store namespace. xLLM automatically generates a distinct `key_component` for the draft cache. If `--model_id` is omitted, xLLM uses the final component of the model path; production deployments should still provide a stable `--model_id` that identifies the model version.

### Prefetch Measurement Logs

Set `XLLM_STORE_PREFETCH_STATS=true` to enable Store prefetch measurements;
they are disabled by default. Each read batch adds a synchronous replica-tier
query. The master's `prefetch_latency` and the worker's `total` include this
overhead. The worker's `get_time` excludes it, and `tier_query_time` reports it
separately.

The master logs `request_id`, `dp_rank`, `prompt_tokens`, `host_hit_tokens`,
`target_tokens`, `fetched_tokens`, `stop_reason`, and `prefetch_latency`.
Workers use the same `request_id` to log per-rank read volume, tier breakdowns,
`queue_wait`, `get_time`, `tier_query_time`, `control_wait`, `total`, and `get_bw`.
The master does not aggregate worker read volume or bandwidth.

Worker `units` means hits/executed/requested, and `blocks` means hits/requested.
`status` is `completed` or `failed`. Failure reports retain completed Get
statistics, including a Get that was still running when the stream closed,
and skip the tail probe. `total` ends at stream closure; a failure report may
be emitted later, after the running Get returns.

`probed_units` is the tail beyond this rank's contiguous gated-hit prefix.
It starts at the first gated miss in the last batch, or after the batch when
every unit hit. It includes units already read later in that batch but unable
to form a contiguous hit prefix after the miss. `probed_present_units` counts
units whose gated blocks are all present in the Store after stream closure.
This is a per-rank snapshot, not a global count of reusable KV or recomputed
tokens. `probe_status` is `complete`, `invalid` for a result-length mismatch,
or `skipped` on failure. The latter two report `probed_present_units=unknown`.

### Disaggregated PD Example

Use the normal [Disaggregated PD](/en/features/disagg_pd/) flags and enable Store on both roles. Use different `store_local_hostname` base ports for Prefill and Decode:

```bash
/path/to/xllm \
  --enable_disagg_pd=true \
  --instance_role=PREFILL \
  --kv_cache_transfer_mode=PUSH \
  --enable_prefix_cache=true \
  --host_blocks_factor=4 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address=127.0.0.1:50051 \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:12345
```

If the backend and model on Prefill support KV split, the same command can include a parallel configuration such as:

```text
--cp_size=4 \
--kv_split_size=2
```

Here, `2` shards KV across two split ranks, and every shard uses an independent Store key. Whether Decode enables KV split, and which topology it uses, depends on its own backend and model support. A different topology uses a different Store key space.

Enable Store on Decode with a different local endpoint range:

```bash
/path/to/xllm \
  --enable_disagg_pd=true \
  --instance_role=DECODE \
  --kv_cache_transfer_mode=PUSH \
  --enable_prefix_cache=true \
  --host_blocks_factor=4 \
  --enable_kvcache_store=true \
  --store_protocol=tcp \
  --store_master_server_address=127.0.0.1:50051 \
  --store_metadata_server=P2PHANDSHAKE \
  --store_local_hostname=127.0.0.1:13345
```

See the [CLI Reference](/en/cli_reference/) for all `KVCacheStoreConfig` parameters. `prefetch_batch_size` controls the batch size for contiguous Store units. After a Worker encounters the first miss in a batch, it stops requesting later batches; the final hit length is the common prefix across all Workers. `layers_wise_copy_batchs` controls how many layers each synchronization event covers when Host-to-Device or Device-to-Host copies are grouped by layer.

## Correctness and Operational Notes

- Store hits use two levels of completeness checks: each Worker must successfully read every registered cache domain for the `BlockType`, and the local Worker group for the current DP rank must then report a hit before the block is mounted into the Host Prefix Cache. With KV split enabled, each `kv_split_rank` shard must hit independently.
- `prefetch_timeout` stops issuing new prefetch batches after the timeout, but admission still waits for every in-flight Worker batch to finish. `0` waits indefinitely.
- H2D registration does not wait for the physical copy. Forward attaches a `LayerSynchronizer` using `batch_id` and waits at the corresponding layers. The Scheduler receives no H2D-complete callback.
- Host Prefix publication depends only on successful D2H completion from every participating Worker. Mooncake `BatchPut` is best-effort; a partial Store write failure is logged but does not invalidate an already successful Host copy.
- `BatchPut` first deduplicates object keys. Mooncake's "object already exists" result counts as success and does not overwrite the existing object. Duplicate objects in one batch are written only once. A logical block counts as a Store success only when every cache-domain object already exists or is written successfully.
- The current Store key version is `xllm-kv-v3`. Host-block capacity is excluded from `schema_hash`, while tensor role, dtype, per-block shape, TP topology, `kv_split_size`, `kv_split_rank`, `BlockType`, and cache-domain identity isolate the key space. Objects written under older key formats do not match the current format and must be written again.
- Weight contents are not automatically encoded in object keys. Use a new `model_id` whenever target or draft weights, quantization, or any other setting that can change KV values is updated, and rotate or clean the old Store namespace as needed.
- In PD, Prefill and Decode both enable Store. They must use disjoint `store_local_hostname` base-port ranges because both roles reuse worker ranks and each worker binds `base_port + worker_rank`.
- The `LINEAR` state in `FLAT_KV_LINEAR` currently supports committed-checkpoint copies between Host and Device and eligible D2H2G checkpoint write-back, but it is not part of the current Store G2H prefetch-unit path. Before relying on cross-process `LINEAR` reuse, verify that the deployed version implements complete admission and rolling-state lifetime management.
