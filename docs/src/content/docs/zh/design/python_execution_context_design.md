---
title: "Python 模型执行 Metadata 设计"
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

## 1. 背景

Python 模型接入新的执行特性时，经常需要模型 forward 参数之外的额外输入，例如：

- scheduler、runtime 或 metadata 提供的动态 Tensor；
- ACL Graph replay 前内容变化、但地址必须稳定的外部输入；
- 模型无法从现有参数和输入中自行推导的执行信息。

如果直接在模型层或 runner 中处理这些需求，通常会产生两类问题。

第一类问题是模型感知执行模式。模型开始判断当前是 eager 还是 Graph Mode，并自行选择临时分配
或持久化分配。这样，同一份数学计算会出现多套控制流，模型正确性和执行引擎的地址生命周期
相互耦合。

第二类问题是 runner 感知具体功能。eager runner 和各类 graph runner 逐渐包含不同模型、算子
和功能模块的输入准备逻辑，最终成为大量特例分支的集合。

当前不少模型采用的正是这种耦合式写法：runner 维护模型专属的静态 Tensor 或 Graph buffer，模型
再直接读取公共 `AttentionMetadata`、`ForwardContext.layer_caches` 等底层载体，并在 layer 内完成
state index 转换、padding、plan 构造以及执行阶段判断。其依赖关系大致如下：

```text
runner 知道模型需要哪些固定地址和特殊字段
        ↓
公共 metadata 同时承载 attention 与其他模块的数据
        ↓
模型读取公共 metadata，并解释 eager/Graph、padding、dummy 等执行细节
        ↓
模型在 layer 内整理算子真正需要的输入
```

这种实现短期接入直接，但模型、runner 和公共 metadata 会随功能增长而同步膨胀。迁移新模型时，
即使数学组图与已有实现一致，也必须重新理解 xLLM 的 Graph 生命周期和历史字段约定。

执行上下文机制的目标，是在二者之间建立清晰边界：

```text
runner 管理执行生命周期
        ↓
InputBatch 承载上游已经确定的执行语义
        ↓
Execution Metadata Builder 准备统一的模型执行输入
        ↓
ForwardContext 传递强类型上下文
        ↓
模型只消费输入并执行计算
```

执行上下文不是通用 scratch allocator，也不是 Graph 中间 Tensor 的注册表。需要固定地址不等于需要
进入 `ForwardContext`：由模型计算产生、并完全在图内消费的中间 Tensor，应继续由组图和 Graph
运行时管理。

## 2. 本次 PR 做了什么

本次 PR 不是单独为 Qwen3.5 增加一组 runner 特例，而是先建立模型执行 metadata 的通用边界，再用
Qwen3.5 的 MegaMoe 和 MegaGDN 验证这条边界能否工作。改造前后的主要差异如下：

| 改造前 | 改造后 |
| --- | --- |
| runner 直接维护模型或功能专属 Tensor | runner 只在统一生命周期调用 metadata builder |
| 模型直接读取并解释公共 metadata | builder 将公共输入转换成模块级 typed metadata |
| 模型在每个 layer 内重复整理 state index、plan 等算子输入 | builder 在模型执行前一次性完成整理和校验 |
| 模型根据 eager、Graph、padding 或 dummy 选择数据准备路径 | 模型始终通过同一种 typed metadata 获取输入 |
| 图内中间结果也可能被提升为 runner 管理的持久化资源 | 图内中间结果继续由模型组图和 Graph 运行时管理 |
| 新功能需要同时修改模型、runner 和公共 metadata | 模型 Registry 显式注册 builder，通用 runner 不增加模型分支 |

### 2.1 稳定的 step 级输入语义

上游新增并透传 request-scoped batch metadata，在 Python 侧构造成 `InputBatch`。当前稳定字段包括请求
数、执行 token 数、每请求 scheduled/computed token 数、query 边界和 prefill 状态。这些字段由真正
掌握语义的上游生产者提供，Python 不再通过 `block_table` 行数、state index 数量或 Tensor shape
猜测请求数。

`InputBatch` 只在当前模型实际注册了 builder 时构造，因此尚未迁移的模型不会新增校验、对象构造或
执行依赖。Graph runner 只在 graph entry 内为它补充固定地址输入、after-padding 容量和 padding mask，
不会改变其中的真实请求语义。

### 2.2 通用 builder 生命周期

本次新增 `ExecutionMetadataBuilder`，作为介质化输入与模型组图之间的模块级适配边界。协议提供三个
生命周期入口：

- `build()`：为 eager forward 构造本轮 metadata；
- `allocate_persistent()`：创建 graph entry 时分配固定地址；
- `update_persistent()`：replay 前按本 step 输入原地更新固定地址内容。

Registry 负责声明模型需要哪些 builder，Executor 只负责实例化和绑定，runner 只负责在正确时机遍历
builder。具体模块需要哪些 Tensor、如何校验以及哪些地址必须持久化，均不进入通用 runner。

### 2.3 Qwen3.5 样板迁移

本次迁移了两类模块：

- MegaMoe：模型只读取 `MegaMoeMetadata.active_token_mask`。topk、padding 后的算子输入及输出都是
  模型图内中间结果，不再为每层建立外部 context；
- MegaGDN：builder 将公共 metadata、`InputBatch` 和层 cache 整理成 `GdnPrefillMetadata` 或
  `GdnDecodeMetadata`。模型不再读取公共 `AttentionMetadata`，也不再在每层重复构造 prefill plan。

Qwen3.5 的模型 forward 同时整理为接近 vLLM 的参数形式，但当前仍只支持 `input_ids`，不在本 PR 中
引入 pipeline parallel 或 `inputs_embeds` 执行能力。

### 2.4 明确不在本次范围内的内容

本次 PR 不声称已经完成整个 Python 执行架构的迁移，以下内容仍保持现状：

- Full Attention 继续复用现有 `AttentionBackend` 和公共 `AttentionMetadata`；
- 当前 `InputBatch` 和 Qwen3.5 GDN builder 不支持 speculative execution；
- DeepSeek-V4 等其他模型不切换到新的 MegaMoe builder；
- ACL Graph runner 中原有的 `mega_moe_token_mask` 兼容路径暂时保留；
- 不迁移 vLLM Ascend 的算子注册、编译系统或完整 runner 类型体系。

因此，这个 PR 的判断标准不是“所有模型已经完成统一”，而是新的模型可见契约是否清晰、通用
runner 是否没有新增 Qwen3.5 特例，以及未迁移模型是否保持原有行为。

## 3. 核心设计哲学

### 3.1 Context 最小原则

先区分“权威输入”和“派生输入”。如果 scheduler、BatchBuilder 或 speculative worker 已经知道某个
字段的准确语义，就必须由该生产者显式写入并沿执行链路透传，不能在 Python 中根据 Tensor shape、
Attention metadata 行数等旁路信息重新猜测。只有上游确实没有提供、且不存在独立业务语义的纯
派生值，才允许由下游计算；例如 `has_prefill` 可以由权威的 per-request `is_prefilling` 计算，但
不能反过来用 attention 行数猜 request 数。所有冗余表达都应做一致性校验，出现冲突时 fail closed。

Context 只应承载模型无法从现有输入推导、且必须由执行层提供的信息。新增字段前依次确认：

1. 如果它是模型参数或正常 forward 输入，直接通过已有接口传递；
2. 如果它能由已有 Tensor 在图内产生，将它保留为普通中间 Tensor；
3. 如果已经存在合适的领域载体，例如 `AttentionMetadata` 或 `LayerCache`，不要重复存储；
4. 只有当数据来自模型 forward 之外、又必须由执行层提供时，才引入 execution context；Graph 固定
   地址只是其中一种生命周期实现，不应改变模型可见契约。

判断一个字段是否必要，可以先问：

> 删除这个 context 字段后，模型能否根据已有输入完成相同计算？

如果答案是“能”，该字段通常不应进入 context。单纯为了复用显存或减少临时分配而扩大 context，
会把性能优化固化为执行契约；这类优化应在 profiling 证明必要后单独设计。

### 3.2 模型只感知计算，不感知地址来源

对模型而言，上游传入的是一个可以使用的 Tensor。该 Tensor 是 eager 临时申请的，还是由某个
graph entry 长期持有的固定地址，不应改变模型的计算逻辑。

因此，模型层不应：

- 查询当前是否处于 Graph Mode；
- 自行调用 graph buffer 管理接口；
- 根据 eager、warmup、capture 或 replay 选择不同数学路径；
- 推导某个 Tensor 应该跨 step 还是跨 layer 复用。

模型只负责按照约定读取输入、执行算子并返回输出。

### 3.3 外部执行输入的生产者负责生命周期

谁创建 Tensor，谁就应该决定：

- shape、dtype 和 device；
- 地址需要保持多久；
- 内容何时更新；
- 能否跨 layer 或跨 step 复用；
- 是只读共享，还是可写独占；
- 不满足约束时是否应在执行前拒绝或选择其他路径。

对于需要固定地址的外部执行输入，Execution Metadata Builder 是这些决策的唯一责任主体。模型
只消费 builder 提供的数据，不反向猜测地址的生命周期。模型内部产生的 topk、临时输出等中间
Tensor 仍由模型和 Graph 运行时管理，不应为了固定地址转移给 builder。

### 3.4 runner 只管理时机，不理解功能语义

runner 知道什么时候创建 graph entry，以及什么时候准备 replay。它天然适合决定调用时机，但不应
理解某个模型功能需要哪些 Tensor。

因此 runner 只调用统一生命周期接口：

- eager forward 前调用 `build()`；
- graph entry 创建时调用 `allocate_persistent()`；
- graph replay 前调用 `update_persistent()`。

具体构建什么、分配什么、更新什么以及如何复用，都由 builder 决定。eager runner 调用 `build()`
构建本轮 metadata；Graph runner 调用持久化生命周期接口。

### 3.5 ForwardContext 只负责传递

`ForwardContext` 是一次模型 forward 的运行时载体。它通过
`execution_contexts: dict[type[object], object]` 保存不同功能提供的强类型上下文。

它不负责构造或解释上下文。模型通过类型获取自己需要的对象：

```python
metadata = get_execution_context(MyFeatureMetadata)
```

以类型作为 key，可以避免持续为 `ForwardContext` 增加功能专属字段，也能在重复注册或类型错误时
尽早失败。

## 4. 通用抽象

`xllm/python/model_executor/execution_context.py` 定义了统一协议：

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

`Metadata` 表达模型消费的数据语义，不承诺底层 Tensor 的生命周期。eager metadata 可以引用本轮
临时 Tensor；Graph metadata 则引用 graph entry 持有的固定地址 Tensor。

### 4.1 `build()`

eager runner 每轮调用 `build()`，将 `InputBatch` 和已介质化 metadata 转成模型直接消费的 typed
metadata。它可以引用上游 Tensor，也可以按当前执行容量构造临时 Tensor。

模型不得通过 `in_acl_graph()` 或 context 是否存在来选择地址准备路径。eager 和 Graph 都必须通过
统一的 typed-context 访问函数取得同一种 metadata 类型。

### 4.2 `allocate_persistent()`

在 graph entry 创建时执行一次，为 context 承载的图外输入分配固定地址。capture 和后续 replay
都使用同一组地址；图内产生的中间 Tensor 继续由 Graph 运行时管理。Builder 可以从 `input_batch`
和 `metadata` 推导容量、shape、dtype、device 等分配规格，但动态内容仍由 `update_persistent()`
在每轮执行前写入。

该接口只负责建立 graph entry 的长期存储，不应混入具体请求每 step 才能确定的值。

### 4.3 `update_persistent()`

在每次 replay 前更新动态内容。更新必须在已有 Tensor 上原地完成，不能替换 capture 时记录的
地址。`InputBatch` 提供请求级执行语义和 Graph padding 信息；`AttentionMetadata` 提供已经介质化的
`block_table`、state indices 等动态 Tensor。Builder 根据自身模块语义选择所需来源。

不是所有 Tensor 都需要 update。builder 应明确区分：

- 跨 step 不变的内容：分配时初始化一次；
- 每 step 变化、但所有 layer 相同的内容：每 step 更新一次并跨层共享；
- 每 layer 不同的内容：由 builder 管理对应 layer 的独立地址。

## 5. 完整执行流程

### 5.1 初始化阶段

模型 Registry 在注册模型实现时，同时显式登记该模型对应的 execution metadata builder 类型。Builder
根据模型配置决定本次是否需要实例化。

`ModelExecutor` 根据当前 Registry entry 调用 Builder 的 `from_config()`，再将成功创建的 builder
绑定到 eager 和 Graph runner。Executor 不扫描模型模块，也不识别具体模型功能；未注册 builder
或未启用对应功能时，原有执行路径保持不变。

### 5.2 eager 阶段

```text
请求输入和 metadata
  -> runner 调用 builder.build(input_batch, metadata)
  -> runner 将 typed metadata 放入 ForwardContext
  -> runner 调用 attention_backend.prepare()
  -> 模型读取 typed metadata 并执行 forward
```

eager 不为对齐 Graph 生命周期而额外分配固定地址。算子要求各 rank 输入 shape 一致时，由 builder
根据当前 step 的权威信息计算容量并构造临时 metadata；模型只消费结果。

### 5.3 Graph Mode 阶段

创建 graph entry 时：

```text
确定 graph bucket
  -> runner 在 InputBatch 中补充 after-padding 容量和 padding mask
  -> runner 调用 builder.allocate_persistent(input_batch, metadata)
  -> builder 分配稳定地址
  -> warmup / capture 使用该上下文
```

每次 replay 前：

```text
本 step InputBatch
  -> 上游实际 InputBatch 与 graph entry 的固定地址组合成 graph InputBatch
  -> runner 调用 builder.update_persistent(persistent_metadata, input_batch, metadata)
  -> builder 从 InputBatch 或 metadata 读取当前 step 的动态值
  -> builder 原地更新动态内容
  -> graph 使用原地址 replay
```

模型在 warmup、capture 和 replay 中看到的是同一种上下文类型和同一套计算接口。模型不需要知道
当前 forward 属于哪个阶段。

## 6. Qwen3.5 样板如何工作

### 6.1 MegaMoe

MegaMoe 真正来自模型外部、且模型无法自行推导的动态输入只有有效 token mask。因此 typed metadata
保持为最小结构：

```python
@dataclass(frozen=True, slots=True)
class MegaMoeMetadata:
    active_token_mask: torch.Tensor
```

eager 下，builder 根据各 DP rank 的执行 token 数构造本轮 mask；Graph 下，builder 为 graph bucket
分配固定地址 mask，并在每次 replay 前根据 padding mask 原地更新。模型侧始终读取
`MegaMoeMetadata`，不判断地址来自 eager 还是 Graph。

每个 MoE layer 自己计算 gate、topk、padding 后的算子输入和输出。这些值完全由本层 hidden states
产生，并在同一个计算图内消费，不属于外部执行输入。因此它们不进入 metadata，也不需要通过
`layer_id` 查找 per-layer buffer。Graph 捕获会自然记录每层图内中间 Tensor 的地址。

为保证迁移期间不影响其他模型，ACL Graph runner 原有的通用 MegaMoe mask 仍然保留。Qwen3.5 使用
builder 提供的新 mask，旧 mask 对它是暂时未消费的兼容资源。只有在其他消费者完成迁移并验证后，
才能单独删除旧路径。

### 6.2 MegaGDN

MegaGDN 与 MegaMoe 不同，它具有跨 step 保存的 Conv/SSM state，并且 prefill、decode 使用不同算子
输入。Builder 在 cache 绑定时按 `layer_id` 建立 `GdnStateCache` 映射，每个 GDN layer 执行时用自己的
`layer_id` 取得对应 cache：

```text
GdnMetadata
  -> state_caches[layer_id]
       -> conv_state
       -> ssm_state
```

prefill 时，builder 一次性完成：

- read/write state index 解析；
- initial-state 无效读槽处理；
- Conv slot 到 SSM checkpoint slot 的转换；
- `cu_seqlens` 和 `num_matrices` 构造；
- prefill/decode 共同能力边界的 fail-fast 校验。

这些结果形成一个 forward 共享的 `GdnPrefillMetadata`，所有 GDN layer 复用同一份 plan，但通过
`layer_id` 使用各自的 state cache。decode 时则形成 `GdnDecodeMetadata`；Graph entry 持有固定地址的
read/write index Tensor，replay 前由 builder 原地更新。

模型只根据 typed metadata 的具体 phase 调用对应数学算子。模型感知 prefill 或 decode 是合理的，
因为二者使用不同的算子语义；模型不应感知的是该 metadata 是否来自 eager、warmup、capture 或
replay。

## 7. Tensor 来源与归属

先判断数据从哪里产生，再决定由谁维护地址。

| Tensor 类型 | 示例 | 推荐归属 |
| --- | --- | --- |
| 模型参数 | 权重、静态通信 context | Module 或模型配置 |
| 正常 forward 输入 | token、position、hidden states | 显式 forward 参数 |
| 图内中间结果 | router topk、临时输出 | 模型和 Graph 运行时 |
| 已有领域输入 | attention block table、KV cache | `AttentionMetadata`、`LayerCache` 等已有载体 |
| 图外动态执行输入 | 有效行 mask、动态 state index | execution context |
| 上述输入的 Graph 固定地址副本 | static mask、static state index | metadata builder 分配并原地更新 |

真正适合 context 的动态输入，在 eager 和 Graph 下可以使用不同存储策略，但对模型暴露相同的
Tensor 语义：

```text
Eager：直接使用上游 Tensor，或从当前输入临时构造
Graph：metadata 引用固定地址，builder 在 replay 前原地更新
模型：执行同一套数学计算，不解释地址生命周期
```

例如，`state_indices` 的值来自上游资源管理，模型无法自行推导；Graph replay 又要求输入地址固定，
因此适合由 builder 在 `allocate_persistent()` 中申请，并在 `update_persistent()` 中写入。相反，router topk 和当前层
输出由模型计算产生，即使 Graph 最终使用固定地址，也只是普通图内中间 Tensor，不应进入 context。

Context 内确实需要多个 buffer 时，复用仍需同时考虑 shape 语义、dtype、device、读写关系和
生命周期。但复用是第二阶段的性能优化，不能替代“这个字段是否应该进入 context”的必要性判断。

## 8. 模块职责

| 模块 | 应负责 | 不应负责 |
| --- | --- | --- |
| 模型或 layer | 声明静态需求、读取 context、执行数学计算 | 判断 eager/graph、管理持久化地址 |
| execution metadata builder | 构建 eager metadata；分配、复用、更新和校验 Graph 固定地址 | 执行模型数学逻辑 |
| Graph runner | 在正确生命周期调用所有 builder | 识别具体模型或功能的 Tensor 语义 |
| eager runner | 调用 builder 构建本轮 typed metadata | 识别具体模型或功能的 Tensor 语义 |
| `ForwardContext` | 承载当前 forward 的强类型上下文 | 创建 context 或决定复用策略 |
| 调度与 batch 构建 | 提供当前 step 的执行视图 | 关心模型内部 buffer 的地址组织 |

这一边界可以概括为：

```text
调度层决定“这一步执行什么”
runner 决定“什么时候准备和执行”
metadata builder 决定“模型读取哪些 metadata，以及 Graph 地址如何维护”
模型决定“拿这些输入做什么计算”
```

## 9. 接入新功能时的约束

新的执行功能若需要额外 Tensor，应先证明它无法通过已有输入产生或已有领域载体传递；确实需要
execution context 时，再遵守以下规则：

1. 定义强类型 metadata，字段只包含来自图外、且模型真正需要消费的最小输入。
2. 在模型 Registry entry 中显式登记 metadata builder 类型，禁止通过遍历模型模块隐式发现功能。
3. Builder 只从配置中读取其 metadata 构建真正需要的模型级信息。
4. eager 通过 `build()` 构建临时 metadata，不分配跨 forward 的固定地址。
5. Graph Mode 在 entry 创建时调用 `allocate_persistent()` 分配稳定地址，replay 通过
   `update_persistent()` 原地更新内容。
6. 图内中间 Tensor 默认由模型和 Graph 管理，不因地址固定或复用诉求进入 context。
7. 不在模型中调用 `in_acl_graph()` 或 `get_execution_buffer()` 决定地址来源。
8. 不向 `AttentionMetadata` 增加与 attention 无关的模型专属字段。
9. 所有不支持条件应在模型执行和 collective 开始前检查。
10. builder 不存在时，runner 的原有路径保持不变。
11. 性能优化不得扩大模型与执行器之间的语义接口，除非 profiling 证明无法在实现内部完成。

## 10. Review 时应该关注什么

### 10.1 模型可见契约是否足够纯粹

重点确认模型及 layer 只消费 forward 参数和 typed metadata，不读取 runner 状态，不判断 Graph
模式，也不负责外部 Tensor 的地址生命周期。prefill/decode 等数学 phase 可以被模型感知；dummy、
padding、warmup、capture、replay 及地址是否持久化不应被模型解释。

### 10.2 metadata 是否满足最小原则

每个字段都应能回答两个问题：为什么模型无法从现有输入推导，以及为什么不能放入已有领域载体。
特别注意不要因为 Graph 需要固定地址，就把 topk、输出或其他图内中间 Tensor 扩大为模型执行契约。

### 10.3 信息来源是否权威

请求数、每请求 token 数、query 边界等独立业务语义必须来自上游。Builder 可以计算纯派生值，但不应
从 block table、state index 或 padded Tensor shape 反推 request-scoped 语义。冗余信息应做一致性
校验，而不是静默选择其中一个。

### 10.4 Graph 生命周期是否正确

每个 graph entry 只能调用一次 `allocate_persistent()`；后续 step 必须通过
`update_persistent()` 原地改值。Review 和测试需要确认：

- capture 与多次 replay 之间 `data_ptr` 不变；
- padding 区域每轮都被完整重置，不保留上一轮脏值；
- 不会用新的 Tensor 替换 persistent metadata 中已捕获的地址；
- 同一个 bucket 不会因请求变化而改变 metadata 类型或 shape。

### 10.5 fail-fast 是否发生在副作用之前

shape、dtype、device、head geometry、并行拓扑及不支持的执行阶段，应尽量在 builder 创建或
runner 进入模型前拒绝。尤其要避免某些 rank 已进入 collective、某些 rank 才发现 metadata 非法；
也要避免 prefill 已修改 cache，首个 decode token 才暴露算子能力不一致。

### 10.6 兼容性和性能是否可解释

未注册 builder 的模型不应构造或校验 `InputBatch`，原有路径必须保持不变。当前保留旧 MegaMoe
mask 是有意的迁移兼容，不应在本 PR 中顺手删除。

性能方面重点观察 builder 是否引入 D2H、重复 `.to()`/`.contiguous()`、每层重复 plan 构造或不必要
的全量 copy。语义清晰优先于提前复用 buffer，但任何额外开销都需要可以定位，并在 profiling 后在
builder 内部优化，而不是让模型重新感知 Graph。

## 11. 为什么不直接使用钩子函数

执行上下文机制在调用形式上类似钩子，但它不是任意 callback 系统。builder 的生命周期接口是
固定且受约束的，返回值也必须以强类型 context 进入 `ForwardContext`。

这种限制有三点价值：

- runner 可以统一编排所有功能，而不允许 builder 任意改变主执行流程；
- 模型依赖的是明确的数据契约，而不是隐藏副作用；
- Graph Mode 的 allocate-once、update-in-place 语义可以被集中测试。

## 12. 后续工作

后续演进应按以下顺序推进：

1. 完成 Qwen3.5 eager prefill、eager decode 和 ACL Graph decode 的数值对齐，并覆盖 TP2/DP2/EP4
   Token Owner MegaMoe 的端到端数据集验证。
2. 补充 persistent metadata 的地址稳定性、跨 step 内容更新、不同 graph bucket 隔离和 dummy rank
   测试。
3. 基于 profiling 检查 MegaMoe padding、mask 更新和 GDN metadata 构造开销；优先复用上游视图，
   只有固定地址确实要求时才 copy。
4. 需要 speculative execution 时，先由上游稳定提供 request 数和每请求执行宽度，再扩展
   `InputBatch` 与对应 builder；不能从展开后的 Tensor 行数反推。
5. 以相同范式逐步迁移 Full Attention、其他 linear attention 和其他模型。迁移完成前保留公共
   `AttentionMetadata` 与旧兼容路径。
6. 当所有消费者都切换到 typed metadata，并完成回归验证后，再删除旧 mask 字段和 runner 中的
   过渡逻辑。
7. 最后对照 vLLM Ascend 的模型可见输入契约迁移 Qwen3.5 模型实现。目标是复用模型数学组图，而
   不是强行让两个框架的 scheduler、runner 和类层次完全相同。

当前最重要的下一步不是继续扩大 builder 接口，而是用真实模型证明：在介质化输入边界之后，xLLM
与 vLLM Ascend 能够提供等价的模型可见信息，并且模型代码不需要理解 xLLM 的调度和 Graph 生命周期。

## 13. 长期维护价值

执行上下文机制建立了一条可复用的扩展路径：功能通过 metadata builder 接入 eager 和 Graph
执行器，而不是继续向模型、runner 或公共 metadata 添加特例。

它带来的长期收益包括：

- eager 和 Graph Mode 复用同一份模型数学代码，更容易保证数值一致；
- 外部执行输入的所有权、更新频率和复用策略有唯一落点，便于 review 和单元测试；
- runner 保持执行编排职责，不会逐渐演变为模型分支集合；
- 新模型和新功能可以独立提供 typed metadata，不需要修改其他功能；
- 新 graph backend 只需遵守统一生命周期，而不需要重新理解模型细节；
- 固定地址和原地更新策略可以发生在 builder 内部，不破坏模型接口和数学语义。

最终目标不是让系统所有层都不知道 Graph Mode，而是让 Graph Mode 只被应该感知它的执行层
感知：runner 管理时机，metadata builder 管理模型输入及持久化地址，模型保持纯计算。
