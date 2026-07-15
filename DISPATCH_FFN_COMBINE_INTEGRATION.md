# dispatch_ffn_combine ATB 集成说明

## 概述
本次集成将 vLLM 使用的 `dispatch_ffn_combine` 算子封装到 xLLM 的 ATB 层，实现单算子融合的 EP dispatch + FFN + combine，对齐 vLLM MC2 mode=1 的通信拓扑。

## 修改文件清单

### 1. 新增文件
- `third_party/xllm_atb_layers/operations/aclnn/ops/dispatch_gmm_combine_decode_operation.h`
  - 定义 `DispatchFfnCombineOperation` 类
  - 定义 `DispatchFfnCombineParam` 参数结构

- `third_party/xllm_atb_layers/operations/aclnn/ops/dispatch_gmm_combine_decode_operation.cpp`
  - 实现 `DispatchFfnCombineOperation`
  - 封装 `aclnnDispatchFFNCombine` 系统算子

### 2. 修改文件
- `third_party/xllm_atb_layers/CMakeLists.txt`
  - 添加 custom_xllm_math include 路径

- `third_party/xllm_atb_layers/operations/fusion/moe/ep/dynamic_ep_moe.h`
  - 添加 `enableDispatchFfnCombine` flag

- `third_party/xllm_atb_layers/operations/fusion/moe/ep/dynamic_ep_moe.cpp`
  - 添加 `CreateDispatchFfnCombineNode()` 函数
  - 在主图构建逻辑添加新分支

- `third_party/xllm_atb_layers/operations/fusion/moe/sparse_moe.h`
  - 添加 `enableDispatchFfnCombine` 参数

- `third_party/xllm_atb_layers/operations/fusion/moe/sparse_moe.cpp`
  - 传递参数到 dynamic_ep_moe
  - 注册 `out_expert_token_nums` 输出张量

- `third_party/xllm_atb_layers/models/deepseekv2/layer/decoder_layer.cpp`
  - 设置触发条件：`!isPrefill && isDynamicEp && enableAllToAllMC2`
  - 添加 debug 日志

## 算子接口

### 输入 (7个张量)
1. **x**: `[bs, h]` - 输入隐藏状态
2. **weight1**: `[num_local_experts, h, n*2]` - gate/up 权重 (TensorList)
3. **weight2**: `[num_local_experts, n, h]` - down 权重 (TensorList)
4. **expert_ids**: `[bs]` - 专家路由索引
5. **scale1**: `[num_local_experts, n*2]` - gate/up 量化缩放 (TensorList)
6. **scale2**: `[num_local_experts, h]` - down 量化缩放 (TensorList)
7. **probs**: `[bs]` - 路由概率

### 输出 (2个张量)
1. **output**: `[bs, h]` - 输出隐藏状态
2. **expert_token_nums**: `[num_local_experts]` - 每个专家处理的token数

### 参数
- `epRankSize`: EP 并行度
- `epRankId`: 当前 EP rank ID
- `maxOutputSize`: 最大输出大小 (0表示自动推断)
- `swigluLimit`: SwiGLU 饱和限制 (0.0表示无限制)
- `localMoeExpertNum`: 本地专家数量
- `epCommName`: EP 通信域名称

## 触发条件

在 `decoder_layer.cpp` 中，decode 阶段满足以下条件时启用：
```cpp
enableDispatchGmmCombineDecode = !isPrefill && isDynamicEp && enableAllToAllMC2
```

其中：
- `!isPrefill`: decode 阶段
- `isDynamicEp`: 启用动态 EP
- `enableAllToAllMC2`: EP 并行度 = 2 (对应 vLLM 的 MC2)

## 对齐 vLLM

### vLLM 配置
```bash
VLLM_ASCEND_ENABLE_FUSED_MC2=1  # mode 1: dispatch_ffn_combine
```

### xLLM 对应逻辑
DeepSeek V2/V3 模型在 `npu_deepseek_v2_decoder_layer_impl.cpp` 中设置：
```cpp
param.enableAllToAllMC2 = (param.expertParallelDegree == 2);
```

当 EP 并行度 = 2 时，decode 阶段自动启用 `dispatch_ffn_combine`。

## 模型覆盖

- ✅ **DeepSeek V2** - 通过 `NpuDeepseekV2DecoderLayer`
- ✅ **DeepSeek V3** - 继承 DeepSeek V2
- ✅ **Kimi K25** - 底层使用 DeepSeek V2 实现

## 编译步骤

```bash
# 1. 编译 xllm_atb_layers
cd third_party/xllm_atb_layers/build
cmake .. -DCMAKE_BUILD_TYPE=Release
ninja

# 2. 编译 xLLM
cd ../../..
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 验证步骤

### 1. 查看 debug 日志
运行 decode 推理时，观察日志输出：
```
[DEBUG-MOE] layer=X isPrefill=0 enableAllToAllMC2=1 isDynamicEp=1 -> enableDispatchGmmCombineDecode=1
```

### 2. 性能对比
对比启用前后的 decode 吞吐和延迟：
- 通信开销应该减少（单算子 AllToAll vs 多次通信）
- 整体延迟应该降低

### 3. 精度验证
对比启用前后的输出精度：
```bash
# 运行测试脚本对比输出
python test_moe_accuracy.py --enable_dispatch_ffn_combine
```

## 预期效果

1. **通信优化**: 单算子内完成 EP dispatch + FFN + combine，减少通信次数
2. **性能提升**: 对齐 vLLM 的通信拓扑，decode 吞吐提升
3. **代码简化**: 从原来的 9 节点子图简化为 1 个节点

## 注意事项

1. **权重格式**: 权重必须是 stacked 3D 格式 `[num_local_experts, ...]`，包装为 TensorList
2. **量化模式**: 当前实现假设使用 w8a8 动态量化
3. **可选输入**: `x_active_mask` 暂未支持，传入 nullptr
4. **系统依赖**: 需要 CANN 9.0.0 及以上，包含 custom_xllm_math 算子库

## 已知限制

- 仅支持 decode 阶段（prefill 仍使用原有路径）
- 仅支持 EP=2 配置
- 暂不支持 shared experts

## 后续优化方向

1. 支持 prefill 阶段的融合算子
2. 支持更灵活的 EP 配置（EP > 2）
3. 添加 `x_active_mask` 支持
4. 性能 profiling 和调优
