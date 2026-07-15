# dispatch_ffn_combine 集成快速参考

## 🎯 核心改动

### 新增文件 (2个)
1. `third_party/xllm_atb_layers/operations/aclnn/ops/dispatch_gmm_combine_decode_operation.h`
2. `third_party/xllm_atb_layers/operations/aclnn/ops/dispatch_gmm_combine_decode_operation.cpp`

### 修改文件 (6个)
1. `third_party/xllm_atb_layers/CMakeLists.txt` - 添加 custom_xllm_math include 路径
2. `third_party/xllm_atb_layers/operations/fusion/moe/ep/dynamic_ep_moe.h` - 添加 flag
3. `third_party/xllm_atb_layers/operations/fusion/moe/ep/dynamic_ep_moe.cpp` - 添加节点创建函数
4. `third_party/xllm_atb_layers/operations/fusion/moe/sparse_moe.h` - 添加参数
5. `third_party/xllm_atb_layers/operations/fusion/moe/sparse_moe.cpp` - 参数传递 + tensor map
6. `third_party/xllm_atb_layers/models/deepseekv2/layer/decoder_layer.cpp` - 设置触发条件

## 🔑 关键参数

### 触发条件（decoder_layer.cpp）
```cpp
enableDispatchFfnCombine = !isPrefill && isDynamicEp && enableAllToAllMC2
```

### 预期 Debug 输出
```
[DEBUG-MOE] layer=X isPrefill=0 enableAllToAllMC2=1 isDynamicEp=1 -> enableDispatchFfnCombine=1
```

## 📊 算子接口

### 输入 (7个)
- x: `[bs, h]`
- weight1: `[num_local_experts, h, n*2]` (TensorList)
- weight2: `[num_local_experts, n, h]` (TensorList)
- expert_ids: `[bs]`
- scale1: `[num_local_experts, n*2]` (TensorList)
- scale2: `[num_local_experts, h]` (TensorList)
- probs: `[bs]`

### 输出 (2个)
- output: `[bs, h]`
- expert_token_nums: `[num_local_experts]`

## 🐛 已修复问题

### 问题 1: Tensor Map 未注册
- **错误**: `outTensorIds[0]: 4294967295 is invalid`
- **修复**: 在 3 个位置添加 `out_expert_token_nums` 到输出列表

### 问题 2: 变量命名
- **旧名**: `enableDispatchGmmCombineDecode`
- **新名**: `enableDispatchFfnCombine`
- **原因**: 对齐实际使用的 `dispatch_ffn_combine` 算子

## 🚀 编译测试步骤

```bash
# 1. 编译 xllm_atb_layers
cd third_party/xllm_atb_layers/build
ninja

# 2. 编译 xLLM
cd ../../../build
make -j

# 3. 运行测试
# 观察日志中的 [DEBUG-MOE] 输出
```

## ✅ 验证清单

- [ ] 编译无错误
- [ ] Debug 日志显示 `enableDispatchFfnCombine=1`
- [ ] 推理无 ATB 错误
- [ ] 性能与 vLLM 对齐
- [ ] 精度验证通过

## 📌 对齐 vLLM

| vLLM | xLLM |
|------|------|
| `VLLM_ASCEND_ENABLE_FUSED_MC2=1` | `enableDispatchFfnCombine=true` |
| EP degree = 2 | `enableAllToAllMC2=true` |
| decode only | `!isPrefill` |
| `dispatch_ffn_combine` | `aclnnDispatchFFNCombine` |

## 🔧 关键设计决策

1. **单算子融合**: 9 节点 → 1 节点
2. **权重格式**: Stacked 3D tensor 包装为 TensorList
3. **仅 decode**: prefill 继续使用原有路径
4. **系统算子**: 使用已安装的 `aclnnDispatchFFNCombine`
