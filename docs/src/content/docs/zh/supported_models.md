---
title: "模型支持列表"
sidebar:
  order: 20
---
表中的 ✅ 表示已接入对应硬件后端，不代表所有模型尺寸、量化格式和推理特性均受支持。部署限制请参阅对应硬件和特性文档。

## LLM
|                          |  NPU  |  MLU  |  ILU  | 海光 DCU |
| ------------------------ | :---: | :---: | :---: | :---: |
| DeepSeek-V2              |   ✅   |   ✅   |   ❌   |   ✅   |
| DeepSeek-V3/R1/V3.1      |   ✅   |   ✅   |   ❌   |   ❌   |
| DeepSeek-V3.2            |   ✅   |   ✅   |   ❌   |   ❌   |
| DeepSeek-V4              |   ✅   |   ✅   |   ❌   |   ❌   |
| DeepSeek-R1-Distill-Qwen |   ✅   |   ❌   |   ❌   |   ❌   |
| Qwen2/2.5/QwQ            |   ✅   |   ✅   |   ✅   |   ✅   |
| Qwen3                    |   ✅   |   ✅   |   ✅   |   ✅   |
| Qwen3 Moe                |   ✅   |   ✅   |   ✅   |   ✅   |
| Qwen3.5                  |   ✅   |   ✅   |   ❌   |   ✅   |
| Qwen3.5-MoE              |   ✅   |   ✅   |   ❌   |   ✅   |
| Kimi-k2                  |   ✅   |   ❌   |   ❌   |   ❌   |
| Llama2/3                 |   ✅   |   ❌   |   ✅   |   ❌   |
| GLM4.5                   |   ✅   |   ❌   |   ❌   |   ❌   |
| GLM4.6                   |   ✅   |   ❌   |   ❌   |   ❌   |
| GLM-4.7                  |   ✅   |   ❌   |   ❌   |   ❌   |
| GLM-5/5.1/5.2            |   ✅   |   ✅   |   ❌   |   ❌   |
| GLM-5.3-Flash            |   ✅   |   ✅   |   ❌   |   ❌   |
| JoyAI-LLM-Flash          |   ✅   |   ✅   |   ❌   |   ❌   |
| Oxygen (text)            |   ✅   |   ✅   |   ❌   |   ❌   |

MLU 上的 GLM-5.3-Flash 对应 `glm5_next`，当前支持文本生成；此处的 ✅ 不包含其多模态变体。Qwen3.5 和 Qwen3.5-MoE 同时支持文本生成和多模态推理，因此分别列在 LLM 和 VLM 表中。

## VLM
|              |  NPU  |  MLU  |  ILU  | 海光 DCU |
| ------------ | :---: | :---: | :---: | :---: |
| MiniCPM-V    |   ✅   |   ❌   |   ❌   |   ❌   |
| MiMo-VL      |   ✅   |   ❌   |   ❌   |   ❌   |
| Qwen2-VL    |   ✅   |   ✅   |   ❌   |   ✅   |
| Qwen2.5-VL   |   ✅   |   ✅   |   ❌   |   ✅   |
| Qwen3-VL     |   ✅   |   ✅   |   ❌   |   ✅   |
| Qwen3-VL-MoE |   ✅   |   ✅   |   ❌   |   ✅   |
| Qwen3.5     |   ✅   |   ✅   |   ❌   |   ✅   |
| Qwen3.5-MoE |   ✅   |   ✅   |   ❌   |   ✅   |
| Oxygen      |   ✅   |   ✅   |   ❌   |   ❌   |
| GLM-4.6V     |   ✅   |   ❌   |   ❌   |   ❌   |
| VLM-R1       |   ✅   |   ❌   |   ❌   |   ❌   |

## Rerank
|                |  NPU  |  MLU  |  ILU  | 海光 DCU |
| -------------- | :---: | :---: | :---: | :---: |
| Qwen3-Reranker |   ✅   |   ❌   |   ❌   |   ❌   |

## DiT
|      |  NPU  |  MLU  |  ILU  | 海光 DCU |
| ---- | :---: | :---: | :---: | :---: |
| Flux |   ✅   |   ✅   |   ❌   |   ❌   |

MLU 已接入 Flux、Flux Control 和 Flux Fill，分别对应 `flux`、`flux_control` 和 `fluxfill`；不包含 Flux2。

## Rec
|     |  NPU  |  MLU  |  ILU  | 海光 DCU |
| --- | :---: | :---: | :---: | :---: |
| OneRec  |   ✅   |   ❌   |   ❌   |   ❌   |
| Qwen2   |   ✅   |   ❌   |   ❌   |   ❌   |
| Qwen2.5 |   ✅   |   ❌   |   ❌   |   ❌   |
| Qwen3   |   ✅   |   ❌   |   ❌   |   ❌   |
