---
title: "Model Support List"
sidebar:
  order: 20
---
A ✅ indicates that the model is integrated with the corresponding hardware backend. Support for model sizes, quantization formats, and inference features may vary. Refer to the hardware and feature documentation for deployment restrictions.

## LLM
|                          |  NPU  |  MLU  |  ILU  | Hygon DCU |
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

On MLU, GLM-5.3-Flash uses `glm5_next` and currently supports text generation; its ✅ does not include multimodal variants. Qwen3.5 and Qwen3.5-MoE support both text generation and multimodal inference and appear in both the LLM and VLM tables.

## VLM
|              |  NPU  |  MLU  |  ILU  | Hygon DCU |
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
|                |  NPU  |  MLU  |  ILU  | Hygon DCU |
| -------------- | :---: | :---: | :---: | :---: |
| Qwen3-Reranker |   ✅   |   ❌   |   ❌   |   ❌   |

## DiT
|      |  NPU  |  MLU  |  ILU  | Hygon DCU |
| ---- | :---: | :---: | :---: | :---: |
| Flux |   ✅   |   ✅   |   ❌   |   ❌   |

MLU integrates Flux, Flux Control, and Flux Fill under `flux`, `flux_control`, and `fluxfill`, respectively. This does not include Flux2.

## Rec
|     |  NPU  |  MLU  |  ILU  | Hygon DCU |
| --- | :---: | :---: | :---: | :---: |
| OneRec  |   ✅   |   ❌   |   ❌   |   ❌   |
| Qwen2   |   ✅   |   ❌   |   ❌   |   ❌   |
| Qwen2.5 |   ✅   |   ❌   |   ❌   |   ❌   |
| Qwen3   |   ✅   |   ❌   |   ❌   |   ❌   |
