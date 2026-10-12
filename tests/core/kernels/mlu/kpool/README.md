<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# KPool correctness tests

`kpool_kernel_test` is a C++ GTest target, enabled by the parent MLU CMake
configuration. Run it after the repository MLU build, together with
`glm5_next_kpool_indexer_test` and `glm5_next_decoder_layer_test`.
Tests call the production launchers in `kernels/mlu/kpool.h`; no external
workspace, Python test harness, golden files, or model weights are required.
There are no timing assertions or benchmarks.

The cases migrate the useful boundaries from the validated xllm-kpool workspace
11 (decode ring updates, paged/ragged scores, compression and split expansion).
Fixtures use GLM-5.3-Flash-W4A8 heads=32, dim=128, pool=4 and top-k=2048.
Their independent Torch references express the xLLM contracts: multi-head weighted ReLU
scores, per-query causality, BF16 rounding before Hadamard, token-level physical
addresses, and all incomplete-tail tokens. The standalone workspace's simplified
score/expansion references are not used as the model oracle.

- `update_test.cpp`: independent CPU sequential pooling, shuffled physical pages,
  independent state slots, ragged chunks, ring wrap and speculative overwrites.
- `score_test.cpp`: paged decode/verify scores with request row mapping;
  finite score precision, exact masks, workspace guards, native top-k ties and
  nonfinite/empty inputs.
- `expand_test.cpp`: token/pool budget distinction, every tail remainder,
  cross-block pools, future-token masks and missing pages.

The layer suite owns integration behavior, including rejected drafts, live graph
metadata, page/state relocation, placeholders and the high-scoring-future-pool
regression. The layer suite also checks an independently generated Torch prefill chain and
Graph replay across the 64K capacity boundary. Full-network and performance
acceptance are recorded separately; passing these tests does not imply either.

The 64K chunked chain test uses an independent CPU Torch cache. Cache comparison
keeps `atol=rtol=0.015625`; isolated scores keep `atol=rtol=1e-4`.
The approved chain bound is `|scale| sum_h |w_h| sum_d |Q_hd| |K_d-Kref_d|`,
plus the unchanged score-stage tolerance, using ReLU's 1-Lipschitz property.
TopK discrepancies are accepted only within the corresponding boundary bound.
The original chain `atol=rtol=0.001` violation count remains a GTest XML property
(`strict_chain_score_violations`); it is not silently reclassified as passing.
