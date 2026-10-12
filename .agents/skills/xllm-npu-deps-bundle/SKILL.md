---
name: xllm-npu-deps-bundle
description: Use when building, rebuilding, or self-testing the unified xLLM NPU deps bundle (xllm-deps-*.run) -- compiling the operator run packages and python wheels declared in docker/npu_docker_depends.yaml, assembling the self-extracting bundle, or installing and verifying it locally. Also covers adding or updating components in the depends config.
---

# xLLM NPU Deps Bundle

Build the unified deps bundle in the current environment — everything runs on the local CANN stack.

## Source of truth

`docker/npu_docker_depends.yaml` owns the component list, build/install/verify
commands. The scripts in this skill are fully data-driven and
contain zero component knowledge: a depends change never requires a skill
change. Dynamic facts (chip flavor, proxy, github acceleration) come from the
environment or CLI at run time.

## Quick start

```bash
bash .agents/skills/xllm-npu-deps-bundle/scripts/build_deps_bundle.sh --soc ascend910_93
```

One bundle per --soc (CANN soc_version), e.g. ascend910_93.

Output: `~/xllm-deps-work/bundle/xllm-deps-{soc}-{arch}.run` (+ `.sha256`).
Install it right where you are: `bash xllm-deps-*.run` (root).

## Workflow: fix an operator, rebuild, reinstall

The bundle supports repeated installation. The normal loop when an operator
misbehaves:

1. fix the operator source (or bump its ref in the depends config)
2. `build_deps_bundle.sh --only <component> --force` (fast single-component rebuild)
3. full `build_deps_bundle.sh` (reuses the other artifacts, reassembles)
4. `bash xllm-deps-*.run` (reinstalls over the previous bundle cleanly:
   vendor dirs are overwritten, load_priority is deduplicated, and
   ASCEND_CUSTOM_OPP_PATH does not accumulate duplicates)

## What the build does (per component, in depends order)

1. fetch source at `ref` (github acceleration via `GH_INSTEAD_OF` if set)
2. clean build: CMake build dirs are always wiped -- incremental state from a
   previous parameter set leaks stale kernels into packages
3. install locally to detect the vendor directory (never hardcoded)
4. run the component's `verify` commands (+ `verify_device` when an NPU is
   attached) and an ABI probe (`ldd -r` reconciled against local symbols)
5. record metadata JSON (repo/ref/commit/file sha256) next to the artifacts

Then `make_bundle.py` assembles everything into one self-extracting run
package (operator run packages + wheels + metadata + env setup), with
private-address and host-path leak scans on the generated text.

## Notes

- pip build requirements from the depends config are NOT installed: their
  loose ranges (e.g. numpy<2) fight the pinned NPU stack; the environment is
  expected to provide python build deps
- artifacts are reused across runs unless `--force`; moving refs print a
  commit-mismatch warning -- use `--force` when freshness matters
- `--only NAME` builds one component (skips assembly); `--jobs N` caps
  compile parallelism (default nproc/2)
- `supported_socs` in the depends config controls which SoC builds include a
  component for; adding a new SoC only extends that list
