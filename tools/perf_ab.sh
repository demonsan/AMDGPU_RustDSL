#!/usr/bin/env bash
# Same-session performance A/B of two commits over every integrated kernel family (docs/perf-ab.md).
#   bash tools/perf_ab.sh --old <ref> --new <ref|.> [--families a,b] [--only REGEX] [--reps 5] [--floor 0.02] [--k 2] [--list]
# Environment of the GPU node (the same as the other GPU tools): HIP_VISIBLE_DEVICES (0-3 only, default 0), AMD_ROCM_MOUNT / AMD_LLVM_BIN for the container cargo, ROCM_PATH, CLANG, AITER_HSA, REF_CO.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
exec python3 tools/perf_ab.py "$@"