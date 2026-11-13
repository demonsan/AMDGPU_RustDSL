#!/usr/bin/env bash
# Hardware probe of exact authored mode (docs/exact-mode.md): a fully pinned kernel compiled verbatim runs and computes out[i] = 2 i + 7.
# Usage: HIP_VISIBLE_DEVICES=2 bash tools/exact_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/exact
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
[ -x build/tileop_probe_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
bash tools/container/cargo.sh test -q --release -p amdgpu-kernels --test exact_mode the_hardware_probe_kernel_compiles_exactly > build/exact/test.log 2>&1 || { tail -20 build/exact/test.log; exit 1; }
if [ -n "${AMDC:-}" ]; then W="$REPO"; else W=/work; fi
${AMDC:-bash tools/container/cargo.sh run -q --release --bin amdc --} emit --from-ir "$W/build/exact/probe_exact_hw.ir" --backend own --out "$W/build/exact/probe.s" > build/exact/emit.log 2>&1 || { tail -5 build/exact/emit.log; exit 1; }
grep -E "^\s+(s_|v_|buffer_)" build/exact/probe.s
"$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/exact/probe.co build/exact/probe.s || exit 1
head -c 256 /dev/zero > build/exact/in.bin
./build/tileop_probe_host build/exact/probe.co probe_exact_hw 64 build/exact/in.bin 256 build/exact/out.bin > /dev/null || exit 1
python3 - <<'EOF'
import struct
d = open("build/exact/out.bin", "rb").read()
v = struct.unpack("<64I", d[:256])
bad = [i for i in range(64) if v[i] != 2 * i + 7]
print("exact probe:", "PASS" if not bad else f"FAIL at {bad[:8]} (got {[v[i] for i in bad[:8]]})")
raise SystemExit(1 if bad else 0)
EOF