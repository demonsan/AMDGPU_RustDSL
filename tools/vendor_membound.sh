#!/usr/bin/env bash
# Build the aiter (vendor) memory-bound harness without torch: the aiter HIP sources are compiled standalone (host + device objects) and linked with
# tests/gpu/vendor_membound_host.cpp, which calls the aiter C++ entry points (activation / quant / rmsnorm_quant) on aiter_tensor_t views of plain HIP buffers.
# prerequisites: an aiter checkout's csrc/{kernels,include,py_itfs_cu} unpacked as $AM/csrc/... (default /tmp/aiter_mb), ROCm with amdclang++ (default rocm-7.0.2), a gfx950 GPU for running.
# usage: tools/vendor_membound.sh [outdir]      then: <outdir>/vendor_membound_host <case> <M> <N>   (cases: see the .cpp)
set -e
R=${ROCM:-/shared/apps/ubuntu/opt/rocm-7.0.2}; AM=${AM:-/tmp/aiter_mb}; OUT=${1:-build/vendor}
mkdir -p "$OUT"
F="-x hip -O2 --offload-arch=gfx950 --rocm-path=$R -std=c++17 -I $AM/csrc/include -I $AM/csrc/kernels -I $AM/csrc/include/ck_tile"
for k in activation_kernels quant_kernels rmsnorm_quant_kernels; do
  [ -f "$OUT/$k.o" ] || $R/bin/amdclang++ $F -c $AM/csrc/kernels/$k.cu -o "$OUT/$k.o"
done
$R/bin/amdclang++ $F -c tests/gpu/vendor_membound_host.cpp -o "$OUT/vh.o"
$R/bin/amdclang++ --rocm-path=$R "$OUT/vh.o" "$OUT"/activation_kernels.o "$OUT"/quant_kernels.o "$OUT"/rmsnorm_quant_kernels.o -o "$OUT/vendor_membound_host" -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib
echo "built $OUT/vendor_membound_host"