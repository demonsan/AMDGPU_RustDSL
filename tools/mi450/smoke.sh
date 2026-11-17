# gfx1250 smoke: build the trivial kernel + host with the node's ROCm clang, run on the idle GPU under gpurun (hard timeout, gpu-lock on a08-1). Needs the repo synced ($SRC).
set -e
export ROCM=${ROCM:-/opt/rocm}
O="$W/smoke"; mkdir -p "$O"
$ROCM/bin/amdclang++ -x hip --offload-arch=gfx1250 --cuda-device-only -O1 -o "$O/smoke.co" "$SRC/tests/gpu/smoke_kernel.hip"
$ROCM/bin/amdclang++ -x c++ -O1 -D__HIP_PLATFORM_AMD__ -I$ROCM/include "$SRC/tests/gpu/smoke_host.cpp" -o "$O/smoke_host" -L$ROCM/lib -lamdhip64 -Wl,-rpath,$ROCM/lib
echo "built: $(ls -la $O/smoke.co | awk '{print $5}') bytes code object"
echo "GPU: ${IDLE_GPU:-none}"
gpurun "$O/smoke_host" "$O/smoke.co"
