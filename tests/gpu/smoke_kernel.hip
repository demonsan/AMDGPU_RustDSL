// The trivial kernel of the MI450 smoke (tests/gpu/smoke_host.cpp): built with the node's ROCm clang as a bare code object until our own emitter targets gfx1250:
//   amdclang++ -x hip --offload-arch=gfx1250 --cuda-device-only -O1 -o smoke.co tests/gpu/smoke_kernel.hip
#include <hip/hip_runtime.h>

extern "C" __global__ void smoke_iota(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i * 3 + 1;
}
