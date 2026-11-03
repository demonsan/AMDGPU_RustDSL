// MFMA issue-interval probe host (tools/mfma_interval.py): launches every kernel of the code object (1 block of 256 threads per CU), reports ns and cycles per MFMA per wave.
// usage: ITERS=4096 ./mfma_interval_host probe.co kernel...
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)
int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s probe.co kernel...\n", argv[0]); return 2; }
    const int iters = std::getenv("ITERS") ? std::atoi(std::getenv("ITERS")) : 4096;
    const int mfma_per_iter = 32;
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    int cus = prop.multiProcessorCount;
    hipModule_t mod; CK(hipModuleLoad(&mod, argv[1]));
    void* out; CK(hipMalloc(&out, 4096));
    double clk_ghz = std::getenv("CLK_GHZ") ? std::atof(std::getenv("CLK_GHZ")) : prop.clockRate / 1e6;
    std::printf("device %s, %d CUs, assumed clock %.2f GHz, %d iterations x %d MFMAs per wave\n", prop.name, cus, clk_ghz, iters, mfma_per_iter);
    for (int k = 2; k < argc; ++k) {
        hipFunction_t f; CK(hipModuleGetFunction(&f, mod, argv[k]));
        struct { void* p; } args = { out };
        size_t sz = sizeof(args);
        void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
        hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
        std::vector<float> ms;
        for (int r = 0; r < 12; ++r) {
            CK(hipEventRecord(a));
            CK(hipModuleLaunchKernel(f, cus, 1, 1, 256, 1, 1, 0, 0, nullptr, cfg));
            CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
            float t; CK(hipEventElapsedTime(&t, a, b)); if (r >= 2) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        double med = ms[ms.size() / 2];
        double ns = med * 1e6 / ((double)iters * mfma_per_iter);
        std::printf("%-22s median %.4f ms  %.3f ns/MFMA/wave = %.2f cycles @%.2fGHz  (min %.4f max %.4f ms)\n", argv[k], med, ns, ns * clk_ghz, clk_ghz, ms.front(), ms.back());
    }
    return 0;
}