// MFMA ceiling probe host (tools/mfma_ceiling.py): launches each kernel with BLOCKS workgroups of 256 threads, reads the per-wave records
// (delta s_memtime, delta s_memrealtime, iterations, wave) and prints cycles/MFMA and the effective clock.
// usage: ITERS=4096 BLOCKS=256 ./mfma_ceiling_host probe.co kernel...      (BLOCKS defaults to the CU count)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)
int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s probe.co kernel...\n", argv[0]); return 2; }
    const int iters = std::getenv("ITERS") ? std::atoi(std::getenv("ITERS")) : 4096;
    const int reps = std::getenv("REPS") ? std::atoi(std::getenv("REPS")) : 6;
    const int mfma_per_iter = std::getenv("MPI") ? std::atoi(std::getenv("MPI")) : 32;   // MFMAs per loop iteration (64 for tools/mfma_block.py)
    const bool trace = std::getenv("TRACE") != nullptr;   // TRACE=1: one line per launch (clock trajectory under sustained load)
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    const int blocks = std::getenv("BLOCKS") ? std::atoi(std::getenv("BLOCKS")) : prop.multiProcessorCount;
    hipModule_t mod; CK(hipModuleLoad(&mod, argv[1]));
    const size_t rec = 16, nwaves = (size_t)blocks * 4;
    void* out; CK(hipMalloc(&out, nwaves * rec));
    std::printf("device %s, %d CUs, %d workgroups x 4 waves, %d iterations x %d MFMAs per wave\n", prop.name, prop.multiProcessorCount, blocks, iters, mfma_per_iter);
    std::printf("%-26s %8s %10s %9s %9s %9s %10s\n", "kernel", "blocks", "cyc/MFMA", "min", "max", "clk GHz", "ns/MFMA");
    for (int k = 2; k < argc; ++k) {
        hipFunction_t f; CK(hipModuleGetFunction(&f, mod, argv[k]));
        struct { void* p; } args = { out };
        size_t sz = sizeof(args);
        void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
        std::vector<double> cyc, ns, clk, lo, hi;
        for (int r = 0; r < reps; ++r) {
            CK(hipMemset(out, 0, nwaves * rec));
            CK(hipModuleLaunchKernel(f, blocks, 1, 1, 256, 1, 1, 0, 0, nullptr, cfg));
            CK(hipDeviceSynchronize());
            std::vector<unsigned> h(nwaves * 4);
            CK(hipMemcpy(h.data(), out, nwaves * rec, hipMemcpyDeviceToHost));
            if (trace) { std::vector<double> gg; for (size_t w = 0; w < nwaves; ++w) { double dt = h[w * 4], drt = h[w * 4 + 1]; if (dt > 0 && drt > 0) gg.push_back(dt / (drt * 10.0)); } if (!gg.empty()) { std::sort(gg.begin(), gg.end()); std::printf("  trace %-24s launch %4d clk %.3f GHz (min %.3f max %.3f)\n", argv[k], r, gg[gg.size() / 2], gg.front(), gg.back()); } }
            if (r == 0) continue;   // warm-up launch
            std::vector<double> c, n, g;
            for (size_t w = 0; w < nwaves; ++w) {
                double dt = h[w * 4], drt = h[w * 4 + 1];
                if (dt <= 0 || drt <= 0) continue;
                c.push_back(dt / ((double)iters * mfma_per_iter));
                n.push_back(drt * 10.0 / ((double)iters * mfma_per_iter));
                g.push_back(dt / (drt * 10.0));
            }
            if (c.empty()) { std::printf("%-26s no records (kernel did not run?)\n", argv[k]); break; }
            std::sort(c.begin(), c.end()); std::sort(n.begin(), n.end()); std::sort(g.begin(), g.end());
            cyc.push_back(c[c.size() / 2]); ns.push_back(n[n.size() / 2]); clk.push_back(g[g.size() / 2]); lo.push_back(c.front()); hi.push_back(c.back());
        }
        if (cyc.empty()) continue;
        std::sort(cyc.begin(), cyc.end()); std::sort(ns.begin(), ns.end()); std::sort(clk.begin(), clk.end());
        std::printf("%-26s %8d %10.2f %9.2f %9.2f %9.3f %10.3f\n", argv[k], blocks, cyc[cyc.size() / 2], *std::min_element(lo.begin(), lo.end()), *std::max_element(hi.begin(), hi.end()), clk[clk.size() / 2], ns[ns.size() / 2]);
    }
    return 0;
}