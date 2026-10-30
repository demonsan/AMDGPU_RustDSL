// Memory-order test for the last-arriver pattern (fused KV-split combine): N workgroups write partials with write-through stores, take a ticket with a
// device-scope atomic add, and the last arriver sums all partials with coherent loads. Repeated for many block counts; the result must always be exact.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_last_arriver.co";
    const char* kname = argc > 2 ? argv[2] : "probe_last_arriver";
    const int reps = argc > 3 ? std::atoi(argv[3]) : 100;
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kname));
    const int counts[] = {2, 7, 8, 16, 31, 64, 100, 256, 304, 608, 1216, 2432};
    const int maxn = 2432;
    void *dpart, *dcnt, *dres;
    CK(hipMalloc(&dpart, (size_t)maxn * 64 * 4)); CK(hipMalloc(&dcnt, 64)); CK(hipMalloc(&dres, 64));
    long total_bad = 0, total_runs = 0;
    for (int n : counts) {
        uint32_t want = 0;
        for (int b = 0; b < n; ++b) for (int t = 0; t < 64; ++t) want += (uint32_t)((b + 1) * 1000 + t);
        int bad = 0; uint32_t first_bad = 0;
        for (int r = 0; r < reps; ++r) {
            const int seed = 7919 * (r + 1) + n;
            const uint32_t want_r = want + (uint32_t)seed * (uint32_t)n * 64u;
            CK(hipMemset(dpart, 0, (size_t)n * 64 * 4)); CK(hipMemset(dcnt, 0, 64)); CK(hipMemset(dres, 0, 64));
            CK(hipDeviceSynchronize());
            void* a0 = dpart; void* a1 = dcnt; void* a2 = dres; int a3 = n; int a4 = seed;
            void* params[] = {&a0, &a1, &a2, &a3, &a4};
            CK(hipModuleLaunchKernel(fn, n, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
            CK(hipDeviceSynchronize());
            uint32_t got = 0, cnt = 0;
            CK(hipMemcpy(&got, dres, 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(&cnt, dcnt, 4, hipMemcpyDeviceToHost));
            if (got != want_r || cnt != (uint32_t)n * 64) { if (!bad) first_bad = got; ++bad; }
        }
        std::printf("[last_arriver] blocks=%5d reps=%d bad=%d%s\n", n, reps, bad, bad ? " (first bad result differs)" : "");
        (void)first_bad;
        total_bad += bad; total_runs += reps;
    }
    std::printf("[last_arriver] %s: %ld bad of %ld runs\n", kname, total_bad, total_runs);
    std::printf(total_bad == 0 ? "LAST_ARRIVER_OK\n" : "LAST_ARRIVER_FAIL\n");
    return total_bad != 0;
}