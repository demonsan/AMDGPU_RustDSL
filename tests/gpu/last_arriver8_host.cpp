// 8-wave (512-thread) last-arriver memory-order test with exact per-element verification (device-side comparison against hash(i+seed)),
// fresh seed every run, staggered block timing, odd block counts, and an optional concurrent streaming copy on a second stream to thrash L2/MALL.
// usage: last_arriver8_host <kernel.co> <kernel name> <verify.co> <total runs> [thrash 0/1] [stagger iterations]
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s kernel.co name verify.co runs [thrash] [stagger]\n", argv[0]); return 2; }
    const char* kco = argv[1]; const char* kname = argv[2]; const char* vco = argv[3];
    const long total_runs_wanted = std::atol(argv[4]);
    const int thrash = argc > 5 ? std::atoi(argv[5]) : 0;
    const int stagger = argc > 6 ? std::atoi(argv[6]) : 4;
    hipModule_t km, vm; hipFunction_t kf, vf;
    CK(hipModuleLoad(&km, kco)); CK(hipModuleGetFunction(&kf, km, kname));
    CK(hipModuleLoad(&vm, vco)); CK(hipModuleGetFunction(&vf, vm, "probe_la_verify"));
    const int counts[] = {3, 8, 17, 64, 101, 255, 304, 617, 1216, 2432};
    const int ncounts = sizeof(counts) / sizeof(counts[0]);
    const int maxn = 2432;
    const size_t part_bytes = (size_t)maxn * 512 * 4;
    void *dpart, *dout, *dcnt, *dbad;
    CK(hipMalloc(&dpart, part_bytes)); CK(hipMalloc(&dout, part_bytes));
    CK(hipMalloc(&dcnt, 4096)); CK(hipMalloc(&dbad, 4096));   // counter on its own page
    void *ta = nullptr, *tb = nullptr; hipStream_t s2;
    const size_t tbytes = 256u << 20;
    if (thrash) { CK(hipMalloc(&ta, tbytes)); CK(hipMalloc(&tb, tbytes)); CK(hipStreamCreate(&s2)); }
    const long per_count = total_runs_wanted / ncounts;
    long total_bad_runs = 0, total_bad_elems = 0, total_runs = 0, counter_bad = 0;
    for (int ci = 0; ci < ncounts; ++ci) {
        const int n = counts[ci];
        long bad_runs = 0, bad_elems = 0;
        for (long r = 0; r < per_count; ++r) {
            const int seed = (int)(2654435761u * (uint32_t)(r + 1) + 40503u * (uint32_t)n);
            if (thrash && (r % 4) == 0) CK(hipMemcpyAsync(tb, ta, tbytes, hipMemcpyDeviceToDevice, s2));
            CK(hipMemsetAsync(dcnt, 0, 4, nullptr)); CK(hipMemsetAsync(dbad, 0, 4, nullptr));
            CK(hipMemsetAsync(dout, 0xA5, (size_t)n * 512 * 4, nullptr));
            void* a0 = dpart; void* a1 = dcnt; void* a2 = dout; int a3 = n; int a4 = seed; int a5 = stagger;
            void* kp[] = {&a0, &a1, &a2, &a3, &a4, &a5};
            CK(hipModuleLaunchKernel(kf, n, 1, 1, 512, 1, 1, 2048, nullptr, kp, nullptr));
            int tot = n * 512; int vblocks = (tot + 255) / 256;
            void* v0 = dout; void* v1 = dbad; int v2 = tot; int v3 = seed;
            void* vp[] = {&v0, &v1, &v2, &v3};
            CK(hipModuleLaunchKernel(vf, vblocks, 1, 1, 256, 1, 1, 0, nullptr, vp, nullptr));
            CK(hipDeviceSynchronize());
            uint32_t bad = 0, cnt = 0;
            CK(hipMemcpy(&bad, dbad, 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(&cnt, dcnt, 4, hipMemcpyDeviceToHost));
            if (cnt != (uint32_t)n) ++counter_bad;
            if (bad) { ++bad_runs; bad_elems += bad; }
        }
        std::printf("[la8] %s blocks=%5d runs=%ld bad_runs=%ld bad_elems=%ld\n", kname, n, per_count, bad_runs, bad_elems);
        total_bad_runs += bad_runs; total_bad_elems += bad_elems; total_runs += per_count;
    }
    std::printf("[la8] %s: %ld bad runs (%ld bad elements, %ld counter errors) of %ld runs thrash=%d stagger=%d\n", kname, total_bad_runs, total_bad_elems, counter_bad, total_runs, thrash, stagger);
    std::printf((total_bad_runs == 0 && counter_bad == 0) ? "LA8_OK\n" : "LA8_FAIL\n");
    return (total_bad_runs != 0 || counter_bad != 0);
}