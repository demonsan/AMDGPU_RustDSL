// Host of the multicast probes (crates/amdgpu-kernels/src/probes/g1250.rs, probe_g1250_mcast_*): a cluster of 2 workgroups, each loads ITS 2048-byte slice of `src` with a group-1 multicast mask into
// LDS and copies the whole 4096-byte LDS to out[position]. Reports, per workgroup, which of the two slices it received intact.
//
//   g1250_mcast_host <co> <kernel> [expect_cluster=2]
//
// Hang policy (docs/mi450-nodes.md): a polled wait with a wall-clock limit (PROBE_WATCHDOG_S, default 10), exit 124 on expiry without touching the device again.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); _exit(2); } } while (0)

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <co> <kernel> [expect_cluster]\n", argv[0]); return 1; }
    const int expect = argc > 3 ? atoi(argv[3]) : 2;
    if (expect != 2) { fprintf(stderr, "the probe declares a cluster of 2 workgroups: refused (%d)\n", expect); return 1; }
    double limit = getenv("PROBE_WATCHDOG_S") ? atof(getenv("PROBE_WATCHDOG_S")) : 10.0;
    hipModule_t mod;
    hipFunction_t fn;
    HC(hipModuleLoad(&mod, argv[1]));
    HC(hipModuleGetFunction(&fn, mod, argv[2]));
    std::vector<uint32_t> src(1024), out(2048, 0xdeadbeef);
    for (int i = 0; i < 1024; ++i) src[i] = 0xC0DE0000u + i;
    uint32_t *dsrc, *dout;
    HC(hipMalloc(&dsrc, 4096)); HC(hipMalloc(&dout, 8192));
    HC(hipMemcpy(dsrc, src.data(), 4096, hipMemcpyHostToDevice));
    HC(hipMemcpy(dout, out.data(), 8192, hipMemcpyHostToDevice));
    struct { void *s, *o; } ka{dsrc, dout};
    size_t ksz = sizeof(ka);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ka, HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksz, HIP_LAUNCH_PARAM_END};
    HIP_LAUNCH_CONFIG lc{};
    lc.gridDimX = 2; lc.gridDimY = 1; lc.gridDimZ = 1;
    lc.blockDimX = 64; lc.blockDimY = 1; lc.blockDimZ = 1;
    lc.sharedMemBytes = 0; lc.hStream = nullptr;
    hipLaunchAttribute at{};
    at.id = hipLaunchAttributeClusterDimension;
    at.val.clusterDim.x = 2; at.val.clusterDim.y = 1; at.val.clusterDim.z = 1;
    lc.attrs = &at; lc.numAttrs = 1;
    hipEvent_t e;
    HC(hipEventCreate(&e));
    HC(hipDrvLaunchKernelEx(&lc, fn, nullptr, cfg));
    HC(hipEventRecord(e, nullptr));
    auto t0 = std::chrono::steady_clock::now();
    while (hipEventQuery(e) == hipErrorNotReady) {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s > limit) { printf("MCAST_HANG after %.0f s (exit 124)\n", s); fflush(stdout); _exit(124); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    HC(hipMemcpy(out.data(), dout, 8192, hipMemcpyDeviceToHost));
    int okall = 1;
    for (int wg = 0; wg < 2; ++wg) {
        int match[2] = {0, 0}, stale = 0, zero = 0, other = 0;
        for (int i = 0; i < 1024; ++i) {
            const uint32_t g = out[wg * 1024 + i];
            const int sl = i / 512;
            if (g == src[i]) match[sl]++;
            else if (g == 0xdeadbeef) stale++;
            else if (g == 0) zero++;
            else other++;
        }
        printf("WG %d: slice0 %3d/512 intact, slice1 %3d/512 intact, untouched(0xdeadbeef) %d, zero %d, other %d\n", wg, match[0], match[1], stale, zero, other);
        if (match[0] != 512 || match[1] != 512) okall = 0;
    }
    // where do the "other" dwords come from? first few of workgroup 0 / 1
    for (int wg = 0; wg < 2; ++wg)
        for (int sl = 0; sl < 2; ++sl) {
            const uint32_t g = out[wg * 1024 + sl * 512];
            printf("  WG %d slice %d first dword: 0x%08x (expect 0x%08x)\n", wg, sl, g, src[sl * 512]);
        }
    printf("{\"host\":\"g1250_mcast\",\"kernel\":\"%s\",\"both_slices_in_both\":%s}\n", argv[2], okall ? "true" : "false");
    return okall ? 0 : 3;
}
