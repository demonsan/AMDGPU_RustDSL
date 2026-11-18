// Host of the multicast landing-place scan (probes probe_g1250_mcast_scan*): a cluster of 2 workgroups fill 64 KiB of LDS with a marker, wave 0 of each loads its 2048-byte slice of `src` (mask in the
// kernel), and each workgroup dumps its whole LDS to out[position]. This reports where the slices' dwords (0xC0DE0000 + i) appear in each workgroup's LDS: runs of consecutive source dwords.
//
//   g1250_mcast_scan_host <co> <kernel> [expect_cluster=2]
// Hang policy: a polled wait with a wall-clock limit (PROBE_WATCHDOG_S, default 10), exit 124 on expiry.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    const int NW = 16384;   // dwords of LDS dumped per workgroup (64 KiB)
    std::vector<uint32_t> src(1024), out(2 * NW, 0xdeadbeef);
    for (int i = 0; i < 1024; ++i) src[i] = 0xC0DE0000u + i;
    uint32_t *dsrc, *dout;
    HC(hipMalloc(&dsrc, 4096)); HC(hipMalloc(&dout, 2 * NW * 4));
    HC(hipMemcpy(dsrc, src.data(), 4096, hipMemcpyHostToDevice));
    HC(hipMemcpy(dout, out.data(), 2 * NW * 4, hipMemcpyHostToDevice));
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
    HC(hipMemcpy(out.data(), dout, 2 * NW * 4, hipMemcpyDeviceToHost));
    for (int wg = 0; wg < 2; ++wg) {
        printf("WG %d LDS dump (%d dwords):\n", wg, NW);
        int marker = 0, srcd = 0, other = 0;
        int i = 0;
        while (i < NW) {
            const uint32_t v = out[wg * NW + i];
            if (v >= 0xC0DE0000u && v < 0xC0DE0000u + 1024) {   // a run of consecutive source dwords
                int j = i + 1;
                while (j < NW && out[wg * NW + j] == v + (j - i)) ++j;
                printf("  src dwords %u..%u at LDS byte %d (%d dwords)\n", v - 0xC0DE0000u, v - 0xC0DE0000u + (j - i) - 1, i * 4, j - i);
                srcd += j - i;
                i = j;
            } else {
                if ((v & 0xFFFF0000u) == 0x7EED0000u) ++marker; else ++other;
                ++i;
            }
        }
        printf("  marker %d, source %d, other %d\n", marker, srcd, other);
    }
    return 0;
}
