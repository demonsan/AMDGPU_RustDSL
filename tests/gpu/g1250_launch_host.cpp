// Generic, hang-safe launcher for a gfx1250 code object (replica harness; docs/exact-mode.md). It launches ONE kernel with a kernarg blob and a
// launch spec and refuses anything the spec does not pin down. Rules (MI450 hang hazards; kernel-wiki mi400 async-multicast / barriers pages):
//   - a kernel that needs a cluster (cluster barriers, cluster ids, TDM cluster loads) is launched ONLY with the cluster dims it was built for;
//     the caller (tools/g1250_launch.py) decides that from the code; this host refuses cluster dims that do not divide the grid;
//   - the kernarg blob size must equal the code object's kernarg segment size (passed in by the caller, read from the metadata);
//   - block <= the function's max threads; static + dynamic LDS <= the device limit;
//   - completion is polled against a deadline; on expiry the process exits non-zero immediately (the caller also runs it under
//     `timeout -s KILL`), never waits forever.
//
//   g1250_launch_host <co> <kernel> <kernarg.bin> <kernarg_size> gx gy gz bx by bz cx cy cz <dyn_lds> <deadline_ms> [iters]
//   (cx cy cz = 0 0 0: no cluster launch). Prints `LAUNCH_OK us=<avg>` or `LAUNCH_FAIL <reason>`.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>
#include <unistd.h>

#define FAIL(...) do { std::printf("LAUNCH_FAIL "); std::printf(__VA_ARGS__); std::printf("\n"); std::fflush(stdout); _exit(2); } while (0)
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) FAIL("%s: %s", #x, hipGetErrorString(e_)); } while (0)

int main(int argc, char** argv) {
    if (argc < 16) {
        std::fprintf(stderr, "usage: %s co kernel kernarg.bin kernarg_size gx gy gz bx by bz cx cy cz dyn_lds deadline_ms [iters]\n", argv[0]);
        return 1;
    }
    const char* co = argv[1];
    const char* kn = argv[2];
    std::ifstream kf(argv[3], std::ios::binary);
    std::vector<char> ka((std::istreambuf_iterator<char>(kf)), std::istreambuf_iterator<char>());
    const size_t ka_size = std::strtoul(argv[4], nullptr, 0);
    unsigned g[3], b[3], c[3];
    for (int i = 0; i < 3; i++) {
        g[i] = std::atoi(argv[5 + i]);
        b[i] = std::atoi(argv[8 + i]);
        c[i] = std::atoi(argv[11 + i]);
    }
    const unsigned dyn_lds = std::atoi(argv[14]);
    const long deadline_ms = std::atol(argv[15]);
    const int iters = argc > 16 ? std::atoi(argv[16]) : 1;

    if (ka.size() != ka_size) FAIL("kernarg blob is %zu bytes, the code object declares %zu", ka.size(), ka_size);
    const bool cluster = c[0] | c[1] | c[2];
    if (cluster) {
        for (int i = 0; i < 3; i++)
            if (c[i] == 0 || g[i] % c[i] != 0) FAIL("cluster dim %d = %u does not divide grid %u", i, c[i], g[i]);
    }
    if (deadline_ms <= 0 || deadline_ms > 600000) FAIL("deadline_ms must be in (0, 600000]");

    hipModule_t m;
    hipFunction_t f;
    HC(hipModuleLoad(&m, co));
    HC(hipModuleGetFunction(&f, m, kn));
    int max_threads = 0, static_lds = 0, dev_lds = 0;
    HC(hipFuncGetAttribute(&max_threads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f));
    HC(hipFuncGetAttribute(&static_lds, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f));
    HC(hipDeviceGetAttribute(&dev_lds, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    if ((int)(b[0] * b[1] * b[2]) > max_threads) FAIL("block %u exceeds the function's max %d", b[0] * b[1] * b[2], max_threads);
    if (static_lds + (int)dyn_lds > dev_lds) FAIL("LDS %d + %u exceeds the device limit %d", static_lds, dyn_lds, dev_lds);

    size_t sz = ka.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ka.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    hipStream_t s;
    HC(hipStreamCreate(&s));
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    HC(hipEventRecord(e0, s));
    for (int it = 0; it < iters; it++) {
        if (cluster) {
            HIP_LAUNCH_CONFIG lc{};
            lc.gridDimX = g[0];
            lc.gridDimY = g[1];
            lc.gridDimZ = g[2];
            lc.blockDimX = b[0];
            lc.blockDimY = b[1];
            lc.blockDimZ = b[2];
            lc.sharedMemBytes = dyn_lds;
            lc.hStream = s;
            hipLaunchAttribute at{};
            at.id = hipLaunchAttributeClusterDimension;
            at.val.clusterDim.x = c[0];
            at.val.clusterDim.y = c[1];
            at.val.clusterDim.z = c[2];
            lc.attrs = &at;
            lc.numAttrs = 1;
            HC(hipDrvLaunchKernelEx(&lc, f, nullptr, cfg));
        } else {
            HC(hipModuleLaunchKernel(f, g[0], g[1], g[2], b[0], b[1], b[2], dyn_lds, s, nullptr, cfg));
        }
    }
    HC(hipEventRecord(e1, s));
    // bounded wait: poll, never block
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        hipError_t q = hipEventQuery(e1);
        if (q == hipSuccess) break;
        if (q != hipErrorNotReady) FAIL("kernel error: %s", hipGetErrorString(q));
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(deadline_ms)) FAIL("deadline of %ld ms exceeded (possible hang)", deadline_ms);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    float ms = 0;
    HC(hipEventElapsedTime(&ms, e0, e1));
    std::printf("LAUNCH_OK us=%.3f\n", 1e3 * ms / iters);
    return 0;
}
