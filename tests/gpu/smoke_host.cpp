// Smoke host for any gfx code object (the MI450 / gfx1250 node check, docs/mi450-nodes.md): loads a code object, launches `out[i] = 3 i + 1` (kernel `smoke_iota(int* out, int n)`,
// kernarg = {ptr, i32 n}), checks the result and prints SMOKE_OK, then times ITERS back-to-back launches and prints one JSON line (the harness format of tools/perf_ab.py).
//
//   smoke_host <co> [kernel=smoke_iota] [n=1048576] [block=256]      env: SMOKE_WATCHDOG_S (default 20): the wall-clock limit of every wait; ITERS (200)
//
// Hang policy (docs/mi450-nodes.md): launches are polled, never waited on with a blocking synchronize. When the limit passes the process prints SMOKE_HANG and leaves with `_exit(124)`
// WITHOUT touching the device again (no reset, no free, no synchronize: those can hang too); the caller (`gpurun`, `timeout -s KILL`) owns the kill, and nobody retries blindly.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); _exit(2); } } while (0)

static double g_limit = 20.0;
static void wait_polled(hipEvent_t ev, const char* what) {
    auto t0 = std::chrono::steady_clock::now();
    while (hipEventQuery(ev) == hipErrorNotReady) {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s > g_limit) { printf("SMOKE_HANG: %s still running after %.0f s; leaving without touching the device (exit 124)\n", what, s); fflush(stdout); _exit(124); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <co> [kernel] [n] [block]\n", argv[0]); return 1; }
    const std::string kn = argc > 2 ? argv[2] : "smoke_iota";
    const int n = argc > 3 ? atoi(argv[3]) : (1 << 20), block = argc > 4 ? atoi(argv[4]) : 256;
    if (const char* w = getenv("SMOKE_WATCHDOG_S")) g_limit = atof(w);
    const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 200;
    hipDeviceProp_t prop;
    HC(hipGetDeviceProperties(&prop, 0));
    printf("device: %s  arch: %s  CUs: %d  wave: %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.warpSize);
    hipModule_t mod;
    hipFunction_t fn;
    HC(hipModuleLoad(&mod, argv[1]));
    HC(hipModuleGetFunction(&fn, mod, kn.c_str()));
    int* d = nullptr;
    HC(hipMalloc(&d, (size_t)n * 4));
    HC(hipMemset(d, 0xff, (size_t)n * 4));
    struct { void* p; int n; } args = { d, n };
    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
    const unsigned grid = (n + block - 1) / block;
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    HC(hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, nullptr, cfg));
    HC(hipEventRecord(e1, nullptr));
    wait_polled(e1, kn.c_str());
    std::vector<int> h(n);
    HC(hipMemcpy(h.data(), d, (size_t)n * 4, hipMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < n; ++i) bad += h[i] != i * 3 + 1;
    printf("kernel %s: n=%d bad=%ld %s\n", kn.c_str(), n, bad, bad == 0 ? "SMOKE_OK" : "SMOKE_FAIL");
    // timing: iters back-to-back launches between two events, polled with the same limit
    HC(hipEventRecord(e0, nullptr));
    for (int i = 0; i < iters; ++i) HC(hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, nullptr, cfg));
    HC(hipEventRecord(e1, nullptr));
    wait_polled(e1, "the timing loop");
    float ms = 0;
    HC(hipEventElapsedTime(&ms, e0, e1));
    const double us = ms * 1000.0 / iters, gbs = (double)n * 4 / (us * 1e-6) / 1e9;
    printf("{\"host\":\"smoke\",\"kernel\":\"%s\",\"shape\":{\"n\":%d,\"block\":%d},\"us\":%.3f,\"gbs\":%.1f,\"bad\":%ld,\"ok\":%s}\n", kn.c_str(), n, block, us, gbs, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}
