// Two-waves-per-SIMD microbenchmark host: g1250_waves_host <co> <kernel> [iters=2000]; env WGS (1) work-groups, PROBE_WATCHDOG_S (20).
// 256 work-items per work-group (8 waves); prints the loop cycles of each wave of work-group 0 and the kernel time.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>
#include <vector>
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); _exit(2); } } while (0)
static void wait_polled(hipEvent_t ev, double lim) {
    auto t0 = std::chrono::steady_clock::now();
    while (hipEventQuery(ev) == hipErrorNotReady) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > lim) { printf("PROBE_HANG\n"); fflush(stdout); _exit(124); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s co kernel [iters]\n", argv[0]); return 1; }
    const int iters = argc > 3 ? atoi(argv[3]) : 2000;
    const int wgs = getenv("WGS") ? atoi(getenv("WGS")) : 1;
    const double lim = getenv("PROBE_WATCHDOG_S") ? atof(getenv("PROBE_WATCHDOG_S")) : 20.0;
    hipModule_t mod; hipFunction_t fn;
    HC(hipModuleLoad(&mod, argv[1])); HC(hipModuleGetFunction(&fn, mod, argv[2]));
    uint32_t* d; HC(hipMalloc(&d, (size_t)wgs * 2048)); HC(hipMemset(d, 0, (size_t)wgs * 2048));
    struct { void* out; int iters; int pad; } ka{d, iters, 0};
    size_t sz = sizeof(ka);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &ka, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
    auto run = [&]() { hipEvent_t e; HC(hipEventCreate(&e)); HC(hipModuleLaunchKernel(fn, wgs, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); HC(hipEventRecord(e, nullptr)); wait_polled(e, lim); };
    run();   // warm-up
    hipEvent_t a, b; HC(hipEventCreate(&a)); HC(hipEventCreate(&b));
    HC(hipEventRecord(a, nullptr)); run(); HC(hipEventRecord(b, nullptr)); HC(hipEventSynchronize(b));
    float ms = 0; HC(hipEventElapsedTime(&ms, a, b));
    std::vector<uint32_t> h((size_t)wgs * 512);
    HC(hipMemcpy(h.data(), d, (size_t)wgs * 2048, hipMemcpyDeviceToHost));
    printf("waves iters=%d wgs=%d: %.2f us cycles/iter per wave:", iters, wgs, ms * 1000.0);
    for (int w = 0; w < 8; ++w) printf(" %.1f", h[w * 64] / (double)iters);
    printf("\n");
    return 0;
}