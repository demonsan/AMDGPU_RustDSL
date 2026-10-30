// Host for the count-matched s_barrier probe: launch one 8-wave (512-thread) block with (o, n); print per-wave elapsed
// cycles / error count / timeout flag, the wall time, and the per-wave cycles-per-iteration.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) { std::fprintf(stderr, "usage: %s file.co kernel n\n", argv[0]); return 2; }
    int n = std::atoi(argv[3]);
    std::vector<unsigned> h(8 * 4, 0xdeadbeefu);
    void* d = nullptr;
    CK(hipMalloc(&d, h.size() * 4));
    CK(hipMemcpy(d, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, argv[1]));
    CK(hipModuleGetFunction(&fn, mod, argv[2]));
    void* a = d; int nn = n; void* params[] = {&a, &nn};
    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
    CK(hipEventRecord(ev0));
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 512, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(ev1)); CK(hipEventSynchronize(ev1));
    float ms = 0.f; CK(hipEventElapsedTime(&ms, ev0, ev1));
    CK(hipMemcpy(h.data(), d, h.size() * 4, hipMemcpyDeviceToHost));
    unsigned long long c[8] = {0}; unsigned err[8] = {0}, tf[8] = {0};
    for (int w = 0; w < 8; ++w) {
        c[w] = (unsigned long long)h[w * 4] | ((unsigned long long)h[w * 4 + 1] << 32);
        err[w] = h[w * 4 + 2]; tf[w] = h[w * 4 + 3];
    }
    double a_cyc = (double)c[0], b_cyc = (double)c[4];
    int aerr = (int)(err[0] + err[1] + err[2] + err[3]), berr = (int)(err[4] + err[5] + err[6] + err[7]);
    std::printf("KERNEL %-9s n=%-6d wall=%8.3fms A_cyc=%-10.0f B_cyc=%-10.0f A_cyc/iter=%-8.2f B_cyc/iter=%-8.2f A_err=%d B_err=%d timeout=%d,%d\n",
                argv[2], n, ms, a_cyc, b_cyc, a_cyc / n, b_cyc / n, aerr, berr, tf[0] | tf[1] | tf[2] | tf[3], tf[4] | tf[5] | tf[6] | tf[7]);
    return 0;
}
