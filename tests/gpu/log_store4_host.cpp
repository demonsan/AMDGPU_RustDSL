// probe_log_store4: out[4*tid+j] = log2(in[tid]) (v_log_f32 + global_store_dwordx4)
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_log_store4.co";
    const int n = 64;
    std::vector<float> in(n), out(4 * n, -1.0f);
    for (int i = 0; i < n; ++i) in[i] = 0.25f + 0.37f * i;
    void *din = nullptr, *dout = nullptr;
    CK(hipMalloc(&din, n * 4)); CK(hipMalloc(&dout, out.size() * 4));
    CK(hipMemcpy(din, in.data(), n * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dout, out.data(), out.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_log_store4"));
    void* a0 = din; void* a1 = dout; void* params[] = {&a0, &a1};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    int bad = 0; double worst = 0;
    for (int t = 0; t < n; ++t) for (int j = 0; j < 4; ++j) {
        double want = std::log2((double)in[t]); double e = std::fabs(out[4*t+j] - want);
        if (e > worst) worst = e;
        if (e > 2e-6 * (1.0 + std::fabs(want)) || out[4*t+j] != out[4*t]) ++bad;
    }
    std::printf("[log_store4] bad=%d worst_abs_err=%.3g out[0..3]=%g %g %g %g\n", bad, worst, out[0], out[1], out[2], out[3]);
    std::printf(bad == 0 ? "LOG_STORE4_OK\n" : "LOG_STORE4_FAIL\n");
    return bad != 0;
}