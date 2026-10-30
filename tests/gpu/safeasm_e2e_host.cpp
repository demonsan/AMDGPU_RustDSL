// Runs probe_safeasm_e2e (a kernel fragment authored in safe Rust through amdgpu-safeasm): out[tid] = sum(in[4*tid..4*tid+4]) + 64.0
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_safeasm_e2e.co";
    const int n = 64;
    std::vector<float> in(4 * n), out(n, -1.0f);
    for (int i = 0; i < 4 * n; ++i) in[i] = (float)((i * 7) % 23) - 5.0f;
    void *din = nullptr, *dout = nullptr;
    CK(hipMalloc(&din, in.size() * 4)); CK(hipMalloc(&dout, out.size() * 4));
    CK(hipMemcpy(din, in.data(), in.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dout, out.data(), out.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_safeasm_e2e"));
    void* a0 = din; void* a1 = dout; void* params[] = {&a0, &a1};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int t = 0; t < n; ++t) {
        float want = ((in[4*t] + in[4*t+1]) + (in[4*t+2] + in[4*t+3])) + 64.0f;
        if (out[t] != want) { if (bad < 6) std::fprintf(stderr, "mismatch tid=%d got=%g want=%g\n", t, out[t], want); ++bad; }
    }
    std::printf("[safeasm_e2e] bad=%d of %d (out[0]=%g)\n", bad, n, out[0]);
    std::printf(bad == 0 ? "SAFEASM_E2E_OK\n" : "SAFEASM_E2E_FAIL\n");
    return bad != 0;
}