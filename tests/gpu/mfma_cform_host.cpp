// probe_mfma_cform: out[tid] must be exactly 128.0 (ones x ones = 64 plus the explicit C operand 64).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_mfma_cform.co";
    std::vector<float> out(64, -1.0f); void* d;
    CK(hipMalloc(&d, 256)); CK(hipMemcpy(d, out.data(), 256, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_mfma_cform"));
    void* a0 = d; void* params[] = {&a0};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr)); CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), d, 256, hipMemcpyDeviceToHost));
    int bad = 0; for (int i = 0; i < 64; ++i) bad += (out[i] != 128.0f);
    std::printf("[mfma_cform] bad=%d out[0]=%g\n%s\n", bad, out[0], bad ? "MFMA_CFORM_FAIL" : "MFMA_CFORM_OK");
    return bad != 0;
}