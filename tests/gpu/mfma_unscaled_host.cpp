// probe_mfma_unscaled: out[0..63] = unscaled (zero-C then in place) = 128, out[64..127] = unscaled separate-C = 128, out[128..191] = scaled(127/127) reference = 128,
// out[192..255] = the zero-C unscaled result = 64. All must be exact.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_mfma_unscaled.co";
    std::vector<float> out(256, -1.0f); void* d;
    CK(hipMalloc(&d, 1024)); CK(hipMemcpy(d, out.data(), 1024, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_mfma_unscaled"));
    void* a0 = d; void* params[] = {&a0};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr)); CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), d, 1024, hipMemcpyDeviceToHost));
    int bad = 0; for (int i = 0; i < 192; ++i) bad += (out[i] != 128.0f); for (int i = 192; i < 256; ++i) bad += (out[i] != 64.0f);
    std::printf("[mfma_unscaled] bad=%d out[0]=%g out[64]=%g out[128]=%g out[192]=%g\n%s\n", bad, out[0], out[64], out[128], out[192], bad ? "MFMA_UNSCALED_FAIL" : "MFMA_UNSCALED_OK");
    return bad != 0;
}