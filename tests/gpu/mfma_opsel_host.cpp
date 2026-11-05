// probe_mfma_scale_opsel: out[64 k + tid] = 64 * scale_a(byte) * scale_b(byte) for (byte_a, byte_b) = (0,0) (1,2) (3,1) (2,3)
// with scale A bytes [127,128,129,126] and scale B bytes [127,126,128,129]: 64, 256, 16, 1024.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_mfma_scale_opsel.co";
    std::vector<float> out(256, -1.0f); void* d;
    CK(hipMalloc(&d, 1024)); CK(hipMemcpy(d, out.data(), 1024, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_mfma_scale_opsel"));
    void* a0 = d; void* params[] = {&a0};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr)); CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), d, 1024, hipMemcpyDeviceToHost));
    const float want[4] = {64.0f, 256.0f, 16.0f, 1024.0f};
    int bad = 0; for (int k = 0; k < 4; ++k) for (int i = 0; i < 64; ++i) bad += (out[64 * k + i] != want[k]);
    std::printf("[mfma_scale_opsel] bad=%d out = %g %g %g %g (want 64 256 16 1024)\n%s\n", bad, out[0], out[64], out[128], out[192], bad ? "MFMA_OPSEL_FAIL" : "MFMA_OPSEL_OK");
    return bad != 0;
}