// Minimal host for the raw-asm hazard probes: loads one .co, runs function <name> with 64 lanes and one pointer argument, prints
// the first three dwords (as f32) of every lane's 16-byte row.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) { std::fprintf(stderr, "usage: %s file.co kernel\n", argv[0]); return 2; }
    std::vector<float> hout(64 * 4, -12345.0f);
    void* dout = nullptr;
    CK(hipMalloc(&dout, hout.size() * 4));
    CK(hipMemcpy(dout, hout.data(), hout.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, argv[1]));
    CK(hipModuleGetFunction(&fn, mod, argv[2]));
    void* a_o = dout;
    void* params[] = {&a_o};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(hout.data(), dout, hout.size() * 4, hipMemcpyDeviceToHost));
    std::printf("KERNEL %s:", argv[2]);
    for (int l = 0; l < 64; ++l) {
        if (l < 4 || l == 63) std::printf(" lane%d=[%.6g %.6g %.6g]", l, hout[l * 4], hout[l * 4 + 1], hout[l * 4 + 2]);
    }
    // compact verdict: first dword histogram
    int nz = 0; for (int l = 0; l < 64; ++l) if (hout[l * 4] != 0.0f) ++nz;
    std::printf(" nonzero_first_dword=%d/64\n", nz);
    return 0;
}
