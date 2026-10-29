// Global -> LDS DMA probe host: expects o[0..256] == i[0..256] (16-byte form) and o[256..320] == i[256..320] (4-byte form).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_dma.co";
    std::vector<unsigned> hi(1024), ho(1024, 0xdeadbeef);
    for (unsigned i = 0; i < 1024; ++i) hi[i] = 0x1000 + i;
    void *di, *dout;
    CK(hipMalloc(&di, 4096)); CK(hipMalloc(&dout, 4096));
    CK(hipMemcpy(di, hi.data(), 4096, hipMemcpyHostToDevice));
    CK(hipMemcpy(dout, ho.data(), 4096, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_dma"));
    void* a_o = dout; void* a_i = di; void* params[] = {&a_o, &a_i};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(ho.data(), dout, 4096, hipMemcpyDeviceToHost));
    int bad16 = 0, bad4 = 0;
    for (int i = 0; i < 256; ++i) if (ho[i] != hi[i]) { if (bad16++ < 8) std::printf("b128 mismatch dword %d: got 0x%x want 0x%x\n", i, ho[i], hi[i]); }
    for (int i = 0; i < 64; ++i) if (ho[256 + i] != hi[256 + i]) { if (bad4++ < 8) std::printf("dword mismatch lane %d: got 0x%x want 0x%x\n", i, ho[256 + i], hi[256 + i]); }
    std::printf("DMA_PROBE %s (b128 bad=%d, b32 bad=%d)\n", (bad16 || bad4) ? "FAIL" : "OK", bad16, bad4);
    return (bad16 || bad4) ? 1 : 0;
}