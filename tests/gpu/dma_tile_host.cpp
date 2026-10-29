// Multi-wave global -> LDS DMA tile probe: o (16 KB) must equal i.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_dma_tile.co";
    const int n = 4096;  // dwords
    std::vector<unsigned> hi(n), ho(n, 0xdeadbeef);
    for (int i = 0; i < n; ++i) hi[i] = 0x10000 + i;
    void *di, *dout;
    CK(hipMalloc(&di, n * 4)); CK(hipMalloc(&dout, n * 4));
    CK(hipMemcpy(di, hi.data(), n * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dout, ho.data(), n * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_dma_tile"));
    void* a_o = dout; void* a_i = di; void* params[] = {&a_o, &a_i};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(ho.data(), dout, n * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) if (ho[i] != hi[i]) { if (bad++ < 12) std::printf("mismatch dword %d (byte %d): got 0x%x want 0x%x\n", i, 4 * i, ho[i], hi[i]); }
    std::printf("DMA_TILE_PROBE %s (bad=%d of %d)\n", bad ? "FAIL" : "OK", bad, n);
    return bad ? 1 : 0;
}