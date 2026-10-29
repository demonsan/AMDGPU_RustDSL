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
    CK(hipModuleLoad(&mod, co)); const char* kn = argc > 2 ? argv[2] : "probe_dma_tile"; int nthr = argc > 3 ? atoi(argv[3]) : 256;
    CK(hipModuleGetFunction(&fn, mod, kn));
    CK(hipModuleGetFunction(&fn, mod, kn));
    void* a_o = dout; void* a_i = di; void* params[] = {&a_o, &a_i};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, nthr, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(ho.data(), dout, n * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) if (ho[i] != hi[i]) { if (bad++ < 12) std::printf("mismatch dword %d (byte %d): got 0x%x want 0x%x\n", i, 4 * i, ho[i], hi[i]); }
    for (int r = 0; r < 16; ++r) {
        int nb = 0; unsigned first = 0;
        for (int i = 0; i < 256; ++i) if (ho[r * 256 + i] != hi[r * 256 + i]) { if (!nb) first = ho[r * 256 + i]; ++nb; }
        if (nb) {
            std::printf("run %2d (LDS byte %5d): %3d bad dwords; src dword of LDS dwords 0,1,2,3,4,16,64,255:", r, r * 1024, nb);
            const int pick[8] = {0, 1, 2, 3, 4, 16, 64, 255};
            for (int p : pick) { unsigned v = ho[r * 256 + p]; if (v >= 0x10000 && v < 0x10000 + 4096) std::printf(" %d", (int)(v - 0x10000)); else std::printf(" ?"); }
            std::printf("\n");
        }
    }
    std::printf("DMA_TILE_PROBE %s (bad=%d of %d)\n", bad ? "FAIL" : "OK", bad, n);
    return bad ? 1 : 0;
}