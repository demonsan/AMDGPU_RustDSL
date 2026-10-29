// DMA + K-fragment read probe host: o[t*64 + k*16 ..+16] must equal i[row*128 + (g ^ 2k)*16 ..+16], row = (t & 31) + 32 (t >> 6), g = (t >> 5) & 1.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_dma_kread.co";
    std::vector<unsigned char> hi(16384), ho(16384, 0xee);
    for (int i = 0; i < 16384; ++i) hi[i] = (unsigned char)(i * 7 + (i >> 7));
    void *di, *dout;
    CK(hipMalloc(&di, 16384)); CK(hipMalloc(&dout, 16384));
    CK(hipMemcpy(di, hi.data(), 16384, hipMemcpyHostToDevice));
    CK(hipMemcpy(dout, ho.data(), 16384, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_dma_kread"));
    void* a_o = dout; void* a_i = di; void* params[] = {&a_o, &a_i};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(ho.data(), dout, 16384, hipMemcpyDeviceToHost));
    int bad = 0; int badrow[128] = {0};
    for (int t = 0; t < 256; ++t) {
        int row = (t & 31) + 32 * (t >> 6), g = (t >> 5) & 1;
        for (int k = 0; k < 4; ++k) {
            const unsigned char* want = &hi[row * 128 + (g ^ (2 * k)) * 16];
            const unsigned char* got = &ho[t * 64 + k * 16];
            if (std::memcmp(want, got, 16)) { ++bad; ++badrow[row]; if (bad <= 12) { int fr = -1, fc = -1; for (int r2 = 0; r2 < 128 && fr < 0; ++r2) for (int c2 = 0; c2 < 8; ++c2) if (!std::memcmp(got, &hi[r2 * 128 + c2 * 16], 16)) { fr = r2; fc = c2; break; } std::printf("t %3d row %3d g %d k %d: expected (row %d chunk %d) got source (row %d chunk %d)\n", t, row, g, k, row, g ^ (2 * k), fr, fc); } }
        }
    }
    for (int r = 0; r < 128; r += 8) { int s = 0; for (int i = 0; i < 8; ++i) s += badrow[r + i]; if (s) std::printf("rows %3d..%3d (LDS run %2d): %d bad fragments\n", r, r + 7, r / 8, s); }
    std::printf("DMA_KREAD_PROBE %s (bad=%d of 1024 fragments)\n", bad ? "FAIL" : "OK", bad);
    return bad ? 1 : 0;
}