// DMA address-pattern probe host: input dword k holds k (so a 16-byte unit identifies its source byte offset);
// prints, for every LDS slot, the source byte offset it holds.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_dma_map.co";
    const int n = 1 << 18;  // 1 MB of dwords
    std::vector<unsigned> hi(n), ho(256, 0xdeadbeef);
    for (int i = 0; i < n; ++i) hi[i] = i;
    void *di, *dout;
    CK(hipMalloc(&di, n * 4)); CK(hipMalloc(&dout, 1024));
    CK(hipMemcpy(di, hi.data(), n * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dout, ho.data(), 1024, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_dma_map"));
    void* a_o = dout; void* a_i = di; void* params[] = {&a_o, &a_i};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(ho.data(), dout, 1024, hipMemcpyDeviceToHost));
    for (int u = 0; u < 64; ++u) std::printf("slot %2d <- source byte %7u%s", u, ho[4 * u] * 4, (u % 4 == 3) ? "\n" : "   ");
    return 0;
}