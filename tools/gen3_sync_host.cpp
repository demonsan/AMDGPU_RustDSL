// Host for the gen3 sync probe: launch one 8-wave (512-thread) block with one output pointer; print per-wave
// (cycles low/high, error count, timeout flag) from the 8 rows of 16 bytes the kernel writes.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) { std::fprintf(stderr, "usage: %s file.co kernel\n", argv[0]); return 2; }
    std::vector<unsigned> h(8 * 4, 0xdeadbeefu);
    void* d = nullptr;
    CK(hipMalloc(&d, h.size() * 4));
    CK(hipMemcpy(d, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, argv[1]));
    CK(hipModuleGetFunction(&fn, mod, argv[2]));
    void* a = d; void* params[] = {&a};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 512, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h.data(), d, h.size() * 4, hipMemcpyDeviceToHost));
    std::printf("KERNEL %-12s", argv[2]);
    unsigned long long a0 = 0, b0 = 0; unsigned ea = 0, eb = 0; int tmo = 0;
    for (int w = 0; w < 8; ++w) {
        unsigned long long c = (unsigned long long)h[w * 4] | ((unsigned long long)h[w * 4 + 1] << 32);
        unsigned err = h[w * 4 + 2], tf = h[w * 4 + 3];
        if (w == 0) a0 = c; if (w == 4) b0 = c;
        if (w < 4) ea += err; else eb += err;
        if (tf) tmo = 1;
        std::printf(" w%d:%llu/%u", w, c, err);
    }
    std::printf(" | A_cycles=%llu B_cycles=%llu A_err=%u B_err=%u timeout=%d\n", a0, b0, ea, eb, tmo);
    return 0;
}
