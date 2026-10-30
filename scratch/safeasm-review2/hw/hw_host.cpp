// Host for the wait-state probes: loads one .co, runs function <name> with 64 lanes and one pointer argument; each lane
// writes one dword at o + lane*16. Prints the first result dword of lanes 0..3 and 63.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) { std::fprintf(stderr, "usage: %s file.co kernel\n", argv[0]); return 2; }
    std::vector<unsigned> h(64 * 4, 0xdeadbeefu);
    void* d = nullptr;
    CK(hipMalloc(&d, h.size() * 4));
    CK(hipMemcpy(d, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, argv[1]));
    CK(hipModuleGetFunction(&fn, mod, argv[2]));
    void* a = d;
    void* params[] = {&a};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h.data(), d, h.size() * 4, hipMemcpyDeviceToHost));
    std::printf("KERNEL %s:", argv[2]);
    for (int l = 0; l < 64; ++l) if (l < 4 || l == 63) std::printf(" lane%d=%u", l, h[l * 4]);
    std::printf("\n");
    return 0;
}
