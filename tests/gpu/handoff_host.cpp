// Runs a self-checking handoff probe (crates/amdgpu-kernels/src/handoff_probe.rs): one wave, out = got XOR expected per (lane, register); all zero = OK.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: handoff_host <co> <kernel> [values]\n"); return 2; }
    int nv = argc > 3 ? std::atoi(argv[3]) : 16;
    hipModule_t mod; hipFunction_t fn;
    if (hipModuleLoad(&mod, argv[1]) != hipSuccess || hipModuleGetFunction(&fn, mod, argv[2]) != hipSuccess) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::vector<unsigned> h(64 * nv, 0xdeadbeef);
    void* d; hipMalloc(&d, h.size() * 4); hipMemcpy(d, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    void* params[] = {&d};
    hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr);
    if (hipDeviceSynchronize() != hipSuccess) { std::fprintf(stderr, "launch failed\n"); return 1; }
    hipMemcpy(h.data(), d, h.size() * 4, hipMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < 64 * nv; ++i) if (h[i]) { if (bad < 5) std::fprintf(stderr, "lane %d reg %d: xor 0x%x\n", i / nv, i % nv, h[i]); ++bad; }
    std::printf("%s: %s (%d bad of %d)\n", argv[2], bad ? "FAIL" : "OK", bad, 64 * nv);
    return bad ? 1 : 0;
}
