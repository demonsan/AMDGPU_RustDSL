// DSL path vs safe-asm path of the conversion chain (probe_cvtchain_*): outputs must be bit-identical, and not trivially constant.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static void run(const char* co, const char* name, const std::vector<float>& in, std::vector<uint32_t>& out) {
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, name));
    void *din, *dout;
    CK(hipMalloc(&din, in.size() * 4)); CK(hipMalloc(&dout, out.size() * 4));
    CK(hipMemcpy(din, in.data(), in.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dout, 0xEE, out.size() * 4));
    void* a0 = din; void* a1 = dout; void* params[] = {&a0, &a1};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    CK(hipFree(din)); CK(hipFree(dout));
}
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s dsl.co asm.co\n", argv[0]); return 2; }
    int bad = 0, distinct_ok = 0;
    for (int seed = 0; seed < 50; ++seed) {
        std::vector<float> in(64 * 8);
        for (size_t i = 0; i < in.size(); ++i) in[i] = (float)(((i * 37 + seed * 101) % 211)) / 140.0f - 0.2f;
        std::vector<uint32_t> a(128, 0), b(128, 0);
        run(argv[1], "probe_cvtchain_dsl", in, a);
        run(argv[2], "probe_cvtchain_asm", in, b);
        if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) { ++bad; if (bad < 4) std::fprintf(stderr, "seed %d: outputs differ (first %08x vs %08x)\n", seed, a[0], b[0]); }
        int nz = 0; for (auto v : a) nz += (v != 0 && v != 0xEEEEEEEEu);
        if (nz > 32) ++distinct_ok;
    }
    std::printf("[cvtchain] 50 seeds: %d mismatches, %d seeds with non-trivial output\n", bad, distinct_ok);
    std::printf((bad == 0 && distinct_ok == 50) ? "CVTCHAIN_OK\n" : "CVTCHAIN_FAIL\n");
    return !(bad == 0 && distinct_ok == 50);
}