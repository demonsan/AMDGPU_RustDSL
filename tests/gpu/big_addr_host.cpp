// probe_big_addr host (docs/op-coverage-tileops.md B1): a 9 GiB buffer, the kernel writes / reads windows just below / above 4 GiB and 8 GiB
// through moved descriptors (scalar ptr_mad) and 64-bit vector pointers and reports per-lane pass flags; the host also copies the four windows
// back (small copies) and checks both patterns.
//   big_addr_host <kernel.co>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <co>\n", argv[0]);
        return 2;
    }
    const size_t big_bytes = (size_t)9 << 30;
    const uint64_t win[4] = {0xffffffull * 256 + 0x80, 0x1000000ull * 256 + 0x80, 0x1ffffffull * 256 + 0x80, 0x2000000ull * 256 + 0x80};
    void *dbig = nullptr, *dres = nullptr;
    CK(hipMalloc(&dbig, big_bytes));
    CK(hipMalloc(&dres, 8 * 256));
    CK(hipMemset(dres, 0, 8 * 256));
    for (int k = 0; k < 4; ++k) CK(hipMemset((char*)dbig + win[k] - 512, 0, 2048));
    hipModule_t mod;
    hipFunction_t fn;
    CK(hipModuleLoad(&mod, argv[1]));
    CK(hipModuleGetFunction(&fn, mod, "probe_big_addr"));
    struct {
        void* big;
        void* res;
    } args{dbig, dres};
    size_t sz = sizeof(args);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipDeviceSynchronize());
    std::vector<uint32_t> res(8 * 64);
    CK(hipMemcpy(res.data(), dres, 8 * 256, hipMemcpyDeviceToHost));
    int pass = 0;
    for (uint32_t f : res) pass += f == 1;
    for (int g = 0; g < 8; ++g) {
        int c = 0;
        for (int l = 0; l < 64; ++l) c += res[g * 64 + l] == 1;
        std::printf("  window %d %s: %d / 64\n", g / 2, g % 2 ? "B (vector write, descriptor read)" : "A (descriptor write, vector read)", c);
    }
    int host_ok = 0;
    for (int k = 0; k < 4; ++k) {
        std::vector<uint32_t> w(128);
        CK(hipMemcpy(w.data(), (char*)dbig + win[k], 512, hipMemcpyDeviceToHost));
        int a = 0, bb = 0;
        for (int l = 0; l < 64; ++l) {
            a += w[l] == (0xA0000000u | (uint32_t)k << 16 | (uint32_t)l);
            bb += w[64 + l] == (0xB0000000u | (uint32_t)k << 16 | (uint32_t)l);
        }
        host_ok += a + bb;
        std::printf("  host window %d: A %d / 64, B %d / 64 (first words %08x %08x)\n", k, a, bb, w[0], w[64]);
    }
    std::printf("probe_big_addr: on-GPU read-back passes %d / 512, host window check %d / 512 (windows at 4 GiB - 128, 4 GiB + 128, 8 GiB - 128, 8 GiB + 128)\n", pass, host_ok);
    return pass == 512 && host_ok == 512 ? 0 : 1;
}
