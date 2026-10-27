// Minimal ABI probe: verifies that kernarg pointers reach the kernel intact.
#include <hip/hip_runtime.h>

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
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* co = argc > 1 ? argv[1] : "build/probe_store.co";
    const int n = 64;

    std::vector<unsigned short> hq(n);
    for (int i = 0; i < n; ++i) {
        hq[i] = (unsigned short)(0x3c00 + i);  // f16 1.0 + i
    }
    std::vector<unsigned char> ho(256, 0);

    void* dq = nullptr;
    void* dout = nullptr;
    CK(hipMalloc(&dq, n * 2));
    CK(hipMalloc(&dout, ho.size()));
    CK(hipMemcpy(dq, hq.data(), n * 2, hipMemcpyHostToDevice));
    CK(hipMemset(dout, 0, ho.size()));

    hipModule_t mod;
    hipFunction_t fn;
    CK(hipModuleLoad(&mod, co));
    CK(hipModuleGetFunction(&fn, mod, "probe_store"));

    void* a_o = dout;
    void* a_q = dq;
    int a_n = n;
    void* params[] = {&a_o, &a_q, &a_n};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    std::printf("[probe] kernel finished\n");

    CK(hipMemcpy(ho.data(), dout, ho.size(), hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        unsigned short got = (unsigned short)(ho[i * 2] | (ho[i * 2 + 1] << 8));
        if (got != hq[i]) {
            if (bad < 6) {
                std::fprintf(stderr, "probe mismatch tid=%d got=0x%04x want=0x%04x\n", i, got, hq[i]);
            }
            ++bad;
        }
    }
    unsigned short n_copy = (unsigned short)(ho[128] | (ho[129] << 8));
    std::printf("[probe] n_copy=%u (want %d), bad=%d\n", n_copy, n, bad);
    if (bad != 0 || n_copy != (unsigned short)n) {
        std::printf("PROBE_FAIL\n");
        return 1;
    }
    std::printf("PROBE_OK\n");
    return 0;
}
