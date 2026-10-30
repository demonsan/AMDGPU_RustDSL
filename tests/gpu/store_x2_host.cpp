// probe_store_x2: out[2*tid] = tid+1000, out[2*tid+1] = tid+2000 (global_store_dwordx2, offset 0 and offset 8 with the same data)
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/probe_store_x2.co";
    const int n = 64;
    std::vector<uint32_t> out(2 * n + 2, 0xEEEEEEEEu);
    void* dout; CK(hipMalloc(&dout, out.size() * 4)); CK(hipMemcpy(dout, out.data(), out.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, "probe_store_x2"));
    void* a0 = dout; void* params[] = {&a0};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    // thread t writes words [2t, 2t+1] (offset 0) and [2t+2, 2t+3] (offset 8): the later store of thread t+1 can overwrite word 2t+2 and 2t+3
    for (int t = 0; t < n; ++t) {
        uint32_t w0 = out[2 * t], w1 = out[2 * t + 1];
        // word 2t is rewritten by thread t-1's offset-8 store with ITS data, so only the last thread's pair and thread 0's are unambiguous; check ownership pattern
        (void)w0; (void)w1;
    }
    // unambiguous: every word is one of the two stored values of some thread; check word pairs follow (1000+k, 2000+k)
    for (int k = 0; k < n + 1; ++k) {
        uint32_t a = out[2 * k], b = out[2 * k + 1];
        uint32_t ka = a - 1000, kb = b - 2000;
        if (ka != kb || ka > (uint32_t)n) ++bad;
    }
    std::printf("[store_x2] bad=%d out[0..3]=%u %u %u %u\n", bad, out[0], out[1], out[2], out[3]);
    std::printf(bad == 0 ? "STORE_X2_OK\n" : "STORE_X2_FAIL\n");
    return bad != 0;
}