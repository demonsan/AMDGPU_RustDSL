// Host for the count-matched s_barrier probe: launch 512 threads (8 waves), one output pointer; print wall time and the
// per-wave (mismatch count, last partner value, wave id) rows.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
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
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 512, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms = 0.f; CK(hipEventElapsedTime(&ms, e0, e1));
    CK(hipMemcpy(h.data(), d, h.size() * 4, hipMemcpyDeviceToHost));
    unsigned mism = 0; int bad_row = 0;
    std::printf("KERNEL %-9s wall=%9.3fms", argv[2], ms);
    for (int w = 0; w < 8; ++w) {
        unsigned m = h[w * 4], v = h[w * 4 + 1], id = h[w * 4 + 2];
        mism += m;
        if (id != (unsigned)w) bad_row = 1;
    }
    std::printf(" mismatches=%u bad_row=%d\n", mism, bad_row);
    for (int w = 0; w < 8; ++w) std::printf("   wave%d: mismatch=%u last_value=%u id=%u\n", w, h[w * 4], h[w * 4 + 1], h[w * 4 + 2]);
    return (bad_row || mism) ? 3 : 0;
}
