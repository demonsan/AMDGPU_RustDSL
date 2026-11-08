// Expert routing (batch 2): topk_softmax (softmax over the experts of a row, then the top-k probabilities and ids; tie = lowest id) against a CPU reference, and latency / bandwidth.
// usage: topk_host <co> <kernel> <T> <E> <K> <plain|renorm> [iters]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s <co> <kernel> <T> <E> <K> <plain|renorm> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long T = std::atol(argv[3]), E = std::atol(argv[4]), K = std::atol(argv[5]); bool renorm = std::string(argv[6]) == "renorm"; int iters = argc > 7 ? std::atoi(argv[7]) : 50;
    std::mt19937 rng(8); std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<uint16_t> hg((size_t)T * E); for (auto& v : hg) v = f32_to_bf16(nd(rng));
    void *dg, *dw, *di; CK(hipMalloc(&dg, hg.size() * 2)); CK(hipMalloc(&dw, (size_t)T * K * 4 + 64)); CK(hipMalloc(&di, (size_t)T * K * 4 + 64));
    CK(hipMemcpy(dg, hg.data(), hg.size() * 2, hipMemcpyHostToDevice)); CK(hipMemset(dw, 0x55, (size_t)T * K * 4 + 64)); CK(hipMemset(di, 0x55, (size_t)T * K * 4 + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *g, *w, *i; int32_t t; } args{dg, dw, di, (int32_t)T};
    size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    int grid = (int)((T + 3) / 4);
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<float> hw((size_t)T * K + 16); std::vector<int32_t> hi((size_t)T * K + 16);
    CK(hipMemcpy(hw.data(), dw, hw.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(hi.data(), di, hi.size() * 4, hipMemcpyDeviceToHost));
    long bad_idx = 0, bad_w = 0; double max_rel = 0;
    std::vector<double> p(E); std::vector<int> ord(E);
    for (long t = 0; t < T; ++t) {
        double mx = -1e300; for (long e = 0; e < E; ++e) mx = std::max(mx, (double)bf16_to_f32(hg[t * E + e]));
        double s = 0; for (long e = 0; e < E; ++e) { p[e] = std::exp(bf16_to_f32(hg[t * E + e]) - mx); s += p[e]; }
        for (long e = 0; e < E; ++e) p[e] /= s;
        std::iota(ord.begin(), ord.end(), 0); std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return bf16_to_f32(hg[t * E + a]) > bf16_to_f32(hg[t * E + b]); });
        double tot = 0; for (long r = 0; r < K; ++r) tot += p[ord[r]];
        for (long r = 0; r < K; ++r) {
            double wref = renorm ? p[ord[r]] / tot : p[ord[r]];
            if (hi[t * K + r] != ord[r]) ++bad_idx;
            double err = std::fabs(hw[t * K + r] - wref); if (err > 1e-5 + 2e-5 * wref) { if (bad_w < 5) std::fprintf(stderr, "bad w (t%ld,r%ld): got %g ref %g\n", t, r, hw[t * K + r], wref); ++bad_w; }
            max_rel = std::max(max_rel, err / wref);
        }
    }
    long guard = 0; for (size_t i = (size_t)T * K; i < (size_t)T * K + 16; ++i) { uint32_t u; std::memcpy(&u, &hw[i], 4); if (u != 0x55555555u || hi[i] != 0x55555555) ++guard; }
    std::printf("check: topk_softmax%s T=%ld E=%ld K=%ld bad_idx=%ld bad_w=%ld max_rel=%.2e past_end_bad=%ld %s\n", renorm ? "_renorm" : "", T, E, K, bad_idx, bad_w, max_rel, guard, (bad_idx == 0 && bad_w == 0 && guard == 0) ? "TOPK_OK" : "TOPK_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)T * E * 2 + (double)T * K * 8;
    std::printf("timing: %.4f ms/iter (%.2f us), %.1f GB/s (traffic %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}