// DeepSeek-V3 style expert gate (batch 2): moe_fused_gate_bf16_e<E>_g<G>_tg<TG>_k<K>: CPU reference (sigmoid, + bias, group top-2 sums, drop groups, top-K, normalised weights) and bandwidth.
// usage: moe_gate_host <co> <kernel> <M> <E> <G> <TG> <K> [iters]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
int main(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: %s <co> <kernel> <M> <E> <G> <TG> <K> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long M = std::atol(argv[3]), E = std::atol(argv[4]), G = std::atol(argv[5]), TG = std::atol(argv[6]), K = std::atol(argv[7]); int iters = argc > 8 ? std::atoi(argv[8]) : 30;
    const long vpt = E / G, rpb = 4 * (64 / G);
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 2.f); std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    std::vector<uint16_t> hx((size_t)M * E), hb(E); for (auto& v : hx) v = f32_to_bf16(nd(rng)); for (auto& v : hb) v = f32_to_bf16(ud(rng));
    if (M > 3) { hx[5] = 0x7fc0; }                       // a NaN logit: sigmoid -> 0
    void *dx, *db, *dw, *di; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&db, E * 2)); CK(hipMalloc(&dw, (size_t)M * K * 4 + 64)); CK(hipMalloc(&di, (size_t)M * K * 4 + 64)); CK(hipMemset(dw, 0x55, (size_t)M * K * 4 + 64)); CK(hipMemset(di, 0x55, (size_t)M * K * 4 + 64));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(db, hb.data(), E * 2, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *x, *b, *w, *i; int32_t m; } args{dx, db, dw, di, (int32_t)M}; size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    int grid = (int)((M + rpb - 1) / rpb);
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<float> hw((size_t)M * K + 16); std::vector<int32_t> hi((size_t)M * K + 16); CK(hipMemcpy(hw.data(), dw, hw.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(hi.data(), di, hi.size() * 4, hipMemcpyDeviceToHost));
    long bad = 0, near = 0;
    std::vector<double> s(E), bi(E);
    for (long m = 0; m < M; ++m) {
        for (long e = 0; e < E; ++e) { double x = bf16_to_f32(hx[m * E + e]); s[e] = (x != x) ? 0.0 : 1.0 / (1.0 + std::exp(-x)); bi[e] = s[e] + bf16_to_f32(hb[e]); }
        std::vector<double> gs(G); for (long g = 0; g < G; ++g) { std::vector<double> v(bi.begin() + g * vpt, bi.begin() + (g + 1) * vpt); std::sort(v.begin(), v.end(), std::greater<double>()); gs[g] = v[0] + v[1]; }
        std::vector<long> go(G); std::iota(go.begin(), go.end(), 0); std::stable_sort(go.begin(), go.end(), [&](long a, long b) { return gs[a] > gs[b]; });   // best first, the lower index first on ties
        std::vector<char> keep(G, 0); for (long i = 0; i < TG; ++i) keep[go[i]] = 1;
        std::vector<long> ord; for (long e = 0; e < E; ++e) if (keep[e / vpt]) ord.push_back(e); std::stable_sort(ord.begin(), ord.end(), [&](long a, long b) { return bi[a] > bi[b]; });
        double sum = 0; for (long i = 0; i < K; ++i) sum += s[ord[i]];
        for (long i = 0; i < K; ++i) {
            long ref = ord[i]; long got = hi[m * K + i]; double wref = s[ref] / sum; double wgot = hw[m * K + i];
            bool idok = got == ref; if (!idok && got >= 0 && got < E && std::fabs(bi[got] - bi[ref]) < 3e-6) { idok = true; ++near; }   // a near-tie of the biased scores (the device sigmoid is the exp2 / rcp form)
            if (idok && got != ref) wref = s[got] / sum;
            if (!idok || !(std::fabs(wgot - wref) <= std::fabs(wref) * 2e-4 + 1e-7)) { if (bad < 5) std::fprintf(stderr, "bad m=%ld i=%ld: id %ld (ref %ld) w %g (ref %g)\n", m, i, got, ref, wgot, wref); ++bad; }
        }
    }
    long guard = 0; for (size_t i = (size_t)M * K; i < (size_t)M * K + 16; ++i) { uint32_t u; std::memcpy(&u, &hw[i], 4); if (u != 0x55555555u || hi[i] != 0x55555555) ++guard; }
    std::printf("check: moe_gate M=%ld E=%ld G=%ld TG=%ld K=%ld bad=%ld near_ties=%ld past_end_bad=%ld %s\n", M, E, G, TG, K, bad, near, guard, (bad == 0 && guard == 0) ? "GATE_OK" : "GATE_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * E * 2 + (double)M * K * 8;
    std::printf("timing: %.4f ms/iter (%.2f us), %.1f GB/s (traffic %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}