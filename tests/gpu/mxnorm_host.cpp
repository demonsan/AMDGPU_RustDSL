// fused RMSNorm + MXFP4 quantisation (batch 2): rmsnorm_quant_mxfp4_bf16_k<K>: x [M][K] bf16, gamma [K] -> q fp4 [M][K/2] + e8m0 scales in the shuffled a4w4 layout. CPU reference of the same rule applied to the f32
// normalised row (no bf16 rounding in between): the scale bytes must match (a few borderline groups may differ by one), every dequantised value must be within the fp4 half-step of the device's own scale.
// usage: mxnorm_host <co> <kernel> <M> <K> [iters]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float e2m1(int c) { static const float t[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6}; return (c & 8 ? -1.f : 1.f) * t[c & 7]; }
int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s <co> <kernel> <M> <K> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long M = std::atol(argv[3]), K = std::atol(argv[4]); int iters = argc > 5 ? std::atoi(argv[5]) : 30;
    const long KT = K / 256, TP = KT / 2; const float eps = 1e-6f;
    std::mt19937 rng(31); std::normal_distribution<float> nd(0.f, 2.f); std::uniform_real_distribution<float> gd(0.5f, 1.5f);
    std::vector<uint16_t> hx((size_t)M * K), hg(K); for (long m = 0; m < M; ++m) { float sc = 0.3f + (m % 7) * 0.9f; for (long k = 0; k < K; ++k) hx[m * K + k] = f32_to_bf16(nd(rng) * sc); } for (auto& v : hg) v = f32_to_bf16(gd(rng));
    void *dx, *dg, *dq, *ds; size_t qb = (size_t)M * K / 2, sb = (size_t)M * K / 32;
    CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dg, K * 2)); CK(hipMalloc(&dq, qb + 64)); CK(hipMalloc(&ds, sb + 64)); CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dg, hg.data(), K * 2, hipMemcpyHostToDevice)); CK(hipMemset(dq, 0x55, qb + 64)); CK(hipMemset(ds, 0x55, sb + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *x, *g, *q, *s; int32_t m; float eps; } args{dx, dg, dq, ds, (int32_t)M, eps}; size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() { CK(hipModuleLaunchKernel(fn, (unsigned)(M / 16), 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(); CK(hipDeviceSynchronize());
    std::vector<uint8_t> hq(qb + 64), hs(sb + 64); CK(hipMemcpy(hq.data(), dq, qb + 64, hipMemcpyDeviceToHost)); CK(hipMemcpy(hs.data(), ds, sb + 64, hipMemcpyDeviceToHost));
    long sbad = 0, qbad = 0, groups = 0; std::vector<float> y(K);
    for (long m = 0; m < M; ++m) {
        float ss = 0; for (long k = 0; k < K; ++k) { float v = bf16_to_f32(hx[m * K + k]); ss += v * v; } float inv = 1.0f / std::sqrt(ss / K + eps);
        for (long k = 0; k < K; ++k) y[k] = bf16_to_f32(hx[m * K + k]) * inv * bf16_to_f32(hg[k]);
        for (long G = 0; G < K / 32; ++G) {
            float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(y[G * 32 + i])); uint32_t ab; std::memcpy(&ab, &amax, 4); uint32_t rnd = (ab + 0x200000u) & 0xff800000u; int e = (int)((rnd >> 23) & 0xff) - 2; e = std::max(0, std::min(254, e));
            long p = G / 16, within = G % 16, g = within % 4, j = within / 4; long lane = 16 * g + (m % 16); long rbk = m / 16; size_t sidx = (((size_t)rbk * TP + p) * 64 + lane) * 4 + j; int edev = hs[sidx];
            ++groups; if (edev != e) { ++sbad; if (std::abs(edev - e) > 1 && sbad < 5) std::fprintf(stderr, "scale m=%ld G=%ld: dev %d ref %d\n", m, G, edev, e); if (std::abs(edev - e) > 1) { ++qbad; continue; } }
            double sc = std::ldexp(1.0, std::max(edev, 1) - 127);   // e = 0 is the smallest normal for the division
            for (int i = 0; i < 32; ++i) { long k = G * 32 + i; int code = (hq[(size_t)m * K / 2 + k / 2] >> (4 * (k & 1))) & 15; double dq_ = e2m1(code) * sc; if (!(std::fabs(dq_ - y[k]) <= sc * 1.01 + 1e-30)) { if (qbad < 5) std::fprintf(stderr, "code bad m=%ld k=%ld: dq %g y %g scale %g\n", m, k, dq_, y[k], sc); ++qbad; } }
        } }
    long guard = 0; for (size_t i = qb; i < qb + 64; ++i) if (hq[i] != 0x55) ++guard; for (size_t i = sb; i < sb + 64; ++i) if (hs[i] != 0x55) ++guard;
    std::printf("check: rmsnorm_quant_mxfp4 M=%ld K=%ld scale_mismatch=%ld (of %ld groups) code_bad=%ld past_end_bad=%ld %s\n", M, K, sbad, groups, qbad, guard, (qbad == 0 && guard == 0 && sbad * 1000 <= groups) ? "MXN_OK" : "MXN_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); for (int i = 0; i < 5; ++i) launch();
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) launch(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * K * 2 + (double)M * K / 2 + (double)M * K / 32; std::printf("timing: %.4f ms/iter (%.1f us), %.1f GB/s (traffic %.1f MB: one read of x, q and scales)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}