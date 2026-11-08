// MXFP6 quantisation (batch 2): quant_mxfp6_{e2m3,e3m2}_bf16_k<K>: x [M][K] bf16 -> q [M][K/32][24 B] + e8m0 scales [M][K/32]; CPU reference of the OCP rule (shared exponent = floor(log2 amax) - emax, saturating RNE).
// usage: mxfp6_host <co> <kernel> <e2m3|e3m2> <M> <K> [iters]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static double fp6_val(int c, bool e3m2) { int s = (c >> 5) & 1; double v; if (!e3m2) { int e = (c >> 3) & 3, m = c & 7; v = e == 0 ? m / 8.0 : (1.0 + m / 8.0) * std::ldexp(1.0, e - 1); } else { int e = (c >> 2) & 7, m = c & 3; v = e == 0 ? m / 16.0 : (1.0 + m / 4.0) * std::ldexp(1.0, e - 3); } return s ? -v : v; }
int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s <co> <kernel> <e2m3|e3m2> <M> <K> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; const bool e3m2 = std::string(argv[3]) == "e3m2"; long M = std::atol(argv[4]), K = std::atol(argv[5]); int iters = argc > 6 ? std::atoi(argv[6]) : 30; const long G = K / 32;
    std::mt19937 rng(47); std::normal_distribution<float> nd(0.f, 2.f); std::vector<uint16_t> hx((size_t)M * K); for (long m = 0; m < M; ++m) { float sc = 0.05f + (m % 7) * 1.3f; for (long k = 0; k < K; ++k) hx[m * K + k] = f32_to_bf16(nd(rng) * sc * (k % 97 == 0 ? 4.f : 1.f)); }
    void *dx, *dq, *ds; size_t qb = (size_t)M * G * 24, sb = (size_t)M * G; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dq, qb + 64)); CK(hipMalloc(&ds, sb + 64)); CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemset(dq, 0x55, qb + 64)); CK(hipMemset(ds, 0x55, sb + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *x, *q, *s; int32_t m; } args{dx, dq, ds, (int32_t)M}; size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    unsigned gx = (unsigned)((G + 255) / 256); auto launch = [&]() { CK(hipModuleLaunchKernel(fn, gx, (unsigned)M, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(); CK(hipDeviceSynchronize());
    std::vector<uint8_t> hq(qb + 64), hs(sb + 64); CK(hipMemcpy(hq.data(), dq, qb + 64, hipMemcpyDeviceToHost)); CK(hipMemcpy(hs.data(), ds, sb + 64, hipMemcpyDeviceToHost));
    const int emax = e3m2 ? 4 : 2; const double maxv = e3m2 ? 28.0 : 7.5; long sbad = 0, qbad = 0;
    for (long m = 0; m < M; ++m) for (long g = 0; g < G; ++g) {
        float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(hx[m * K + g * 32 + i]))); uint32_t ab; std::memcpy(&ab, &amax, 4); int e = (int)((ab >> 23) & 0xff) - emax; e = std::max(0, std::min(254, e)); int edev = hs[m * G + g];
        if (edev != e) { ++sbad; if (sbad < 5) std::fprintf(stderr, "scale m=%ld g=%ld: dev %d ref %d\n", m, g, edev, e); continue; }
        double sc = std::ldexp(1.0, std::max(e, 1) - 127); const uint8_t* gq = &hq[((size_t)m * G + g) * 24];
        for (int i = 0; i < 32; ++i) { int bit = 6 * i; int code = ((gq[bit / 8] | (gq[bit / 8 + 1] << 8)) >> (bit % 8)) & 63; double y = bf16_to_f32(hx[m * K + g * 32 + i]) / sc; double dq_ = fp6_val(code, e3m2); double sat = std::max(-maxv, std::min(maxv, y)); double half = e3m2 ? std::max(std::fabs(sat) / 8.0, 1.0 / 32) : std::max(std::fabs(sat) / 16.0, 1.0 / 16);
            if (!(std::fabs(dq_ - sat) <= half * 1.01)) { if (qbad < 5) std::fprintf(stderr, "code bad m=%ld g=%ld i=%d: dq %g y %g (code %d)\n", m, g, i, dq_, y, code); ++qbad; } } }
    long guard = 0; for (size_t i = qb; i < qb + 64; ++i) if (hq[i] != 0x55) ++guard; for (size_t i = sb; i < sb + 64; ++i) if (hs[i] != 0x55) ++guard;
    std::printf("check: mxfp6_%s M=%ld K=%ld scale_bad=%ld code_bad=%ld past_end_bad=%ld %s\n", argv[3], M, K, sbad, qbad, guard, (sbad == 0 && qbad == 0 && guard == 0) ? "MX6_OK" : "MX6_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); for (int i = 0; i < 5; ++i) launch();
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) launch(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * K * 2 + (double)qb + (double)sb; std::printf("timing: %.4f ms/iter (%.1f us), %.1f GB/s (traffic %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}