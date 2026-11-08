// MX quantization host: ours `quant_mxfp4_bf16_k<K>` / `quant_mxfp8_bf16_k<K>` vs a CPU reference of the same rule (exact codes + scales),
// dequantization error, and bandwidth (bytes read + written).
//   mxquant_host <co> <mxfp4|mxfp8> M K     env: ITERS (100), SEED (1), SHOW_BAD (0)
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
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)
static int envi(const char* n, int d) { const char* v = getenv(n); return v ? atoi(v) : d; }
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t tobf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float e2m1_val(int c) { static const float t[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6}; return (c & 8 ? -1 : 1) * t[c & 7]; }
static float e4m3_val(int c) {
    if ((c & 0x7f) == 0x7f) return NAN;
    int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}
// nearest representable value (ties: even mantissa), saturating
template <class F> static float nearest(float x, int n, F val) {
    float best = 0, bd = INFINITY;
    int bc = 0;
    for (int c = 0; c < n; c++) {
        float v = val(c);
        if (std::isnan(v)) continue;
        float d = std::fabs(v - x);
        if (d < bd || (d == bd && (c & 1) == 0 && (bc & 1))) { bd = d; best = v; bc = c; }
    }
    return best;
}
int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s co mxfp4|mxfp8 M K\n", argv[0]); return 1; }
    const std::string kind = argv[2];
    const int M = atoi(argv[3]), K = atoi(argv[4]);
    const bool f4 = kind == "mxfp4";
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0, 1);
    std::vector<uint16_t> x((size_t)M * K);
    for (size_t i = 0; i < x.size(); i++) x[i] = tobf(N(rng) * std::exp2((float)((i / 32) % 9) - 4));
    const size_t qn = f4 ? (size_t)M * K / 2 : (size_t)M * K, sn = f4 ? (size_t)M * K / 32 : (size_t)M * K / 128;
    uint16_t* dx;
    uint8_t *dq, *ds;
    HC(hipMalloc(&dx, x.size() * 2)); HC(hipMalloc(&dq, qn)); HC(hipMalloc(&ds, sn));
    HC(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    hipModule_t mod;
    hipFunction_t fn;
    std::string kn = std::string(f4 ? "quant_mxfp4_bf16_k" : "quant_mxfp8_bf16_k") + std::to_string(K);
    HC(hipModuleLoad(&mod, argv[1])); HC(hipModuleGetFunction(&fn, mod, kn.c_str()));
    std::vector<uint8_t> a(32, 0);
    memcpy(&a[0], &dx, 8); memcpy(&a[8], &dq, 8); memcpy(&a[16], &ds, 8); memcpy(&a[24], &M, 4);
    size_t asz = a.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    int gx = f4 ? (K / 512 + 3) / 4 : K / 128, gy = M / 16;
    auto launch = [&]() { HC(hipModuleLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch();
    HC(hipDeviceSynchronize());
    std::vector<uint8_t> q(qn), s(sn);
    HC(hipMemcpy(q.data(), dq, qn, hipMemcpyDeviceToHost)); HC(hipMemcpy(s.data(), ds, sn, hipMemcpyDeviceToHost));
    long bad_s = 0, bad_q = 0;
    double maxrel = 0;
    const int G = f4 ? 32 : 128, KT = K / 256;
    for (int m = 0; m < M; m++)
        for (int gi = 0; gi < K / G; gi++) {
            float amax = 0;
            for (int i = 0; i < G; i++) amax = std::max(amax, std::fabs(bf(x[(size_t)m * K + gi * G + i])));
            int e;
            if (f4) {
                uint32_t r = (fbits(amax) + 0x200000) & 0xff800000u;
                e = (int)((r >> 23) & 0xff) - 2;
            } else {
                uint32_t t = fbits(amax * (1.0f / 448.0f));
                e = (int)((t >> 23) & 0xff) + ((t & 0x7fffff) ? 1 : 0);
            }
            e = std::min(254, std::max(0, e));
            size_t sidx;
            if (f4) {
                int rb = m / 16, r = m % 16, t = gi / 8, gg = gi % 8, p = t / 2, hh = t % 2, g = gg % 4, hi = gg / 4;
                int lane = 16 * g + r;
                sidx = (((size_t)rb * (KT / 2) + p) * 64 + lane) * 4 + 2 * hh + hi;
            } else {
                sidx = (size_t)gi * M + m;
            }
            if (s[sidx] != e) {
                if (bad_s < envi("SHOW_BAD", 0)) printf("  scale m=%d g=%d got %d ref %d\n", m, gi, s[sidx], e);
                bad_s++;
            }
            float sc = std::ldexp(1.0f, std::max(e, 1) - 127);
            for (int i = 0; i < G; i++) {
                size_t k = (size_t)gi * G + i;
                float xv = bf(x[(size_t)m * K + k]), v = xv / sc;
                float rv = f4 ? nearest(v, 16, e2m1_val) : nearest(v, 256, e4m3_val);
                int got = f4 ? ((q[((size_t)m * K + k) / 2] >> (4 * (k & 1))) & 15) : q[(size_t)m * K + k];
                float gv = f4 ? e2m1_val(got) : e4m3_val(got);
                if (gv != rv) {
                    if (bad_q < envi("SHOW_BAD", 0)) printf("  code m=%d k=%zu x/s=%g got %g ref %g\n", m, k, v, gv, rv);
                    bad_q++;
                }
                if (amax > 0) maxrel = std::max(maxrel, (double)std::fabs(gv * sc - xv) / amax);
            }
        }
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    for (int i = 0; i < 5; i++) launch();
    int it = envi("ITERS", 100);
    HC(hipEventRecord(e0));
    for (int i = 0; i < it; i++) launch();
    HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
    float ms;
    HC(hipEventElapsedTime(&ms, e0, e1));
    double us = 1e3 * ms / it, bytes = x.size() * 2.0 + qn + sn;
    printf("%s M=%d K=%d: %.2f us %.0f GB/s bad_scale=%ld bad_code=%ld (of %zu) max |deq - x| / amax = %.3f %s\n", kn.c_str(), M, K, us, bytes / (us * 1e-6) / 1e9,
           bad_s, bad_q, (size_t)M * K, maxrel, (bad_s == 0 && bad_q <= (long)((size_t)M * K / 10000)) ? "MXQ_OK" : "MXQ_FAIL");
    return 0;
}
