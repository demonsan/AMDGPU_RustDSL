// Same-process host for bf16 attention forward kernels with the aiter fmha_v3 656-byte ABI (csrc/include/mha_fwd.h fmha_fwd_v3_args: every
// field in a 16-byte slot): aiter's prebuilt gfx950 fwd_hd128_bf16[_causal].co and ours, same data, CPU reference on sampled query rows.
//
//   attn_host <vendor|ours> <co> <kernel> B H HK SQ SK causal [D=128]
//   env: ITERS (50), WARMUP (10), CHECK (32 sampled (b, h, q) rows), SEED (1), ROWS (query rows per workgroup: vendor 256, ours 128),
//        BLOCK (threads: vendor 512, ours 256), LSE (1: request / check the log-sum-exp)
// Layouts: Q [B, SQ, H, D], K / V [B, SK, HK, D], O [B, SQ, H, D] bf16 (BSHD, contiguous); LSE fp32 [B, H, SQ] (natural log). Causal =
// bottom-right aligned (key k visible to query q iff k <= q + SK - SQ).
#include <hip/hip_runtime.h>
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

int main(int argc, char** argv) {
    if (argc < 10) { fprintf(stderr, "usage: %s vendor|ours co kernel B H HK SQ SK causal [D]\n", argv[0]); return 1; }
    const bool ours = std::string(argv[1]) == "ours";
    const char* co = argv[2];
    const char* kn = argv[3];
    const int B = atoi(argv[4]), H = atoi(argv[5]), HK = atoi(argv[6]), SQ = atoi(argv[7]), SK = atoi(argv[8]), causal = atoi(argv[9]);
    const int D = argc > 10 ? atoi(argv[10]) : 128;
    const int rows = envi("ROWS", ours ? 128 : 256), block = envi("BLOCK", ours ? 256 : 512), want_lse = envi("LSE", 1);
    const int iters = envi("ITERS", 50), warm = envi("WARMUP", 10), ncheck = envi("CHECK", 32);
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0.0f, 1.0f);
    size_t nq = (size_t)B * SQ * H * D, nk = (size_t)B * SK * HK * D;
    std::vector<uint16_t> hq(nq), hk(nk), hv(nk);
    for (auto& x : hq) x = tobf(N(rng));
    for (auto& x : hk) x = tobf(N(rng));
    for (auto& x : hv) x = tobf(N(rng));
    uint16_t *dq, *dk, *dv, *dout;
    float* dlse;
    HC(hipMalloc(&dq, nq * 2)); HC(hipMalloc(&dk, nk * 2)); HC(hipMalloc(&dv, nk * 2)); HC(hipMalloc(&dout, nq * 2));
    HC(hipMalloc(&dlse, (size_t)B * H * SQ * 4 + 256));
    HC(hipMemcpy(dq, hq.data(), nq * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dk, hk.data(), nk * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dv, hv.data(), nk * 2, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0, nq * 2));
    hipModule_t mod; hipFunction_t fn;
    HC(hipModuleLoad(&mod, co));
    HC(hipModuleGetFunction(&fn, mod, kn));
    // fmha_fwd_v3_args: field i at 16 i
    // our ABI extension: sliding window (WL / WR env, >= 0) at 656 / 672
    const int WL = envi("WL", -1), WR = envi("WR", -1);
    std::vector<uint8_t> a(WL >= 0 ? 688 : 656, 0);
    auto P = [&](int i, const void* p) { memcpy(&a[16 * i], &p, 8); };
    auto U = [&](int i, uint32_t v) { memcpy(&a[16 * i], &v, 4); };
    auto F = [&](int i, float v) { memcpy(&a[16 * i], &v, 4); };
    const float scale = 1.0f / std::sqrt((float)D);
    const uint32_t qs = H * D * 2, ks = HK * D * 2;
    P(0, dout); P(1, dq); P(2, dk); P(3, dv); P(4, want_lse ? dlse : nullptr); F(5, scale);
    U(6, SQ); U(7, qs); U(8, rows * qs); U(9, D * 2); U(10, SQ * qs); U(11, H / HK);
    U(12, ks); U(13, D * 2); U(14, SK * ks);
    U(15, causal && (H % 8 != 0 || SQ > 16384) ? 3 : 5); U(16, want_lse); U(17, SK); U(18, D); U(19, D); U(20, H);
    U(21, ks); U(22, D * 2); U(23, SK * ks);
    U(24, qs); U(25, D * 2); U(26, SQ * qs);
    U(29, SQ * 4);
    if (WL >= 0) { U(41, WL); U(42, WR); }
    int tiles = (SQ + rows - 1) / rows;
    int gx = ours ? tiles : (tiles + (causal ? 1 : 0)) / (causal ? 2 : 1);
    size_t asz = a.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() { HC(hipModuleLaunchKernel(fn, gx, H, B, block, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch();
    HC(hipDeviceSynchronize());
    std::vector<uint16_t> ho(nq);
    std::vector<float> hl((size_t)B * H * SQ);
    HC(hipMemcpy(ho.data(), dout, nq * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hl.data(), dlse, hl.size() * 4, hipMemcpyDeviceToHost));
    // CPU reference on sampled rows (always the first and the last query row of the first / last batch and head)
    long bad = 0, badl = 0;
    double maxe = 0;
    std::vector<double> s(SK), o(D);
    for (int c = 0; c < ncheck; c++) {
        int b = c == 0 ? 0 : c == 1 ? B - 1 : (int)(rng() % B);
        int h = c == 0 ? 0 : c == 1 ? H - 1 : (int)(rng() % H);
        int q = c == 0 ? 0 : c == 1 ? SQ - 1 : (int)(rng() % SQ);
        int hk_ = h / (H / HK);
        int kmax = causal ? std::min(SK - 1, q + SK - SQ) : SK - 1;
        int kmin = 0;
        if (WL >= 0) {
            kmin = std::max(0, q + SK - SQ - WL);
            kmax = std::min(kmax, q + SK - SQ + WR);
        }
        double m = -1e300;
        for (int k = kmin; k <= kmax; k++) {
            double acc = 0;
            for (int d = 0; d < D; d++)
                acc += (double)bf(hq[(((size_t)b * SQ + q) * H + h) * D + d]) * bf(hk[(((size_t)b * SK + k) * HK + hk_) * D + d]);
            s[k] = acc * scale;
            m = std::max(m, s[k]);
        }
        double l = 0;
        std::fill(o.begin(), o.end(), 0.0);
        for (int k = kmin; k <= kmax; k++) {
            double p = std::exp(s[k] - m);
            l += p;
            for (int d = 0; d < D; d++) o[d] += p * bf(hv[(((size_t)b * SK + k) * HK + hk_) * D + d]);
        }
        if (kmax < kmin) l = 1;  // fully masked row: O = 0
        for (int d = 0; d < D; d++) {
            double ref = kmax < kmin ? 0 : o[d] / l;
            double got = bf(ho[(((size_t)b * SQ + q) * H + h) * D + d]);
            double e = std::fabs(got - ref);
            maxe = std::max(maxe, e);
            // bf16 output rounding + bf16-P accumulation error
            if (e > 1e-2 * std::fabs(ref) + 2e-2) {
                if (bad < envi("SHOW_BAD", 0)) printf("  bad b=%d h=%d q=%d d=%d got=%g ref=%g\n", b, h, q, d, got, ref);
                bad++;
            }
        }
        if (want_lse && kmax >= kmin) {
            double lr = m + std::log(l), lg = hl[((size_t)b * H + h) * SQ + q];
            if (std::fabs(lg - lr) > 1e-2 + 1e-3 * std::fabs(lr)) badl++;
        }
    }
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    for (int i = 0; i < warm; i++) launch();
    HC(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch();
    HC(hipEventRecord(e1));
    HC(hipEventSynchronize(e1));
    float ms; HC(hipEventElapsedTime(&ms, e0, e1));
    double us = 1e3 * ms / iters;
    double flops = 4.0 * B * H * (double)SQ * SK * D * (causal ? 0.5 : 1.0);
    printf("%s %s B=%d H=%d HK=%d SQ=%d SK=%d causal=%d: %.2f us %.1f TF/s rows=%d bad=%ld badlse=%ld maxerr=%.3e %s\n", ours ? "ours" : "vendor", kn, B, H, HK,
           SQ, SK, causal, us, flops / (us * 1e-6) / 1e12, ncheck, bad, badl, maxe, bad || badl ? "ATTN_FAIL" : "ATTN_OK");
    return bad || badl ? 3 : 0;
}
