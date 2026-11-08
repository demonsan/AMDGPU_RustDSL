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
    // DV: the V / O head dim (MLA prefill: D = 192, DV = 128)
    const int DV = envi("DV", D);
    const int rows = envi("ROWS", ours ? 128 : 256), block = envi("BLOCK", ours ? 256 : 512), want_lse = envi("LSE", 1);
    const int iters = envi("ITERS", 50), warm = envi("WARMUP", 10), ncheck = envi("CHECK", 32);
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0.0f, 1.0f);
    // GROUP=1: varlen (THD) with per-sequence lengths in [SQ / 2, SQ] / [SK / 2, SK]; batch mode = the same rows with uniform lengths
    const int group = envi("GROUP", 0);
    std::vector<int> cuq(B + 1, 0), cuk(B + 1, 0);
    for (int i = 0; i < B; i++) {
        int lq = group ? SQ - (int)(rng() % (SQ / 2 + 1)) : SQ, lk = group ? SK - (int)(rng() % (SK / 2 + 1)) : SK;
        cuq[i + 1] = cuq[i] + lq;
        cuk[i + 1] = cuk[i] + lk;
    }
    const int TQ = cuq[B], TK = cuk[B];
    size_t nq = (size_t)TQ * H * D, nk = (size_t)TK * HK * D, nv = (size_t)TK * HK * DV, no = (size_t)TQ * H * DV;
    std::vector<uint16_t> hq(nq), hk(nk), hv(nv);
    for (auto& x : hq) x = tobf(N(rng));
    for (auto& x : hk) x = tobf(N(rng));
    for (auto& x : hv) x = tobf(N(rng));
    uint16_t *dq, *dk, *dv, *dout;
    float* dlse;
    HC(hipMalloc(&dq, nq * 2)); HC(hipMalloc(&dk, nk * 2)); HC(hipMalloc(&dv, nv * 2)); HC(hipMalloc(&dout, no * 2));
    HC(hipMalloc(&dlse, (size_t)H * TQ * 4 + 256));
    int *dcuq, *dcuk;
    HC(hipMalloc(&dcuq, (B + 1) * 4)); HC(hipMalloc(&dcuk, (B + 1) * 4));
    HC(hipMemcpy(dcuq, cuq.data(), (B + 1) * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(dcuk, cuk.data(), (B + 1) * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(dq, hq.data(), nq * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dk, hk.data(), nk * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dv, hv.data(), nv * 2, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0, no * 2));
    hipModule_t mod; hipFunction_t fn;
    HC(hipModuleLoad(&mod, co));
    HC(hipModuleGetFunction(&fn, mod, kn));
    // fmha_fwd_v3_args: field i at 16 i
    // our ABI extension: sliding window (WL / WR env, >= 0) at 656 / 672
    const int WL = envi("WL", -1), WR = envi("WR", -1);
    // KS=lg (ours *_v3_ks): 2^lg key parts per q tile, partials + attn_combine_bf16_hd128_q (COMBINE_CO)
    const int KS = envi("KS", -1);
    std::vector<uint8_t> a(KS >= 0 ? 704 : WL >= 0 ? 688 : 656, 0);
    auto P = [&](int i, const void* p) { memcpy(&a[16 * i], &p, 8); };
    auto U = [&](int i, uint32_t v) { memcpy(&a[16 * i], &v, 4); };
    auto F = [&](int i, float v) { memcpy(&a[16 * i], &v, 4); };
    const float scale = 1.0f / std::sqrt((float)D);
    const uint32_t qs = H * D * 2, ks = HK * D * 2, vs_ = HK * DV * 2, os = H * DV * 2;
    P(0, dout); P(1, dq); P(2, dk); P(3, dv); P(4, want_lse ? dlse : nullptr); F(5, scale);
    U(6, group ? TQ : SQ); U(7, qs); U(8, rows * qs); U(9, D * 2); U(10, group ? 0 : SQ * qs); U(11, H / HK);
    U(12, ks); U(13, D * 2); U(14, group ? 0 : SK * ks);
    U(15, causal && (H % 8 != 0 || SQ > 16384) ? 3 : 5); U(16, want_lse); U(17, group ? TK : SK); U(18, D); U(19, DV); U(20, H);
    U(21, vs_); U(22, DV * 2); U(23, group ? 0 : SK * vs_);
    U(24, os); U(25, DV * 2); U(26, group ? 0 : SQ * os);
    U(29, (group ? TQ : SQ) * 4);
    if (group) { P(27, dcuq); P(28, dcuk); P(30, dcuq); P(31, dcuk); }
    if (WL >= 0) { U(41, WL); U(42, WR); }
    void *dpo = nullptr, *dpl = nullptr;
    const int S = KS >= 0 ? 1 << KS : 1;
    hipFunction_t fcomb = nullptr;
    std::vector<uint8_t> ca(48, 0);
    if (KS >= 0) {
        HC(hipMalloc(&dpo, (size_t)B * H * SQ * S * D * 4)); HC(hipMalloc(&dpl, (size_t)B * H * SQ * S * 4));
        uint32_t lg = KS, sc = S;
        memcpy(&a[656], &lg, 4); memcpy(&a[660], &sc, 4); memcpy(&a[672], &dpo, 8); memcpy(&a[688], &dpl, 8);
        hipModule_t cm; HC(hipModuleLoad(&cm, getenv("COMBINE_CO"))); HC(hipModuleGetFunction(&fcomb, cm, "attn_combine_bf16_hd128_q"));
        float* lsep = want_lse ? dlse : nullptr;
        memcpy(&ca[0], &dout, 8); memcpy(&ca[8], &dpo, 8); memcpy(&ca[16], &dpl, 8); memcpy(&ca[24], &lsep, 8);
        uint32_t hh = H, sqq = SQ, lo = want_lse;
        memcpy(&ca[32], &hh, 4); memcpy(&ca[36], &sqq, 4); memcpy(&ca[40], &sc, 4); memcpy(&ca[44], &lo, 4);
    }
    int tiles = ((group ? TQ : SQ) + rows - 1) / rows;
    // PAIR=1 (ours v3 causal): the vendor's pairing grid
    const bool pair = !ours || envi("PAIR", 0);
    int gt = pair ? (tiles + (causal ? 1 : 0)) / (causal ? 2 : 1) : tiles;
    int gx = group ? H : gt, gy = group ? B : H, gz = group ? gt : B;
    size_t asz = a.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    size_t csz = ca.size();
    void* ccfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ca.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &csz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() {
        HC(hipModuleLaunchKernel(fn, gx, gy, gz * S, block, 1, 1, 0, nullptr, nullptr, cfg));
        if (fcomb) HC(hipModuleLaunchKernel(fcomb, SQ, H, B, 128, 1, 1, 0, nullptr, nullptr, ccfg));
    };
    launch();
    HC(hipDeviceSynchronize());
    std::vector<uint16_t> ho(no);
    std::vector<float> hl((size_t)H * TQ);
    HC(hipMemcpy(ho.data(), dout, no * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hl.data(), dlse, hl.size() * 4, hipMemcpyDeviceToHost));
    // CPU reference on sampled rows (always the first and the last query row of the first / last batch and head)
    long bad = 0, badl = 0;
    double maxe = 0;
    std::vector<double> s(SK), o(DV);
    for (int c = 0; c < ncheck; c++) {
        int b = c == 0 ? 0 : c == 1 ? B - 1 : (int)(rng() % B);
        int h = c == 0 ? 0 : c == 1 ? H - 1 : (int)(rng() % H);
        const int sqb = cuq[b + 1] - cuq[b], skb = cuk[b + 1] - cuk[b];
        int q = c == 0 ? 0 : c == 1 ? sqb - 1 : (int)(rng() % sqb);
        int hk_ = h / (H / HK);
        const int SQ = sqb, SK = skb;  // this sequence
        auto qi = [&](int d) { return (((size_t)cuq[b] + q) * H + h) * D + d; };
        auto ki = [&](int k, int d) { return (((size_t)cuk[b] + k) * HK + hk_) * D + d; };
        auto vi = [&](int k, int d) { return (((size_t)cuk[b] + k) * HK + hk_) * DV + d; };
        auto oi = [&](int d) { return (((size_t)cuq[b] + q) * H + h) * DV + d; };
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
                acc += (double)bf(hq[qi(d)]) * bf(hk[ki(k, d)]);
            s[k] = acc * scale;
            m = std::max(m, s[k]);
        }
        double l = 0;
        std::fill(o.begin(), o.end(), 0.0);
        for (int k = kmin; k <= kmax; k++) {
            double p = std::exp(s[k] - m);
            l += p;
            for (int d = 0; d < DV; d++) o[d] += p * bf(hv[vi(k, d)]);
        }
        if (kmax < kmin) l = 1;  // fully masked row: O = 0
        for (int d = 0; d < DV; d++) {
            double ref = kmax < kmin ? 0 : o[d] / l;
            double got = bf(ho[oi(d)]);
            double e = std::fabs(got - ref);
            maxe = std::max(maxe, e);
            // bf16 output rounding + bf16-P accumulation error
            if (e > 1e-2 * std::fabs(ref) + 2e-2) {
                if (bad < envi("SHOW_BAD", 0)) printf("  bad b=%d h=%d q=%d d=%d got=%g ref=%g\n", b, h, q, d, got, ref);
                bad++;
            }
        }
        if (want_lse && kmax >= kmin) {
            double lr = m + std::log(l), lg = group ? hl[(size_t)h * TQ + cuq[b] + q] : hl[((size_t)b * H + h) * SQ + q];
            if (std::fabs(lg - lr) > 1e-2 + 1e-3 * std::fabs(lr)) { if (badl < envi("SHOW_BAD", 0)) printf("  badlse b=%d h=%d q=%d got %g ref %g\n", b, h, q, lg, lr); badl++; }
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
    double flops = 0;
    for (int i = 0; i < B; i++) flops += 4.0 * H * (double)(cuq[i + 1] - cuq[i]) * (cuk[i + 1] - cuk[i]) * D * (causal ? 0.5 : 1.0);
    printf("%s %s B=%d H=%d HK=%d SQ=%d SK=%d causal=%d: %.2f us %.1f TF/s rows=%d bad=%ld badlse=%ld maxerr=%.3e %s\n", ours ? "ours" : "vendor", kn, B, H, HK,
           SQ, SK, causal, us, flops / (us * 1e-6) / 1e12, ncheck, bad, badl, maxe, bad || badl ? "ATTN_FAIL" : "ATTN_OK");
    return bad || badl ? 3 : 0;
}
