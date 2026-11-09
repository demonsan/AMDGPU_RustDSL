// Same-process host for per-tensor FP8 attention forward (aiter fmha_v3 ABI, fp8bf16: fmha_fwd_v3_args slots 32-40 = q / k / v descale
// pointers + byte strides): aiter's prebuilt gfx950 fwd_hd128_fp8[_causal].co and ours (fmha_fwd_hd128_fp8pt_v3[_causal]) on the same data,
// both checked against a CPU reference on sampled query rows, then timed alternately (same run).
//
//   fp8pt_host <vendor_co> <vendor_kernel> <our_co> <our_kernel> B H HK SQ SK causal
//   env: ITERS (50), WARMUP (10), CHECK (32 sampled (b, h, q) rows), SEED (1), ROUNDS (3: alternating timing rounds), SCALAR_DS (1: descales
//        as one scalar, all strides 0 - the aiter [1] form)
// Data: Q, K, V ~ N(0, 1) (V: N(0,1) + 0.5 to keep outputs away from 0), quantised per (batch, KV head) to e4m3 with descale = amax / 448
// (aiter's [B, HK] descale form; Q's descale is indexed by the KV head of the query head). Layouts BSHD contiguous, O bf16. Causal =
// bottom-right (key k visible to query q iff k <= q + SK - SQ).
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
// OCP e4m3fn
static float e4m3_val(uint8_t x) {
    int s = x >> 7, e = (x >> 3) & 15, m = x & 7;
    float v = e == 0 ? std::ldexp((float)m / 8.0f, -6) : std::ldexp(1.0f + (float)m / 8.0f, e - 7);
    return s ? -v : v;
}
static uint8_t to_e4m3(float f) {
    // round to nearest even over the 254 finite non-negative codes (0x7F = NaN excluded)
    uint8_t s = f < 0 ? 0x80 : 0;
    float a = std::fabs(f);
    if (a >= 448.0f) return s | 0x7E;
    int lo = 0, hi = 0x7E;
    while (hi - lo > 1) { int mid = (lo + hi) / 2; if (e4m3_val((uint8_t)mid) <= a) lo = mid; else hi = mid; }
    float dl = a - e4m3_val((uint8_t)lo), dh = e4m3_val((uint8_t)hi) - a;
    int c = dl < dh ? lo : dh < dl ? hi : ((lo & 1) ? hi : lo);
    return s | (uint8_t)c;
}

struct Launch {
    hipFunction_t fn;
    std::vector<uint8_t> a;
    int gx, gy, gz, block;
    // KV split (ours): the combine launch after the main kernel
    hipFunction_t comb = nullptr;
    std::vector<uint8_t> ca;
    int cgrid = 0;
};

int main(int argc, char** argv) {
    if (argc < 11) { fprintf(stderr, "usage: %s vendor_co vendor_kernel our_co our_kernel B H HK SQ SK causal\n", argv[0]); return 1; }
    const int B = atoi(argv[5]), H = atoi(argv[6]), HK = atoi(argv[7]), SQ = atoi(argv[8]), SK = atoi(argv[9]), causal = atoi(argv[10]);
    const int D = 128, G = H / HK;
    const int iters = envi("ITERS", 50), warm = envi("WARMUP", 10), ncheck = envi("CHECK", 32), rounds = envi("ROUNDS", 3);
    const int scalar_ds = envi("SCALAR_DS", 0);
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0.0f, 1.0f);
    size_t nq = (size_t)B * SQ * H * D, nk = (size_t)B * SK * HK * D, no = nq;
    std::vector<float> fq(nq), fk(nk), fv(nk);
    for (auto& x : fq) x = N(rng);
    for (auto& x : fk) x = N(rng);
    for (auto& x : fv) x = N(rng) + 0.5f;
    // per (b, hk) descales; Q over the G query heads of the KV head
    std::vector<float> dsq(B * HK), dsk(B * HK), dsv(B * HK);
    auto qi = [&](int b, int s, int h, int d) { return (((size_t)b * SQ + s) * H + h) * D + d; };
    auto ki = [&](int b, int s, int h, int d) { return (((size_t)b * SK + s) * HK + h) * D + d; };
    for (int b = 0; b < B; b++)
        for (int g = 0; g < HK; g++) {
            float aq = 0, ak = 0, av = 0;
            for (int s = 0; s < SQ; s++) for (int j = 0; j < G; j++) for (int d = 0; d < D; d++) aq = std::max(aq, std::fabs(fq[qi(b, s, g * G + j, d)]));
            for (int s = 0; s < SK; s++) for (int d = 0; d < D; d++) { ak = std::max(ak, std::fabs(fk[ki(b, s, g, d)])); av = std::max(av, std::fabs(fv[ki(b, s, g, d)])); }
            if (scalar_ds) { aq = 4.5f; ak = 4.5f; av = 5.0f; }
            dsq[b * HK + g] = aq / 448.0f; dsk[b * HK + g] = ak / 448.0f; dsv[b * HK + g] = av / 448.0f;
        }
    std::vector<uint8_t> hq(nq), hk(nk), hv(nk);
    for (int b = 0; b < B; b++)
        for (int s = 0; s < SQ; s++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) hq[qi(b, s, h, d)] = to_e4m3(fq[qi(b, s, h, d)] / dsq[b * HK + h / G]);
    for (int b = 0; b < B; b++)
        for (int s = 0; s < SK; s++) for (int h = 0; h < HK; h++) for (int d = 0; d < D; d++) {
            hk[ki(b, s, h, d)] = to_e4m3(fk[ki(b, s, h, d)] / dsk[b * HK + h]);
            hv[ki(b, s, h, d)] = to_e4m3(fv[ki(b, s, h, d)] / dsv[b * HK + h]);
        }
    uint8_t *dq, *dk, *dv;
    uint16_t *dout;
    float *dds, *dlse;
    HC(hipMalloc(&dq, nq)); HC(hipMalloc(&dk, nk)); HC(hipMalloc(&dv, nk)); HC(hipMalloc(&dout, no * 2));
    HC(hipMalloc(&dlse, (size_t)B * H * SQ * 4 + 256));
    HC(hipMalloc(&dds, 3 * B * HK * 4));
    HC(hipMemcpy(dq, hq.data(), nq, hipMemcpyHostToDevice));
    HC(hipMemcpy(dk, hk.data(), nk, hipMemcpyHostToDevice));
    HC(hipMemcpy(dv, hv.data(), nk, hipMemcpyHostToDevice));
    HC(hipMemcpy(dds, dsq.data(), B * HK * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(dds + B * HK, dsk.data(), B * HK * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(dds + 2 * B * HK, dsv.data(), B * HK * 4, hipMemcpyHostToDevice));
    const float scale = 1.0f / std::sqrt((float)D);
    // VSCALE_MUL: a factor on the softmax scale passed to the vendor kernel only (ABI probing)
    const float vmul = getenv("VSCALE_MUL") ? (float)atof(getenv("VSCALE_MUL")) : 1.0f;
    // KVS=n (2 / 4 / 8): our kernel is a `_kvsplit` form (kernarg 0x290 Ohat, 0x298 lse, 0x2A0 splits, 0x2A4 rows_total; grid z = B n)
    // merged by COMBINE_CO / COMBINE_KN (default fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h, 8 elements per thread); both are timed together
    const int kvs = envi("KVS", 0);
    void *dohat = nullptr, *dlsep = nullptr;
    const long rows_total = (long)B * SQ * H;
    if (kvs) { HC(hipMalloc(&dohat, (size_t)kvs * rows_total * D * 4)); HC(hipMalloc(&dlsep, (size_t)kvs * rows_total * 4)); }
    auto make = [&](const char* co, const char* kn, float smul) {
        Launch L;
        hipModule_t mod;
        HC(hipModuleLoad(&mod, co));
        HC(hipModuleGetFunction(&L.fn, mod, kn));
        L.a.assign(656, 0);
        auto P = [&](int i, const void* p) { memcpy(&L.a[16 * i], &p, 8); };
        auto U = [&](int i, uint32_t v) { memcpy(&L.a[16 * i], &v, 4); };
        auto F = [&](int i, float v) { memcpy(&L.a[16 * i], &v, 4); };
        const uint32_t qs = H * D, ks = HK * D, os = H * D * 2;
        P(0, dout); P(1, dq); P(2, dk); P(3, dv); P(4, nullptr); F(5, scale * smul);
        U(6, SQ); U(7, qs); U(8, 256 * qs); U(9, D); U(10, SQ * qs); U(11, G);
        U(12, ks); U(13, D); U(14, SK * ks);
        U(15, causal && (H % 8 != 0 || SQ > 16384) ? 3 : 5); U(16, 0); U(17, SK); U(18, D); U(19, D); U(20, H);
        U(21, ks); U(22, D); U(23, SK * ks);
        U(24, os); U(25, D * 2); U(26, SQ * os);
        U(29, SQ * 4);
        P(32, dds); P(33, dds + B * HK); P(34, dds + 2 * B * HK);
        // VDS_DIV: divide the vendor's descale strides (ABI probing: element vs byte strides)
        const uint32_t dv_ = smul != 1.0f || getenv("VDS_DIV") == nullptr || kn[0] != '_' ? 1 : (uint32_t)atoi(getenv("VDS_DIV"));
        const uint32_t hs = scalar_ds ? 0 : 4 / dv_, bs = scalar_ds ? 0 : HK * 4 / dv_;
        U(35, bs); U(36, hs); U(37, bs); U(38, hs); U(39, bs); U(40, hs);
        const int tiles = (SQ + 255) / 256;
        // OUR_PAIR=0: our kernel without the q-tile pairing (one q tile per block)
        const bool pair = causal && (kn[0] == '_' || envi("OUR_PAIR", 1));
        L.gx = pair ? (tiles + 1) / 2 : tiles; L.gy = H; L.gz = B; L.block = 512;
        if (kvs && kn[0] != '_') {
            L.a.resize(688, 0);
            memcpy(&L.a[0x290], &dohat, 8); memcpy(&L.a[0x298], &dlsep, 8);
            const uint32_t sp = kvs, rt = (uint32_t)rows_total;
            memcpy(&L.a[0x2A0], &sp, 4); memcpy(&L.a[0x2A4], &rt, 4);
            L.gx = tiles; L.gz = B * kvs;
            hipModule_t cm;
            HC(hipModuleLoad(&cm, getenv("COMBINE_CO")));
            HC(hipModuleGetFunction(&L.comb, cm, getenv("COMBINE_KN") ? getenv("COMBINE_KN") : "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"));
            L.ca.assign(48, 0);
            memcpy(&L.ca[0x00], &dohat, 8); memcpy(&L.ca[0x08], &dlsep, 8); memcpy(&L.ca[0x10], &dout, 8);
            const uint32_t d128 = D;
            memcpy(&L.ca[0x18], &rt, 4); memcpy(&L.ca[0x1C], &d128, 4); memcpy(&L.ca[0x20], &sp, 4);
            L.cgrid = (int)((rows_total * D / 8 + 255) / 256);
        }
        return L;
    };
    std::vector<Launch> Ls = {make(argv[1], argv[2], vmul), make(argv[3], argv[4], 1.0f)};
    const char* names[2] = {"vendor", "ours"};
    auto run = [&](Launch& L) {
        size_t asz = L.a.size();
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, L.a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
        HC(hipModuleLaunchKernel(L.fn, L.gx, L.gy, L.gz, L.block, 1, 1, 0, nullptr, nullptr, cfg));
        if (L.comb) {
            size_t csz = L.ca.size();
            void* ccfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, L.ca.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &csz, HIP_LAUNCH_PARAM_END};
            HC(hipModuleLaunchKernel(L.comb, L.cgrid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, ccfg));
        }
    };
    // sampled rows (first / last query of the first / last batch and head always)
    std::vector<int> cb(ncheck), ch(ncheck), cq(ncheck);
    for (int c = 0; c < ncheck; c++) {
        cb[c] = c == 0 ? 0 : c == 1 ? B - 1 : (int)(rng() % B);
        ch[c] = c == 0 ? 0 : c == 1 ? H - 1 : (int)(rng() % H);
        cq[c] = c == 0 ? 0 : c == 1 ? SQ - 1 : (int)(rng() % SQ);
    }
    std::vector<std::vector<double>> ref(ncheck, std::vector<double>(D));
    for (int c = 0; c < ncheck; c++) {
        int b = cb[c], h = ch[c], q = cq[c], g = h / G;
        int kmax = causal ? std::min(SK - 1, q + SK - SQ) : SK - 1;
        std::vector<double> s(SK);
        double m = -1e300;
        for (int k = 0; k <= kmax; k++) {
            double acc = 0;
            for (int d = 0; d < D; d++) acc += (double)e4m3_val(hq[qi(b, q, h, d)]) * e4m3_val(hk[ki(b, k, g, d)]);
            s[k] = acc * dsq[b * HK + g] * dsk[b * HK + g] * scale;
            m = std::max(m, s[k]);
        }
        double l = 0;
        for (int k = 0; k <= kmax; k++) {
            double p = std::exp(s[k] - m);
            l += p;
            for (int d = 0; d < D; d++) ref[c][d] += p * e4m3_val(hv[ki(b, k, g, d)]) * dsv[b * HK + g];
        }
        for (int d = 0; d < D; d++) ref[c][d] /= (kmax >= 0 ? l : 1);
    }
    int fails = 0;
    double mean_e[2] = {0, 0}, max_e[2] = {0, 0};
    long bads[2] = {0, 0};
    for (int w = 0; w < 2; w++) {
        HC(hipMemset(dout, 0, no * 2));
        run(Ls[w]);
        HC(hipDeviceSynchronize());
        std::vector<uint16_t> ho(no);
        HC(hipMemcpy(ho.data(), dout, no * 2, hipMemcpyDeviceToHost));
        long bad = 0;
        double maxe = 0, sume = 0, sumr = 0;
        for (int c = 0; c < ncheck; c++)
            for (int d = 0; d < D; d++) {
                double got = bf(ho[qi(cb[c], cq[c], ch[c], d)]), r = ref[c][d];
                double e = std::fabs(got - r);
                maxe = std::max(maxe, e);
                sume += e; sumr += std::fabs(r);
                // fp8 P (3-bit mantissa) and the bf16 output: 2 % of |ref| + 2e-2
                if (!(e <= 2e-2 * std::fabs(r) + 2e-2)) {
                    if (bad < envi("SHOW_BAD", 0)) printf("  %s bad b=%d h=%d q=%d d=%d got=%g ref=%g\n", names[w], cb[c], ch[c], cq[c], d, got, r);
                    bad++;
                }
            }
        printf("%s check: bad=%ld maxerr=%.3e mean_abs_err=%.3e mean_abs_ref=%.3e\n", names[w], bad, maxe, sume / (ncheck * D), sumr / (ncheck * D));
        mean_e[w] = sume / (ncheck * D); max_e[w] = maxe; bads[w] = bad;
    }
    // verdict on our kernel: no element outside the tolerance, or the project's vendor parity (docs/current-status.md: bad <= vendor and mean
    // error <= 1.10x the vendor's on the same data; meaningful with SCALAR_DS=1, the descale form the vendor binary honours)
    fails = bads[1] != 0 && !(bads[1] <= bads[0] && mean_e[1] <= 1.1 * mean_e[0]);
    printf("parity: ours/vendor mean_err %.3f, max_err %.3f, bad %ld / %ld\n", mean_e[1] / mean_e[0], max_e[1] / max_e[0], bads[1], bads[0]);
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    double best[2] = {1e30, 1e30};
    for (int r = 0; r < rounds; r++)
        for (int w = 0; w < 2; w++) {
            for (int i = 0; i < warm; i++) run(Ls[w]);
            HC(hipEventRecord(e0));
            for (int i = 0; i < iters; i++) run(Ls[w]);
            HC(hipEventRecord(e1));
            HC(hipEventSynchronize(e1));
            float ms; HC(hipEventElapsedTime(&ms, e0, e1));
            best[w] = std::min(best[w], 1e3 * ms / iters);
        }
    double flops = 4.0 * B * H * (double)SQ * SK * D * (causal ? 0.5 : 1.0);
    for (int w = 0; w < 2; w++) printf("%s B=%d H=%d HK=%d SQ=%d SK=%d causal=%d: %.2f us %.1f TF/s\n", names[w], B, H, HK, SQ, SK, causal, best[w], flops / (best[w] * 1e-6) / 1e12);
    printf("ratio vendor/ours = %.3f %s\n", best[0] / best[1], fails ? "FP8PT_FAIL" : "FP8PT_OK");
    return fails ? 3 : 0;
}
