// Fused-MoE host: ours (`gemm_moe1_<dt>_<act>` + `gemm_moe2_<dt>` + `moe_sum_bf16_k<K>`) on aiter-style sorted routes, and (VENDOR1=<co>,
// fp8 only) aiter's 2-stage ASM stage 1 `fmoe_stage1_bf16_pertokenFp8_g1u1_*` on the same data; CPU reference on sampled tokens.
//
//   moe_host <bf16|fp8> <co_dir> M D I E TOPK [silu|gelu]
//   co_dir holds gemm_moe1_<dt>_<act>.co, gemm_moe2_<dt>.co, moe_sum_bf16_k<TOPK>.co
//   env: ITERS (20), WARMUP (5), SEED (1), CHECK (16 tokens), BLOCK_M (128; vendor stage 1 = its tile_m), VENDOR1 / VENDOR1_KN
// Weights: W1 [E, 2I, D] (gate rows, then up rows), W2 [E, D, I], both in aiter's shuffle_weight(16, 16) layout. fp8: per-token X scale,
// per-row weight scales; the stage-1 output is re-quantized per row on the host for stage 2 (the vendor flow quantizes it in a kernel).
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
static float e4m3_to_f(uint8_t b) { int s = b >> 7, e = (b >> 3) & 15, m = b & 7; float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7); return s ? -v : v; }
static uint8_t f_to_e4m3(float f) {  // round to nearest (by search; fine for a host test)
    static std::vector<float> tab;
    if (tab.empty()) for (int i = 0; i < 256; i++) tab.push_back((i & 0x7f) == 0x7f ? 1e30f : e4m3_to_f((uint8_t)i));
    int best = 0; float bd = 1e30f;
    for (int i = 0; i < 256; i++) { float d = std::fabs(tab[i] - f); if (d < bd) { bd = d; best = i; } }
    return (uint8_t)best;
}
template <class T> static T* dev(const std::vector<T>& h) { T* d; HC(hipMalloc(&d, h.size() * sizeof(T) + 256)); HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }
// aiter shuffle_weight(16, 16) of [rows, K] with `es`-byte elements
static std::vector<uint8_t> shuffle_b(const std::vector<uint8_t>& src, long rows, int K, int es) {
    const int Kp = 16 / es, BK = 32;
    std::vector<uint8_t> dst(src.size());
    for (long n = 0; n < rows; n++)
        for (int k = 0; k < K; k++) {
            long n0 = n / 16, ni = n % 16, k0 = k / BK, kk = (k % BK) / Kp, kj = k % Kp;
            size_t d = (size_t)n0 * 16 * K + (size_t)k0 * 16 * BK + (size_t)kk * 16 * Kp + (size_t)ni * Kp + kj;
            memcpy(&dst[d * es], &src[((size_t)n * K + k) * es], es);
        }
    return dst;
}

int main(int argc, char** argv) {
    if (argc < 9) { fprintf(stderr, "usage: %s bf16|fp8 co_dir M D I E TOPK [silu|gelu]\n", argv[0]); return 1; }
    const bool fp8 = std::string(argv[1]) == "fp8";
    const std::string dir = argv[2];
    const int M = atoi(argv[3]), D = atoi(argv[4]), I = atoi(argv[5]), E = atoi(argv[6]), TK = atoi(argv[7]);
    const std::string act = argc > 8 ? argv[8] : "silu";
    const bool gelu = act == "gelu";
    const int BM = envi("BLOCK_M", 128), iters = envi("ITERS", 20), warm = envi("WARMUP", 5);
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0.0f, 1.0f);
    // ---- routing: TOPK distinct experts per token, weights normalized ----
    std::vector<int> ids((size_t)M * TK);
    std::vector<float> tw((size_t)M * TK);
    for (int t = 0; t < M; t++) {
        std::vector<int> e(E);
        for (int i = 0; i < E; i++) e[i] = i;
        std::shuffle(e.begin(), e.end(), rng);
        if (envi("SKIP_E0", 0)) for (int k = 0; k < TK; k++) if (e[k] == 0) std::swap(e[k], e[E - 1]);
        float s = 0;
        for (int k = 0; k < TK; k++) { ids[t * TK + k] = e[k]; tw[t * TK + k] = 0.1f + (rng() % 1000) / 1000.0f; s += tw[t * TK + k]; }
        for (int k = 0; k < TK; k++) tw[t * TK + k] /= s;
    }
    // ---- moe_sorting (aiter format) ----
    int maxp = M * TK + E * BM - TK;
    const int nblk = (maxp + BM - 1) / BM;
    maxp = nblk * BM;
    std::vector<int> sid(maxp, M | (TK << 24)), seid(nblk, E), nv(2);
    std::vector<float> sw(maxp, 0.0f);
    int pos = 0;
    for (int e = 0; e < E; e++) {
        int start = pos;
        for (int t = 0; t < M; t++)
            for (int k = 0; k < TK; k++)
                if (ids[t * TK + k] == e) { sid[pos] = t | (k << 24); sw[pos] = tw[t * TK + k]; pos++; }
        if (pos == start) continue;
        int pad = (pos + BM - 1) / BM * BM;
        for (int b = start / BM; b < pad / BM; b++) seid[b] = e;
        pos = pad;
    }
    nv[0] = pos; nv[1] = M;
    // ---- data ----
    const int es = fp8 ? 1 : 2;
    std::vector<float> xf((size_t)M * D), w1f((size_t)E * 2 * I * D), w2f((size_t)E * D * I), xs(M, 1.0f), w1s((size_t)E * 2 * I, 1.0f), w2s((size_t)E * D, 1.0f);
    std::vector<uint8_t> xb((size_t)M * D * es), w1b(w1f.size() * es), w2b(w2f.size() * es);
    auto fill = [&](std::vector<float>& f, std::vector<uint8_t>& raw, float sc, std::vector<float>* rs, int rowlen) {
        for (size_t i = 0; i < f.size(); i++) {
            float v = N(rng) * sc;
            if (fp8) {
                // per-row scales: values quantized as code * rs[row]
                float s = (*rs)[i / rowlen];
                uint8_t c = f_to_e4m3(v / s);
                raw[i] = c;
                f[i] = e4m3_to_f(c) * s;
            } else {
                uint16_t h = tobf(v);
                memcpy(&raw[2 * i], &h, 2);
                f[i] = bf(h);
            }
        }
    };
    if (fp8) {
        for (auto& s : xs) s = 0.05f + (rng() % 100) / 1000.0f;
        for (auto& s : w1s) s = 0.005f + (rng() % 100) / 20000.0f;
        for (auto& s : w2s) s = 0.005f + (rng() % 100) / 20000.0f;
    }
    fill(xf, xb, 1.0f, &xs, D);
    fill(w1f, w1b, 0.05f, &w1s, D);
    fill(w2f, w2b, 0.05f, &w2s, I);
    // ours: gate / up rows interleaved in 16-row blocks (row 32 q + r = gate 16 q + r, 32 q + 16 + r = up 16 q + r); the vendor: stacked
    std::vector<uint8_t> w1i(w1b.size());
    std::vector<float> w1si(w1s.size());
    for (int e = 0; e < E; e++)
        for (int r = 0; r < 2 * I; r++) {
            int q = r / 32, rr = r % 32, src = rr < 16 ? 16 * q + rr : I + 16 * q + rr - 16;
            memcpy(&w1i[(((size_t)e * 2 * I) + r) * D * es], &w1b[(((size_t)e * 2 * I) + src) * D * es], (size_t)D * es);
            w1si[(size_t)e * 2 * I + r] = w1s[(size_t)e * 2 * I + src];
        }
    auto w1sh = shuffle_b(w1i, (long)E * 2 * I, D, es);
    auto w1sh_v = shuffle_b(w1b, (long)E * 2 * I, D, es);
    auto w2sh = shuffle_b(w2b, (long)E * D, I, es);
    void *dX = dev(xb), *dW1 = dev(w1sh), *dW2 = dev(w2sh), *dsid = dev(sid), *dseid = dev(seid), *dnv = dev(nv), *dsw = dev(sw);
    void *dxs = dev(xs), *dw1s = dev(w1si), *dw2s = dev(w2s), *dW1v = dev(w1sh_v), *dw1sv = dev(w1s);
    void *dY1, *dY1q, *dy1s, *dT, *dO;
    HC(hipMalloc(&dY1, (size_t)M * TK * I * 2)); HC(hipMalloc(&dY1q, (size_t)M * TK * I)); HC(hipMalloc(&dy1s, (size_t)M * TK * 4));
    HC(hipMalloc(&dT, (size_t)M * TK * D * 4)); HC(hipMalloc(&dO, (size_t)M * D * 2));
    // ---- kernels ----
    std::string dt = fp8 ? "fp8" : "bf16";
    auto load = [&](const std::string& name) { hipModule_t m; hipFunction_t f; HC(hipModuleLoad(&m, (dir + "/" + name + ".co").c_str())); HC(hipModuleGetFunction(&f, m, name.c_str())); return f; };
    hipFunction_t f1 = load("gemm_moe1_" + dt + "_" + act), f2 = load("gemm_moe2_" + dt), fs = load("moe_sum_bf16_k" + std::to_string(TK));
    // 96-byte GEMM ABI + MoE extension (144 bytes)
    auto gemm_args = [&](void* A, void* B, void* C, int m, int n, int k, void* sa, void* sb, int nout) {
        std::vector<uint8_t> a(144, 0);
        auto P = [&](int o, const void* p) { memcpy(&a[o], &p, 8); };
        auto U = [&](int o, int v) { memcpy(&a[o], &v, 4); };
        P(0, A); P(8, B); P(16, C); U(24, m); U(28, n); U(32, k); U(36, 1); U(40, k); U(44, k); U(48, nout);
        P(64, sa); P(72, sb); U(80, m); U(84, k * es / 128);
        P(96, dsid); P(104, dseid); P(112, dnv); P(120, dsw); U(128, TK); U(132, E); U(136, nout); U(140, M);
        return a;
    };
    auto a1 = gemm_args(dX, dW1, dY1, M, 2 * I, D, dxs, dw1s, I);
    auto a2 = gemm_args(fp8 ? dY1q : dY1, dW2, dT, M * TK, D, I, dy1s, dw2s, D);
    std::vector<uint8_t> as(24, 0);
    memcpy(&as[0], &dT, 8); memcpy(&as[8], &dO, 8); memcpy(&as[16], &M, 4); memcpy(&as[20], &D, 4);
    auto launch = [&](hipFunction_t f, std::vector<uint8_t>& a, int gx, int gy) {
        size_t sz = a.size();
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
        HC(hipModuleLaunchKernel(f, gx, gy, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    };
    // fp8: the stage-1 output re-quantized per row on the host (between the stages)
    auto requant = [&]() {
        std::vector<uint16_t> y((size_t)M * TK * I);
        HC(hipMemcpy(y.data(), dY1, y.size() * 2, hipMemcpyDeviceToHost));
        std::vector<uint8_t> q(y.size());
        std::vector<float> s(M * TK);
        for (int r = 0; r < M * TK; r++) {
            float amax = 1e-12f;
            for (int c = 0; c < I; c++) amax = std::max(amax, std::fabs(bf(y[(size_t)r * I + c])));
            s[r] = amax / 448.0f;
            for (int c = 0; c < I; c++) q[(size_t)r * I + c] = f_to_e4m3(bf(y[(size_t)r * I + c]) / s[r]);
        }
        HC(hipMemcpy(dY1q, q.data(), q.size(), hipMemcpyHostToDevice));
        HC(hipMemcpy(dy1s, s.data(), s.size() * 4, hipMemcpyHostToDevice));
    };
    auto run = [&](bool host_q) {
        launch(f1, a1, I / envi("UP_N", 128), nblk);
        if (fp8 && host_q) { HC(hipDeviceSynchronize()); requant(); }
        launch(f2, a2, D / 128, nblk);
        launch(fs, as, D / 1024, M);
    };
    HC(hipMemset(dY1, 0, (size_t)M * TK * I * 2));
    run(true);
    HC(hipDeviceSynchronize());
    std::vector<uint16_t> out((size_t)M * D), y1((size_t)M * TK * I);
    HC(hipMemcpy(out.data(), dO, out.size() * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(y1.data(), dY1, y1.size() * 2, hipMemcpyDeviceToHost));
    if (envi("MAPDBG", 0)) {
        // full stage-1 reference for every (token, slot), then: for wrong entries, where does the GPU value occur in the reference?
        std::vector<double> ref((size_t)M * TK * I);
        for (int t = 0; t < M; t++)
            for (int k = 0; k < TK; k++) {
                int e = ids[t * TK + k];
                for (int i = 0; i < I; i++) {
                    double g = 0, u = 0;
                    for (int d = 0; d < D; d++) { g += (double)xf[(size_t)t * D + d] * w1f[((size_t)e * 2 * I + i) * D + d]; u += (double)xf[(size_t)t * D + d] * w1f[((size_t)e * 2 * I + I + i) * D + d]; }
                    ref[((size_t)t * TK + k) * I + i] = g / (1 + std::exp(-g)) * u;
                }
            }
        int shown = 0;
        for (size_t x = 0; x < ref.size() && shown < 12; x++) {
            double got = bf(y1[x]);
            if (std::fabs(got - ref[x]) <= 2e-2 * std::fabs(ref[x]) + 2e-2) continue;
            long best = -1;
            for (size_t y = 0; y < ref.size(); y++) if (std::fabs(bf(y1[x]) - ref[y]) < 1e-2 * std::fabs(ref[y]) + 1e-3 && got != 0) { best = y; break; }
            // candidates: gate / up of every row i' of this token (which pair would produce got?)
            int t = (int)(x / I / TK), e = ids[(x / I)], ii = (int)(x % I);
            std::vector<double> G(2 * I);
            for (int r = 0; r < 2 * I; r++) { double s = 0; for (int d = 0; d < D; d++) s += (double)xf[(size_t)t * D + d] * w1f[((size_t)e * 2 * I + r) * D + d]; G[r] = s; }
            int bg = -1, bu = -1;
            for (int a = 0; a < 2 * I && bg < 0; a++) for (int c = 0; c < 2 * I; c++) { double v = G[a] / (1 + std::exp(-G[a])) * G[c]; if (std::fabs(v - got) < 3e-3 * std::fabs(got) + 1e-3) { bg = a; bu = c; break; } }
            printf("  wrong row %zu col %d got %g ref %g; = silu(row %d) * row %d (stacked rows: gate < %d)\n", x / I, ii, got, ref[x], bg, bu, I);
            shown++;
        }
    }
    // ---- CPU reference (stage 1 exact, stage 2 from the GPU's y1 so the bf16 rounding of y1 matches) ----
    long bad1 = 0, bad2 = 0;
    double me1 = 0, me2 = 0;
    std::vector<double> tmp(D);
    for (int c = 0; c < envi("CHECK", 16); c++) {
        int t = c == 0 ? 0 : c == 1 ? M - 1 : (int)(rng() % M);
        std::fill(tmp.begin(), tmp.end(), 0.0);
        for (int k = 0; k < TK; k++) {
            int e = ids[t * TK + k];
            std::vector<double> ycpu(I);
            for (int i = 0; i < I; i++) {
                double g = 0, u = 0;
                for (int d = 0; d < D; d++) {
                    g += (double)xf[(size_t)t * D + d] * w1f[((size_t)e * 2 * I + i) * D + d];
                    u += (double)xf[(size_t)t * D + d] * w1f[((size_t)e * 2 * I + I + i) * D + d];
                }
                double a = gelu ? 0.5 * g * (1 + std::tanh(0.7978845608 * (g + 0.044715 * g * g * g))) : g / (1 + std::exp(-g));
                ycpu[i] = a * u;
                double got = bf(y1[((size_t)t * TK + k) * I + i]);
                double err = std::fabs(got - ycpu[i]);
                me1 = std::max(me1, err);
                if (err > 2e-2 * std::fabs(ycpu[i]) + 2e-2) {
                    if (bad1 < envi("SHOW_BAD", 0)) { int p = 0; while (p < maxp && sid[p] != (t | (k << 24))) p++; printf("  bad1 t=%d k=%d e=%d i=%d tile=%d row=%d got=%g ref=%g\n", t, k, e, i, p / BM, p % BM, got, ycpu[i]); }
                    bad1++;
                }
            }
            // stage 2 from the GPU y1 (quantized as the GPU consumed it for fp8)
            std::vector<double> yin(I);
            if (fp8) {
                float amax = 1e-12f;
                for (int i = 0; i < I; i++) amax = std::max(amax, std::fabs(bf(y1[((size_t)t * TK + k) * I + i])));
                float s = amax / 448.0f;
                for (int i = 0; i < I; i++) yin[i] = e4m3_to_f(f_to_e4m3(bf(y1[((size_t)t * TK + k) * I + i]) / s)) * s;
            } else {
                for (int i = 0; i < I; i++) yin[i] = bf(y1[((size_t)t * TK + k) * I + i]);
            }
            for (int d = 0; d < D; d++) {
                double acc = 0;
                for (int i = 0; i < I; i++) acc += yin[i] * w2f[((size_t)e * D + d) * I + i];
                tmp[d] += tw[t * TK + k] * acc;
            }
        }
        for (int d = 0; d < D; d++) {
            double got = bf(out[(size_t)t * D + d]), err = std::fabs(got - tmp[d]);
            me2 = std::max(me2, err);
            if (err > 2e-2 * std::fabs(tmp[d]) + 2e-2) bad2++;
        }
    }
    // ---- timing (stage 1, stage 2 + sum; the fp8 host re-quantization excluded) ----
    hipEvent_t e0, e1, e2;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1)); HC(hipEventCreate(&e2));
    for (int i = 0; i < warm; i++) run(false);
    HC(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch(f1, a1, I / envi("UP_N", 128), nblk);
    HC(hipEventRecord(e1));
    for (int i = 0; i < iters; i++) { launch(f2, a2, D / 128, nblk); launch(fs, as, D / 1024, M); }
    HC(hipEventRecord(e2));
    HC(hipEventSynchronize(e2));
    float t1, t2;
    HC(hipEventElapsedTime(&t1, e0, e1)); HC(hipEventElapsedTime(&t2, e1, e2));
    double fl1 = 2.0 * M * TK * 2.0 * I * D, fl2 = 2.0 * M * TK * (double)I * D;
    printf("ours %s %s M=%d D=%d I=%d E=%d topk=%d: stage1 %.2f us (%.1f TF/s) stage2+sum %.2f us (%.1f TF/s) bad1=%ld bad2=%ld maxerr1=%.3e maxerr2=%.3e %s\n", dt.c_str(),
           act.c_str(), M, D, I, E, TK, 1e3 * t1 / iters, fl1 / (1e-3 * t1 / iters) / 1e12, 1e3 * t2 / iters, fl2 / (1e-3 * t2 / iters) / 1e12, bad1, bad2, me1, me2,
           bad1 || bad2 ? "MOE_FAIL" : "MOE_OK");
    // ---- vendor stage 1 (fp8 per-token g1u1, same sorted buffers; block_m must equal its tile_m) ----
    if (const char* vco = getenv("VENDOR1")) {
        hipModule_t vm; hipFunction_t vf;
        HC(hipModuleLoad(&vm, vco));
        HC(hipModuleGetFunction(&vf, vm, getenv("VENDOR1_KN")));
        void* dV;
        HC(hipMalloc(&dV, (size_t)M * TK * I * 2));
        HC(hipMemset(dV, 0, (size_t)M * TK * I * 2));
        std::vector<uint8_t> va(368, 0);
        auto P = [&](int o, const void* p) { memcpy(&va[o], &p, 8); };
        auto U = [&](int o, int v) { memcpy(&va[o], &v, 4); };
        P(0, dV); P(16, dX); P(32, dW1v); P(48, dnv); P(64, dxs); P(80, dw1sv); P(96, nullptr); P(112, dsid); P(128, dseid);
        U(144, D); U(160, I); U(176, M); U(192, E); U(208, D); U(224, D); U(240, TK * I * 2); U(256, D * 2 * I); U(272, 2 * I * 4); U(288, I * 4);
        U(304, TK); U(320, 0); U(336, gelu ? 1 : 0); P(352, nullptr);
        int tn = envi("VENDOR1_TN", 128);
        launch(vf, va, I / tn, nblk);
        HC(hipDeviceSynchronize());
        std::vector<uint16_t> vy((size_t)M * TK * I);
        HC(hipMemcpy(vy.data(), dV, vy.size() * 2, hipMemcpyDeviceToHost));
        double md = 0;
        long vbad = 0;
        for (size_t i = 0; i < vy.size(); i++) {
            double a = bf(vy[i]), b = bf(y1[i]);
            md = std::max(md, std::fabs(a - b));
            if (std::fabs(a - b) > 2e-2 * std::fabs(b) + 2e-2) vbad++;
        }
        HC(hipEventRecord(e0));
        for (int i = 0; i < iters; i++) launch(vf, va, I / tn, nblk);
        HC(hipEventRecord(e1));
        HC(hipEventSynchronize(e1));
        HC(hipEventElapsedTime(&t1, e0, e1));
        printf("vendor stage1 %s: %.2f us (%.1f TF/s) vs ours y1: maxdiff=%.3e diffs=%ld\n", getenv("VENDOR1_KN"), 1e3 * t1 / iters, fl1 / (1e-3 * t1 / iters) / 1e12, md, vbad);
    }
    return bad1 || bad2 ? 3 : 0;
}
