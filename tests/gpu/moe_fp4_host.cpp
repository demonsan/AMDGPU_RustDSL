// MXFP4 fused-MoE host: ours gemm_moe1_fp4_<act>[_bm64] + gemm_moe2_fp4[_bm64] + moe_sum_bf16_k<TOPK> on aiter-style sorted routes.
// X [M][D] and W1 [E][2I][D], W2 [E][D][I] are fp4 e2m1 codes with 1 x 32 E8M0 scales (random, exact data); the stage-1 output is
// re-quantized to MXFP4 on the host (aiter rule) for stage 2. CPU references: stage 1 on the exact data, stage 2 on the uploaded bytes.
//   moe_fp4_host <co_dir> M D I E TOPK   (D % 1024 == 0, I % 256 == 0)   env: BLOCK_M (128 | 64), ITERS (20), SEED (1), CHECK (tokens checked, 16)
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
static float e2m1(int c) { static const float t[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6}; return (c & 8 ? -1 : 1) * t[c & 7]; }
static float e8(int e) { return std::ldexp(1.0f, e - 127); }
template <class T> static T* dev(const std::vector<T>& h) { T* d; HC(hipMalloc(&d, h.size() * sizeof(T) + 256)); HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }
// aiter shuffle_weight(16, 16) of [rows][Kb] bytes
static std::vector<uint8_t> shuffle_b(const std::vector<uint8_t>& src, long rows, int Kb) {
    std::vector<uint8_t> dst(src.size());
    for (long n = 0; n < rows; n++)
        for (int k0 = 0; k0 < Kb; k0 += 16) memcpy(&dst[(size_t)(n / 16) * 16 * Kb + (size_t)(k0 / 16) * 256 + (n % 16) * 16], &src[(size_t)n * Kb + k0], 16);
    return dst;
}
// fp4 codes + e8m0 scales of a [rows][K] matrix: random (exact data)
struct Mx { std::vector<uint8_t> q, s; int rows, K; float v(size_t r, int k) const { int c = (q[(r * K + k) / 2] >> (4 * (k & 1))) & 15; return e2m1(c) * e8(s[r * (K / 32) + k / 32]); } };
static Mx rand_mx(std::mt19937& rng, int rows, int K, int e0) {
    Mx m{std::vector<uint8_t>((size_t)rows * K / 2), std::vector<uint8_t>((size_t)rows * K / 32), rows, K};
    for (auto& b : m.q) b = rng() & 0xff;
    for (auto& s : m.s) s = e0 + rng() % 3;
    return m;
}
// aiter dynamic_mxfp4_quant of one row group of 32
static void quant32(const float* x, uint8_t* codes /*32 nibbles*/, uint8_t& e) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[i]));
    uint32_t u; memcpy(&u, &amax, 4); u = (u + 0x200000) & 0xff800000u;
    int ee = (int)((u >> 23) & 0xff) - 2;
    e = (uint8_t)std::min(254, std::max(0, ee));
    float sc = e8(std::max((int)e, 1));
    for (int i = 0; i < 32; i++) {
        float v = x[i] / sc;
        int best = 0; float bd = INFINITY;
        for (int c = 0; c < 16; c++) { float d = std::fabs(e2m1(c) - v); if (d < bd) { bd = d; best = c; } }
        codes[i] = (uint8_t)best;
    }
}
int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s co_dir M D I E TOPK\n", argv[0]); return 1; }
    const std::string dir = argv[1];
    const int M = atoi(argv[2]), D = atoi(argv[3]), I = atoi(argv[4]), E = atoi(argv[5]), TK = atoi(argv[6]);
    const int BM = envi("BLOCK_M", 128), iters = envi("ITERS", 20), CHK = envi("CHECK", 16);
    std::mt19937 rng(envi("SEED", 1));
    // ---- routing + aiter moe_sorting ----
    std::vector<int> ids((size_t)M * TK);
    std::vector<float> tw((size_t)M * TK);
    for (int t = 0; t < M; t++) {
        std::vector<int> e(E);
        for (int i = 0; i < E; i++) e[i] = i;
        std::shuffle(e.begin(), e.end(), rng);
        float s = 0;
        for (int k = 0; k < TK; k++) { ids[t * TK + k] = e[k]; tw[t * TK + k] = 0.1f + (rng() % 1000) / 1000.0f; s += tw[t * TK + k]; }
        for (int k = 0; k < TK; k++) tw[t * TK + k] /= s;
    }
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
    Mx X = rand_mx(rng, M, D, 125), W1 = rand_mx(rng, E * 2 * I, D, 122), W2 = rand_mx(rng, E * D, I, 122);
    // A16=1 (gemm_moe*_bf16w4*_bm64): bf16 activations X (exact values k / 8), the fp4 weights K-permuted inside every 256-K tile (the
    // A16W4 GEMM layout, L(p) below), stage 2 takes the bf16 stage-1 output as is
    const bool a16 = envi("A16", 0) != 0;
    std::vector<uint16_t> xb16((size_t)M * D);
    std::vector<float> xf((size_t)M * D);
    for (size_t i = 0; i < xb16.size(); i++) { float v = ((int)(rng() % 33) - 16) / 8.0f; uint32_t u; memcpy(&u, &v, 4); xb16[i] = (uint16_t)(u >> 16); xf[i] = v; }
    auto xval = [&](size_t t, int d) -> double { return a16 ? xf[t * D + d] : X.v(t, d); };
    auto kperm = [&](std::vector<uint8_t>& q, long rows, int K) {
        std::vector<uint8_t> o(q.size(), 0);
        for (long r = 0; r < rows; r++)
            for (int k = 0; k < K; k++) {
                int T = k / 256, p = k % 256;
                int L = 128 * (p / 128) + 16 * ((p / 64) % 2) + 8 * ((p / 32) % 2) + 32 * ((p / 8) % 4) + p % 8;
                size_t src = (size_t)r * K + 256 * T + L, dst = (size_t)r * K + k;
                o[dst / 2] |= ((q[src / 2] >> (4 * (src & 1))) & 15) << (4 * (dst & 1));
            }
        q.swap(o);
    };
    // ours: W1 gate / up rows interleaved in 16-row blocks (row 32 q + r = gate 16 q + r, 32 q + 16 + r = up 16 q + r), scales the same order
    std::vector<uint8_t> w1q(W1.q.size()), w1s(W1.s.size());
    for (int e = 0; e < E; e++)
        for (int r = 0; r < 2 * I; r++) {
            int q = r / 32, rr = r % 32, src = rr < 16 ? 16 * q + rr : I + 16 * q + rr - 16;
            size_t d0 = (size_t)e * 2 * I + r, s0 = (size_t)e * 2 * I + src;
            memcpy(&w1q[d0 * D / 2], &W1.q[s0 * D / 2], D / 2);
            memcpy(&w1s[d0 * D / 32], &W1.s[s0 * D / 32], D / 32);
        }
    std::vector<uint8_t> w2q = W2.q;
    if (a16) { kperm(w1q, (long)E * 2 * I, D); kperm(w2q, (long)E * D, I); }
    auto w1sh = shuffle_b(w1q, (long)E * 2 * I, D / 2), w2sh = shuffle_b(w2q, (long)E * D, I / 2);
    void *dX = a16 ? (void*)dev(xb16) : (void*)dev(X.q), *dXs = dev(X.s), *dW1 = dev(w1sh), *dW1s = dev(w1s), *dW2 = dev(w2sh), *dW2s = dev(W2.s);
    void *dsid = dev(sid), *dseid = dev(seid), *dnv = dev(nv), *dsw = dev(sw);
    void *dY1, *dY1q, *dY1s, *dT, *dO;
    HC(hipMalloc(&dY1, (size_t)M * TK * I * 2)); HC(hipMalloc(&dY1q, (size_t)M * TK * I / 2 + 256)); HC(hipMalloc(&dY1s, (size_t)M * TK * I / 32 + 256));
    HC(hipMalloc(&dT, (size_t)M * TK * D * 4)); HC(hipMalloc(&dO, (size_t)M * D * 2));
    auto load = [&](const std::string& name) { hipModule_t m; hipFunction_t f; HC(hipModuleLoad(&m, (dir + "/" + name + ".co").c_str())); HC(hipModuleGetFunction(&f, m, name.c_str())); return f; };
    const std::string bms = BM == 64 ? "_bm64" : "";
    // OSS=1: the gpt-oss form (_oss): per-expert fp32 biases b1 [E][2I] (gate then up), b2 [E][D]; clamped swiglu
    const bool oss = envi("OSS", 0) != 0;
    std::vector<float> b1((size_t)E * 2 * I), b2((size_t)E * D);
    for (auto& v : b1) v = ((int)(rng() % 64) - 32) / 8.0f;
    for (auto& v : b2) v = ((int)(rng() % 64) - 32) / 8.0f;
    void *db1 = dev(b1), *db2 = dev(b2);
    const std::string dt = a16 ? "bf16w4" : "fp4";
    hipFunction_t f1 = load("gemm_moe1_" + dt + (oss ? "_oss" : "_silu") + bms), f2 = load("gemm_moe2_" + dt + (oss ? "_oss" : "") + bms),
                  fs = load("moe_sum_bf16_k" + std::to_string(TK));
    auto gemm_args = [&](void* A, void* B, void* C, int m, int n, int k, void* sa, void* sb, int nout) {
        std::vector<uint8_t> a(oss ? 152 : 144, 0);
        auto P = [&](int o, const void* p) { memcpy(&a[o], &p, 8); };
        auto U = [&](int o, int v) { memcpy(&a[o], &v, 4); };
        P(0, A); P(8, B); P(16, C); U(24, m); U(28, n); U(32, k); U(36, 1); U(40, k); U(44, k); U(48, nout);
        P(64, sa); P(72, sb); U(80, m); U(84, k / 256);
        P(96, dsid); P(104, dseid); P(112, dnv); P(120, dsw); U(128, TK); U(132, E); U(136, nout); U(140, M);
        return a;
    };
    auto a1 = gemm_args(dX, dW1, dY1, M, 2 * I, D, dXs, dW1s, I);
    auto a2 = gemm_args(a16 ? dY1 : dY1q, dW2, dT, M * TK, D, I, dY1s, dW2s, D);
    if (oss) { memcpy(&a1[144], &db1, 8); memcpy(&a2[144], &db2, 8); }
    std::vector<uint8_t> as(24, 0);
    memcpy(&as[0], &dT, 8); memcpy(&as[8], &dO, 8); memcpy(&as[16], &M, 4); memcpy(&as[20], &D, 4);
    auto launch = [&](hipFunction_t f, std::vector<uint8_t>& a, int gx, int gy) {
        size_t sz = a.size();
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
        HC(hipModuleLaunchKernel(f, gx, gy, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    };
    // ---- stage 1 + check ----
    launch(f1, a1, 2 * I / 256, nblk);
    HC(hipDeviceSynchronize());
    std::vector<uint16_t> y1((size_t)M * TK * I);
    HC(hipMemcpy(y1.data(), dY1, y1.size() * 2, hipMemcpyDeviceToHost));
    long bad1 = 0; double me1 = 0;
    for (int t = 0; t < std::min(M, CHK); t++)
        for (int k = 0; k < TK; k++) {
            int e = ids[t * TK + k];
            for (int c = 0; c < I; c++) {
                double g = 0, u = 0;
                for (int d = 0; d < D; d++) { double x = xval(t, d); g += x * W1.v((size_t)e * 2 * I + c, d); u += x * W1.v((size_t)e * 2 * I + I + c, d); }
                if (oss) { g += b1[(size_t)e * 2 * I + c]; u += b1[(size_t)e * 2 * I + I + c]; g = std::min(g, 7.0); u = std::max(std::min(u, 7.0), -7.0); }
                double ref = oss ? (u + 1) * g / (1 + std::exp(-1.702 * g)) : g / (1 + std::exp(-g)) * u, got = bf(y1[((size_t)t * TK + k) * I + c]);
                double err = std::fabs(got - ref);
                me1 = std::max(me1, err / (std::fabs(ref) + 1e-3));
                if (err > 2e-2 * std::fabs(ref) + 2e-2) { if (bad1 < envi("SHOW_BAD", 0)) printf("  s1 t=%d k=%d c=%d got %g ref %g\n", t, k, c, got, ref); bad1++; }
            }
        }
    // ---- host requant of y1 to MXFP4 (aiter rule), upload ----
    std::vector<uint8_t> yq((size_t)M * TK * I / 2, 0), ys((size_t)M * TK * I / 32);
    for (size_t r = 0; r < (size_t)M * TK; r++)
        for (int g = 0; g < I / 32; g++) {
            float x[32]; uint8_t c[32];
            for (int i = 0; i < 32; i++) x[i] = bf(y1[r * I + 32 * g + i]);
            quant32(x, c, ys[r * (I / 32) + g]);
            for (int i = 0; i < 32; i++) { size_t p = r * I + 32 * g + i; yq[p / 2] |= c[i] << (4 * (p & 1)); }
        }
    HC(hipMemcpy(dY1q, yq.data(), yq.size(), hipMemcpyHostToDevice)); HC(hipMemcpy(dY1s, ys.data(), ys.size(), hipMemcpyHostToDevice));
    Mx YQ{yq, ys, M * TK, I};
    auto yval = [&](size_t r, int i) -> double { return a16 ? (double)bf(y1[r * I + i]) : YQ.v(r, i); };
    // ---- stage 2 + sum + check ----
    launch(f2, a2, D / 128, nblk);
    launch(fs, as, D / 1024, M);
    HC(hipDeviceSynchronize());
    std::vector<uint16_t> out((size_t)M * D);
    HC(hipMemcpy(out.data(), dO, out.size() * 2, hipMemcpyDeviceToHost));
    long bad2 = 0; double me2 = 0;
    for (int t = 0; t < std::min(M, CHK); t++)
        for (int d = 0; d < D; d++) {
            double ref = 0;
            for (int k = 0; k < TK; k++) {
                int e = ids[t * TK + k];
                double acc = 0;
                for (int i = 0; i < I; i++) acc += yval((size_t)t * TK + k, i) * W2.v((size_t)e * D + d, i);
                if (oss) acc += b2[(size_t)e * D + d];
                ref += tw[t * TK + k] * acc;
            }
            double got = bf(out[(size_t)t * D + d]), err = std::fabs(got - ref);
            me2 = std::max(me2, err / (std::fabs(ref) + 1e-3));
            if (err > 2e-2 * std::fabs(ref) + 2e-2) { if (bad2 < envi("SHOW_BAD", 0)) printf("  s2 t=%d d=%d got %g ref %g\n", t, d, got, ref); bad2++; }
        }
    // ---- timing (stage 1; stage 2 + sum) ----
    hipEvent_t e0, e1, e2;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1)); HC(hipEventCreate(&e2));
    for (int i = 0; i < 3; i++) { launch(f1, a1, 2 * I / 256, nblk); launch(f2, a2, D / 128, nblk); launch(fs, as, D / 1024, M); }
    HC(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch(f1, a1, 2 * I / 256, nblk);
    HC(hipEventRecord(e1));
    for (int i = 0; i < iters; i++) { launch(f2, a2, D / 128, nblk); launch(fs, as, D / 1024, M); }
    HC(hipEventRecord(e2)); HC(hipEventSynchronize(e2));
    float ms1, ms2; HC(hipEventElapsedTime(&ms1, e0, e1)); HC(hipEventElapsedTime(&ms2, e1, e2));
    double us1 = 1e3 * ms1 / iters, us2 = 1e3 * ms2 / iters, fl1 = 2.0 * M * TK * 2 * I * D, fl2 = 2.0 * M * TK * D * I;
    printf("ours fp4 M=%d D=%d I=%d E=%d topk=%d bm=%d: stage1 %.2f us (%.1f TF/s) stage2+sum %.2f us (%.1f TF/s) bad1=%ld bad2=%ld maxrel1=%.3e maxrel2=%.3e %s\n",
           M, D, I, E, TK, BM, us1, fl1 / (us1 * 1e-6) / 1e12, us2, fl2 / (us2 * 1e-6) / 1e12, bad1, bad2, me1, me2, bad1 == 0 && bad2 == 0 ? "MOE_OK" : "MOE_FAIL");
    return 0;
}
