// gfx1250 MLA decode host (replica harness): fp8 Q [B, H, 576] x fp8 paged KV [pages, 64, 1, 576] -> bf16 O [B, H, 512] (+ fp32 LSE), one
// query per sequence, one KV split, for a code object with the aiter gfx1250 MLA ABI (public aiter csrc/py_itfs_cu/asm_mla.cu: the 120-byte packed
// preload kernarg, block 128, grid (ceil(sq * gqa / sub_Q), B, kv_split), page size 64, fp32 per-tensor q / kv scales; with one split the kernel
// writes the final bf16 output into R).
//
// Device layouts follow aiter op_tests/test_mla_decode_pagesize64.py (Q heads padded to 768 bytes, KV pages split into nope / rope planes).
//
//   g1250_mla_host <co> <kernel> B ctx [H=16] [iters=20] [deadline_ms=10000]
// Env: SEED (1), CHECK_B (4 sequences checked against an fp32 CPU reference), SCALE (softmax scale, default 1/sqrt(576)), WARMUP (5).
// Prints `MLA_OK` / `MLA_FAIL`, time and KV bandwidth. Hang safety: polled deadlines; on expiry exit 124 without touching the device again.
#include <hip/hip_runtime.h>
#include "g1250_timing.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#include <unistd.h>

#define FAIL(...) do { std::printf("MLA_FAIL "); std::printf(__VA_ARGS__); std::printf("\n"); std::fflush(stdout); _exit(2); } while (0)
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) FAIL("%s: %s", #x, hipGetErrorString(e_)); } while (0)

static int envi(const char* n, int d) { const char* v = getenv(n); return v ? atoi(v) : d; }
static float e4m3(uint8_t b) {  // OCP e4m3fn
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.f + m / 8.f, e - 7);
    if (e == 15 && m == 7) v = NAN;
    return s ? -v : v;
}
static float bf16f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }

static void wait_deadline(hipEvent_t e, long ms, const char* what) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        hipError_t q = hipEventQuery(e);
        if (q == hipSuccess) return;
        if (q != hipErrorNotReady) FAIL("%s: %s", what, hipGetErrorString(q));
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(ms)) {
            std::printf("MLA_HANG %s exceeded %ld ms\n", what, ms);
            std::fflush(stdout);
            _exit(124);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s co kernel B ctx [H] [iters] [deadline_ms]\n", argv[0]); return 1; }
    const char* co = argv[1];
    const char* kn = argv[2];
    const int B = atoi(argv[3]), ctx = atoi(argv[4]);
    const int H = argc > 5 ? atoi(argv[5]) : 16;
    const int iters = argc > 6 ? atoi(argv[6]) : 20;
    const long deadline = argc > 7 ? atol(argv[7]) : 10000;
    const int D = 576, DV = 512, PS = 64;
    if (B < 1 || ctx < 1 || B > 65535) FAIL("shape");
    const float sm = getenv("SCALE") ? atof(getenv("SCALE")) : 1.f / std::sqrt((float)D);
    const float qs = 0.5f, ks = 0.25f;

    std::mt19937 rng(envi("SEED", 1));
    auto f8 = [&]() -> uint8_t { uint8_t b; do { b = rng() & 0xFF; } while (((b >> 3) & 15) > 9 || (b & 0x7F) == 0x7F); return b; };  // |x| < 2^3
    const int pages_per = (ctx + PS - 1) / PS, npages = B * pages_per;
    std::vector<uint8_t> Q((size_t)B * H * D), KV((size_t)npages * PS * D);
    for (auto& x : Q) x = f8();
    for (auto& x : KV) x = f8();
    std::vector<int> perm(npages);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> kv_indptr(B + 1), page_idx(npages), last(B), qo_indptr(B + 1), vsc(B, 1), splits_indptr(B + 1);
    for (int b = 0; b <= B; b++) { kv_indptr[b] = b * pages_per; qo_indptr[b] = b; splits_indptr[b] = b; }
    for (int i = 0; i < npages; i++) page_idx[i] = perm[i];
    for (int b = 0; b < B; b++) last[b] = ctx - PS * (pages_per - 1);

    HC(hipSetDevice(0));
    auto up = [](const void* h, size_t n) { void* d; HC(hipMalloc(&d, n)); HC(hipMemcpy(d, h, n, hipMemcpyHostToDevice)); return d; };
    // device layouts (aiter op_tests/test_mla_decode_pagesize64.py): Q heads padded to a 768-byte row ([nope 512][rope 64][zero 192]);
    // KV pages "rope split2": per page the 64 x 512 nope plane, then the 64 x 64 rope plane
    const int QP = 768;
    std::vector<uint8_t> Qd((size_t)B * H * QP, 0), KVd(KV.size());
    for (size_t r = 0; r < (size_t)B * H; r++) std::memcpy(&Qd[r * QP], &Q[r * D], D);
    for (int pg = 0; pg < npages; pg++)
        for (int t = 0; t < PS; t++) {
            const uint8_t* src = &KV[((size_t)pg * PS + t) * D];
            std::memcpy(&KVd[(size_t)pg * PS * D + (size_t)t * DV], src, DV);
            std::memcpy(&KVd[(size_t)pg * PS * D + (size_t)PS * DV + (size_t)t * (D - DV)], src + DV, D - DV);
        }
    void* dQ = up(Qd.data(), Qd.size());
    void* dKV = up(KVd.data(), KVd.size());
    void* dLTP = up(kv_indptr.data(), kv_indptr.size() * 4);
    void* dLTD = up(page_idx.data(), page_idx.size() * 4);
    void* dLTL = up(last.data(), last.size() * 4);
    void* dQTP = up(qo_indptr.data(), qo_indptr.size() * 4);
    void* dQS = up(&qs, 4);
    void* dKS = up(&ks, 4);
    void* dVSC = up(vsc.data(), vsc.size() * 4);
    void *dR, *dLSE;
    HC(hipMalloc(&dR, (size_t)B * H * DV * 2));
    HC(hipMalloc(&dLSE, (size_t)B * H * 4));
    HC(hipMemset(dR, 0xFF, (size_t)B * H * DV * 2));

    struct __attribute__((packed)) KA {
        void *R, *LSE, *Q, *KV, *LTP, *LTD, *LTL, *QTP, *QS, *KS, *VSC;
        uint32_t out16;
        float scalar;
        uint32_t qlens, passes, stride_page, log2_page, stride_q, flags;
    } ka{dR, dLSE, dQ, dKV, dLTP, dLTD, dLTL, dQTP, dQS, dKS, dVSC, 1u, sm, (uint32_t)H, 1u, (uint32_t)(PS * D), 6u, (uint32_t)(H * D), 0u};
    static_assert(sizeof(KA) == 120, "kernarg size");
    size_t ksz = sizeof(ka);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ka, HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksz, HIP_LAUNCH_PARAM_END};
    hipModule_t mod;
    hipFunction_t f;
    HC(hipModuleLoad(&mod, co));
    HC(hipModuleGetFunction(&f, mod, kn));
    // PAIRED mode: CO2 [KERNEL2] = a second kernel of the same ABI, timed alternately with the first on the same buffers (tests/gpu/g1250_timing.h)
    hipFunction_t f2 = nullptr;
    if (const char* co2 = getenv("CO2")) {
        hipModule_t mod2;
        HC(hipModuleLoad(&mod2, co2));
        HC(hipModuleGetFunction(&f2, mod2, getenv("KERNEL2") ? getenv("KERNEL2") : kn));
    }
    hipStream_t s;
    HC(hipStreamCreate(&s));
    const unsigned gdx = (H + H - 1) / H;  // sub_Q = gqa * sq = H for one query per sequence
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    std::printf("launch grid %ux%dx1 block 128\n", gdx, B);
    std::fflush(stdout);
    HC(hipEventRecord(e0, s));
    HC(hipModuleLaunchKernel(f, gdx, B, 1, 128, 1, 1, 0, s, nullptr, cfg));
    HC(hipEventRecord(e1, s));
    wait_deadline(e1, deadline, "first launch");
    std::vector<uint16_t> R((size_t)B * H * DV);
    HC(hipMemcpy(R.data(), dR, R.size() * 2, hipMemcpyDeviceToHost));

    // fp32 reference: logits = (q qs) . (k ks) * scale over 576 dims, softmax over ctx, out = p . (v ks) over the first 512 dims
    const int cb = std::min(B, envi("CHECK_B", 4));
    long bad = 0, checked = 0;
    double maxe = 0, dot = 0, nn = 0;
    std::vector<double> lg(ctx);
    for (int bi = 0; bi < cb; bi++) {
        const int b = cb == 1 ? 0 : (int)((long)bi * (B - 1) / (cb - 1));
        for (int h = 0; h < H; h++) {
            const uint8_t* q = &Q[((size_t)b * H + h) * D];
            double mx = -1e30;
            for (int t = 0; t < ctx; t++) {
                const uint8_t* k = &KV[((size_t)page_idx[kv_indptr[b] + t / PS] * PS + t % PS) * D];
                double a = 0;
                for (int d = 0; d < D; d++) a += (double)e4m3(q[d]) * e4m3(k[d]);
                lg[t] = a * qs * ks * sm;
                mx = std::max(mx, lg[t]);
            }
            double z = 0;
            for (int t = 0; t < ctx; t++) z += std::exp(lg[t] - mx);
            for (int d = 0; d < DV; d++) {
                double o = 0;
                for (int t = 0; t < ctx; t++) {
                    const uint8_t* v = &KV[((size_t)page_idx[kv_indptr[b] + t / PS] * PS + t % PS) * D];
                    o += std::exp(lg[t] - mx) / z * e4m3(v[d]) * ks;
                }
                const double got = bf16f(R[((size_t)b * H + h) * DV + d]);
                if (envi("DEBUG", 0) && bi == 0 && h < 2 && d < 6) {
                    // alternatives: no V scale; logits without q / kv scales; plain mean of V (uniform softmax)
                    double o2 = 0, mean = 0, z2 = 0, mx2 = -1e30;
                    std::vector<double> l2(ctx);
                    for (int t = 0; t < ctx; t++) { l2[t] = lg[t] / (qs * ks); mx2 = std::max(mx2, l2[t]); }
                    for (int t = 0; t < ctx; t++) z2 += std::exp(l2[t] - mx2);
                    for (int t = 0; t < ctx; t++) {
                        const uint8_t* v = &KV[((size_t)page_idx[kv_indptr[b] + t / PS] * PS + t % PS) * D];
                        o2 += std::exp(l2[t] - mx2) / z2 * e4m3(v[d]) * ks;
                        mean += e4m3(v[d]) * ks / ctx;
                    }
                    std::printf("dbg h=%d d=%d got=%.5f ref=%.5f noV=%.5f noQKscale=%.5f mean=%.5f\n", h, d, got, o, o / ks, o2, mean);
                }
                // aiter's check (test_mla_decode_pagesize64.py): |got - ref| <= 6e-2 + 6e-2 |ref| on >= 95 % of the elements (the kernel
                // rounds P to fp8 for the PV product), plus the cosine difference
                const double err = std::fabs(got - o);
                maxe = std::max(maxe, err);
                if (!(err <= 6e-2 + 6e-2 * std::fabs(o))) bad++;
                dot += got * o;
                nn += got * got + o * o;
                checked++;
            }
        }
    }
    // timing: median of SAMPLES samples of back-to-back launches (>= MIN_SAMPLE_MS each), FLUSH=1 evicts the caches before every launch (tests/gpu/g1250_timing.h); `iters` is kept for the command line and ignored
    (void)iters;
    const G1250Timing tmg = g1250_time([&](hipStream_t st) { HC(hipModuleLaunchKernel(f, gdx, B, 1, 128, 1, 1, 0, st, nullptr, cfg)); }, s, wait_deadline, deadline);
    const double us = tmg.median_us;
    if (f2) {
        const G1250Paired pr = g1250_time_pair([&](hipStream_t st) { HC(hipModuleLaunchKernel(f, gdx, B, 1, 128, 1, 1, 0, st, nullptr, cfg)); }, [&](hipStream_t st) { HC(hipModuleLaunchKernel(f2, gdx, B, 1, 128, 1, 1, 0, st, nullptr, cfg)); }, s, wait_deadline, deadline);
        std::printf("{\"host\":\"g1250_mla\",\"paired\":true,\"kernel\":\"%s\",\"us\":%.3f,\"us2\":%.3f,\"ratio\":%.4f,\"pair_spread\":%.4f,\"ok\":true}\n", kn, pr.a_us, pr.b_us, pr.ratio_ba, pr.spread_ratio);
    }
    const double cosd = 1 - 2 * dot / std::max(nn, 1e-12);
    const bool ok = bad <= checked / 20 && cosd < 1e-3;
    std::printf("mla B=%d H=%d ctx=%d: %.2f us %.1f GB/s (KV bytes) checked=%ld bad=%ld max_abs=%.3e cos_diff=%.2e %s\n", B, H, ctx, us,
                (double)B * ctx * D / (us * 1e-6) / 1e9, checked, bad, maxe, cosd, ok ? "MLA_OK" : "MLA_FAIL");
    std::printf("{\"host\":\"g1250_mla\",\"kernel\":\"%s\",\"shape\":{\"B\":%d,\"H\":%d,\"ctx\":%d},\"grid\":[%u,%d,1],\"cluster\":[0,0,0],"
                "\"us\":%.3f,\"gbs\":%.1f,\"bad\":%ld,\"cos_diff\":%.3e,\"ok\":%s}\n", kn, B, H, ctx, gdx, B, us, (double)B * ctx * D / (us * 1e-6) / 1e9, bad,
                cosd, ok ? "true" : "false");
    return ok ? 0 : 3;
}
