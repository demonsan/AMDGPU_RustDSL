// gfx1250 MLA decode host (replica harness): fp8 Q [B, H, 576] x fp8 paged KV [pages, 64, 1, 576] -> bf16 O [B, H, 512] (+ fp32 LSE), one
// query per sequence, one KV split, for a code object with the aiter gfx1250 MLA ABI (public aiter csrc/py_itfs_cu/asm_mla.cu: the 120-byte packed
// preload kernarg, block 128, grid (ceil(sq * gqa / sub_Q), B, kv_split), page size 64, fp32 per-tensor q / kv scales; with one split the kernel
// writes the final bf16 output into R).
//
//   g1250_mla_host <co> <kernel> B ctx [H=16] [iters=20] [deadline_ms=10000]
// Env: SEED (1), CHECK_B (4 sequences checked against an fp32 CPU reference), SCALE (softmax scale, default 1/sqrt(576)), WARMUP (5).
// Prints `MLA_OK` / `MLA_FAIL`, time and KV bandwidth. Hang safety: polled deadlines; on expiry exit 124 without touching the device again.
#include <hip/hip_runtime.h>
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
    void* dQ = up(Q.data(), Q.size());
    void* dKV = up(KV.data(), KV.size());
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
    double maxe = 0;
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
                const double err = std::fabs(got - o) / std::max(0.05, std::fabs(o));
                maxe = std::max(maxe, err);
                if (!(err < 0.02)) bad++;
                checked++;
            }
        }
    }
    const int warm = envi("WARMUP", 5);
    for (int i = 0; i < warm; i++) HC(hipModuleLaunchKernel(f, gdx, B, 1, 128, 1, 1, 0, s, nullptr, cfg));
    HC(hipEventRecord(e0, s));
    for (int i = 0; i < iters; i++) HC(hipModuleLaunchKernel(f, gdx, B, 1, 128, 1, 1, 0, s, nullptr, cfg));
    HC(hipEventRecord(e1, s));
    wait_deadline(e1, deadline + (long)iters * 1000, "timed launches");
    float ms = 0;
    HC(hipEventElapsedTime(&ms, e0, e1));
    const double us = 1e3 * ms / iters;
    std::printf("mla B=%d H=%d ctx=%d: %.2f us %.1f GB/s (KV bytes) checked=%ld bad=%ld max_rel=%.3e %s\n", B, H, ctx, us,
                (double)B * ctx * D / (us * 1e-6) / 1e9, checked, bad, maxe, bad ? "MLA_FAIL" : "MLA_OK");
    return bad ? 3 : 0;
}
