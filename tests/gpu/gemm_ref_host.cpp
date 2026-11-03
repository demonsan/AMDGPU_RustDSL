// MXFP8 blockscale GEMM reference harness (aiter PR #6696 ABI, `opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950`, 96 bytes):
//   C[m, n] (bf16) = sum_k A[m, k] * 2^(sa[k/128][m] - 127) * B[n, k] * 2^(sb[n/128][k/128] - 127)
//   A  [M, K] fp8 e4m3 row-major;  B [N, K] fp8 e4m3 preshuffled with aiter shuffle_weight(layout=(16,16)):
//      Bsh[n0][k0][kk][ni][ki] = B[16 n0 + ni][32 k0 + 16 kk + ki]   (n0 < N/16, k0 < K/32, kk < 2, ni < 16, ki < 16)
//   sa E8M0 [K/128][M] (column-major view of the 1x128 x_scale, stride_sfa = M);  sb E8M0 [N/128][K/128] row-major (stride_sfb = K/128)
// Launches one or two code objects with the same ABI on identical inputs (REF_CO = the aiter .co, OURS_CO = ours, optional), checks a sample
// of output rows against a double-precision CPU reference and times each kernel (median of REF_REPS repetitions of REF_ITERS launches).
//
// env: GEMM_M/N/K (4096/4096/8192), REF_CO + REF_KNAME (symbol), OURS_CO + OURS_KNAME, REF_ROWS (rows checked, 64), REF_WARMUP (300),
//      REF_ITERS (100), REF_REPS (7), GRID_MODE (nm: grid(N/256, M/256) [default]; xcd: grid(roundup8(tiles)); linear: grid(tiles)),
//      TILE_M/TILE_N (256), BLOCK (256), SEED
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static double fp8_e4m3(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    double v;
    if (e == 0xF && m == 0x7) v = 0.0;
    else if (e == 0) v = (m / 8.0) * std::ldexp(1.0, -6);
    else v = (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return s ? -v : v;
}
static float bf2f(uint16_t x) { uint32_t u = (uint32_t)x << 16; float f; std::memcpy(&f, &u, 4); return f; }
static int envi(const char* n, int d) { const char* v = std::getenv(n); return v && *v ? std::atoi(v) : d; }

struct Kargs {
    const void* ptr_a; const void* ptr_b; void* ptr_c;
    int m, n, k, batch, stride_a, stride_b, stride_c, stride_a_batch, stride_b_batch, stride_c_batch;
    const void* ptr_sfa; const void* ptr_sfb;
    int stride_sfa, stride_sfb, stride_sfa_batch, stride_sfb_batch;
};
static_assert(sizeof(Kargs) == 96, "ABI");

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int M = envi("GEMM_M", 4096), N = envi("GEMM_N", 4096), K = envi("GEMM_K", 8192);
    const int TM = envi("TILE_M", 256), TN = envi("TILE_N", 256), BLOCK = envi("BLOCK", 256);
    if (M % TM || N % TN || K % 128) { std::printf("shape must be a multiple of the tile (%dx%d) and K of 128\n", TM, TN); return 1; }
    const int KB = K / 128;
    unsigned seed = (unsigned)envi("SEED", 20261103);
    auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    // fp8 values: sign, exponent 4..9 (|x| in [2^-3, 2^3)), any mantissa (no NaN/subnormal)
    auto rfp8 = [&]() -> unsigned char { unsigned r = rnd(); return (unsigned char)(((r & 1) << 7) | ((4 + (r >> 1) % 6) << 3) | ((r >> 4) & 7)); };
    std::vector<unsigned char> A((size_t)M * K), B((size_t)N * K), Bsh((size_t)N * K), sa((size_t)KB * M), sb((size_t)(N / 128) * KB);
    for (auto& x : A) x = rfp8();
    for (auto& x : B) x = rfp8();
    for (auto& x : sa) x = (unsigned char)(124 + rnd() % 7);
    for (auto& x : sb) x = (unsigned char)(124 + rnd() % 7);
    // aiter shuffle_weight(layout=(16,16)) for 1-byte elements
    for (int n0 = 0; n0 < N / 16; ++n0)
        for (int k0 = 0; k0 < K / 32; ++k0)
            for (int kk = 0; kk < 2; ++kk)
                for (int ni = 0; ni < 16; ++ni)
                    for (int ki = 0; ki < 16; ++ki) {
                        const size_t dst = ((((size_t)n0 * (K / 32) + k0) * 2 + kk) * 16 + ni) * 16 + ki;
                        Bsh[dst] = B[(size_t)(16 * n0 + ni) * K + 32 * k0 + 16 * kk + ki];
                    }
    void *dA, *dB, *dC, *dSa, *dSb;
    CK(hipMalloc(&dA, A.size())); CK(hipMalloc(&dB, Bsh.size())); CK(hipMalloc(&dC, (size_t)M * N * 2));
    CK(hipMalloc(&dSa, sa.size())); CK(hipMalloc(&dSb, sb.size()));
    CK(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dB, Bsh.data(), Bsh.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dSa, sa.data(), sa.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dSb, sb.data(), sb.size(), hipMemcpyHostToDevice));
    Kargs ka{};
    ka.ptr_a = dA; ka.ptr_b = dB; ka.ptr_c = dC; ka.m = M; ka.n = N; ka.k = K; ka.batch = 1;
    ka.stride_a = K; ka.stride_b = K; ka.stride_c = N;
    ka.stride_a_batch = M * K; ka.stride_b_batch = N * K; ka.stride_c_batch = (int)((uint64_t)M * N);
    ka.ptr_sfa = dSa; ka.ptr_sfb = dSb; ka.stride_sfa = M; ka.stride_sfb = KB;
    ka.stride_sfa_batch = M * KB; ka.stride_sfb_batch = (N / 128) * KB;

    // CPU reference on a sample of rows (spread over all M tiles), every column
    const int rows = std::min(M, envi("REF_ROWS", 64));
    std::vector<int> rlist; for (int i = 0; i < rows; ++i) rlist.push_back((int)(((long long)i * 2654435761LL) % M));
    std::vector<double> ref((size_t)rows * N);
    {
        std::vector<double> af(K), bf((size_t)K);
        std::vector<double> bdq((size_t)N * K);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) bdq[(size_t)n * K + k] = fp8_e4m3(B[(size_t)n * K + k]) * std::ldexp(1.0, (int)sb[(size_t)(n / 128) * KB + k / 128] - 127);
        for (int i = 0; i < rows; ++i) {
            const int m = rlist[i];
            for (int k = 0; k < K; ++k) af[k] = fp8_e4m3(A[(size_t)m * K + k]) * std::ldexp(1.0, (int)sa[(size_t)(k / 128) * M + m] - 127);
            for (int n = 0; n < N; ++n) { double acc = 0; const double* bp = &bdq[(size_t)n * K]; for (int k = 0; k < K; ++k) acc += af[k] * bp[k]; ref[(size_t)i * N + n] = acc; }
        }
    }
    const int iters = envi("REF_ITERS", 100), warm = envi("REF_WARMUP", 300), reps = envi("REF_REPS", 7);
    const double flops = 2.0 * M * N * K;
    const char* gm = std::getenv("GRID_MODE");
    const std::string grid_mode = gm && *gm ? gm : "nm";
    int rc = 0;
    for (const char* which : {"REF", "OURS"}) {
        // per-kernel CTA tile (REF_TILE / OURS_TILE, default TILE_M x TILE_N): grid (N / tile, M / tile)
        const int tm = envi((std::string(which) + "_TILE").c_str(), TM), tn = envi((std::string(which) + "_TILE").c_str(), TN);
        const int tiles = (M / tm) * (N / tn);
        dim3 grid = grid_mode == "xcd" ? dim3((tiles + 7) / 8 * 8) : grid_mode == "linear" ? dim3(tiles) : dim3(N / tn, M / tm);
        const char* co = std::getenv((std::string(which) + "_CO").c_str());
        const char* kn = std::getenv((std::string(which) + "_KNAME").c_str());
        if (!co || !*co) continue;
        hipModule_t mod; hipFunction_t fn;
        CK(hipModuleLoad(&mod, co));
        CK(hipModuleGetFunction(&fn, mod, kn));
        size_t asz = sizeof(ka);
        void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ka, HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
        auto launch = [&] { CK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, BLOCK, 1, 1, 0, nullptr, nullptr, extra)); };
        CK(hipMemset(dC, 0xFF, (size_t)M * N * 2));   // NaN pattern: unwritten outputs fail the check
        launch(); CK(hipDeviceSynchronize());
        std::vector<uint16_t> C((size_t)M * N);
        CK(hipMemcpy(C.data(), dC, C.size() * 2, hipMemcpyDeviceToHost));
        size_t bad = 0, cnt = 0; double worst = 0, sum = 0, maxref = 0;
        for (int i = 0; i < rows; ++i)
            for (int n = 0; n < N; ++n) {
                const double r = ref[(size_t)i * N + n], g = bf2f(C[(size_t)rlist[i] * N + n]);
                const double e = std::fabs(g - r); ++cnt;
                if (!(e <= 1e-2 * std::fabs(r) + 1e-3 * 4.0)) { if (bad < 4) std::fprintf(stderr, "%s C[%d][%d] want %.6f got %.6f\n", which, rlist[i], n, r, g); ++bad; }
                if (e == e) { worst = std::max(worst, e / (std::fabs(r) + 1e-3)); sum += e; } else worst = INFINITY;
                maxref = std::max(maxref, std::fabs(r));
            }
        std::printf("%s %s: checked=%zu bad=%zu worst_rel=%.3e mean_abs=%.3e max|ref|=%.3f -> %s\n", which, kn, cnt, bad, worst, sum / cnt, maxref, bad ? "GEMM_FAIL" : "GEMM_OK");
        if (bad) rc = 1;
        if (iters > 0) {
            for (int w = 0; w < warm; ++w) launch();
            CK(hipDeviceSynchronize());
            std::vector<double> tfs;
            hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
            for (int r = 0; r < reps; ++r) {
                CK(hipEventRecord(e0));
                for (int it = 0; it < iters; ++it) launch();
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
                tfs.push_back(flops / ((double)ms / iters * 1e-3) / 1e12);
            }
            std::vector<double> s = tfs; std::sort(s.begin(), s.end());
            std::printf("timing %s: median %.2f TFLOP/s (%.4f ms) over %d reps x %d iters; runs:", which, s[s.size() / 2], flops / (s[s.size() / 2] * 1e12) * 1e3, reps, iters);
            for (double t : tfs) std::printf(" %.1f", t);
            std::printf("  [M=%d N=%d K=%d grid=(%u,%u) block=%d]\n", M, N, K, grid.x, grid.y, BLOCK);
        }
        CK(hipModuleUnload(mod));
    }
    return rc;
}
