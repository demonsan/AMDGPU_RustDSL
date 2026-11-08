// Implicit-GEMM convolution host (bf16, NDHWC / NHWC): ours `conv{2,3}d_bf16_*` (weights [K][C / 64][T][R][S][64] + shuffle_weight(16, 16));
// CPU reference on sampled outputs; TF/s of the equivalent GEMM (2 M K C T R S).
//
//   conv_host <co> <kernel> NB DI H W C K T R S stride pad dil [cta]
//   (2d: DI = 1, T = 1; depth stride 1, depth padding 0). env: ITERS (20), SEED (1), CHECK (256)
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

int main(int argc, char** argv) {
    if (argc < 15) { fprintf(stderr, "usage: %s co kernel NB DI H W C K T R S stride pad dil [cta]\n", argv[0]); return 1; }
    int a = 3;
    const int NB = atoi(argv[a++]), DI = atoi(argv[a++]), H = atoi(argv[a++]), W = atoi(argv[a++]), C = atoi(argv[a++]), K = atoi(argv[a++]);
    const int T = atoi(argv[a++]), R = atoi(argv[a++]), S = atoi(argv[a++]), st = atoi(argv[a++]), pad = atoi(argv[a++]), dil = atoi(argv[a++]);
    const int cta = argc > a ? atoi(argv[a]) : 256;
    // SPLITK=S (the _c128_sk kernel): fp32 partials [S][MP][K] + gemm_splitk_reduce_s<S> (REDUCE_CO)
    const int S = envi("SPLITK", 1);
    const int sd = T == 1 ? 1 : st, pd = T == 1 ? 0 : pad;
    const int OD = (DI + 2 * pd - dil * (T - 1) - 1) / sd + 1, OH = (H + 2 * pad - dil * (R - 1) - 1) / st + 1, OW = (W + 2 * pad - dil * (S - 1) - 1) / st + 1;
    const long M = (long)NB * OD * OH * OW;
    const int KR = C * T * R * S, CB = C / 64;
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<uint16_t> x((size_t)NB * DI * H * W * C), w((size_t)K * T * R * S * C);
    for (auto& v : x) v = tobf(N(rng));
    for (auto& v : w) v = tobf(N(rng) * 0.05f);
    // logical w [K][T][R][S][C] -> ours [K][CB][T][R][S][64] -> shuffle_weight(16, 16) of [K, KR] bf16
    std::vector<uint16_t> wl(w.size()), ws(w.size());
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++)
            for (int r = 0; r < R; r++)
                for (int s = 0; s < S; s++)
                    for (int c = 0; c < C; c++) {
                        size_t src = ((((size_t)k * T + t) * R + r) * S + s) * C + c;
                        size_t dst = (size_t)k * KR + ((((size_t)(c / 64) * T + t) * R + r) * S + s) * 64 + c % 64;
                        wl[dst] = w[src];
                    }
    const int Kp = 8, BK = 32;
    for (long n = 0; n < K; n++)
        for (int kk = 0; kk < KR; kk++) {
            long n0 = n / 16, ni = n % 16, k0 = kk / BK, k1 = (kk % BK) / Kp, kj = kk % Kp;
            ws[(size_t)n0 * 16 * KR + (size_t)k0 * 16 * BK + (size_t)k1 * 16 * Kp + ni * Kp + kj] = wl[(size_t)n * KR + kk];
        }
    uint16_t *dx, *dw, *dy;
    HC(hipMalloc(&dx, x.size() * 2 + 256)); HC(hipMalloc(&dw, ws.size() * 2 + 256)); HC(hipMalloc(&dy, (size_t)M * K * 2 + 256));
    HC(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dw, ws.data(), ws.size() * 2, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn;
    HC(hipModuleLoad(&mod, argv[1])); HC(hipModuleGetFunction(&fn, mod, argv[2]));
    std::vector<uint8_t> ab(128, 0);
    auto P = [&](int o, const void* p) { memcpy(&ab[o], &p, 8); };
    auto U = [&](int o, int v) { memcpy(&ab[o], &v, 4); };
    const long MP = (M + cta - 1) / cta * cta;
    float* dws = nullptr;
    if (S > 1) HC(hipMalloc(&dws, (size_t)S * MP * K * 4 + 256));
    P(0, dx); P(8, dw); P(16, S > 1 ? (void*)dws : (void*)dy); U(24, (int)M); U(28, K); U(32, KR); U(36, S); U(40, KR); U(44, KR); U(48, K); U(80, (int)M);
    U(84, KR * 2 / 128); U(88, KR * 2 / 128 / S); U(92, (int)MP);
    U(96, NB); U(100, DI); U(104, H); U(108, W); U(112, C); U(116, OD); U(120, OH); U(124, OW);
    size_t asz = ab.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ab.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    hipFunction_t rfn = nullptr;
    std::vector<uint8_t> ra(48, 0);
    if (S > 1) {
        hipModule_t rm;
        HC(hipModuleLoad(&rm, getenv("REDUCE_CO")));
        std::string rk = "gemm_splitk_reduce_s" + std::to_string(S) + "_n256";
        HC(hipModuleGetFunction(&rfn, rm, rk.c_str()));
        int mi = (int)M, mpi = (int)MP;
        memcpy(&ra[0], &dws, 8); memcpy(&ra[8], &dy, 8); memcpy(&ra[32], &mi, 4); memcpy(&ra[36], &K, 4); memcpy(&ra[40], &S, 4); memcpy(&ra[44], &mpi, 4);
    }
    size_t rsz = ra.size();
    void* rcfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ra.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &rsz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() {
        HC(hipModuleLaunchKernel(fn, K / cta, (int)((M + cta - 1) / cta), S, 256, 1, 1, 0, nullptr, nullptr, cfg));
        if (rfn) HC(hipModuleLaunchKernel(rfn, K / 256, (int)M, 1, 64, 1, 1, 0, nullptr, nullptr, rcfg));
    };
    launch();
    HC(hipDeviceSynchronize());
    std::vector<uint16_t> y((size_t)M * K);
    HC(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0;
    double me = 0;
    for (int c = 0; c < envi("CHECK", 256); c++) {
        long m = c == 0 ? 0 : c == 1 ? M - 1 : (long)(rng() % M);
        int k = rng() % K;
        long rem = m;
        int ow = rem % OW; rem /= OW;
        int oh = rem % OH; rem /= OH;
        int od = rem % OD; rem /= OD;
        int nb = (int)rem;
        double acc = 0, mag = 0;
        for (int t = 0; t < T; t++)
            for (int r = 0; r < R; r++)
                for (int s = 0; s < S; s++) {
                    int zi = od * sd - pd + t * dil, yi = oh * st - pad + r * dil, xi = ow * st - pad + s * dil;
                    if (zi < 0 || zi >= DI || yi < 0 || yi >= H || xi < 0 || xi >= W) continue;
                    for (int ci = 0; ci < C; ci++) {
                        double p = (double)bf(x[((((size_t)nb * DI + zi) * H + yi) * W + xi) * C + ci]) * bf(w[((((size_t)k * T + t) * R + r) * S + s) * C + ci]);
                        acc += p;
                        mag += std::fabs(p);
                    }
                }
        double got = bf(y[(size_t)m * K + k]), e = std::fabs(got - acc);
        me = std::max(me, e);
        if (e > 1e-2 * std::fabs(acc) + 1e-3 * mag + 1e-3) {
            if (bad < envi("SHOW_BAD", 0)) printf("  bad m=%ld (nb %d od %d oh %d ow %d) k=%d got %g ref %g\n", m, nb, od, oh, ow, k, got, acc);
            bad++;
        }
    }
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    for (int i = 0; i < 3; i++) launch();
    HC(hipEventRecord(e0));
    int iters = envi("ITERS", 20);
    for (int i = 0; i < iters; i++) launch();
    HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
    float ms; HC(hipEventElapsedTime(&ms, e0, e1));
    double us = 1e3 * ms / iters, fl = 2.0 * M * K * (double)KR;
    printf("%s NB=%d DI=%d H=%d W=%d C=%d K=%d T=%d R=%d S=%d st=%d pad=%d dil=%d -> M=%ld (OD %d OH %d OW %d) KR=%d: %.2f us %.1f TF/s bad=%ld maxerr=%.3e %s\n",
           argv[2], NB, DI, H, W, C, K, T, R, S, st, pad, dil, M, OD, OH, OW, KR, us, fl / (us * 1e-6) / 1e12, bad, me, bad ? "CONV_FAIL" : "CONV_OK");
    return bad ? 3 : 0;
}
