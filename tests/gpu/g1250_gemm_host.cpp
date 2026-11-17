// Host of the gfx1250 TDM-ring WMMA GEMM (crates/amdgpu-kernels/src/probes/g1250_gemm.rs): C[M][N] f32 = A[M][K] x B[N][K]^T, fp8 e4m3 operands, against a CPU reference.
//
//   g1250_gemm_host <co> <kernel> [M=32] [N=32] [K=192] [iters=0]
//
// Kernargs: a @0, b @8, c @16 (pointers), k @24, n @28, passes @32 (K / 192, the looped kernel; ignored by the unrolled ones). Grid (N / 32, M / 32), 128 work-items.
// Hang policy (docs/mi450-nodes.md): every wait is polled with a wall-clock limit (PROBE_WATCHDOG_S, default 15); on expiry the program prints GEMM_HANG and leaves with _exit(124) without touching the device again.
// One JSON line per run: {"host":"g1250_gemm","kernel":..,"M":..,"N":..,"K":..,"maxerr":..,"maxrel":..,"bad":N,"us":..,"tflops":..,"ok":bool}
#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); _exit(2); } } while (0)

static double g_limit = 15.0;
static void wait_polled(hipEvent_t ev, const char* what) {
    auto t0 = std::chrono::steady_clock::now();
    while (hipEventQuery(ev) == hipErrorNotReady) {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s > g_limit) { printf("GEMM_HANG: %s still running after %.0f s; leaving without touching the device (exit 124)\n", what, s); fflush(stdout); _exit(124); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// fp8 e4m3 (OCP "fn": bias 7, no infinities, 0x7f / 0xff are NaN)
static float dec_e4m3(uint8_t v) {
    int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    if (e == 15 && m == 7) return NAN;
    float x = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -x : x;
}

static void put64(std::vector<uint8_t>& a, size_t off, const void* p) { memcpy(&a[off], &p, 8); }
static void put32(std::vector<uint8_t>& a, size_t off, uint32_t v) { memcpy(&a[off], &v, 4); }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <co> <kernel> [M N K iters]\n", argv[0]); return 1; }
    const char* kn = argv[2];
    const int M = argc > 3 ? atoi(argv[3]) : 32, N = argc > 4 ? atoi(argv[4]) : 32, K = argc > 5 ? atoi(argv[5]) : 192;
    const int iters = argc > 6 ? atoi(argv[6]) : 0;
    if (M % 32 || N % 32 || K % 192) { fprintf(stderr, "M, N multiples of 32 and K a multiple of 192 are required\n"); return 1; }
    if (const char* w = getenv("PROBE_WATCHDOG_S")) g_limit = atof(w);
    hipDeviceProp_t prop;
    HC(hipGetDeviceProperties(&prop, 0));
    printf("device: %s  arch: %s  CUs: %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    hipModule_t mod;
    hipFunction_t fn;
    HC(hipModuleLoad(&mod, argv[1]));
    HC(hipModuleGetFunction(&fn, mod, kn));

    // operands: finite e4m3 codes of magnitude <= 2
    std::vector<uint8_t> codes;
    for (int c = 0; c < 256; ++c) { float x = dec_e4m3((uint8_t)c); if (!std::isnan(x) && std::fabs(x) <= 2.0f) codes.push_back((uint8_t)c); }
    std::mt19937 rng(1234);
    std::vector<uint8_t> A((size_t)M * K), B((size_t)N * K);
    for (auto& x : A) x = codes[rng() % codes.size()];
    for (auto& x : B) x = codes[rng() % codes.size()];
    uint8_t *dA, *dB; float* dC;
    HC(hipMalloc(&dA, A.size())); HC(hipMalloc(&dB, B.size())); HC(hipMalloc(&dC, (size_t)M * N * 4));
    HC(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice));
    HC(hipMemset(dC, 0xff, (size_t)M * N * 4));   // NaN pattern: an unwritten element fails the check

    std::vector<uint8_t> args(40, 0);
    put64(args, 0, dA); put64(args, 8, dB); put64(args, 16, dC);
    put32(args, 24, K); put32(args, 28, N); put32(args, 32, K / 192);
    size_t sz = args.size();
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
    // the smallest launch first: ONE work-group, then the full grid
    for (int stage = 0; stage < 2; ++stage) {
        const unsigned gx = stage == 0 ? 1 : N / 32, gy = stage == 0 ? 1 : M / 32;
        if (stage == 1 && gx == 1 && gy == 1) break;
        hipEvent_t e;
        HC(hipEventCreate(&e));
        HC(hipModuleLaunchKernel(fn, gx, gy, 1, 128, 1, 1, 0, nullptr, nullptr, cfg));
        HC(hipEventRecord(e, nullptr));
        wait_polled(e, kn);
    }
    std::vector<float> C((size_t)M * N);
    HC(hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> fa(A.size()), fb(B.size());
    for (size_t i = 0; i < A.size(); ++i) fa[i] = dec_e4m3(A[i]);
    for (size_t i = 0; i < B.size(); ++i) fb[i] = dec_e4m3(B[i]);
    double maxerr = 0, maxrel = 0; long bad = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double ref = 0, scale = 0;
            for (int k = 0; k < K; ++k) { double p = (double)fa[(size_t)i * K + k] * fb[(size_t)j * K + k]; ref += p; scale += std::fabs(p); }
            double got = C[(size_t)i * N + j];
            double err = std::fabs(got - ref);
            bool ok = !std::isnan(got) && err <= 5e-4 * scale + 1e-3;
            if (!ok) { if (bad < 8) printf("  mismatch C[%d][%d]: got %g ref %g\n", i, j, got, ref); ++bad; }
            if (!std::isnan(err)) { maxerr = std::max(maxerr, err); maxrel = std::max(maxrel, err / (scale + 1e-9)); }
        }
    double us = 0, tflops = 0;
    if (iters > 0 && bad == 0) {
        std::vector<hipEvent_t> ev(2);
        HC(hipEventCreate(&ev[0])); HC(hipEventCreate(&ev[1]));
        const unsigned gx = N / 32, gy = M / 32;
        for (int w = 0; w < 3; ++w) HC(hipModuleLaunchKernel(fn, gx, gy, 1, 128, 1, 1, 0, nullptr, nullptr, cfg));
        HC(hipEventRecord(ev[0], nullptr));
        for (int i = 0; i < iters; ++i) HC(hipModuleLaunchKernel(fn, gx, gy, 1, 128, 1, 1, 0, nullptr, nullptr, cfg));
        HC(hipEventRecord(ev[1], nullptr));
        wait_polled(ev[1], "timing");
        float ms = 0;
        HC(hipEventElapsedTime(&ms, ev[0], ev[1]));
        us = 1000.0 * ms / iters;
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12;
    }
    printf("{\"host\":\"g1250_gemm\",\"kernel\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"maxerr\":%.3g,\"maxrel\":%.3g,\"bad\":%ld,\"us\":%.2f,\"tflops\":%.3f,\"ok\":%s}\n", kn, M, N, K, maxerr, maxrel, bad, us, tflops, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}
