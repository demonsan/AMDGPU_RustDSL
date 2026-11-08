// Quantization kernels (batch 2, docs/op-coverage-membound.md): per-token / per-group(128) dynamic fp8 and silu_mul + per-token fp8: correctness against a CPU reference and bandwidth.
// usage: quant_host <co> <kernel> <mode: token|group128|silu_token> <M> <N> [iters]     (silu_token: N = D, input [M][2D])
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
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float fp8_to_f32(uint8_t v) { int s = v >> 7, e = (v >> 3) & 15, m = v & 7; float r = (e == 0) ? std::ldexp((float)m / 8.0f, -6) : std::ldexp(1.0f + (float)m / 8.0f, e - 7); return s ? -r : r; }

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s <co> <kernel> <token|group128|silu_token> <M> <N> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; std::string mode = argv[3];
    long M = std::atol(argv[4]); long N = std::atol(argv[5]); int iters = argc > 6 ? std::atoi(argv[6]) : 50;
    const bool silu = mode == "silu_token", grp = mode == "group128", i8 = mode == "token_int8";
    const long in_cols = silu ? 2 * N : N, ns = grp ? N / 128 : 1;
    std::mt19937 rng(99); std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<uint16_t> hx((size_t)M * in_cols);
    for (auto& v : hx) v = f32_to_bf16(nd(rng));
    for (size_t i = 0; i < hx.size(); i += 777) hx[i] = f32_to_bf16(((i / 777) % 2 ? 1 : -1) * 90.f);   // outliers
    void *dx, *dq, *ds;
    CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dq, (size_t)M * N + 64)); CK(hipMalloc(&ds, (size_t)M * ns * 4 + 64));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemset(dq, 0x55, (size_t)M * N + 64)); CK(hipMemset(ds, 0x77, (size_t)M * ns * 4 + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void* x; void* q; void* s; int32_t m; } args{dx, dq, ds, (int32_t)M};
    size_t sz = sizeof(args);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    int grid = (int)((M + 3) / 4);
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint8_t> hq((size_t)M * N + 64); std::vector<float> hs((size_t)M * ns + 16);
    CK(hipMemcpy(hq.data(), dq, hq.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(hs.data(), ds, hs.size() * 4, hipMemcpyDeviceToHost));
    long bad = 0, cnt = 0; double max_rel_scale = 0;
    std::vector<float> y(N);
    for (long m = 0; m < M; ++m) {
        for (long k = 0; k < N; ++k) {
            double v;
            if (silu) { double a = bf16_to_f32(hx[(size_t)m * in_cols + k]), g = bf16_to_f32(hx[(size_t)m * in_cols + N + k]); v = a / (1.0 + std::exp(-a)) * g; }
            else v = bf16_to_f32(hx[(size_t)m * in_cols + k]);
            y[k] = (float)v;
        }
        long gsz = grp ? 128 : N;
        for (long g0 = 0; g0 < N; g0 += gsz) {
            float amax = 0; for (long k = g0; k < g0 + gsz; ++k) amax = std::fmax(amax, std::fabs(y[k]));
            float sref = std::fmax(amax, 1e-12f) / (i8 ? 127.0f : 448.0f), sdev = hs[(size_t)m * ns + g0 / gsz];
            max_rel_scale = std::fmax(max_rel_scale, std::fabs(sdev - sref) / sref);
            for (long k = g0; k < g0 + gsz; ++k) {
                float d = (i8 ? (float)(int8_t)hq[(size_t)m * N + k] : fp8_to_f32(hq[(size_t)m * N + k])) * sdev, e = std::fabs(d - y[k]);
                float tol = i8 ? sref * 0.5f * 1.01f + 1e-6f : std::fabs(y[k]) / 16.0f * 1.02f + sref * std::ldexp(1.0f, -9);
                if (!(e <= tol)) { if (bad < 5) std::fprintf(stderr, "bad at (%ld,%ld): got %g ref %g\n", m, k, d, y[k]); ++bad; }
                ++cnt;
            }
        }
    }
    long guard_bad = 0; for (size_t i = (size_t)M * N; i < (size_t)M * N + 64; ++i) if (hq[i] != 0x55) ++guard_bad;
    for (size_t i = (size_t)M * ns; i < (size_t)M * ns + 16; ++i) { uint32_t u; std::memcpy(&u, &hs[i], 4); if (u != 0x77777777u) ++guard_bad; }
    bool ok = bad == 0 && max_rel_scale < 1e-4 && guard_bad == 0;
    std::printf("check: %s M=%ld N=%ld bad=%ld (of %ld) max_rel_scale=%.3e past_end_bad=%ld %s\n", mode.c_str(), M, N, bad, cnt, max_rel_scale, guard_bad, ok ? "QUANT_OK" : "QUANT_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * in_cols * 2 + (double)M * N + (double)M * ns * 4;
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}