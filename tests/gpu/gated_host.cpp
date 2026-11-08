// Gated RMSNorm (batch 2): gated_rmsnorm_bf16_n<N> (out = rmsnorm(x) * gamma * silu(z), bf16) and gated_rmsnorm_fp8_n<N> (the same, then per-row dynamic fp8 + f32 scale): CPU reference and bandwidth.
// usage: gated_host <co> <kernel> <bf16|fp8> <M> <N> [iters]
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
    if (argc < 6) { std::fprintf(stderr, "usage: %s <co> <kernel> <bf16|fp8> <M> <N> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; bool q = std::string(argv[3]) == "fp8"; long M = std::atol(argv[4]), N = std::atol(argv[5]); int iters = argc > 6 ? std::atoi(argv[6]) : 30;
    long g = 64; if (N < 512) { g = 4; while (g < N / 8) g *= 2; } long rpw = N < 512 ? std::max(8L, 64 / g) : (N < 1024 ? 2 : 1); int grid = (int)((M + 4 * rpw - 1) / (4 * rpw));
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 2.f); std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    std::vector<uint16_t> hx((size_t)M * N), hz((size_t)M * N), hg(N); for (auto& v : hx) v = f32_to_bf16(nd(rng)); for (auto& v : hz) v = f32_to_bf16(nd(rng)); for (auto& v : hg) v = f32_to_bf16(ud(rng));
    float eps = 1e-6f; void *dx, *dz, *dg, *dout, *dsc = nullptr; size_t obytes = q ? (size_t)M * N : (size_t)M * N * 2;
    CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dz, hz.size() * 2)); CK(hipMalloc(&dg, N * 2)); CK(hipMalloc(&dout, obytes + 64)); CK(hipMemset(dout, 0x55, obytes + 64)); if (q) { CK(hipMalloc(&dsc, M * 4 + 64)); CK(hipMemset(dsc, 0x55, M * 4 + 64)); }
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dz, hz.data(), hz.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dg, hg.data(), N * 2, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[64]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    put(&dx, 8); put(&dz, 8); put(&dg, 8); put(&dout, 8); if (q) put(&dsc, 8); int32_t m32 = (int32_t)M; put(&m32, 4); put(&eps, 4);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint8_t> ho(obytes + 32); CK(hipMemcpy(ho.data(), dout, obytes + 32, hipMemcpyDeviceToHost)); std::vector<float> hs(M + 16); if (q) CK(hipMemcpy(hs.data(), dsc, (M + 16) * 4, hipMemcpyDeviceToHost));
    long bad = 0, sbad = 0; std::vector<double> y(N);
    for (long m = 0; m < M; ++m) {
        double ss = 0; for (long k = 0; k < N; ++k) { double v = bf16_to_f32(hx[m * N + k]); ss += v * v; } double inv = 1.0 / std::sqrt(ss / N + eps), amax = 0;
        for (long k = 0; k < N; ++k) { double zf = bf16_to_f32(hz[m * N + k]); y[k] = bf16_to_f32(hx[m * N + k]) * inv * bf16_to_f32(hg[k]) * (zf / (1.0 + std::exp(-zf))); amax = std::max(amax, std::fabs(y[k])); }
        if (!q) { for (long k = 0; k < N; ++k) { uint16_t hb; std::memcpy(&hb, &ho[(m * N + k) * 2], 2); double got = bf16_to_f32(hb); if (!(std::fabs(got - y[k]) <= std::fabs(y[k]) / 128.0 * 1.1 + 1e-6)) { if (bad < 5) std::fprintf(stderr, "bad at m=%ld k=%ld: got %g ref %g\n", m, k, got, y[k]); ++bad; } } }
        else { double sref = std::max(amax, 1e-12) / 448.0; if (std::fabs(hs[m] - sref) > sref * 1e-3 + 1e-12) { if (sbad < 5) std::fprintf(stderr, "scale m=%ld: got %g ref %g\n", m, hs[m], sref); ++sbad; }
            for (long k = 0; k < N; ++k) { double d = fp8_to_f32(ho[m * N + k]) * (double)hs[m]; if (!(std::fabs(d - y[k]) <= std::fabs(y[k]) / 16.0 * 1.05 + sref * std::ldexp(1.0, -9) * 1.1)) { if (bad < 5) std::fprintf(stderr, "bad at m=%ld k=%ld: got %g ref %g\n", m, k, d, y[k]); ++bad; } } }
    }
    long guard = 0; for (size_t i = obytes; i < obytes + 32; ++i) if (ho[i] != 0x55) ++guard; if (q) for (long i = M; i < M + 16; ++i) { uint32_t u; std::memcpy(&u, &hs[i], 4); if (u != 0x55555555u) ++guard; }
    std::printf("check: gated_%s M=%ld N=%ld bad=%ld scale_bad=%ld past_end_bad=%ld %s\n", q ? "fp8" : "bf16", M, N, bad, sbad, guard, (bad == 0 && sbad == 0 && guard == 0) ? "GATED_OK" : "GATED_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * N * 2 * 2 + (double)M * N * (q ? 1 : 2) + (q ? M * 4.0 : 0);
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}