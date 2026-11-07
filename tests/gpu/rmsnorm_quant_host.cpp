// RMSNorm + per-token fp8 (e4m3) quantization: correctness against a CPU reference and timing (rmsnorm_quant_fp8_n<N>).
// usage: rmsnorm_quant_host <co> <kernel> <M> <N> [iters]
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
// e4m3 (OCP, max 448) decode
static float fp8_to_f32(uint8_t v) {
    int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float r = (e == 0) ? std::ldexp((float)m / 8.0f, -6) : std::ldexp(1.0f + (float)m / 8.0f, e - 7);
    return s ? -r : r;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s <co> <kernel> <M> <N> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; int M = std::atoi(argv[3]); int N = std::atoi(argv[4]); int iters = argc > 5 ? std::atoi(argv[5]) : 100;
    float eps = 1e-6f;
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 1.f); std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    std::vector<uint16_t> hx((size_t)M * N), hg(N);
    for (auto& v : hx) v = f32_to_bf16(nd(rng));
    for (auto& v : hg) v = f32_to_bf16(ud(rng));
    void *dx, *dg, *dq, *ds;
    CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dg, hg.size() * 2)); CK(hipMalloc(&dq, (size_t)M * N)); CK(hipMalloc(&ds, (size_t)M * 4));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dg, hg.data(), hg.size() * 2, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void* x; void* g; void* q; void* s; int32_t m; float eps; } args{dx, dg, dq, ds, M, eps};
    size_t sz = sizeof(args);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    int grid = (M + 3) / 4;
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint8_t> hq((size_t)M * N); std::vector<float> hs(M);
    CK(hipMemcpy(hq.data(), dq, hq.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(hs.data(), ds, (size_t)M * 4, hipMemcpyDeviceToHost));
    // reference (fp32): y = x rsqrt(mean x^2 + eps) g; scale = max|y| / 448; compare dequantized values with the reference y
    long bad = 0; double max_rel_scale = 0, sum_err = 0; long cnt = 0;
    for (int m = 0; m < M; ++m) {
        double ss = 0; for (int k = 0; k < N; ++k) { double x = bf16_to_f32(hx[(size_t)m * N + k]); ss += x * x; }
        float r = 1.0f / std::sqrt((float)(ss / N) + eps);
        std::vector<float> y(N); float amax = 0;
        for (int k = 0; k < N; ++k) { y[k] = bf16_to_f32(hx[(size_t)m * N + k]) * r * bf16_to_f32(hg[k]); amax = std::fmax(amax, std::fabs(y[k])); }
        float sref = std::fmax(amax, 1e-12f) / 448.0f;
        max_rel_scale = std::fmax(max_rel_scale, std::fabs(hs[m] - sref) / sref);
        for (int k = 0; k < N; ++k) {
            float d = fp8_to_f32(hq[(size_t)m * N + k]) * hs[m];
            float e = std::fabs(d - y[k]);
            // e4m3 has 3 mantissa bits: half an ulp of |y| is |y| / 16 (+ the subnormal step near 0)
            float tol = std::fabs(y[k]) / 16.0f * 1.01f + sref * std::ldexp(1.0f, -9);
            if (e > tol) ++bad;
            sum_err += e; ++cnt;
        }
    }
    std::printf("check: M=%d N=%d bad=%ld (of %ld) mean_abs_err=%.3e max_rel_scale_err=%.3e %s\n", M, N, bad, cnt, sum_err / cnt, max_rel_scale, (bad == 0 && max_rel_scale < 1e-4) ? "RMSQ_OK" : "RMSQ_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * N * 2 + (double)N * 2 + (double)M * N + (double)M * 4;
    std::printf("timing: %.4f ms/iter, %.1f GB/s\n", ms, bytes / ms / 1e6);
    return 0;
}