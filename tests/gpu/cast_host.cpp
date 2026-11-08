// Casts and static fp8 quantization (batch 2): cvt_f32_bf16 / cvt_bf16_f32 (bit exact) and quant_bf16_fp8_static (half-ulp fp8): correctness and bandwidth.
// usage: cast_host <co> <kernel> <f32_bf16|bf16_f32|quant_static> <n> [iters]
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
    if (argc < 5) { std::fprintf(stderr, "usage: %s <co> <kernel> <f32_bf16|bf16_f32|quant_static> <n> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; std::string mode = argv[3]; long n = std::atol(argv[4]); int iters = argc > 5 ? std::atoi(argv[5]) : 50;
    const float scale = 0.5f;   // static quant scale: x / 0.5, outliers clamp at 448
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 30.f);
    size_t in_b = mode == "f32_bf16" ? 4 : 2, out_b = mode == "bf16_f32" ? 4 : (mode == "quant_static" ? 1 : 2);
    std::vector<uint8_t> hin(n * in_b); std::vector<float> xs(n);
    for (long i = 0; i < n; ++i) {
        float v = nd(rng); if (i % 1001 == 0) v = (i % 2002 == 0 ? 1 : -1) * 5000.f;
        if (mode == "f32_bf16") { std::memcpy(&hin[i * 4], &v, 4); xs[i] = v; }
        else { uint16_t h = f32_to_bf16(v); std::memcpy(&hin[i * 2], &h, 2); xs[i] = bf16_to_f32(h); }
    }
    void *din, *dout; CK(hipMalloc(&din, hin.size() + 64)); CK(hipMalloc(&dout, n * out_b + 64));
    CK(hipMemcpy(din, hin.data(), hin.size(), hipMemcpyHostToDevice)); CK(hipMemset(dout, 0x55, n * out_b + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[64]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    int32_t n32 = (int32_t)n; put(&din, 8); put(&dout, 8); if (mode == "quant_static") put(&scale, 4); put(&n32, 4);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    int per = (std::string(kn).find("_x16") != std::string::npos) ? 4096 : 2048; int grid = (int)((n + per - 1) / per);
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint8_t> ho(n * out_b + 64); CK(hipMemcpy(ho.data(), dout, ho.size(), hipMemcpyDeviceToHost));
    long bad = 0;
    for (long i = 0; i < n; ++i) {
        if (mode == "f32_bf16") { uint16_t g; std::memcpy(&g, &ho[i * 2], 2); if (g != f32_to_bf16(xs[i])) ++bad; }
        else if (mode == "bf16_f32") { float g; std::memcpy(&g, &ho[i * 4], 4); if (g != xs[i]) ++bad; }
        else { float y = std::fmin(std::fmax(xs[i] / scale, -448.f), 448.f); float d = fp8_to_f32(ho[i]); if (!(std::fabs(d - y) <= std::fabs(y) / 16.f * 1.02f + std::ldexp(1.0f, -9))) { if (bad < 5) std::fprintf(stderr, "bad at %ld: got %g ref %g\n", i, d, y); ++bad; } }
    }
    long guard = 0; for (size_t i = n * out_b; i < n * out_b + 64; ++i) if (ho[i] != 0x55) ++guard;
    std::printf("check: %s n=%ld bad=%ld past_end_bad=%ld %s\n", mode.c_str(), n, bad, guard, (bad == 0 && guard == 0) ? "CAST_OK" : "CAST_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)n * (in_b + out_b);
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}