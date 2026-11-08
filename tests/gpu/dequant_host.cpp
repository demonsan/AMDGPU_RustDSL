// fp8 -> bf16 dequantization (batch 2): static / per-token / per-group128 scale; correctness against a CPU reference and bandwidth.
// usage: dequant_host <co> <kernel> <static|token|group128> <M> <N> [iters]      (static: n = M * N elements)
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
static float fp8_to_f32(uint8_t v) { int s = v >> 7, e = (v >> 3) & 15, m = v & 7; float r = (e == 0) ? std::ldexp((float)m / 8.0f, -6) : std::ldexp(1.0f + (float)m / 8.0f, e - 7); return s ? -r : r; }
int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s <co> <kernel> <static|token|group128> <M> <N> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; std::string mode = argv[3]; long M = std::atol(argv[4]), N = std::atol(argv[5]); int iters = argc > 6 ? std::atoi(argv[6]) : 50;
    const bool st = mode == "static", grp = mode == "group128"; const size_t n = (size_t)M * N; const long ns = grp ? N / 128 : 1;
    std::mt19937 rng(31); std::vector<uint8_t> hq(n); for (auto& v : hq) { do v = rng() & 255; while ((v & 0x7f) == 0x7f); }   // no NaN codes
    std::uniform_real_distribution<float> ud(0.01f, 2.0f); std::vector<float> hs(st ? 1 : M * ns); for (auto& v : hs) v = ud(rng);
    void *dq, *dout, *ds; CK(hipMalloc(&dq, n)); CK(hipMalloc(&dout, n * 2 + 64)); CK(hipMalloc(&ds, hs.size() * 4)); CK(hipMemset(dout, 0x55, n * 2 + 64));
    CK(hipMemcpy(dq, hq.data(), n, hipMemcpyHostToDevice)); CK(hipMemcpy(ds, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[40]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    int gx, gy = 1, blockx;
    if (st) { float s = hs[0]; int32_t n32 = (int32_t)n; put(&dq, 8); put(&dout, 8); put(&s, 4); put(&n32, 4); blockx = 256; gx = (int)((n + 2047) / 2048); }
    else { int32_t m32 = (int32_t)M; put(&dq, 8); put(&dout, 8); put(&ds, 8); put(&m32, 4); long d8 = N / 8; blockx = (int)(std::min<long>(4, (d8 + 63) / 64) * 64); gx = (int)((d8 + blockx - 1) / blockx); gy = (int)M; }
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, gx, gy, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint16_t> ho(n + 32); CK(hipMemcpy(ho.data(), dout, ho.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < n; ++i) {
        float sc = st ? hs[0] : (grp ? hs[(i / N) * ns + (i % N) / 128] : hs[i / N]);
        double ref = (double)fp8_to_f32(hq[i]) * sc, got = bf16_to_f32(ho[i]);
        if (!(std::fabs(got - ref) <= std::fabs(ref) * (1.0 / 128.0) * 1.05 + 1e-30)) { if (bad < 5) std::fprintf(stderr, "bad at %zu: got %g ref %g\n", i, got, ref); ++bad; }
    }
    long guard = 0; for (size_t i = n; i < n + 32; ++i) if (ho[i] != 0x5555) ++guard;
    std::printf("check: dequant_%s M=%ld N=%ld bad=%ld past_end_bad=%ld %s\n", mode.c_str(), M, N, bad, guard, (bad == 0 && guard == 0) ? "DEQ_OK" : "DEQ_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, gx, gy, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, gx, gy, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)n * 3 + hs.size() * 4;
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}