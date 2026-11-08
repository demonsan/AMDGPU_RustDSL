// MoE combine (batch 2): moe_reduce_bf16_k<K>_d<D> (weighted) and moe_combine_bf16_k<K>_d<D>: CPU reference and bandwidth.
// usage: moe_reduce_host <co> <kernel> <reduce|sum> <T> <K> <D> [iters] [sample_stride]   (sample_stride > 0: a periodic input pattern and only every stride-th token + the last one are checked -- for tensors > 4 GiB)
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
int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s <co> <kernel> <reduce|sum> <T> <K> <D> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; bool weighted = std::string(argv[3]) == "reduce"; long T = std::atol(argv[4]), K = std::atol(argv[5]), D = std::atol(argv[6]); int iters = argc > 7 ? std::atoi(argv[7]) : 30; long stride = argc > 8 ? std::atol(argv[8]) : 0;
    std::mt19937 rng(9); std::normal_distribution<float> nd(0.f, 1.f); std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<uint16_t> hx((size_t)T * K * D); if (stride > 0) { std::vector<uint16_t> pat(65521); for (auto& v : pat) v = f32_to_bf16(nd(rng)); for (size_t i = 0; i < hx.size(); ++i) hx[i] = pat[i % 65521]; } else for (auto& v : hx) v = f32_to_bf16(nd(rng));
    std::vector<float> hw((size_t)T * K); for (auto& v : hw) v = ud(rng);
    void *dx, *dw, *dout; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dw, hw.size() * 4)); CK(hipMalloc(&dout, (size_t)T * D * 2 + 64)); CK(hipMemset(dout, 0x55, (size_t)T * D * 2 + 64));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[40]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    int32_t t32 = (int32_t)T; if (weighted) { put(&dx, 8); put(&dw, 8); put(&dout, 8); put(&t32, 4); } else { put(&dx, 8); put(&dout, 8); put(&t32, 4); }
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long d8 = D / 8; long maxw = std::min<long>(4, (d8 + 63) / 64); long bestw = INT32_MAX; int blockx = 64; for (long w = 1; w <= maxw; ++w) { long th = 64 * w, waste = ((d8 + th - 1) / th) * th - d8; if (waste <= bestw) { bestw = waste; blockx = (int)th; } }   // pick_threads in moe/reduce.rs
    int gx = (int)((d8 + blockx - 1) / blockx);
    CK(hipModuleLaunchKernel(fn, gx, (int)T, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint16_t> ho((size_t)T * D + 32); CK(hipMemcpy(ho.data(), dout, ho.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0;
    for (long t = 0; t < T; ++t) { if (stride > 0 && t % stride != 0 && t != T - 1) continue; for (long d = 0; d < D; ++d) {
        double ref = 0, mag = 0; for (long k = 0; k < K; ++k) { double term = (double)bf16_to_f32(hx[((size_t)t * K + k) * D + d]) * (weighted ? hw[(size_t)t * K + k] : 1.0); ref += term; mag += std::fabs(term); }
        double got = bf16_to_f32(ho[(size_t)t * D + d]);
        if (!(std::fabs(got - ref) <= std::fabs(ref) / 128.0 * 1.02 + 1e-6 * mag + 1e-30)) { if (bad < 5) std::fprintf(stderr, "bad at t=%ld d=%ld: got %g ref %g\n", t, d, got, ref); ++bad; }
    } }
    long guard = 0; for (size_t i = (size_t)T * D; i < (size_t)T * D + 32; ++i) if (ho[i] != 0x5555) ++guard;
    std::printf("check: moe_%s T=%ld K=%ld D=%ld bad=%ld past_end_bad=%ld %s\n", weighted ? "reduce" : "sum", T, K, D, bad, guard, (bad == 0 && guard == 0) ? "MOER_OK" : "MOER_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, gx, (int)T, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, gx, (int)T, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)T * K * D * 2 + (double)T * D * 2 + (weighted ? (double)T * K * 4 : 0);
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}