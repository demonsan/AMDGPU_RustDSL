// Streaming elementwise kernels (batch 2, docs/op-coverage-memtemplates.md T1): correctness against a CPU reference (bf16) and bandwidth.
// usage: eltwise_host <co> <kernel> <mode: silu_mul|gelu_tanh_mul|add> <M> <D> [iters]      (add: n = M * D elements)
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
    if (argc < 6) { std::fprintf(stderr, "usage: %s <co> <kernel> <silu_mul|gelu_tanh_mul|add> <M> <D> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; std::string mode = argv[3];
    long M = std::atol(argv[4]); long D = std::atol(argv[5]); int iters = argc > 6 ? std::atoi(argv[6]) : 50;
    const bool flat = mode == "add";
    const size_t nin = flat ? (size_t)M * D : (size_t)M * 2 * D, nout = (size_t)M * D;
    std::mt19937 rng(77); std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<uint16_t> hx(nin), hy(flat ? nin : 0);
    for (auto& v : hx) v = f32_to_bf16(nd(rng));
    for (auto& v : hy) v = f32_to_bf16(nd(rng));
    for (size_t i = 0; i < hx.size(); i += 4099) hx[i] = f32_to_bf16((i / 4099) % 2 ? 30.f : -30.f);
    void *dx, *dy = nullptr, *dout;
    CK(hipMalloc(&dx, nin * 2)); CK(hipMalloc(&dout, nout * 2 + 64)); CK(hipMemset(dout, 0x55, nout * 2 + 64));
    CK(hipMemcpy(dx, hx.data(), nin * 2, hipMemcpyHostToDevice));
    if (flat) { CK(hipMalloc(&dy, nin * 2)); CK(hipMemcpy(dy, hy.data(), nin * 2, hipMemcpyHostToDevice)); }
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    int blockx, gx, gy = 1;
    void* cfg_args = nullptr; size_t sz = 0;
    struct __attribute__((packed)) A2 { void* x; void* o; int32_t m; } a2{dx, dout, (int32_t)M};
    struct __attribute__((packed)) A3 { void* a; void* b; void* o; int32_t n; } a3{dx, dy, dout, (int32_t)((size_t)M * D)};
    if (flat) { blockx = 256; gx = (int)(((size_t)M * D + 2047) / 2048); cfg_args = &a3; sz = sizeof(a3); }
    else { long d8 = D / 8; blockx = (int)(std::min<long>(4, (d8 + 63) / 64) * 64); gy = (int)M; gx = (int)((d8 + blockx - 1) / blockx); cfg_args = &a2; sz = sizeof(a2); }
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, cfg_args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, gx, gy, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint16_t> ho(nout + 32);
    CK(hipMemcpy(ho.data(), dout, (nout + 32) * 2, hipMemcpyDeviceToHost));
    long bad = 0, cnt = 0; double max_rel = 0;
    for (size_t i = 0; i < nout; ++i) {
        double ref;
        if (flat) ref = (double)bf16_to_f32(hx[i]) + bf16_to_f32(hy[i]);
        else {
            size_t m = i / D, d = i % D;
            double a = bf16_to_f32(hx[m * 2 * D + d]), g = bf16_to_f32(hx[m * 2 * D + D + d]), y;
            if (mode == "silu_mul") y = a / (1.0 + std::exp(-a));
            else y = 0.5 * a * (1.0 + std::tanh(0.7978845608028654 * (a + 0.044715 * a * a * a)));
            ref = y * g;
        }
        double got = bf16_to_f32(ho[i]);
        double err = std::fabs(got - ref), tol = std::fabs(ref) * (1.0 / 128.0) * 1.05 + 1e-6;   // <= 1 bf16 ulp (+ a tiny absolute floor for the exp2 / rcp tails)
        if (!(err <= tol)) { if (bad < 5) std::fprintf(stderr, "bad at %zu: got %g ref %g\n", i, got, ref); ++bad; }
        if (std::fabs(ref) > 1e-3) max_rel = std::fmax(max_rel, err / std::fabs(ref));
        ++cnt;
    }
    long guard_bad = 0; for (size_t i = nout; i < nout + 32; ++i) if (ho[i] != 0x5555) ++guard_bad;
    std::printf("check: %s M=%ld D=%ld bad=%ld (of %ld) max_rel=%.3e past_end_bad=%ld %s\n", mode.c_str(), M, D, bad, cnt, max_rel, guard_bad, (bad == 0 && guard_bad == 0) ? "ELT_OK" : "ELT_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, gx, gy, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, gx, gy, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * D * 2 * 3;
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}