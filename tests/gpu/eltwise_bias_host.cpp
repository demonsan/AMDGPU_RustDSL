// MoE-bias activations (batch 2): {silu,gelu_tanh,gelu,swiglu_oai}_mul_bias_bf16_d<D>: out[m][d] = act(bf16(x + bias[e][d])) * bf16(y + bias[e][D + d]), e = expert_ids[m]; invalid experts give zero rows. CPU reference.
// usage: eltwise_bias_host <co> <kernel> <silu|gelu_tanh|gelu|swiglu_oai> <M> <D> <E> [iters]
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
    if (argc < 7) { std::fprintf(stderr, "usage: %s <co> <kernel> <act> <M> <D> <E> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; std::string act = argv[3]; long M = std::atol(argv[4]), D = std::atol(argv[5]), E = std::atol(argv[6]); int iters = argc > 7 ? std::atoi(argv[7]) : 30;
    std::mt19937 rng(43); std::normal_distribution<float> nd(0.f, 2.f); std::vector<uint16_t> hx((size_t)M * 2 * D), hb((size_t)E * 2 * D); for (auto& v : hx) v = f32_to_bf16(nd(rng)); for (auto& v : hb) v = f32_to_bf16(nd(rng) * 0.5f);
    std::vector<int32_t> ex(M); for (long m = 0; m < M; ++m) ex[m] = (m % 11 == 3) ? -1 : (m % 17 == 5 ? (int32_t)E + 2 : (int32_t)(rng() % E));
    void *dx, *db, *de, *dout; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&db, hb.size() * 2)); CK(hipMalloc(&de, M * 4)); CK(hipMalloc(&dout, (size_t)M * D * 2 + 64)); CK(hipMemset(dout, 0x55, (size_t)M * D * 2 + 64));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(db, hb.data(), hb.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(de, ex.data(), M * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *x, *b, *e, *o; int32_t m, ne; } args{dx, db, de, dout, (int32_t)M, (int32_t)E}; size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long d8 = D / 8; int blockx = (int)(std::min<long>(4, (d8 + 63) / 64) * 64); int gx = (int)((d8 + blockx - 1) / blockx);
    auto launch = [&]() { CK(hipModuleLaunchKernel(fn, gx, (int)M, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(); CK(hipDeviceSynchronize());
    std::vector<uint16_t> ho((size_t)M * D + 32); CK(hipMemcpy(ho.data(), dout, ho.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0;
    for (long m = 0; m < M; ++m) for (long d = 0; d < D; ++d) {
        double ref; if (ex[m] < 0 || ex[m] >= E) ref = 0; else { long e = ex[m]; double a = bf16_to_f32(f32_to_bf16(bf16_to_f32(hx[m * 2 * D + d]) + bf16_to_f32(hb[(e * 2) * D + d]))), g = bf16_to_f32(f32_to_bf16(bf16_to_f32(hx[m * 2 * D + D + d]) + bf16_to_f32(hb[(e * 2 + 1) * D + d])));
            if (act == "silu") ref = a / (1.0 + std::exp(-a)) * g; else if (act == "gelu_tanh") ref = 0.5 * a * (1.0 + std::tanh(0.7978845608028654 * (a + 0.044715 * a * a * a))) * g; else if (act == "gelu") ref = 0.5 * a * (1.0 + std::erf(a / std::sqrt(2.0))) * g;
            else { double gt = std::min(a, 7.0), lin = std::max(-7.0, std::min(g, 7.0)); ref = gt / (1.0 + std::exp(-1.702 * gt)) * (lin + 1.0); } }
        double got = bf16_to_f32(ho[m * D + d]); if (!(std::fabs(got - ref) <= std::fabs(ref) / 128.0 * 1.05 + 1e-4)) { if (bad < 5) std::fprintf(stderr, "bad m=%ld d=%ld: got %g ref %g\n", m, d, got, ref); ++bad; } }
    long guard = 0; for (size_t i = (size_t)M * D; i < (size_t)M * D + 32; ++i) if (ho[i] != 0x5555) ++guard;
    std::printf("check: %s_mul_bias M=%ld D=%ld E=%ld bad=%ld past_end_bad=%ld %s\n", act.c_str(), M, D, E, bad, guard, (bad == 0 && guard == 0) ? "BIAS_OK" : "BIAS_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); for (int i = 0; i < 5; ++i) launch();
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) launch(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * D * 2 * 3 + (double)M * D * 2 * 2; std::printf("timing: %.4f ms/iter (%.1f us), %.1f GB/s (x + out + the expert bias rows read)\n", ms, ms * 1e3, bytes / ms / 1e6);
    return 0;
}