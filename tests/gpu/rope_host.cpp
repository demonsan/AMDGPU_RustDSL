// RoPE (batch 2): neox / gptj, bf16 [T][H][D], positions i32, cos_sin bf16 [P][D] (cos | sin halves): correctness against a CPU double reference (the table values are the bf16 ones) and bandwidth.
// usage: rope_host <co> <kernel> <neox|gptj> <T> <H> <D> [iters]
#include <hip/hip_runtime.h>
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
    if (argc < 7) { std::fprintf(stderr, "usage: %s <co> <kernel> <neox|gptj> <T> <H> <D> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; std::string mode = argv[3]; long T = std::atol(argv[4]), H = std::atol(argv[5]), D = std::atol(argv[6]); int iters = argc > 7 ? std::atoi(argv[7]) : 50;
    const bool neox = mode == "neox"; const long P = 4096;
    std::mt19937 rng(13); std::normal_distribution<float> nd(0.f, 2.f); std::uniform_int_distribution<int> pd(0, P - 1);
    std::vector<uint16_t> hx((size_t)T * H * D), hcs((size_t)P * D); std::vector<int32_t> hpos(T);
    for (auto& v : hx) v = f32_to_bf16(nd(rng));
    for (long p = 0; p < P; ++p) for (long i = 0; i < D / 2; ++i) { double ang = p * std::pow(10000.0, -2.0 * i / D); hcs[(size_t)p * D + i] = f32_to_bf16((float)std::cos(ang)); hcs[(size_t)p * D + D / 2 + i] = f32_to_bf16((float)std::sin(ang)); }
    for (auto& p : hpos) p = pd(rng);
    if (T > 2) hpos[T - 1] = (int32_t)P + 5;   // out of the table: cos = sin = 0, the output row is 0
    void *dx, *dout, *dpos, *dcs;
    CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dout, hx.size() * 2 + 64)); CK(hipMalloc(&dpos, T * 4)); CK(hipMalloc(&dcs, hcs.size() * 2));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dpos, hpos.data(), T * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dcs, hcs.data(), hcs.size() * 2, hipMemcpyHostToDevice)); CK(hipMemset(dout, 0x55, hx.size() * 2 + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void* x; void* o; void* pos; void* cs; int32_t t; int32_t p; } args{dx, dout, dpos, dcs, (int32_t)T, (int32_t)P};
    size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, (int)T, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint16_t> ho(hx.size() + 32); CK(hipMemcpy(ho.data(), dout, ho.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0, cnt = 0;
    for (long t = 0; t < T; ++t) for (long h = 0; h < H; ++h) for (long i = 0; i < D; ++i) {
        size_t base = ((size_t)t * H + h) * D; int p = hpos[t]; bool oob = p >= P;
        auto C = [&](long j) { return oob ? 0.0 : (double)bf16_to_f32(hcs[(size_t)p * D + j]); };
        auto S = [&](long j) { return oob ? 0.0 : (double)bf16_to_f32(hcs[(size_t)p * D + D / 2 + j]); };
        double ref;
        if (neox) { long j = i % (D / 2); double a = bf16_to_f32(hx[base + j]), c = bf16_to_f32(hx[base + D / 2 + j]); ref = i < D / 2 ? a * C(j) - c * S(j) : c * C(j) + a * S(j); }
        else { long j = i / 2; double a = bf16_to_f32(hx[base + 2 * j]), c = bf16_to_f32(hx[base + 2 * j + 1]); ref = (i % 2 == 0) ? a * C(j) - c * S(j) : c * C(j) + a * S(j); }
        double got = bf16_to_f32(ho[base + i]), err = std::fabs(got - ref), tol = std::fabs(ref) * (1.0 / 128.0) * 1.05 + 2e-3;   // the products and the bf16 output: <= 1 ulp of the larger term
        if (!(err <= tol)) { if (bad < 5) std::fprintf(stderr, "bad (t%ld,h%ld,i%ld): got %g ref %g\n", t, h, i, got, ref); ++bad; }
        ++cnt;
    }
    long guard = 0; for (size_t i = hx.size(); i < hx.size() + 32; ++i) if (ho[i] != 0x5555) ++guard;
    std::printf("check: rope_%s T=%ld H=%ld D=%ld bad=%ld (of %ld) past_end_bad=%ld %s\n", mode.c_str(), T, H, D, bad, cnt, guard, (bad == 0 && guard == 0) ? "ROPE_OK" : "ROPE_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, (int)T, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, (int)T, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)T * H * D * 2 * 2 + (double)T * (4 + D * 2);
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}