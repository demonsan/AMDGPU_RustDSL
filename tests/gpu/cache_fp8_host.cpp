// fp8 KV-cache write (batch 2): reshape_cache_flash_fp8_h<Hk>_d<D>: k, v bf16 [T][Hk][D] -> key_cache, value_cache fp8 e4m3 [S][Hk][D] at slot_mapping[t] (negative / >= S: nothing), values divided by k_scale / v_scale.
// usage: cache_fp8_host <co> <kernel> <T> <Hk> <D> <S> [iters]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float fp8_to_f32(uint8_t v) { int s = v >> 7, e = (v >> 3) & 15, m = v & 7; float r = (e == 0) ? std::ldexp((float)m / 8.0f, -6) : std::ldexp(1.0f + (float)m / 8.0f, e - 7); return s ? -r : r; }
int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s <co> <kernel> <T> <Hk> <D> <S> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long T = std::atol(argv[3]), Hk = std::atol(argv[4]), D = std::atol(argv[5]), S = std::atol(argv[6]); int iters = argc > 7 ? std::atoi(argv[7]) : 30;
    const size_t erow = (size_t)Hk * D;
    std::mt19937 rng(21); std::normal_distribution<float> nd(0.f, 3.f); std::vector<uint16_t> hk(T * erow), hv(T * erow); for (auto& v : hk) v = f32_to_bf16(nd(rng)); for (auto& v : hv) v = f32_to_bf16(nd(rng));
    float ksc = 0.07f, vsc = 0.13f;
    std::vector<int32_t> perm(S); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng); std::vector<int32_t> slots(T); for (long t = 0; t < T; ++t) slots[t] = (t % 9 == 4) ? -1 : (t % 17 == 6 ? (int32_t)(S + t) : perm[t]);
    void *dk, *dv, *dkc, *dvc, *dsl; CK(hipMalloc(&dk, T * erow * 2)); CK(hipMalloc(&dv, T * erow * 2)); CK(hipMalloc(&dkc, S * erow + 64)); CK(hipMalloc(&dvc, S * erow + 64)); CK(hipMalloc(&dsl, T * 4));
    CK(hipMemcpy(dk, hk.data(), T * erow * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), T * erow * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dsl, slots.data(), T * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dkc, 0xAB, S * erow + 64)); CK(hipMemset(dvc, 0xCD, S * erow + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *k, *v, *kc, *vc, *sl; int32_t t, s; float ks, vs; } args{dk, dv, dkc, dvc, dsl, (int32_t)T, (int32_t)S, ksc, vsc};
    size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    size_t row = erow * 2; long nv = (long)(row / 16); long lt = std::min<long>(4, (nv + 63) / 64) * 64, tpb = lt <= 128 ? 256 / lt : 1; int blockx = (int)(lt * tpb); int grid = (int)((T + tpb - 1) / tpb);
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint8_t> hkc(S * erow + 64), hvc(S * erow + 64); CK(hipMemcpy(hkc.data(), dkc, hkc.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(hvc.data(), dvc, hvc.size(), hipMemcpyDeviceToHost));
    std::vector<char> written(S, 0); long bad = 0;
    for (long t = 0; t < T; ++t) { long s = slots[t]; if (s < 0 || s >= S) continue; written[s] = 1;
        for (size_t i = 0; i < erow; ++i) {
            double yk = bf16_to_f32(hk[t * erow + i]) / ksc, yv = bf16_to_f32(hv[t * erow + i]) / vsc;
            double dk_ = fp8_to_f32(hkc[s * erow + i]), dv_ = fp8_to_f32(hvc[s * erow + i]);
            auto okf = [&](double y, double d) { double sat = std::max(-448.0, std::min(448.0, y)); return std::fabs(d - sat) <= std::fabs(sat) / 16.0 * 1.05 + std::ldexp(1.0, -9) * 1.1; };
            if (!okf(yk, dk_)) { if (bad < 5) std::fprintf(stderr, "key bad t=%ld i=%zu: got %g ref %g\n", t, i, dk_, yk); ++bad; }
            if (!okf(yv, dv_)) { if (bad < 5) std::fprintf(stderr, "value bad t=%ld i=%zu: got %g ref %g\n", t, i, dv_, yv); ++bad; }
        } }
    long ubad = 0; for (long s = 0; s < S; ++s) if (!written[s]) for (size_t i = 0; i < erow; ++i) { if (hkc[s * erow + i] != 0xAB || hvc[s * erow + i] != 0xCD) { ++ubad; break; } }
    long guard = 0; for (size_t i = S * erow; i < S * erow + 64; ++i) if (hkc[i] != 0xAB || hvc[i] != 0xCD) ++guard;
    std::printf("check: cache_fp8 T=%ld Hk=%ld D=%ld S=%ld bad=%ld untouched_bad=%ld past_end_bad=%ld %s\n", T, Hk, D, S, bad, ubad, guard, (bad == 0 && ubad == 0 && guard == 0) ? "CFP8_OK" : "CFP8_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)T * erow * (2 + 2 + 1 + 1);
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}