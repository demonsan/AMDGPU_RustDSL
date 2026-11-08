// KV-cache write, flash layout (batch 2): k, v [T][Hk][D] bf16 scattered to caches [S][Hk][D] at slot_mapping[t] (negative = padding): byte-exact check, untouched slots keep the sentinel; bandwidth.
// usage: cache_host <co> <kernel> <T> <Hk> <D> <S> [iters]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s <co> <kernel> <T> <Hk> <D> <S> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long T = std::atol(argv[3]), Hk = std::atol(argv[4]), D = std::atol(argv[5]), S = std::atol(argv[6]); int iters = argc > 7 ? std::atoi(argv[7]) : 50;
    const size_t row = (size_t)Hk * D * 2;
    std::mt19937 rng(21); std::vector<uint8_t> hk(T * row), hv(T * row); for (auto& v : hk) v = rng() & 255; for (auto& v : hv) v = rng() & 255;
    std::vector<int32_t> perm(S); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> slots(T); for (long t = 0; t < T; ++t) slots[t] = (t % 7 == 3) ? -1 : perm[t];   // distinct slots, some padding tokens
    void *dk, *dv, *dkc, *dvc, *dsl; CK(hipMalloc(&dk, T * row)); CK(hipMalloc(&dv, T * row)); CK(hipMalloc(&dkc, S * row + 64)); CK(hipMalloc(&dvc, S * row + 64)); CK(hipMalloc(&dsl, T * 4));
    CK(hipMemcpy(dk, hk.data(), T * row, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), T * row, hipMemcpyHostToDevice)); CK(hipMemcpy(dsl, slots.data(), T * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dkc, 0xAB, S * row + 64)); CK(hipMemset(dvc, 0xCD, S * row + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *k, *v, *kc, *vc, *sl; int32_t t, s; } args{dk, dv, dkc, dvc, dsl, (int32_t)T, (int32_t)S};
    size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long nv = (long)(row / 16); int blockx = (int)(std::min<long>(4, (nv + 63) / 64) * 64);
    CK(hipModuleLaunchKernel(fn, (int)T, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint8_t> hkc(S * row + 64), hvc(S * row + 64); CK(hipMemcpy(hkc.data(), dkc, hkc.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(hvc.data(), dvc, hvc.size(), hipMemcpyDeviceToHost));
    std::vector<int32_t> owner(S, -1); for (long t = 0; t < T; ++t) if (slots[t] >= 0) owner[slots[t]] = (int32_t)t;
    long bad = 0;
    for (long s = 0; s < S; ++s) for (size_t i = 0; i < row; ++i) {
        uint8_t ek = owner[s] >= 0 ? hk[(size_t)owner[s] * row + i] : 0xAB, ev = owner[s] >= 0 ? hv[(size_t)owner[s] * row + i] : 0xCD;
        if (hkc[s * row + i] != ek || hvc[s * row + i] != ev) ++bad;
    }
    long guard = 0; for (size_t i = S * row; i < S * row + 64; ++i) if (hkc[i] != 0xAB || hvc[i] != 0xCD) ++guard;
    std::printf("check: T=%ld Hk=%ld D=%ld S=%ld bad_bytes=%ld past_end_bad=%ld %s\n", T, Hk, D, S, bad, guard, (bad == 0 && guard == 0) ? "CACHE_OK" : "CACHE_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, (int)T, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, (int)T, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = 2.0 * 2.0 * (double)T * row * ((double)(T - (T + 6) / 7) / T);   // k and v read + written for the non-padding tokens
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}