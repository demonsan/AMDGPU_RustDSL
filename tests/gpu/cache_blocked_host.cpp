// KV-cache write, vLLM / aiter blocked layout (batch 2): k, v [T][Hk][D] bf16 -> key_cache [NB][Hk][D/8][BS][8], value_cache [NB][Hk][D][BS] at slot_mapping[t] (negative = padding): byte-exact check; untouched bytes keep the sentinel; bandwidth.
// usage: cache_blocked_host <co> <kernel> <T> <Hk> <D> <BS> <NB> [iters]
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
    if (argc < 8) { std::fprintf(stderr, "usage: %s <co> <kernel> <T> <Hk> <D> <BS> <NB> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long T = std::atol(argv[3]), Hk = std::atol(argv[4]), D = std::atol(argv[5]), BS = std::atol(argv[6]), NB = std::atol(argv[7]); int iters = argc > 8 ? std::atoi(argv[8]) : 50;
    const long S = NB * BS; const size_t row = (size_t)Hk * D * 2;
    std::mt19937 rng(77); std::vector<uint16_t> hk(T * Hk * D), hv(T * Hk * D); for (auto& v : hk) v = rng() & 0xffff; for (auto& v : hv) v = rng() & 0xffff;
    std::vector<int32_t> perm(S); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> slots(T); for (long t = 0; t < T; ++t) slots[t] = (t % 7 == 3) ? -1 : perm[t];
    void *dk, *dv, *dkc, *dvc, *dsl; CK(hipMalloc(&dk, T * row)); CK(hipMalloc(&dv, T * row)); CK(hipMalloc(&dkc, S * row + 64)); CK(hipMalloc(&dvc, S * row + 64)); CK(hipMalloc(&dsl, T * 4));
    CK(hipMemcpy(dk, hk.data(), T * row, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), T * row, hipMemcpyHostToDevice)); CK(hipMemcpy(dsl, slots.data(), T * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dkc, 0xAB, S * row + 64)); CK(hipMemset(dvc, 0xCD, S * row + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *k, *v, *kc, *vc, *sl; int32_t t, s; } args{dk, dv, dkc, dvc, dsl, (int32_t)T, (int32_t)S};
    size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long nv = (long)(row / 16); long lt = std::min<long>(4, (nv + 63) / 64) * 64, tpb = lt <= 128 ? 256 / lt : 1; int blockx = (int)(lt * tpb); int grid = (int)((T + tpb - 1) / tpb);
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint8_t> hkc(S * row + 64), hvc(S * row + 64); CK(hipMemcpy(hkc.data(), dkc, hkc.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(hvc.data(), dvc, hvc.size(), hipMemcpyDeviceToHost));
    std::vector<uint8_t> ekc(S * row + 64), evc(S * row + 64); std::memset(ekc.data(), 0xAB, ekc.size()); std::memset(evc.data(), 0xCD, evc.size());
    for (long t = 0; t < T; ++t) {
        long s = slots[t]; if (s < 0) continue; long blk = s / BS, off = s % BS;
        for (long h = 0; h < Hk; ++h) for (long d = 0; d < D; ++d) {
            uint16_t kval = hk[(t * Hk + h) * D + d], vval = hv[(t * Hk + h) * D + d];
            size_t ki = ((((size_t)blk * Hk + h) * (D / 8) + d / 8) * BS + off) * 8 + d % 8;   // key_cache [NB][Hk][D/8][BS][8]
            size_t vi = (((size_t)blk * Hk + h) * D + d) * BS + off;                            // value_cache [NB][Hk][D][BS]
            std::memcpy(&ekc[ki * 2], &kval, 2); std::memcpy(&evc[vi * 2], &vval, 2);
        }
    }
    long bad = 0; for (size_t i = 0; i < S * row; ++i) { if (hkc[i] != ekc[i]) ++bad; if (hvc[i] != evc[i]) ++bad; }
    long guard = 0; for (size_t i = S * row; i < S * row + 64; ++i) if (hkc[i] != 0xAB || hvc[i] != 0xCD) ++guard;
    std::printf("check: blocked T=%ld Hk=%ld D=%ld BS=%ld NB=%ld bad_bytes=%ld past_end_bad=%ld %s\n", T, Hk, D, BS, NB, bad, guard, (bad == 0 && guard == 0) ? "CACHEB_OK" : "CACHEB_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = 2.0 * 2.0 * (double)T * row * ((double)(T - (T + 6) / 7) / T);
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}