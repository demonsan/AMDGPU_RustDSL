// per-token (per-head) dynamic fp8 blocked KV-cache write (batch 2): reshape_cache_blocked_fp8_h<Hk>_d<D>_b<BS>[_i64]: k, v bf16 [T][Hk][D] -> key_cache fp8 [NB][Hk][D/16][BS][16], value_cache fp8
// [NB][Hk][D][BS], k_scales / v_scales f32 [Hk][S] (S = NB * BS); every written (token, head) is checked against x / scale (e4m3 half-step tolerance) and the scale against amax / 448; untouched slots keep their sentinel.
// usage: cache_blocked_fp8_host <co> <kernel> <T> <Hk> <D> <BS> <NB> <i64 0|1> [iters]
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
    if (argc < 9) { std::fprintf(stderr, "usage: %s <co> <kernel> <T> <Hk> <D> <BS> <NB> <i64> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long T = std::atol(argv[3]), H = std::atol(argv[4]), D = std::atol(argv[5]), BS = std::atol(argv[6]), NB = std::atol(argv[7]); bool s64 = std::atoi(argv[8]); int iters = argc > 9 ? std::atoi(argv[9]) : 20;
    const long S = NB * BS; const size_t erow = (size_t)H * D;
    std::mt19937 rng(41); std::normal_distribution<float> nd(0.f, 3.f); std::vector<uint16_t> hk(T * erow), hv(T * erow);
    for (long t = 0; t < T; ++t) { float sc = 0.2f + (t % 9) * 0.7f; for (size_t i = 0; i < erow; ++i) { hk[t * erow + i] = f32_to_bf16(nd(rng) * sc); hv[t * erow + i] = f32_to_bf16(nd(rng) * (1.f + (t % 5))); } }
    std::vector<int32_t> perm(S); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng); std::vector<int64_t> slots(T);
    for (long t = 0; t < T; ++t) slots[t] = (t % 9 == 4) ? -1 : perm[t % S];     // T <= S required for distinct slots
    std::vector<int32_t> slots32(T); for (long t = 0; t < T; ++t) slots32[t] = (int32_t)slots[t];
    void *dk, *dv, *dkc, *dvc, *dks, *dvs, *dsl; size_t cb = (size_t)S * erow;
    CK(hipMalloc(&dk, T * erow * 2)); CK(hipMalloc(&dv, T * erow * 2)); CK(hipMalloc(&dkc, cb + 64)); CK(hipMalloc(&dvc, cb + 64)); CK(hipMalloc(&dks, (size_t)S * H * 4 + 64)); CK(hipMalloc(&dvs, (size_t)S * H * 4 + 64)); CK(hipMalloc(&dsl, T * 8));
    CK(hipMemcpy(dk, hk.data(), T * erow * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), T * erow * 2, hipMemcpyHostToDevice)); if (s64) CK(hipMemcpy(dsl, slots.data(), T * 8, hipMemcpyHostToDevice)); else CK(hipMemcpy(dsl, slots32.data(), T * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dkc, 0xAB, cb + 64)); CK(hipMemset(dvc, 0xCD, cb + 64)); CK(hipMemset(dks, 0x55, (size_t)S * H * 4 + 64)); CK(hipMemset(dvs, 0x55, (size_t)S * H * 4 + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *k, *v, *kc, *vc, *ks, *vs, *sl; int32_t t, s; } args{dk, dv, dkc, dvc, dks, dvs, dsl, (int32_t)T, (int32_t)S};
    size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long nv = (long)(erow * 2 / 16); long lt = std::min<long>(4, (nv + 63) / 64) * 64, tpb = lt <= 128 ? 256 / lt : 1; int blockx = (int)(lt * tpb); int grid = (int)((T + tpb - 1) / tpb);
    auto launch = [&]() { CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(); CK(hipDeviceSynchronize());
    std::vector<uint8_t> kc(cb + 64), vc(cb + 64); std::vector<float> ksc((size_t)S * H + 16), vsc((size_t)S * H + 16); CK(hipMemcpy(kc.data(), dkc, cb + 64, hipMemcpyDeviceToHost)); CK(hipMemcpy(vc.data(), dvc, cb + 64, hipMemcpyDeviceToHost)); CK(hipMemcpy(ksc.data(), dks, ksc.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(vsc.data(), dvs, vsc.size() * 4, hipMemcpyDeviceToHost));
    long bad = 0, sbad = 0; std::vector<char> written(S, 0);
    for (long t = 0; t < T; ++t) { long sl = slots[t]; if (sl < 0 || sl >= S) continue; written[sl] = 1; long blk = sl / BS, off = sl % BS;
        for (long h = 0; h < H; ++h) {
            double ak = 0, av = 0; for (long d = 0; d < D; ++d) { ak = std::max(ak, (double)std::fabs(bf16_to_f32(hk[t * erow + h * D + d]))); av = std::max(av, (double)std::fabs(bf16_to_f32(hv[t * erow + h * D + d]))); }
            double sk = std::max(ak, 1e-12) / 448.0, sv = std::max(av, 1e-12) / 448.0; float gk = ksc[(size_t)h * S + sl], gv = vsc[(size_t)h * S + sl];
            if (!(std::fabs(gk - sk) <= sk * 1e-3 + 1e-12) || !(std::fabs(gv - sv) <= sv * 1e-3 + 1e-12)) { if (sbad < 5) std::fprintf(stderr, "scale t=%ld h=%ld: k %g (ref %g) v %g (ref %g)\n", t, h, gk, sk, gv, sv); ++sbad; }
            for (long d = 0; d < D; ++d) {
                size_t kidx = ((((size_t)blk * H + h) * (D / 16) + d / 16) * BS + off) * 16 + d % 16, vidx = (((size_t)blk * H + h) * D + d) * BS + off;
                double yk = bf16_to_f32(hk[t * erow + h * D + d]) / sk, yv = bf16_to_f32(hv[t * erow + h * D + d]) / sv; double dk_ = fp8_to_f32(kc[kidx]), dv_ = fp8_to_f32(vc[vidx]);
                auto okf = [&](double y, double q) { return std::fabs(q - y) <= std::fabs(y) / 16.0 * 1.05 + std::ldexp(1.0, -9) * 1.1; };
                if (!okf(yk, dk_)) { if (bad < 5) std::fprintf(stderr, "key bad t=%ld h=%ld d=%ld: got %g ref %g\n", t, h, d, dk_, yk); ++bad; }
                if (!okf(yv, dv_)) { if (bad < 5) std::fprintf(stderr, "value bad t=%ld h=%ld d=%ld: got %g ref %g\n", t, h, d, dv_, yv); ++bad; }
            } } }
    long ubad = 0; for (long sl = 0; sl < S; ++sl) if (!written[sl]) { long blk = sl / BS, off = sl % BS; for (long h = 0; h < H && !ubad; ++h) { uint32_t u; std::memcpy(&u, &ksc[(size_t)h * S + sl], 4); if (u != 0x55555555u) ++ubad; } (void)blk; (void)off; }
    // the cache bytes of unwritten slots: sample
    long cbad = 0; for (long sl = 0; sl < S; ++sl) if (!written[sl]) { long blk = sl / BS, off = sl % BS; for (long h = 0; h < H; ++h) for (long d = 0; d < D; d += 7) { size_t kidx = ((((size_t)blk * H + h) * (D / 16) + d / 16) * BS + off) * 16 + d % 16, vidx = (((size_t)blk * H + h) * D + d) * BS + off; if (kc[kidx] != 0xAB || vc[vidx] != 0xCD) ++cbad; } }
    long guard = 0; for (size_t i = cb; i < cb + 64; ++i) if (kc[i] != 0xAB || vc[i] != 0xCD) ++guard;
    std::printf("check: cache_blocked_fp8 T=%ld Hk=%ld D=%ld BS=%ld S=%ld%s bad=%ld scale_bad=%ld untouched_scale_bad=%ld untouched_cache_bad=%ld past_end_bad=%ld %s\n", T, H, D, BS, S, s64 ? " i64" : "", bad, sbad, ubad, cbad, guard, (bad == 0 && sbad == 0 && ubad == 0 && cbad == 0 && guard == 0) ? "CBF8_OK" : "CBF8_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) launch();
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) launch(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)T * erow * (2 + 2 + 1 + 1) + (double)T * H * 8; std::printf("timing: %.4f ms/iter (%.1f us), %.1f GB/s (traffic %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}