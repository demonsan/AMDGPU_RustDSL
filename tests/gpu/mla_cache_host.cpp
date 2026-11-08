// MLA KV-cache write (batch 2): concat_cache_mla_{bf16,fp8}_r<R>_p<P>[_i64] (flat [S][R+P]) and concat_cache_mla_seg_{bf16,fp8}_r<R>_p<P>_b<BS>[_i64] (blocks: BS kv_c rows then BS k_pe rows).
// usage: mla_cache_host <co> <kernel> <flat|seg> <bf16|fp8> <T> <R> <P> <BS> <NB> <i64 0|1> [iters]      (S = NB * BS slots)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float fp8_to_f32(uint8_t v) { int s = v >> 7, e = (v >> 3) & 15, m = v & 7; float r = (e == 0) ? std::ldexp((float)m / 8.0f, -6) : std::ldexp(1.0f + (float)m / 8.0f, e - 7); return s ? -r : r; }
int main(int argc, char** argv) {
    if (argc < 11) { std::fprintf(stderr, "usage: %s <co> <kernel> <flat|seg> <bf16|fp8> <T> <R> <P> <BS> <NB> <i64> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; const bool seg = std::string(argv[3]) == "seg", fp8 = std::string(argv[4]) == "fp8"; long T = std::atol(argv[5]), R = std::atol(argv[6]), P = std::atol(argv[7]), BS = std::atol(argv[8]), NB = std::atol(argv[9]); const bool s64 = std::atoi(argv[10]); int iters = argc > 11 ? std::atoi(argv[11]) : 20;
    const long S = NB * BS, ENT = R + P; const size_t esz = fp8 ? 1 : 2; const size_t cbytes = (size_t)S * ENT * esz; const float scale = 0.11f;
    std::mt19937 rng(37); std::normal_distribution<float> nd(0.f, 2.f); std::vector<uint16_t> hc((size_t)T * R), hp((size_t)T * P); for (auto& v : hc) v = f32_to_bf16(nd(rng)); for (auto& v : hp) v = f32_to_bf16(nd(rng));
    std::vector<int32_t> perm(S); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng); std::vector<int64_t> slots(T); for (long t = 0; t < T; ++t) slots[t] = (t % 9 == 4) ? -1 : (t % 13 == 6 ? S + t : perm[t % S]);
    std::vector<int32_t> slots32(slots.begin(), slots.end());
    void *dc, *dp, *dcache, *dsl; CK(hipMalloc(&dc, hc.size() * 2)); CK(hipMalloc(&dp, hp.size() * 2)); CK(hipMalloc(&dcache, cbytes + 64)); CK(hipMalloc(&dsl, T * 8));
    CK(hipMemcpy(dc, hc.data(), hc.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dp, hp.data(), hp.size() * 2, hipMemcpyHostToDevice)); if (s64) CK(hipMemcpy(dsl, slots.data(), T * 8, hipMemcpyHostToDevice)); else CK(hipMemcpy(dsl, slots32.data(), T * 4, hipMemcpyHostToDevice)); CK(hipMemset(dcache, 0xAB, cbytes + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[64]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; }; int32_t t32 = (int32_t)T, s32 = (int32_t)S; put(&dc, 8); put(&dp, 8); put(&dcache, 8); put(&dsl, 8); put(&t32, 4); put(&s32, 4); if (fp8) put(&scale, 4);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long nv = (R + P) / 8; long lt = std::min<long>(4, (nv + 63) / 64) * 64, tpb = lt <= 128 ? 256 / lt : 1; int blockx = (int)(lt * tpb); int grid = (int)((T + tpb - 1) / tpb);
    auto launch = [&]() { CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(); CK(hipDeviceSynchronize());
    std::vector<uint8_t> hc2(cbytes + 64); CK(hipMemcpy(hc2.data(), dcache, cbytes + 64, hipMemcpyDeviceToHost));
    long bad = 0; std::vector<char> written(S, 0);
    auto check = [&](size_t byte_off, double ref) { if (!fp8) { uint16_t v; std::memcpy(&v, &hc2[byte_off], 2); double got = bf16_to_f32(v); return got == ref; } double got = fp8_to_f32(hc2[byte_off]) * scale; double sat = std::max(-448.0 * scale, std::min(448.0 * scale, ref)); return std::fabs(got - sat) <= std::fabs(sat) / 16.0 * 1.05 + std::ldexp(1.0, -9) * scale * 1.1; };
    for (long t = 0; t < T; ++t) { long sl = slots[t]; if (sl < 0 || sl >= S) continue; written[sl] = 1; long blk = sl / BS, off = sl % BS;
        for (long i = 0; i < ENT; ++i) { double ref = i < R ? bf16_to_f32(hc[t * R + i]) : bf16_to_f32(hp[t * P + i - R]); size_t el = seg ? (size_t)blk * BS * ENT + (i < R ? (size_t)off * R + i : (size_t)BS * R + (size_t)off * P + (i - R)) : (size_t)sl * ENT + i; if (!check(el * esz, ref)) { if (bad < 5) std::fprintf(stderr, "bad t=%ld i=%ld slot=%ld\n", t, i, sl); ++bad; } } }
    long ubad = 0; for (long sl = 0; sl < S; ++sl) if (!written[sl]) { long blk = sl / BS, off = sl % BS; for (long i = 0; i < ENT; i += 7) { size_t el = seg ? (size_t)blk * BS * ENT + (i < R ? (size_t)off * R + i : (size_t)BS * R + (size_t)off * P + (i - R)) : (size_t)sl * ENT + i; if (hc2[el * esz] != 0xAB) { ++ubad; break; } } }
    long guard = 0; for (size_t i = cbytes; i < cbytes + 64; ++i) if (hc2[i] != 0xAB) ++guard;
    std::printf("check: mla_%s %s %s T=%ld R=%ld P=%ld BS=%ld S=%ld%s bad=%ld untouched_bad=%ld past_end_bad=%ld %s\n", seg ? "seg" : "flat", argv[4], "", T, R, P, BS, S, s64 ? " i64" : "", bad, ubad, guard, (bad == 0 && ubad == 0 && guard == 0) ? "MLA_OK" : "MLA_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); for (int i = 0; i < 5; ++i) launch();
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) launch(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)T * ENT * (2 + esz); std::printf("timing: %.4f ms/iter (%.1f us), %.1f GB/s (traffic %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}