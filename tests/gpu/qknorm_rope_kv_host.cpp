// fused per-head RMSNorm + RoPE + KV-cache write (batch 2): qknorm_rope_kvcache_{neox,gptj}_{bf16,fp8}_h<H>_d<D>[_i64]: k [T][H][D] is normalised (gamma, eps) and rotated, v [T][H][D] copied (bf16 cache) or
// quantised (fp8 cache, static k_scale / v_scale); both land at slot_mapping[t] of the flash-layout caches [S][H][D]. CPU reference; unwritten rows keep their sentinel.
// usage: qknorm_rope_kv_host <co> <kernel> <neox|gptj> <T> <H> <D> <S> <fp8 0|1> <i64 0|1> [iters]
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
    if (argc < 10) { std::fprintf(stderr, "usage: %s <co> <kernel> <neox|gptj> <T> <H> <D> <S> <fp8> <i64> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; const bool neox = std::string(argv[3]) == "neox"; long T = std::atol(argv[4]), H = std::atol(argv[5]), D = std::atol(argv[6]), S = std::atol(argv[7]); const bool fp8 = std::atoi(argv[8]), s64 = std::atoi(argv[9]); int iters = argc > 10 ? std::atoi(argv[10]) : 20;
    const long P = 4096; const size_t erow = (size_t)H * D; const size_t esz = fp8 ? 1 : 2;
    std::mt19937 rng(29); std::normal_distribution<float> nd(0.f, 2.f); std::uniform_int_distribution<int> pd(0, P - 1); std::uniform_real_distribution<float> gd(0.5f, 1.5f);
    std::vector<uint16_t> hk(T * erow), hv(T * erow), hcs((size_t)P * D), hg(D); for (auto& v : hk) v = f32_to_bf16(nd(rng)); for (auto& v : hv) v = f32_to_bf16(nd(rng)); for (auto& v : hg) v = f32_to_bf16(gd(rng));
    for (long p = 0; p < P; ++p) for (long i = 0; i < D / 2; ++i) { double ang = p * std::pow(10000.0, -2.0 * i / D); hcs[(size_t)p * D + i] = f32_to_bf16((float)std::cos(ang)); hcs[(size_t)p * D + D / 2 + i] = f32_to_bf16((float)std::sin(ang)); }
    std::vector<int32_t> hpos(T); for (auto& p : hpos) p = pd(rng); if (T > 2) hpos[T - 1] = (int32_t)P + 5;
    std::vector<int32_t> perm(S); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng); std::vector<int64_t> slots(T); for (long t = 0; t < T; ++t) slots[t] = (t % 9 == 4) ? -1 : (t % 13 == 6 ? S + t : perm[t % S]);
    std::vector<int32_t> slots32(slots.begin(), slots.end());
    const float ksc = 0.09f, vsc = 0.21f, eps = 1e-6f;
    void *dk, *dv, *dpos, *dcs, *dg, *dkc, *dvc, *dsl, *dout; size_t cbytes = (size_t)S * erow * esz;
    CK(hipMalloc(&dk, hk.size() * 2)); CK(hipMalloc(&dv, hv.size() * 2)); CK(hipMalloc(&dpos, T * 4)); CK(hipMalloc(&dcs, hcs.size() * 2)); CK(hipMalloc(&dg, D * 2)); CK(hipMalloc(&dkc, cbytes + 64)); CK(hipMalloc(&dvc, cbytes + 64)); CK(hipMalloc(&dsl, T * 8)); CK(hipMalloc(&dout, 64));
    CK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), hv.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dpos, hpos.data(), T * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dcs, hcs.data(), hcs.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dg, hg.data(), D * 2, hipMemcpyHostToDevice));
    if (s64) CK(hipMemcpy(dsl, slots.data(), T * 8, hipMemcpyHostToDevice)); else CK(hipMemcpy(dsl, slots32.data(), T * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dkc, 0xAB, cbytes + 64)); CK(hipMemset(dvc, 0xCD, cbytes + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[128]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    int32_t t32 = (int32_t)T, p32 = (int32_t)P, s32 = (int32_t)S; put(&dk, 8); put(&dout, 8); put(&dpos, 8); put(&dcs, 8); put(&t32, 4); put(&p32, 4); put(&dg, 8); put(&eps, 4); sz = 56; put(&dv, 8); put(&dkc, 8); put(&dvc, 8); put(&dsl, 8); put(&s32, 4); if (fp8) { put(&ksc, 4); put(&vsc, 4); }
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long Lr = neox ? D / 16 : D / 8, rpw = 64 / Lr, nw = std::min<long>(4, (H + rpw - 1) / rpw); long hl = H * Lr; bool packed = hl <= 32 && (hl & (hl - 1)) == 0; long tpw = packed ? 64 / hl : 1; if (packed) nw = 4; int blockx = (int)(64 * nw); int grid = packed ? (int)((T + 4 * tpw - 1) / (4 * tpw)) : (int)T;
    auto launch = [&]() { CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(); CK(hipDeviceSynchronize());
    std::vector<uint8_t> kc(cbytes + 64), vc(cbytes + 64); CK(hipMemcpy(kc.data(), dkc, cbytes + 64, hipMemcpyDeviceToHost)); CK(hipMemcpy(vc.data(), dvc, cbytes + 64, hipMemcpyDeviceToHost));
    long kbad = 0, vbad = 0; std::vector<char> written(S, 0); std::vector<double> xn(D), yr(D);
    for (long t = 0; t < T; ++t) { long sl = slots[t]; if (sl < 0 || sl >= S) continue; written[sl] = 1; int p = hpos[t]; bool oob = p >= P;
        for (long h = 0; h < H; ++h) {
            const uint16_t* x = &hk[(t * H + h) * D]; double ss = 0; for (long i = 0; i < D; ++i) { double v = bf16_to_f32(x[i]); ss += v * v; } double inv = 1.0 / std::sqrt(ss / D + eps); for (long i = 0; i < D; ++i) xn[i] = bf16_to_f32(x[i]) * inv * bf16_to_f32(hg[i]);
            auto C = [&](long j) { return oob ? 0.0 : (double)bf16_to_f32(hcs[(size_t)p * D + j]); }; auto Sn = [&](long j) { return oob ? 0.0 : (double)bf16_to_f32(hcs[(size_t)p * D + D / 2 + j]); };
            for (long i = 0; i < D; ++i) { if (neox) { long j = i % (D / 2); double a = xn[j], c = xn[D / 2 + j]; yr[i] = i < D / 2 ? a * C(j) - c * Sn(j) : c * C(j) + a * Sn(j); } else { long j = i / 2; double a = xn[2 * j], c = xn[2 * j + 1]; yr[i] = (i % 2 == 0) ? a * C(j) - c * Sn(j) : c * C(j) + a * Sn(j); } }
            for (long i = 0; i < D; ++i) { size_t idx = ((size_t)sl * H + h) * D + i; double yk = yr[i], yv = bf16_to_f32(hv[(t * H + h) * D + i]);
                if (!fp8) { double gk = bf16_to_f32(*(uint16_t*)&kc[idx * 2]); uint16_t vraw = *(uint16_t*)&vc[idx * 2]; if (!(std::fabs(gk - yk) <= std::fabs(yk) / 128.0 * 1.05 + 2e-3)) { if (kbad < 5) std::fprintf(stderr, "k bad t=%ld h=%ld i=%ld: got %g ref %g\n", t, h, i, gk, yk); ++kbad; } if (vraw != hv[(t * H + h) * D + i]) { if (vbad < 5) std::fprintf(stderr, "v bad t=%ld h=%ld i=%ld\n", t, h, i); ++vbad; } }
                else { double gk = fp8_to_f32(kc[idx]) * ksc, gv = fp8_to_f32(vc[idx]) * vsc; auto okf = [&](double y, double q, double sc) { double sat = std::max(-448.0 * sc, std::min(448.0 * sc, y)); return std::fabs(q - sat) <= std::fabs(sat) / 16.0 * 1.05 + std::ldexp(1.0, -9) * sc * 1.1 + 2e-3; }; if (!okf(yk, gk, ksc)) { if (kbad < 5) std::fprintf(stderr, "k bad t=%ld h=%ld i=%ld: got %g ref %g\n", t, h, i, gk, yk); ++kbad; } if (!okf(yv, gv, vsc)) { if (vbad < 5) std::fprintf(stderr, "v bad t=%ld h=%ld i=%ld: got %g ref %g\n", t, h, i, gv, yv); ++vbad; } } } } }
    long ubad = 0; for (long s = 0; s < S; ++s) if (!written[s]) for (size_t i = 0; i < erow * esz; i += 5) if (kc[(size_t)s * erow * esz + i] != 0xAB || vc[(size_t)s * erow * esz + i] != 0xCD) { ++ubad; break; }
    long guard = 0; for (size_t i = cbytes; i < cbytes + 64; ++i) if (kc[i] != 0xAB || vc[i] != 0xCD) ++guard;
    std::printf("check: qknorm_rope_kv %s %s%s T=%ld H=%ld D=%ld S=%ld k_bad=%ld v_bad=%ld untouched_bad=%ld past_end_bad=%ld %s\n", argv[3], fp8 ? "fp8" : "bf16", s64 ? " i64" : "", T, H, D, S, kbad, vbad, ubad, guard, (kbad == 0 && vbad == 0 && ubad == 0 && guard == 0) ? "QKV_OK" : "QKV_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) launch();
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) launch(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)T * erow * (2 + 2 + 2 * esz);
    std::printf("timing: %.4f ms/iter (%.1f us), %.1f GB/s (traffic %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}