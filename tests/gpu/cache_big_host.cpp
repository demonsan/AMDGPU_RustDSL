// KV-cache write into caches larger than 4 GiB (batch 2, B1 64-bit descriptor bases): reshape_cache_flash_bf16_h<Hk>_d<D> and reshape_cache_blocked_bf16_h<Hk>_d<D>_b<BS>.
// Scattered slots over the WHOLE cache (including past 4 GiB and 8 GiB), a few invalid slots (negative, >= S); every touched slot / block is compared byte-exactly with a CPU reference, and a sample of untouched
// rows keeps its sentinel.   usage: cache_big_host <co> <kernel> <flash|blocked> <T> <Hk> <D> <BS> <NB>      (flash: S = NB * BS slots; caches are S * Hk * D * 2 bytes each)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    if (argc < 9) { std::fprintf(stderr, "usage: %s <co> <kernel> <flash|blocked> <T> <Hk> <D> <BS> <NB>\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; bool blocked = std::string(argv[3]) == "blocked"; long T = std::atol(argv[4]), Hk = std::atol(argv[5]), D = std::atol(argv[6]), BS = std::atol(argv[7]), NB = std::atol(argv[8]);
    const long S = NB * BS; const size_t row = (size_t)Hk * D * 2; const size_t cbytes = (size_t)S * row;
    std::printf("cache bytes per tensor: %.2f GiB\n", cbytes / 1073741824.0);
    std::mt19937_64 rng(123); std::vector<uint8_t> hk(T * row), hv(T * row); for (auto& v : hk) v = rng() & 255; for (auto& v : hv) v = rng() & 255;
    std::vector<int32_t> slots(T); std::set<int32_t> used;
    for (long t = 0; t < T; ++t) { int32_t s; do { s = (int32_t)(rng() % (uint64_t)S); } while (used.count(s)); if (t % 11 == 5) s = -1 - (int32_t)(t % 7); else if (t % 13 == 4) s = (int32_t)std::min<long>(S + t, INT32_MAX); else used.insert(s); slots[t] = s; }
    if (!used.empty()) { slots[0] = 0; used.insert(0); slots[1] = (int32_t)(S - 1); used.insert((int32_t)(S - 1)); }   // the first and the last slot of the cache (0 may collide with a random one: harmless, the later token wins in the reference below)
    void *dk, *dv, *dkc, *dvc, *dsl; CK(hipMalloc(&dk, T * row)); CK(hipMalloc(&dv, T * row)); CK(hipMalloc(&dkc, cbytes)); CK(hipMalloc(&dvc, cbytes)); CK(hipMalloc(&dsl, T * 4));
    CK(hipMemcpy(dk, hk.data(), T * row, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, hv.data(), T * row, hipMemcpyHostToDevice)); CK(hipMemcpy(dsl, slots.data(), T * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dkc, 0xAB, cbytes)); CK(hipMemset(dvc, 0xCD, cbytes));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *k, *v, *kc, *vc, *sl; int32_t t, s; } args{dk, dv, dkc, dvc, dsl, (int32_t)T, (int32_t)S};
    size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long nv = (long)(row / 16); long lt = std::min<long>(4, (nv + 63) / 64) * 64, tpb = lt <= 128 ? 256 / lt : 1; int blockx = (int)(lt * tpb); int grid = (int)((T + tpb - 1) / tpb);
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, blockx, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    // the last writer of a slot wins (in the kernel the order of two tokens with the same slot is undefined; the slots chosen above are distinct except possibly slot 0 / S-1, which are skipped in the check if duplicated)
    std::vector<int> count(0); std::set<int32_t> dup; { std::set<int32_t> seen; for (long t = 0; t < T; ++t) { if (slots[t] < 0 || slots[t] >= S) continue; if (!seen.insert(slots[t]).second) dup.insert(slots[t]); } }
    long bad = 0, checked = 0, untouched_bad = 0;
    auto fetch = [&](void* base, size_t off, size_t n, std::vector<uint8_t>& out) { out.resize(n); CK(hipMemcpy(out.data(), (char*)base + off, n, hipMemcpyDeviceToHost)); };
    std::vector<uint8_t> got;
    for (long t = 0; t < T; ++t) {
        long s = slots[t]; if (s < 0 || s >= S || dup.count((int32_t)s)) continue;
        const uint8_t* ks = &hk[t * row]; const uint8_t* vs = &hv[t * row];
        if (!blocked) {
            fetch(dkc, (size_t)s * row, row, got); if (std::memcmp(got.data(), ks, row)) ++bad;
            fetch(dvc, (size_t)s * row, row, got); if (std::memcmp(got.data(), vs, row)) ++bad; ++checked;
        } else {
            long blk = s / BS, off = s % BS; size_t bb = (size_t)BS * row;
            // the touched elements of the block: key [Hk][D/8][BS][8], value [Hk][D][BS] (2-byte elements); everything else in the block (written by other tokens of the same block) is skipped
            std::vector<uint8_t> kb, vb; fetch(dkc, (size_t)blk * bb, bb, kb); fetch(dvc, (size_t)blk * bb, bb, vb);
            for (long h = 0; h < Hk; ++h) for (long d = 0; d < D; ++d) {
                size_t ki = (((size_t)h * (D / 8) + d / 8) * BS + off) * 8 + d % 8, vi = ((size_t)h * D + d) * BS + off;
                if (std::memcmp(&kb[ki * 2], &ks[((size_t)h * D + d) * 2], 2)) ++bad;
                if (std::memcmp(&vb[vi * 2], &vs[((size_t)h * D + d) * 2], 2)) ++bad;
            }
            ++checked;
        }
    }
    // untouched rows (flash) / blocks (blocked): a sample keeps its sentinel
    std::mt19937_64 r2(7); for (int i = 0; i < 2000; ++i) { long s = (long)(r2() % (uint64_t)S); bool touched = used.count((int32_t)s) > 0; if (blocked) { touched = false; for (long u = 0; u < BS && !touched; ++u) touched = used.count((int32_t)((s / BS) * BS + u)) > 0; } if (touched) continue;
        size_t n = blocked ? (size_t)BS * row : row, off = (blocked ? (size_t)(s / BS) * BS * row : (size_t)s * row); fetch(dkc, off, n, got); for (uint8_t x : got) if (x != 0xAB) { ++untouched_bad; break; } fetch(dvc, off, n, got); for (uint8_t x : got) if (x != 0xCD) { ++untouched_bad; break; } }
    std::printf("check: cache_big %s T=%ld Hk=%ld D=%ld BS=%ld S=%ld slots checked=%ld bad=%ld untouched_bad=%ld %s\n", argv[3], T, Hk, D, BS, S, checked, bad, untouched_bad, (bad == 0 && untouched_bad == 0) ? "CBIG_OK" : "CBIG_FAIL");
    return 0;
}