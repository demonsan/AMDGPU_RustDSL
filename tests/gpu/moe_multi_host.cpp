// MoE alignment / sorting (batch 2): moe_sorting_e<E>_k<K>_b<BM> against a CPU reference (multiset per expert, padding, expert ids per tile, tile starts, counts) and latency.
// usage: moe_multi_host <co_count> <co_scan> <co_scatter> <M> <E> <K> <BM> [iters] [skew]   (the three kernels moe_count_e<E>_k<K>, moe_scan_e<E>_k<K>_b<BM>, moe_scatter_e<E>_k<K> of moe/multi.rs; G = min(64, ceil(M K / 4096)) blocks)     (skew=1: half of the pairs go to expert 0; invalid ids -1 / E are sprinkled in)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: %s <co_count> <co_scan> <co_scatter> <M> <E> <K> <BM> [iters] [skew]\n", argv[0]); return 2; }
    long M = std::atol(argv[4]), E = std::atol(argv[5]), K = std::atol(argv[6]), BM = std::atol(argv[7]); int iters = argc > 8 ? std::atoi(argv[8]) : 20; bool skew = argc > 9 && std::atoi(argv[9]);
    const long N = M * K, maxp = N + E * BM - K, ntile = maxp / BM;
    std::mt19937 rng(5); std::vector<int32_t> ids(N); std::vector<float> w(N);
    for (long i = 0; i < N; ++i) { ids[i] = skew && (rng() & 1) ? 0 : (int32_t)(rng() % E); w[i] = (float)(rng() % 1000) / 1000.f; if (skew && i % 53 == 7) ids[i] = (i % 2) ? -1 : (int32_t)E; }
    void *did, *dw, *dsid, *dsw, *deid, *dnv, *dts; CK(hipMalloc(&did, N * 4)); CK(hipMalloc(&dw, N * 4)); CK(hipMalloc(&dsid, maxp * 4 + 64)); CK(hipMalloc(&dsw, maxp * 4 + 64)); CK(hipMalloc(&deid, ntile * 4 + 64)); CK(hipMalloc(&dnv, 64)); CK(hipMalloc(&dts, (E + 1) * 4 + 64));
    CK(hipMemcpy(did, ids.data(), N * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, w.data(), N * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dsid, 0x55, maxp * 4 + 64)); CK(hipMemset(dsw, 0x55, maxp * 4 + 64)); CK(hipMemset(deid, 0x55, ntile * 4 + 64)); CK(hipMemset(dnv, 0x55, 64)); CK(hipMemset(dts, 0x55, (E + 1) * 4 + 64));
    char kc[64], ks[64], kt[64]; std::snprintf(kc, 64, "moe_count_e%ld_k%ld", E, K); std::snprintf(ks, 64, "moe_scan_e%ld_k%ld_b%ld", E, K, BM); std::snprintf(kt, 64, "moe_scatter_e%ld_k%ld", E, K);
    hipModule_t m1, m2, m3; hipFunction_t f1, f2, f3; CK(hipModuleLoad(&m1, argv[1])); CK(hipModuleGetFunction(&f1, m1, kc)); CK(hipModuleLoad(&m2, argv[2])); CK(hipModuleGetFunction(&f2, m2, ks)); CK(hipModuleLoad(&m3, argv[3])); CK(hipModuleGetFunction(&f3, m3, kt));
    long G = std::min<long>(64, (N + 4095) / 4096); if (G < 1) G = 1; long P = (N + G - 1) / G;
    void* dws; CK(hipMalloc(&dws, E * 256 + 64));
    struct __attribute__((packed)) { void *i, *ws; int32_t m, per; } a1{did, dws, (int32_t)M, (int32_t)P};
    struct __attribute__((packed)) { void *ws, *sid, *sw, *eid, *nv, *ts; int32_t m, g; } a2{dws, dsid, dsw, deid, dnv, dts, (int32_t)M, (int32_t)G};
    struct __attribute__((packed)) { void *i, *w, *sid, *sw, *ws; int32_t m, per; } a3{did, dw, dsid, dsw, dws, (int32_t)M, (int32_t)P};
    size_t s1 = sizeof(a1), s2 = sizeof(a2), s3 = sizeof(a3);
    void* c1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &s1, HIP_LAUNCH_PARAM_END};
    void* c2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a2, HIP_LAUNCH_PARAM_BUFFER_SIZE, &s2, HIP_LAUNCH_PARAM_END};
    void* c3[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a3, HIP_LAUNCH_PARAM_BUFFER_SIZE, &s3, HIP_LAUNCH_PARAM_END};
    auto run = [&]() { CK(hipModuleLaunchKernel(f1, (int)G, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, c1)); CK(hipModuleLaunchKernel(f2, 1, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, c2)); CK(hipModuleLaunchKernel(f3, (int)G, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, c3)); };
    run(); CK(hipDeviceSynchronize());    std::vector<uint32_t> sid(maxp + 16); std::vector<float> sw(maxp + 16); std::vector<int32_t> eid(ntile + 16), nv(16), ts(E + 1 + 16);
    CK(hipMemcpy(sid.data(), dsid, (maxp + 16) * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(sw.data(), dsw, (maxp + 16) * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(eid.data(), deid, (ntile + 16) * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(nv.data(), dnv, 64, hipMemcpyDeviceToHost)); CK(hipMemcpy(ts.data(), dts, (E + 17) * 4, hipMemcpyDeviceToHost));
    // reference
    std::vector<long> cnt(E, 0); for (long i = 0; i < N; ++i) if (ids[i] >= 0 && ids[i] < E) ++cnt[ids[i]];
    std::vector<long> padded(E), off(E + 1, 0); for (long e = 0; e < E; ++e) { padded[e] = (cnt[e] + BM - 1) / BM * BM; off[e + 1] = off[e] + padded[e]; }
    long total = off[E], bad = 0; const uint32_t sentinel = ((uint32_t)K << 24) | (uint32_t)M;
    if (nv[0] != total || nv[1] != M) { std::fprintf(stderr, "num_valid_ids: got [%d, %d] ref [%ld, %ld]\n", nv[0], nv[1], total, M); ++bad; }
    for (long e = 0; e <= E; ++e) if (ts[e] != off[e] / BM) { if (bad < 5) std::fprintf(stderr, "tile_start[%ld]: got %d ref %ld\n", e, ts[e], off[e] / BM); ++bad; }
    for (long j = 0; j < ntile; ++j) { long ref = E; if (j * BM < total) { long e = 0; while (off[e + 1] <= j * BM) ++e; ref = e; } if (eid[j] != ref) { if (bad < 5) std::fprintf(stderr, "expert_ids[%ld]: got %d ref %ld\n", j, eid[j], ref); ++bad; } }
    for (long e = 0; e < E; ++e) {
        std::multiset<std::pair<uint32_t, uint32_t>> got, ref;   // (packed id, weight bits)
        for (long p = off[e]; p < off[e] + cnt[e]; ++p) { uint32_t wb; std::memcpy(&wb, &sw[p], 4); got.insert({sid[p], wb}); }
        for (long i = 0; i < N; ++i) if (ids[i] == e) { uint32_t wb; std::memcpy(&wb, &w[i], 4); ref.insert({(uint32_t)(((i % K) << 24) | (i / K)), wb}); }
        if (got != ref) { if (bad < 5) std::fprintf(stderr, "expert %ld: group differs\n", e); ++bad; }
        for (long p = off[e] + cnt[e]; p < off[e + 1]; ++p) if (sid[p] != sentinel || sw[p] != 0.f) { if (bad < 5) std::fprintf(stderr, "pad at %ld wrong: %08x\n", p, sid[p]); ++bad; }
    }
    for (long p = total; p < maxp; ++p) if (sid[p] != sentinel || sw[p] != 0.f) { if (bad < 5) std::fprintf(stderr, "tail at %ld wrong: %08x\n", p, sid[p]); ++bad; }
    long guard = 0; for (long p = maxp; p < maxp + 16; ++p) if (sid[p] != 0x55555555u) ++guard; for (long j = ntile; j < ntile + 16; ++j) if (eid[j] != 0x55555555) ++guard; for (long e = E + 1; e < E + 17; ++e) if (ts[e] != 0x55555555) ++guard;
    std::printf("check: moe_multi G=%ld M=%ld E=%ld K=%ld BM=%ld%s bad=%ld past_end_bad=%ld %s\n", G, M, E, K, BM, skew ? " skew" : "", bad, guard, (bad == 0 && guard == 0) ? "MOE_OK" : "MOE_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 10; ++i) run();
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) run();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    std::printf("timing: %.4f ms/iter (%.2f us) for %ld routed pairs\n", ms, ms * 1e3, N);
    return 0;
}