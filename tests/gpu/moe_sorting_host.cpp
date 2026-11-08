// MoE alignment / sorting (batch 2): moe_sorting_e<E>_k<K>_b<BM> against a CPU reference (multiset per expert, padding, expert ids per tile, tile starts, counts) and latency.
// usage: moe_sorting_host <co> <kernel> <M> <E> <K> <BM> [iters] [skew]     (skew=1: half of the pairs go to expert 0; invalid ids -1 / E are sprinkled in)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s <co> <kernel> <M> <E> <K> <BM> [iters] [skew]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long M = std::atol(argv[3]), E = std::atol(argv[4]), K = std::atol(argv[5]), BM = std::atol(argv[6]); int iters = argc > 7 ? std::atoi(argv[7]) : 20; bool skew = argc > 8 && std::atoi(argv[8]);
    const long N = M * K, maxp = N + E * BM - K, ntile = maxp / BM;
    std::mt19937 rng(5); std::vector<int32_t> ids(N); std::vector<float> w(N);
    for (long i = 0; i < N; ++i) { ids[i] = skew && (rng() & 1) ? 0 : (int32_t)(rng() % E); w[i] = (float)(rng() % 1000) / 1000.f; if (skew && i % 53 == 7) ids[i] = (i % 2) ? -1 : (int32_t)E; }
    void *did, *dw, *dsid, *dsw, *deid, *dnv, *dts; CK(hipMalloc(&did, N * 4)); CK(hipMalloc(&dw, N * 4)); CK(hipMalloc(&dsid, maxp * 4 + 64)); CK(hipMalloc(&dsw, maxp * 4 + 64)); CK(hipMalloc(&deid, ntile * 4 + 64)); CK(hipMalloc(&dnv, 64)); CK(hipMalloc(&dts, (E + 1) * 4 + 64));
    CK(hipMemcpy(did, ids.data(), N * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, w.data(), N * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dsid, 0x55, maxp * 4 + 64)); CK(hipMemset(dsw, 0x55, maxp * 4 + 64)); CK(hipMemset(deid, 0x55, ntile * 4 + 64)); CK(hipMemset(dnv, 0x55, 64)); CK(hipMemset(dts, 0x55, (E + 1) * 4 + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct __attribute__((packed)) { void *i, *w, *sid, *sw, *eid, *nv, *ts; int32_t m; } args{did, dw, dsid, dsw, deid, dnv, dts, (int32_t)M};
    size_t sz = sizeof(args); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint32_t> sid(maxp + 16); std::vector<float> sw(maxp + 16); std::vector<int32_t> eid(ntile + 16), nv(16), ts(E + 1 + 16);
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
    std::printf("check: moe_sorting M=%ld E=%ld K=%ld BM=%ld%s bad=%ld past_end_bad=%ld %s\n", M, E, K, BM, skew ? " skew" : "", bad, guard, (bad == 0 && guard == 0) ? "MOE_OK" : "MOE_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 10; ++i) CK(hipModuleLaunchKernel(fn, 1, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, 1, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    std::printf("timing: %.4f ms/iter (%.2f us) for %ld routed pairs\n", ms, ms * 1e3, N);
    return 0;
}