// per-row top-K of f32 logits (batch 2): topk_per_row_{decode,prefill}_f32_k<K>: the selected columns must be distinct, inside the row, and carry exactly the K largest values (a multiset check: ties may pick any
// of the equal elements); rows shorter than K are returned whole in column order followed by -1.
// usage: topk_per_row_host <co> <kernel> <decode|prefill> <B or R> <next_n (decode; 1 for prefill)> <width = stride> <len (decode: the longest seq len; prefill: the longest row)> <K> [iters] [ties 0|1]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
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
    if (argc < 9) { std::fprintf(stderr, "usage: %s <co> <kernel> <decode|prefill> <B|R> <next_n> <width> <len> <K> [iters] [ties]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; bool prefill = std::string(argv[3]) == "prefill"; long B = std::atol(argv[4]), NN = std::atol(argv[5]), W = std::atol(argv[6]), LEN = std::atol(argv[7]), K = std::atol(argv[8]); int iters = argc > 9 ? std::atoi(argv[9]) : 10; bool ties = argc > 10 && std::atoi(argv[10]);
    const long R = prefill ? B : B * NN;
    std::mt19937 rng(17); std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> hl((size_t)R * W); for (auto& v : hl) v = nd(rng);
    std::vector<int32_t> starts(R, 0), ends(R, 0), seq(prefill ? 0 : B), rs(R), re(R);
    if (!prefill) { const long lens[4] = {LEN, std::max(NN, LEN / 8), std::max(NN, LEN / 64), std::max(NN, LEN / 2)}; for (long b = 0; b < B; ++b) seq[b] = (int32_t)std::min<long>(W + NN - 1, lens[b % 4] + (b % 7 == 3 ? 5 : 0)); for (long r = 0; r < R; ++r) { rs[r] = 0; re[r] = seq[r / NN] - (int32_t)NN + (int32_t)(r % NN) + 1; } }
    else { for (long r = 0; r < R; ++r) { long len = (r % 5 == 1) ? std::max<long>(1, K / 3) : (r % 5 == 3 ? K : (LEN - (r % 11) * 7)); len = std::min<long>(len, W); long st = (r % 3 == 0) ? 0 : (r * 13) % std::max<long>(1, W - len + 1); rs[r] = (int32_t)st; re[r] = (int32_t)(st + len); } }
    if (ties) for (auto& v : hl) v = std::round(v * 2.f) / 2.f;      // many equal values
    void *dl, *ds, *de, *dsq, *dout; CK(hipMalloc(&dl, hl.size() * 4)); CK(hipMalloc(&dout, (size_t)R * K * 4 + 64)); CK(hipMemset(dout, 0x55, (size_t)R * K * 4 + 64)); CK(hipMemcpy(dl, hl.data(), hl.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&ds, R * 4 + 4)); CK(hipMalloc(&de, R * 4 + 4)); CK(hipMalloc(&dsq, B * 4 + 4)); CK(hipMemcpy(ds, rs.data(), R * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(de, re.data(), R * 4, hipMemcpyHostToDevice)); if (!prefill) CK(hipMemcpy(dsq, seq.data(), B * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[64]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    int32_t w32 = (int32_t)W, nn32 = (int32_t)NN, r32 = (int32_t)R;
    if (prefill) { put(&dl, 8); put(&ds, 8); put(&de, 8); put(&dout, 8); put(&w32, 4); put(&r32, 4); } else { put(&dl, 8); put(&dsq, 8); put(&dout, 8); put(&w32, 4); put(&nn32, 4); put(&r32, 4); }
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() { CK(hipModuleLaunchKernel(fn, (unsigned)(prefill ? R : B), (unsigned)(prefill ? 1 : NN), 1, 1024, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(); CK(hipDeviceSynchronize());
    std::vector<int32_t> ho((size_t)R * K + 16); CK(hipMemcpy(ho.data(), dout, ho.size() * 4, hipMemcpyDeviceToHost));
    long bad = 0, short_rows = 0;
    for (long r = 0; r < R; ++r) {
        long st = rs[r], L = std::max<long>(0, std::min<long>(re[r] - rs[r], W)); const int32_t* o = &ho[(size_t)r * K]; const float* row = &hl[(size_t)r * W];
        if (L <= K) { ++short_rows; for (long p = 0; p < K; ++p) { long exp = p < L ? st + p : -1; if (o[p] != exp) { if (bad < 5) std::fprintf(stderr, "short row %ld pos %ld: got %d want %ld\n", r, p, o[p], exp); ++bad; break; } } continue; }
        std::set<int32_t> seen; std::vector<float> got; bool ok = true;
        for (long p = 0; p < K; ++p) { int32_t c = o[p]; if (c < st || c >= st + L || !seen.insert(c).second) { ok = false; if (bad < 5) std::fprintf(stderr, "row %ld pos %ld: bad column %d (range [%ld,%ld))\n", r, p, c, st, st + L); break; } got.push_back(row[c]); }
        if (ok) { std::vector<float> all(row + st, row + st + L); std::nth_element(all.begin(), all.begin() + K, all.end(), std::greater<float>()); std::vector<float> want(all.begin(), all.begin() + K); std::sort(want.begin(), want.end()); std::sort(got.begin(), got.end()); if (want != got) { ok = false; if (bad < 5) std::fprintf(stderr, "row %ld: the selected values are not the top-%ld\n", r, K); } }
        if (!ok) ++bad;
    }
    long guard = 0; for (size_t i = (size_t)R * K; i < (size_t)R * K + 16; ++i) if (ho[i] != 0x55555555) ++guard;
    std::printf("check: topk_per_row_%s R=%ld W=%ld K=%ld%s bad_rows=%ld (short rows %ld) past_end_bad=%ld %s\n", prefill ? "prefill" : "decode", R, W, K, ties ? " ties" : "", bad, short_rows, guard, (bad == 0 && guard == 0) ? "TKR_OK" : "TKR_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) launch();
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) launch();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = 0; for (long r = 0; r < R; ++r) bytes += (double)std::max<long>(0, std::min<long>(re[r] - rs[r], W)) * 4 + K * 4.0;
    std::printf("timing: %.4f ms/iter (%.1f us), %.1f GB/s of one row read (+ output)\n", ms, ms * 1e3, bytes / ms / 1e6);
    return 0;
}