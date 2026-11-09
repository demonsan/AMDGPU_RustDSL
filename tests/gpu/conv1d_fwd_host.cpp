// causal_conv1d fwd (prefill, varlen; batch 2): causal_conv1d_fwd_bf16_w<W>[_silu] and causal_conv1d_fwd_split_bf16_w<W>[_silu]: CPU reference (outputs and the final states) and bandwidth.
// usage: conv1d_fwd_host <co> <kernel> <plain|split> <B> <dim> <W> <k_dim (split)> <silu 0|1> <maxlen> [iters]
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
int main(int argc, char** argv) {
    if (argc < 10) { std::fprintf(stderr, "usage: %s <co> <kernel> <plain|split> <B> <dim> <W> <k_dim> <silu> <maxlen> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; bool split = std::string(argv[3]) == "split"; long B = std::atol(argv[4]), dim = std::atol(argv[5]), W = std::atol(argv[6]), KD = std::atol(argv[7]); bool silu = std::atoi(argv[8]); long maxlen = std::atol(argv[9]); int iters = argc > 10 ? std::atoi(argv[10]) : 10; bool uniform = argc > 11 && std::atoi(argv[11]); const bool sched = std::string(kn).find("_sched") != std::string::npos;   // uniform: every sequence has `maxlen` tokens
    const long SL = W - 1, SLOTS = B + B / 2 + 3, VD = dim - 2 * KD;
    std::mt19937 rng(23); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int32_t> cu(B + 1, 0), cidx(B); std::vector<uint8_t> hinit(B); std::vector<int32_t> perm(SLOTS); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
    for (long b = 0; b < B; ++b) { long len = uniform ? maxlen : (b % 6 == 1) ? 1 : (b % 6 == 3 ? 2 : (b % 6 == 5 ? SL + 1 : 1 + (long)(rng() % (uint64_t)maxlen))); cu[b + 1] = cu[b] + (int32_t)len; cidx[b] = (b % 7 == 4) ? -1 : perm[b]; hinit[b] = (b % 3 != 0); }
    const long T = cu[B];
    std::vector<uint16_t> hx((size_t)T * dim), hw((size_t)dim * W), hb(dim), hs((size_t)SLOTS * dim * SL); for (auto& v : hx) v = f32_to_bf16(nd(rng)); for (auto& v : hw) v = f32_to_bf16(nd(rng) * 0.5f); for (auto& v : hb) v = f32_to_bf16(nd(rng)); for (auto& v : hs) v = f32_to_bf16(nd(rng));
    void *dx, *dw, *db, *dcu, *dci, *dhi, *ds; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dw, hw.size() * 2)); CK(hipMalloc(&db, dim * 2)); CK(hipMalloc(&dcu, (B + 1) * 4)); CK(hipMalloc(&dci, B * 4)); CK(hipMalloc(&dhi, B + 4)); CK(hipMalloc(&ds, hs.size() * 2 + 64));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(db, hb.data(), dim * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dcu, cu.data(), (B + 1) * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dci, cidx.data(), B * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dhi, hinit.data(), B, hipMemcpyHostToDevice)); CK(hipMemcpy(ds, hs.data(), hs.size() * 2, hipMemcpyHostToDevice));
    void *dout[3] = {nullptr, nullptr, nullptr}; size_t osz[3] = {(size_t)T * dim * 2, 0, 0}; if (split) { osz[0] = (size_t)T * KD * 2; osz[1] = (size_t)T * KD * 2; osz[2] = (size_t)T * VD * 2; } int nout = split ? 3 : 1;
    for (int i = 0; i < nout; ++i) { CK(hipMalloc(&dout[i], osz[i] + 64)); CK(hipMemset(dout[i], 0x55, osz[i] + 64)); }
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[128]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    put(&dx, 8); put(&dw, 8); put(&db, 8); put(&dcu, 8); put(&dci, 8); put(&dhi, 8); put(&ds, 8); for (int i = 0; i < nout; ++i) put(&dout[i], 8);
    int32_t d32 = (int32_t)dim, k32 = (int32_t)KD, t32 = (int32_t)T, s32 = (int32_t)SLOTS; put(&d32, 4); if (split) put(&k32, 4); put(&t32, 4); put(&s32, 4);
    std::vector<int32_t> sq, ck; for (long b = 0; b < B; ++b) { long len = cu[b + 1] - cu[b]; for (long c = 0; c * 8 < len; ++c) { sq.push_back((int32_t)b); ck.push_back((int32_t)c); } } void *dsq = nullptr, *dck = nullptr; int32_t nc32 = (int32_t)sq.size();
    if (sched) { CK(hipMalloc(&dsq, sq.size() * 4 + 4)); CK(hipMalloc(&dck, ck.size() * 4 + 4)); CK(hipMemcpy(dsq, sq.data(), sq.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dck, ck.data(), ck.size() * 4, hipMemcpyHostToDevice)); put(&dsq, 8); put(&dck, 8); put(&nc32, 4); int32_t nseq32 = (int32_t)B; put(&nseq32, 4); }
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long maxl = 0; for (long b = 0; b < B; ++b) maxl = std::max<long>(maxl, cu[b + 1] - cu[b]);
    unsigned gx = (unsigned)((dim + 2047) / 2048), gz = (unsigned)((maxl + 7) / 8);
    auto launch = [&]() { if (sched) CK(hipModuleLaunchKernel(fn, gx, (unsigned)nc32, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); else CK(hipModuleLaunchKernel(fn, gx, (unsigned)B, gz, 256, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(); CK(hipDeviceSynchronize());
    std::vector<std::vector<uint8_t>> ho(nout); for (int i = 0; i < nout; ++i) { ho[i].resize(osz[i] + 64); CK(hipMemcpy(ho[i].data(), dout[i], osz[i] + 64, hipMemcpyDeviceToHost)); }
    std::vector<uint16_t> hs2(hs.size()); CK(hipMemcpy(hs2.data(), ds, hs.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0, sbad = 0; std::vector<uint16_t> hs_ref = hs;
    for (long b = 0; b < B; ++b) { long s0 = cu[b], len = cu[b + 1] - cu[b]; long slot = cidx[b]; bool hi = hinit[b];
        if (slot < 0) { for (long t = 0; t < len; ++t) for (long f = 0; f < dim; ++f) { long tk = s0 + t; const uint8_t* p; size_t off; if (!split) { p = ho[0].data(); off = ((size_t)tk * dim + f) * 2; } else if (f < KD) { p = ho[0].data(); off = ((size_t)tk * KD + f) * 2; } else if (f < 2 * KD) { p = ho[1].data(); off = ((size_t)tk * KD + f - KD) * 2; } else { p = ho[2].data(); off = ((size_t)tk * VD + f - 2 * KD) * 2; } uint16_t v; std::memcpy(&v, p + off, 2); if (v != 0x5555) ++bad; } continue; }
        for (long f = 0; f < dim; ++f) {
            auto X = [&](long p) -> double { if (p >= 0) return bf16_to_f32(hx[(size_t)(s0 + p) * dim + f]); long j = p + SL; return hi ? bf16_to_f32(hs[((size_t)slot * dim + f) * SL + j]) : 0.0; };
            for (long t = 0; t < len; ++t) {
                double acc = bf16_to_f32(hb[f]); for (long i = 0; i < W; ++i) acc += (double)bf16_to_f32(hw[f * W + i]) * X(t - (W - 1) + i); if (silu) acc = acc / (1.0 + std::exp(-acc));
                long tk = s0 + t; const uint8_t* p; size_t off; if (!split) { p = ho[0].data(); off = ((size_t)tk * dim + f) * 2; } else if (f < KD) { p = ho[0].data(); off = ((size_t)tk * KD + f) * 2; } else if (f < 2 * KD) { p = ho[1].data(); off = ((size_t)tk * KD + f - KD) * 2; } else { p = ho[2].data(); off = ((size_t)tk * VD + f - 2 * KD) * 2; }
                uint16_t v; std::memcpy(&v, p + off, 2); double got = bf16_to_f32(v); if (!(std::fabs(got - acc) <= std::fabs(acc) / 128.0 * 1.05 + 1e-5)) { if (bad < 5) std::fprintf(stderr, "bad b=%ld t=%ld f=%ld: got %g ref %g\n", b, t, f, got, acc); ++bad; }
            }
            for (long j = 0; j < SL; ++j) hs_ref[((size_t)slot * dim + f) * SL + j] = f32_to_bf16((float)X(len - SL + j));
        } }
    for (size_t i = 0; i < hs.size(); ++i) if (hs2[i] != hs_ref[i]) { if (sbad < 5) std::fprintf(stderr, "state bad at %zu: got %04x want %04x\n", i, hs2[i], hs_ref[i]); ++sbad; }
    long guard = 0; for (int i = 0; i < nout; ++i) for (size_t k = osz[i]; k < osz[i] + 64; ++k) if (ho[i][k] != 0x55) ++guard;
    std::printf("check: conv1d_fwd%s B=%ld T=%ld dim=%ld W=%ld%s bad=%ld state_bad=%ld past_end_bad=%ld %s\n", split ? "_split" : "", B, T, dim, W, silu ? " silu" : "", bad, sbad, guard, (bad == 0 && sbad == 0 && guard == 0) ? "CONVF_OK" : "CONVF_FAIL");
    hipEvent_t a, bb; CK(hipEventCreate(&a)); CK(hipEventCreate(&bb));
    for (int i = 0; i < 3; ++i) launch();
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) launch(); CK(hipEventRecord(bb)); CK(hipEventSynchronize(bb)); float ms; CK(hipEventElapsedTime(&ms, a, bb)); ms /= iters;
    double bytes = (double)T * dim * 4 + (double)B * dim * SL * 4;
    std::printf("timing: %.4f ms/iter (%.1f us), %.1f GB/s (traffic %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}