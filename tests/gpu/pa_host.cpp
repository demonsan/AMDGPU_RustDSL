// Paged-KV decode host (bf16, D = 128): ours (`pa_decode_bf16_hd128_p<P>` + `attn_combine_bf16_hd128`, NHD pages [num_pages, P, HK, D]) and
// aiter's gfx950 `pa_bf16_noquant_gqa{8,16}_1tg_4w.co` (block 16, K [blocks, HK, D / 8, 16, 8], V [blocks, HK, 16 / 8, D, 8], 272-byte
// KernelArgs, grid (HK, B), 256 threads) on the same logical data; CPU reference for every (sequence, head).
//
//   pa_host ours <decode.co> <combine.co> B H HK CTX P S
//   pa_host vendor <pa.co> <kernel> B H HK CTX
//   env: ITERS (50), WARMUP (10), SEED (1); context lengths random in [CTX / 2, CTX]
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

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)
static int envi(const char* n, int d) { const char* v = getenv(n); return v ? atoi(v) : d; }
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t tobf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
template <class T> static T* dev(const std::vector<T>& h) { T* d; HC(hipMalloc(&d, h.size() * sizeof(T) + 256)); HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }

int main(int argc, char** argv) {
    const bool ours = std::string(argv[1]) == "ours";
    const int a0 = 4;
    const int B = atoi(argv[a0]), H = atoi(argv[a0 + 1]), HK = atoi(argv[a0 + 2]), CTX = atoi(argv[a0 + 3]);
    const int P = ours ? atoi(argv[a0 + 4]) : 16, S = ours ? atoi(argv[a0 + 5]) : 1;
    const int D = 128, iters = envi("ITERS", 50), warm = envi("WARMUP", 10);
    // QLEN=q (ours, pa_decode16_bf16_hd128_p<P>_mtp): q new query tokens per sequence (Q / out [B][q][H][D], causal among them);
    // SINK=1: attention sinks (attn_combine_bf16_sink_hd128, per-head logit sink[h])
    const int QL = envi("QLEN", 1);
    const bool sinks = envi("SINK", 0) != 0;
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<int> cl(B);
    for (auto& c : cl) c = CTX - (int)(rng() % (CTX / 2 + 1));
    const int maxp = (CTX + P - 1) / P;
    int npages = 0;
    for (int c : cl) npages += (c + P - 1) / P;
    npages += 3;  // spare pages
    // logical K / V per (sequence, key, kv head, d); pages assigned in a random permutation
    std::vector<int> perm(npages);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> bt((size_t)B * maxp, 0);
    int nxt = 0;
    for (int b = 0; b < B; b++)
        for (int p = 0; p < (cl[b] + P - 1) / P; p++) bt[(size_t)b * maxp + p] = perm[nxt++];
    // NHD pages
    std::vector<uint16_t> kc((size_t)npages * P * HK * D), vc(kc.size()), q((size_t)B * QL * H * D);
    for (auto& x : kc) x = tobf(N(rng));
    for (auto& x : vc) x = tobf(N(rng));
    for (auto& x : q) x = tobf(N(rng));
    // FP8KV=1 (pa_decode16_fp8_hd128_p<P>): fp8 e4m3 KV with per-tensor scales k 0.25 / v 0.5; the reference sees code * scale (exact in bf16)
    // FP8KV=2 (pa_decode16_fp8tok_hd128_b<P>): the BLOCKED layout (K [NB][HK][D/16][P][16], V [NB][HK][D][P]) with per-token scales
    // ks / vs f32 [HK][NB P] (powers of two: the reference values code * scale stay exact in bf16)
    const int fp8mode = envi("FP8KV", 0);
    const bool fp8kv = fp8mode != 0;
    const float ksc = 0.25f, vsc = 0.5f;
    std::vector<uint8_t> kq, vq;
    std::vector<float> kss, vss;
    if (fp8mode == 2) {
        auto e4 = [](uint8_t c) { int s = c >> 7, e = (c >> 3) & 15, m = c & 7; float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7); return s ? -v : v; };
        const size_t S_ = (size_t)npages * P;
        kss.resize((size_t)HK * S_); vss.resize(kss.size());
        for (auto& s : kss) s = std::ldexp(1.0f, -(int)(rng() % 4));
        for (auto& s : vss) s = std::ldexp(1.0f, -(int)(rng() % 4));
        if (envi("FP8UNIT", 0)) { std::fill(kss.begin(), kss.end(), 1.0f); std::fill(vss.begin(), vss.end(), 1.0f); }  // (debug: unit per-token scales)
        kq.assign(kc.size(), 0); vq.assign(vc.size(), 0);
        for (int pg = 0; pg < npages; pg++)
            for (int t = 0; t < P; t++)
                for (int hk = 0; hk < HK; hk++)
                    for (int d = 0; d < D; d++) {
                        uint8_t ck = (uint8_t)(((rng() % 2) << 7) | ((4 + rng() % 7) << 3) | (rng() % 8)), cv = (uint8_t)(((rng() % 2) << 7) | ((4 + rng() % 7) << 3) | (rng() % 8));
                        size_t slot = (size_t)pg * P + t, nhd = (((size_t)pg * P + t) * HK + hk) * D + d;
                        kq[(((size_t)pg * HK + hk) * (D / 16) + d / 16) * P * 16 + t * 16 + d % 16] = ck;
                        vq[(((size_t)pg * HK + hk) * D + d) * P + t] = cv;
                        kc[nhd] = tobf(e4(ck) * kss[hk * S_ + slot]); vc[nhd] = tobf(e4(cv) * vss[hk * S_ + slot]);
                    }
    } else if (fp8kv) {
        auto e4 = [](uint8_t c) { int s = c >> 7, e = (c >> 3) & 15, m = c & 7; float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7); return s ? -v : v; };
        kq.resize(kc.size()); vq.resize(vc.size());
        for (size_t i = 0; i < kc.size(); i++) {
            uint8_t ck = (uint8_t)(((rng() % 2) << 7) | ((4 + rng() % 7) << 3) | (rng() % 8)), cv = (uint8_t)(((rng() % 2) << 7) | ((4 + rng() % 7) << 3) | (rng() % 8));
            kq[i] = ck; vq[i] = cv; kc[i] = tobf(e4(ck) * ksc); vc[i] = tobf(e4(cv) * vsc);
        }
    }
    auto kidx = [&](int b, int key, int hk, int d) { return (((size_t)bt[(size_t)b * maxp + key / P] * P + key % P) * HK + hk) * D + d; };
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<uint16_t> out((size_t)B * QL * H * D);
    std::vector<float> sink(H);
    for (auto& s : sink) s = N(rng) * 2.0f;
    uint16_t* dout;
    HC(hipMalloc(&dout, out.size() * 2));
    std::vector<uint8_t> a, ca;
    hipFunction_t fn, cfn = nullptr;
    int gx, gy, gz, blk;
    std::vector<void*> keep;
    // ours: the kernel names are pa_decode_bf16_hd128_p<P> / attn_combine_bf16_hd128
    hipModule_t mod;
    HC(hipModuleLoad(&mod, argv[2]));
    // KPREFIX: the kernel family of ours (pa_decode_bf16_hd128_p / pa_decode16_bf16_hd128_p)
    std::string kname = ours ? (getenv("KPREFIX") ? std::string(getenv("KPREFIX")) : std::string("pa_decode_bf16_hd128_p")) + std::to_string(P) + (getenv("KSUFFIX") ? getenv("KSUFFIX") : "") : argv[3];
    HC(hipModuleGetFunction(&fn, mod, kname.c_str()));
    void *dq = dev(q), *dbt = dev(bt), *dcl = dev(cl);
    const int NS = 4 * S;
    if (ours) {
        hipModule_t cm;
        HC(hipModuleLoad(&cm, argv[3]));
        HC(hipModuleGetFunction(&cfn, cm, sinks ? "attn_combine_bf16_sink_hd128" : "attn_combine_bf16_hd128"));
        void *dk = fp8kv ? (void*)dev(kq) : (void*)dev(kc), *dv = fp8kv ? (void*)dev(vq) : (void*)dev(vc), *dpo, *dpl;
        HC(hipMalloc(&dpo, (size_t)B * QL * H * NS * D * 4)); HC(hipMalloc(&dpl, (size_t)B * QL * H * NS * 4));
        a.assign(16 * 16, 0);
        memcpy(&a[240], &QL, 4);  // (qlen: read by the _mtp kernels only)
        if (fp8mode == 2) { void *dks = dev(kss), *dvs = dev(vss); memcpy(&a[208], &dks, 8); memcpy(&a[224], &dvs, 8); }
        else if (fp8kv) { memcpy(&a[208], &ksc, 4); memcpy(&a[224], &vsc, 4); }
        auto P_ = [&](int i, const void* p) { memcpy(&a[16 * i], &p, 8); };
        auto U = [&](int i, int v) { memcpy(&a[16 * i], &v, 4); };
        // FUSED=1 (ours `_f`, S = 1): the kernel writes the bf16 output itself (ptr slot 0), no combine
        const bool fused = envi("FUSED", 0) != 0;
        P_(0, fused ? (void*)dout : dpo); P_(1, dq); P_(2, dk); P_(3, dv); P_(4, dbt); P_(5, dcl); memcpy(&a[96], &scale, 4); U(7, H); U(8, HK); U(9, maxp); U(10, npages); U(11, S); P_(12, dpl);
        if (fused) cfn = nullptr;
        ca.assign(16 * 6, 0);
        if (sinks) { void* ds = dev(sink); memcpy(&ca[80], &ds, 8); }
        memcpy(&ca[0], &dout, 8); memcpy(&ca[16], &dpo, 8); memcpy(&ca[32], &dpl, 8); memcpy(&ca[48], &H, 4); memcpy(&ca[64], &NS, 4);
        gx = HK; gy = B; gz = S; blk = envi("BLOCK", 256);  // (BLOCK=512: the 8-wave _f8 form)
    } else if (fp8mode == 2) {
        // vendor pertokenFp8 (pa_bf16_pertokenFp8_*, P = 16): K [blk][hk][d / 16][16][16] and V [blk][hk][d][16] fp8 = the blocked layout
        // built above; KQ / VQ f32 [HK][NB 16] per-token scales (kernarg names O Q K V BT CL KQ VQ sclg mblk batch Qs Bs KVs mtp GQA [QTP])
        // (aiter asm PA: K_QScale / V_QScale [NB][HK][BS] f32 - pa_ps_fwd's stride_scale_page / stride_scale_blk)
        std::vector<float> kv1(kss.size()), vv1(vss.size());
        const size_t S_ = (size_t)npages * P;
        for (int hk = 0; hk < HK; hk++)
            for (size_t sl = 0; sl < S_; sl++) {
                size_t d_ = ((sl / P) * HK + hk) * P + sl % P;
                kv1[d_] = kss[hk * S_ + sl]; vv1[d_] = vss[hk * S_ + sl];
            }
        void *dk = dev(kq), *dv = dev(vq), *dks = dev(kv1), *dvs = dev(vv1);
        a.assign(envi("VARGS", 272), 0);
        auto P_ = [&](int i, const void* p) { memcpy(&a[16 * i], &p, 8); };
        auto U = [&](int i, int v) { memcpy(&a[16 * i], &v, 4); };
        P_(0, dout); P_(1, dq); P_(2, dk); P_(3, dv); P_(4, dbt); P_(5, dcl); P_(6, dks); P_(7, dvs);
        float sl = std::log2(std::exp(1.0f)) * scale;
        memcpy(&a[128], &sl, 4);
        U(9, maxp); U(10, HK); U(11, H * D * 2); U(12, HK * 16 * D); U(13, 16 * D); U(14, envi("VMTP", 0)); U(15, H / HK);
        gx = HK; gy = B; gz = 1; blk = 256;
    } else {
        // vendor layouts (x = 8 bf16 per 16 bytes): K [blk][hk][d / 8][16][8], V [blk][hk][16 / 8][d][8]
        std::vector<uint16_t> kv2(kc.size()), vv2(vc.size());
        for (int pg = 0; pg < npages; pg++)
            for (int t = 0; t < 16; t++)
                for (int hk = 0; hk < HK; hk++)
                    for (int d = 0; d < D; d++) {
                        size_t src = (((size_t)pg * 16 + t) * HK + hk) * D + d;
                        kv2[((((size_t)pg * HK + hk) * (D / 8) + d / 8) * 16 + t) * 8 + d % 8] = kc[src];
                        vv2[((((size_t)pg * HK + hk) * 2 + t / 8) * D + d) * 8 + t % 8] = vc[src];
                    }
        void *dk = dev(kv2), *dv = dev(vv2);
        a.assign(288, 0);
        auto P_ = [&](int i, const void* p) { memcpy(&a[16 * i], &p, 8); };
        auto U = [&](int i, int v) { memcpy(&a[16 * i], &v, 4); };
        P_(0, dout); P_(1, dq); P_(2, dk); P_(3, dv); P_(4, dbt); P_(5, dcl);
        float sl = std::log2(std::exp(1.0f)) * scale;
        memcpy(&a[128], &sl, 4);
        U(9, maxp); U(10, HK); U(11, H * D * 2); U(12, HK * 16 * D * 2); U(13, 16 * D * 2); U(14, QL - 1); U(15, H / HK);
        // MTP (aiter pa_fwd: mtp = max_qlen - 1, ptr_QTP = qo_indptr [B + 1])
        if (QL > 1) { std::vector<int> qtp(B + 1); for (int i = 0; i <= B; i++) qtp[i] = i * QL; void* dqtp = dev(qtp); P_(16, dqtp); }
        gx = HK; gy = B; gz = 1; blk = 256;
    }
    size_t asz = a.size(), csz = ca.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    void* ccfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ca.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &csz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() {
        HC(hipModuleLaunchKernel(fn, gx, gy, gz, blk, 1, 1, 0, nullptr, nullptr, cfg));
        // (NOCOMB=1: time the decode kernel alone - the check then reads stale output, use for timing only)
        if (cfn && !envi("NOCOMB", 0)) HC(hipModuleLaunchKernel(cfn, H, B * QL, 1, 128, 1, 1, 0, nullptr, nullptr, ccfg));
    };
    launch();
    HC(hipDeviceSynchronize());
    HC(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0;
    double maxe = 0;
    for (int b = 0; b < B; b++)
      for (int qp = 0; qp < QL; qp++)
        for (int h = 0; h < H; h++) {
            int hk = h / (H / HK);
            const int kl = std::max(0, cl[b] - QL + 1 + qp);  // the keys query position qp sees
            const size_t qr = (size_t)b * QL + qp;
            std::vector<double> s(kl);
            double m = sinks ? (double)sink[h] : -1e300;
            for (int k = 0; k < kl; k++) {
                double acc = 0;
                for (int d = 0; d < D; d++) acc += (double)bf(q[(qr * H + h) * D + d]) * bf(kc[kidx(b, k, hk, d)]);
                s[k] = acc * scale;
                m = std::max(m, s[k]);
            }
            double l = sinks ? std::exp(sink[h] - m) : 0;
            std::vector<double> o(D, 0.0);
            for (int k = 0; k < kl; k++) {
                double p = std::exp(s[k] - m);
                l += p;
                for (int d = 0; d < D; d++) o[d] += p * bf(vc[kidx(b, k, hk, d)]);
            }
            for (int d = 0; d < D; d++) {
                double ref = kl ? o[d] / l : 0, got = bf(out[(qr * H + h) * D + d]);
                double e = std::fabs(got - ref);
                maxe = std::max(maxe, e);
                if (e > 1e-2 * std::fabs(ref) + 2e-2) {
                    if (bad < envi("SHOW_BAD", 0)) printf("  bad b=%d h=%d d=%d got=%g ref=%g\n", b, h, d, got, ref);
                    bad++;
                }
            }
        }
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    for (int i = 0; i < warm; i++) launch();
    HC(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch();
    HC(hipEventRecord(e1));
    HC(hipEventSynchronize(e1));
    float ms; HC(hipEventElapsedTime(&ms, e0, e1));
    double us = 1e3 * ms / iters, bytes = 0;
    for (int c : cl) bytes += 2.0 * c * HK * D * 2;
    printf("%s %s B=%d H=%d HK=%d CTX=%d P=%d S=%d: %.2f us %.0f GB/s (KV) bad=%ld maxerr=%.3e %s\n", ours ? "ours" : "vendor", kname.c_str(), B, H, HK, CTX, P, S, us,
           bytes / (us * 1e-6) / 1e9, bad, maxe, bad ? "PA_FAIL" : "PA_OK");
    return bad ? 3 : 0;
}
