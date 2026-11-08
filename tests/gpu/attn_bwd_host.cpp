// Attention backward host (bf16, D = 128, BSHD, bottom-right causal): ours (`fmha_bwd_dot` + `fmha_bwd_dq` + `fmha_bwd_dkdv`, dK / dV per Q head)
// and aiter's gfx950 fmha_v3_bwd ASM pipeline (odo + dqdkdv a32 + dq_convert) on the same data; O and LSE from a CPU forward; CPU reference
// for sampled query rows (dQ) and key rows (dK, dV).
//
//   attn_bwd_host <co_dir> B H HK SQ SK causal
//   co_dir: fmha_bwd_dot_hd128_bf16.co, fmha_bwd_dq_hd128_bf16[_causal].co, fmha_bwd_dkdv_hd128_bf16[_causal].co
//   env: VENDOR_DIR (aiter hsa/gfx950/fmha_v3_bwd), ITERS (20), SEED (1), CHECK (16)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)
static int envi(const char* n, int d) { const char* v = getenv(n); return v ? atoi(v) : d; }
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t tobf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
template <class T> static T* dev(const std::vector<T>& h) { T* d; HC(hipMalloc(&d, h.size() * sizeof(T) + 256)); HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }
static hipFunction_t load(const std::string& path, const std::string& name) {
    hipModule_t m; hipFunction_t f;
    HC(hipModuleLoad(&m, path.c_str())); HC(hipModuleGetFunction(&f, m, name.c_str()));
    return f;
}
static void launch(hipFunction_t f, std::vector<uint8_t>& a, int gx, int gy, int gz) {
    size_t sz = a.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    HC(hipModuleLaunchKernel(f, gx, gy, gz, 256, 1, 1, 0, nullptr, nullptr, cfg));
}

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s co_dir B H HK SQ SK causal\n", argv[0]); return 1; }
    const std::string dir = argv[1];
    const int B = atoi(argv[2]), H = atoi(argv[3]), HK = atoi(argv[4]), SQ = atoi(argv[5]), SK = atoi(argv[6]), causal = atoi(argv[7]);
    const int D = 128, G = H / HK, iters = envi("ITERS", 20);
    const float scale = 1.0f / std::sqrt((float)D);
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0.0f, 1.0f);
    const size_t nq = (size_t)B * SQ * H * D, nk = (size_t)B * SK * HK * D;
    std::vector<uint16_t> q(nq), k(nk), v(nk), dO(nq), o(nq);
    for (auto& x : q) x = tobf(N(rng));
    for (auto& x : k) x = tobf(N(rng));
    for (auto& x : v) x = tobf(N(rng));
    for (auto& x : dO) x = tobf(N(rng));
    auto Q = [&](int b, int i, int h, int d) { return bf(q[(((size_t)b * SQ + i) * H + h) * D + d]); };
    auto K = [&](int b, int j, int h, int d) { return bf(k[(((size_t)b * SK + j) * HK + h / G) * D + d]); };
    auto V = [&](int b, int j, int h, int d) { return bf(v[(((size_t)b * SK + j) * HK + h / G) * D + d]); };
    auto DO = [&](int b, int i, int h, int d) { return bf(dO[(((size_t)b * SQ + i) * H + h) * D + d]); };
    auto vis = [&](int i, int j) { return !causal || j <= i + SK - SQ; };
    // ---- CPU forward: P rows, O (bf16) and LSE (natural) for every row; keep P (double) per (b, h) ----
    std::vector<float> lse((size_t)B * H * SQ), dot((size_t)B * H * SQ);
    std::vector<double> P((size_t)B * H * SQ * SK);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < SQ; i++) {
                double* p = &P[(((size_t)b * H + h) * SQ + i) * SK];
                double m = -1e300;
                for (int j = 0; j < SK; j++) {
                    double s = 0;
                    if (vis(i, j)) for (int d = 0; d < D; d++) s += (double)Q(b, i, h, d) * K(b, j, h, d);
                    p[j] = vis(i, j) ? s * scale : -1e300;
                    m = std::max(m, p[j]);
                }
                double l = 0;
                for (int j = 0; j < SK; j++) { p[j] = vis(i, j) ? std::exp(p[j] - m) : 0; l += p[j]; }
                for (int j = 0; j < SK; j++) p[j] /= l;
                lse[((size_t)b * H + h) * SQ + i] = (float)(m + std::log(l));
                for (int d = 0; d < D; d++) {
                    double acc = 0;
                    for (int j = 0; j < SK; j++) acc += p[j] * V(b, j, h, d);
                    o[(((size_t)b * SQ + i) * H + h) * D + d] = tobf((float)acc);
                }
                double dd = 0;
                for (int d = 0; d < D; d++) dd += (double)DO(b, i, h, d) * bf(o[(((size_t)b * SQ + i) * H + h) * D + d]);
                dot[((size_t)b * H + h) * SQ + i] = (float)dd;
            }
    // dS[q][k] = P (dP - Dot), dP = dO . V
    auto dS = [&](int b, int h, int i, int j) {
        double p = P[(((size_t)b * H + h) * SQ + i) * SK + j];
        if (p == 0) return 0.0;
        double dp = 0;
        for (int d = 0; d < D; d++) dp += (double)DO(b, i, h, d) * V(b, j, h, d);
        return p * (dp - dot[((size_t)b * H + h) * SQ + i]);
    };
    // ---- device buffers ----
    void *dq_ = dev(q), *dk_ = dev(k), *dv_ = dev(v), *ddo = dev(dO), *dO_ = dev(o), *dlse = dev(lse);
    void *ddot, *ddq, *ddk, *ddv;
    HC(hipMalloc(&ddot, lse.size() * 4)); HC(hipMalloc(&ddq, nq * 2)); HC(hipMalloc(&ddk, (size_t)B * SK * H * D * 2)); HC(hipMalloc(&ddv, (size_t)B * SK * H * D * 2));
    auto check = [&](const char* who) {
        std::vector<uint16_t> gq(nq), gk((size_t)B * SK * H * D), gv(gk.size());
        HC(hipMemcpy(gq.data(), ddq, nq * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(gk.data(), ddk, gk.size() * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(gv.data(), ddv, gv.size() * 2, hipMemcpyDeviceToHost));
        long bad[3] = {0, 0, 0};
        double me[3] = {0, 0, 0};
        auto cmp = [&](int w, double got, double ref) {
            double e = std::fabs(got - ref);
            me[w] = std::max(me[w], e);
            if (e > 2e-2 * std::fabs(ref) + 2e-2) bad[w]++;
        };
        for (int c = 0; c < envi("CHECK", 16); c++) {
            int b = rng() % B, h = rng() % H, i = c == 0 ? 0 : c == 1 ? SQ - 1 : rng() % SQ, j = c == 0 ? 0 : c == 1 ? SK - 1 : rng() % SK;
            std::vector<double> ds(SK);
            for (int jj = 0; jj < SK; jj++) ds[jj] = dS(b, h, i, jj);
            for (int d = 0; d < D; d++) {
                double acc = 0;
                for (int jj = 0; jj < SK; jj++) acc += ds[jj] * K(b, jj, h, d);
                cmp(0, bf(gq[(((size_t)b * SQ + i) * H + h) * D + d]), acc * scale);
            }
            std::vector<double> dsk(SQ), pk(SQ);
            for (int ii = 0; ii < SQ; ii++) { dsk[ii] = dS(b, h, ii, j); pk[ii] = P[(((size_t)b * H + h) * SQ + ii) * SK + j]; }
            for (int d = 0; d < D; d++) {
                double ak = 0, av = 0;
                for (int ii = 0; ii < SQ; ii++) { ak += dsk[ii] * Q(b, ii, h, d); av += pk[ii] * DO(b, ii, h, d); }
                cmp(1, bf(gk[(((size_t)b * SK + j) * H + h) * D + d]), ak * scale);
                cmp(2, bf(gv[(((size_t)b * SK + j) * H + h) * D + d]), av);
            }
        }
        printf("  %s: bad dq=%ld dk=%ld dv=%ld maxerr %.3e %.3e %.3e\n", who, bad[0], bad[1], bad[2], me[0], me[1], me[2]);
        return bad[0] + bad[1] + bad[2];
    };
    double flops = 2.5 * 4.0 * B * H * (double)SQ * SK * D * (causal ? 0.5 : 1.0);  // ~2.5x the forward
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    long fails = 0;
    {
        // ---- ours ----
        std::string cs = causal ? "_causal" : "";
        hipFunction_t fd = load(dir + "/fmha_bwd_dot_hd128_bf16.co", "fmha_bwd_dot_hd128_bf16");
        hipFunction_t fq = load(dir + "/fmha_bwd_dq_hd128_bf16" + cs + ".co", "fmha_bwd_dq_hd128_bf16" + cs);
        hipFunction_t fk = load(dir + "/fmha_bwd_dkdv_hd128_bf16" + cs + ".co", "fmha_bwd_dkdv_hd128_bf16" + cs);
        std::vector<uint8_t> a(16 * 15, 0);
        auto P_ = [&](int i, const void* p) { memcpy(&a[16 * i], &p, 8); };
        auto U = [&](int i, int x) { memcpy(&a[16 * i], &x, 4); };
        P_(0, dq_); P_(1, dk_); P_(2, dv_); P_(3, ddo); P_(4, dlse); P_(5, ddot); P_(6, ddq); P_(7, ddk); P_(8, ddv); P_(14, dO_);
        memcpy(&a[144], &scale, 4); U(10, SQ); U(11, SK); U(12, H); U(13, HK);
        auto run = [&]() {
            launch(fd, a, (SQ + 255) / 256, H, B);
            launch(fq, a, (SQ + 127) / 128, H, B);
            launch(fk, a, (SK + 127) / 128, H, B);
        };
        run();
        HC(hipDeviceSynchronize());
        fails += check("ours");
        HC(hipEventRecord(e0));
        for (int i = 0; i < iters; i++) run();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms; HC(hipEventElapsedTime(&ms, e0, e1));
        double us = 1e3 * ms / iters;
        printf("ours bwd B=%d H=%d HK=%d SQ=%d SK=%d causal=%d: %.1f us %.1f TF/s\n", B, H, HK, SQ, SK, causal, us, flops / (us * 1e-6) / 1e12);
        // per-kernel split
        hipFunction_t fs[3] = {fd, fq, fk};
        int gxs[3] = {(SQ + 255) / 256, (SQ + 127) / 128, (SK + 127) / 128};
        const char* nm[3] = {"dot", "dq", "dkdv"};
        for (int kk = 0; kk < 3; kk++) {
            HC(hipEventRecord(e0));
            for (int i = 0; i < iters; i++) launch(fs[kk], a, gxs[kk], H, B);
            HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
            float m2; HC(hipEventElapsedTime(&m2, e0, e1));
            printf("   %s: %.1f us\n", nm[kk], 1e3 * m2 / iters);
        }
    }
    if (const char* vd = getenv("VENDOR_DIR")) {
        // ---- aiter: odo + dqdkdv (a32, causal = bottom-right) + dq_convert; dK / dV per Q head ----
        std::string vdir = vd;
        std::string kn = causal ? "_ZN5aiter41fmha_bwd_hd128_bf16_causal_br_a32_psskddvE" : "_ZN5aiter31fmha_bwd_hd128_bf16_a32_psskddvE";
        std::string co = causal ? "bwd_hd128_bf16_causal_br_a32_psskddv.co" : "bwd_hd128_bf16_a32_psskddv.co";
        hipFunction_t fo = load(vdir + "/bwd_hd128_odo_bf16.co", "_ZN5aiter23fmha_bwd_hd128_odo_bf16E");
        hipFunction_t fm = load(vdir + "/" + co, kn);
        hipFunction_t fc = load(vdir + "/bwd_hd128_dq_convert_bf16.co", "_ZN5aiter30fmha_bwd_hd128_dq_convert_bf16E");
        void* dacc;
        size_t accb = (size_t)B * H * SQ * D * 4;
        HC(hipMalloc(&dacc, accb));
        const int qs = H * D * 2, qh = D * 2, qb = SQ * H * D * 2, ks = HK * D * 2, kh = D * 2, kb = SK * HK * D * 2;
        std::vector<uint8_t> ao(256, 0), am(704, 0), ac(192, 0);
        auto Pp = [](std::vector<uint8_t>& a, int o, const void* p) { memcpy(&a[o], &p, 8); };
        auto Uu = [](std::vector<uint8_t>& a, int o, int x) { memcpy(&a[o], &x, 4); };
        Pp(ao, 0x00, dO_); Pp(ao, 0x10, ddo); Pp(ao, 0x20, ddot);
        Uu(ao, 0x30, qh); Uu(ao, 0x40, qb); Uu(ao, 0x50, qs); Uu(ao, 0x60, qh); Uu(ao, 0x70, qb); Uu(ao, 0x80, qs);
        Uu(ao, 0x90, SQ * 4); Uu(ao, 0xA0, H * SQ * 4); Uu(ao, 0xB0, 4); Uu(ao, 0xC0, SQ); Uu(ao, 0xD0, D);
        float l2e = 1.4426950408889634f;
        Pp(am, 0x000, dacc); Pp(am, 0x010, ddk); Pp(am, 0x020, ddv); Pp(am, 0x030, dq_); Pp(am, 0x040, dk_); Pp(am, 0x050, dv_); Pp(am, 0x060, ddo);
        Pp(am, 0x070, dlse); Pp(am, 0x080, ddot); memcpy(&am[0x90], &scale, 4); memcpy(&am[0xA0], &l2e, 4);
        Uu(am, 0x0B0, SQ); Uu(am, 0x0C0, 256 * ks); Uu(am, 0x0D0, qh); Uu(am, 0x0E0, qb); Uu(am, 0x0F0, qs); Uu(am, 0x100, G);
        Uu(am, 0x110, kh); Uu(am, 0x120, kb); Uu(am, 0x130, ks); Uu(am, 0x140, H * D * 2); Uu(am, 0x150, SK); Uu(am, 0x160, D); Uu(am, 0x170, D);
        Uu(am, 0x180, H); Uu(am, 0x190, kh); Uu(am, 0x1A0, kb); Uu(am, 0x1B0, ks); Uu(am, 0x1C0, qh); Uu(am, 0x1D0, qb); Uu(am, 0x1E0, qs);
        Uu(am, 0x1F0, D * 2); Uu(am, 0x200, SK * H * D * 2); Uu(am, 0x210, D * 2); Uu(am, 0x220, SK * H * D * 2); Uu(am, 0x230, H * D * 2);
        Uu(am, 0x240, SQ * 4); Uu(am, 0x290, SQ);
        Pp(ac, 0x00, dacc); Pp(ac, 0x10, ddq);
        Uu(ac, 0x20, SQ * D * 4); Uu(ac, 0x30, B == 1 ? 0 : H * SQ * D * 4); Uu(ac, 0x40, D * 4);
        Uu(ac, 0x50, qh); Uu(ac, 0x60, qb); Uu(ac, 0x70, qs); Uu(ac, 0x80, SQ); Uu(ac, 0x90, D);
        int gx = (SK + 255) / 256;
        if (causal) gx = (gx + 1) / 2;
        auto run = [&]() {
            HC(hipMemsetAsync(dacc, 0, accb));
            launch(fo, ao, (SQ + 127) / 128, H, B);
            launch(fm, am, gx, H, B);
            launch(fc, ac, (SQ + 63) / 64, H, B);
        };
        HC(hipMemset(ddq, 0, nq * 2)); HC(hipMemset(ddk, 0, (size_t)B * SK * H * D * 2)); HC(hipMemset(ddv, 0, (size_t)B * SK * H * D * 2));
        run();
        HC(hipDeviceSynchronize());
        check("vendor");
        HC(hipEventRecord(e0));
        for (int i = 0; i < iters; i++) run();
        HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
        float ms; HC(hipEventElapsedTime(&ms, e0, e1));
        double us = 1e3 * ms / iters;
        printf("vendor bwd (odo + dqdkdv a32 + dq_convert): %.1f us %.1f TF/s\n", us, flops / (us * 1e-6) / 1e12);
    }
    printf("%s\n", fails ? "BWD_FAIL" : "BWD_OK");
    return fails ? 3 : 0;
}
