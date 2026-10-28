// Runs the vendor reference kernel (aiter fwd_hd128_mxfp8.co) on random fp8 data with E8M0 1x32
// Q/K descales and compares it with a CPU reference. Also dumps the bf16 output so our own kernel
// can be compared against the vendor kernel on identical inputs.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static double fp8_e4m3(unsigned char b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    double v;
    if (e == 0xF && m == 0x7) v = 0.0;  // NaN encoding never generated
    else if (e == 0) v = (m / 8.0) * std::ldexp(1.0, -6);
    else v = (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return s ? -v : v;
}

static void put64(unsigned char* a, int off, const void* p) {
    uint64_t v = (uint64_t)(uintptr_t)p;
    std::memcpy(a + off, &v, 8);
}
static void put32(unsigned char* a, int off, uint32_t v) { std::memcpy(a + off, &v, 4); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* co = argc > 1 ? argv[1] : "ref_mxfp8.co";
    auto envi = [](const char* n, int d) { const char* v = std::getenv(n); return v ? std::atoi(v) : d; };
    const int B = envi("REF_B", 1), H = envi("REF_H", 2), S = envi("REF_S", 512), SK = envi("REF_SK", 512);
    const int D = 128;
    const float softmax_scale = 1.0f / std::sqrt((float)D);

    unsigned int seed = 20260101u;
    auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (seed >> 16) & 0xFFu; };
    static const unsigned char palette[11] = {0x00, 0x20, 0x28, 0x30, 0x38, 0x3C, 0x40, 0xA0, 0xA8, 0xB0, 0xB8};

    const size_t nq = (size_t)B * S * H * D, nk = (size_t)B * SK * H * D;
    std::vector<unsigned char> q(nq), k(nk), v(nk);
    for (auto& x : q) x = palette[rnd() % 11];
    for (auto& x : k) x = palette[rnd() % 11];
    for (auto& x : v) x = palette[rnd() % 11];
    if (std::getenv("REF_QZ")) std::fill(q.begin(), q.end(), 0);
    // E8M0 descales: [B, S, H, 4], padded with zeroed slack rows for the unguarded tile gathers.
    // Descale tensors are contiguous [B, S, H, 4] (batch stride S * H * 4); only the end of the
    // allocation carries zeroed slack for the kernel's unguarded tile gathers.
    const size_t qrows = (size_t)S, krows = (size_t)SK;
    const size_t q_slack = (size_t)(256 + 256) * H * 4, k_slack = (size_t)(512 + 128) * H * 4;
    std::vector<unsigned char> qs((size_t)B * qrows * H * 4 + q_slack, 0), ks((size_t)B * krows * H * 4 + k_slack, 0);    auto qs_at = [&](int b, int s, int h, int blk) -> unsigned char& { return qs[(((size_t)b * qrows + s) * H + h) * 4 + blk]; };
    auto ks_at = [&](int b, int s, int h, int blk) -> unsigned char& { return ks[(((size_t)b * krows + s) * H + h) * 4 + blk]; };
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < H; ++h)
                for (int blk = 0; blk < 4; ++blk) qs_at(b, s, h, blk) = (unsigned char)(126 + rnd() % 3);
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < SK; ++s)
            for (int h = 0; h < H; ++h)
                for (int blk = 0; blk < 4; ++blk) ks_at(b, s, h, blk) = (unsigned char)(126 + rnd() % 3);
    if (std::getenv("REF_SC1")) {
        const int ev = std::atoi(std::getenv("REF_SC1"));
        std::fill(qs.begin(), qs.end(), (unsigned char)(ev > 1 ? ev : 127));
        std::fill(ks.begin(), ks.end(), (unsigned char)(ev > 1 ? ev : 127));
    }
    float vdescale = 1.0f;
    const size_t no = (size_t)B * S * H * D;
    std::vector<uint16_t> out(no, 0);

    void *dq, *dk, *dv, *dqs, *dks, *dvs, *dout;
    CK(hipMalloc(&dq, nq)); CK(hipMalloc(&dk, nk)); CK(hipMalloc(&dv, nk));
    CK(hipMalloc(&dqs, qs.size())); CK(hipMalloc(&dks, ks.size())); CK(hipMalloc(&dvs, 4096));
    CK(hipMalloc(&dout, no * 2));
    CK(hipMemcpy(dq, q.data(), nq, hipMemcpyHostToDevice));
    CK(hipMemcpy(dk, k.data(), nk, hipMemcpyHostToDevice));
    CK(hipMemcpy(dv, v.data(), nk, hipMemcpyHostToDevice));
    CK(hipMemcpy(dqs, qs.data(), qs.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dks, ks.data(), ks.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dvs, &vdescale, 4, hipMemcpyHostToDevice));
    CK(hipMemset(dout, 0, no * 2));

    unsigned char args[656];
    std::memset(args, 0, sizeof(args));
    put64(args, 0x000, dout); put64(args, 0x010, dq); put64(args, 0x020, dk); put64(args, 0x030, dv);
    float passed_scale = softmax_scale * (std::getenv("REF_SCALE_MUL") ? (float)std::atof(std::getenv("REF_SCALE_MUL")) : 1.0f);
    uint32_t sbits; std::memcpy(&sbits, &passed_scale, 4);
    put32(args, 0x050, sbits);
    put32(args, 0x060, S);
    put32(args, 0x070, H * D);                 // Q seq stride (bytes)
    put32(args, 0x080, 256 * H * D);           // Q tile stride
    put32(args, 0x090, D);                     // Q head stride
    put32(args, 0x0A0, (uint32_t)((size_t)S * H * D));  // Q batch stride
    put32(args, 0x0B0, 1);                     // gqa
    put32(args, 0x0C0, H * D);                 // K seq stride
    put32(args, 0x0D0, D);                     // K head stride
    put32(args, 0x0E0, (uint32_t)((size_t)SK * H * D));
    put32(args, 0x0F0, 5);                     // opt
    put32(args, 0x110, SK);
    put32(args, 0x120, D); put32(args, 0x130, D);
    put32(args, 0x140, H);
    put32(args, 0x150, H * D); put32(args, 0x160, D); put32(args, 0x170, (uint32_t)((size_t)SK * H * D));
    put32(args, 0x180, H * D * 2); put32(args, 0x190, D * 2); put32(args, 0x1A0, (uint32_t)((size_t)S * H * D * 2));
    put64(args, 0x200, dqs); put64(args, 0x210, dks); put64(args, 0x220, dvs);
    put32(args, 0x230, (uint32_t)((size_t)qrows * H * 4)); put32(args, 0x240, 4);
    put32(args, 0x250, (uint32_t)((size_t)krows * H * 4)); put32(args, 0x260, 4);

    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co));
    CK(hipModuleGetFunction(&fn, mod, "_ZN5aiter28fmha_fwd_hd128_mxfp8_gfx950E"));
    size_t asz = sizeof(args);
    void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, (S + 255) / 256, H, B, 512, 1, 1, 0, nullptr, nullptr, extra));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dout, no * 2, hipMemcpyDeviceToHost));
    const int iters = envi("REF_ITERS", 0);
    const double flops = 4.0 * B * H * (double)S * SK * D;
    auto time_kernel = [&](hipFunction_t f, const char* label) {
        if (iters <= 0) return;
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        for (int w = 0; w < 3; ++w) CK(hipModuleLaunchKernel(f, (S + 255) / 256, H, B, 512, 1, 1, 0, nullptr, nullptr, extra));
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0));
        for (int w = 0; w < iters; ++w) CK(hipModuleLaunchKernel(f, (S + 255) / 256, H, B, 512, 1, 1, 0, nullptr, nullptr, extra));
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
        const double per = ms / iters;
        std::printf("timing %-8s: %.4f ms/iter, %.2f TFLOP/s (%d iters)\n", label, per, flops / (per * 1e-3) / 1e12, iters);
    };
    time_kernel(fn, "vendor");

    // CPU reference.
    auto bf2f = [](uint16_t x) { uint32_t u = (uint32_t)x << 16; float f; std::memcpy(&f, &u, 4); return f; };
    const int check_rows = envi("REF_ROWS", 64);
    const double cpu_scale = std::getenv("REF_CPU_SCALE") ? std::atof(std::getenv("REF_CPU_SCALE")) : 0.6931471805599453;  // ln 2: the kernel computes exp2(S)
    auto check = [&](const std::vector<uint16_t>& o, const char* label) -> size_t {
        double worst = 0.0, sum_abs = 0.0, max_ref = 0.0;
        size_t bad = 0, cnt = 0;
        std::vector<double> sc(SK), qd(D);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < S && i < check_rows; ++i) {
                    for (int kk = 0; kk < D; ++kk)
                        qd[kk] = fp8_e4m3(q[(((size_t)b * S + i) * H + h) * D + kk]) *
                                 std::ldexp(1.0, (int)qs_at(b, i, h, kk / 32) - 127);
                    double mx = -1e300;
                    for (int t = 0; t < SK; ++t) {
                        double acc = 0.0;
                        for (int kk = 0; kk < D; ++kk)
                            acc += qd[kk] * fp8_e4m3(k[(((size_t)b * SK + t) * H + h) * D + kk]) *
                                   std::ldexp(1.0, (int)ks_at(b, t, h, kk / 32) - 127);
                        sc[t] = acc * cpu_scale;
                        mx = std::max(mx, sc[t]);
                    }
                    double den = 0.0;
                    for (int t = 0; t < SK; ++t) { sc[t] = std::exp(sc[t] - mx); den += sc[t]; }
                    for (int d = 0; d < D; ++d) {
                        double acc = 0.0;
                        for (int t = 0; t < SK; ++t) acc += sc[t] / den * fp8_e4m3(v[(((size_t)b * SK + t) * H + h) * D + d]);
                        acc *= vdescale;
                        const double got = bf2f(o[(((size_t)b * S + i) * H + h) * D + d]);
                        const double err = std::fabs(got - acc);
                        worst = std::max(worst, err); sum_abs += err; max_ref = std::max(max_ref, std::fabs(acc)); ++cnt;
                        if (err > 0.05 + 0.05 * std::fabs(acc)) { if (bad < 5) std::fprintf(stderr, "%s O[%d][%d][%d][%d] want %.5f got %.5f\n", label, b, h, i, d, acc, got); ++bad; }
                    }
                }
        std::printf("%s: checked=%zu bad=%zu worst_err=%.4e mean_err=%.4e max|ref|=%.4f\n", label, cnt, bad, worst, sum_abs / cnt, max_ref);
        return bad;
    };
    size_t bad = check(out, "vendor kernel vs CPU");

    // Optional second kernel (ours) on the same inputs.
    size_t bad2 = 0;
    if (const char* co2 = std::getenv("REF_CO2")) {
        const char* kname2 = std::getenv("REF_KNAME2") ? std::getenv("REF_KNAME2") : "fmha_fwd_hd128_mxfp8";
        void* dout2;
        CK(hipMalloc(&dout2, no * 2));
        CK(hipMemset(dout2, 0, no * 2));
        put64(args, 0x000, dout2);
        hipModule_t mod2; hipFunction_t fn2;
        CK(hipModuleLoad(&mod2, co2));
        CK(hipModuleGetFunction(&fn2, mod2, kname2));
        CK(hipModuleLaunchKernel(fn2, (S + 255) / 256, H, B, 512, 1, 1, 0, nullptr, nullptr, extra));
        CK(hipDeviceSynchronize());
        std::vector<uint16_t> out2(no, 0);
        CK(hipMemcpy(out2.data(), dout2, no * 2, hipMemcpyDeviceToHost));
        bad2 = check(out2, "our kernel   vs CPU");
        time_kernel(fn2, "ours");
        double wd = 0.0, md = 0.0;
        for (size_t i = 0; i < no; ++i) {
            const double dd = std::fabs((double)bf2f(out[i]) - (double)bf2f(out2[i]));
            wd = std::max(wd, dd); md += dd;
        }
        std::printf("our kernel vs vendor kernel (all %zu outputs): worst_diff=%.4e mean_diff=%.4e\n", no, wd, md / no);
        if (const char* dump2 = std::getenv("REF_DUMP2")) {
            FILE* fo = std::fopen(dump2, "wb"); std::fwrite(out2.data(), 2, no, fo); std::fclose(fo);
        }
    }    if (const char* dump = std::getenv("REF_DUMP")) {
        FILE* fo = std::fopen(dump, "wb");
        std::fwrite(out.data(), 2, no, fo);
        std::fclose(fo);
        std::printf("dumped %zu bf16 values to %s\n", no, dump);
    }
    std::printf((bad || bad2) ? "REF_KERNEL_FAIL\n" : "REF_KERNEL_OK\n");
    return (bad || bad2) != 0;
}
