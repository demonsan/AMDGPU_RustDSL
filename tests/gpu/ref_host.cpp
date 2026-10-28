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
    const size_t qrows = (size_t)((S + 255) / 256) * 256 + 256, krows = (size_t)((SK + 127) / 128) * 128 + 512;
    std::vector<unsigned char> qs((size_t)B * qrows * H * 4, 0), ks((size_t)B * krows * H * 4, 0);
    auto qs_at = [&](int b, int s, int h, int blk) -> unsigned char& { return qs[(((size_t)b * qrows + s) * H + h) * 4 + blk]; };
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
        std::fill(qs.begin(), qs.end(), 127);
        std::fill(ks.begin(), ks.end(), 127);
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

    // CPU reference.
    auto bf2f = [](uint16_t x) { uint32_t u = (uint32_t)x << 16; float f; std::memcpy(&f, &u, 4); return f; };
    double worst = 0.0, sum_abs = 0.0, max_ref = 0.0;
    size_t bad = 0, cnt = 0;
    const int check_rows = envi("REF_ROWS", 64);
    const double cpu_scale = std::getenv("REF_CPU_SCALE") ? std::atof(std::getenv("REF_CPU_SCALE")) : (double)softmax_scale;
    std::vector<double> sc(SK), qd(D), kd(D);
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
                    const double got = bf2f(out[(((size_t)b * S + i) * H + h) * D + d]);
                    const double err = std::fabs(got - acc);
                    worst = std::max(worst, err); sum_abs += err; max_ref = std::max(max_ref, std::fabs(acc)); ++cnt;
                    if (err > 0.05 + 0.05 * std::fabs(acc)) { if (bad < 5) std::fprintf(stderr, "ref O[%d][%d][%d][%d] want %.5f got %.5f\n", b, h, i, d, acc, got); ++bad; }
                }
            }
    std::printf("ref kernel: checked=%zu bad=%zu worst_err=%.4e mean_err=%.4e max|ref|=%.4f\n", cnt, bad, worst, sum_abs / cnt, max_ref);
    if (const char* dump = std::getenv("REF_DUMP")) {
        FILE* fo = std::fopen(dump, "wb");
        std::fwrite(out.data(), 2, no, fo);
        std::fclose(fo);
        std::printf("dumped %zu bf16 values to %s\n", no, dump);
    }
    std::printf(bad ? "REF_KERNEL_FAIL\n" : "REF_KERNEL_OK\n");
    return bad != 0;
}
