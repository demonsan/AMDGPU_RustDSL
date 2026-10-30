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
#include <string>
#include <limits>

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
    const int HK = envi("REF_HK", H), GQA = H / HK;
    const int D = 128;
    const float softmax_scale = 1.0f / std::sqrt((float)D);

    // ---- tiling configuration / dispatch (defaults preserve the historical 256-row / 512-thread run) ----
    // Vendor: fixed 256-row tiling (ts_qo=256), 512 threads, grid (ceil(S/256), H, B).
    // Ours: two tilings share the ABI - 256 rows/512 threads (_vs128) and 128 rows/256 threads (_t128).
    //   REF_CO2 / REF_KNAME2             = the 256-row kernel (or a single kernel)
    //   REF_CO2_SMALL / REF_KNAME2_SMALL = the 128-row kernel; when present, dispatch is automatic:
    //       t128 iff 2*ceil(S/256)*H*B <= 256 (its own grid fits the 256 CUs in one wave), else _vs128.
    //   Overrides: REF_ROWS_PER_BLOCK=128|256, REF_GRID128=1, REF_OURS_BLOCK_THREADS.
    const char* kname_env = std::getenv("REF_KNAME2");
    const char* co_small = std::getenv("REF_CO2_SMALL");
    const char* kname_small = std::getenv("REF_KNAME2_SMALL") ? std::getenv("REF_KNAME2_SMALL") : "fmha_fwd_hd128_mxfp8_t128";
    // KV-split candidate (docs/kvsplit-design.md): split kernel writes Ohat/lse partials, combine kernel merges.
    const char* co_split = std::getenv("REF_CO2_SPLIT");
    const char* kname_split = std::getenv("REF_KNAME2_SPLIT") ? std::getenv("REF_KNAME2_SPLIT") : "fmha_fwd_hd128_mxfp8_kvsplit";
    const char* co_comb = std::getenv("REF_CO2_COMBINE");
    const char* kname_comb = std::getenv("REF_KNAME2_COMBINE") ? std::getenv("REF_KNAME2_COMBINE") : "fmha_fwd_hd128_mxfp8_kvsplit_combine4h";
    const int splits = envi("REF_SPLIT", 0);
    const bool force_split = envi("REF_SPLIT_FORCE", 0) != 0;
    const long rows_total = (long)B * S * H;
    const int vendor_rows = envi("REF_VENDOR_ROWS_PER_BLOCK", 256);
    const int cus = envi("REF_CUS", 256);                 // MI350: 256 active CUs/package
    const unsigned vendor_block = 512;                    // vendor: 512 threads = 8 waves
    const int forced_rows = envi("REF_ROWS_PER_BLOCK", 0);
    const bool k_t128 = kname_env && std::strstr(kname_env, "t128") != nullptr;
    const long blocks256 = (long)((S + 255) / 256) * H * B;
    const int s_rows_pb = 256;                            // split kernel uses the _vs128 8-wave block shape
    const int s_gx = (S + s_rows_pb - 1) / s_rows_pb;
    const long s_blocks = (long)s_gx * H * B * (splits > 1 ? splits : 1);
    // The split kernel writes fp16 partials by default (FMHA_OPT=nop16 -> fp32); the buffer is allocated
    // fp32-sized so both variants fit. The guard model uses REF_SPLIT_PART_BYTES (default 2 = fp16).
    const double part_elem_bytes = (double)envi("REF_SPLIT_PART_BYTES", 2);
    const double ohat_bytes = (double)splits * rows_total * D * part_elem_bytes;
    const double ohat_alloc_bytes = (double)splits * rows_total * D * 4;
    const double lse_bytes = (double)splits * rows_total * 4;
    const double comb_threads_d = (double)envi("REF_COMBINE_THREADS", 256);

    // Dispatch guard for the KV-split candidate (docs/gen4-tiling-design.md section 11):
    //   use it only when N256 <= 128 and N256*s <= 256 (one full block wave) and the modelled
    //   T_kvsplit(s) < 0.95 * T_current; otherwise fall back to the certified _t128/_vs128 rule.
    bool use_split = false; const char* split_why = "REF_SPLIT unset";
    if (splits > 1 && co_split) {
        // Calibrated against the Step A/B measurements (docs/kvsplit-results.md section validation):
        // the split kernel does not reach the _vs128 per-CU rate at 128/256 CUs (L2/HBM contention), and the
        // fp16 partials are written serially in the epilogue (no overlap with the MFMA stream), so the model
        // is rate(CUs) + serial write + combine + launch gap. Coefficients are measured, not peak numbers.
        const double F = 4.0 * B * H * (double)S * SK * D;
        const double cus_kv = std::min<double>(256.0, (double)blocks256 * splits);
        const double t_current = (blocks256 <= 128)
            ? F / (4.03e12 * std::min<double>(256.0, 2.0 * (double)blocks256))
            : F / (5.90e12 * std::min<double>(256.0, (double)blocks256));
        // Calibrated on the final fp16 build (docs/kvsplit-results.md "validation" section):
        //   compute-only 18.8 us at 256 CUs (= 3.57 TFLOP/s/CU; 5.9 at 64 CUs, 4.88 at 128 CUs measured/derived),
        //   fp16 partial writes are serial in the epilogue: 17.1 MB -> ~4.2 us (4.1 TB/s),
        //   combine4h: 10.6 MB -> 3.8 us / 21.3 MB -> 5.2 us (6.25 TB/s + 1.8 us fixed).
        auto rate_at = [](double cus) {          // measured TFLOP/s/CU for the kvsplit phase-1 kernel
            if (cus <= 64.0) return 5.90;
            if (cus <= 128.0) return 5.90 + (4.88 - 5.90) * (cus - 64.0) / 64.0;
            return 4.88 + (3.57 - 4.88) * (cus - 128.0) / 128.0;
        };
        const double part_b = ohat_bytes + lse_bytes;
        const double comb_b = part_b + rows_total * D * 2;
        const double t_kv = F / (1e12 * rate_at(cus_kv) * cus_kv)
                          + part_b / 4.1e12 + 1.8e-6 + comb_b / 6.25e12;
        // Split key ranges are 128-key tile aligned (kernel-opt contract): tiles_per_split = ceil(ntiles/splits),
        // split i owns [i*tps, min(ntiles,(i+1)*tps)); every split must own >= 1 tile for the Step A kernel.
        const int ntiles = (SK + 127) / 128;
        const int tps = (ntiles + splits - 1) / splits;
        const bool tiles_ok = (splits >= 1) && (tps >= 1) && ((splits - 1) * tps < ntiles);
        const bool guard = (blocks256 <= 128) && (blocks256 * splits <= 256) && tiles_ok && (t_kv < 0.95 * t_current);
        if (force_split) { use_split = true; split_why = "REF_SPLIT_FORCE"; }
        else if (guard) { use_split = true; split_why = "auto-split guard: N256<=128 && N256*s<=256 && model win"; }
        else { split_why = "split guard rejected (fallback)"; }
        std::printf("[split] s=%d %s: blocks256=%ld phase1 blocks=%ld ntiles=%d tps=%d; model T_current=%.2f us T_kvsplit=%.2f us; %s\n",
                    splits, use_split ? "CHOSEN" : "not chosen", blocks256, s_blocks,
                    (SK + 127) / 128, ((SK + 127) / 128 + splits - 1) / splits,
                    t_current * 1e6, t_kv * 1e6, split_why);
    }
    int rows_pb; const char* why;
    if (use_split) { rows_pb = s_rows_pb; why = "KV-split (main+combine)"; }
    else if (forced_rows == 128 || forced_rows == 256) { rows_pb = forced_rows; why = "REF_ROWS_PER_BLOCK override"; }
    else if (envi("REF_GRID128", 0) != 0) { rows_pb = 128; why = "REF_GRID128=1"; }
    else if (co_small) { rows_pb = blocks256 <= 128 ? 128 : 256; why = blocks256 <= 128 ? "auto: 2*ceil(S/256)*H*B <= 256 CUs" : "auto: 2*ceil(S/256)*H*B > 256 CUs"; }
    else { rows_pb = k_t128 ? 128 : 256; why = "kernel name"; }
    const char* use_co = use_split ? co_split : ((rows_pb == 128 && co_small) ? co_small : std::getenv("REF_CO2"));
    const char* use_kn = use_split ? kname_split : ((rows_pb == 128 && co_small) ? kname_small : (kname_env ? kname_env : "fmha_fwd_hd128_mxfp8"));
    const int ours_rows = use_split ? s_rows_pb : rows_pb;
    const unsigned ours_block = (unsigned)envi("REF_OURS_BLOCK_THREADS", (use_split || rows_pb == 256) ? 512 : 256);
    const int vendor_stride_rows = envi("REF_VENDOR_TILE_STRIDE_ROWS", 256);
    const int ours_stride_rows = envi("REF_OURS_TILE_STRIDE_ROWS", ours_rows);
    const int gvx = (S + vendor_rows - 1) / vendor_rows, gy = H, gz = B;
    const int gox = use_split ? s_gx : (S + ours_rows - 1) / ours_rows;
    const int goz = use_split ? B * splits : B;
    const long gv_blocks = (long)gvx * gy * gz, go_blocks = (long)gox * gy * goz;
    std::printf("[grid] vendor rows/block=%d block=%u grid=(%d,%d,%d) blocks=%ld (%.1f%% of %d CUs); "
                "ours (%s): rows/block=%d block=%u grid=(%d,%d,%d) blocks=%ld (%.1f%% of %d CUs) [%s; blocks256=%ld]%s\n",
                vendor_rows, vendor_block, gvx, gy, gz, gv_blocks, 100.0 * (double)gv_blocks / cus, cus,
                use_kn, ours_rows, ours_block, gox, gy, use_split ? goz : gz, go_blocks, 100.0 * (double)go_blocks / cus, cus,
                why, blocks256, use_split ? ", split phase-1" : "");
    if (use_split) {
        const long cgrid = (long)std::ceil((double)rows_total * D / comb_threads_d);
        std::printf("[split] s=%d: phase1 Ohat=%.1f MB (model %.0f B/elem) + lse=%.2f MB, grid=(%d,%d,%d) blocks=%ld block=512; "
                    "combine grid=%ld block=%u; guard=%s\n",
                    splits, ohat_bytes / 1e6, part_elem_bytes, lse_bytes / 1e6, gox, gy, goz, s_blocks,
                    cgrid, (unsigned)comb_threads_d, split_why);
        if (!co_comb) std::printf("[split] WARNING: REF_CO2_SPLIT is set but REF_CO2_COMBINE is not - combine cannot launch\n");
    }
    if (vendor_rows != 256)
        std::printf("[grid] WARNING: the vendor binary is fixed at 256 rows/block; REF_VENDOR_ROWS_PER_BLOCK=%d is not supported by it\n", vendor_rows);
    if (std::getenv("REF_GRID_ONLY")) return 0;

    unsigned int seed = 20260101u;
    auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (seed >> 16) & 0xFFu; };
    static const unsigned char palette[11] = {0x00, 0x20, 0x28, 0x30, 0x38, 0x3C, 0x40, 0xA0, 0xA8, 0xB0, 0xB8};

    const size_t nq = (size_t)B * S * H * D, nk = (size_t)B * SK * HK * D;
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
    std::vector<unsigned char> qs((size_t)B * qrows * H * 4 + q_slack, 0), ks((size_t)B * krows * HK * 4 + k_slack, 0);    auto qs_at = [&](int b, int s, int h, int blk) -> unsigned char& { return qs[(((size_t)b * qrows + s) * H + h) * 4 + blk]; };
    auto ks_at = [&](int b, int s, int h, int blk) -> unsigned char& { return ks[(((size_t)b * krows + s) * HK + h) * 4 + blk]; };
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < H; ++h)
                for (int blk = 0; blk < 4; ++blk) qs_at(b, s, h, blk) = (unsigned char)(126 + rnd() % 3);
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < SK; ++s)
            for (int h = 0; h < HK; ++h)
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

    unsigned char args[720];   // 656-byte base ABI + KV-split fields at 0x270/0x278/0x280
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
    put32(args, 0x0B0, GQA);                   // gqa
    put32(args, 0x0C0, HK * D);                // K seq stride
    put32(args, 0x0D0, D);                     // K head stride
    put32(args, 0x0E0, (uint32_t)((size_t)SK * HK * D));
    put32(args, 0x0F0, 5);                     // opt
    put32(args, 0x110, SK);
    put32(args, 0x120, D); put32(args, 0x130, D);
    put32(args, 0x140, H);
    put32(args, 0x150, HK * D); put32(args, 0x160, D); put32(args, 0x170, (uint32_t)((size_t)SK * HK * D));
    put32(args, 0x180, H * D * 2); put32(args, 0x190, D * 2); put32(args, 0x1A0, (uint32_t)((size_t)S * H * D * 2));
    put64(args, 0x200, dqs); put64(args, 0x210, dks); put64(args, 0x220, dvs);
    put32(args, 0x230, (uint32_t)((size_t)qrows * H * 4)); put32(args, 0x240, 4);
    put32(args, 0x250, (uint32_t)((size_t)krows * HK * 4)); put32(args, 0x260, 4);

    hipModule_t mod; hipFunction_t fn;
    CK(hipModuleLoad(&mod, co));
    CK(hipModuleGetFunction(&fn, mod, "_ZN5aiter28fmha_fwd_hd128_mxfp8_gfx950E"));
    size_t asz = 656;          // vendor / _vs128 / _t128 use the 656-byte ABI; KV-split uses 720
    void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    // Per-kernel ABI fields that depend on the tiling: output pointer and the Q tile stride (offset 0x080).
    auto set_launch = [&](void* outp, int stride_rows) {
        put64(args, 0x000, outp);
        put32(args, 0x080, (uint32_t)((size_t)stride_rows * H * D));
    };
    // KV-split kernarg extension (docs/kvsplit-design.md section 3, agreed with kernel-opt): 0x270 ptr Ohat,
    // 0x278 ptr lse, 0x280 u32 splits, 0x284 u32 rows_total (= B*S*H, the per-split stride); segment 720 bytes.
    auto set_split_args = [&](void* ohat, void* lse) {
        put64(args, 0x270, ohat);
        put64(args, 0x278, lse);
        put32(args, 0x280, (uint32_t)splits);
        put32(args, 0x284, (uint32_t)rows_total);
        asz = 720;
    };
    asz = 656; set_launch(dout, vendor_stride_rows);
    CK(hipModuleLaunchKernel(fn, gvx, gy, gz, vendor_block, 1, 1, 0, nullptr, nullptr, extra));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dout, no * 2, hipMemcpyDeviceToHost));
    const int iters = envi("REF_ITERS", 0);
    const double flops = 4.0 * B * H * (double)S * SK * D;
    auto time_kernel = [&](hipFunction_t f, const char* label, void* outp, int rows, int gx, unsigned block) {
        if (iters <= 0) return;
        set_launch(outp, rows);
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        // Warm-up launches: on a shared node the GPU sits in its idle clock state, so a cold 3-launch
        // warm-up under-states the first kernel timed. REF_WARMUP (default 3 = historical) lets the
        // measurement protocol pump enough work through the GPU to reach the boost clock first.
        const int warm = envi("REF_WARMUP", 3);
        for (int w = 0; w < warm; ++w) CK(hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, nullptr, nullptr, extra));
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0));
        for (int w = 0; w < iters; ++w) CK(hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, nullptr, nullptr, extra));
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
        const double per = ms / iters;
        std::printf("timing %-8s: %.4f ms/iter, %.2f TFLOP/s (%d iters, %d rows/block, %d blocks)\n",
                    label, per, flops / (per * 1e-3) / 1e12, iters, rows, gx * gy * gz);
    };
    time_kernel(fn, "vendor", dout, vendor_rows, gvx, vendor_block);

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
                            acc += qd[kk] * fp8_e4m3(k[(((size_t)b * SK + t) * HK + h / GQA) * D + kk]) *
                                   std::ldexp(1.0, (int)ks_at(b, t, h / GQA, kk / 32) - 127);
                        sc[t] = acc * cpu_scale;
                        mx = std::max(mx, sc[t]);
                    }
                    double den = 0.0;
                    for (int t = 0; t < SK; ++t) { sc[t] = std::exp(sc[t] - mx); den += sc[t]; }
                    for (int d = 0; d < D; ++d) {
                        double acc = 0.0;
                        for (int t = 0; t < SK; ++t) acc += sc[t] / den * fp8_e4m3(v[(((size_t)b * SK + t) * HK + h / GQA) * D + d]);
                        acc *= vdescale;
                        const double got = bf2f(o[(((size_t)b * S + i) * H + h) * D + d]);
                        const double err = std::fabs(got - acc);
                        if (err != err) worst = std::numeric_limits<double>::infinity(); else worst = std::max(worst, err); sum_abs += err; max_ref = std::max(max_ref, std::fabs(acc)); ++cnt;
                        if (!(err <= 0.05 + 0.05 * std::fabs(acc))) { /* NaN counts as bad */ if (bad < 5) std::fprintf(stderr, "%s O[%d][%d][%d][%d] want %.5f got %.5f\n", label, b, h, i, d, acc, got); ++bad; }
                    }
                }
        std::printf("%s: checked=%zu bad=%zu worst_err=%.4e mean_err=%.4e max|ref|=%.4f\n", label, cnt, bad, worst, sum_abs / cnt, max_ref);
        return bad;
    };
    size_t bad = check(out, "vendor kernel vs CPU");

    // Optional second kernel (ours) on the same inputs: either the single-kernel path or KV-split main+combine.
    size_t bad2 = 0;
    if (std::getenv("REF_CO2") || std::getenv("REF_CO2_SMALL") || std::getenv("REF_CO2_SPLIT")) {
        void* dout2;
        CK(hipMalloc(&dout2, no * 2));
        CK(hipMemset(dout2, 0, no * 2));
        std::vector<uint16_t> out2(no, 0);
        if (use_split) {
            if (!co_comb) { std::printf("[split] missing REF_CO2_COMBINE\n"); return 1; }
            // Partials: allocated ONCE, outside every timed window (docs/kvsplit-design.md section 3).
            void *ohat, *lse;
            CK(hipMalloc(&ohat, (size_t)ohat_alloc_bytes));
            CK(hipMalloc(&lse, (size_t)lse_bytes));
            CK(hipMemset(ohat, 0, (size_t)ohat_alloc_bytes));
            CK(hipMemset(lse, 0, (size_t)lse_bytes));
            hipModule_t mod2; hipFunction_t fn2;
            CK(hipModuleLoad(&mod2, use_co));
            CK(hipModuleGetFunction(&fn2, mod2, use_kn));
            hipModule_t modc; hipFunction_t fnc;
            CK(hipModuleLoad(&modc, co_comb));
            CK(hipModuleGetFunction(&fnc, modc, kname_comb));
            unsigned char cargs[40]; std::memset(cargs, 0, sizeof(cargs));
            put64(cargs, 0x00, ohat); put64(cargs, 0x08, lse); put64(cargs, 0x10, dout2);
            put32(cargs, 0x18, (uint32_t)rows_total); put32(cargs, 0x1C, (uint32_t)D); put32(cargs, 0x20, (uint32_t)splits);
            size_t casz = sizeof(cargs);
            void* cextra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, cargs, HIP_LAUNCH_PARAM_BUFFER_SIZE, &casz, HIP_LAUNCH_PARAM_END};
            const unsigned cthreads = (unsigned)envi("REF_COMBINE_THREADS", 256);
            // `..._combine4` / `..._combine4h` kernels handle 4 output elements per thread
            const double comb_ept = std::string(kname_comb).find("combine4") != std::string::npos ? 4.0 : 1.0;
            long cgrid = (long)std::ceil((double)rows_total * D / ((double)cthreads * comb_ept));
            if (const char* cg = std::getenv("REF_COMBINE_GRID")) cgrid = std::atol(cg);
            if (cgrid < 1) cgrid = 1;
            auto launch_split = [&] {
                asz = 720; set_split_args(ohat, lse); set_launch(dout2, s_rows_pb);
                CK(hipModuleLaunchKernel(fn2, gox, gy, goz, ours_block, 1, 1, 0, nullptr, nullptr, extra));
            };
            auto launch_comb = [&] {
                CK(hipModuleLaunchKernel(fnc, (unsigned)cgrid, 1, 1, cthreads, 1, 1, 0, nullptr, nullptr, cextra));
            };
            const int warm = envi("REF_WARMUP", 3);
            for (int w = 0; w < warm; ++w) { launch_split(); launch_comb(); }
            CK(hipDeviceSynchronize());
            auto timed = [&](auto&& launch, int n) -> double {
                hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
                launch(); CK(hipDeviceSynchronize());
                CK(hipEventRecord(e0));
                for (int i = 0; i < n; ++i) launch();
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
                CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
                return (double)ms / (double)n;
            };
            // one real main+combine pass -> combined output for the CPU/vendor/ULP checks
            launch_split(); launch_comb(); CK(hipDeviceSynchronize());
            CK(hipMemcpy(out2.data(), dout2, no * 2, hipMemcpyDeviceToHost));
            bad2 = check(out2, "our kernel   vs CPU");
            if (iters > 0) {
                const double ms_main = timed([&]{ launch_split(); }, iters);
                launch_split(); launch_comb(); CK(hipDeviceSynchronize());
                const double ms_comb = timed([&]{ launch_comb(); }, iters);
                const double ms_total = timed([&]{ launch_split(); launch_comb(); }, iters);
                std::printf("timing ours main: %.4f ms/iter, %.2f TFLOP/s (phase-1 only, %d iters, split=%d)\n",
                            ms_main, flops / (ms_main * 1e-3) / 1e12, iters, splits);
                std::printf("timing ours combine: %.4f ms/iter (%d iters, %ld blocks x %u threads)\n",
                            ms_comb, iters, cgrid, cthreads);
                std::printf("timing ours total: %.4f ms/iter, %.2f TFLOP/s (%d iters, split=%d main+combine)\n",
                            ms_total, flops / (ms_total * 1e-3) / 1e12, iters, splits);
            }
            double wd = 0.0, md = 0.0;
            for (size_t i = 0; i < no; ++i) { const double dd = std::fabs((double)bf2f(out[i]) - (double)bf2f(out2[i])); wd = std::max(wd, dd); md += dd; }
            std::printf("our kernel vs vendor kernel (all %zu outputs): worst_diff=%.4e mean_diff=%.4e\n", no, wd, md / no);
            // Split-vs-_vs128 ULP histogram (REF_CO2 is the _vs128 kernel in auto_split mode).
            if (envi("REF_SPLIT_ULP", 1) != 0) {
                if (const char* co_vs = std::getenv("REF_CO2")) {
                    void* dout_ref; CK(hipMalloc(&dout_ref, no * 2)); CK(hipMemset(dout_ref, 0, no * 2));
                    hipModule_t modv; hipFunction_t fnv;
                    CK(hipModuleLoad(&modv, co_vs));
                    CK(hipModuleGetFunction(&fnv, modv, kname_env ? kname_env : "fmha_fwd_hd128_mxfp8_vs128"));
                    asz = 656; set_launch(dout_ref, 256);
                    CK(hipModuleLaunchKernel(fnv, gvx, gy, gz, 512, 1, 1, 0, nullptr, nullptr, extra));
                    CK(hipDeviceSynchronize());
                    std::vector<uint16_t> ref(no, 0);
                    CK(hipMemcpy(ref.data(), dout_ref, no * 2, hipMemcpyDeviceToHost));
                    // Gate in kernel-opt's suggested form: max_abs <= 1 bf16 ULP of max|reference| plus
                    // mean_abs <= 1e-4. Word-level ULP counts are informational only: on near-zero outputs
                    // a 1-code difference is tens of thousands of word ULPs and is meaningless as a gate.
                    double mean_abs = 0.0, max_abs = 0.0, max_ref = 0.0;
                    long gt1 = 0, nz = 0;
                    for (size_t i = 0; i < no; ++i) {
                        const double rr = std::fabs((double)bf2f(ref[i]));
                        max_ref = std::max(max_ref, rr);
                        const double dd = std::fabs((double)bf2f(out2[i]) - (double)bf2f(ref[i]));
                        mean_abs += dd; max_abs = std::max(max_abs, dd);
                        const int u = std::abs((int)out2[i] - (int)ref[i]);
                        if (u > 1) ++gt1; if (u) ++nz;
                    }
                    mean_abs /= (double)no;
                    int rexp = 0;
                    std::frexp(max_ref, &rexp);
                    const double ulp_ref = std::ldexp(1.0, rexp - 8);   // bf16: 7-bit mantissa
                    const double mean_max = std::getenv("SPLIT_MEAN_ABS_MAX") ? std::atof(std::getenv("SPLIT_MEAN_ABS_MAX")) : 1e-4;
                    const double abs_tol = (std::getenv("SPLIT_ABS_TOL") ? std::atof(std::getenv("SPLIT_ABS_TOL")) : 1.0) * ulp_ref;
                    std::printf("SPLIT_ULP split=%d: max_abs=%.3e ulp_ref=%.3e (%.2f ULP) mean_abs=%.3e frac_diff_words=%.5f frac_gt1_word_ulp=%.5f max|ref|=%.3f\n",
                                splits, max_abs, ulp_ref, max_abs / ulp_ref, mean_abs, (double)nz / (double)no,
                                (double)gt1 / (double)no, max_ref);
                    std::printf("%s\n", (max_abs <= abs_tol && mean_abs <= mean_max) ? "SPLIT_ULP_OK" : "SPLIT_ULP_FAIL");
                    CK(hipFree(dout_ref));
                }
            }
            if (const char* dump2 = std::getenv("REF_DUMP2")) { FILE* fo = std::fopen(dump2, "wb"); std::fwrite(out2.data(), 2, no, fo); std::fclose(fo); }
            if (const char* dumps = std::getenv("REF_SPLIT_DUMP")) { FILE* fo = std::fopen(dumps, "wb"); std::fwrite(out2.data(), 2, no, fo); std::fclose(fo); }
            CK(hipFree(ohat)); CK(hipFree(lse));
        } else {
            // Single-kernel path (unchanged behaviour).
            asz = 656; set_launch(dout2, ours_stride_rows);
            hipModule_t mod2; hipFunction_t fn2;
            CK(hipModuleLoad(&mod2, use_co));
            CK(hipModuleGetFunction(&fn2, mod2, use_kn));
            CK(hipModuleLaunchKernel(fn2, gox, gy, gz, ours_block, 1, 1, 0, nullptr, nullptr, extra));
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(out2.data(), dout2, no * 2, hipMemcpyDeviceToHost));
            bad2 = check(out2, "our kernel   vs CPU");
            time_kernel(fn2, "ours", dout2, ours_rows, gox, ours_block);
            double wd = 0.0, md = 0.0;
            for (size_t i = 0; i < no; ++i) { const double dd = std::fabs((double)bf2f(out[i]) - (double)bf2f(out2[i])); wd = std::max(wd, dd); md += dd; }
            std::printf("our kernel vs vendor kernel (all %zu outputs): worst_diff=%.4e mean_diff=%.4e\n", no, wd, md / no);
            if (const char* dump2 = std::getenv("REF_DUMP2")) { FILE* fo = std::fopen(dump2, "wb"); std::fwrite(out2.data(), 2, no, fo); std::fclose(fo); }
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
