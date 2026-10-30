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
    int splits = envi("REF_SPLIT", 0);
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
    const double part_elem_bytes = (double)envi("REF_PARTIAL_BYTES", 2);   // fp16 partials (default of the split kernel; `nop16` builds use 4)
    const double comb_threads_d = (double)envi("REF_COMBINE_THREADS", 256);

    // Dispatch guard for the KV-split candidate (docs/kvsplit-results.md, docs/gen4-tiling-design.md section 11): use it only when
    // N256 <= 128 and N256*s <= 256 (one full block wave), every split owns a tile, and the modelled T_kvsplit(s) < 0.95 * T_current;
    // otherwise fall back to the certified _t128/_vs128 rule. The model is the one fitted to the measurements (kernel-opt, B1H8S2048 and 1 8 1024):
    //   T_t128    = F / (5.9e12 * min(256, 2 N256)) + 9.4 us             (4-wave blocks: compute + the fixed per-launch cost; 9.4 us is the floor measured at SK = 1)
    //   T_vs128   = F / (5.9e12 * min(256, N256))                        (filling grids)
    //   T_kvsplit = F / (5.9e12 * min(256, N256 s)) + 7 us               (phase 1: compute + fixed prologue/epilogue)
    //             + partial bytes / 4.0 TB/s                              (epilogue write, not hidden: the normalised partial exists only after
    //                                                                     the last tile and all blocks finish together)
    //             + max(3 us, (partial + out bytes) / 4.2 TB/s) + 1.5 us  (combine (kernel floor 3 us) + launch gap)
    // The split launch counts BOTH kernels in the timing window.
    struct SplitEval { bool guard; double t_cur; double t_kv; int tps; bool tiles_ok; };
    auto eval_split = [&](int s) {
        SplitEval r{};
        const double F = 4.0 * B * H * (double)S * SK * D;
        r.t_cur = (blocks256 <= 128)
            ? F / (5.9e12 * std::min<double>(256.0, 2.0 * (double)blocks256)) + 9.4e-6
            : F / (5.9e12 * std::min<double>(256.0, (double)blocks256));
        const double part_b = (double)s * rows_total * D * part_elem_bytes + (double)s * rows_total * 4;
        const double comb_b = part_b + (double)rows_total * D * 2;
        r.t_kv = F / (5.9e12 * std::min<double>(256.0, (double)blocks256 * s)) + 7.0e-6 + part_b / 4.0e12 + std::max(3.0e-6, comb_b / 4.2e12) + 1.5e-6;
        const int ntiles = (SK + 127) / 128;
        r.tps = (ntiles + s - 1) / s;
        r.tiles_ok = (s >= 1) && (r.tps >= 1) && ((s - 1) * r.tps < ntiles);
        r.guard = (blocks256 <= 128) && (blocks256 * s <= 256) && r.tiles_ok && (r.t_kv < 0.95 * r.t_cur);
        return r;
    };
    bool use_split = false; const char* split_why = "split not requested";
    if (co_split && co_comb && splits <= 1 && envi("REF_NO_AUTO_SPLIT", 0) == 0) {
        // default auto dispatch: best guard-passing split among {2, 4}
        int best = 0; double best_t = 1e9;
        for (int s : {2, 4}) {
            const SplitEval r = eval_split(s);
            std::printf("[split] auto candidate s=%d: blocks256=%ld ntiles=%d tps=%d tiles_ok=%d model T_current=%.2f us T_kvsplit=%.2f us guard=%d\n",
                        s, blocks256, (SK + 127) / 128, r.tps, (int)r.tiles_ok, r.t_cur * 1e6, r.t_kv * 1e6, (int)r.guard);
            if (r.guard && r.t_kv < best_t) { best = s; best_t = r.t_kv; }
        }
        if (best > 0) { splits = best; use_split = true; split_why = "auto dispatch: guard N256<=128 && N256*s<=256 && tiles && T_kv < 0.95 T_current"; }
        else split_why = "auto dispatch: no split passes the guard (fallback to t128/_vs128)";
    } else if (splits > 1 && co_split) {
        const SplitEval r = eval_split(splits);
        if (force_split) { use_split = true; split_why = "REF_SPLIT_FORCE"; }
        else if (r.guard) { use_split = true; split_why = "REF_SPLIT + guard: N256<=128 && N256*s<=256 && model win"; }
        else { split_why = "split guard rejected (fallback)"; }
        std::printf("[split] s=%d %s: blocks256=%ld ntiles=%d tps=%d; model T_current=%.2f us T_kvsplit=%.2f us; %s\n",
                    splits, use_split ? "CHOSEN" : "not chosen", blocks256, (SK + 127) / 128, r.tps, r.t_cur * 1e6, r.t_kv * 1e6, split_why);
    }
    const long s_blocks = (long)s_gx * H * B * (splits > 1 ? splits : 1);
    const double ohat_bytes = (double)splits * rows_total * D * part_elem_bytes;
    const double lse_bytes = (double)splits * rows_total * 4;    int rows_pb; const char* why;
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
        std::printf("[split] s=%d: phase1 Ohat=%.1f MB (fp32) + lse=%.2f MB, grid=(%d,%d,%d) blocks=%ld block=512; "
                    "combine grid=%ld block=%u; guard=%s\n",
                    splits, ohat_bytes / 1e6, lse_bytes / 1e6, gox, gy, goz, s_blocks,
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
            CK(hipMalloc(&ohat, (size_t)ohat_bytes));
            CK(hipMalloc(&lse, (size_t)lse_bytes));
            CK(hipMemset(ohat, 0, (size_t)ohat_bytes));
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
                    double mean_abs = 0.0, max_abs = 0.0;
                    int max_ulp = 0; long gt1 = 0, nz = 0;
                    std::vector<int> all_ulps; all_ulps.reserve((size_t)no);
                    for (size_t i = 0; i < no; ++i) {
                        const double dd = std::fabs((double)bf2f(out2[i]) - (double)bf2f(ref[i]));
                        mean_abs += dd; max_abs = std::max(max_abs, dd);
                        const int u = std::abs((int)out2[i] - (int)ref[i]);
                        all_ulps.push_back(u); max_ulp = std::max(max_ulp, u);
                        if (u > 1) ++gt1; if (u) ++nz;
                    }
                    mean_abs /= (double)no;
                    std::sort(all_ulps.begin(), all_ulps.end());
                    const long p99 = all_ulps.empty() ? 0 : (long)all_ulps[(size_t)(0.99 * (double)(all_ulps.size() - 1))];
                    const int ulp_max = envi("SPLIT_ULP_MAX", 2);
                    const double mean_max = std::getenv("SPLIT_MEAN_ABS_MAX") ? std::atof(std::getenv("SPLIT_MEAN_ABS_MAX")) : 1e-4;
                    std::printf("SPLIT_ULP split=%d: max_ulp=%d p99_ulp=%ld mean_abs=%.3e max_abs=%.3e frac_gt1ulp=%.5f frac_nonzero=%.5f\n",
                                splits, max_ulp, p99, mean_abs, max_abs, (double)gt1 / (double)no, (double)nz / (double)no);
                    std::printf("%s\n", (max_ulp <= ulp_max && mean_abs <= mean_max) ? "SPLIT_ULP_OK" : "SPLIT_ULP_FAIL");
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
