// Same-process host for GEMM code objects: aiter's prebuilt gfx950 ASM GEMMs (bf16gemm, i8gemm, fp8gemm_blockscale, f4gemm), a standalone
// Opus instance (opus_a16w16), and ours (same data, same checks) - for the op-coverage batch 1 comparisons (docs/op-coverage-design.md §7.2).
// The kernarg layouts and data layouts follow aiter's wrappers (csrc/py_itfs_cu/asm_gemm_*.cu, aiter/ops/shuffle.py); every field of the
// padded structs sits on a 16-byte boundary.
//
//   vendor_gemm_host <family> <co> <kernel> M N K [tileM tileN]
//   family: asm_bf16 | asm_bf16_pre | opus_a16w16 | asm_i8 | asm_fp8bs | asm_f4 | ours_bf16 | ours_i8 | ours_fp8bs | ours_f4
//   env: ITERS (100), WARMUP (20), CHECK_ROWS (16, rows spread over M checked against the CPU reference), SPLITK (1), SEED (1)
//
// The "ours_*" families take our kernels' ABI: ptr A, B, C, [scales], then M, N, K as int32 (8-byte pointers, packed), grid = (N/tileN, M/tileM),
// block 256 unless BLOCK is set; operands in the plain (unshuffled) layout unless PRESHUF=1.
#include <hip/hip_runtime.h>
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

// ---- number formats ----
static float bf16_to_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float e4m3_to_f(uint8_t b) {  // OCP e4m3fn
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}
static float e2m1_to_f(uint8_t n) { static const float t[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6}; return (n & 8) ? -t[n & 7] : t[n & 7]; }
static float e8m0_to_f(uint8_t e) { return std::ldexp(1.0f, (int)e - 127); }

// aiter shuffle_weight(layout = (16, 16)) on a [N, Kb] array of `es`-byte elements (Kb = elements per row): Kp = 16 / es, BK = 32 elements
static std::vector<uint8_t> shuffle_b(const std::vector<uint8_t>& src, int N, int Kb, int es) {
    const int Kp = 16 / es, BK = 32;
    std::vector<uint8_t> dst(src.size());
    for (int n = 0; n < N; n++)
        for (int k = 0; k < Kb; k++) {
            int n0 = n / 16, ni = n % 16, k0 = k / BK, kk = (k % BK) / Kp, kj = k % Kp;
            size_t d = (size_t)n0 * 16 * Kb + (size_t)k0 * 16 * BK + (size_t)kk * 16 * Kp + (size_t)ni * Kp + kj;
            memcpy(&dst[d * es], &src[((size_t)n * Kb + k) * es], es);
        }
    return dst;
}
// aiter e8m0_shuffle / shuffle_scale on [rows, groups] bytes: rows padded to 256, columns to 8 (pad 0x7F)
static std::vector<uint8_t> shuffle_scale(const std::vector<uint8_t>& s, int rows, int groups, int& sm, int& sn) {
    sm = (rows + 255) / 256 * 256;
    sn = (groups + 7) / 8 * 8;
    std::vector<uint8_t> d((size_t)sm * sn, 0x7F);
    for (int r = 0; r < sm; r++)
        for (int g = 0; g < sn; g++) {
            uint8_t v = (r < rows && g < groups) ? s[(size_t)r * groups + g] : 0x7F;
            size_t o = (size_t)(r / 32) * 32 * sn + (size_t)(g / 8) * 256 + (g % 4) * 64 + (r % 16) * 4 + ((g % 8) / 4) * 2 + (r % 32) / 16;
            d[o] = v;
        }
    return d;
}

struct Arg { std::vector<uint8_t> b; void put(size_t off, const void* p, size_t n) { if (b.size() < off + n) b.resize(off + n, 0); memcpy(&b[off], p, n); }
    void ptr(size_t off, void* p) { put(off, &p, 8); } void u32(size_t off, uint32_t v) { put(off, &v, 4); } void f32(size_t off, float v) { put(off, &v, 4); } };

template <class T> static T* dev(const std::vector<T>& h) { T* d; HC(hipMalloc(&d, h.size() * sizeof(T) + 256)); HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s family co kernel M N K [tileM tileN]\n", argv[0]); return 1; }
    std::string fam = argv[1];
    const char* co = argv[2];
    const char* kn = argv[3];
    const int M = atoi(argv[4]), N = atoi(argv[5]), K = atoi(argv[6]);
    const int tM = argc > 7 ? atoi(argv[7]) : 256, tN = argc > 8 ? atoi(argv[8]) : 256;
    const int iters = envi("ITERS", 100), warm = envi("WARMUP", 20), crow = envi("CHECK_ROWS", 16), splitk = envi("SPLITK", 1);
    std::mt19937 rng(envi("SEED", 1));
    std::uniform_real_distribution<float> U(-1, 1);
    const bool ours = fam.rfind("ours_", 0) == 0;
    const bool pre = ours && fam.rfind("ours_v4_", 0) != 0 ? envi("PRESHUF", 0) != 0 : true;

    hipModule_t mod; hipFunction_t fn;
    HC(hipModuleLoad(&mod, co));
    HC(hipModuleGetFunction(&fn, mod, kn));

    // ---- operands (host values kept as float for the reference) ----
    std::vector<float> af((size_t)M * K), bf((size_t)N * K), sa_row, sb_col, sa_blk, sb_blk;
    std::vector<uint8_t> ha, hb;  // raw device bytes (A row-major [M, K], B [N, K] before any shuffle)
    std::vector<uint8_t> hb_w4perm;  // bf16w4: the K-permuted B
    std::vector<uint8_t> sa8, sb8;  // e8m0 scales (f4)
    int es = 2;
    std::string kind = fam.rfind("ours_v4_", 0) == 0 ? fam.substr(8) : fam.substr(fam.find('_') + 1);
    if (kind == "rowcol_fp8") {} // fp8 per-token data

    if (kind == "bf16" || kind == "bf16_pre" || kind == "a16w16") {
        es = 2;
        ha.resize((size_t)M * K * 2); hb.resize((size_t)N * K * 2);
        for (size_t i = 0; i < af.size(); i++) { uint16_t h = f_to_bf16(U(rng)); memcpy(&ha[2 * i], &h, 2); af[i] = bf16_to_f(h); }
        for (size_t i = 0; i < bf.size(); i++) { uint16_t h = f_to_bf16(U(rng)); memcpy(&hb[2 * i], &h, 2); bf[i] = bf16_to_f(h); }
    } else if (kind == "i8" || kind == "rowcol_fp8") {
        es = 1;
        ha.resize((size_t)M * K); hb.resize((size_t)N * K);
        if (kind == "i8") {
            for (size_t i = 0; i < af.size(); i++) { int8_t v = (int8_t)(rng() % 255 - 127); ha[i] = (uint8_t)v; af[i] = v; }
            for (size_t i = 0; i < bf.size(); i++) { int8_t v = (int8_t)(rng() % 255 - 127); hb[i] = (uint8_t)v; bf[i] = v; }
        } else {  // fp8 e4m3 codes (|x| in [2^-4, 2^4])
            auto code = [&]() { uint8_t e = 3 + rng() % 9, m = rng() % 8, s = rng() % 2; return (uint8_t)((s << 7) | (e << 3) | m); };
            for (size_t i = 0; i < af.size(); i++) { ha[i] = code(); af[i] = e4m3_to_f(ha[i]); }
            for (size_t i = 0; i < bf.size(); i++) { hb[i] = code(); bf[i] = e4m3_to_f(hb[i]); }
        }
        sa_row.resize(M); sb_col.resize(N);
        for (auto& s : sa_row) s = 1e-3f * (1 + (rng() % 100) / 100.0f);
        for (auto& s : sb_col) s = 1e-3f * (1 + (rng() % 100) / 100.0f);
        // PERTENSOR=1 (gemm_a8w8_pertensor_fp8): one scale per tensor (the device gets 1-element arrays; the reference sees them broadcast)
        if (envi("PERTENSOR", 0)) { for (auto& s : sa_row) s = sa_row[0]; for (auto& s : sb_col) s = sb_col[0]; }
    } else if (kind == "fp8bs") {
        es = 1;
        ha.resize((size_t)M * K); hb.resize((size_t)N * K);
        auto code = [&]() { uint8_t e = 3 + rng() % 9, m = rng() % 8, s = rng() % 2; return (uint8_t)((s << 7) | (e << 3) | m); };
        for (size_t i = 0; i < af.size(); i++) { ha[i] = code(); af[i] = e4m3_to_f(ha[i]); }
        for (size_t i = 0; i < bf.size(); i++) { hb[i] = code(); bf[i] = e4m3_to_f(hb[i]); }
        sa_blk.resize((size_t)M * (K / 128)); sb_blk.resize((size_t)(N / 128) * (K / 128));
        for (auto& s : sa_blk) s = std::ldexp(1.0f, -(int)(rng() % 4)) * (1 + (rng() % 8) / 8.0f);
        for (auto& s : sb_blk) s = std::ldexp(1.0f, -(int)(rng() % 4)) * (1 + (rng() % 8) / 8.0f);
    } else if (kind == "f4") {
        es = 1;  // bytes; K/2 per row
        ha.resize((size_t)M * K / 2); hb.resize((size_t)N * K / 2);
        for (size_t i = 0; i < ha.size(); i++) { ha[i] = rng() & 0xff; af[2 * i] = e2m1_to_f(ha[i] & 15); af[2 * i + 1] = e2m1_to_f(ha[i] >> 4); }
        for (size_t i = 0; i < hb.size(); i++) { hb[i] = rng() & 0xff; bf[2 * i] = e2m1_to_f(hb[i] & 15); bf[2 * i + 1] = e2m1_to_f(hb[i] >> 4); }
        sa8.resize((size_t)M * (K / 32)); sb8.resize((size_t)N * (K / 32));
        for (auto& s : sa8) s = 124 + rng() % 6;
        for (auto& s : sb8) s = 124 + rng() % 6;
    } else if (kind == "f8f4") {
        // A8W4: A fp8 e4m3 codes [M][K], B fp4 [N][K/2] bytes, e8m0 1 x 32 scales on both (shuffled like f4)
        es = 1;
        ha.resize((size_t)M * K); hb.resize((size_t)N * K / 2);
        auto code = [&]() { uint8_t e = 3 + rng() % 9, m = rng() % 8, s = rng() % 2; return (uint8_t)((s << 7) | (e << 3) | m); };
        for (size_t i = 0; i < ha.size(); i++) { ha[i] = code(); af[i] = e4m3_to_f(ha[i]); }
        for (size_t i = 0; i < hb.size(); i++) { hb[i] = rng() & 0xff; bf[2 * i] = e2m1_to_f(hb[i] & 15); bf[2 * i + 1] = e2m1_to_f(hb[i] >> 4); }
        sa8.resize((size_t)M * (K / 32)); sb8.resize((size_t)N * (K / 32));
        for (auto& s : sa8) s = 124 + rng() % 6;
        for (auto& s : sb8) s = 124 + rng() % 6;
    } else if (kind == "bf16w4") {
        // A16W4: A bf16 [M][K] (exact), B fp4 [N][K/2] bytes with e8m0 1 x 32 scales [N][K/32] (unshuffled); A "scales" all 1 (127, unused by
        // the kernel). The device B gets the K permutation of gemm_a16w4 inside every 256-K tile (physical fp4 position p -> logical K),
        // applied below before the standard preshuffle.
        es = 2;
        ha.resize((size_t)M * K * 2); hb.resize((size_t)N * K / 2);
        for (size_t i = 0; i < (size_t)M * K; i++) { float v = (float)((int)(rng() % 17) - 8) / 8.0f; uint16_t bb = f_to_bf16(v); memcpy(&ha[2 * i], &bb, 2); af[i] = bf16_to_f(bb); }
        for (size_t i = 0; i < hb.size(); i++) { hb[i] = rng() & 0xff; bf[2 * i] = e2m1_to_f(hb[i] & 15); bf[2 * i + 1] = e2m1_to_f(hb[i] >> 4); }
        sa8.assign((size_t)M * (K / 32), 127); sb8.resize((size_t)N * (K / 32));
        for (auto& s : sb8) s = 124 + rng() % 6;
        // W4DBG=1: unit B scales + A = 1 (tests the B path); W4DBG=2: unit B scales + B = 1 (code 2) (tests the A path)
        const int dbg = envi("W4DBG", 0);
        if (dbg && dbg != 5) for (auto& s : sb8) s = 127;  // (W4DBG=4: unit scales only; W4DBG=5: A = 1, scales random)
        if (dbg == 5) for (size_t i = 0; i < (size_t)M * K; i++) { uint16_t bb = f_to_bf16(1.0f); memcpy(&ha[2 * i], &bb, 2); af[i] = 1.0f; }
        if (dbg == 1) for (size_t i = 0; i < (size_t)M * K; i++) { uint16_t bb = f_to_bf16(1.0f); memcpy(&ha[2 * i], &bb, 2); af[i] = 1.0f; }
        if (dbg == 2) for (size_t i = 0; i < hb.size(); i++) { hb[i] = 0x22; bf[2 * i] = 1.0f; bf[2 * i + 1] = 1.0f; }
        // permute: phys[p] = logical K 256 T + L(p), L(p) = 128 (p / 128) + 16 ((p / 64) % 2) + 8 ((p / 32) % 2) + 32 ((p / 8) % 4) + p % 8
        // (measured with W4DBG=3: the A K index the MFMA pairs with physical fp4 position p of the tile)
        std::vector<uint8_t> hp(hb.size(), 0);
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++) {
                int T = k / 256, p = k % 256;
                int L = 128 * (p / 128) + 16 * ((p / 64) % 2) + 8 * ((p / 32) % 2) + 32 * ((p / 8) % 4) + p % 8;
                size_t src = (size_t)n * K + 256 * T + L;
                uint8_t code = (hb[src / 2] >> (4 * (src & 1))) & 15;
                size_t dst = (size_t)n * K + k;
                hp[dst / 2] |= code << (4 * (dst & 1));
            }
        // W4DBG=3: probe the pairing - A[m][k] = k (exact), B device row n = only physical fp4 position P = KK (env) set to 1.0
        if (dbg == 3) {
            for (size_t i = 0; i < (size_t)M * K; i++) { uint16_t bb = f_to_bf16((float)(i % K)); memcpy(&ha[2 * i], &bb, 2); }
            std::fill(hp.begin(), hp.end(), 0);
            const int P = envi("KK", 0);
            for (int n = 0; n < N; n++) { size_t dst = (size_t)n * K + P; hp[dst / 2] |= 2 << (4 * (dst & 1)); }
        }
        // W4DBG=6: probe the scale bytes - A[m][k] = k, B = 1.0 everywhere, scale byte GRP of every row = 2 (others 1)
        if (dbg == 6) {
            for (size_t i = 0; i < (size_t)M * K; i++) { uint16_t bb = f_to_bf16((float)(i % K)); memcpy(&ha[2 * i], &bb, 2); }
            std::fill(hp.begin(), hp.end(), 0x22);
            for (int n = 0; n < N; n++) for (int g2 = 0; g2 < K / 32; g2++) sb8[(size_t)n * (K / 32) + g2] = g2 == envi("GRP", 0) ? 128 : 127;
        }
        hb_w4perm = hp;
    } else { fprintf(stderr, "unknown family %s\n", fam.c_str()); return 1; }

    // ---- device buffers ----
    const bool b4 = kind == "f4" || kind == "f8f4" || kind == "bf16w4";
    if (kind == "bf16w4") hb.swap(hb_w4perm);  // the device gets the permuted B (the reference uses bf)
    const int Kb = b4 ? K / 2 : K;  // B row length in device elements
    std::vector<uint8_t> hb_dev = pre && kind != "a16w16" && fam != "asm_bf16" ? shuffle_b(hb, N, b4 ? Kb : K, b4 ? 1 : es) : hb;
    void* dA = dev(ha);
    void* dB = dev(hb_dev);
    const int out_rows = (M + 31) / 32 * 32;
    void* dC; HC(hipMalloc(&dC, (size_t)out_rows * N * 2 + 256)); HC(hipMemset(dC, 0, (size_t)out_rows * N * 2));
    void *dSA = nullptr, *dSB = nullptr, *dBias = nullptr, *dSem = nullptr;
    Arg a;
    int gx = (N + tN - 1) / tN, gy = (M + tM - 1) / tM, gz = 1, block = envi("BLOCK", 256);
    if (fam == "asm_bf16" || fam == "asm_bf16_pre") {
        HC(hipMalloc(&dSem, 4096)); HC(hipMemset(dSem, 0, 4096));
        a.ptr(0, dC); a.ptr(16, nullptr); a.ptr(32, dA); a.ptr(48, dB); a.f32(64, 1.0f); a.f32(80, 0.0f);
        a.u32(96, N * 2); a.u32(128, N * 2); a.u32(160, K * 2); a.u32(192, K * 2); a.u32(224, M); a.u32(240, N); a.u32(256, K);
        a.u32(272, splitk); a.u32(288, 1); a.ptr(304, nullptr); a.u32(320, 0); a.ptr(336, splitk > 1 ? dSem : nullptr); a.b.resize(352);
        gz = splitk;
    } else if (fam == "opus_a16w16") {
        // opus_gemm_noscale_kargs_gfx950: 4 pointers, then 11 int32 (strides in elements)
        a.ptr(0, dA); a.ptr(8, dB); a.ptr(16, dC); a.ptr(24, nullptr);
        int vals[11] = {M, N, K, 1, K, K, N, 0, 0, 0, 0};
        for (int i = 0; i < 11; i++) a.u32(32 + 4 * i, vals[i]);
        a.b.resize(80);
        gx = ((M + tM - 1) / tM) * ((N + tN - 1) / tN); gy = 1; block = 512;
    } else if (fam == "asm_i8") {
        dSA = dev(sa_row); dSB = dev(sb_col); dBias = dev(std::vector<float>(N, 0.0f));
        a.ptr(0, dC); a.ptr(16, dA); a.ptr(32, dB); a.ptr(48, dSA); a.ptr(64, dSB); a.ptr(80, dBias);
        a.u32(96, M); a.u32(112, N); a.u32(128, K); a.u32(144, K); a.u32(160, K); a.u32(176, N * 2); a.u32(192, 1); a.b.resize(208);
    } else if (fam == "asm_fp8bs") {
        std::vector<float> sat((size_t)(K / 128) * M);  // K-major
        for (int m = 0; m < M; m++) for (int kb = 0; kb < K / 128; kb++) sat[(size_t)kb * M + m] = sa_blk[(size_t)m * (K / 128) + kb];
        dSA = dev(sat); dSB = dev(sb_blk); dBias = dev(std::vector<float>(N, 0.0f));
        a.ptr(0, dC); a.ptr(16, dA); a.ptr(32, dB); a.ptr(48, dSA); a.ptr(64, dSB); a.ptr(80, dBias);
        a.u32(96, M); a.u32(112, N); a.u32(128, K); a.u32(144, K); a.u32(160, K); a.u32(176, N * 2); a.u32(192, 1);
        a.u32(208, M); a.u32(224, N / 128); a.u32(240, K / 128); a.b.resize(256);
    } else if (fam == "asm_f4") {
        int sm, sn;
        auto ssa = shuffle_scale(sa8, M, K / 32, sm, sn);
        int sm2, sn2;
        auto ssb = shuffle_scale(sb8, N, K / 32, sm2, sn2);
        dSA = dev(ssa); dSB = dev(ssb);
        a.ptr(0, dC); a.ptr(16, nullptr); a.ptr(32, dA); a.ptr(48, dB); a.f32(64, 1.0f); a.f32(80, 0.0f);
        a.u32(96, N); a.u32(128, N); a.u32(160, K); a.u32(192, K); a.u32(224, M); a.u32(240, N); a.u32(256, K);
        a.ptr(272, dSA); a.ptr(288, dSB); a.u32(304, sn); a.u32(336, sn2); a.u32(368, 0); a.b.resize(372);
    } else if (fam == "ours_v4_fp8bs") {
        // our v4 structure, fp8 blockscale: A scales fp32 [M][K/128] row-major (ours), B scales [N/128][K/128], KT = K / 128
        dSA = dev(sa_blk); dSB = dev(sb_blk);
        a.ptr(0, dA); a.ptr(8, dB); a.ptr(16, dC);
        int iv[10] = {M, N, K, 1, K, K, N, 0, 0, 0};
        for (int i = 0; i < 10; i++) a.u32(24 + 4 * i, iv[i]);
        a.ptr(64, dSA); a.ptr(72, dSB); a.u32(80, M); a.u32(84, K / 128); a.u32(88, 0); a.u32(92, 0); a.b.resize(96);
    } else if (fam == "ours_v4_f4" || fam == "ours_v4_f8f4" || fam == "ours_v4_bf16w4") {
        // our v4 structure, MXFP4: A [M][K/2] bytes, B preshuffled bytes, KT = 128-byte K tiles = K / 256; e8m0 scales SHUFFLED
        // (gemm_a4w4_mxfp4 scale_shuf): [rows/16][KT/2][64 lanes][4] - lane 16 g + r: groups g, 4+g of tile 2p, then of tile 2p+1 (K % 512 == 0)
        {
            auto shuf = [&](const std::vector<uint8_t>& s, int rows) {
                const int KG = K / 32, KT = K / 256;
                std::vector<uint8_t> o(s.size());
                for (int rb = 0; rb < rows / 16; rb++)
                    for (int p = 0; p < KT / 2; p++)
                        for (int l = 0; l < 64; l++) {
                            int r = rb * 16 + l % 16, g = l / 16;
                            size_t d = (((size_t)rb * (KT / 2) + p) * 64 + l) * 4;
                            for (int h = 0; h < 2; h++) {
                                int t = 2 * p + h;
                                o[d + 2 * h] = s[(size_t)r * KG + 8 * t + g];
                                o[d + 2 * h + 1] = s[(size_t)r * KG + 8 * t + 4 + g];
                            }
                        }
                return o;
            };
            if (kind == "bf16w4") { dSA = dev(sa8); dSB = dev(sb8); }  // A16W4: B scales [N][K/32] unshuffled (A ones unused)
            else { dSA = dev(shuf(sa8, M)); dSB = dev(shuf(sb8, N)); }
        }
        a.ptr(0, dA); a.ptr(8, dB); a.ptr(16, dC);
        int iv[10] = {M, N, K, 1, K, K, N, 0, 0, 0};
        for (int i = 0; i < 10; i++) a.u32(24 + 4 * i, iv[i]);
        a.ptr(64, dSA); a.ptr(72, dSB); a.u32(80, M); a.u32(84, K / 256); a.u32(88, 0); a.u32(92, 0); a.b.resize(96);
    } else if (fam == "ours_v4_bf16") {
        // our v4 structure, bf16: K in elements, KT = 128-byte K tiles
        a.ptr(0, dA); a.ptr(8, dB); a.ptr(16, dC);
        int iv[10] = {M, N, K, 1, K, K, N, 0, 0, 0};
        for (int i = 0; i < 10; i++) a.u32(24 + 4 * i, iv[i]);
        a.ptr(64, nullptr); a.ptr(72, nullptr); a.u32(80, M); a.u32(84, K * 2 / 128); a.u32(88, 0); a.u32(92, 0); a.b.resize(96);
    } else if (fam == "ours_v4_rowcol_fp8" || fam == "ours_v4_i8") {
        // our v4 structure, 96-byte ABI: A, B (preshuffled), C, M N K batch, strides, sfa = row scales f32 [M], sfb = column scales f32 [N], KT
        dSA = dev(sa_row); dSB = dev(sb_col);
        a.ptr(0, dA); a.ptr(8, dB); a.ptr(16, dC);
        int iv[10] = {M, N, K, 1, K, K, N, 0, 0, 0};
        for (int i = 0; i < 10; i++) a.u32(24 + 4 * i, iv[i]);
        a.ptr(64, dSA); a.ptr(72, dSB); a.u32(80, M); a.u32(84, K / 128); a.u32(88, 0); a.u32(92, 0); a.b.resize(96);
    } else if (ours) {
        // ours: A, B, C, then scales (i8: row, col; fp8bs: A [M, K/128] row-major, B [N/128, K/128]; f4: A, B e8m0 [rows, K/32]), then M N K
        size_t o = 0;
        a.ptr(o, dA); o += 8; a.ptr(o, dB); o += 8; a.ptr(o, dC); o += 8;
        if (kind == "i8") { dSA = dev(sa_row); dSB = dev(sb_col); }
        if (kind == "fp8bs") { dSA = dev(sa_blk); dSB = dev(sb_blk); }
        if (kind == "f4") { dSA = dev(sa8); dSB = dev(sb8); }
        if (dSA) { a.ptr(o, dSA); o += 8; a.ptr(o, dSB); o += 8; }
        a.u32(o, M); a.u32(o + 4, N); a.u32(o + 8, K); a.b.resize(o + 12);
    }

    // ours with split-K (SPLITK > 1): fp32 partials [S][M][N] into a workspace, then REDUCE_CO / REDUCE_KN (gemm_splitk_reduce_s<S>[_rc]) into C
    void* dWS = nullptr;
    Arg ra;
    hipFunction_t rfn = nullptr;
    if (fam.rfind("ours_v4_", 0) == 0 && splitk > 1) {
        const int MP = (M + tM - 1) / tM * tM;
        HC(hipMalloc(&dWS, (size_t)splitk * MP * N * 4 + 256));
        a.ptr(16, dWS);
        a.u32(36, splitk);
        a.u32(92, MP);
        a.u32(88, (kind == "f4" ? K / 256 : kind == "bf16" ? K / 64 : K / 128) / splitk);
        gz = splitk;
        const std::string knm(kn);
        if (knm.size() > 3 && knm.compare(knm.size() - 3, 3, "_lc") == 0) {
            // in-kernel last-arriver reduce: bf16 C at 96, the tile counters (u32 [MP / tileM][N / tileN][4 waves], zero; the kernel resets them) at 104
            void* dCnt = nullptr;
            const size_t cb = (size_t)(MP / tM) * (N / tN) * 4 * 4;
            HC(hipMalloc(&dCnt, cb + 256));
            HC(hipMemset(dCnt, 0, cb + 256));
            if (a.b.size() < 112) a.b.resize(112);
            a.ptr(96, dC);
            a.ptr(104, dCnt);
        } else {
        hipModule_t rmod;
        HC(hipModuleLoad(&rmod, getenv("REDUCE_CO")));
        HC(hipModuleGetFunction(&rfn, rmod, getenv("REDUCE_KN")));
        ra.ptr(0, dWS); ra.ptr(8, dC); ra.ptr(16, dSA); ra.ptr(24, dSB); ra.u32(32, M); ra.u32(36, N); ra.u32(40, splitk); ra.u32(44, MP); ra.b.resize(48);
        }
    }
    // BATCH=n (ours_v4_bf16 / ours_v4_rowcol_fp8 with a `_bmm` kernel): n dense batches; the LAST batch holds the reference data (the others
    // are filled with 0x3c bytes), grid z = n; the check reads the last batch's C
    const int nbat = envi("BATCH", 1);
    void* dCcheck = dC;
    if (nbat > 1) {
        auto rep = [&](void* src, size_t bytes) {
            uint8_t* d;
            HC(hipMalloc(&d, bytes * nbat + 256));
            HC(hipMemset(d, 0x3c, bytes * nbat));
            HC(hipMemcpy(d + bytes * (nbat - 1), src, bytes, hipMemcpyDeviceToDevice));
            return (void*)d;
        };
        a.ptr(0, rep(dA, ha.size())); a.ptr(8, rep(dB, hb_dev.size()));
        uint8_t* dCb; HC(hipMalloc(&dCb, (size_t)M * N * 2 * nbat + 256)); HC(hipMemset(dCb, 0, (size_t)M * N * 2 * nbat));
        a.ptr(16, dCb); dCcheck = dCb + (size_t)M * N * 2 * (nbat - 1);
        if (fam == "ours_v4_rowcol_fp8") { a.ptr(64, rep(dSA, (size_t)M * 4)); a.ptr(72, rep(dSB, (size_t)N * 4)); }
        a.u32(36, nbat);
        gz = nbat;
    }
    size_t rsz = ra.b.size();
    void* rcfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ra.b.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &rsz, HIP_LAUNCH_PARAM_END};
    size_t asz = a.b.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.b.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() {
        HC(hipModuleLaunchKernel(fn, gx, gy, gz, block, 1, 1, 0, nullptr, nullptr, cfg));
        if (rfn) HC(hipModuleLaunchKernel(rfn, N / 1024, M, 1, 256, 1, 1, 0, nullptr, nullptr, rcfg));
    };
    launch();
    HC(hipDeviceSynchronize());

    // ---- check against the CPU reference on CHECK_ROWS rows ----
    std::vector<uint16_t> out((size_t)M * N);
    HC(hipMemcpy(out.data(), dCcheck, out.size() * 2, hipMemcpyDeviceToHost));
    if (kind == "bf16w4" && envi("W4DBG", 0) == 6) { printf("SPROBE G=%d extra=%g row17=%g\n", envi("GRP", 0), bf16_to_f(out[0]) - 32640.0, bf16_to_f(out[17 * N + 5]) - 32640.0); return 0; }
    if (kind == "bf16w4" && envi("W4DBG", 0) == 3) { printf("PROBE P=%d got=%g %g\n", envi("KK", 0), bf16_to_f(out[0]), bf16_to_f(out[17 * N + 5])); return 0; }
    double max_rel = 0; long bad = 0, checked = 0;
    for (int ri = 0; ri < crow && ri < M; ri++) {
        int m = crow >= M ? ri : (int)((long)ri * (M - 1) / (crow - 1 > 0 ? crow - 1 : 1));
        for (int n = 0; n < N; n++) {
            double ref = 0, mag = 0;
            if (kind == "fp8bs") {
                for (int kb = 0; kb < K / 128; kb++) {
                    double p = 0;
                    for (int k = kb * 128; k < kb * 128 + 128; k++) p += (double)af[(size_t)m * K + k] * bf[(size_t)n * K + k];
                    double s = (double)sa_blk[(size_t)m * (K / 128) + kb] * sb_blk[(size_t)(n / 128) * (K / 128) + kb];
                    ref += p * s;
                    mag += std::fabs(p * s);
                }
            } else if (kind == "f4" || kind == "f8f4" || kind == "bf16w4") {
                for (int g = 0; g < K / 32; g++) {
                    double p = 0;
                    for (int k = g * 32; k < g * 32 + 32; k++) p += (double)af[(size_t)m * K + k] * bf[(size_t)n * K + k];
                    double s = (double)e8m0_to_f(sa8[(size_t)m * (K / 32) + g]) * e8m0_to_f(sb8[(size_t)n * (K / 32) + g]);
                    ref += p * s;
                    mag += std::fabs(p * s);
                }
            } else {
                for (int k = 0; k < K; k++) { double t = (double)af[(size_t)m * K + k] * bf[(size_t)n * K + k]; ref += t; mag += std::fabs(t); }
                if (kind == "i8" || kind == "rowcol_fp8") { ref *= (double)sa_row[m] * sb_col[n]; mag *= (double)sa_row[m] * sb_col[n]; }
            }
            double got = bf16_to_f(out[(size_t)m * N + n]);
            // bf16 output (2^-8 relative) + the worst-case fp32 summation error over K terms of magnitude sum `mag`: K 2^-24 mag (x2)
            double err = std::fabs(got - ref), tol = 1e-2 * std::fabs(ref) + 2.0 * K * std::ldexp(1.0, -24) * mag + 1e-6;
            if (err > tol) {
                if (bad < envi("SHOW_BAD", 0)) printf("  bad m=%d n=%d got=%.6g ref=%.6g mag=%.6g\n", m, n, got, ref, mag);
                bad++;
            }
            max_rel = std::max(max_rel, err / (std::fabs(ref) + mag / K + 1e-12));
            checked++;
        }
    }

    // ---- timing ----
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    for (int i = 0; i < warm; i++) launch();
    HC(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch();
    HC(hipEventRecord(e1));
    HC(hipEventSynchronize(e1));
    float ms; HC(hipEventElapsedTime(&ms, e0, e1));
    double us = 1e3 * ms / iters, tf = 2.0 * nbat * M * N * K / (us * 1e-6) / 1e12;
    printf("%s %s M=%d N=%d K=%d: %.2f us %.1f TF/s check rows=%d checked=%ld bad=%ld max_rel=%.3e %s\n", fam.c_str(), kn, M, N, K, us, tf,
           crow < M ? crow : M, checked, bad, max_rel, bad ? "GEMM_FAIL" : "GEMM_OK");
    return bad ? 3 : 0;
}
