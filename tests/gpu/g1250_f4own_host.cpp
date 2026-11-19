// Host of our MXFP4 TDM GEMM (crates/amdgpu-kernels/src/probes/g1250_gemm.rs, gemm_g1250_tdm_stage_f4): C[M][N] f32 = A[M][K] x B[N][K]^T, fp4 e2m1 (two per byte, low nibble first) with E8M0
// scales per 32 K, against an fp32 CPU reference on sampled rows.
//
//   g1250_f4own_host <co> <kernel> M N K [iters=0] [BM=128] [BN=64]
//
// (argv[9] = work-items per work-group, default 128)  Kernargs: a @0, b @8, sa @16, sb @24, c @32 (pointers), k @40, n @44. Grid (N / BN, M / BM), 128 work-items. Hang policy (docs/mi450-nodes.md): polled waits with a wall-clock limit
// (PROBE_WATCHDOG_S, default 15), exit 124 on expiry without touching the device again. Env: SEED (1), CHECK_ROWS (16).
// One JSON line: {"host":"g1250_f4own","kernel":..,"M":..,"N":..,"K":..,"bad":N,"maxrel":..,"us":..,"tflops":..,"ok":bool}
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); _exit(2); } } while (0)

static double g_limit = 15.0;
static int g_cx = 1, g_cy = 1;   // CLUSTER=cx,cy: launch with the cluster attribute (a multicast kernel); the grid must divide by it

// one launch: plain, or with the cluster dimension attribute (hipDrvLaunchKernelEx)
static hipError_t launch(hipFunction_t fn, unsigned gx, unsigned gy, unsigned thr, void** cfg) {
    if (g_cx * g_cy <= 1) return hipModuleLaunchKernel(fn, gx, gy, 1, thr, 1, 1, 0, nullptr, nullptr, cfg);
    HIP_LAUNCH_CONFIG lc{};
    lc.gridDimX = gx; lc.gridDimY = gy; lc.gridDimZ = 1;
    lc.blockDimX = thr; lc.blockDimY = 1; lc.blockDimZ = 1;
    lc.sharedMemBytes = 0; lc.hStream = nullptr;
    hipLaunchAttribute at{};
    at.id = hipLaunchAttributeClusterDimension;
    at.val.clusterDim.x = g_cx; at.val.clusterDim.y = g_cy; at.val.clusterDim.z = 1;
    lc.attrs = &at; lc.numAttrs = 1;
    return hipDrvLaunchKernelEx(&lc, fn, nullptr, cfg);
}
static void wait_polled(hipEvent_t ev, const char* what) {
    auto t0 = std::chrono::steady_clock::now();
    while (hipEventQuery(ev) == hipErrorNotReady) {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s > g_limit) { printf("F4OWN_HANG: %s still running after %.0f s; leaving without touching the device (exit 124)\n", what, s); fflush(stdout); _exit(124); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static float fp4(uint8_t n) {
    static const float t[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t[n & 15];
}
static float e8m0(uint8_t e) { uint32_t u = (uint32_t)e << 23; float f; std::memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s <co> <kernel> M N K [iters] [BM] [BN]\n", argv[0]); return 1; }
    const char* kn = argv[2];
    const int M = atoi(argv[3]), N = atoi(argv[4]), K = atoi(argv[5]);
    const int iters = argc > 6 ? atoi(argv[6]) : 0;
    const int BM = argc > 7 ? atoi(argv[7]) : 128, BN = argc > 8 ? atoi(argv[8]) : 64, THR = argc > 9 ? atoi(argv[9]) : 128;
    if (M % BM || N % BN || K % 128) { fprintf(stderr, "M %% %d, N %% %d, K %% 128 required\n", BM, BN); return 1; }
    if (const char* w = getenv("PROBE_WATCHDOG_S")) g_limit = atof(w);
    if (const char* cl = getenv("CLUSTER")) std::sscanf(cl, "%d,%d", &g_cx, &g_cy);
    // the cluster the kernel was compiled for (its kd cluster_dims, exported by the payload): refuse any other launch shape
    if (const char* ex = getenv("EXPECT_CLUSTER")) {
        if (atoi(ex) != g_cx * g_cy) { fprintf(stderr, "the kernel declares a cluster of %s workgroups, the launch asks for %d x %d: refused\n", ex, g_cx, g_cy); return 1; }
    }
    if (g_cx * g_cy > 1 && ((N / BN) % g_cx || (M / BM) % g_cy)) { fprintf(stderr, "the grid (%d x %d) must divide by the cluster (%d x %d)\n", N / BN, M / BM, g_cx, g_cy); return 1; }
    hipDeviceProp_t prop;
    HC(hipGetDeviceProperties(&prop, 0));
    printf("device: %s  arch: %s  CUs: %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    hipModule_t mod;
    hipFunction_t fn;
    HC(hipModuleLoad(&mod, argv[1]));
    HC(hipModuleGetFunction(&fn, mod, kn));

    std::mt19937 rng(getenv("SEED") ? atoi(getenv("SEED")) : 1);
    const size_t KB = K / 2, KS = K / 32;
    std::vector<uint8_t> A((size_t)M * KB), B((size_t)N * KB), SA((size_t)M * KS), SB((size_t)N * KS);
    for (auto& x : A) x = (uint8_t)(rng() & 0xff);
    for (auto& x : B) x = (uint8_t)(rng() & 0xff);
    for (auto& x : SA) x = (uint8_t)(125 + rng() % 5);   // 2^-2 .. 2^2
    for (auto& x : SB) x = (uint8_t)(125 + rng() % 5);
    // ONLY_K=lo,hi: zero every A element outside the K range [lo, hi) (a bisect aid: the output is nonzero only when the K tiles of that range reach the WMMA correctly)
    if (const char* ok = getenv("ONLY_K")) {
        int lo = 0, hi = K;
        std::sscanf(ok, "%d,%d", &lo, &hi);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                if (k < lo || k >= hi) A[(size_t)i * KB + k / 2] &= (k & 1) ? 0x0f : 0xf0;
    }
    uint8_t *dA, *dB, *dSA, *dSB; float* dC;
    HC(hipMalloc(&dA, A.size())); HC(hipMalloc(&dB, B.size())); HC(hipMalloc(&dSA, SA.size())); HC(hipMalloc(&dSB, SB.size())); const bool bf16 = getenv("OUT_BF16") != nullptr;   // the kernel writes bf16 (RNE), M x N x 2 bytes
    HC(hipMalloc(&dC, (size_t)M * N * (bf16 ? 2 : 4)));
    HC(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice)); HC(hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice));
    // SCALE_SHUF=1: the scales in the vendor's preshuffled layout (aiter shuffle_scale_f4: [rows / 32][K / 128][32 rows][4 bytes]; rows and
    // K / 32 here are multiples of 32 and 4); the reference below keeps reading the row-major SA / SB
    auto shuf_scale = [](const std::vector<uint8_t>& s, int rows, int cols) {
        std::vector<uint8_t> out(s.size());
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) out[(((size_t)(r / 32) * (cols / 4) + c / 4) * 32 + r % 32) * 4 + c % 4] = s[(size_t)r * cols + c];
        return out;
    };
    if (getenv("SCALE_SHUF")) {
        const std::vector<uint8_t> sa2 = shuf_scale(SA, M, K / 32), sb2 = shuf_scale(SB, N, K / 32);
        HC(hipMemcpy(dSA, sa2.data(), sa2.size(), hipMemcpyHostToDevice)); HC(hipMemcpy(dSB, sb2.data(), sb2.size(), hipMemcpyHostToDevice));
    } else {
        HC(hipMemcpy(dSA, SA.data(), SA.size(), hipMemcpyHostToDevice)); HC(hipMemcpy(dSB, SB.data(), SB.size(), hipMemcpyHostToDevice));
    }
    HC(hipMemset(dC, 0xff, (size_t)M * N * (bf16 ? 2 : 4)));

    // PERSIST=G: a persistent kernel (a tile loop inside): G workgroups (a multiple of the tile columns N / BN), each takes the tile column
    // w % (N / BN) and the rows w / (N / BN), + G / (N / BN), ..; extra kernargs: tile rows M / BM @48, log2(N / BN) @52, the row step @56.
    // The first (smoke) launch is one workgroup per tile column (row step 1).
    const int persist = getenv("PERSIST") ? atoi(getenv("PERSIST")) : 0;
    const int tx = N / BN, ty = M / BM;
    int lgx = 0;
    while ((1 << lgx) < tx) ++lgx;
    if (persist && ((1 << lgx) != tx || persist % tx || persist > tx * ty)) { fprintf(stderr, "PERSIST=%d: N / BN = %d must be a power of two dividing it, at most %d tiles\n", persist, tx, tx * ty); return 1; }
    // STAMPS=1: the kernel stores per-wave K-loop phase sums (shader cycles: stage issue, window, wait, barrier; iterations) into a debug buffer (kernarg @64):
    // [tile row][tile column & 63][wave] x 8 dwords; N / BN must be <= 64
    const bool stamps = getenv("STAMPS") != nullptr;
    void* dDbg = nullptr;
    const size_t dbg_bytes = (size_t)ty * 64 * 4 * 64;
    if (stamps) { if (tx > 64) { fprintf(stderr, "STAMPS: N / BN = %d must be at most 64\n", tx); return 1; } HC(hipMalloc(&dDbg, dbg_bytes)); HC(hipMemset(dDbg, 0, dbg_bytes)); }
    struct { void *a, *b, *sa, *sb, *c; int k, n, ty, lgx, ystep; int pad_; void* dbg; } ka{dA, dB, dSA, dSB, dC, K, N, ty, lgx, 1, 0, dDbg};
    // SWIZZLE=G: the kernel remaps the tile order (rows grouped by G); it needs the log2 of the tile columns (kernarg @52)
    const int swizzle = getenv("SWIZZLE") ? atoi(getenv("SWIZZLE")) : 0;
    if (swizzle && (((1 << lgx) != tx) || ty % swizzle)) { fprintf(stderr, "SWIZZLE=%d: N / BN = %d must be a power of two and the tile rows %d a multiple of it\n", swizzle, tx, ty); return 1; }
    size_t ksz = (persist || swizzle || stamps) ? sizeof(ka) : 48;
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ka, HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksz, HIP_LAUNCH_PARAM_END};
    for (int stage = 0; stage < 2; ++stage) {
        unsigned gx = stage == 0 ? g_cx : N / BN, gy = stage == 0 ? g_cy : M / BM;
        if (persist) { gx = stage == 0 ? tx : persist; gy = 1; ka.ystep = gx / tx; }
        if (stage == 1 && gx == 1 && gy == 1) break;
        hipEvent_t e;
        HC(hipEventCreate(&e));
        HC(launch(fn, gx, gy, THR, cfg));
        HC(hipEventRecord(e, nullptr));
        wait_polled(e, kn);
    }
    std::vector<float> C((size_t)M * N);
    if (bf16) {
        std::vector<uint16_t> Cb((size_t)M * N);
        HC(hipMemcpy(Cb.data(), dC, Cb.size() * 2, hipMemcpyDeviceToHost));
        for (size_t x = 0; x < Cb.size(); ++x) { uint32_t u = (uint32_t)Cb[x] << 16; float f; memcpy(&f, &u, 4); C[x] = f; }
    } else {
        HC(hipMemcpy(C.data(), dC, C.size() * 4, hipMemcpyDeviceToHost));
    }

    // fp32 reference on sampled rows (all columns)
    const int nrows = getenv("CHECK_ROWS") ? atoi(getenv("CHECK_ROWS")) : 16;
    long bad = 0; double maxrel = 0;
    // MISMAP=1: per checked row, the wrong elements per workgroup tile column (a bisect aid for cluster / multicast kernels: which operand slices never arrived)
    const bool mismap = getenv("MISMAP") != nullptr;
    std::vector<std::vector<int>> rowbad;
    std::vector<int> rowid;
    for (int s = 0; s < std::min(nrows, M); ++s) {
        const int i = nrows >= M ? s : (int)((long)s * (M - 1) / std::max(1, nrows - 1));
        std::vector<float> arow(K);
        for (int k = 0; k < K; ++k) arow[k] = fp4(A[(size_t)i * KB + k / 2] >> (4 * (k & 1))) * e8m0(SA[(size_t)i * KS + k / 32]);
        if (mismap) { rowid.push_back(i); rowbad.emplace_back(N / BN, 0); }
        for (int j = 0; j < N; ++j) {
            double ref = 0, scale = 0;
            for (int k = 0; k < K; ++k) {
                double p = (double)arow[k] * fp4(B[(size_t)j * KB + k / 2] >> (4 * (k & 1))) * e8m0(SB[(size_t)j * KS + k / 32]);
                ref += p; scale += std::fabs(p);
            }
            const double got = C[(size_t)i * N + j];
            const double err = std::fabs(got - ref);
            const bool ok = !std::isnan(got) && err <= (bf16 ? 8e-3 * std::fabs(ref) : 0.0) + 5e-4 * scale + 1e-3;
            if (!ok) { if (bad < 8) printf("  mismatch C[%d][%d]: got %g ref %g\n", i, j, got, ref); ++bad; if (mismap) rowbad.back()[j / BN]++; }
            if (!std::isnan(err)) maxrel = std::max(maxrel, err / (scale + 1e-9));
        }
    }
    if (mismap) {
        for (size_t r = 0; r < rowid.size(); ++r) {
            printf("  row %4d:", rowid[r]);
            for (int c : rowbad[r]) printf(" %3d", c);
            printf("\n");
        }
    }
    double us = 0, tflops = 0;
    // SKIPCHECK=1: time even when the result is wrong (profiling kernels with parts removed from the assembly; never for a kernel that is supposed to be correct)
    if (iters > 0 && (bad == 0 || getenv("SKIPCHECK"))) {
        hipEvent_t e0, e1;
        HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
        unsigned gx = N / BN, gy = M / BM;
        if (persist) { gx = persist; gy = 1; ka.ystep = persist / tx; }
        for (int w = 0; w < 3; ++w) HC(launch(fn, gx, gy, THR, cfg));
        HC(hipEventRecord(e0, nullptr));
        for (int i = 0; i < iters; ++i) HC(launch(fn, gx, gy, THR, cfg));
        HC(hipEventRecord(e1, nullptr));
        wait_polled(e1, "timing");
        float ms = 0;
        HC(hipEventElapsedTime(&ms, e0, e1));
        us = 1000.0 * ms / iters;
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12;
    }
    if (stamps && iters > 0) {
        HC(hipDeviceSynchronize());
        std::vector<unsigned> rec(dbg_bytes / 4);
        HC(hipMemcpy(rec.data(), dDbg, dbg_bytes, hipMemcpyDeviceToHost));
        const char* ph[4] = {"issue", "window", "wait", "barrier"};
        for (int w = 0; w < 4; ++w) {
            double sum[4] = {0, 0, 0, 0}; long cnt = 0; double it = 0, pro = 0, pro_setup = 0, lp = 0, epi = 0, e1 = 0, e2 = 0, e3 = 0;
            for (int r = 0; r < ty; ++r) for (int c = 0; c < tx; ++c) {
                const unsigned* q = &rec[(((size_t)r * 64 + c) * 4 + w) * 16];
                if (q[4] == 0) continue;
                for (int k = 0; k < 4; ++k) sum[k] += (double)q[k] / q[4];
                it += q[4]; ++cnt; pro += (double)(unsigned)(q[5] - q[8]); pro_setup += (double)(unsigned)(q[11] - q[8]); lp += (double)(unsigned)(q[6] - q[5]); epi += (double)(unsigned)(q[7] - q[6]); e1 += (double)(unsigned)(q[9] - q[6]); e2 += (double)(unsigned)(q[10] - q[9]); e3 += (double)(unsigned)(q[7] - q[10]);
            }
            if (cnt) printf("{\"stamps\":{\"wave\":%d,\"records\":%ld,\"iters\":%.1f,\"cycles_per_iter\":{\"issue\":%.1f,\"window\":%.1f,\"wait\":%.1f,\"barrier\":%.1f,\"total\":%.1f}}}\n", w, cnt, it / cnt, sum[0] / cnt, sum[1] / cnt, sum[2] / cnt, sum[3] / cnt, (sum[0] + sum[1] + sum[2] + sum[3]) / cnt);
            if (cnt) printf("{\"stamps_kernel\":{\"wave\":%d,\"prologue\":%.0f,\"pro_setup\":%.0f,\"loop\":%.0f,\"epilogue\":%.0f,\"epi_stage\":%.0f,\"epi_issue\":%.0f,\"epi_store_wait\":%.0f}}\n", w, pro / cnt, pro_setup / cnt, lp / cnt, epi / cnt, e1 / cnt, e2 / cnt, e3 / cnt);
            (void)ph;
        }
    }
    printf("{\"host\":\"g1250_f4own\",\"kernel\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"bad\":%ld,\"maxrel\":%.3g,\"us\":%.2f,\"tflops\":%.3f,\"ok\":%s}\n", kn, M, N, K, bad, maxrel, us, tflops, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}
