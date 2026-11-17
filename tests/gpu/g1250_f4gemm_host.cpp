// gfx1250 MXFP4 GEMM host (replica harness): D[M,N] bf16 = A[M,K] mxfp4 x B[N,K]^T mxfp4 with E8M0 block-32 scales, for a code object with the
// aiter gfx1250 f4gemm ABI (public aiter csrc/py_itfs_cu/asm_f4gemm.cu: 80-byte preload kernarg, block 128, HIP cluster launch, persistent
// 256-workgroup grid with log2(grid) in kernarg dwords 18 / 19). Data layouts follow aiter/ops/shuffle.py (shuffle_weight_f4: 16 x 16 byte-tile
// transpose of B; shuffle_scale_f4: 32-row x 4-column tiles, padding 0x7F rows). A is not preshuffled (a_preshuffle 0).
//
//   g1250_f4gemm_host <co> <kernel> M N K [persistent=1] [iters=20] [deadline_ms=10000]
// Env: SEED (1), CHECK_ROWS (16 rows spread over M, compared against an fp32 CPU reference), CLUSTER=cx,cy,grid_y (else aiter's automatic
// selection), WARMUP (5). Prints `F4GEMM_OK` / `F4GEMM_FAIL ...`, the time and TF/s. Hang safety: every wait is a polled event with a deadline;
// on expiry the process exits 124 WITHOUT touching the device again (no synchronize / free / reset).
#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#define FAIL(...) do { std::printf("F4GEMM_FAIL "); std::printf(__VA_ARGS__); std::printf("\n"); std::fflush(stdout); _exit(2); } while (0)
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) FAIL("%s: %s", #x, hipGetErrorString(e_)); } while (0)

static int envi(const char* n, int d) { const char* v = getenv(n); return v ? atoi(v) : d; }
static float fp4(uint8_t n) {
    static const float t[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t[n & 15];
}
static float e8m0(uint8_t e) {
    if (e == 0) { uint32_t u = 0x00400000; float f; std::memcpy(&f, &u, 4); return f; }
    uint32_t u = (uint32_t)e << 23; float f; std::memcpy(&f, &u, 4); return f;
}
static float bf16f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }

static void wait_deadline(hipEvent_t e, long ms, const char* what) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        hipError_t q = hipEventQuery(e);
        if (q == hipSuccess) return;
        if (q != hipErrorNotReady) FAIL("%s: %s", what, hipGetErrorString(q));
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(ms)) {
            std::printf("F4GEMM_HANG %s exceeded %ld ms\n", what, ms);
            std::fflush(stdout);
            _exit(124);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

static bool pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s co kernel M N K [persistent] [iters] [deadline_ms]\n", argv[0]); return 1; }
    const char* co = argv[1];
    const char* kn = argv[2];
    const int M = atoi(argv[3]), N = atoi(argv[4]), K = atoi(argv[5]);
    const int persistent = argc > 6 ? atoi(argv[6]) : 1;
    const int iters = argc > 7 ? atoi(argv[7]) : 20;
    const long deadline = argc > 8 ? atol(argv[8]) : 10000;
    if (N % 16 || K % 128 || M <= 0) FAIL("shape: N %% 16, K %% 128 and M > 0 required (got %d %d %d)", M, N, K);

    // launch geometry (aiter select_launch_geometry / make_launch_geometry; tile 256 x 256, 256 persistent workgroups)
    int cx = 4, cy = 4, gy = 1;
    const int tm = (M + 255) / 256, tn = (N + 255) / 256;
    if (const char* c = getenv("CLUSTER")) {
        if (std::sscanf(c, "%d,%d,%d", &cx, &cy, &gy) != 3) FAIL("CLUSTER=cx,cy,grid_y");
    } else {
        while (cx > 1 && tn % cx) cx >>= 1;
        while (cy > 1 && tm % cy) cy >>= 1;
        if (cx * cy == 1) { if (tn >= tm) cx = 2; else cy = 2; }
        const int rows = (tm + cy - 1) / cy;
        while (gy < 4 && gy < rows) gy <<= 1;
    }
    // the cluster rules (a single axis of 16 or a 1x1 "cluster" is refused: these kernels need cluster ids and cluster barriers)
    if (!pow2(cx) || !pow2(cy) || cx > 8 || cy > 8 || cx * cy < 2 || cx * cy > 16) FAIL("cluster %dx%d not allowed", cx, cy);
    const int clusters = 256 / (cx * cy);
    if (!pow2(gy) || gy > clusters) FAIL("grid_y %d", gy);
    const int gxc = persistent ? clusters / gy : N / (256 * cx), gyc = persistent ? gy : M / (192 * cy);
    const unsigned gdx = gxc * cx, gdy = gyc * cy;
    unsigned lgx = 0, lgy = 0;
    for (int v = gxc; v > 1; v >>= 1) lgx++;
    for (int v = gyc; v > 1; v >>= 1) lgy++;

    // data: A [M][K/2], B [N][K/2] packed fp4 (low nibble = even k); scales [rows][K/32] E8M0 near 2^0
    std::mt19937 rng(envi("SEED", 1));
    std::vector<uint8_t> A((size_t)M * K / 2), B((size_t)N * K / 2), SA((size_t)M * (K / 32)), SB((size_t)N * (K / 32));
    for (auto& x : A) x = rng() & 0xFF;
    for (auto& x : B) x = rng() & 0xFF;
    for (auto& x : SA) x = 124 + rng() % 7;
    for (auto& x : SB) x = 124 + rng() % 7;
    // B preshuffle: 16 x 16 byte tiles transposed per tile ([tileN, tileK, n, k])
    std::vector<uint8_t> Bs(B.size());
    const int kp = K / 2;
    for (int n = 0; n < N; n++)
        for (int k = 0; k < kp; k++)
            Bs[((size_t)(n / 16) * (kp / 16) + k / 16) * 256 + (n % 16) * 16 + k % 16] = B[(size_t)n * kp + k];
    // scale preshuffle: rows padded to 32 (0x7F), cols to 4; tiles [rows/32][cols/4], inside m outer, k inner
    auto shuf_scale = [](const std::vector<uint8_t>& s, int rows, int cols) {
        const int rp = (rows + 31) / 32 * 32, cp = (cols + 3) / 4 * 4;
        std::vector<uint8_t> pad((size_t)rp * cp, 0x7F), out((size_t)rp * cp);
        for (int r = 0; r < rp; r++)
            for (int c = 0; c < cp; c++) pad[(size_t)r * cp + c] = r < rows ? (c < cols ? s[(size_t)r * cols + c] : 0) : 0x7F;
        for (int r = 0; r < rp; r++)
            for (int c = 0; c < cp; c++) out[(((size_t)(r / 32) * (cp / 4) + c / 4) * 32 + r % 32) * 4 + c % 4] = pad[(size_t)r * cp + c];
        return out;
    };
    const std::vector<uint8_t> SAs = shuf_scale(SA, M, K / 32), SBs = shuf_scale(SB, N, K / 32);

    HC(hipSetDevice(0));
    uint8_t *dA, *dB, *dSA, *dSB;
    uint16_t* dD;
    HC(hipMalloc(&dA, A.size()));
    HC(hipMalloc(&dB, Bs.size()));
    HC(hipMalloc(&dSA, SAs.size()));
    HC(hipMalloc(&dSB, SBs.size()));
    HC(hipMalloc(&dD, (size_t)M * N * 2));
    HC(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(dB, Bs.data(), Bs.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(dSA, SAs.data(), SAs.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(dSB, SBs.data(), SBs.size(), hipMemcpyHostToDevice));
    HC(hipMemset(dD, 0xFF, (size_t)M * N * 2));

    // kernarg (80 bytes): 5 pointers, strides D / A / B / SA / SB, M, N, K, log2 grid x / y (MXFP4 overlays the GlobalScale slots)
    struct __attribute__((packed)) KA {
        void *D, *A, *B, *SA, *SB;
        uint32_t sd, sa, sb, ssa, ssb, m, n, k, lgx, lgy;
    } ka{dD, dA, dB, dSA, dSB, (uint32_t)N * 2, (uint32_t)K / 2, (uint32_t)K / 2, (uint32_t)(K / 128 * 4), (uint32_t)(K / 128 * 4),
         (uint32_t)M, (uint32_t)N, (uint32_t)K, persistent ? lgx : 0u, persistent ? lgy : 0u};
    static_assert(sizeof(KA) == 80, "kernarg size");
    size_t ksz = sizeof(ka);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ka, HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksz, HIP_LAUNCH_PARAM_END};

    hipModule_t mod;
    hipFunction_t f;
    HC(hipModuleLoad(&mod, co));
    HC(hipModuleGetFunction(&f, mod, kn));
    hipStream_t s;
    HC(hipStreamCreate(&s));
    HIP_LAUNCH_CONFIG lc{};
    lc.gridDimX = gdx; lc.gridDimY = gdy; lc.gridDimZ = 1;
    lc.blockDimX = 128; lc.blockDimY = 1; lc.blockDimZ = 1;
    lc.sharedMemBytes = 0; lc.hStream = s;
    hipLaunchAttribute at{};
    at.id = hipLaunchAttributeClusterDimension;
    at.val.clusterDim.x = cx; at.val.clusterDim.y = cy; at.val.clusterDim.z = 1;
    lc.attrs = &at; lc.numAttrs = 1;
    std::printf("launch grid %ux%u cluster %dx%d (cluster grid %dx%d, log2 %u/%u)\n", gdx, gdy, cx, cy, gxc, gyc, lgx, lgy);
    std::fflush(stdout);

    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    // one launch first (correctness), each wait bounded
    HC(hipEventRecord(e0, s));
    HC(hipDrvLaunchKernelEx(&lc, f, nullptr, cfg));
    HC(hipEventRecord(e1, s));
    wait_deadline(e1, deadline, "first launch");
    std::vector<uint16_t> D((size_t)M * N);
    HC(hipMemcpy(D.data(), dD, D.size() * 2, hipMemcpyDeviceToHost));
    // a hash of the whole output (FNV-1a 64): two code objects on the same inputs give the same hash exactly when every output bit agrees
    uint64_t dhash = 1469598103934665603ull;
    for (uint16_t x : D) {
        dhash = (dhash ^ (x & 0xff)) * 1099511628211ull;
        dhash = (dhash ^ (x >> 8)) * 1099511628211ull;
    }
    std::printf("output hash %016llx\n", (unsigned long long)dhash);

    // reference on sampled rows (fp32)
    const int cr = std::min(M, envi("CHECK_ROWS", 16));
    long bad = 0, checked = 0;
    double maxe = 0;
    for (int ri = 0; ri < cr; ri++) {
        const int m = cr == 1 ? 0 : (int)((long)ri * (M - 1) / (cr - 1));
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++) {
                const float a = fp4(A[(size_t)m * kp + k / 2] >> (4 * (k & 1))) * e8m0(SA[(size_t)m * (K / 32) + k / 32]);
                const float b = fp4(B[(size_t)n * kp + k / 2] >> (4 * (k & 1))) * e8m0(SB[(size_t)n * (K / 32) + k / 32]);
                acc += (double)a * b;
            }
            const double got = bf16f(D[(size_t)m * N + n]);
            const double err = std::fabs(got - acc) / std::max(1.0, std::fabs(acc));
            maxe = std::max(maxe, err);
            if (!(err <= 1.0 / 128)) bad++;
            checked++;
        }
    }

    const int warm = envi("WARMUP", 5);
    for (int i = 0; i < warm; i++) HC(hipDrvLaunchKernelEx(&lc, f, nullptr, cfg));
    HC(hipEventRecord(e0, s));
    for (int i = 0; i < iters; i++) HC(hipDrvLaunchKernelEx(&lc, f, nullptr, cfg));
    HC(hipEventRecord(e1, s));
    wait_deadline(e1, deadline + (long)iters * 1000, "timed launches");
    float ms = 0;
    HC(hipEventElapsedTime(&ms, e0, e1));
    const double us = 1e3 * ms / iters;
    std::printf("f4gemm M=%d N=%d K=%d: %.2f us %.1f TF/s check rows=%d checked=%ld bad=%ld max_rel=%.3e %s\n", M, N, K, us,
                2.0 * M * N * (double)K / (us * 1e-6) / 1e12, cr, checked, bad, maxe, bad ? "F4GEMM_FAIL" : "F4GEMM_OK");
    std::printf("{\"host\":\"g1250_f4gemm\",\"kernel\":\"%s\",\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},\"grid\":[%u,%u,1],\"cluster\":[%d,%d,1],"
                "\"us\":%.3f,\"tflops\":%.2f,\"bad\":%ld,\"ok\":%s}\n", kn, M, N, K, gdx, gdy, cx, cy, us, 2.0 * M * N * (double)K / (us * 1e-6) / 1e12, bad,
                bad ? "false" : "true");
    return bad ? 3 : 0;
}
