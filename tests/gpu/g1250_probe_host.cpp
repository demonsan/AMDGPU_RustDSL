// Host of the gfx1250 G1 probes (docs/gfx1250-plan.md): probe_g1250_add (3 f32 buffers), probe_g1250_math (transcendentals against the CPU), probe_g1250_tdm (tensor_load_to_lds -> wait ->
// barrier -> LDS read -> store), probe_g1250_select (lane masks: compare / v_cndmask) and probe_g1250_buffer (buffer loads / stores / a returning atomic add). The first own-compiled gfx1250 kernels on hardware.
//
//   g1250_probe_host add  <co> [kernel=probe_g1250_add]  [N workgroups = 4096]
//   g1250_probe_host math <co> [kernel=probe_g1250_math]
//   g1250_probe_host tdm  <co> [kernel=probe_g1250_tdm]  [dwords = 64]
//   g1250_probe_host select <co> [kernel=probe_g1250_select]
//   g1250_probe_host buffer <co> [kernel=probe_g1250_buffer]
//   env: PROBE_WATCHDOG_S (default 15): the wall-clock limit of every wait; ITERS (100) for the timing of `add`
//
// Smallest configuration first: `add` runs ONE workgroup before the full grid; `tdm` runs the smallest tile (1 dword per work-item). Hang policy (docs/mi450-nodes.md): every wait is POLLED against a
// deadline; on expiry the program prints PROBE_HANG and leaves with `_exit(124)` without touching the device again (no synchronize / free / reset); a failed check exits 3 and nothing retries.
// One JSON line per successful run: {"host":"g1250_probe","kernel":..,"shape":{..},"us":..,"gbs":..,"maxrel":..,"bad":N,"ok":bool}
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
#include <unistd.h>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); _exit(2); } } while (0)

static double g_limit = 15.0;
static void wait_polled(hipEvent_t ev, const char* what) {
    auto t0 = std::chrono::steady_clock::now();
    while (hipEventQuery(ev) == hipErrorNotReady) {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s > g_limit) { printf("PROBE_HANG: %s still running after %.0f s; leaving without touching the device (exit 124)\n", what, s); fflush(stdout); _exit(124); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

struct Kernel {
    hipFunction_t fn;
    // one launch with a kernarg blob; the completion is awaited by polling
    void launch(unsigned grid, unsigned block, std::vector<uint8_t>& args, const char* what) {
        size_t sz = args.size();
        void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
        hipEvent_t e;
        HC(hipEventCreate(&e));
        HC(hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, nullptr, cfg));
        HC(hipEventRecord(e, nullptr));
        wait_polled(e, what);
    }
    // a cluster launch: `grid` workgroups in clusters of `cx` (x only); the cluster dims must divide the grid
    void launch_cluster(unsigned grid, unsigned block, unsigned cx, std::vector<uint8_t>& args, const char* what) {
        if (cx < 1 || grid % cx != 0) {
            printf("PROBE_FAIL cluster %u does not divide grid %u (refused)\n", cx, grid);
            fflush(stdout);
            _exit(2);
        }
        size_t sz = args.size();
        void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
        HIP_LAUNCH_CONFIG lc{};
        lc.gridDimX = grid; lc.gridDimY = 1; lc.gridDimZ = 1;
        lc.blockDimX = block; lc.blockDimY = 1; lc.blockDimZ = 1;
        lc.sharedMemBytes = 0; lc.hStream = nullptr;
        hipLaunchAttribute at{};
        at.id = hipLaunchAttributeClusterDimension;
        at.val.clusterDim.x = cx; at.val.clusterDim.y = 1; at.val.clusterDim.z = 1;
        lc.attrs = &at; lc.numAttrs = 1;
        hipEvent_t e;
        HC(hipEventCreate(&e));
        HC(hipDrvLaunchKernelEx(&lc, fn, nullptr, cfg));
        HC(hipEventRecord(e, nullptr));
        wait_polled(e, what);
    }
};

static void put64(std::vector<uint8_t>& a, size_t off, const void* p) { memcpy(&a[off], &p, 8); }
static void put32(std::vector<uint8_t>& a, size_t off, uint32_t v) { memcpy(&a[off], &v, 4); }
static void putf(std::vector<uint8_t>& a, size_t off, float v) { memcpy(&a[off], &v, 4); }

static int run_add(Kernel& k, const char* kn, int nwg) {
    for (int stage = 0; stage < 2; ++stage) {
        const int wgs = stage == 0 ? 1 : nwg;
        const size_t n = (size_t)wgs * 256;
        std::vector<float> a(n), b(n), o(n);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> d(-100.f, 100.f);
        for (size_t i = 0; i < n; ++i) { a[i] = d(rng); b[i] = d(rng); }
        float *da, *db, *dout;
        HC(hipMalloc(&da, n * 4)); HC(hipMalloc(&db, n * 4)); HC(hipMalloc(&dout, n * 4));
        HC(hipMemcpy(da, a.data(), n * 4, hipMemcpyHostToDevice));
        HC(hipMemcpy(db, b.data(), n * 4, hipMemcpyHostToDevice));
        HC(hipMemset(dout, 0xff, n * 4));
        std::vector<uint8_t> args(24, 0);
        put64(args, 0, da); put64(args, 8, db); put64(args, 16, dout);
        k.launch(wgs, 256, args, kn);
        HC(hipMemcpy(o.data(), dout, n * 4, hipMemcpyDeviceToHost));
        long bad = 0;
        for (size_t i = 0; i < n; ++i) bad += o[i] != a[i] + b[i];
        printf("add: %d workgroup(s), %zu elements: bad=%ld %s\n", wgs, n, bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
        if (bad) return 3;
        if (stage == 1) {
            const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 100;
            hipEvent_t e0, e1;
            HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
            size_t sz = args.size();
            void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
            HC(hipEventRecord(e0, nullptr));
            for (int i = 0; i < iters; ++i) HC(hipModuleLaunchKernel(k.fn, wgs, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
            HC(hipEventRecord(e1, nullptr));
            wait_polled(e1, "the add timing loop");
            float ms; HC(hipEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / iters, gbs = (double)n * 12 / (us * 1e-6) / 1e9;
            printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"workgroups\":%d,\"n\":%zu},\"us\":%.3f,\"gbs\":%.1f,\"bad\":0,\"ok\":true}\n", kn, wgs, n, us, gbs);
        }
    }
    return 0;
}

static int run_math(Kernel& k, const char* kn) {
    struct P { float x, y, z; } ps[] = { { 1.5f, 4.0f, 9.0f }, { -3.25f, 0.37f, 123.456f }, { 10.f, 1e-3f, 1e4f } };
    double worst = 0;
    for (auto& p : ps) {
        float* dout;
        HC(hipMalloc(&dout, 64 * 4));
        HC(hipMemset(dout, 0xff, 64 * 4));
        std::vector<uint8_t> args(24, 0);
        put64(args, 0, dout); putf(args, 8, p.x); putf(args, 12, p.y); putf(args, 16, p.z);
        k.launch(1, 64, args, kn);
        std::vector<float> o(64);
        HC(hipMemcpy(o.data(), dout, 64 * 4, hipMemcpyDeviceToHost));
        long bad = 0;
        double maxrel = 0;
        for (int t = 0; t < 64; ++t) {
            const double ref = std::exp2((double)p.x) / std::sqrt((double)p.y) + std::sqrt((double)p.z) + t;
            const double rel = std::fabs(o[t] - ref) / std::fabs(ref);
            maxrel = std::fmax(maxrel, rel);
            bad += !(rel <= 4e-6);
        }
        worst = std::fmax(worst, maxrel);
        printf("math(x=%g y=%g z=%g): out[0]=%.7g out[63]=%.7g maxrel=%.3g bad=%ld %s\n", p.x, p.y, p.z, o[0], o[63], maxrel, bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
        if (bad) return 3;
    }
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"items\":64},\"maxrel\":%.3g,\"bad\":0,\"ok\":true}\n", kn, worst);
    return 0;
}

static int run_tdm(Kernel& k, const char* kn, int dwords) {
    // the source tensor: `dwords` 32-bit elements, one 1-D tile; the kernel's tile lands at LDS 0 and every work-item copies LDS dword tid to out[tid] (64 work-items: dwords <= 64)
    if (dwords < 1 || dwords > 64) { fprintf(stderr, "dwords must be 1..64\n"); return 1; }
    std::vector<uint32_t> src(64, 0);
    for (int i = 0; i < 64; ++i) src[i] = 0x1000u + 7u * i;
    uint32_t* dsrc;
    uint32_t* dout;
    HC(hipMalloc(&dsrc, 64 * 4));
    HC(hipMalloc(&dout, 64 * 4));
    HC(hipMemcpy(dsrc, src.data(), 64 * 4, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0xee, 64 * 4));
    const uint64_t ga = (uint64_t)dsrc;
    uint32_t d[12] = {};
    // group 0: [1:0] valid, [63:32] lds address (bytes), [120:64] global address, [127:126] type 2
    d[0] = 1;
    d[1] = 0;
    d[2] = (uint32_t)ga;
    d[3] = (uint32_t)((ga >> 32) & 0x1ffffff) | (2u << 30);
    // group 1: workgroup_mask 0, data_size 4 B (encoding 2), tensor_dim0 = dwords, tensor_dim1 = 1, tile_dim0 = dwords, tile_dim1 = 1, stride0 = dwords
    const uint32_t dim0 = dwords, dim1 = 1, tile0 = dwords, tile1 = 1;
    d[4] = 2u << 16;
    d[5] = (dim0 & 0xffff) << 16;
    d[6] = (dim0 >> 16) | ((dim1 & 0xffff) << 16);
    d[7] = (dim1 >> 16) | (tile0 << 16);
    d[8] = tile1;
    d[9] = dwords;   // tensor_dim0_stride [207:160]
    d[10] = 0;
    d[11] = 0;
    std::vector<uint8_t> args(64, 0);
    put64(args, 0, dout);
    for (int i = 0; i < 12; ++i) put32(args, 16 + 4 * i, d[i]);
    k.launch(1, 64, args, kn);
    std::vector<uint32_t> o(64);
    HC(hipMemcpy(o.data(), dout, 64 * 4, hipMemcpyDeviceToHost));
    long bad = 0;
    for (int t = 0; t < 64; ++t) {
        const uint32_t ref = t < dwords ? src[t] : 0u;   // LDS beyond the tile: not written by the tile; only the tile's dwords are checked below
        if (t < dwords) bad += o[t] != ref;
    }
    printf("tdm: tile of %d dword(s): out[0]=0x%x out[%d]=0x%x bad=%ld %s\n", dwords, o[0], dwords - 1, o[dwords - 1], bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"dwords\":%d},\"bad\":%ld,\"ok\":%s}\n", kn, dwords, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

static int run_select(Kernel& k, const char* kn) {
    std::vector<float> a(256), b(256), o(512);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-50.f, 50.f);
    for (int i = 0; i < 256; ++i) { a[i] = d(rng); b[i] = d(rng); }
    a[7] = b[7];   // equal operands: the `>` compare is false, the select takes 2 b
    float *da, *db, *dout;
    HC(hipMalloc(&da, 1024)); HC(hipMalloc(&db, 1024)); HC(hipMalloc(&dout, 2048));
    HC(hipMemcpy(da, a.data(), 1024, hipMemcpyHostToDevice));
    HC(hipMemcpy(db, b.data(), 1024, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0xff, 2048));
    std::vector<uint8_t> args(24, 0);
    put64(args, 0, da); put64(args, 8, db); put64(args, 16, dout);
    k.launch(1, 256, args, kn);
    HC(hipMemcpy(o.data(), dout, 2048, hipMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < 256; ++i) {
        const float ref = a[i] > b[i] ? a[i] : 2.f * b[i];
        bad += o[i] != ref;
        int32_t v; memcpy(&v, &o[256 + i], 4);
        bad += v != ((i & 1) ? 7 : -i);
    }
    printf("select: 256 items: bad=%ld %s\n", bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"items\":256},\"bad\":%ld,\"ok\":%s}\n", kn, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

static int run_buffer(Kernel& k, const char* kn) {
    std::vector<float> a(256), b(256);
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> d(-10.f, 10.f);
    for (int i = 0; i < 256; ++i) { a[i] = d(rng); b[i] = d(rng); }
    float *da, *db, *dout;
    HC(hipMalloc(&da, 1024)); HC(hipMalloc(&db, 1024)); HC(hipMalloc(&dout, 1028));
    HC(hipMemcpy(da, a.data(), 1024, hipMemcpyHostToDevice));
    HC(hipMemcpy(db, b.data(), 1024, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0, 1028));
    std::vector<uint8_t> args(32, 0);
    put64(args, 0, da); put64(args, 8, db); put64(args, 16, dout); put32(args, 24, 1024);   // nb = bytes of b's buffer resource
    k.launch(1, 256, args, kn);
    std::vector<uint32_t> o(257);
    HC(hipMemcpy(o.data(), dout, 1028, hipMemcpyDeviceToHost));
    long bad = 0;
    double maxrel = 0;
    for (int i = 0; i < 256; ++i) {
        float v; memcpy(&v, &o[i], 4);
        const double ref = 3.0 * a[i] + b[i];
        const double rel = std::fabs(v - ref) / std::fmax(std::fabs(ref), 1e-3);
        maxrel = std::fmax(maxrel, rel);
        bad += !(rel <= 1e-6);
    }
    const bool atomic_ok = o[256] == 256u;   // 256 work-items each add 1 to out[256] (initially 0)
    bad += !atomic_ok;
    printf("buffer: 256 items: maxrel=%.3g out[256]=%u (atomic add of 256 ones) bad=%ld %s\n", maxrel, o[256], bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"items\":256},\"maxrel\":%.3g,\"bad\":%ld,\"ok\":%s}\n", kn, maxrel, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_tdm2d: a 16 x 8 f32 tile from the typed descriptor builder. The tensor is rows x cols (row-major); the tile starts at (r0, c0);
// dim0 / dim1 are the extents from the tile start (cols - c0, rows - r0), so a tile reaching past the tensor reads zeros there. Cases:
// inside, and clipped on both axes.
static int run_tdm2d(Kernel& k, const char* kn) {
    const int rows = 20, cols = 40;
    std::vector<float> src(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) src[r * cols + c] = 1000.f * r + c;
    float* dsrc;
    float* dout;
    HC(hipMalloc(&dsrc, src.size() * 4));
    HC(hipMalloc(&dout, 128 * 4));
    HC(hipMemcpy(dsrc, src.data(), src.size() * 4, hipMemcpyHostToDevice));
    long bad_all = 0;
    const int cases[3][2] = {{0, 0}, {5, 9}, {15, 30}};
    for (auto& cs : cases) {
        const int r0 = cs[0], c0 = cs[1];
        HC(hipMemset(dout, 0xee, 128 * 4));
        std::vector<uint8_t> args(32, 0);
        put64(args, 0, dsrc);
        put64(args, 8, dout);
        put32(args, 16, (uint32_t)(cols - c0));
        put32(args, 20, (uint32_t)(rows - r0));
        put32(args, 24, (uint32_t)cols);
        put32(args, 28, (uint32_t)((r0 * cols + c0) * 4));
        k.launch(1, 128, args, kn);
        std::vector<float> o(128);
        HC(hipMemcpy(o.data(), dout, 128 * 4, hipMemcpyDeviceToHost));
        long bad = 0;
        for (int i = 0; i < 128; ++i) {
            const int tr = i / 16, tc = i % 16;
            const float ref = (r0 + tr < rows && c0 + tc < cols) ? src[(r0 + tr) * cols + c0 + tc] : 0.f;
            bad += o[i] != ref;
        }
        printf("tdm2d: tile at (%d, %d): out[0]=%g out[127]=%g bad=%ld %s\n", r0, c0, o[0], o[127], bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
        bad_all += bad;
    }
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"tile\":[16,8]},\"bad\":%ld,\"ok\":%s}\n", kn, bad_all, bad_all == 0 ? "true" : "false");
    return bad_all == 0 ? 0 : 3;
}

// probe_g1250_tdm_ring: 4 tiles of 16 x 8 f32 (a 32 x 16 tensor) through a 2-slot TDM ring; out[t] = sum over the tiles of element t.
static int run_tdm_ring(Kernel& k, const char* kn) {
    std::vector<float> src(4 * 128);
    for (int i = 0; i < 4 * 128; ++i) src[i] = (float)((i * 37) % 101) - 50.f;
    float* dsrc;
    float* dout;
    HC(hipMalloc(&dsrc, src.size() * 4));
    HC(hipMalloc(&dout, 128 * 4));
    HC(hipMemcpy(dsrc, src.data(), src.size() * 4, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0xee, 128 * 4));
    std::vector<uint8_t> args(16, 0);
    put64(args, 0, dsrc);
    put64(args, 8, dout);
    k.launch(1, 128, args, kn);
    std::vector<float> o(128);
    HC(hipMemcpy(o.data(), dout, 128 * 4, hipMemcpyDeviceToHost));
    long bad = 0;
    for (int t = 0; t < 128; ++t) {
        float ref = 0.f;
        for (int i = 0; i < 4; ++i) ref += src[i * 128 + t];
        bad += o[t] != ref;
    }
    printf("tdm_ring: 4 tiles, 2 slots: out[0]=%g out[127]=%g bad=%ld %s\n", o[0], o[127], bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"tiles\":4,\"slots\":2},\"bad\":%ld,\"ok\":%s}\n", kn, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_named: LDS exchange across 4 waves synchronised by named barrier 1; out[t] = 3 * ((t + 32) % 128).
static int run_named(Kernel& k, const char* kn) {
    uint32_t* dout;
    HC(hipMalloc(&dout, 128 * 4));
    HC(hipMemset(dout, 0xee, 128 * 4));
    std::vector<uint8_t> args(8, 0);
    put64(args, 0, dout);
    long bad = 0;
    for (int rep = 0; rep < 50; ++rep) {
        k.launch(256, 128, args, kn);
        std::vector<uint32_t> o(128);
        HC(hipMemcpy(o.data(), dout, 128 * 4, hipMemcpyDeviceToHost));
        for (int t = 0; t < 128; ++t) bad += o[t] != 3u * (uint32_t)((t + 32) % 128);
    }
    printf("named: 50 x 256 workgroups x 4 waves: bad=%ld %s\n", bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"waves\":4},\"bad\":%ld,\"ok\":%s}\n", kn, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_tr8: ds_load_tr8_b64 lane mapping. LDS byte i = i (i < 256), lane l reads at (l % 16) * 16 + (l / 16) * 8; prints, per lane, the 8 memory
// bytes (= addresses) it received, and the decoded (row, col) of a 16 x 16 byte tile (row = addr / 16, col = addr % 16).
static int run_tr8(Kernel& k, const char* kn) {
    uint32_t* dout;
    HC(hipMalloc(&dout, 64 * 4));
    HC(hipMemset(dout, 0xee, 64 * 4));
    std::vector<uint8_t> args(8, 0);
    put64(args, 0, dout);
    k.launch(1, 32, args, kn);
    std::vector<uint8_t> o(256);
    HC(hipMemcpy(o.data(), dout, 256, hipMemcpyDeviceToHost));
    for (int l = 0; l < 32; ++l) {
        printf("lane %2d:", l);
        for (int j = 0; j < 8; ++j) printf(" (%2d,%2d)", o[l * 8 + j] / 16, o[l * 8 + j] % 16);
        printf("\n");
    }
    printf("tr8: PROBE_OK (mapping printed)\n");
    return 0;
}

// probe_g1250_varscale{4,32}[_br]: NV loop-carried WMMA accumulators scaled in place per row each iteration by alpha = in[(i*8+r)*32+lane],
// then += 128 (a WMMA of fp8 ones x twos). Reference O = O * alpha + 128 per iteration (exact in f32 for the alphas used). env NV (4 / 32), N (iterations).
static int run_varscale(Kernel& k, const char* kn) {
    const int nv = getenv("NV") ? atoi(getenv("NV")) : 4, n = getenv("N") ? atoi(getenv("N")) : 4;
    const float choice[4] = {1.0f, 0.5f, 0.25f, 1.0f};
    std::vector<float> in(64 * 8 * 32, 1.0f);  // the kernel's declared extent: 64 iterations
    uint32_t s = 12345;
    for (int j = 0; j < n * 8 * 32; ++j) { s = s * 1103515245u + 12345u; in[j] = choice[(s >> 16) & 3]; }
    float *din, *dout;
    HC(hipMalloc(&din, in.size() * 4));
    HC(hipMalloc(&dout, nv * 8 * 32 * 4));
    HC(hipMemcpy(din, in.data(), in.size() * 4, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0, nv * 8 * 32 * 4));
    std::vector<uint8_t> args(24, 0);
    put64(args, 0, dout);
    put64(args, 8, din);
    memcpy(args.data() + 16, &n, 4);
    k.launch(1, 32, args, kn);
    std::vector<float> o(nv * 8 * 32);
    HC(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
    long bad = 0;
    for (int v = 0; v < nv; ++v)
        for (int r = 0; r < 8; ++r)
            for (int l = 0; l < 32; ++l) {
                float ref = 0;
                // the scale of element r of lane l is alpha[r] of lane l
                for (int i = 0; i < n; ++i) ref = ref * in[(i * 8 + r) * 32 + l] + 128.0f;
                float got = o[(v * 8 + r) * 32 + l];
                if (got != ref) { if (bad < 8) printf("  v%d r%d lane %d: got %g ref %g\n", v, r, l, got, ref); ++bad; }
            }
    printf("varscale: nv=%d n=%d bad=%ld %s\n", nv, n, bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    return bad == 0 ? 0 : 3;
}
// probe_g1250_permlane: the gfx1250 lane permutes against this CPU model (wave32). Selects: P16 / X16: lane i of a 16-lane group reads
// lane (sel nibble i of the 64-bit {b:a}) of its own / the other group; *_VAR: the per-lane select (3 lane) & 15; BCAST: lane a % size of the
// group of `size` lanes; UP: lane i - a (values move to higher lanes), DOWN: lane i + a, a lane whose source would leave the group keeping its\n// own value (measured); XOR: i ^ (a & (size - 1)).
// EXEC (env MASK, default all lanes; the partial-EXEC variant narrows EXEC around each permute): a disabled destination lane keeps old; a
// disabled source lane is read anyway with fetch-invalid, else it is invalid: 0 with bound-ctrl, else the destination keeps old. The
// v_permlane_idx_gen results are printed (DUMP_IDX=1) and not checked. WGS workgroups (default 1).
static int run_permlane(Kernel& k, const char* kn) {
    const int wgs = getenv("WGS") ? atoi(getenv("WGS")) : 1;
    const uint32_t mask = getenv("MASK") ? (uint32_t)strtoul(getenv("MASK"), nullptr, 0) : 0xffffffffu;
    struct C { int kind; long long a; uint32_t b; int fi, bc; };
    const C cs[] = {
        {0, 0x76543210u, 4275878552u, 0, 0}, {0, 0x89abcdefu, 19088743u, 0, 0}, {0, 0x13579bdfu, 38177486u, 1, 1},
        {1, 0x76543210u, 4275878552u, 0, 0}, {1, 0x89abcdefu, 19088743u, 1, 0}, {2, -1, 0u, 0, 0}, {3, -1, 0u, 0, 1},
        {4, 3, 4u, 0, 0}, {4, 5, 8u, 0, 0}, {4, 7, 16u, 0, 0}, {4, 0, 32u, 0, 0},
        {5, 1, 4u, 0, 0}, {5, 3, 8u, 0, 0}, {5, 5, 16u, 0, 0}, {5, 1, 32u, 0, 0},
        {6, 1, 4u, 0, 0}, {6, 3, 8u, 0, 0}, {6, 5, 16u, 0, 0}, {6, 1, 32u, 0, 0},
        {7, 1, 4u, 0, 0}, {7, 5, 8u, 0, 0}, {7, 3, 16u, 0, 0}, {7, 16, 32u, 0, 0},
        {4, 1, 2u, 0, 0}, {5, 7, 8u, 0, 0}, {7, 1, 2u, 0, 0},
    };
    const int n = sizeof(cs) / sizeof(cs[0]), stride = n + 2;
    uint32_t* dout;
    HC(hipMalloc(&dout, (size_t)wgs * stride * 32 * 4));
    HC(hipMemset(dout, 0, (size_t)wgs * stride * 32 * 4));
    std::vector<uint8_t> args(8, 0);
    put64(args, 0, dout);
    k.launch(wgs, 32, args, kn);
    std::vector<uint32_t> o((size_t)wgs * stride * 32);
    HC(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
    long bad = 0;
    for (int w = 0; w < wgs; ++w) {
        for (int c = 0; c < n; ++c) {
            const C& cc = cs[c];
            for (int l = 0; l < 32; ++l) {
                const uint32_t old = 0xdead0000u + l;
                auto x = [&](int lane) { return (uint32_t)((w << 16) + 1000 + lane); };
                int s;
                const int i = l & 15, g = l & 16;
                const uint64_t sel64 = ((uint64_t)cc.b << 32) | (uint32_t)cc.a;
                switch (cc.kind) {
                    case 0: s = g + (int)((sel64 >> (4 * i)) & 15); break;
                    case 1: s = (g ^ 16) + (int)((sel64 >> (4 * i)) & 15); break;
                    case 2: s = g + ((3 * l) & 15); break;
                    case 3: s = (g ^ 16) + ((3 * l) & 15); break;
                    default: {
                        const int size = (int)cc.b, gb = l & ~(size - 1), j = l & (size - 1), a = (int)cc.a;
                        if (cc.kind == 4) s = gb + (a % size);
                        // UP / DOWN (measured on MI450): a lane whose source would leave the group reads its own value
                        else if (cc.kind == 5) s = j >= a ? l - a : l;
                        else if (cc.kind == 6) s = j + a < size ? l + a : l;
                        else s = gb + ((j ^ a) & (size - 1));
                    }
                }
                uint32_t ref;
                if (!((mask >> l) & 1)) ref = old;
                // (the group forms always read the source lane: they ignore EXEC for reads)
                else if (!((mask >> s) & 1) && !cc.fi && cc.kind < 4) ref = cc.bc ? 0u : old;
                else ref = x(s);
                const uint32_t got = o[((size_t)w * stride + c) * 32 + l];
                if (got != ref) { if (bad < 16) printf("  wg %d case %d lane %d: got %#x ref %#x (src lane %d)\n", w, c, l, got, ref, s); ++bad; }
            }
        }
        if (w == 0 && getenv("DUMP_IDX"))
            for (int j = 0; j < 2; ++j) {
                printf("idx_gen a=%d:", j ? 5 : 1);
                for (int l = 0; l < 32; ++l) printf(" %u", o[((size_t)w * stride + n + j) * 32 + l]);
                printf("\n");
            }
    }
    printf("permlane: wgs=%d mask=%#x bad=%ld %s\n", wgs, mask, bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    return bad == 0 ? 0 : 3;
}
// probe_g1250_dpp: every DPP control against this CPU model (wave32: rows of 16 lanes, banks of 4 lanes in a row; a lane whose source is
// invalid keeps old unless bound_ctrl (then 0); a lane in a disabled row / bank is not written: old). WGS workgroups (default 64).
static int run_dpp(Kernel& k, const char* kn) {
    const int wgs = getenv("WGS") ? atoi(getenv("WGS")) : 64;
    const uint32_t OLD = 0xdeadbeefu;
    uint32_t* dout;
    HC(hipMalloc(&dout, (size_t)wgs * 20 * 32 * 4));
    HC(hipMemset(dout, 0, (size_t)wgs * 20 * 32 * 4));
    std::vector<uint8_t> args(8, 0);
    put64(args, 0, dout);
    k.launch(wgs, 32, args, kn);
    std::vector<uint32_t> o((size_t)wgs * 20 * 32);
    HC(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
    // control k: source lane within the row (-1 invalid) for row lane i (0..15) or wave lane l
    auto src = [](int c, int l) -> int {
        int r = l & ~15, i = l & 15;
        auto inrow = [&](int j) { return (j >= 0 && j < 16) ? r + j : -1; };
        switch (c) {
            case 0: { const int p[4] = {1, 0, 3, 2}; return (l & ~3) | p[l & 3]; }
            case 1: { const int p[4] = {3, 2, 1, 0}; return (l & ~3) | p[l & 3]; }
            case 2: return inrow(i + 1);
            case 3: return inrow(i + 3);
            case 4: return inrow(i - 1);
            case 5: return inrow(i - 15);
            case 6: return r + ((i - 1) & 15);
            case 7: return r + ((i - 4) & 15);
            case 8: return r + 15 - i;
            case 9: return r + (i & 8) + 7 - (i & 7);
            case 10: return r + 0;
            case 11: return r + 5;
            case 12: return r + (i ^ 1);
            case 13: return r + (i ^ 15);
            case 14: { const int s[8] = {1, 0, 3, 2, 5, 4, 7, 6}; return (l & ~7) + s[l & 7]; }
            case 15: { const int s[8] = {7, 6, 5, 4, 3, 2, 1, 0}; return (l & ~7) + s[l & 7]; }
            case 16: return inrow(i + 1);
            case 17: { const int p[4] = {1, 0, 3, 2}; return (l & ~3) | p[l & 3]; }
            case 18: return inrow(i - 1);
            default: return r + (i ^ 1);
        }
    };
    long bad = 0;
    for (int w = 0; w < wgs; ++w)
        for (int c = 0; c < 20; ++c)
            for (int l = 0; l < 32; ++l) {
                auto x = [&](int lane) { return (uint32_t)((w << 8) + 3 * lane + 1); };
                const int row = l >> 4, bank = (l & 15) >> 2;
                bool written = true;
                if (c == 16 && !((0x2 >> row) & 1)) written = false;
                if (c == 17 && !((0x5 >> bank) & 1)) written = false;
                const int s = src(c, l);
                uint32_t ref;
                if (!written) ref = OLD;
                else if (s < 0) ref = (c == 18) ? 0u : OLD;
                else ref = (c == 19) ? x(s) + x(l) : x(s);
                const uint32_t got = o[((size_t)w * 20 + c) * 32 + l];
                if (got != ref) { if (bad < 12) printf("  wg %d ctrl %d lane %d: got %#x ref %#x\n", w, c, l, got, ref); ++bad; }
            }
    printf("dpp: wgs=%d bad=%ld %s\n", wgs, bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_lds_fadd: 128 work-items add x = (tid + 1) * 2^-r into acc[(tid + 7 r) mod 256] (r = 0..7) with ds_add_f32 after a zeroing
// barrier; out[wg * 256 + i] = acc[i]. Every partial sum is a dyadic rational far inside f32's range: the result is exact, compared bit for bit.
// WGS workgroups (default 1024), REPS repetitions (default 5).
static int run_lds_fadd(Kernel& k, const char* kn) {
    const int wgs = getenv("WGS") ? atoi(getenv("WGS")) : 1024, reps = getenv("REPS") ? atoi(getenv("REPS")) : 5;
    std::vector<float> ref(256, 0.0f);
    for (int tid = 0; tid < 128; ++tid)
        for (int r = 0; r < 8; ++r) ref[(tid + 7 * r) & 255] += (float)(tid + 1) * (1.0f / (float)(1 << r));
    float* dout;
    HC(hipMalloc(&dout, (size_t)wgs * 1024));
    long bad = 0;
    for (int rep = 0; rep < reps; ++rep) {
        HC(hipMemset(dout, 0xff, (size_t)wgs * 1024));
        std::vector<uint8_t> args(8, 0);
        put64(args, 0, dout);
        k.launch(wgs, 128, args, kn);
        std::vector<float> o((size_t)wgs * 256);
        HC(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
        for (size_t x = 0; x < o.size(); ++x)
            if (o[x] != ref[x & 255]) { if (bad < 8) printf("  wg %zu i %zu: got %g ref %g\n", x / 256, x & 255, o[x], ref[x & 255]); ++bad; }
    }
    printf("lds_fadd: wgs=%d reps=%d bad=%ld %s\n", wgs, reps, bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_barriers: named barrier 1 (init / join / signal / wait / leave) then the cluster barrier, out[tid] = tid. Launched ONLY as a
// cluster (CLUSTER, default 2 workgroups along x; a cluster barrier needs a cluster id), GRID workgroups (default 64), 20 repetitions.
static int run_barriers(Kernel& k, const char* kn) {
    const unsigned cx = getenv("CLUSTER") ? atoi(getenv("CLUSTER")) : 2;
    const unsigned grid = getenv("GRID") ? atoi(getenv("GRID")) : 64;
    uint32_t* dout;
    HC(hipMalloc(&dout, 128 * 4));
    std::vector<uint8_t> args(8, 0);
    put64(args, 0, dout);
    long bad = 0;
    for (int rep = 0; rep < 20; ++rep) {
        HC(hipMemset(dout, 0xee, 128 * 4));
        k.launch_cluster(grid, 128, cx, args, kn);
        std::vector<uint32_t> o(128);
        HC(hipMemcpy(o.data(), dout, 128 * 4, hipMemcpyDeviceToHost));
        for (int t = 0; t < 128; ++t) bad += o[t] != (uint32_t)t;
    }
    printf("barriers: 20 x %u workgroups in clusters of %u x 4 waves: bad=%ld %s\n", grid, cx, bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"grid\":%u,\"cluster\":%u},\"bad\":%ld,\"ok\":%s}\n", kn, grid, cx, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_wide: 320 values per lane (one wave), all live at once (more than 256 VGPRs: the MSB window), summed in reverse order.
static int run_wide(Kernel& k, const char* kn) {
    const int N = 320;
    std::vector<float> src(N * 32);
    for (int i = 0; i < N * 32; ++i) src[i] = (float)((i * 7919) % 1000) * 0.125f - 60.f;
    float* dsrc;
    float* dout;
    HC(hipMalloc(&dsrc, src.size() * 4));
    HC(hipMalloc(&dout, 32 * 4));
    HC(hipMemcpy(dsrc, src.data(), src.size() * 4, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0xee, 32 * 4));
    std::vector<uint8_t> args(16, 0);
    put64(args, 0, dsrc);
    put64(args, 8, dout);
    k.launch(1, 32, args, kn);
    std::vector<float> o(32);
    HC(hipMemcpy(o.data(), dout, 32 * 4, hipMemcpyDeviceToHost));
    long bad = 0;
    for (int l = 0; l < 32; ++l) {
        float acc = src[(N - 1) * 32 + l];
        for (int kk = N - 2; kk >= 0; --kk) acc = acc + src[kk * 32 + l];
        bad += o[l] != acc;
    }
    printf("wide: %d live values per lane: out[0]=%g bad=%ld %s\n", N, o[0], bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"values\":%d},\"bad\":%ld,\"ok\":%s}\n", kn, N, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_lds_<bytes>: LDS round trip over the whole allocation, `wgs` workgroups of 256 at once; every out word must be 0.
static int run_lds_big(Kernel& k, const char* kn, int wgs) {
    uint32_t* dout;
    size_t n = (size_t)wgs * 256;
    HC(hipMalloc(&dout, n * 4));
    HC(hipMemset(dout, 0xee, n * 4));
    std::vector<uint8_t> args(8, 0);
    put64(args, 0, dout);
    k.launch(wgs, 256, args, kn);
    std::vector<uint32_t> o(n);
    HC(hipMemcpy(o.data(), dout, n * 4, hipMemcpyDeviceToHost));
    long bad = 0, firstbad = -1;
    for (size_t i = 0; i < n; ++i)
        if (o[i] != 0) { if (firstbad < 0) firstbad = (long)i; ++bad; }
    printf("lds_big %s: %d workgroups: bad=%ld first=%ld %s\n", kn, wgs, bad, firstbad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"shape\":{\"workgroups\":%d},\"bad\":%ld,\"ok\":%s}\n", kn, wgs, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_tdm_hi_<addr>: a 4 KiB tensor load to a high LDS address, read back; out must equal src.
static int run_tdm_hi(Kernel& k, const char* kn) {
    std::vector<uint32_t> src(1024);
    for (int i = 0; i < 1024; ++i) src[i] = 0x5a000000u ^ (uint32_t)(i * 2654435761u);
    uint32_t *dsrc, *dout;
    HC(hipMalloc(&dsrc, 4096));
    HC(hipMalloc(&dout, 4096));
    HC(hipMemcpy(dsrc, src.data(), 4096, hipMemcpyHostToDevice));
    HC(hipMemset(dout, 0xee, 4096));
    std::vector<uint8_t> args(16, 0);
    put64(args, 0, dsrc);
    put64(args, 8, dout);
    k.launch(1, 256, args, kn);
    std::vector<uint32_t> o(1024);
    HC(hipMemcpy(o.data(), dout, 4096, hipMemcpyDeviceToHost));
    long bad = 0, first = -1;
    for (int i = 0; i < 1024; ++i) if (o[i] != src[i]) { if (first < 0) first = i; ++bad; }
    printf("tdm_hi %s: bad=%ld first=%ld %s\n", kn, bad, first, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    printf("{\"host\":\"g1250_probe\",\"kernel\":\"%s\",\"bad\":%ld,\"ok\":%s}\n", kn, bad, bad == 0 ? "true" : "false");
    return bad == 0 ? 0 : 3;
}

// probe_g1250_trans: run the kernel `runs` times x `wgs` workgroups (contention) and write the bits of workgroup 0's outputs of every run to
// `dump` (the script compares the v_nop and the no-v_nop builds); also checks that every run and every workgroup agree with run 0 / wg 0.
static int run_trans(Kernel& k, const char* kn, int runs, int wgs, const char* dump) {
    size_t n = 512 * 8;
    uint32_t* dout;
    HC(hipMalloc(&dout, (size_t)wgs * n * 4));
    std::vector<uint8_t> args(8, 0);
    long bad = 0;
    std::vector<uint32_t> ref;
    FILE* fd = fopen(dump, "wb");
    for (int r = 0; r < runs; ++r) {
        HC(hipMemset(dout, 0xee, (size_t)wgs * n * 4));
        put64(args, 0, dout);
        k.launch(wgs, 512, args, kn);
        std::vector<uint32_t> o((size_t)wgs * n);
        HC(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
        if (r == 0) ref.assign(o.begin(), o.begin() + n);
        for (size_t i = 0; i < o.size(); ++i) bad += o[i] != ref[i % n];
        fwrite(o.data(), 4, n, fd);
    }
    fclose(fd);
    printf("trans %s: %d runs x %d workgroups: disagreements=%ld %s\n", kn, runs, wgs, bad, bad == 0 ? "PROBE_OK" : "PROBE_FAIL");
    return bad == 0 ? 0 : 3;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s add|math|tdm <co> [kernel] [arg]\n", argv[0]); return 1; }
    const std::string mode = argv[1];
    const char* kn = argc > 3 ? argv[3] : (mode == "add" ? "probe_g1250_add" : mode == "math" ? "probe_g1250_math" : mode == "select" ? "probe_g1250_select" : mode == "buffer" ? "probe_g1250_buffer" : mode == "tdm2d" ? "probe_g1250_tdm2d" : mode == "tdm_ring" ? "probe_g1250_tdm_ring" : mode == "named" ? "probe_g1250_named" : mode == "wide" ? "probe_g1250_wide" : mode == "lds_big" ? (argc > 3 ? argv[3] : "probe_g1250_lds_86784") : "probe_g1250_tdm");
    if (const char* w = getenv("PROBE_WATCHDOG_S")) g_limit = atof(w);
    hipDeviceProp_t prop;
    HC(hipGetDeviceProperties(&prop, 0));
    printf("device: %s  arch: %s  CUs: %d  wave: %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.warpSize);
    hipModule_t mod;
    Kernel k;
    HC(hipModuleLoad(&mod, argv[2]));
    HC(hipModuleGetFunction(&k.fn, mod, kn));
    const int arg = argc > 4 ? atoi(argv[4]) : -1;
    if (mode == "varscale") return run_varscale(k, kn);
    if (mode == "lds_fadd") return run_lds_fadd(k, kn);
    if (mode == "dpp") return run_dpp(k, kn);
    if (mode == "permlane") return run_permlane(k, kn);
    if (mode == "add") return run_add(k, kn, arg > 0 ? arg : 4096);
    if (mode == "math") return run_math(k, kn);
    if (mode == "tdm") return run_tdm(k, kn, arg > 0 ? arg : 64);
    if (mode == "select") return run_select(k, kn);
    if (mode == "buffer") return run_buffer(k, kn);
    if (mode == "tdm2d") return run_tdm2d(k, kn);
    if (mode == "tdm_ring") return run_tdm_ring(k, kn);
    if (mode == "named") return run_named(k, kn);
    if (mode == "barriers") return run_barriers(k, kn);
    if (mode == "tr8") return run_tr8(k, kn);
    if (mode == "wide") return run_wide(k, kn);
    if (mode == "trans") return run_trans(k, kn, atoi(argv[4]), atoi(argv[5]), argv[6]);
    if (mode == "tdm_hi") return run_tdm_hi(k, kn);
    if (mode == "lds_big") return run_lds_big(k, kn, argc > 4 ? atoi(argv[4]) : 1);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
