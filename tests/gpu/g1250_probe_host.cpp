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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s add|math|tdm <co> [kernel] [arg]\n", argv[0]); return 1; }
    const std::string mode = argv[1];
    const char* kn = argc > 3 ? argv[3] : (mode == "add" ? "probe_g1250_add" : mode == "math" ? "probe_g1250_math" : mode == "select" ? "probe_g1250_select" : mode == "buffer" ? "probe_g1250_buffer" : mode == "tdm2d" ? "probe_g1250_tdm2d" : mode == "tdm_ring" ? "probe_g1250_tdm_ring" : mode == "named" ? "probe_g1250_named" : "probe_g1250_tdm");
    if (const char* w = getenv("PROBE_WATCHDOG_S")) g_limit = atof(w);
    hipDeviceProp_t prop;
    HC(hipGetDeviceProperties(&prop, 0));
    printf("device: %s  arch: %s  CUs: %d  wave: %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.warpSize);
    hipModule_t mod;
    Kernel k;
    HC(hipModuleLoad(&mod, argv[2]));
    HC(hipModuleGetFunction(&k.fn, mod, kn));
    const int arg = argc > 4 ? atoi(argv[4]) : -1;
    if (mode == "add") return run_add(k, kn, arg > 0 ? arg : 4096);
    if (mode == "math") return run_math(k, kn);
    if (mode == "tdm") return run_tdm(k, kn, arg > 0 ? arg : 64);
    if (mode == "select") return run_select(k, kn);
    if (mode == "buffer") return run_buffer(k, kn);
    if (mode == "tdm2d") return run_tdm2d(k, kn);
    if (mode == "tdm_ring") return run_tdm_ring(k, kn);
    if (mode == "named") return run_named(k, kn);
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
