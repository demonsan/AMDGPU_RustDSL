// Row kernels (batch 2, docs/op-coverage-membound.md): rmsnorm / add+rmsnorm / layernorm / softmax (bf16 rows, one row per wave): correctness against a CPU double reference and bandwidth.
// usage: rows_host <co> <kernel> <mode: rmsnorm|add_rmsnorm|layernorm|softmax> <M> <N> [iters]
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
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
template <class T> static void* up(const std::vector<T>& v) { void* d; CK(hipMalloc(&d, v.size() * sizeof(T) + 64)); CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s <co> <kernel> <rmsnorm|add_rmsnorm|layernorm|softmax> <M> <N> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; std::string mode = argv[3];
    long M = std::atol(argv[4]); long N = std::atol(argv[5]); int iters = argc > 6 ? std::atoi(argv[6]) : 50;
    const float eps = 1e-5f;
    std::mt19937 rng(4242); std::normal_distribution<float> nd(0.f, 2.f); std::uniform_real_distribution<float> ud(0.5f, 1.5f); std::normal_distribution<float> nb(0.f, 0.5f);
    std::vector<uint16_t> hx((size_t)M * N), hr(mode == "add_rmsnorm" ? (size_t)M * N : 0), hg(N), hb(N);
    for (auto& v : hx) v = f32_to_bf16(nd(rng));
    for (auto& v : hr) v = f32_to_bf16(nd(rng));
    for (auto& v : hg) v = f32_to_bf16(ud(rng));
    for (auto& v : hb) v = f32_to_bf16(nb(rng));
    void *dx = up(hx), *dr = hr.empty() ? nullptr : up(hr), *dg = up(hg), *db = up(hb), *dout, *drout = nullptr;
    CK(hipMalloc(&dout, (size_t)M * N * 2 + 64)); CK(hipMemset(dout, 0x55, (size_t)M * N * 2 + 64));
    if (mode == "add_rmsnorm") { CK(hipMalloc(&drout, (size_t)M * N * 2 + 64)); CK(hipMemset(drout, 0x55, (size_t)M * N * 2 + 64)); }
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[128]; size_t sz = 0;
    auto put = [&](const void* p, size_t n) { std::memcpy(args + sz, p, n); sz += n; };
    auto putp = [&](void* p) { put(&p, 8); };
    int32_t m32 = (int32_t)M;
    if (mode == "rmsnorm") { putp(dx); putp(dg); putp(dout); put(&m32, 4); put(&eps, 4); }
    else if (mode == "add_rmsnorm") { putp(dx); putp(dr); putp(dg); putp(dout); putp(drout); put(&m32, 4); put(&eps, 4); }
    else if (mode == "layernorm") { putp(dx); putp(dg); putp(db); putp(dout); put(&m32, 4); put(&eps, 4); }
    else { putp(dx); putp(dout); put(&m32, 4); }
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    long rpw = N < 512 ? 4 : (N < 1024 ? 2 : 1);   // rows per wave (rows_per_wave in rows/mod.rs)
    int grid = (int)((M + 4 * rpw - 1) / (4 * rpw));
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint16_t> ho((size_t)M * N + 32), hro(drout ? (size_t)M * N + 32 : 0);
    CK(hipMemcpy(ho.data(), dout, ho.size() * 2, hipMemcpyDeviceToHost));
    if (drout) CK(hipMemcpy(hro.data(), drout, hro.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0, cnt = 0, rbad = 0; double max_rel = 0;
    std::vector<double> y(N), s(N);
    for (long m = 0; m < M; ++m) {
        for (long k = 0; k < N; ++k) {
            double v = bf16_to_f32(hx[(size_t)m * N + k]);
            if (mode == "add_rmsnorm") { uint16_t rb = f32_to_bf16(bf16_to_f32(hx[(size_t)m * N + k]) + bf16_to_f32(hr[(size_t)m * N + k])); if (hro[(size_t)m * N + k] != rb) ++rbad; v = bf16_to_f32(rb); }
            s[k] = v;
        }
        if (mode == "rmsnorm" || mode == "add_rmsnorm") {
            double ss = 0; for (long k = 0; k < N; ++k) ss += s[k] * s[k];
            double inv = 1.0 / std::sqrt(ss / N + eps);
            for (long k = 0; k < N; ++k) y[k] = s[k] * inv * bf16_to_f32(hg[k]);
        } else if (mode == "layernorm") {
            double mu = 0; for (long k = 0; k < N; ++k) mu += s[k]; mu /= N;
            double var = 0; for (long k = 0; k < N; ++k) var += (s[k] - mu) * (s[k] - mu); var /= N;
            double inv = 1.0 / std::sqrt(var + eps);
            for (long k = 0; k < N; ++k) y[k] = (s[k] - mu) * inv * bf16_to_f32(hg[k]) + bf16_to_f32(hb[k]);
        } else {
            double mx = -1e300; for (long k = 0; k < N; ++k) mx = std::max(mx, s[k]);
            double sm = 0; for (long k = 0; k < N; ++k) sm += std::exp(s[k] - mx);
            for (long k = 0; k < N; ++k) y[k] = std::exp(s[k] - mx) / sm;
        }
        for (long k = 0; k < N; ++k) {
            double got = bf16_to_f32(ho[(size_t)m * N + k]);
            double err = std::fabs(got - y[k]), tol = std::fabs(y[k]) * (1.0 / 128.0) * 1.05 + 1e-6;
            if (!(err <= tol)) { if (bad < 5) std::fprintf(stderr, "bad at (%ld,%ld): got %g ref %g\n", m, k, got, y[k]); ++bad; }
            if (std::fabs(y[k]) > 1e-3) max_rel = std::fmax(max_rel, err / std::fabs(y[k]));
            ++cnt;
        }
    }
    long guard_bad = 0; for (size_t i = (size_t)M * N; i < (size_t)M * N + 32; ++i) { if (ho[i] != 0x5555) ++guard_bad; if (!hro.empty() && hro[i] != 0x5555) ++guard_bad; }
    bool ok = bad == 0 && rbad == 0 && guard_bad == 0;
    std::printf("check: %s M=%ld N=%ld bad=%ld (of %ld) res_out_bad=%ld max_rel=%.3e past_end_bad=%ld %s\n", mode.c_str(), M, N, bad, cnt, rbad, max_rel, guard_bad, ok ? "ROW_OK" : "ROW_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)M * N * 2 * (mode == "add_rmsnorm" ? 4 : 2) + (double)N * 2;
    std::printf("timing: %.4f ms/iter, %.1f GB/s (traffic %.1f MB)\n", ms, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}