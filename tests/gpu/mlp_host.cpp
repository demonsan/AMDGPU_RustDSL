// Fused MLP chain vs the unfused pair (crates/amdgpu-kernels/src/mlp_fp8.rs): Y = fp8(relu(alpha X W1)) W2.
// usage: mlp_host <dir with mlp_fp8_{fused,g1,g2}.co> [M]   prints accuracy vs a host reference and interleaved timings.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static const int K1 = 256, N1 = 256, N2 = 256; static const float ALPHA = 0.125f;
static float e4m3(unsigned char v) {  // OCP E4M3FN
    int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    if (e == 15 && m == 7) return NAN;
    float f = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
    return s ? -f : f;
}
static unsigned char to_e4m3(float x) {  // round to nearest even over the representable set (small range here)
    unsigned char best = 0; float bd = INFINITY;
    for (int c = 0; c < 256; ++c) { if ((c & 0x7f) == 0x7f) continue; float d = std::fabs(e4m3((unsigned char)c) - x); if (d < bd || (d == bd && (c & 1) == 0 && (best & 1))) { bd = d; best = (unsigned char)c; } }
    return best;
}
static float bf16(unsigned short v) { unsigned u = (unsigned)v << 16; float f; std::memcpy(&f, &u, 4); return f; }
static int sigma(int k) { return 32 * (k / 32) + 8 * ((k % 16) / 4) + 4 * ((k % 32) / 16) + k % 4; }
int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "build";
    int M = argc > 2 ? std::atoi(argv[2]) : 65536;
    int iters = 50, rounds = 7;
    std::srand(1);
    std::vector<unsigned char> x((size_t)M * K1), w1(N1 * K1), w2n(N2 * N1), w2p(N2 * N1);
    // small fp8 values: exponents around 2^-2 .. 2^0
    auto rnd = [] { int s = std::rand() & 1, e = 4 + std::rand() % 4, m = std::rand() & 7; return (unsigned char)((s << 7) | (e << 3) | m); };
    for (auto& v : x) v = rnd();
    for (auto& v : w1) v = rnd();
    std::vector<unsigned char> w2(N1 * N2);  // logical W2[n1][n2]
    for (auto& v : w2) v = rnd();
    for (int n2 = 0; n2 < N2; ++n2) for (int n1 = 0; n1 < N1; ++n1) {
        w2n[n2 * N1 + n1] = w2[n1 * N2 + n2];
        int jb = n1 / 64, k = n1 % 64;
        w2p[n2 * N1 + n1] = w2[(64 * jb + sigma(k)) * N2 + n2];
    }
    unsigned char *dx, *dw1, *dw2n, *dw2p, *dh; unsigned short *dyf, *dyu;
    CK(hipMalloc(&dx, x.size())); CK(hipMalloc(&dw1, w1.size())); CK(hipMalloc(&dw2n, w2n.size())); CK(hipMalloc(&dw2p, w2p.size()));
    CK(hipMalloc(&dh, (size_t)M * N1)); CK(hipMalloc(&dyf, (size_t)M * N2 * 2)); CK(hipMalloc(&dyu, (size_t)M * N2 * 2));
    CK(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dw1, w1.data(), w1.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dw2n, w2n.data(), w2n.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dw2p, w2p.data(), w2p.size(), hipMemcpyHostToDevice));
    hipModule_t mf, m1, m2; hipFunction_t ff, f1, f2;
    CK(hipModuleLoad(&mf, (dir + "/mlp_fp8_fused.co").c_str())); CK(hipModuleGetFunction(&ff, mf, "mlp_fp8_fused"));
    CK(hipModuleLoad(&m1, (dir + "/mlp_fp8_g1.co").c_str())); CK(hipModuleGetFunction(&f1, m1, "mlp_fp8_g1"));
    CK(hipModuleLoad(&m2, (dir + "/mlp_fp8_g2.co").c_str())); CK(hipModuleGetFunction(&f2, m2, "mlp_fp8_g2"));
    struct A { void *in, *w1, *w2, *out; int m; int pad; };
    A af{dx, dw1, dw2p, dyf, M, 0}, a1{dx, dw1, dw2n, dh, M, 0}, a2{dh, dw1, dw2n, dyu, M, 0};
    size_t sz = sizeof(A);
    auto launch = [&](hipFunction_t f, A& a) { void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END}; CK(hipModuleLaunchKernel(f, M / 128, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); };
    launch(ff, af); launch(f1, a1); launch(f2, a2); CK(hipDeviceSynchronize());
    // accuracy on a row sample
    std::vector<unsigned short> yf((size_t)M * N2), yu((size_t)M * N2);
    CK(hipMemcpy(yf.data(), dyf, yf.size() * 2, hipMemcpyDeviceToHost)); CK(hipMemcpy(yu.data(), dyu, yu.size() * 2, hipMemcpyDeviceToHost));
    double worst_f = 0, worst_u = 0; int bad = 0, same = 0, tot = 0;
    for (int r = 0; r < M; r += std::max(1, M / 512)) {
        float hrow[N1];
        for (int n1 = 0; n1 < N1; ++n1) { double s = 0; for (int k = 0; k < K1; ++k) s += (double)e4m3(x[(size_t)r * K1 + k]) * e4m3(w1[n1 * K1 + k]); hrow[n1] = e4m3(to_e4m3(std::max((float)s * ALPHA, 0.0f))); }
        for (int n2 = 0; n2 < N2; ++n2) {
            double s = 0, sa = 0; for (int n1 = 0; n1 < N1; ++n1) { double p = (double)hrow[n1] * e4m3(w2[n1 * N2 + n2]); s += p; sa += std::fabs(p); }
            float gf = bf16(yf[(size_t)r * N2 + n2]), gu = bf16(yu[(size_t)r * N2 + n2]);
            double tol = 0.02 * sa + 1e-3;
            double ef = std::fabs(gf - s) / (sa + 1e-6), eu = std::fabs(gu - s) / (sa + 1e-6);
            worst_f = std::max(worst_f, ef); worst_u = std::max(worst_u, eu);
            if (std::fabs(gf - s) > tol || std::fabs(gu - s) > tol) ++bad;
            same += (yf[(size_t)r * N2 + n2] == yu[(size_t)r * N2 + n2]); ++tot;
        }
    }
    std::printf("M=%d: checked %d, bad %d, worst rel (|err|/sum|p|) fused %.3e unfused %.3e, fused==unfused bitwise %d/%d -> %s\n", M, tot, bad, worst_f, worst_u, same, tot, bad ? "FAIL" : "OK");
    // interleaved timing
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    std::vector<float> tf, tu;
    for (int rr = 0; rr < rounds; ++rr) {
        for (int w = 0; w < 2; ++w) {
            CK(hipEventRecord(e0)); for (int i = 0; i < iters; ++i) { if (w == 0) launch(ff, af); else { launch(f1, a1); launch(f2, a2); } } CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); (w == 0 ? tf : tu).push_back(1000.0f * ms / iters);
        }
    }
    std::sort(tf.begin(), tf.end()); std::sort(tu.begin(), tu.end());
    double flops = 2.0 * M * (double)K1 * N1 + 2.0 * M * (double)N1 * N2;
    std::printf("fused %.2f us (%.0f TF/s) | unfused pair %.2f us (%.0f TF/s) | speedup %.3fx  [medians of %d x %d]\n", tf[rounds / 2], flops / tf[rounds / 2] / 1e6, tu[rounds / 2], flops / tu[rounds / 2] / 1e6, tu[rounds / 2] / tf[rounds / 2], rounds, iters);
    return bad ? 1 : 0;
}
