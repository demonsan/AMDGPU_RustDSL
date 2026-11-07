// GEMM 1 + generic epilogue chain (mlp_fp8_ws_g1q_{relu,silu}): H = fp8(act(alpha X W1^T + bias) / scale[m]), scale[m] = rowmax|.| / 448.
// Correctness vs a CPU reference and timing against mlp_fp8_ws_g1 (same GEMM, relu + fp8 pack, no bias / row scale).
// usage: gemm_epi_host <dir with mlp_fp8_ws_g1.co, mlp_fp8_ws_g1q_<act>.co> <act: relu|silu> [M] [G]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static const int K1 = 256, N1 = 256; static const float ALPHA = 0.125f;
static float e4m3(unsigned char v) { int s = v >> 7, e = (v >> 3) & 15, m = v & 7; if (e == 15 && m == 7) return NAN; float f = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6); return s ? -f : f; }
int main(int argc, char** argv) {
    std::string dir = argv[1], act = argv[2];
    int M = argc > 3 ? std::atoi(argv[3]) : 65536; int G = std::min(M / 128, argc > 4 ? std::atoi(argv[4]) : 256);
    std::srand(7);
    std::vector<unsigned char> x((size_t)M * K1), w1(N1 * K1);
    auto rnd = [] { int s = std::rand() & 1, e = 4 + std::rand() % 4, m = std::rand() & 7; return (unsigned char)((s << 7) | (e << 3) | m); };
    for (auto& v : x) v = rnd(); for (auto& v : w1) v = rnd();
    std::vector<float> bias(N1); for (auto& v : bias) v = (std::rand() % 2001 - 1000) / 1000.0f;
    std::vector<unsigned char> img(256 * 272, 0); for (int r = 0; r < 256; ++r) std::memcpy(&img[r * 272], &w1[r * 256], 256);
    unsigned char *dx, *dw, *dh, *dh0; float *db, *ds;
    CK(hipMalloc(&dx, x.size())); CK(hipMalloc(&dw, img.size())); CK(hipMalloc(&dh, (size_t)M * N1)); CK(hipMalloc(&dh0, (size_t)M * N1)); CK(hipMalloc(&db, N1 * 4)); CK(hipMalloc(&ds, (size_t)M * 4));
    CK(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dw, img.data(), img.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(db, bias.data(), N1 * 4, hipMemcpyHostToDevice));
    hipModule_t mq, m1; hipFunction_t fq, f1;
    std::string kq = "mlp_fp8_ws_g1q_" + act;
    CK(hipModuleLoad(&mq, (dir + "/" + kq + ".co").c_str())); CK(hipModuleGetFunction(&fq, mq, kq.c_str()));
    CK(hipModuleLoad(&m1, (dir + "/mlp_fp8_ws_g1.co").c_str())); CK(hipModuleGetFunction(&f1, m1, "mlp_fp8_ws_g1"));
    struct __attribute__((packed)) { void* in; void* w1; void* w2; void* out; int m; int g; void* bias; void* scale; } aq{dx, dw, dw, dh, M, G, db, ds};
    struct __attribute__((packed)) { void* in; void* w1; void* w2; void* out; int m; int g; } a1{dx, dw, dw, dh0, M, G};
    size_t sq = sizeof(aq), s1 = sizeof(a1);
    void* cq[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &aq, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sq, HIP_LAUNCH_PARAM_END};
    void* c1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &s1, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fq, G, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cq)); CK(hipDeviceSynchronize());
    std::vector<unsigned char> h((size_t)M * N1); std::vector<float> sc(M);
    CK(hipMemcpy(h.data(), dh, h.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(sc.data(), ds, (size_t)M * 4, hipMemcpyDeviceToHost));
    long bad = 0, cnt = 0; double maxrel = 0, sumerr = 0;
    std::vector<float> xf(K1), y(N1);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K1; ++k) xf[k] = e4m3(x[(size_t)m * K1 + k]);
        float amax = 0;
        for (int n = 0; n < N1; ++n) {
            double acc = 0; for (int k = 0; k < K1; ++k) acc += (double)xf[k] * e4m3(w1[n * K1 + k]);
            float z = ALPHA * (float)acc + bias[n];
            y[n] = act == "relu" ? std::fmax(z, 0.0f) : z / (1.0f + std::exp(-z));
            amax = std::fmax(amax, std::fabs(y[n]));
        }
        float sref = std::fmax(amax, 1e-12f) / 448.0f;
        maxrel = std::fmax(maxrel, std::fabs(sc[m] - sref) / sref);
        for (int n = 0; n < N1; ++n) {
            float d = e4m3(h[(size_t)m * N1 + n]) * sc[m]; float e = std::fabs(d - y[n]);
            float tol = std::fabs(y[n]) / 16.0f * 1.02f + sref * std::ldexp(1.0f, -8);
            if (e > tol) ++bad; sumerr += e; ++cnt;
        }
    }
    std::printf("check %s: M=%d bad=%ld (of %ld) mean_abs_err=%.3e max_rel_scale_err=%.3e %s\n", kq.c_str(), M, bad, cnt, sumerr / cnt, maxrel, (bad == 0 && maxrel < 1e-3) ? "EPI_OK" : "EPI_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    auto timeit = [&](hipFunction_t f, void** c) { for (int i = 0; i < 10; ++i) CK(hipModuleLaunchKernel(f, G, 1, 1, 256, 1, 1, 0, nullptr, nullptr, c)); CK(hipEventRecord(a)); for (int i = 0; i < 50; ++i) CK(hipModuleLaunchKernel(f, G, 1, 1, 256, 1, 1, 0, nullptr, nullptr, c)); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms / 50; };
    std::vector<float> tq, t1;
    for (int r = 0; r < 7; ++r) { tq.push_back(timeit(fq, cq)); t1.push_back(timeit(f1, c1)); }
    std::sort(tq.begin(), tq.end()); std::sort(t1.begin(), t1.end());
    double flops = 2.0 * M * N1 * K1;
    std::printf("timing (median of 7): g1q_%s %.4f ms (%.1f TF/s)  g1 (relu + pack, no bias / row scale) %.4f ms (%.1f TF/s)  ratio %.3f\n", act.c_str(), tq[3], flops / tq[3] / 1e9, t1[3], flops / t1[3] / 1e9, t1[3] / tq[3]);
    return 0;
}