// MLA decode host (bf16, 576 / 512, one latent KV head): ours (`mla_decode_bf16_p<P>` + `attn_combine_bf16_hd512`) on a paged cache
// [num_pages, P, 576] with a block table; CPU reference for every (sequence, head).
//
//   mla_host <mla.co> <combine.co> B H CTX P S
//   env: ITERS (50), WARMUP (10), SEED (1); context lengths random in [CTX / 2, CTX]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)
static int envi(const char* n, int d) { const char* v = getenv(n); return v ? atoi(v) : d; }
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t tobf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
template <class T> static T* dev(const std::vector<T>& h) { T* d; HC(hipMalloc(&d, h.size() * sizeof(T) + 256)); HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s mla.co combine.co B H CTX P S\n", argv[0]); return 1; }
    const int B = atoi(argv[3]), H = atoi(argv[4]), CTX = atoi(argv[5]), P = atoi(argv[6]), S = atoi(argv[7]);
    const int DK = 576, DV = 512, iters = envi("ITERS", 50), warm = envi("WARMUP", 10);
    std::mt19937 rng(envi("SEED", 1));
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::vector<int> cl(B);
    for (auto& c : cl) c = CTX - (int)(rng() % (CTX / 2 + 1));
    const int maxp = (CTX + P - 1) / P;
    int npages = 3;
    for (int c : cl) npages += (c + P - 1) / P;
    std::vector<int> perm(npages);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> bt((size_t)B * maxp, 0);
    int nx = 0;
    for (int b = 0; b < B; b++)
        for (int p = 0; p < (cl[b] + P - 1) / P; p++) bt[(size_t)b * maxp + p] = perm[nx++];
    std::vector<uint16_t> kv((size_t)npages * P * DK), q((size_t)B * H * DK);
    for (auto& x : kv) x = tobf(N(rng));
    for (auto& x : q) x = tobf(N(rng));
    auto ki = [&](int b, int key, int d) { return ((size_t)bt[(size_t)b * maxp + key / P] * P + key % P) * DK + d; };
    const float scale = 1.0f / std::sqrt((float)DK);
    const int NS = S;
    uint16_t* dout;
    void *dpo, *dpl;
    HC(hipMalloc(&dout, (size_t)B * H * DV * 2));
    HC(hipMalloc(&dpo, (size_t)B * H * NS * DV * 4)); HC(hipMalloc(&dpl, (size_t)B * H * NS * 4));
    void *dq = dev(q), *dkv = dev(kv), *dbt = dev(bt), *dcl = dev(cl);
    hipModule_t m1, m2;
    hipFunction_t fn, cfn;
    HC(hipModuleLoad(&m1, argv[1]));
    std::string kn = "mla_decode_bf16_p" + std::to_string(P);
    HC(hipModuleGetFunction(&fn, m1, kn.c_str()));
    HC(hipModuleLoad(&m2, argv[2]));
    HC(hipModuleGetFunction(&cfn, m2, "attn_combine_bf16_hd512"));
    std::vector<uint8_t> a(16 * 11, 0), ca(16 * 5, 0);
    auto Pp = [&](int i, const void* p) { memcpy(&a[16 * i], &p, 8); };
    auto U = [&](int i, int v) { memcpy(&a[16 * i], &v, 4); };
    Pp(0, dpo); Pp(1, dq); Pp(2, dkv); Pp(3, dbt); Pp(4, dcl); memcpy(&a[80], &scale, 4); U(6, H); U(7, maxp); U(8, npages); U(9, S); Pp(10, dpl);
    memcpy(&ca[0], &dout, 8); memcpy(&ca[16], &dpo, 8); memcpy(&ca[32], &dpl, 8); memcpy(&ca[48], &H, 4); memcpy(&ca[64], &NS, 4);
    size_t asz = a.size(), csz = ca.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
    void* ccfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ca.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &csz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() {
        HC(hipModuleLaunchKernel(fn, (H + 31) / 32, B, S, 256, 1, 1, 0, nullptr, nullptr, cfg));
        HC(hipModuleLaunchKernel(cfn, H, B, 1, 512, 1, 1, 0, nullptr, nullptr, ccfg));
    };
    launch();
    HC(hipDeviceSynchronize());
    std::vector<uint16_t> out((size_t)B * H * DV);
    HC(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0;
    double maxe = 0;
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            std::vector<double> s(cl[b]);
            double m = -1e300;
            for (int k = 0; k < cl[b]; k++) {
                double acc = 0;
                for (int d = 0; d < DK; d++) acc += (double)bf(q[((size_t)b * H + h) * DK + d]) * bf(kv[ki(b, k, d)]);
                s[k] = acc * scale;
                m = std::max(m, s[k]);
            }
            double l = 0;
            std::vector<double> o(DV, 0.0);
            for (int k = 0; k < cl[b]; k++) {
                double p = std::exp(s[k] - m);
                l += p;
                for (int d = 0; d < DV; d++) o[d] += p * bf(kv[ki(b, k, d)]);
            }
            for (int d = 0; d < DV; d++) {
                double ref = cl[b] ? o[d] / l : 0, got = bf(out[((size_t)b * H + h) * DV + d]);
                double e = std::fabs(got - ref);
                maxe = std::max(maxe, e);
                if (e > 1e-2 * std::fabs(ref) + 2e-2) {
                    if (bad < envi("SHOW_BAD", 0)) printf("  bad b=%d h=%d d=%d got=%g ref=%g\n", b, h, d, got, ref);
                    bad++;
                }
            }
        }
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    for (int i = 0; i < warm; i++) launch();
    HC(hipEventRecord(e0));
    for (int i = 0; i < iters; i++) launch();
    HC(hipEventRecord(e1));
    HC(hipEventSynchronize(e1));
    float ms; HC(hipEventElapsedTime(&ms, e0, e1));
    double us = 1e3 * ms / iters, bytes = 0;
    for (int c : cl) bytes += (double)c * DK * 2;
    printf("ours %s B=%d H=%d CTX=%d P=%d S=%d: %.2f us %.0f GB/s (KV) bad=%ld maxerr=%.3e %s\n", kn.c_str(), B, H, CTX, P, S, us, bytes / (us * 1e-6) / 1e9, bad, maxe,
           bad ? "MLA_FAIL" : "MLA_OK");
    return bad ? 3 : 0;
}
