// GPU harness for `attention_fwd`.
//
// Loads the code object produced by `amdc`, runs it on gfx950 and compares
// against a straightforward CPU reference. Shapes are compile-time constants of
// the kernel: D = 64, 64 query rows per block, S a multiple of 64.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

struct Args {
    void* q;
    void* k;
    void* v;
    void* o;
    int seqlen_k;
    float scale;
    int nhead;
    int seqlen_q;
};

static_assert(sizeof(Args) == 48, "kernarg layout must match attention_fwd");

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* co = argc > 1 ? argv[1] : "build/attention.co";
    auto envi = [](const char* name, int fallback) {
        const char* v = std::getenv(name);
        return v ? std::atoi(v) : fallback;
    };
    const int B = envi("ATTN_B", 1);
    const int H = envi("ATTN_H", 2);
    const int S = envi("ATTN_S", 128);  // multiple of 64
    const int D = 64;
    const char* s_env = std::getenv("ATTN_SCALE");
    const float scale = s_env ? (float)std::atof(s_env) : 1.0f / std::sqrt((float)D);

    std::printf("[harness] shape B=%d H=%d S=%d D=%d scale=%.4f\n", B, H, S, D, scale);

    const size_t elems = (size_t)B * H * S * D;
    std::vector<__half> hq(elems), hk(elems), hv(elems);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elems; ++i) {
        hq[i] = __float2half(dist(rng));
        hk[i] = __float2half(dist(rng));
        hv[i] = __float2half(dist(rng));
    }

    __half* dq = nullptr;
    __half* dk = nullptr;
    __half* dv = nullptr;
    __half* dout = nullptr;
    CK(hipMalloc((void**)&dq, elems * sizeof(__half)));
    CK(hipMalloc((void**)&dk, elems * sizeof(__half)));
    CK(hipMalloc((void**)&dv, elems * sizeof(__half)));
    CK(hipMalloc((void**)&dout, elems * sizeof(__half)));
    CK(hipMemcpy(dq, hq.data(), elems * sizeof(__half), hipMemcpyHostToDevice));
    CK(hipMemcpy(dk, hk.data(), elems * sizeof(__half), hipMemcpyHostToDevice));
    CK(hipMemcpy(dv, hv.data(), elems * sizeof(__half), hipMemcpyHostToDevice));
    CK(hipMemset(dout, 0, elems * sizeof(__half)));
    std::printf("[harness] buffers ready\n");

    hipModule_t mod;
    hipFunction_t fn;
    CK(hipModuleLoad(&mod, co));
    std::printf("[harness] module loaded\n");
    CK(hipModuleGetFunction(&fn, mod, "attention_fwd"));
    std::printf("[harness] function resolved\n");

    // hipModuleLaunchKernel wants one pointer per kernel argument, matching the
    // metadata argument order (q, k, v, o, seqlen_k, scale, nhead, seqlen_q).
    void* a_q = dq;
    void* a_k = dk;
    void* a_v = dv;
    void* a_o = dout;
    const int SK = envi("ATTN_SK", S);
    int a_sk = SK;
    float a_scale = scale;
    int a_nhead = H;
    int a_sq = S;
    void* params[] = {&a_q, &a_k, &a_v, &a_o, &a_sk, &a_scale, &a_nhead, &a_sq};
    dim3 grid(S / 64, H, B);
    std::printf("[harness] launching grid=(%u,%u,%u)\n", grid.x, grid.y, grid.z);
    CK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, 64, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    std::printf("[harness] kernel finished\n");

    std::vector<__half> got(elems);
    CK(hipMemcpy(got.data(), dout, elems * sizeof(__half), hipMemcpyDeviceToHost));

    // --- CPU reference -------------------------------------------------
    std::vector<float> ref(elems, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const size_t base = ((size_t)b * H + h) * S * D;
            for (int qi = 0; qi < S; ++qi) {
                std::vector<float> score(SK);
                float mx = -INFINITY;
                for (int s = 0; s < SK; ++s) {
                    float acc = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        acc += __half2float(hq[base + (size_t)qi * D + d]) *
                               __half2float(hk[base + (size_t)s * D + d]);
                    }
                    score[s] = acc * scale;
                    mx = std::max(mx, score[s]);
                }
                float sum = 0.0f;
                for (int s = 0; s < SK; ++s) {
                    score[s] = std::exp(score[s] - mx);
                    sum += score[s];
                }
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int s = 0; s < SK; ++s) {
                        acc += score[s] / sum * __half2float(hv[base + (size_t)s * D + d]);
                    }
                    ref[base + (size_t)qi * D + d] = acc;
                }
            }
        }
    }

    // --- compare -------------------------------------------------------
    double max_abs = 0.0;
    double sum_abs = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < elems; ++i) {
        double a = __half2float(got[i]);
        double e = ref[i];
        double d = std::fabs(a - e);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        if (d > 2e-2 + 2e-2 * std::fabs(e)) {
            if (bad < 8) {
                std::fprintf(stderr, "mismatch idx=%zu got=%.6f ref=%.6f\n", i, a, e);
            }
            ++bad;
        }
    }
    for (int i = 0; i < 4; ++i) {
        std::printf("  [0] got=%.6f ref=%.6f\n", __half2float(got[i]), ref[i]);
    }
    std::printf("max_abs_err=%.6g mean_abs_err=%.6g bad=%zu/%zu\n", max_abs,
                sum_abs / (double)elems, bad, elems);
    if (bad != 0) {
        std::printf("ATTENTION_FWD FAIL\n");
        return 1;
    }
    std::printf("ATTENTION_FWD OK\n");
    return 0;
}
