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
#include <cstring>
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
    const int BLOCK = envi("ATTN_BLOCK", 64);
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
    if (std::getenv("ATTN_CONST")) {
        // q = k = 0, v = 0.5 => attention output is exactly 0.5 everywhere.
        for (size_t i = 0; i < elems; ++i) {
            hq[i] = __float2half(0.0f);
            hk[i] = __float2half(0.0f);
            hv[i] = __float2half(0.5f);
        }
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
    const char* kern_env = std::getenv("ATTN_KERNEL");
    const char* kern = kern_env ? kern_env : "attention_fwd";
    CK(hipModuleGetFunction(&fn, mod, kern));
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
    // Probe kernels take the output pointer as their only argument.
    void* probe_params[] = {&a_o};
    const bool probe_kernel = std::strcmp(kern, "probe_math") == 0 ||
                             std::strcmp(kern, "probe_mfma") == 0 || std::strcmp(kern, "probe_lds") == 0 ||
                             std::strcmp(kern, "probe_mfma_wide") == 0 ||
                             std::strcmp(kern, "probe_mfma_fp8") == 0 ||
                             std::strcmp(kern, "probe_mfma_scale") == 0 ||
                             std::strcmp(kern, "probe_roles") == 0 ||
                             std::strcmp(kern, "probe_waves2") == 0 ||
                             std::strcmp(kern, "hw_roles") == 0 ||
                             std::strcmp(kern, "hw_lds_neighbour") == 0 ||
                             std::strcmp(kern, "hw_lds_own") == 0 ||
                             std::strncmp(kern, "hw_bisect", 9) == 0 || std::strcmp(kern, "hw_store2") == 0 || std::strcmp(kern, "probe_lds_own") == 0 ||
                             std::strcmp(kern, "probe_lds_self2") == 0 ||
                             std::strcmp(kern, "probe_roles_loop") == 0 ||
                             std::strcmp(kern, "probe_fmha_datapath") == 0 ||
                             std::strcmp(kern, "probe_ab_loops") == 0 ||
                             std::strcmp(kern, "probe_nested_acc") == 0;
    void** launch_params = probe_kernel ? probe_params : params;
    dim3 grid(S / 64, H, B);
    std::printf("[harness] launching grid=(%u,%u,%u) block=%d\n", grid.x, grid.y, grid.z, BLOCK);
    CK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, (unsigned)BLOCK, 1, 1, 0, nullptr, launch_params,
                             nullptr));
    CK(hipDeviceSynchronize());
    std::printf("[harness] kernel finished\n");

    std::vector<__half> got(elems);
    CK(hipMemcpy(got.data(), dout, elems * sizeof(__half), hipMemcpyDeviceToHost));

    if (std::strcmp(kern, "probe_math") == 0) {
        float vals[3] = {0.0f, 0.0f, 0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        std::printf("  exp(1)=%.9f exp(2)=%.9f rcp(3)=%.9f\n", vals[0], vals[1], vals[2]);
        return 0;
    }

    if (std::strcmp(kern, "probe_mfma") == 0) {
        // All-ones A and B with a zero accumulator: every element of the 16x16
        // result must be exactly K = 16.0, independent of the fragment layout.
        float vals[256] = {0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        int mfma_bad = 0;
        for (int i = 0; i < 256; ++i) {
            if (std::fabs(vals[i] - 16.0f) > 1e-3f) {
                if (mfma_bad < 6) {
                    std::fprintf(stderr, "mfma mismatch lane=%d reg=%d got=%.6f\n", i / 4, i % 4,
                                 vals[i]);
                }
                ++mfma_bad;
            }
        }
        std::printf("  mfma[0..3]=%.6f %.6f %.6f %.6f bad=%d\n", vals[0], vals[1], vals[2], vals[3],
                    mfma_bad);
        std::printf(mfma_bad ? "PROBE_MFMA_FAIL\n" : "PROBE_MFMA_OK\n");
        return mfma_bad != 0;
    }

    if (std::strcmp(kern, "probe_lds") == 0) {
        // Lane i must read the value lane (i+1)%64 wrote into LDS.
        unsigned int bits[64] = {0};
        std::memcpy(bits, got.data(), sizeof(bits));
        int lds_bad = 0;
        for (int i = 0; i < 64; ++i) {
            unsigned int want = (unsigned int)((i + 1) & 63);
            if (bits[i] != want) {
                if (lds_bad < 6) {
                    std::fprintf(stderr, "lds mismatch lane=%d got=%u want=%u\n", i, bits[i], want);
                }
                ++lds_bad;
            }
        }
        std::printf("  lds[0..3]=%u %u %u %u bad=%d\n", bits[0], bits[1], bits[2], bits[3],
                    lds_bad);
        std::printf(lds_bad ? "PROBE_LDS_FAIL\n" : "PROBE_LDS_OK\n");
        return lds_bad != 0;
    }
    if (std::strcmp(kern, "probe_mfma_wide") == 0) {
        // All-ones A and B, zero accumulator, K = 8: every element must be 8.0.
        float vals[1024] = {0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        int wide_bad = 0;
        for (int i = 0; i < 1024; ++i) {
            if (std::fabs(vals[i] - 8.0f) > 1e-3f) {
                if (wide_bad < 6) {
                    std::fprintf(stderr, "wide mfma mismatch lane=%d reg=%d got=%.6f\n", i / 16,
                                 i % 16, vals[i]);
                }
                ++wide_bad;
            }
        }
        std::printf("  wide mfma[0..3]=%.6f %.6f %.6f %.6f bad=%d\n", vals[0], vals[1], vals[2],
                    vals[3], wide_bad);
        std::printf(wide_bad ? "PROBE_MFMA_WIDE_FAIL\n" : "PROBE_MFMA_WIDE_OK\n");
        return wide_bad != 0;
    }
    if (std::strcmp(kern, "probe_mfma_fp8") == 0) {
        // Lane layout: 4 floats from the 16x16x32 result (K = 32), then 16 from
        // the 32x32x16 result (K = 16).
        float vals[1280] = {0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        int fp8_bad = 0;
        for (int lane = 0; lane < 64; ++lane) {
            for (int i = 0; i < 4; ++i) {
                if (std::fabs(vals[lane * 20 + i] - 32.0f) > 1e-3f) {
                    ++fp8_bad;
                }
            }
            for (int i = 0; i < 16; ++i) {
                if (std::fabs(vals[lane * 20 + 4 + i] - 16.0f) > 1e-3f) {
                    ++fp8_bad;
                }
            }
        }
        std::printf("  fp8 mfma lane0: %.6f %.6f %.6f %.6f | %.6f %.6f bad=%d\n", vals[0],
                    vals[1], vals[2], vals[3], vals[4], vals[5], fp8_bad);
        std::printf(fp8_bad ? "PROBE_MFMA_FP8_FAIL\n" : "PROBE_MFMA_FP8_OK\n");
        return fp8_bad != 0;
    }
    if (std::strcmp(kern, "probe_mfma_scale") == 0) {
        // 64 floats per lane: four scale variants of 16 elements each.
        float vals[2048] = {0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        std::printf("  scale variants (lane0): %.6f | %.6f | %.6f | %.6f\n", vals[0], vals[16],
                    vals[32], vals[48]);
        // E8M0 127 = 2^0, 128 = 2^1, and only the low byte is used, so the four
        // variants must be 64, 128, 0, 0.
        const float want[4] = {64.0f, 128.0f, 0.0f, 0.0f};
        int scale_bad = 0;
        for (int v = 0; v < 4; ++v) {
            if (std::fabs(vals[v * 16] - want[v]) > 1e-3f) {
                std::fprintf(stderr, "scale variant %d got %.6f want %.6f\n", v, vals[v * 16],
                             want[v]);
                ++scale_bad;
            }
        }
        std::printf(scale_bad ? "PROBE_MFMA_SCALE_FAIL\n" : "PROBE_MFMA_SCALE_OK\n");
        return scale_bad != 0;
    }
    if (std::strcmp(kern, "probe_roles") == 0) {
        // The kernel stores the raw 32-bit integer it read from LDS, so compare
        // bit patterns rather than floats.
        unsigned int bits[512] = {0};
        std::memcpy(bits, got.data(), sizeof(bits));
        int roles_bad = 0;
        for (int tid = 0; tid < 512; ++tid) {
            int w = tid / 64;
            int l = tid % 64;
            int pw = (w + 4) & 7;
            unsigned int want = (unsigned int)(pw * 64 + l);
            if (bits[tid] != want) {
                if (roles_bad < 6) {
                    std::fprintf(stderr, "roles mismatch tid=%d got=%u want=%u\n", tid, bits[tid],
                                 want);
                }
                ++roles_bad;
            }
        }
        std::printf("  roles bad=%d sample=%u %u %u %u\n", roles_bad, bits[0], bits[64], bits[256],
                    bits[320]);
        std::printf(roles_bad ? "PROBE_ROLES_FAIL\n" : "PROBE_ROLES_OK\n");
        return roles_bad != 0;
    }
    if (std::strcmp(kern, "probe_waves2") == 0) {
        // Two waves: o[tid] holds the lane's own id, o[128+tid] the id of the
        // partner wave's lane. Raw integer bit patterns, not floats.
        unsigned int bits[256] = {0};
        std::memcpy(bits, got.data(), sizeof(bits));
        int own_bad = 0;
        int partner_bad = 0;
        for (int tid = 0; tid < 128; ++tid) {
            int w = tid / 64;
            int l = tid % 64;
            if (bits[tid] != (unsigned int)tid) {
                ++own_bad;
            }
            unsigned int want = (unsigned int)((w ^ 1) * 64 + l);
            if (bits[128 + tid] != want) {
                if (partner_bad < 6) {
                    std::fprintf(stderr, "waves2 mismatch tid=%d got=%u want=%u\n", tid,
                                 bits[128 + tid], want);
                }
                ++partner_bad;
            }
        }
        std::printf("  waves2 own_bad=%d partner_bad=%d own[0]=%u partner[0]=%u\n", own_bad,
                    partner_bad, bits[0], bits[128]);
        std::printf((own_bad || partner_bad) ? "PROBE_WAVES2_FAIL\n" : "PROBE_WAVES2_OK\n");
        return (own_bad || partner_bad) ? 1 : 0;
    }
    if (std::strcmp(kern, "probe_nested_acc") == 0) {
        // Expected raw integer per lane: 3 + 2*tid.
        unsigned int bits[64] = {0};
        std::memcpy(bits, got.data(), sizeof(bits));
        int na_bad = 0;
        for (int tid = 0; tid < 64; ++tid) {
            unsigned int want = (unsigned int)(3 + 2 * tid);
            if (bits[tid] != want) {
                if (na_bad < 6) {
                    std::fprintf(stderr, "nested_acc mismatch tid=%d got=%u want=%u\n", tid, bits[tid],
                                 want);
                }
                ++na_bad;
            }
        }
        std::printf("  nested_acc bad=%d sample=%u %u %u\n", na_bad, bits[0], bits[1], bits[63]);
        std::printf(na_bad ? "PROBE_NESTED_ACC_FAIL\n" : "PROBE_NESTED_ACC_OK\n");
        return na_bad != 0;
    }
    if (std::strcmp(kern, "probe_ab_loops") == 0) {
        // Accumulator closed form: role A lanes read the B partner's writes
        // (14112 + 8*tid); role B lanes read the A partner's writes
        // (8*tid - 1984). Stored as raw integer bit patterns.
        unsigned int bits[512] = {0};
        std::memcpy(bits, got.data(), sizeof(bits));
        int ab_bad = 0;
        for (int tid = 0; tid < 512; ++tid) {
            long want = (tid < 256) ? (14112 + 8L * tid) : (8L * tid - 1984);
            if (bits[tid] != (unsigned int)want) {
                if (ab_bad < 6) {
                    std::fprintf(stderr, "ab_loops mismatch tid=%d got=%u want=%ld\n", tid, bits[tid],
                                 want);
                }
                ++ab_bad;
            }
        }
        std::printf("  ab_loops bad=%d sample=%u %u %u %u\n", ab_bad, bits[0], bits[256], bits[511],
                    bits[1]);
        std::printf(ab_bad ? "PROBE_AB_LOOPS_FAIL\n" : "PROBE_AB_LOOPS_OK\n");
        return ab_bad != 0;
    }
    if (std::strcmp(kern, "probe_roles_loop") == 0) {
        // Four iterations x 512 lanes. Lane tid of wave w reads the value the
        // partner wave (w+4)%8 wrote, which carries the +100000 role-B marker.
        unsigned int bits[2048] = {0};
        std::memcpy(bits, got.data(), sizeof(bits));
        int loop_bad = 0;
        for (int it = 0; it < 4; ++it) {
            for (int tid = 0; tid < 512; ++tid) {
                int w = tid / 64;
                int l = tid % 64;
                int pw = (w + 4) & 7;
                unsigned int want = (unsigned int)(it * 1000 + pw * 64 + l + (pw >= 4 ? 100000 : 0));
                unsigned int got_v = bits[it * 512 + tid];
                if (got_v != want) {
                    if (loop_bad < 6) {
                        std::fprintf(stderr, "roles_loop mismatch it=%d tid=%d got=%u want=%u\n", it,
                                     tid, got_v, want);
                    }
                    ++loop_bad;
                }
            }
        }
        std::printf("  roles_loop bad=%d sample=%u %u %u %u\n", loop_bad, bits[0], bits[64],
                    bits[1024], bits[1088]);
        std::printf(loop_bad ? "PROBE_ROLES_LOOP_FAIL\n" : "PROBE_ROLES_LOOP_OK\n");
        return loop_bad != 0;
    }
    if (std::strcmp(kern, "probe_fmha_datapath") == 0) {
        // 2048 dwords, all of which must be the f32 bit pattern of 64.0.
        const unsigned int want = 0x42800000u;  // 64.0f
        unsigned int bits[2048] = {0};
        std::memcpy(bits, got.data(), sizeof(bits));
        int dp_bad = 0;
        for (int i = 0; i < 2048; ++i) {
            if (bits[i] != want) {
                if (dp_bad < 6) {
                    std::fprintf(stderr, "datapath mismatch i=%d got=0x%08x want=0x%08x\n", i,
                                 bits[i], want);
                }
                ++dp_bad;
            }
        }
        int hi_bad = 0;
        int lo_bad = 0;
        for (int i = 0; i < 1024; ++i) {
            if (bits[i] != want) {
                ++hi_bad;
            }
        }
        for (int i = 1024; i < 2048; ++i) {
            if (bits[i] != want) {
                ++lo_bad;
            }
        }
        std::printf("  datapath bad=%d hi_bad=%d lo_bad=%d sample=0x%08x 0x%08x\n", dp_bad, hi_bad,
                    lo_bad, bits[0], bits[1024]);
        std::printf(dp_bad ? "PROBE_FMHA_DATAPATH_FAIL\n" : "PROBE_FMHA_DATAPATH_OK\n");
        return dp_bad != 0;
    }
    if (std::strcmp(kern, "hw_roles") == 0) {
        // Hand-written kernel: o[tid] must equal tid for every lane.
        float vals[128] = {0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        int hw_bad = 0;
        for (int tid = 0; tid < 128; ++tid) {
            if (std::fabs(vals[tid] - (float)((tid + 63) & 63)) > 0.5f) {
                if (hw_bad < 6) {
                    std::fprintf(stderr, "hw_roles mismatch tid=%d got=%.1f\n", tid, vals[tid]);
                }
                ++hw_bad;
            }
        }
        std::printf("  hw_roles bad=%d sample=%.1f %.1f %.1f %.1f\n", hw_bad, vals[0], vals[1],
                    vals[64], vals[65]);
        std::printf(hw_bad ? "HW_ROLES_FAIL\n" : "HW_ROLES_OK\n");
        return hw_bad != 0;
    }
    if (std::strcmp(kern, "hw_lds_neighbour") == 0 || std::strcmp(kern, "hw_lds_own") == 0) {
        const bool neighbour = std::strcmp(kern, "hw_lds_neighbour") == 0;
        float vals[64] = {0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        int bad = 0;
        for (int tid = 0; tid < 64; ++tid) {
            int want = neighbour ? ((tid + 1) & 63) : tid;
            if (std::fabs(vals[tid] - (float)want) > 0.5f) {
                if (bad < 6) {
                    std::fprintf(stderr, "%s mismatch tid=%d got=%.1f want=%d\n", kern, tid,
                                 vals[tid], want);
                }
                ++bad;
            }
        }
        std::printf("  %s bad=%d sample=%.1f %.1f %.1f %.1f\n", kern, bad, vals[0], vals[1],
                    vals[2], vals[3]);
        std::printf(bad ? "HW_LDS_FAIL\n" : "HW_LDS_OK\n");
        return bad != 0;
    }
    if (std::strcmp(kern, "probe_lds_self2") == 0) {
        unsigned int bits[64] = {0};
        std::memcpy(bits, got.data(), sizeof(bits));
        int bad = 0;
        for (int tid = 0; tid < 64; ++tid) {
            if (bits[tid] != (unsigned int)tid) {
                if (bad < 4) {
                    std::fprintf(stderr, "self2 mismatch tid=%d got=%u\n", tid, bits[tid]);
                }
                ++bad;
            }
        }
        std::printf("  self2 bad=%d sample=%u %u %u %u\n", bad, bits[0], bits[1], bits[2], bits[3]);
        std::printf(bad ? "SELF2_FAIL\n" : "SELF2_OK\n");
        return bad != 0;
    }
    if (std::strcmp(kern, "probe_lds_own") == 0) {
        // o[tid] is a float constant 42.0; o[64+tid] holds the raw LDS bit pattern.
        float vals[64] = {0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        unsigned int bits[64] = {0};
        std::memcpy(bits, got.data() + 64 * sizeof(float), sizeof(bits));
        int first_bad = 0;
        int read_bad = 0;
        for (int tid = 0; tid < 64; ++tid) {
            if (std::fabs(vals[tid] - 42.0f) > 0.5f) {
                ++first_bad;
            }
            if (bits[tid] != (unsigned int)tid) {
                if (read_bad < 4) {
                    std::fprintf(stderr, "own read mismatch tid=%d got=%u\n", tid, bits[tid]);
                }
                ++read_bad;
            }
        }
        std::printf("  lds_own first_bad=%d second_bad=%d first[0]=%.1f second[0]=%u\n", first_bad,
                    read_bad, vals[0], bits[0]);
        std::printf((first_bad || read_bad) ? "LDS_OWN_FAIL\n" : "LDS_OWN_OK\n");
        return (first_bad || read_bad) ? 1 : 0;
    }
    if (std::strncmp(kern, "hw_bisect", 9) == 0 || std::strcmp(kern, "hw_store2") == 0) {
        float vals[64] = {0.0f};
        std::memcpy(vals, got.data(), sizeof(vals));
        int bad = 0;
        for (int tid = 0; tid < 64; ++tid) {
            if (std::fabs(vals[tid] - (float)((tid + 63) & 63)) > 0.5f) {
                if (bad < 4) {
                    std::fprintf(stderr, "%s mismatch tid=%d got=%.1f\n", kern, tid, vals[tid]);
                }
                ++bad;
            }
        }
        std::printf("  %s bad=%d sample=%.1f %.1f %.1f %.1f\n", kern, bad, vals[0], vals[1],
                    vals[2], vals[3]);
        std::printf(bad ? "HW_BISECT_FAIL\n" : "HW_BISECT_OK\n");
        return bad != 0;
    }
    if (std::strcmp(kern, "attention_fwd") != 0) {
        // Diagnostic kernel layout per row: 0 l, 1 m, 2 acc0, 3 q0,
        // 4 corr, 5 p, 6 l_new, 7 scaled (last iteration).
        for (int r = 0; r < 4; ++r) {
            std::printf(
                "  row %d: l=%.6f m=%.6f acc0=%.6f q0=%.6f corr=%.6f p=%.6f l_new=%.6f s=%.6f\n",
                r, __half2float(got[(size_t)r * D + 0]), __half2float(got[(size_t)r * D + 1]),
                __half2float(got[(size_t)r * D + 2]), __half2float(got[(size_t)r * D + 3]),
                __half2float(got[(size_t)r * D + 4]), __half2float(got[(size_t)r * D + 5]),
                __half2float(got[(size_t)r * D + 6]), __half2float(got[(size_t)r * D + 7]));
        }
        return 0;
    }

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
    // Per-row bad counts: distinguishes broken lanes from broken math.
    {
        size_t rows_bad = 0;
        size_t bad_in_lo = 0;
        size_t bad_in_hi = 0;
        for (int r = 0; r < B * H * S; ++r) {
            size_t row_bad = 0;
            for (int d = 0; d < D; ++d) {
                size_t i = (size_t)r * D + d;
                double a = __half2float(got[i]);
                double e = ref[i];
                if (std::fabs(a - e) > 2e-2 + 2e-2 * std::fabs(e)) {
                    ++row_bad;
                }
            }
            if (row_bad) {
                ++rows_bad;
                if (rows_bad <= 40) {
                    std::printf("   bad_row=%d tid=%d block=%d\n", r, r % 64, r / 64);
                }
                if ((r % S) < 32) {
                    bad_in_lo += row_bad;
                } else {
                    bad_in_hi += row_bad;
                }
            }
            if (r < 4) {
                std::printf("  row %d bad=%zu\n", r, row_bad);
            }
        }
        std::printf("rows_with_bad=%zu/%d bad_in_rows0_31=%zu bad_in_rows32_63=%zu\n",
                    rows_bad, B * H * S, bad_in_lo, bad_in_hi);
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
