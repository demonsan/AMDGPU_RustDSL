// Function / ABI level feature probes (probe_llvm_*): each case launches one kernel with a raw kernarg buffer (explicit byte layout) and checks the result.
// Usage: llvm_feature_host <case> <code-object>   (case = kernarg ids roles lds branch loop atomic maskge twokernel)
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                    \
    do {                                                                         \
        hipError_t e_ = (x);                                                     \
        if (e_ != hipSuccess) {                                                  \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_));  \
            std::exit(2);                                                        \
        }                                                                        \
    } while (0)

static hipFunction_t load(const char* co, const char* name) {
    hipModule_t mod;
    hipFunction_t fn;
    CK(hipModuleLoad(&mod, co));
    CK(hipModuleGetFunction(&fn, mod, name));
    return fn;
}

static void launch(hipFunction_t fn, int gx, int gy, int gz, int bx, std::vector<unsigned char>& ka) {
    size_t sz = ka.size();
    void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ka.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, gx, gy, gz, bx, 1, 1, 0, nullptr, nullptr, extra));
    CK(hipDeviceSynchronize());
}

template <class T> static void put(std::vector<unsigned char>& ka, size_t off, T v) { std::memcpy(ka.data() + off, &v, sizeof(T)); }
static float bits2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static int verdict(const char* name, int bad, const char* what) {
    std::printf("%s %s (%s, bad=%d)\n", bad == 0 ? "FEATURE_PASS" : "FEATURE_FAIL", name, what, bad);
    return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) { std::fprintf(stderr, "usage: %s <case> <co> [co2]\n", argv[0]); return 2; }
    std::string c = argv[1];
    const char* co = argv[2];
    int bad = 0;
    if (c == "kernarg") {
        // three pointers, i32, f32 at offset 20, i32 at 32; 40 byte kernarg
        const int n = 64;
        std::vector<float> ha(n), hb(n), hc(80, -1.f);
        for (int i = 0; i < n; ++i) { ha[i] = (float)i; hb[i] = 2.f * i; }
        void *da, *db, *dc;
        CK(hipMalloc(&da, n * 4)); CK(hipMalloc(&db, n * 4)); CK(hipMalloc(&dc, 80 * 4));
        CK(hipMemcpy(da, ha.data(), n * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(db, hb.data(), n * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dc, hc.data(), 80 * 4, hipMemcpyHostToDevice));
        std::vector<unsigned char> ka(40, 0);
        put(ka, 0, (uint64_t)da); put(ka, 8, (uint64_t)db); put<int32_t>(ka, 16, 7); put<float>(ka, 20, 0.5f); put(ka, 24, (uint64_t)dc); put<int32_t>(ka, 32, -3);
        launch(load(co, "probe_llvm_kernarg"), 1, 1, 1, 64, ka);
        CK(hipMemcpy(hc.data(), dc, 80 * 4, hipMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) if (hc[i] != 3.f * i + 0.5f) ++bad;
        uint32_t u64, u65; std::memcpy(&u64, &hc[64], 4); std::memcpy(&u65, &hc[65], 4);
        if ((int)u64 != 7 || (int)u65 != -3) ++bad;
        return verdict("kernarg_layout", bad, "3 pointers + i32 + f32@4-aligned + i32, 40-byte kernarg");
    }
    if (c == "ids") {
        const int G = 12, n = G * 64;
        std::vector<int> h(n, -1);
        void* d; CK(hipMalloc(&d, n * 4)); CK(hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice));
        std::vector<unsigned char> ka(8, 0); put(ka, 0, (uint64_t)d);
        launch(load(co, "probe_llvm_ids"), 3, 2, 2, 64, ka);
        CK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
        for (int z = 0; z < 2; ++z) for (int y = 0; y < 2; ++y) for (int x = 0; x < 3; ++x) for (int t = 0; t < 64; ++t) {
            int idx = ((z * 2 + y) * 3 + x) * 64 + t;
            if (h[idx] != (z * 100 + y * 10 + x) * 1000 + t) ++bad;
        }
        return verdict("workgroup_workitem_ids", bad, "grid 3x2x2, block 64, WorkgroupIdX/Y/Z + WorkitemId");
    }
    if (c == "roles") {
        const int n = 512;
        std::vector<int> h(n, -1);
        void* d; CK(hipMalloc(&d, n * 4)); CK(hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice));
        std::vector<unsigned char> ka(8, 0); put(ka, 0, (uint64_t)d);
        launch(load(co, "probe_llvm_roles"), 1, 1, 1, 512, ka);
        CK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
        for (int t = 0; t < n; ++t) { int want = t < 256 ? 2000 + t + 256 : 1000 + t - 256; if (h[t] != want) ++bad; }
        return verdict("wave_roles_ab_if_else_lds_barrier", bad, "512 threads, waves 0-3 / 4-7 if/else, static LDS, barrier, waitcnt");
    }
    if (c == "lds") {
        const int n = 128;
        std::vector<int> h(n, -1);
        void* d; CK(hipMalloc(&d, n * 4)); CK(hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice));
        std::vector<unsigned char> ka(8, 0); put(ka, 0, (uint64_t)d);
        launch(load(co, "probe_llvm_lds"), 1, 1, 1, 64, ka);
        CK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
        for (int t = 0; t < 64; ++t) { if (h[t] != 63 - t) ++bad; if (h[64 + t] != (63 - t) + 2000) ++bad; }
        return verdict("static_lds_b32_b128_barrier", bad, "8 KiB static LDS, ds_write/read b32 + b128, barrier");
    }
    if (c == "branch") {
        const int G = 8;
        std::vector<int> h(G, -1);
        void* d; CK(hipMalloc(&d, G * 4)); CK(hipMemcpy(d, h.data(), G * 4, hipMemcpyHostToDevice));
        std::vector<unsigned char> ka(8, 0); put(ka, 0, (uint64_t)d);
        launch(load(co, "probe_llvm_branch"), G, 1, 1, 64, ka);
        CK(hipMemcpy(h.data(), d, G * 4, hipMemcpyDeviceToHost));
        for (int w = 0; w < G; ++w) { int want = (w % 2 == 0) ? (w < 4 ? 11 : 12) : (w < 4 ? 21 : 22); if (h[w] != want) ++bad; }
        return verdict("uniform_branching_nested_if_else", bad, "wave-uniform nested begin_if/else on workgroup id");
    }
    if (c == "loop") {
        const int n = 64, N = 10;
        std::vector<unsigned> h(128, 0);
        void* d; CK(hipMalloc(&d, 128 * 4)); CK(hipMemcpy(d, h.data(), 128 * 4, hipMemcpyHostToDevice));
        std::vector<unsigned char> ka(16, 0); put(ka, 0, (uint64_t)d); put<int32_t>(ka, 8, N);
        launch(load(co, "probe_llvm_loop"), 1, 1, 1, 64, ka);
        CK(hipMemcpy(h.data(), d, 128 * 4, hipMemcpyDeviceToHost));
        for (int t = 0; t < n; ++t) { if ((int)h[t] != N * (N - 1) / 2) ++bad; if (bits2f(h[64 + t]) != 0.5f * N) ++bad; }
        return verdict("loop_carried_variables", bad, "3 loop-carried vars (i32, i32, f32), data-dependent trip count from kernarg");
    }
    if (c == "atomic") {
        const int G = 5, n = G * 64;
        std::vector<unsigned> h(n, 0xffffffffu);
        void *d, *ctr; CK(hipMalloc(&d, n * 4)); CK(hipMalloc(&ctr, 4)); CK(hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice)); CK(hipMemset(ctr, 0, 4));
        std::vector<unsigned char> ka(16, 0); put(ka, 0, (uint64_t)d); put(ka, 8, (uint64_t)ctr);
        launch(load(co, "probe_llvm_atomic"), G, 1, 1, 64, ka);
        CK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
        unsigned fin = 0; CK(hipMemcpy(&fin, ctr, 4, hipMemcpyDeviceToHost));
        std::vector<int> seen(n, 0);
        for (int i = 0; i < n; ++i) { if (h[i] < (unsigned)n) seen[h[i]]++; else ++bad; }
        for (int i = 0; i < n; ++i) if (seen[i] != 1) ++bad;
        if (fin != (unsigned)n) ++bad;
        return verdict("global_atomic_add_returning", bad, "atomic_add_u32 with return, old values are a permutation, final counter = 64 * grid");
    }
    if (c == "maskge") {
        const int n = 64, thr = 20;
        std::vector<float> h(128, -1.f);
        void* d; CK(hipMalloc(&d, 128 * 4)); CK(hipMemcpy(d, h.data(), 128 * 4, hipMemcpyHostToDevice));
        std::vector<unsigned char> ka(16, 0); put(ka, 0, (uint64_t)d); put<int32_t>(ka, 8, thr);
        launch(load(co, "probe_llvm_maskge"), 1, 1, 1, 64, ka);
        CK(hipMemcpy(h.data(), d, 128 * 4, hipMemcpyDeviceToHost));
        // out[t] = (t >= thr) ? 1 : 2 ; out[64 + t] = (bits2f(t) < 1.0f) ? 5 : 6 (every small integer pattern is a denormal or 0: always 5)
        for (int t = 0; t < n; ++t) { if (h[t] != (t >= thr ? 1.0f : 2.0f)) ++bad; if (h[64 + t] != 5.0f) ++bad; }
        return verdict("compare_select_ge_lt", bad, "i32 >= and f32 < compare feeding selects");
    }
    if (c == "twokernel") {
        const int n = 64;
        const char* co2 = argc > 3 ? argv[3] : co;
        std::vector<int> h(n, -1);
        void* d; CK(hipMalloc(&d, n * 4)); CK(hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice));
        std::vector<unsigned char> ka(8, 0); put(ka, 0, (uint64_t)d);
        launch(load(co, "probe_llvm_k1"), 1, 1, 1, 64, ka);
        launch(load(co2, "probe_llvm_k2"), 1, 1, 1, 64, ka);
        CK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
        for (int t = 0; t < n; ++t) if (h[t] != 2 * t + 1) ++bad;
        return verdict("two_kernels_two_descriptors", bad, "two code objects, each with its own kernel descriptor, chained on one buffer");
    }
    std::fprintf(stderr, "unknown case %s\n", c.c_str());
    return 2;
}
