// Sampling from logits (batch 2): argmax_bf16_v<V> (greedy, lowest index on ties) and sample_bf16_v<V> (temperature + inverse-CDF with the caller's uniform number): against a CPU reference and bandwidth.
// usage: sampling_host <co> <kernel> <argmax|sample> <B> <V> [iters] [temperature]
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
int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s <co> <kernel> <argmax|sample> <B> <V> [iters] [temperature]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; bool sample = std::string(argv[3]) == "sample"; long B = std::atol(argv[4]), V = std::atol(argv[5]); int iters = argc > 6 ? std::atoi(argv[6]) : 20; float temp = argc > 7 ? (float)std::atof(argv[7]) : 0.8f;
    std::mt19937 rng(17); std::normal_distribution<float> nd(0.f, 3.f); std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<uint16_t> hl((size_t)B * V); for (auto& v : hl) v = f32_to_bf16(nd(rng));
    for (long r = 0; r < B; ++r) { hl[(size_t)r * V + (r * 7919) % V] = f32_to_bf16(30.f); if (r % 3 == 0) hl[(size_t)r * V + (r * 104729) % V] = f32_to_bf16(30.f); }   // peaks, and a tie in every third row
    std::vector<float> hu(B); for (auto& v : hu) v = ud(rng);
    void *dl, *du, *dout; CK(hipMalloc(&dl, hl.size() * 2)); CK(hipMalloc(&du, B * 4)); CK(hipMalloc(&dout, B * 4 + 64)); CK(hipMemset(dout, 0x55, B * 4 + 64));
    CK(hipMemcpy(dl, hl.data(), hl.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(du, hu.data(), B * 4, hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[40]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    float inv_t = 1.0f / temp; int32_t b32 = (int32_t)B;
    if (sample) { put(&dl, 8); put(&du, 8); put(&dout, 8); put(&inv_t, 4); put(&b32, 4); } else { put(&dl, 8); put(&dout, 8); put(&b32, 4); }
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, (int)B, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<int32_t> ho(B + 16); CK(hipMemcpy(ho.data(), dout, (B + 16) * 4, hipMemcpyDeviceToHost));
    long bad = 0, near = 0;
    for (long r = 0; r < B; ++r) {
        const uint16_t* row = &hl[(size_t)r * V];
        if (!sample) { long best = 0; for (long i = 1; i < V; ++i) if (bf16_to_f32(row[i]) > bf16_to_f32(row[best])) best = i; if (ho[r] != best) { if (bad < 5) std::fprintf(stderr, "row %ld: got %d ref %ld\n", r, ho[r], best); ++bad; } }
        else {
            double mx = -1e300; for (long i = 0; i < V; ++i) mx = std::max(mx, (double)bf16_to_f32(row[i]) * inv_t);
            std::vector<double> p(V); double s = 0; for (long i = 0; i < V; ++i) { p[i] = std::exp((double)bf16_to_f32(row[i]) * inv_t - mx); s += p[i]; }
            double target = std::min((double)hu[r], 0.99999994) * s, acc = 0; long ref = V - 1; for (long i = 0; i < V; ++i) { acc += p[i]; if (acc >= target) { ref = i; break; } }
            // the kernel's float cumulative sums differ from the double ones in the last bits: accept the neighbours whose cumulative value is within 1e-4 of the target
            if (ho[r] != ref) { double a2 = 0; long lo = std::min<long>(ho[r], ref), hi = std::max<long>(ho[r], ref); for (long i = 0; i <= lo; ++i) a2 += p[i]; double between = 0; for (long i = lo + 1; i <= hi; ++i) between += p[i]; if (between / s < 2e-4 && ho[r] >= 0 && ho[r] < V) ++near; else { if (bad < 5) std::fprintf(stderr, "row %ld: got %d ref %ld (u=%g)\n", r, ho[r], ref, hu[r]); ++bad; } }
        }
    }
    long guard = 0; for (long i = B; i < B + 16; ++i) if (ho[i] != 0x55555555) ++guard;
    std::printf("check: %s B=%ld V=%ld bad=%ld near_boundary=%ld past_end_bad=%ld %s\n", sample ? "sample" : "argmax", B, V, bad, near, guard, (bad == 0 && guard == 0) ? "SAMP_OK" : "SAMP_FAIL");
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 10; ++i) CK(hipModuleLaunchKernel(fn, (int)B, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, (int)B, 1, 1, 1024, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= iters;
    double bytes = (double)B * V * 2;
    std::printf("timing: %.4f ms/iter (%.2f us), %.1f GB/s (read %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}