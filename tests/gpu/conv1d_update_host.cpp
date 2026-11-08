// causal_conv1d_update (batch 2): causal_conv1d_update_bf16_w<W>_s<SL>[_idx][_silu]: CPU reference (output and the shifted state) and bandwidth.
// usage: conv1d_update_host <co> <kernel> <B> <dim> <W> <SL> <idx 0|1> <silu 0|1> [iters]      (idx: continuous batching, a random slot per sequence, some padding = -1)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
int main(int argc, char** argv) {
    if (argc < 9) { std::fprintf(stderr, "usage: %s <co> <kernel> <B> <dim> <W> <SL> <idx> <silu> [iters]\n", argv[0]); return 2; }
    const char* co = argv[1]; const char* kn = argv[2]; long B = std::atol(argv[3]), dim = std::atol(argv[4]), W = std::atol(argv[5]), SL = std::atol(argv[6]); bool idx = std::atoi(argv[7]), silu = std::atoi(argv[8]); int iters = argc > 9 ? std::atoi(argv[9]) : 30;
    const long SLOTS = idx ? B + B / 3 + 2 : B;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> hx(B * dim), hw(dim * W), hb(dim), hs((size_t)SLOTS * dim * SL); for (auto& v : hx) v = f32_to_bf16(nd(rng)); for (auto& v : hw) v = f32_to_bf16(nd(rng) * 0.5f); for (auto& v : hb) v = f32_to_bf16(nd(rng)); for (auto& v : hs) v = f32_to_bf16(nd(rng));
    std::vector<int32_t> slots(B); { std::vector<int32_t> perm(SLOTS); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng); for (long b = 0; b < B; ++b) slots[b] = idx ? ((b % 5 == 2) ? -1 : perm[b]) : (int32_t)b; }
    void *dx, *dw, *db, *ds, *dout, *di; CK(hipMalloc(&dx, hx.size() * 2)); CK(hipMalloc(&dw, hw.size() * 2)); CK(hipMalloc(&db, hb.size() * 2)); CK(hipMalloc(&ds, hs.size() * 2 + 64)); CK(hipMalloc(&dout, hx.size() * 2 + 64)); CK(hipMalloc(&di, B * 4));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(db, hb.data(), hb.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(ds, hs.data(), hs.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(di, slots.data(), B * 4, hipMemcpyHostToDevice)); CK(hipMemset(dout, 0x55, hx.size() * 2 + 64));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    char args[80]; size_t sz = 0; auto put = [&](const void* p, size_t k) { std::memcpy(args + sz, p, k); sz += k; };
    put(&dx, 8); put(&dw, 8); put(&db, 8); put(&ds, 8); put(&dout, 8); if (idx) put(&di, 8); int32_t d32 = (int32_t)dim, s32 = (int32_t)SLOTS; put(&d32, 4); put(&s32, 4);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    int gx = (int)((dim + 255) / 256);
    CK(hipModuleLaunchKernel(fn, gx, (int)B, 1, 256, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize());
    std::vector<uint16_t> ho(hx.size() + 32), hs2(hs.size()); CK(hipMemcpy(ho.data(), dout, ho.size() * 2, hipMemcpyDeviceToHost)); CK(hipMemcpy(hs2.data(), ds, hs.size() * 2, hipMemcpyDeviceToHost));
    long bad = 0, sbad = 0; std::vector<uint16_t> hs_ref = hs;
    for (long b = 0; b < B; ++b) { long sl = slots[b]; if (sl < 0) { for (long c = 0; c < dim; ++c) if (ho[b * dim + c] != 0x5555) ++bad; continue; }
        for (long c = 0; c < dim; ++c) {
            uint16_t* st = &hs_ref[((size_t)sl * dim + c) * SL]; double xv[5]; for (long l = 0; l < W - 1; ++l) xv[l] = bf16_to_f32(st[SL - (W - 1) + l]); xv[W - 1] = bf16_to_f32(hx[b * dim + c]);
            double acc = bf16_to_f32(hb[c]); for (long j = 0; j < W; ++j) acc += (double)bf16_to_f32(hw[c * W + j]) * xv[j]; if (silu) acc = acc / (1.0 + std::exp(-acc));
            double got = bf16_to_f32(ho[b * dim + c]); if (!(std::fabs(got - acc) <= std::fabs(acc) / 128.0 * 1.05 + 1e-5)) { if (bad < 5) std::fprintf(stderr, "bad out b=%ld c=%ld: got %g ref %g\n", b, c, got, acc); ++bad; }
            for (long l = 0; l + 1 < SL; ++l) st[l] = st[l + 1]; st[SL - 1] = hx[b * dim + c];
        } }
    for (size_t i = 0; i < hs.size(); ++i) if (hs2[i] != hs_ref[i]) { if (sbad < 5) std::fprintf(stderr, "state bad at %zu\n", i); ++sbad; }
    long guard = 0; for (size_t i = hx.size(); i < hx.size() + 32; ++i) if (ho[i] != 0x5555) ++guard;
    std::printf("check: conv1d_update B=%ld dim=%ld W=%ld SL=%ld%s%s bad=%ld state_bad=%ld past_end_bad=%ld %s\n", B, dim, W, SL, idx ? " idx" : "", silu ? " silu" : "", bad, sbad, guard, (bad == 0 && sbad == 0 && guard == 0) ? "CONV_OK" : "CONV_FAIL");
    hipEvent_t a, bb; CK(hipEventCreate(&a)); CK(hipEventCreate(&bb));
    for (int i = 0; i < 20; ++i) CK(hipModuleLaunchKernel(fn, gx, (int)B, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) CK(hipModuleLaunchKernel(fn, gx, (int)B, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipEventRecord(bb)); CK(hipEventSynchronize(bb)); float ms; CK(hipEventElapsedTime(&ms, a, bb)); ms /= iters;
    double bytes = (double)B * dim * (2.0 * SL * 2 + 2 + 2) + (double)dim * (2 * W + 2);
    std::printf("timing: %.4f ms/iter (%.2f us), %.1f GB/s (traffic %.1f MB)\n", ms, ms * 1e3, bytes / ms / 1e6, bytes / 1e6);
    return 0;
}