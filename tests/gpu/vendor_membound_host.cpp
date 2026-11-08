// Same-run vendor comparison for the memory-bound kernels (batch 2): the aiter HIP entry points (csrc/kernels: activation, quant, rmsnorm_quant), compiled standalone (no torch), timed with the same
// method as our hosts (hipEvent over back-to-back launches, ~1 GB working sets).  Build: see tools/vendor_membound.sh.
// usage: vendor_membound_host <case> <M> <N> [iters]   cases: silu_mul gelu_tanh_mul rmsnorm add_rmsnorm rmsnorm_quant quant_token quant_group128 quant_static silu_mul_quant
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "aiter_tensor.h"
#include "activation.h"
#include "quant.h"
#include "rmsnorm_quant.h"
#include "moe_topk_op.h"
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static uint16_t f32_to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static AiterTensor bf16_rand(std::initializer_list<int64_t> dims, size_t n, float sd, uint32_t seed) {
    AiterTensor t = AiterTensor::empty(dims, AITER_DTYPE_bf16, 0);
    std::vector<uint16_t> h(n); std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, sd);
    for (auto& v : h) v = f32_to_bf16(nd(rng));
    CK(hipMemcpy(t.ptr, h.data(), n * 2, hipMemcpyHostToDevice)); return t;
}
template <class F> static double bench(F f, int iters) {
    for (int i = 0; i < 20; ++i) f();
    CK(hipDeviceSynchronize());
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    CK(hipEventRecord(a)); for (int i = 0; i < iters; ++i) f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms / iters;
}
int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <case> <M> <N> [iters]\n", argv[0]); return 2; }
    std::string c = argv[1]; int64_t M = std::atol(argv[2]), N = std::atol(argv[3]); int iters = argc > 4 ? std::atoi(argv[4]) : 50;
    double bytes = 0, ms = 0; const size_t MN = (size_t)M * N;
    if (c == "silu_mul" || c == "gelu_tanh_mul") {
        AiterTensor in = bf16_rand({M, 2 * N}, MN * 2, 2.f, 1), out = AiterTensor::empty({M, N}, AITER_DTYPE_bf16, 0);
        if (c == "silu_mul") ms = bench([&] { aiter::silu_and_mul(out, in, 0.0f); }, iters); else ms = bench([&] { aiter::gelu_tanh_and_mul(out, in); }, iters);
        bytes = (double)MN * 6;
    } else if (c == "rmsnorm") {
        AiterTensor in = bf16_rand({M, N}, MN, 2.f, 2), w = bf16_rand({N}, N, 1.f, 3), out = AiterTensor::empty({M, N}, AITER_DTYPE_bf16, 0);
        ms = bench([&] { aiter::rmsnorm(out, in, w, 1e-5); }, iters); bytes = (double)MN * 4 + N * 2;
    } else if (c == "add_rmsnorm") {
        AiterTensor in = bf16_rand({M, N}, MN, 2.f, 2), res = bf16_rand({M, N}, MN, 2.f, 4), w = bf16_rand({N}, N, 1.f, 3), out = AiterTensor::empty({M, N}, AITER_DTYPE_bf16, 0), rout = AiterTensor::empty({M, N}, AITER_DTYPE_bf16, 0);
        ms = bench([&] { aiter::add_rmsnorm(out, in, res, rout, w, 1e-5); }, iters); bytes = (double)MN * 8 + N * 2;
    } else if (c == "rmsnorm_quant") {
        AiterTensor in = bf16_rand({M, N}, MN, 2.f, 2), w = bf16_rand({N}, N, 1.f, 3), out = AiterTensor::empty({M, N}, AITER_DTYPE_fp8, 0), sc = AiterTensor::empty({M, 1}, AITER_DTYPE_fp32, 0);
        ms = bench([&] { aiter::rmsnorm_quant(out, in, sc, w, 1e-5, 0); }, iters); bytes = (double)MN * 3 + M * 4 + N * 2;
    } else if (c == "quant_token") {
        AiterTensor in = bf16_rand({M, N}, MN, 2.f, 2), out = AiterTensor::empty({M, N}, AITER_DTYPE_fp8, 0), sc = AiterTensor::empty({M, 1}, AITER_DTYPE_fp32, 0);
        ms = bench([&] { aiter::dynamic_per_token_scaled_quant(out, in, sc); }, iters); bytes = (double)MN * 3 + M * 4;
    } else if (c == "quant_group128") {
        AiterTensor in = bf16_rand({M, N}, MN, 2.f, 2), out = AiterTensor::empty({M, N}, AITER_DTYPE_fp8, 0), sc = AiterTensor::empty({M, N / 128}, AITER_DTYPE_fp32, 0);
        ms = bench([&] { aiter::dynamic_per_group_scaled_quant(out, in, sc, 128, false); }, iters); bytes = (double)MN * 3 + (double)M * (N / 128) * 4;
    } else if (c == "quant_static") {
        AiterTensor in = bf16_rand({M, N}, MN, 30.f, 2), out = AiterTensor::empty({M, N}, AITER_DTYPE_fp8, 0), sc = AiterTensor::empty({1}, AITER_DTYPE_fp32, 0);
        float s = 0.5f; CK(hipMemcpy(sc.ptr, &s, 4, hipMemcpyHostToDevice));
        ms = bench([&] { aiter::static_per_tensor_quant(out, in, sc); }, iters); bytes = (double)MN * 3;
    } else if (c == "silu_mul_quant") {
        AiterTensor in = bf16_rand({M, 2 * N}, MN * 2, 2.f, 1), out = AiterTensor::empty({M, N}, AITER_DTYPE_fp8, 0), sc = AiterTensor::empty({M, 1}, AITER_DTYPE_fp32, 0);
        ms = bench([&] { aiter::silu_and_mul_quant(out, in, sc, (int)N); }, iters); bytes = (double)MN * 5 + M * 4;
    } else if (c == "topk_softmax") {   // M tokens, N experts, topk from env TOPK (default 8): gating bf16 [M][N]
        int K = std::getenv("TOPK") ? std::atoi(std::getenv("TOPK")) : 8;
        AiterTensor g = bf16_rand({M, N}, MN, 2.f, 1), w = AiterTensor::empty({M, (int64_t)K}, AITER_DTYPE_fp32, 0), id = AiterTensor::empty({M, (int64_t)K}, AITER_DTYPE_i32, 0), tid = AiterTensor::empty({M, (int64_t)K}, AITER_DTYPE_i32, 0), ws = AiterTensor::empty({M * N}, AITER_DTYPE_fp32, 0);
        ms = bench([&] { ::topk_softmax(w, id, tid, g, ws, false, 0, "", std::nullopt); }, iters); bytes = (double)MN * 2 + (double)M * K * 8;
    } else { std::fprintf(stderr, "unknown case %s\n", c.c_str()); return 2; }
    std::printf("vendor %-15s M=%ld N=%ld: %.4f ms/iter, %.1f GB/s\n", c.c_str(), (long)M, (long)N, ms, bytes / ms / 1e6);
    return 0;
}