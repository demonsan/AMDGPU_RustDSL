// VALU issue-rate microbenchmark on gfx1250 (wave32): a loop of 32 independent v_fmac_f32 (A), the same work as 16 VOPD pairs
// v_dual_fmac_f32 :: v_dual_fmac_f32 (B), and 32 dependent v_fmac_f32 chains of 4 (C: latency-bound reference). One workgroup per CU with
// WAVES waves (argv[1], default 4 = one per SIMD). Prints the cycles per VALU instruction per wave from s_memtime-free host timing
// (ns per loop iteration and the instruction count; the clock from hipDeviceProp).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HC(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

#define CLOB "v40","v41","v42","v43","v44","v45","v46","v47","v48","v49","v50","v51","v52","v53","v54","v55","v56","v57","v58","v59","v60","v61","v62","v63","v64","v65","v66","v67","v68","v69","v70","v71","v72","v73","v74","v75","s90","scc"

#define INIT "v_mov_b32 v72, 1.0\n v_mov_b32 v73, 0.5\n v_mov_b32 v74, 2.0\n v_mov_b32 v75, 0.25\n" \
  "v_mov_b32 v40, 0\n v_mov_b32 v41, 0\n v_mov_b32 v42, 0\n v_mov_b32 v43, 0\n v_mov_b32 v44, 0\n v_mov_b32 v45, 0\n v_mov_b32 v46, 0\n v_mov_b32 v47, 0\n" \
  "v_mov_b32 v48, 0\n v_mov_b32 v49, 0\n v_mov_b32 v50, 0\n v_mov_b32 v51, 0\n v_mov_b32 v52, 0\n v_mov_b32 v53, 0\n v_mov_b32 v54, 0\n v_mov_b32 v55, 0\n" \
  "v_mov_b32 v56, 0\n v_mov_b32 v57, 0\n v_mov_b32 v58, 0\n v_mov_b32 v59, 0\n v_mov_b32 v60, 0\n v_mov_b32 v61, 0\n v_mov_b32 v62, 0\n v_mov_b32 v63, 0\n" \
  "v_mov_b32 v64, 0\n v_mov_b32 v65, 0\n v_mov_b32 v66, 0\n v_mov_b32 v67, 0\n v_mov_b32 v68, 0\n v_mov_b32 v69, 0\n v_mov_b32 v70, 0\n v_mov_b32 v71, 0\n"

#define F(d) "v_fmac_f32 v" #d ", v72, v73\n"
#define F8(a) F(a) F(a##1)
#define BODY_A F(40) F(41) F(42) F(43) F(44) F(45) F(46) F(47) F(48) F(49) F(50) F(51) F(52) F(53) F(54) F(55) \
  F(56) F(57) F(58) F(59) F(60) F(61) F(62) F(63) F(64) F(65) F(66) F(67) F(68) F(69) F(70) F(71)
#define D(x, y) "v_dual_fmac_f32 v" #x ", v72, v73 :: v_dual_fmac_f32 v" #y ", v74, v75\n"
#define BODY_B D(40, 41) D(42, 43) D(44, 45) D(46, 47) D(48, 49) D(50, 51) D(52, 53) D(54, 55) D(56, 57) D(58, 59) D(60, 61) D(62, 63) \
  D(64, 65) D(66, 67) D(68, 69) D(70, 71)
// C: 8 independent chains of 4 dependent fmacs (latency reference)
#define BODY_C F(40) F(41) F(42) F(43) F(44) F(45) F(46) F(47) F(40) F(41) F(42) F(43) F(44) F(45) F(46) F(47) \
  F(40) F(41) F(42) F(43) F(44) F(45) F(46) F(47) F(40) F(41) F(42) F(43) F(44) F(45) F(46) F(47)

#define KERNEL(name, BODY) \
__global__ void name(float* out, int iters) { \
  float r; \
  asm volatile(INIT "s_mov_b32 s90, %1\n" \
    "1:\n" BODY BODY BODY BODY \
    "s_sub_co_u32 s90, s90, 1\n s_cmp_lg_u32 s90, 0\n s_cbranch_scc1 1b\n" \
    "v_add_f32 %0, v40, v71\n" : "=v"(r) : "s"(iters) : CLOB); \
  out[blockIdx.x * blockDim.x + threadIdx.x] = r; \
}
KERNEL(kA, BODY_A)
KERNEL(kB, BODY_B)
KERNEL(kC, BODY_C)

int main(int argc, char** argv) {
  const int waves = argc > 1 ? atoi(argv[1]) : 4;
  const int iters = argc > 2 ? atoi(argv[2]) : 20000;
  hipDeviceProp_t p;
  HC(hipGetDeviceProperties(&p, 0));
  const int cus = p.multiProcessorCount;
  const double mhz = p.clockRate / 1000.0;
  float* out;
  HC(hipMalloc(&out, (size_t)cus * waves * 32 * 4));
  hipEvent_t a, b;
  HC(hipEventCreate(&a));
  HC(hipEventCreate(&b));
  void (*ks[3])(float*, int) = {kA, kB, kC};
  const char* names[3] = {"A 32 fmac", "B 16 v_dual (32 fmac)", "C 8x4 dependent fmac"};
  const int insts[3] = {128, 64, 128};
  for (int k = 0; k < 3; ++k) {
    hipLaunchKernelGGL(ks[k], dim3(cus), dim3(waves * 32), 0, 0, out, 10);
    HC(hipDeviceSynchronize());
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
      HC(hipEventRecord(a));
      hipLaunchKernelGGL(ks[k], dim3(cus), dim3(waves * 32), 0, 0, out, iters);
      HC(hipEventRecord(b));
      HC(hipEventSynchronize(b));
      float ms;
      HC(hipEventElapsedTime(&ms, a, b));
      if (ms < best) best = ms;
    }
    const double ns_iter = best * 1e6 / iters;
    const double cyc_iter = ns_iter * mhz / 1000.0;
    printf("%-26s waves/CU %d: %.1f ns/iter = %.0f cycles/iter at %.0f MHz -> %.2f cycles per issued instruction, %.2f cycles per fmac (per wave)\n",
           names[k], waves, ns_iter, cyc_iter, mhz, cyc_iter / insts[k], cyc_iter / 128.0);
  }
  printf("VOPD_BENCH_OK\n");
  return 0;
}
