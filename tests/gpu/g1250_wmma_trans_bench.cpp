// Do TRANS ops (v_exp_f32) issue in the shadow of WMMA execution on gfx1250? Loop bodies (per wave, one wave per SIMD by default):
//   W  : 8 v_wmma_f32_16x16x64_fp8_fp8, two independent accumulator chains
//   E  : 16 v_exp_f32 on independent registers
//   WE : the same 8 WMMAs with the 16 exps interleaved (2 after each WMMA)
//   WV : the 8 WMMAs with 16 plain VALU ops (v_mul_f32) interleaved (reference: a non-TRANS VALU in the shadow)
// If WE ~= W, the exps hide in the WMMA shadow. argv: waves per workgroup (default 4), iterations.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HC(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

#define CLOB "v40","v41","v42","v43","v44","v45","v46","v47","v48","v49","v50","v51","v52","v53","v54","v55","v56","v57","v58","v59","v60","v61","v62","v63", \
  "v64","v65","v66","v67","v68","v69","v70","v71","v72","v73","v74","v75","v76","v77","v78","v79","v80","v81","v82","v83","v84","v85","v86","v87", \
  "v88","v89","v90","v91","v92","v93","v94","v95","v96","s90","scc"

#define Z8(a) "v_mov_b32 v" #a ", 0\n"
#define INIT Z8(40) Z8(41) Z8(42) Z8(43) Z8(44) Z8(45) Z8(46) Z8(47) Z8(48) Z8(49) Z8(50) Z8(51) Z8(52) Z8(53) Z8(54) Z8(55) \
  Z8(56) Z8(57) Z8(58) Z8(59) Z8(60) Z8(61) Z8(62) Z8(63) Z8(64) Z8(65) Z8(66) Z8(67) Z8(68) Z8(69) Z8(70) Z8(71) \
  Z8(72) Z8(73) Z8(74) Z8(75) Z8(76) Z8(77) Z8(78) Z8(79) "v_mov_b32 v96, 0.5\n"

#define W1 "v_wmma_f32_16x16x64_fp8_fp8 v[40:47], v[56:63], v[64:71], v[40:47]\n"
#define W2 "v_wmma_f32_16x16x64_fp8_fp8 v[48:55], v[56:63], v[64:71], v[48:55]\n"
#define E(r) "v_exp_f32 v" #r ", v96\n"
#define M(r) "v_mul_f32 v" #r ", v96, v96\n"
#define BODY_W W1 W2 W1 W2 W1 W2 W1 W2
#define BODY_E E(80) E(81) E(82) E(83) E(84) E(85) E(86) E(87) E(88) E(89) E(90) E(91) E(92) E(93) E(94) E(95)
#define BODY_WE W1 E(80) E(81) W2 E(82) E(83) W1 E(84) E(85) W2 E(86) E(87) W1 E(88) E(89) W2 E(90) E(91) W1 E(92) E(93) W2 E(94) E(95)
#define BODY_WV W1 M(80) M(81) W2 M(82) M(83) W1 M(84) M(85) W2 M(86) M(87) W1 M(88) M(89) W2 M(90) M(91) W1 M(92) M(93) W2 M(94) M(95)

#define KERNEL(name, BODY) \
__global__ void name(float* out, int iters) { \
  float r; \
  asm volatile(INIT "s_mov_b32 s90, %1\n" \
    "1:\n" BODY BODY \
    "s_sub_co_u32 s90, s90, 1\n s_cmp_lg_u32 s90, 0\n s_cbranch_scc1 1b\n" \
    "s_nop 7\n v_add_f32 %0, v40, v95\n" : "=v"(r) : "s"(iters) : CLOB); \
  out[blockIdx.x * blockDim.x + threadIdx.x] = r; \
}
KERNEL(kW, BODY_W)
KERNEL(kE, BODY_E)
KERNEL(kWE, BODY_WE)
KERNEL(kWV, BODY_WV)

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
  void (*ks[4])(float*, int) = {kW, kE, kWE, kWV};
  const char* names[4] = {"W  8 wmma", "E  16 exp", "WE 8 wmma + 16 exp", "WV 8 wmma + 16 mul"};
  for (int k = 0; k < 4; ++k) {
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
    const double cyc = best * 1e6 / iters * mhz / 1000.0 / 2.0;   // per body (the loop holds two)
    printf("%-22s waves/WG %d: %.0f cycles per body\n", names[k], waves, cyc);
  }
  printf("WMMA_TRANS_BENCH_OK\n");
  return 0;
}
