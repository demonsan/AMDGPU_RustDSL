// Launch a sched_* microkernel (see kernels.rs): sched_host <co> <kernel> <block> <grid> <n> [reps]
// kernarg: {void* o; int n} ; prints min ms over reps, cycles/iteration at CLK_GHZ (default 2.4) and an output hash.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)
int main(int argc, char** argv) {
  if (argc < 6) { std::printf("usage\n"); return 2; }
  const char* co = argv[1]; const char* name = argv[2];
  int block = atoi(argv[3]), grid = atoi(argv[4]), n = atoi(argv[5]);
  int reps = argc > 6 ? atoi(argv[6]) : 10;
  double ghz = getenv("CLK_GHZ") ? atof(getenv("CLK_GHZ")) : 2.4;
  hipModule_t mod; hipFunction_t fn;
  CK(hipModuleLoad(&mod, co));
  CK(hipModuleGetFunction(&fn, mod, name));
  size_t obytes = 1 << 20; void* d; CK(hipMalloc(&d, obytes)); CK(hipMemset(d, 0, obytes));
  struct { void* o; int n; int pad; } args = {d, n, 0};
  size_t asz = sizeof(args);
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &asz, HIP_LAUNCH_PARAM_END};
  for (int w = 0; w < 3; ++w) CK(hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, nullptr, extra));
  CK(hipDeviceSynchronize());
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  float best = 1e30f;
  for (int r = 0; r < reps; ++r) {
    CK(hipEventRecord(e0));
    CK(hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, nullptr, extra));
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
  }
  std::vector<uint32_t> out(obytes / 4); CK(hipMemcpy(out.data(), d, obytes, hipMemcpyDeviceToHost));
  uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < 4096 * 4 && i < out.size(); ++i) { h ^= out[i]; h *= 1099511628211ull; }
  double cyc = best * 1e-3 * ghz * 1e9;
  if (getenv("SH_DUMP")) { int nd = atoi(getenv("SH_DUMP")); for (int i = 0; i < nd; ++i) { float f; std::memcpy(&f, &out[i], 4); std::printf("out[%d]=%08x(%g) ", i, out[i], f); } std::printf("\n"); }
  std::printf("RESULT kernel=%s block=%d n=%d ms=%.4f cycles_per_iter=%.1f hash=%016llx\n", name, block, n, best, cyc / n, (unsigned long long)h);
  return 0;
}