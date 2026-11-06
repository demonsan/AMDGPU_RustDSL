// probe_ovl host: one block, prints the per-wave loop cycles (s_memtime) of the probe kernel.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    const char* co = argv[1]; const char* kn = argv[2]; int waves = std::atoi(argv[3]); int blocks = argc > 4 ? std::atoi(argv[4]) : 1;
    std::vector<unsigned char> hi(64 * 32);
    for (size_t i = 0; i < hi.size(); ++i) hi[i] = (unsigned char)(0x30 + (i * 7) % 16);
    void *di, *dout; CK(hipMalloc(&di, hi.size())); CK(hipMalloc(&dout, 4096 * blocks)); CK(hipMemcpy(di, hi.data(), hi.size(), hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, kn));
    struct { void* o; void* i; } args{dout, di}; size_t sz = sizeof(args);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    for (int r = 0; r < 3; ++r) { CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 64 * waves, 1, 1, 0, nullptr, nullptr, cfg)); CK(hipDeviceSynchronize()); }
    std::vector<unsigned> ho(64); CK(hipMemcpy(ho.data(), dout, 256, hipMemcpyDeviceToHost));
    double s = 0; for (int w = 0; w < waves; ++w) s += ho[w];
    std::printf("%s waves=%d mean_cycles_per_wave=%.0f (w0 %u)\n", kn, waves, s / waves, ho[0]);
}