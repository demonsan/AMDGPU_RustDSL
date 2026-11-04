// stamp_probe_host: run the probe_stamp kernel (crates/amdgpu-dsl kernels::probe_stamp) and print its per-wave stamp deltas.
// usage: stamp_probe_host probe_stamp.co [workgroups=256] [dump.u32]
#include "stamp_host.h"
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s probe_stamp.co [workgroups] [dump.u32]\n", argv[0]); return 2; }
    int wgs = argc > 2 ? std::atoi(argv[2]) : 256;
    hipModule_t mod; hipFunction_t f;
    if (hipModuleLoad(&mod, argv[1]) != hipSuccess || hipModuleGetFunction(&f, mod, "probe_stamp") != hipSuccess) { std::printf("load failed\n"); return 1; }
    StampBuffer sb(wgs, 4, 8);
    struct { void* p; } args = { sb.ptr() };
    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };
    for (int r = 0; r < 3; ++r) {
        hipMemset(sb.ptr(), 0, (size_t)wgs * 4 * 8 * 4);
        if (hipModuleLaunchKernel(f, wgs, 1, 1, 256, 1, 1, 0, 0, nullptr, cfg) != hipSuccess) { std::printf("launch failed\n"); return 1; }
        hipDeviceSynchronize();
    }
    sb.read();
    sb.summary(stdout);
    std::printf("slot 4 (x, keeps the chain alive) wave 0: %u   slot 5 (y): %u\n", sb.at(0, 0, 4), sb.at(0, 0, 5));
    if (argc > 3) std::printf("dump: %s\n", sb.dump(argv[3]) ? "written" : "FAILED");
    return 0;
}