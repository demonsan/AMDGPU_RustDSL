// Generic harness: mbhost x.co name1,name2,... [waves list e.g. 1,4,8] [reps]
// One workgroup of waves*64 threads; per-wave record at out + wave*512 bytes:
//   dw0 = s_memtime delta, dw1 = s_memrealtime delta, dw2 = mark delta, dw3 = check value, dw8.. extras
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); exit(1);} } while (0)
static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> r; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) r.push_back(t);
    return r;
}
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) return 1;
    auto names = split(argv[2]);
    auto waves = split(argc > 3 ? argv[3] : "1,4,8");
    int reps = argc > 4 ? atoi(argv[4]) : 3;
    hipModule_t mod; CK(hipModuleLoad(&mod, argv[1]));
    const size_t OUT = 16 * 512, IN = 1 << 20;
    void *dout, *din; CK(hipMalloc(&dout, OUT)); CK(hipMalloc(&din, IN)); CK(hipMemset(din, 0, IN));
    for (auto& nm : names) {
        hipFunction_t fn;
        if (hipModuleGetFunction(&fn, mod, nm.c_str()) != hipSuccess) { printf("MISSING %s\n", nm.c_str()); continue; }
        for (auto& ws : waves) {
            int w = atoi(ws.c_str());
            for (int r = 0; r < reps; ++r) {
                CK(hipMemset(dout, 0, OUT));
                void* a0 = dout; void* a1 = din; void* params[] = {&a0, &a1};
                // kernarg as packed struct
                struct { void* o; void* i; } ka = {dout, din}; size_t sz = sizeof(ka);
                void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ka, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
                hipError_t e = hipModuleLaunchKernel(fn, 1, 1, 1, 64 * w, 1, 1, 0, nullptr, nullptr, cfg);
                if (e == hipSuccess) e = hipDeviceSynchronize();
                if (e != hipSuccess) { printf("ERR %s w=%d: %s\n", nm.c_str(), w, hipGetErrorString(e)); break; }
                std::vector<unsigned> h(OUT / 4); CK(hipMemcpy(h.data(), dout, OUT, hipMemcpyDeviceToHost));
                printf("R %s %d %d |", nm.c_str(), w, r);
                for (int k = 0; k < w; ++k) printf(" %u/%u/%u/%08x", h[k*128], h[k*128+1], h[k*128+2], h[k*128+3]);
                printf("\n");
                if (nm.find("haz") != std::string::npos) {
                    printf("X %s %d %d |", nm.c_str(), w, r);
                    for (int k = 0; k < 28; ++k) printf(" %08x", h[8 + k]);
                    printf("\n");
                }
            }
        }
    }
    return 0;
}


