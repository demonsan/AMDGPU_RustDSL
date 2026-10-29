// Driver for the sparse-fp8 MFMA layout probe (cmake-free, mirrors probe_host.cpp).
// Loads build/vendor/probe_smfmac_ones.co, launches one 64-lane wave, prints the per-slot decode
// of D for every lane and a histogram per slot. `tools/vendor_smfmac_probe.sh` runs this and
// interprets the numbers; see docs/vendor-spec.md "smfmac sparse layout".
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static constexpr int kSlots = 16;   // must match the probe kernel's dump slots
static constexpr int kLanes = 64;

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* co = argc > 1 ? argv[1] : "build/vendor/probe_smfmac_ones.co";
    const int slots = argc > 2 ? std::atoi(argv[2]) : kSlots;

    std::vector<unsigned char> hout((size_t)slots * 1024, 0);
    void* dout = nullptr;
    CK(hipMalloc(&dout, hout.size()));
    CK(hipMemset(dout, 0, hout.size()));

    hipModule_t mod;
    hipFunction_t fn;
    CK(hipModuleLoad(&mod, co));
    CK(hipModuleGetFunction(&fn, mod, "probe_smfmac_ones"));
    void* a_o = dout;
    void* params[] = {&a_o};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, kLanes, 1, 1, 0, nullptr, params, nullptr));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(hout.data(), dout, hout.size(), hipMemcpyDeviceToHost));

    for (int e = 0; e < slots; ++e) {
        float vals[4 * kLanes];
        std::memcpy(vals, hout.data() + (size_t)e * 1024, sizeof(vals));
        // Per-dword statistics across lanes.
        std::printf("slot %2d:", e);
        for (int j = 0; j < 4; ++j) {
            double mn = 1e30, mx = -1e30, sum = 0;
            std::map<float, int> hist;
            for (int l = 0; l < kLanes; ++l) {
                const float v = vals[4 * l + j];
                mn = std::fmin(mn, (double)v);
                mx = std::fmax(mx, (double)v);
                sum += v;
                hist[v]++;
            }
            std::printf(" | d%d min=%.6g max=%.6g mean=%.6g distinct=%zu", j, mn, mx,
                        sum / kLanes, hist.size());
        }
        std::printf("\n");
        // Sample lanes: print the four dwords of lanes 0, 1, 31, 32, 63.
        for (int l : {0, 1, 2, 31, 32, 33, 63}) {
            std::printf("    slot %2d lane %2d: %.6g %.6g %.6g %.6g\n", e, l, vals[4 * l + 0],
                        vals[4 * l + 1], vals[4 * l + 2], vals[4 * l + 3]);
        }
    }
    if (const char* det = std::getenv("DETAIL")) {
        // DETAIL="slot[,slot...]" prints all 64 lanes x 4 dwords for those slots.
        std::vector<int> want;
        for (const char* q = det; *q;) {
            want.push_back(std::atoi(q));
            const char* comma = std::strchr(q, ',');
            if (!comma) break;
            q = comma + 1;
        }
        for (int e : want) {
            for (int l = 0; l < kLanes; ++l) {
                const float* v = (const float*)(hout.data() + (size_t)e * 1024) + 4 * l;
                std::printf("DETAIL slot %d lane %2d: %.9g %.9g %.9g %.9g\n", e, l, v[0], v[1], v[2],
                            v[3]);
            }
        }
    }
    std::printf("SMFMAC_PROBE_OK\n");
    return 0;
}
