// Hardware check of the gfx1250 WMMA atoms (crates/amdgpu-isa/src/gfx1250.rs) with the probes `probe_g1250_wmma_*`
// (crates/amdgpu-kernels/src/probes/g1250.rs): every lane's A / B / C fragments are written per lane (lane l's dwords at l * n), filled from
// random matrices through the atom layouts below (the same formulas as the atoms, CDNA5 ISA 7.12), and the result is compared with
// D = A x B (x scales) + C computed on the CPU, mapped through the C layout. A wrong layout shows up as wrong elements.
//
//   g1250_wmma_probe_host <co> <kernel> <fp8|fp8k128|f4> <with_c 0|1> [deadline_ms]
// Kernarg: pointers a, b, c, d, sa, sb (48 bytes). One workgroup of 32 lanes. Prints `WMMA_OK` / `WMMA_FAIL ...` and a JSON line.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#define FAIL(...) do { std::printf("WMMA_FAIL "); std::printf(__VA_ARGS__); std::printf("\n"); std::fflush(stdout); _exit(2); } while (0)
#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) FAIL("%s: %s", #x, hipGetErrorString(e_)); } while (0)

static float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}
static float fp4(uint8_t n) {
    static const float t[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t[n & 15];
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s co kernel fp8|fp8k128|f4 with_c [deadline_ms]\n", argv[0]); return 1; }
    const std::string kind = argv[3];
    const bool with_c = atoi(argv[4]) != 0;
    const long deadline = argc > 5 ? atol(argv[5]) : 10000;
    const bool f4 = kind == "f4";
    const int M = f4 ? 32 : 16, N = 16, K = kind == "fp8" ? 64 : 128;
    const int na = f4 ? 16 : K / 8, nb = f4 ? 8 : K / 8, nc = f4 ? 16 : 8;  // dwords per lane
    std::mt19937 rng(1);

    // logical matrices: element codes (fp8 byte / fp4 nibble), C in f32, E8M0 scales per (row, k / 32)
    std::vector<uint8_t> A(M * K), B(N * K), SA(M * 4, 127), SB(N * 4, 127);
    auto code = [&]() -> uint8_t {
        if (f4) return rng() & 15;
        uint8_t b;
        do { b = rng() & 0xFF; } while (((b >> 3) & 15) > 9 || ((b >> 3) & 15) < 3);
        return b;
    };
    for (auto& x : A) x = code();
    for (auto& x : B) x = code();
    if (f4) { for (auto& x : SA) x = 125 + rng() % 5; for (auto& x : SB) x = 125 + rng() % 5; }
    std::vector<float> C(M * N, 0.f);
    if (with_c) for (auto& x : C) x = (int)(rng() % 17) - 8;
    auto val = [&](bool isA, int r, int k) {
        const uint8_t c = isA ? A[r * K + k] : B[r * K + k];
        const float s = f4 ? std::ldexp(1.f, (isA ? SA[r * 4 + k / 32] : SB[r * 4 + k / 32]) - 127) : 1.f;
        return (f4 ? fp4(c) : e4m3(c)) * s;
    };

    // per-lane operand buffers through the atom layouts
    std::vector<uint32_t> la(32 * na, 0), lb(32 * nb, 0), lc(32 * nc, 0), lsa(32, 0), lsb(32, 0);
    for (int l = 0; l < 32; l++) {
        const int h = l / 16;
        if (!f4) {
            for (int isA = 0; isA < 2; isA++) {
                const int n = isA ? na : nb;
                for (int e = 0; e < 4 * n; e++) {
                    const int b = e % 4, v = e / 4;
                    const int k = b + 4 * (v % 2) + 8 * h + 16 * ((v / 2) % 2) + 32 * ((v / 4) % 2) + 64 * (v / 8);
                    const uint8_t c = isA ? A[(l % 16) * K + k] : B[(l % 16) * K + k];
                    (isA ? la : lb)[l * n + v] |= (uint32_t)c << (8 * b);
                }
            }
            for (int v = 0; v < 8; v++) {
                float x = C[(v + 8 * h) * N + l % 16];
                std::memcpy(&lc[l * nc + v], &x, 4);
            }
        } else {
            for (int e = 0; e < 8 * na; e++) {  // A 32 x 128: 16 dwords of 8 nibbles
                const int j = e % 8, v = e / 8;
                const int k = j + 8 * (v % 4) + 32 * h + 64 * ((v / 4) % 2), m = l % 16 + 16 * (v / 8);
                la[l * na + v] |= (uint32_t)(A[m * K + k] & 15) << (4 * j);
            }
            for (int e = 0; e < 8 * nb; e++) {  // B 16 x 128: 8 dwords
                const int j = e % 8, v = e / 8;
                const int k = j + 8 * (v % 4) + 32 * h + 64 * ((v / 4) % 2);
                lb[l * nb + v] |= (uint32_t)(B[(l % 16) * K + k] & 15) << (4 * j);
            }
            for (int v = 0; v < 16; v++) {
                float x = C[((v % 8) + 8 * h + 16 * (v / 8)) * N + l % 16];
                std::memcpy(&lc[l * nc + v], &x, 4);
            }
            for (int j = 0; j < 4; j++) {
                lsa[l] |= (uint32_t)SA[l * 4 + j] << (8 * j);
                if (l < 16) lsb[l] |= (uint32_t)SB[l * 4 + j] << (8 * j);
            }
        }
    }

    HC(hipSetDevice(0));
    auto up = [](const std::vector<uint32_t>& v) { void* d; HC(hipMalloc(&d, v.size() * 4)); HC(hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice)); return d; };
    void *da = up(la), *db = up(lb), *dc = up(lc), *dsa = up(lsa), *dsb = up(lsb), *dd;
    HC(hipMalloc(&dd, 32 * nc * 4));
    HC(hipMemset(dd, 0xFF, 32 * nc * 4));
    struct { void *a, *b, *c, *d, *sa, *sb; } ka{da, db, dc, dd, dsa, dsb};
    size_t ksz = sizeof(ka);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ka, HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksz, HIP_LAUNCH_PARAM_END};
    hipModule_t mod;
    hipFunction_t fn;
    HC(hipModuleLoad(&mod, argv[1]));
    HC(hipModuleGetFunction(&fn, mod, argv[2]));
    hipStream_t s;
    HC(hipStreamCreate(&s));
    hipEvent_t e1;
    HC(hipEventCreate(&e1));
    HC(hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, s, nullptr, cfg));
    HC(hipEventRecord(e1, s));
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        hipError_t q = hipEventQuery(e1);
        if (q == hipSuccess) break;
        if (q != hipErrorNotReady) FAIL("kernel: %s", hipGetErrorString(q));
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(deadline)) { std::printf("WMMA_HANG\n"); std::fflush(stdout); _exit(124); }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    std::vector<uint32_t> ld(32 * nc);
    HC(hipMemcpy(ld.data(), dd, ld.size() * 4, hipMemcpyDeviceToHost));

    long bad = 0;
    double maxe = 0;
    int shown = 0;
    for (int l = 0; l < 32; l++)
        for (int v = 0; v < nc; v++) {
            const int h = l / 16, n = l % 16, m = f4 ? (v % 8) + 8 * h + 16 * (v / 8) : v + 8 * h;
            double ref = C[m * N + n], mag = std::fabs(ref);
            for (int k = 0; k < K; k++) {
                const double p = (double)val(true, m, k) * val(false, n, k);
                ref += p;
                mag += std::fabs(p);
            }
            float got;
            std::memcpy(&got, &ld[l * nc + v], 4);
            // relative to the magnitude of the dot product (the hardware's fp8 accumulation is not exact fp32; a layout error is O(1))
            const double err = std::fabs(got - ref) / std::max(1.0, mag);
            maxe = std::max(maxe, err);
            if (!(err < 1e-3)) {
                bad++;
                if (shown++ < 6) std::printf("  lane %d value %d (m %d, n %d): got %g ref %g\n", l, v, m, n, got, ref);
            }
        }
    std::printf("wmma %s with_c=%d: checked=%d bad=%ld max_rel=%.2e %s\n", kind.c_str(), with_c, 32 * nc, bad, maxe, bad ? "WMMA_FAIL" : "WMMA_OK");
    std::printf("{\"host\":\"g1250_wmma_probe\",\"kernel\":\"%s\",\"kind\":\"%s\",\"bad\":%ld,\"ok\":%s}\n", argv[2], kind.c_str(), bad, bad ? "false" : "true");
    return bad ? 3 : 0;
}
