// Generic byte-level probe harness (docs/op-coverage-tileops.md §6.5): kernel(in, out) with one block of THREADS threads.
//   tileop_probe_host <kernel.co> <kernel name> <threads> <input.bin> <out bytes> <output.bin>
// The output buffer is pre-filled with 0xA5 so bytes the kernel does not write are visible.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <co> <kernel> <threads> <input.bin> <out bytes> <output.bin>\n", argv[0]);
        return 2;
    }
    const char* co = argv[1];
    const char* name = argv[2];
    const int threads = std::atoi(argv[3]);
    FILE* fi = std::fopen(argv[4], "rb");
    if (!fi) {
        std::perror("input");
        return 2;
    }
    std::vector<unsigned char> in;
    unsigned char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, fi)) > 0) in.insert(in.end(), buf, buf + n);
    std::fclose(fi);
    const size_t out_bytes = (size_t)std::atol(argv[5]);
    std::vector<unsigned char> out(out_bytes, 0xA5);

    void *din = nullptr, *dout = nullptr;
    CK(hipMalloc(&din, in.size() + 4096));
    CK(hipMalloc(&dout, out_bytes + 4096));
    CK(hipMemcpy(din, in.data(), in.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dout, out.data(), out_bytes, hipMemcpyHostToDevice));
    hipModule_t mod;
    hipFunction_t fn;
    CK(hipModuleLoad(&mod, co));
    CK(hipModuleGetFunction(&fn, mod, name));
    struct {
        void* in;
        void* out;
    } args{din, dout};
    size_t sz = sizeof(args);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dout, out_bytes, hipMemcpyDeviceToHost));
    FILE* fo = std::fopen(argv[6], "wb");
    std::fwrite(out.data(), 1, out_bytes, fo);
    std::fclose(fo);
    std::printf("%s: ran %d threads, %zu input bytes, %zu output bytes\n", name, threads, in.size(), out_bytes);
    return 0;
}
