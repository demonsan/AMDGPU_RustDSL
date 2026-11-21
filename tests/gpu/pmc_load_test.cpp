// Container loader check (tools/mi450/pmc_run.sh bring-up): runs a HIP kernel of its own, then loads a code object by file and from
// memory and prints the HIP status of each.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>

__global__ void k(int* p) { p[threadIdx.x] = threadIdx.x; }

int main(int argc, char** argv) {
    int* d = nullptr;
    printf("malloc %s\n", hipGetErrorString(hipMalloc(&d, 256 * sizeof(int))));
    k<<<1, 64>>>(d);
    printf("own kernel %s\n", hipGetErrorString(hipDeviceSynchronize()));
    if (argc < 2) return 0;
    hipModule_t m;
    printf("hipModuleLoad %s\n", hipGetErrorString(hipModuleLoad(&m, argv[1])));
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    printf("bytes %zu\n", b.size());
    printf("hipModuleLoadData %s\n", hipGetErrorString(hipModuleLoadData(&m, b.data())));
    return 0;
}
