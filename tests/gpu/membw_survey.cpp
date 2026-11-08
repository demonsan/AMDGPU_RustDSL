// streaming bandwidth survey on MI355X: copy / read-only reduce / fill; vector width, unroll, grid-stride vs per-tile, nontemporal
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x) do { hipError_t e_=(x); if(e_!=hipSuccess){ printf("%s: %s\n", #x, hipGetErrorString(e_)); exit(1);} } while(0)
template<int W> struct V;                 // W dwords per thread per access
template<> struct V<1> { typedef unsigned t; };
template<> struct V<2> { typedef uint2 t; };
template<> struct V<4> { typedef uint4 t; };
template<typename T> __device__ __forceinline__ T ld(const T* p, bool nt) { return nt ? __builtin_nontemporal_load(p) : *p; }
template<typename T> __device__ __forceinline__ void st(T* p, T v, bool nt) { if (nt) __builtin_nontemporal_store(v, p); else *p = v; }
// copy: each block handles U vectors per thread per step; grid-stride when gs
template<int W, int U, bool NT, bool GS>
__global__ void copyk(const typename V<W>::t* __restrict__ in, typename V<W>::t* __restrict__ out, size_t nvec) {
  typedef typename V<W>::t T;
  size_t tile = (size_t)256 * U;
  size_t step = GS ? (size_t)gridDim.x * tile : 0;
  for (size_t base = (size_t)blockIdx.x * tile; base < nvec; base += (GS ? step : nvec)) {
    T r[U];
#pragma unroll
    for (int u = 0; u < U; ++u) { size_t i = base + (size_t)u * 256 + threadIdx.x; if (i < nvec) r[u] = ld(in + i, NT); }
#pragma unroll
    for (int u = 0; u < U; ++u) { size_t i = base + (size_t)u * 256 + threadIdx.x; if (i < nvec) st(out + i, r[u], NT); }
    if (!GS) break;
  }
}
template<int W, int U, bool GS>
__global__ void redk(const typename V<W>::t* __restrict__ in, unsigned* __restrict__ out, size_t nvec) {
  typedef typename V<W>::t T;
  size_t tile = (size_t)256 * U; size_t step = GS ? (size_t)gridDim.x * tile : 0;
  unsigned acc = 0;
  for (size_t base = (size_t)blockIdx.x * tile; base < nvec; base += (GS ? step : nvec)) {
    T r[U];
#pragma unroll
    for (int u = 0; u < U; ++u) { size_t i = base + (size_t)u * 256 + threadIdx.x; if (i < nvec) r[u] = in[i]; }
#pragma unroll
    for (int u = 0; u < U; ++u) { size_t i = base + (size_t)u * 256 + threadIdx.x; if (i < nvec) { unsigned* q = (unsigned*)&r[u]; for (int k = 0; k < W; ++k) acc ^= q[k]; } }
    if (!GS) break;
  }
  if (acc == 0x12345678u) out[blockIdx.x] = acc;   // keep alive
}
template<int W, int U, bool NT, bool GS>
__global__ void fillk(typename V<W>::t* __restrict__ out, size_t nvec) {
  typedef typename V<W>::t T; T z; unsigned* q = (unsigned*)&z; for (int k = 0; k < W; ++k) q[k] = 7;
  size_t tile = (size_t)256 * U; size_t step = GS ? (size_t)gridDim.x * tile : 0;
  for (size_t base = (size_t)blockIdx.x * tile; base < nvec; base += (GS ? step : nvec)) {
#pragma unroll
    for (int u = 0; u < U; ++u) { size_t i = base + (size_t)u * 256 + threadIdx.x; if (i < nvec) st(out + i, z, NT); }
    if (!GS) break;
  }
}
static double med(std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
template<typename F> double timeit(F f, int reps = 15) {
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) f();
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) { CK(hipEventRecord(a)); f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); t.push_back(ms); }
  return med(t) * 1e-3;
}
int main(int argc, char** argv) {
  size_t bytes = (size_t)(argc > 1 ? atoi(argv[1]) : 1024) << 20;
  void *din, *dout; unsigned* dr; CK(hipMalloc(&din, bytes)); CK(hipMalloc(&dout, bytes)); CK(hipMalloc(&dr, 1 << 20));
  CK(hipMemset(din, 1, bytes)); CK(hipMemset(dout, 0, bytes));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0)); printf("%s CUs=%d bytes=%zu MiB\n", p.name, p.multiProcessorCount, bytes >> 20);
  { double s = timeit([&] { CK(hipMemcpyAsync(dout, din, bytes, hipMemcpyDeviceToDevice)); }); printf("hipMemcpy D2D            %8.1f GB/s (read+write counted: %.1f)\n", bytes / s / 1e9, 2.0 * bytes / s / 1e9); }
  const int CU = p.multiProcessorCount;
#define COPY(W, U, NT, GS, GRIDMUL) do { size_t nvec = bytes / (4 * W); size_t tile = 256 * U; int grid = GS ? CU * GRIDMUL : (int)((nvec + tile - 1) / tile); \
    double s = timeit([&] { copyk<W, U, NT, GS><<<grid, 256>>>((const V<W>::t*)din, (V<W>::t*)dout, nvec); }); \
    printf("copy w=%2dB U=%d %s %s grid=%-6d %8.1f GB/s (r+w)\n", 4 * W, U, NT ? "nt " : "   ", GS ? "gridstride" : "per-tile  ", grid, 2.0 * bytes / s / 1e9); } while (0)
  COPY(1, 4, false, false, 1); COPY(2, 4, false, false, 1); COPY(4, 1, false, false, 1); COPY(4, 2, false, false, 1); COPY(4, 4, false, false, 1); COPY(4, 8, false, false, 1);
  COPY(4, 4, true, false, 1); COPY(4, 2, true, false, 1); COPY(4, 8, true, false, 1);
  COPY(4, 4, false, true, 1); COPY(4, 4, false, true, 2); COPY(4, 4, false, true, 4); COPY(4, 4, false, true, 8); COPY(4, 4, false, true, 16);
  COPY(4, 4, true, true, 4); COPY(4, 8, true, true, 4); COPY(4, 8, false, true, 4); COPY(4, 2, false, true, 8);
#define RED(W, U, GS, GRIDMUL) do { size_t nvec = bytes / (4 * W); size_t tile = 256 * U; int grid = GS ? CU * GRIDMUL : (int)((nvec + tile - 1) / tile); \
    double s = timeit([&] { redk<W, U, GS><<<grid, 256>>>((const V<W>::t*)din, dr, nvec); }); \
    printf("read w=%2dB U=%d %s grid=%-6d %8.1f GB/s (read only)\n", 4 * W, U, GS ? "gridstride" : "per-tile  ", grid, bytes / s / 1e9); } while (0)
  RED(4, 4, false, 1); RED(4, 8, false, 1); RED(4, 4, true, 4); RED(4, 8, true, 4); RED(4, 8, true, 8); RED(2, 8, true, 4); RED(1, 8, true, 4);
#define FILL(W, U, NT, GS, GRIDMUL) do { size_t nvec = bytes / (4 * W); size_t tile = 256 * U; int grid = GS ? CU * GRIDMUL : (int)((nvec + tile - 1) / tile); \
    double s = timeit([&] { fillk<W, U, NT, GS><<<grid, 256>>>((V<W>::t*)dout, nvec); }); \
    printf("fill w=%2dB U=%d %s %s grid=%-6d %8.1f GB/s (write only)\n", 4 * W, U, NT ? "nt " : "   ", GS ? "gridstride" : "per-tile  ", grid, bytes / s / 1e9); } while (0)
  FILL(4, 4, false, false, 1); FILL(4, 4, true, false, 1); FILL(4, 4, false, true, 4); FILL(4, 8, true, true, 4);
  return 0;
}