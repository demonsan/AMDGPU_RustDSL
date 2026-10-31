// MUBUF probes: probe_buffer_rw (descriptor loads/stores vs the FLAT path, immediates, SGPR soffset, out-of-range behaviour) and probe_buffer_lds
// (buffer_load ... lds vs global_load_lds). usage: buffer_host rw|lds kernel.co
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
static int cmp(const char* what, const uint8_t* a, const uint8_t* b, size_t n) {
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) { if (first == (size_t)-1) first = i; ++bad; }
    std::printf("  %-44s %s%s", what, bad ? "MISMATCH " : "ok", "");
    if (bad) std::printf("(%zu bytes differ, first at +%zu: got %02x want %02x)", bad, first, a[first], b[first]);
    std::printf("\n");
    return bad != 0;
}
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s rw|lds kernel.co\n", argv[0]); return 2; }
    const bool rw = std::strcmp(argv[1], "rw") == 0;
    const char* kname = rw ? "probe_buffer_rw" : "probe_buffer_lds";
    std::vector<uint32_t> in(4096);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (uint32_t)(i * 2654435761u + 7u);
    std::vector<uint8_t> out(16384, 0xA5);
    void *din, *dout; CK(hipMalloc(&din, 16384)); CK(hipMalloc(&dout, 16384));
    CK(hipMemcpy(din, in.data(), in.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dout, out.data(), out.size(), hipMemcpyHostToDevice));
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, argv[2])); CK(hipModuleGetFunction(&fn, mod, kname));
    int soff = rw ? 3072 : 2048;
    void *a0 = din, *a1 = dout; void* params[] = {&a0, &a1, &soff};
    CK(hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, params, nullptr)); CK(hipDeviceSynchronize());
    CK(hipMemcpy(out.data(), dout, out.size(), hipMemcpyDeviceToHost));
    const uint8_t* I = reinterpret_cast<const uint8_t*>(in.data());
    int bad = 0;
    if (rw) {
        bad += cmp("buffer x4 copy        == in[0..1024)", &out[0], I, 1024);
        bad += cmp("buffer x2 (imm 1024)  == in[1024..1536)", &out[1024], I + 1024, 512);
        bad += cmp("buffer x1 (imm 2048)  == in[2048..2304)", &out[1536], I + 2048, 256);
        bad += cmp("buffer x1 (SGPR soff) == in[3072..3328)", &out[2048], I + 3072, 256);
        bad += cmp("FLAT x4 copy          == in[0..1024)", &out[4096], I, 1024);
        bad += cmp("FLAT x2               == in[1024..1536)", &out[4096 + 1024], I + 1024, 512);
        bad += cmp("FLAT x1               == in[2048..2304)", &out[4096 + 1536], I + 2048, 256);
        bad += cmp("FLAT x1 (offset)      == in[3072..3328)", &out[4096 + 2048], I + 3072, 256);
        bad += cmp("buffer == FLAT (x4)", &out[0], &out[4096], 1024);
        // out of range: sized descriptor (512 B): lanes 0..31 read data, lanes 32..63 read ZERO
        bad += cmp("OOB load, in-range lanes (0..31)", &out[8192], I, 512);
        std::vector<uint8_t> zeros(512, 0);
        bad += cmp("OOB load, out-of-range lanes (32..63) == 0", &out[8192 + 512], zeros.data(), 512);
        // OOB store: lanes 32..63 dropped (sentinel survives)
        bad += cmp("OOB store, in-range lanes (0..31)", &out[9216], I, 512);
        std::vector<uint8_t> sent(512, 0xA5);
        bad += cmp("OOB store, out-of-range lanes keep the sentinel", &out[9216 + 512], sent.data(), 512);
    } else {
        bad += cmp("buffer lds x4  slot0 == in[0..1024)", &out[0], I, 1024);
        bad += cmp("buffer lds x4 +SGPR soffset == in[2048..3072)", &out[1024], I + 2048, 1024);
        bad += cmp("buffer lds x4 imm 3072 == in[3072..4096)", &out[2048], I + 3072, 1024);
        bad += cmp("FLAT lds x4 == in[0..1024)", &out[3072], I, 1024);
        bad += cmp("FLAT lds x4 soffset-equivalent == in[2048..3072)", &out[4096], I + 2048, 1024);
        bad += cmp("FLAT lds x4 imm 3072 == in[3072..4096)", &out[5120], I + 3072, 1024);
        bad += cmp("buffer lds x1 == in[0..256)", &out[6144], I, 256);
        bad += cmp("FLAT lds x1 == in[0..256)", &out[6400], I, 256);
        bad += cmp("buffer == FLAT (x4 slot0)", &out[0], &out[3072], 1024);
    }
    std::printf("%s\n", bad ? "BUFFER_FAIL" : "BUFFER_OK");
    return bad != 0;
}