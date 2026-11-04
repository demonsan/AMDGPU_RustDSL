// stamp_host.h - host side of the DSL timing stamps (FnBuilder::memtime_lo / stamp_rec_off / stamp_store).
// The kernel writes u32 stamps into a debug buffer laid out [workgroup][wave][slot]; this header allocates and zeroes it, reads it back and dumps it for tools/stamps.py.
//   StampBuffer sb(workgroups, waves_per_wg, slots);   // device buffer, zero-filled
//   ... pass sb.ptr() as the kernel's debug-buffer argument, launch, synchronize ...
//   sb.read(); sb.summary(stdout); sb.dump("stamps.u32");   // dump header: 4 x u32 = workgroups, waves, slots, magic 0x53544d50
// Stamps are the LOW 32 bits of s_memtime (core-clock ticks); deltas are computed modulo 2^32.
#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

struct StampBuffer {
    int wgs, waves, slots;
    void* dev = nullptr;
    std::vector<uint32_t> h;
    StampBuffer(int workgroups, int waves_per_wg, int slots_) : wgs(workgroups), waves(waves_per_wg), slots(slots_) {
        size_t n = (size_t)wgs * waves * slots * sizeof(uint32_t);
        if (hipMalloc(&dev, n) != hipSuccess || hipMemset(dev, 0, n) != hipSuccess) { std::fprintf(stderr, "StampBuffer: hip alloc failed\n"); dev = nullptr; }
        h.assign((size_t)wgs * waves * slots, 0);
    }
    ~StampBuffer() { if (dev) hipFree(dev); }
    void* ptr() const { return dev; }
    bool read() { return dev && hipMemcpy(h.data(), dev, h.size() * sizeof(uint32_t), hipMemcpyDeviceToHost) == hipSuccess; }
    uint32_t at(int wg, int wave, int slot) const { return h[((size_t)wg * waves + wave) * slots + slot]; }
    // per slot-to-slot interval over all waves that wrote both slots: median / min / max cycles (wrapping u32 subtraction)
    void summary(FILE* f) const {
        std::fprintf(f, "stamps: %d workgroups x %d waves x %d slots\n", wgs, waves, slots);
        for (int s = 1; s < slots; ++s) {
            std::vector<uint32_t> d;
            for (int g = 0; g < wgs; ++g)
                for (int w = 0; w < waves; ++w) {
                    uint32_t a = at(g, w, s - 1), b = at(g, w, s);
                    if (a && b) d.push_back(b - a);
                }
            if (d.empty()) continue;
            std::sort(d.begin(), d.end());
            std::fprintf(f, "  slot %2d - slot %2d: median %10u  min %10u  max %10u  (%zu waves)\n", s, s - 1, d[d.size() / 2], d.front(), d.back(), d.size());
        }
    }
    bool dump(const char* path) const {
        FILE* o = std::fopen(path, "wb");
        if (!o) return false;
        uint32_t hdr[4] = {(uint32_t)wgs, (uint32_t)waves, (uint32_t)slots, 0x53544d50u};
        std::fwrite(hdr, sizeof hdr, 1, o);
        std::fwrite(h.data(), sizeof(uint32_t), h.size(), o);
        std::fclose(o);
        return true;
    }
};