// Launch-contract check from a code object's own metadata (the AMDGPU note: a msgpack map per kernel with .kernarg_segment_size, .max_flat_workgroup_size, .name):
// a host refuses a launch whose kernarg blob is SMALLER than what the kernel reads (the kernel would read garbage: a persistent kernel launched without PERSIST reads its tile-loop
// kernargs past the blob -> an out-of-range address -> a memory fault) or whose work-group size differs from the one the kernel was built for (a role-split kernel with half its
// waves missing hangs at its first barrier). Own hosts call `g1250_check_launch` BEFORE the first launch; a failed check exits 2 and never touches the device.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

struct CoMeta { long kernarg = -1; long max_flat = -1; long cluster = 0; };   // cluster: work-groups per cluster the kernel was built for (0 = not a cluster kernel)

static inline bool co_msgpack_int(const std::vector<unsigned char>& d, size_t p, long* v) {
    if (p >= d.size()) return false;
    const unsigned char c = d[p];
    if (c < 0x80) { *v = c; return true; }
    if (c == 0xcc && p + 1 < d.size()) { *v = d[p + 1]; return true; }
    if (c == 0xcd && p + 2 < d.size()) { *v = (d[p + 1] << 8) | d[p + 2]; return true; }
    if (c == 0xce && p + 4 < d.size()) { *v = ((long)d[p + 1] << 24) | (d[p + 2] << 16) | (d[p + 3] << 8) | d[p + 4]; return true; }
    return false;
}

static inline size_t co_find(const std::vector<unsigned char>& d, size_t from, size_t to, const std::string& pat) {
    if (to > d.size()) to = d.size();
    if (pat.empty() || to < pat.size() || from > to - pat.size()) return (size_t)-1;
    for (size_t i = from; i + pat.size() <= to; ++i)
        if (d[i] == (unsigned char)pat[0] && memcmp(&d[i], pat.data(), pat.size()) == 0) return i;
    return (size_t)-1;
}

// the metadata of kernel `kname` in the code object `path` (kernarg = -1: not found)
static inline CoMeta co_meta(const char* path, const char* kname) {
    CoMeta m;
    FILE* f = fopen(path, "rb");
    if (!f) return m;
    std::vector<unsigned char> d;
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) d.insert(d.end(), buf, buf + n);
    fclose(f);
    const std::string key_ka = std::string(1, (char)(0xa0 | 21)) + ".kernarg_segment_size";
    const std::string key_mf = std::string(1, (char)(0xa0 | 24)) + ".max_flat_workgroup_size";
    const std::string key_nm = std::string(1, (char)(0xa0 | 5)) + ".name";
    const std::string key_cl = std::string(1, (char)(0xa0 | 19)) + ".cluster_workgroups";
    const size_t kl = strlen(kname);
    std::string name_val = kl < 32 ? std::string(1, (char)(0xa0 | kl)) : std::string("\xd9") + (char)kl;
    name_val += kname;
    size_t from = 0;
    for (;;) {
        const size_t p = co_find(d, from, d.size(), key_ka);
        if (p == (size_t)-1) break;
        from = p + 1;
        // the same kernel's map: its .name follows within a few hundred bytes (the keys are sorted)
        const size_t win = p + 900;
        const size_t pn = co_find(d, p, win, key_nm + name_val);
        if (pn == (size_t)-1) continue;
        // the next kernel's .kernarg_segment_size must not lie between: take the nearest kernarg key before the name
        const size_t pk2 = co_find(d, p + 1, pn, key_ka);
        if (pk2 != (size_t)-1) continue;
        long v = 0;
        if (co_msgpack_int(d, p + key_ka.size(), &v)) m.kernarg = v;
        const size_t pm = co_find(d, p, pn, key_mf);
        if (pm != (size_t)-1 && co_msgpack_int(d, pm + key_mf.size(), &v)) m.max_flat = v;
        // the cluster key sorts before .group_segment_fixed_size, i.e. before the kernarg key of the same map: look back a little
        const size_t back = p > 700 ? p - 700 : 0;
        size_t pc = (size_t)-1, at = back;
        for (;;) { const size_t q = co_find(d, at, p, key_cl); if (q == (size_t)-1) break; pc = q; at = q + 1; }
        if (pc != (size_t)-1 && co_msgpack_int(d, pc + key_cl.size(), &v)) m.cluster = v;
        break;
    }
    return m;
}

// refuse the launch (exit 2, nothing launched) when the contract recorded in the code object is violated; threads <= 0: do not check the work-group size
static inline void g1250_check_launch(const char* path, const char* kname, size_t provided_kargs, long threads, bool strict = true, bool exact = true) {
    const CoMeta m = co_meta(path, kname);
    if (m.kernarg < 0) {
        fprintf(stderr, "LAUNCH-%s: cannot read the metadata of kernel %s in %s\n", strict ? "REFUSED" : "WARNING", kname, path);
        if (strict) _exit(2);
        return;
    }
    if ((long)provided_kargs < m.kernarg) {
        fprintf(stderr, "LAUNCH-REFUSED: kernel %s reads %ld kernarg bytes, the host passes %zu (a persistent / swizzled / stamped kernel needs its option: PERSIST=G, SWIZZLE=G, STAMPS=1)\n", kname, m.kernarg, provided_kargs);
        _exit(2);
    }
    if (threads > 0 && m.max_flat >= 0 && (exact ? threads != m.max_flat : threads > m.max_flat)) {
        fprintf(stderr, "LAUNCH-REFUSED: kernel %s was built for %ld work-items per work-group, the host launches %ld (a role-split kernel would hang at its first barrier)\n", kname, m.max_flat, threads);
        _exit(2);
    }
}
// the cluster launch shape: a kernel built for a cluster of N work-groups must be launched with exactly N per cluster, and a plain kernel must not be (its work-group ids come from different registers)
static inline void g1250_check_cluster(const char* path, const char* kname, long launch_cluster /* work-groups per cluster the host launches (1 = none) */) {
    const CoMeta m = co_meta(path, kname);
    const long built = m.cluster > 1 ? m.cluster : 1;
    if (m.kernarg < 0) return;   // the metadata was not found: g1250_check_launch reports it
    if (built != launch_cluster) {
        fprintf(stderr, "LAUNCH-REFUSED: kernel %s was built for clusters of %ld work-groups, the host launches clusters of %ld\n", kname, built, launch_cluster);
        _exit(2);
    }
}