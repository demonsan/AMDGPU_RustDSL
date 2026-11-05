// ISA one-hot probes (docs/isa-probes-smfmac-tr.md): recovers the exact lane / register / byte contribution maps of
//   (a) v_smfmac_f32_16x16x128_fp8_fp8   (kernel probe_smfmac_gen)
//   (b) ds_read_b64_tr_b8                (kernel probe_lds_tr_b8_gen)
// on the GPU and writes them as text maps (the recorded maps that crates/amdgpu-isa/tests replays).
// usage: isa_probe_host <dir with probe_smfmac_gen.co and probe_lds_tr_b8_gen.co> <out dir>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s (line %d)\n", #x, hipGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

static int g_bad = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_bad; std::printf("CHECK FAILED (line %d): ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Dev { hipModule_t mod; hipFunction_t fn; };
static Dev load(const std::string& co, const char* name) { Dev d; CK(hipModuleLoad(&d.mod, co.c_str())); CK(hipModuleGetFunction(&d.fn, d.mod, name)); return d; }
template <class... P> static void launch(const Dev& d, int blocks, P... ptrs) {
    struct { void* p[sizeof...(P)]; } args = {{(void*)ptrs...}};
    size_t sz = sizeof(args);
    void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(d.fn, blocks, 1, 1, 64, 1, 1, 0, nullptr, nullptr, extra));
    CK(hipDeviceSynchronize());
}

// ================================================================ (a) smfmac
struct SmfProbe { uint8_t a[64 * 16]; uint8_t b[64 * 32]; uint32_t idx[64]; };
static const uint8_t ONE = 0x38;   // E4M3 1.0
static void a_one_hot(SmfProbe& p, int lane, int byte) { std::memset(p.a, 0, sizeof p.a); p.a[lane * 16 + byte] = ONE; }
static void a_ones(SmfProbe& p) { std::memset(p.a, ONE, sizeof p.a); }
static void b_ones(SmfProbe& p) { std::memset(p.b, ONE, sizeof p.b); }
static void b_one_hot(SmfProbe& p, int lane, int byte) { std::memset(p.b, 0, sizeof p.b); p.b[lane * 32 + byte] = ONE; }
static void idx_all(SmfProbe& p, uint32_t w) { for (int l = 0; l < 64; ++l) p.idx[l] = w; }

struct SmfRunner {
    Dev dev;
    void* da = nullptr; void* db = nullptr; void* di = nullptr; void* dd = nullptr;
    size_t cap = 0;
    // runs the probes; returns D (n * 256 floats: lane * 4 + dword)
    std::vector<float> run(const std::vector<SmfProbe>& ps) {
        const size_t n = ps.size();
        if (n > cap) {
            if (da) { hipFree(da); hipFree(db); hipFree(di); hipFree(dd); }
            CK(hipMalloc(&da, n * 1024)); CK(hipMalloc(&db, n * 2048)); CK(hipMalloc(&di, n * 256)); CK(hipMalloc(&dd, n * 1024));
            cap = n;
        }
        std::vector<uint8_t> A(n * 1024), B(n * 2048); std::vector<uint32_t> I(n * 64);
        for (size_t k = 0; k < n; ++k) { std::memcpy(&A[k * 1024], ps[k].a, 1024); std::memcpy(&B[k * 2048], ps[k].b, 2048); std::memcpy(&I[k * 64], ps[k].idx, 256); }
        CK(hipMemcpy(da, A.data(), A.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(db, B.data(), B.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(di, I.data(), I.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemset(dd, 0, n * 1024));
        launch(dev, (int)n, da, db, di, dd);
        std::vector<float> D(n * 256);
        CK(hipMemcpy(D.data(), dd, D.size() * 4, hipMemcpyDeviceToHost));
        return D;
    }
};
typedef std::array<uint64_t, 4> Bits;   // 256 D positions: lane * 4 + dword
static Bits nz(const float* d) { Bits b{}; for (int i = 0; i < 256; ++i) if (std::fabs(d[i]) > 0.5f) b[i / 64] |= 1ull << (i % 64); return b; }
static int popc(const Bits& b) { int c = 0; for (auto w : b) c += __builtin_popcountll(w); return c; }
static bool has(const Bits& b, int i) { return (b[i / 64] >> (i % 64)) & 1; }

static const uint32_t NIB[6] = {0x4, 0x8, 0xc, 0x9, 0xd, 0xe};   // (p0,p1) = (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
static uint32_t uniform(uint32_t nib) { return nib * 0x11111111u; }

static void smfmac_part(const std::string& dir, FILE* out) {
    SmfRunner R; R.dev = load(dir + "/probe_smfmac_gen.co", "probe_smfmac_gen");
    std::printf("== (a) v_smfmac_f32_16x16x128_fp8_fp8\n");
    // ---- sanity: all ones, vendor pattern 0x44444444 -> every D element = 64 (64 selected products of 1)
    {
        std::vector<SmfProbe> ps(1); a_ones(ps[0]); b_ones(ps[0]); idx_all(ps[0], 0x44444444);
        auto D = R.run(ps); int bad = 0; for (int i = 0; i < 256; ++i) if (D[i] != 64.0f) ++bad;
        std::printf("sanity (A ones, B ones, idx 0x44444444): D = %g .. %g, %d elements != 64\n", *std::min_element(D.begin(), D.end()), *std::max_element(D.begin(), D.end()), bad);
        CHECK(bad == 0, "all-ones smfmac must give 64 everywhere");
    }
    // ---- stage 1: A one-hot (1024) x B ones, idx 0x44444444 -> D positions fed by each A byte
    std::vector<Bits> S(1024);
    {
        std::vector<SmfProbe> ps(1024);
        for (int a = 0; a < 1024; ++a) { a_one_hot(ps[a], a / 16, a % 16); b_ones(ps[a]); idx_all(ps[a], 0x44444444); }
        auto D = R.run(ps);
        for (int a = 0; a < 1024; ++a) { S[a] = nz(&D[a * 256]); for (int i = 0; i < 256; ++i) if (has(S[a], i)) CHECK(D[a * 256 + i] == 1.0f, "A one-hot D value"); }
    }
    // ---- stage 2: B one-hot (2048) x A ones: both complementary patterns (the union is the position set of the B element)
    std::vector<Bits> T(2048);
    {
        std::vector<SmfProbe> ps(2048 * 2);
        for (int b = 0; b < 2048; ++b) for (int q = 0; q < 2; ++q) { SmfProbe& p = ps[b * 2 + q]; a_ones(p); b_one_hot(p, b / 32, b % 32); idx_all(p, q ? 0xeeeeeeee : 0x44444444); }
        auto D = R.run(ps);
        for (int b = 0; b < 2048; ++b) { Bits x = nz(&D[(b * 2) * 256]), y = nz(&D[(b * 2 + 1) * 256]); for (int w = 0; w < 4; ++w) T[b][w] = x[w] | y[w]; CHECK(popc(x) + popc(y) == popc(T[b]), "B element %d selected under both complementary patterns", b); }
    }
    // ---- row / column classes -> D (lane, dword) -> (m, n)
    std::map<Bits, std::vector<int>> rows, cols;
    for (int a = 0; a < 1024; ++a) rows[S[a]].push_back(a);
    for (int b = 0; b < 2048; ++b) cols[T[b]].push_back(b);
    std::printf("A row classes: %zu (expect 16), B column classes: %zu (expect 16)\n", rows.size(), cols.size());
    CHECK(rows.size() == 16 && cols.size() == 16, "16 row and 16 column classes");
    std::vector<int> mrow(256, -1), ncol(256, -1);
    int arow[1024], bcol[2048];
    for (auto& kv : rows) { int lab = kv.second[0] / 16 % 16; for (int a : kv.second) { CHECK(a / 16 % 16 == lab, "A row class mixes lanes %% 16"); arow[a] = lab; } for (int i = 0; i < 256; ++i) if (has(kv.first, i)) mrow[i] = lab; CHECK(popc(kv.first) == 16 && kv.second.size() == 64, "row class size"); }
    for (auto& kv : cols) { int lab = kv.second[0] / 32 % 16; for (int b : kv.second) { CHECK(b / 32 % 16 == lab, "B column class mixes lanes %% 16"); bcol[b] = lab; } for (int i = 0; i < 256; ++i) if (has(kv.first, i)) ncol[i] = lab; CHECK(popc(kv.first) == 16 && kv.second.size() == 128, "column class size"); }
    {   // D layout hypothesis: lane l dword r = (m = 4 (l / 16) + r, n = l % 16)
        int viol = 0; for (int l = 0; l < 64; ++l) for (int r = 0; r < 4; ++r) if (mrow[l * 4 + r] != 4 * (l / 16) + r || ncol[l * 4 + r] != l % 16) ++viol;
        std::printf("D (lane l, dword r) = (m = 4 (l/16) + r, n = l%%16): %d of 256 positions violate\n", viol);
        CHECK(viol == 0, "D layout");
    }
    // ---- stage 3: for index pattern P and A element a: which B element (column n) has the same k as a?  B bit planes j = 0..10 + existence
    std::vector<std::pair<std::string, std::array<uint32_t, 8>>> pats;   // per-nibble patterns of a word (nibble 0 = low)
    auto word = [](const std::array<uint32_t, 8>& n) { uint32_t w = 0; for (int i = 0; i < 8; ++i) w |= n[i] << (4 * i); return w; };
    for (int q = 0; q < 6; ++q) { std::array<uint32_t, 8> n; n.fill(NIB[q]); char nm[32]; std::snprintf(nm, sizeof nm, "u%x", NIB[q]); pats.push_back({nm, n}); }
    for (int j = 0; j < 8; ++j) { std::array<uint32_t, 8> n; n.fill(0x4); n[j] = 0xe; char nm[32]; std::snprintf(nm, sizeof nm, "w%d", j); pats.push_back({nm, n}); }
    const int NP = (int)pats.size();
    // one stage-3 run: index word wl[l] per lane; returns the matched k label per A byte (-1 none, -2 inconsistent) and counts the problems
    int missing = 0, inconsistent = 0;
    auto stage3 = [&](const uint32_t* wl, std::vector<int>& kOut) {
        std::vector<SmfProbe> ps(1024 * 12);
        for (int a = 0; a < 1024; ++a) for (int j = 0; j < 12; ++j) {
            SmfProbe& p = ps[a * 12 + j]; a_one_hot(p, a / 16, a % 16); for (int l = 0; l < 64; ++l) p.idx[l] = wl[l];
            if (j == 11) b_ones(p); else for (int b = 0; b < 2048; ++b) p.b[b] = ((b >> j) & 1) ? ONE : 0;
        }
        auto D = R.run(ps);
        kOut.assign(1024, -9);
        for (int a = 0; a < 1024; ++a) {
            const int m = arow[a];
            int kk = -9;
            for (int n = 0; n < 16; ++n) {
                int pos = -1; for (int i = 0; i < 256; ++i) if (mrow[i] == m && ncol[i] == n) pos = i;
                int b = 0; for (int j = 0; j < 11; ++j) if (D[(a * 12 + j) * 256 + pos] > 0.5f) b |= 1 << j;
                if (!(D[(a * 12 + 11) * 256 + pos] > 0.5f)) { ++missing; continue; }   // (B ones: D = 1 whenever the A byte contributes)
                CHECK(bcol[b] == n, "matched B element %d of column %d lies in column %d", b, n, bcol[b]);
                const int lane = b / 32, r = (b % 32) / 4, y = b % 4;
                const int k = 32 * (lane / 16) + 4 * r + y;   // the k LABEL of a B element: lane group, register, byte; every column must give the same label
                if (kk == -9) kk = k; else if (kk != k) kk = -2;
            }
            if (kk == -2) ++inconsistent;
            kOut[a] = kk;
        }
    };
    std::vector<int> kA((size_t)NP * 1024, -9);
    for (int pi = 0; pi < NP; ++pi) { uint32_t wl[64]; for (int l = 0; l < 64; ++l) wl[l] = word(pats[pi].second); std::vector<int> k; stage3(wl, k); std::copy(k.begin(), k.end(), kA.begin() + (size_t)pi * 1024); }
    std::printf("stage 3: %d patterns x 1024 A bytes x 16 columns: %d unmatched, %d A elements whose 16 matched B elements disagree on k\n", NP, missing, inconsistent);
    CHECK(missing == 0 && inconsistent == 0, "every A element matches one k in every column");
    // measured: A lane l (lane group g = l/16), stored byte s (pair q = s/2), index word w of the SAME lane:  k = 64 (g&1) + 16 (g>>1) + 32 (q/4) + 4 (q%4) + ((w >> 2 s) & 3)
    auto kformula = [](int l, int s, uint32_t w) { const int g = l / 16, q = s / 2; return 64 * (g & 1) + 16 * (g >> 1) + 32 * (q / 4) + 4 * (q % 4) + (int)((w >> (2 * s)) & 3); };
    int viol = 0;
    for (int pi = 0; pi < NP; ++pi) for (int a = 0; a < 1024; ++a) if (kA[(size_t)pi * 1024 + a] != kformula(a / 16, a % 16, word(pats[pi].second))) ++viol;
    std::printf("k(l, s, w) = 64 (g&1) + 16 (g>>1) + 32 (q/4) + 4 (q%%4) + ((w >> 2 s) & 3), g = l/16, q = s/2: %d of %d violate\n", viol, NP * 1024);
    CHECK(viol == 0, "A compressed-index semantics");
    // per-lane index words (each lane its own): every A byte follows the index word of ITS OWN lane only
    int viol_lane = 0;
    for (int variant = 0; variant < 3; ++variant) {
        uint32_t wl[64]; std::mt19937 rg(777 + variant);
        for (int l = 0; l < 64; ++l) { uint32_t w = 0; for (int i = 0; i < 8; ++i) w |= NIB[rg() % 6] << (4 * i); wl[l] = w; }
        std::vector<int> k; stage3(wl, k);
        for (int a = 0; a < 1024; ++a) if (k[a] != kformula(a / 16, a % 16, wl[a / 16])) ++viol_lane;
    }
    std::printf("per-lane random index words (3 variants x 1024 bytes): %d violate\n", viol_lane);
    CHECK(viol_lane == 0, "each A byte follows its own lane's index word");
    viol += viol_lane;
    // B element k labels and the (n, k) map
    std::fprintf(out, "# v_smfmac_f32_16x16x128_fp8_fp8 one-hot probe maps (tests/gpu/isa_probe_host.cpp, kernel probe_smfmac_gen); m, n, k are measured class labels\n");
    std::fprintf(out, "SMFMAC_VIOLATIONS %d\n", viol + g_bad);
    std::fprintf(out, "DMAP");   for (int i = 0; i < 256; ++i) std::fprintf(out, " %d,%d", mrow[i], ncol[i]); std::fprintf(out, "\n");
    std::fprintf(out, "AROW");   for (int a = 0; a < 1024; ++a) std::fprintf(out, " %d", arow[a]); std::fprintf(out, "\n");
    std::fprintf(out, "BCOL");   for (int b = 0; b < 2048; ++b) std::fprintf(out, " %d", bcol[b]); std::fprintf(out, "\n");
    for (int pi = 0; pi < NP; ++pi) {
        std::fprintf(out, "PAT %s %08x", pats[pi].first.c_str(), word(pats[pi].second)); std::fprintf(out, "\nKA %s", pats[pi].first.c_str());
        for (int a = 0; a < 1024; ++a) std::fprintf(out, " %d", kA[(size_t)pi * 1024 + a]); std::fprintf(out, "\n");
    }
    // the B element that carries label k in column n (all patterns agree: B elements are matched by k)
    std::fprintf(out, "BK");     // per B element: the matched k label (from stage 3 inverse): k of element b
    for (int b = 0; b < 2048; ++b) std::fprintf(out, " %d", 32 * (b / 32 / 16) + 4 * ((b % 32) / 4) + (b % 4)); std::fprintf(out, "\n");
    // ---- vendor / own pattern check: 0x44444444 on a single row sum (all-ones A and B -> 64) was the sanity run above
}

// ================================================================ (b) ds_read_b64_tr_b8
static void tr_part(const std::string& dir, FILE* out) {
    Dev dev = load(dir + "/probe_lds_tr_b8_gen.co", "probe_lds_tr_b8_gen");
    std::printf("== (b) ds_read_b64_tr_b8\n");
    std::vector<std::vector<uint32_t>> addr_sets;   // per probe: 64 lane addresses
    auto row = [&](auto f) { std::vector<uint32_t> v(64); for (int l = 0; l < 64; ++l) v[l] = f(l); addr_sets.push_back(v); };
    row([](int l) { return 8u * l; });                       // consecutive qwords
    row([](int l) { return 16u * l; });
    row([](int l) { return 64u * (l % 8) + 8u * (l / 8); }); // column-major over an 8 x 8 qword block
    row([](int l) { return 3072u + 8u * l; });
    row([](int l) { return 8u * (l ^ 5); });
    std::mt19937 rng(12345);
    for (int t = 0; t < 40; ++t) { std::vector<int> blk(512); for (int i = 0; i < 512; ++i) blk[i] = i; std::shuffle(blk.begin(), blk.end(), rng); std::vector<uint32_t> v(64); for (int l = 0; l < 64; ++l) v[l] = 8u * blk[l]; addr_sets.push_back(v); }
    const int NPB = (int)addr_sets.size();
    void *dsrc, *dadr, *dout; CK(hipMalloc(&dsrc, 4096)); CK(hipMalloc(&dadr, NPB * 64 * 4)); CK(hipMalloc(&dout, NPB * 64 * 8));
    std::vector<uint32_t> A(NPB * 64); for (int p = 0; p < NPB; ++p) for (int l = 0; l < 64; ++l) A[p * 64 + l] = addr_sets[p][l];
    CK(hipMemcpy(dadr, A.data(), A.size() * 4, hipMemcpyHostToDevice));
    std::vector<uint8_t> id[2];
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<uint8_t> src(4096); for (int i = 0; i < 4096; ++i) src[i] = pass == 0 ? (i & 0xff) : ((i >> 8) & 0xff);
        CK(hipMemcpy(dsrc, src.data(), 4096, hipMemcpyHostToDevice)); CK(hipMemset(dout, 0, NPB * 512));
        launch(dev, NPB, dsrc, dadr, dout);
        id[pass].resize(NPB * 512); CK(hipMemcpy(id[pass].data(), dout, id[pass].size(), hipMemcpyDeviceToHost));
    }
    // per probe, lane d, byte j: the LDS byte address it received -> (source lane s, offset h) with addr[s] = address & ~7
    int smap[64][8], hmap[64][8]; bool first = true; int viol = 0;
    for (int p = 0; p < NPB; ++p) {
        std::map<uint32_t, int> lane_of; for (int l = 0; l < 64; ++l) lane_of[addr_sets[p][l]] = l;
        for (int d = 0; d < 64; ++d) for (int j = 0; j < 8; ++j) {
            const uint32_t a = id[0][p * 512 + d * 8 + j] + 256u * id[1][p * 512 + d * 8 + j];
            auto it = lane_of.find(a & ~7u);
            const int s = it == lane_of.end() ? -1 : it->second, h = a & 7;
            if (first) { smap[d][j] = s; hmap[d][j] = h; } else if (smap[d][j] != s || hmap[d][j] != h) ++viol;
        }
        first = false;
    }
    int unmatched = 0; for (int d = 0; d < 64; ++d) for (int j = 0; j < 8; ++j) if (smap[d][j] < 0) ++unmatched;
    std::printf("tr_b8: %d address sets (consecutive, strided, 8x8 column-major, 40 random permutations of 8-byte blocks): %d (lane, byte) results differ from the first set's (source lane, byte offset); %d unmatched\n", NPB, viol, unmatched);
    CHECK(viol == 0 && unmatched == 0, "tr_b8 map is address independent");
    // measured: dest lane d, byte j (register j/4, byte j%4) = byte (d % 8) of the qword read by lane 16 (d/16) + 2 j + (d%16)/8
    //   i.e. inside each 16-lane group the 8 qwords of the even lanes are 8x8-byte transposed into dest lanes 0..7, the odd lanes' into dest lanes 8..15
    int hv = 0; for (int d = 0; d < 64; ++d) for (int j = 0; j < 8; ++j) if (smap[d][j] != 16 * (d / 16) + 2 * j + (d % 16) / 8 || hmap[d][j] != d % 8) ++hv;
    std::printf("tr_b8 formula (source lane 16 (d/16) + 2 j + (d%%16)/8, byte d%%8): %d of 512 violate\n", hv);
    CHECK(hv == 0, "tr_b8 formula");
    std::fprintf(out, "# ds_read_b64_tr_b8 one-hot probe map (tests/gpu/isa_probe_host.cpp, kernel probe_lds_tr_b8_gen): dest lane d, byte j (reg j/4, byte j%%4) = byte h of the qword read by lane s\n");
    std::fprintf(out, "TR_VIOLATIONS %d\n", viol + unmatched);
    std::fprintf(out, "TR_S"); for (int d = 0; d < 64; ++d) for (int j = 0; j < 8; ++j) std::fprintf(out, " %d", smap[d][j]); std::fprintf(out, "\n");
    std::fprintf(out, "TR_H"); for (int d = 0; d < 64; ++d) for (int j = 0; j < 8; ++j) std::fprintf(out, " %d", hmap[d][j]); std::fprintf(out, "\n");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) { std::printf("usage: %s <co dir> <out dir>\n", argv[0]); return 2; }
    std::string dir = argv[1], od = argv[2];
    FILE* fa = std::fopen((od + "/smfmac_16x16x128_fp8.map").c_str(), "w"); FILE* fb = std::fopen((od + "/lds_tr_b8.map").c_str(), "w");
    if (!fa || !fb) { std::printf("cannot write %s\n", od.c_str()); return 1; }
    smfmac_part(dir, fa);
    tr_part(dir, fb);
    std::fclose(fa); std::fclose(fb);
    std::printf("%s (%d failed checks)\n", g_bad ? "ISA_PROBE_FAIL" : "ISA_PROBE_OK", g_bad);
    return g_bad ? 1 : 0;
}