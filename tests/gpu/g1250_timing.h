// Timing of the gfx1250 hosts: median of SAMPLES samples, every sample at least MIN_SAMPLE_MS of back-to-back launches (so launch / clock jitter of a 10 - 30 us kernel averages out), optionally with a cache
// flush before every launch (FLUSH=1: the working set is then read from HBM, not from L2 / MALL; the flush is NOT inside the timed span).
//
//   env: WARMUP (20 launches), SAMPLES (7 per stream), MIN_SAMPLE_MS (20), STREAMS (1: the number of streams the samples are spread over), FLUSH (0|1), FLUSH_MB (512: must exceed the
//        last-level cache), FLUSH_LAUNCHES (240: the launches timed one by one in FLUSH mode)
//
// Why STREAMS: a small grid (one workgroup per sequence) runs on one CU of one die, and which CU / die a stream's dispatches reach is fixed per process (hardware queue assignment), not per launch: the same binary
// shows discrete modes (e.g. 4.7 / 8.1 / 10.7 us for the same 1-workgroup kernel) from one process to the next while each process is stable to 0.5 %. Samples spread over several streams (each its own
// hardware queue) average over the placements, so a process's median is a placement-independent number.
//
// Hang safety: every wait is the host's polled `wait(event, deadline_ms, what)` (exit 124 on expiry without touching the device again); no hipEventSynchronize / hipDeviceSynchronize anywhere.
#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct G1250Timing {
    double median_us = 0, min_us = 0, max_us = 0;
    int samples = 0, iters = 0, streams = 1;
    bool flush = false;
};

typedef void (*G1250Wait)(hipEvent_t, long, const char*);

static inline int g1250_envi(const char* n, int d) { const char* v = getenv(n); return v ? atoi(v) : d; }
#define G1250_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

// launch(stream): a callable that enqueues ONE kernel launch on `stream`. The caller's `HC` aborts on error.
template <class Launch>
static G1250Timing g1250_time(Launch launch, hipStream_t s, G1250Wait wait, long deadline_ms) {
    G1250Timing t;
    const int warm = g1250_envi("WARMUP", 20), nsamp = std::max(1, g1250_envi("SAMPLES", 7)), nstreams = std::max(1, g1250_envi("STREAMS", 1));
    const double min_ms = g1250_envi("MIN_SAMPLE_MS", 20);
    t.flush = g1250_envi("FLUSH", 0) != 0;
    t.streams = nstreams;
    std::vector<hipStream_t> ss(1, s);
    for (int i = 1; i < nstreams; i++) { hipStream_t x; G1250_OK(hipStreamCreate(&x)); ss.push_back(x); }
    hipEvent_t e0, e1;
    G1250_OK(hipEventCreate(&e0));
    G1250_OK(hipEventCreate(&e1));
    std::vector<double> us;
    std::vector<double> per_stream;
    void* fb = nullptr;
    const size_t mb = (size_t)std::max(1, g1250_envi("FLUSH_MB", 512));
    if (t.flush) G1250_OK(hipMalloc(&fb, mb << 20));
    for (hipStream_t st : ss) {
        for (int i = 0; i < warm; i++) launch(st);
        G1250_OK(hipEventRecord(e1, st));
        wait(e1, deadline_ms + 1000, "warm-up launches");
        std::vector<double> mine;
        if (!t.flush) {
            const int cal = 50;   // calibrate the launches per sample from one batch
            G1250_OK(hipEventRecord(e0, st));
            for (int i = 0; i < cal; i++) launch(st);
            G1250_OK(hipEventRecord(e1, st));
            wait(e1, deadline_ms + 1000, "calibration launches");
            float ms = 0;
            G1250_OK(hipEventElapsedTime(&ms, e0, e1));
            const double est_us = std::max(0.05, 1e3 * ms / cal);
            t.iters = (int)std::min(200000.0, std::max(20.0, min_ms * 1000.0 / est_us));
            for (int k = 0; k < nsamp; k++) {
                G1250_OK(hipEventRecord(e0, st));
                for (int i = 0; i < t.iters; i++) launch(st);
                G1250_OK(hipEventRecord(e1, st));
                wait(e1, deadline_ms + 2000 + (long)(min_ms * 4), "timed launches");
                G1250_OK(hipEventElapsedTime(&ms, e0, e1));
                mine.push_back(1e3 * ms / t.iters);
            }
        } else {
            const int n = std::max(nsamp * 8, g1250_envi("FLUSH_LAUNCHES", 240));
            std::vector<hipEvent_t> a(n), b(n);
            for (int i = 0; i < n; i++) { G1250_OK(hipEventCreate(&a[i])); G1250_OK(hipEventCreate(&b[i])); }
            for (int i = 0; i < n; i++) {
                G1250_OK(hipMemsetAsync(fb, i & 0xff, mb << 20, st));   // evicts the working set (not timed: the events bracket only the launch)
                G1250_OK(hipEventRecord(a[i], st));
                launch(st);
                G1250_OK(hipEventRecord(b[i], st));
            }
            wait(b[n - 1], deadline_ms + 2000 + (long)n * 5, "flushed launches");
            for (int i = 0; i < n; i++) { float ms = 0; G1250_OK(hipEventElapsedTime(&ms, a[i], b[i])); mine.push_back(1e3 * ms); }
            t.iters = 1;
        }
        std::vector<double> v = mine;
        std::sort(v.begin(), v.end());
        per_stream.push_back(v[v.size() / 2]);
        us.insert(us.end(), mine.begin(), mine.end());
    }
    std::vector<double> v = us;
    std::sort(v.begin(), v.end());
    t.samples = (int)v.size();
    // the median of the per-stream medians when several streams (placements) are sampled: robust to one stream sitting on a far die
    std::vector<double> ps = per_stream;
    std::sort(ps.begin(), ps.end());
    t.median_us = nstreams > 1 ? ps[ps.size() / 2] : v[v.size() / 2];
    t.min_us = v.front();
    t.max_us = v.back();
    std::printf("timing: median %.3f us (min %.3f, max %.3f, spread %.1f %%) over %d %s, %d launch(es) per sample, %d stream(s)%s\n", t.median_us, t.min_us, t.max_us,
                100.0 * (t.max_us - t.min_us) / t.median_us, t.samples, t.flush ? "flushed launches" : "samples", t.iters, nstreams, t.flush ? ", cache flushed before each" : "");
    if (nstreams > 1) {
        std::printf("timing: per-stream medians:");
        for (double x : per_stream) std::printf(" %.3f", x);
        std::printf("\n");
    }
    return t;
}

// PAIRED timing (CO2 / KERNEL2 of the hosts): two kernels of the SAME ABI on the SAME buffers in ONE process, alternating blocks (A B, B A, ...): per-process placement effects (the physical location of the
// buffers, the die a queue lands on) hit both kernels equally, so the ratio is stable where the absolute times are not. Returns the per-kernel medians and the median of the per-pair ratios B / A.
struct G1250Paired {
    double a_us = 0, b_us = 0, ratio_ba = 0, spread_ratio = 0;
    int pairs = 0, iters = 0;
};

template <class LaunchA, class LaunchB>
static G1250Paired g1250_time_pair(LaunchA la, LaunchB lb, hipStream_t st, G1250Wait wait, long deadline_ms) {
    G1250Paired r;
    const int warm = g1250_envi("WARMUP", 20), npairs = std::max(3, g1250_envi("PAIRS", 15));
    const double min_ms = g1250_envi("MIN_SAMPLE_MS", 20);
    hipEvent_t e0, e1;
    G1250_OK(hipEventCreate(&e0));
    G1250_OK(hipEventCreate(&e1));
    for (int i = 0; i < warm; i++) { la(st); lb(st); }
    G1250_OK(hipEventRecord(e1, st));
    wait(e1, deadline_ms + 1000, "paired warm-up");
    auto block = [&](auto& l, int iters, const char* what) {
        G1250_OK(hipEventRecord(e0, st));
        for (int i = 0; i < iters; i++) l(st);
        G1250_OK(hipEventRecord(e1, st));
        wait(e1, deadline_ms + 2000 + (long)(min_ms * 4), what);
        float ms = 0;
        G1250_OK(hipEventElapsedTime(&ms, e0, e1));
        return 1e3 * ms / iters;
    };
    const int cal = 50;
    const double est = std::max(0.05, std::max(block(la, cal, "paired calibration A"), block(lb, cal, "paired calibration B")));
    r.iters = (int)std::min(200000.0, std::max(20.0, min_ms * 1000.0 / est));
    std::vector<double> A, B, R;
    for (int k = 0; k < npairs; k++) {
        double ta, tb;
        if (k % 2 == 0) { ta = block(la, r.iters, "paired block A"); tb = block(lb, r.iters, "paired block B"); }
        else { tb = block(lb, r.iters, "paired block B"); ta = block(la, r.iters, "paired block A"); }
        A.push_back(ta); B.push_back(tb); R.push_back(tb / ta);
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    r.a_us = med(A); r.b_us = med(B); r.ratio_ba = med(R); r.pairs = npairs;
    std::vector<double> rs = R; std::sort(rs.begin(), rs.end());
    r.spread_ratio = (rs.back() - rs.front()) / r.ratio_ba;
    std::printf("timing(paired): A median %.3f us, B median %.3f us, B/A %.4f (pair spread %.2f %%) over %d pairs, %d launches per block\n", r.a_us, r.b_us, r.ratio_ba, 100 * r.spread_ratio, npairs, r.iters);
    return r;
}
