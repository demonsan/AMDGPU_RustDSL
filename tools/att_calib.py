#!/usr/bin/env python3
"""Per-instruction-class issue / latency calibration from a rocprofv3 --att wave timeline (one dispatch dir with code.json and se*_wv*.json).
Each event is [time(issue), type, stall, duration, code_line]. Reports, for the wave(s) given:
  * per class: issue stall (cycles the instruction waited before issue) median / p90, issue-to-next-issue median / p90;
  * MFMA issue interval, MFMA -> first dependent reader (VALU / accvgpr / MFMA srcC / ds / branch) latency, split by reader kind;
  * ds_read issue -> first consumer latency by read kind and by the number of ds_reads outstanding at issue;
  * v_cmp(vcc) -> s_cbranch_vcc* issue-to-issue; s_nop states;
  * barrier arrival skew across all waves found (same occurrence index) and the 5 biggest stall sites (PC) with their text.
usage: att_calib.py <ui_output_dir> [steady_from_mfma=40]"""
import json, sys, glob, re, collections, statistics

d = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 40
code = json.load(open(d + "/code.json"))["code"]
waves = sorted(glob.glob(d + "/se*_wv*.json"))


def text(e):
    return code[e[4]][0]


def klass(t):
    m = t.split()[0]
    if m.startswith("v_mfma") or m.startswith("v_smfmac"):
        return "MFMA" if "scale_f32_32x32x64" in m else "MFMA-other"
    if m.startswith("s_waitcnt"):
        return "waitcnt"
    if m == "s_barrier":
        return "s_barrier"
    if m == "s_nop":
        return "s_nop"
    if m.startswith("buffer_load") or m.startswith("global_load"):
        return "buffer_load(lds)" if " lds" in t else "vmem_load"
    if m.startswith("buffer_store") or m.startswith("global_store"):
        return "vmem_store"
    if m.startswith("ds_read_b64_tr"):
        return "ds_read_tr_b8"
    if m.startswith("ds_read_b128"):
        return "ds_read_b128"
    if m.startswith("ds_read"):
        return "ds_read_b32/64"
    if m.startswith("ds_write"):
        return "ds_write"
    if m.startswith("v_accvgpr_read"):
        return "accvgpr_read"
    if m.startswith("v_accvgpr_write"):
        return "accvgpr_write"
    if m.startswith("v_"):
        return "VALU"
    if m.startswith("s_cbranch") or m == "s_branch":
        return "branch"
    if m.startswith("s_"):
        return "SALU"
    return m


RE = re.compile(r"\b([va])(?:\[(\d+):(\d+)\]|(\d+))")


def regs(op):
    out = set()
    for m in RE.finditer(op):
        f = m.group(1)
        if m.group(4) is not None:
            out.add((f, int(m.group(4))))
        else:
            for r in range(int(m.group(2)), int(m.group(3)) + 1):
                out.add((f, r))
    return out


def split_ops(t):
    p = t.split(None, 1)
    if len(p) < 2:
        return [], []
    ops = [o.strip() for o in re.split(r",(?![^\[]*\])", p[1])]
    return ops


def dst_src(t):
    m = t.split()[0]
    ops = split_ops(t)
    if not ops:
        return set(), set()
    if m.startswith("s_") or m.startswith("buffer_store") or m.startswith("global_store") or m.startswith("ds_write"):
        return set(), set().union(*[regs(o) for o in ops])
    if m.startswith("buffer_load") and " lds" in t:
        return set(), set().union(*[regs(o) for o in ops])
    return regs(ops[0]), set().union(*[regs(o) for o in ops[1:]])


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def fmt(xs):
    return f"n={len(xs):5d} med={pct(xs, 0.5):6.0f} p90={pct(xs, 0.9):6.0f}" if xs else "n=0"


barrier_times = collections.defaultdict(list)  # occurrence -> [time per wave]
for wi, wf in enumerate(waves):
    w = json.load(open(wf))["wave"]
    ev = w["instructions"]
    mf = [i for i, e in enumerate(ev) if text(e).split()[0].startswith("v_mfma")]
    if len(mf) <= skip + 10:
        continue
    lo, hi = mf[skip], mf[-10]
    steady = ev[lo:hi]
    print(f"===== {wf.split('/')[-1]}: {len(ev)} events, {len(mf)} MFMAs; steady window = MFMA #{skip}..#{len(mf)-10} ({ev[hi][0]-ev[lo][0]} cycles)")
    # per class stall + issue-to-next
    st = collections.defaultdict(list)
    nx = collections.defaultdict(list)
    for k in range(lo, hi):
        c = klass(text(ev[k]))
        st[c].append(ev[k][2])
        nx[c].append(ev[k + 1][0] - ev[k][0])
    print(f"{'class':18s} {'issue stall':34s} {'issue -> next issue':34s}")
    for c in sorted(st, key=lambda c: -sum(st[c])):
        print(f"{c:18s} {fmt(st[c]):34s} {fmt(nx[c]):34s} sum_stall={sum(st[c])}")
    # MFMA interval
    mfi = [ev[b][0] - ev[a][0] for a, b in zip(mf, mf[1:]) if lo <= a < hi]
    print("MFMA(all) issue interval:", fmt(mfi))
    # dependency latencies (issue -> first reader issue)
    dep = collections.defaultdict(list)
    lds = collections.defaultdict(list)
    outstanding = []  # (issue idx, regs, consumed?)
    for k in range(lo, hi):
        t = text(ev[k])
        m = t.split()[0]
        dk, _ = dst_src(t)
        if not dk:
            continue
        is_mfma = m.startswith("v_mfma") or m.startswith("v_smfmac")
        is_ds = m.startswith("ds_read")
        if not (is_mfma or is_ds or m.startswith("v_cmp")):
            continue
        if m.startswith("v_cmp") and "vcc" not in t:
            continue
        depth = 0
        if is_ds:
            # ds_reads issued before k whose first consumer is after k
            depth = sum(1 for (j, tc) in outstanding if j < k and (tc is None or tc > k))
        found = None
        for j in range(k + 1, min(len(ev), k + 400)):
            tj = text(ev[j])
            mj = tj.split()[0]
            if m.startswith("v_cmp"):
                if mj.startswith("s_cbranch_vcc"):
                    found = j
                    break
                if mj.startswith("v_cmp"):
                    break
                continue
            dj, sj = dst_src(tj)
            if mj.startswith("v_mfma") and dj == dk:
                # in-place accumulation of the same tile: srcC read
                found = j
                break
            if dk & sj:
                found = j
                break
            if dk & dj:
                break
        if is_ds:
            outstanding.append((k, found))
        if found is None:
            continue
        lat = ev[found][0] - ev[k][0]
        rk = klass(text(ev[found]))
        if is_mfma:
            kind = "MFMA32x64" if "32x32x64" in m else m.replace("v_", "")[:18]
            dep[(kind, rk)].append(lat)
        elif is_ds:
            lds[(klass(t), min(depth, 12) // 3 * 3)].append(lat)
        else:
            dep[("v_cmp vcc", "branch")].append(lat)
    print("issue -> first dependent reader issue (cycles):")
    for key in sorted(dep, key=lambda k: -len(dep[k])):
        print(f"  {key[0]:18s} -> {key[1]:16s} {fmt(dep[key])}")
    print("ds_read issue -> first consumer issue, by ds_reads outstanding at issue (bucket of 3):")
    for key in sorted(lds):
        print(f"  {key[0]:16s} depth {key[1]:2d}-{key[1]+2:2d}: {fmt(lds[key])}")
    nops = [int(text(ev[k]).split()[1]) + 1 for k in range(lo, hi) if text(ev[k]).startswith("s_nop")]
    print("s_nop states:", fmt(nops), "total states", sum(nops))
    # top stall sites
    site = collections.defaultdict(lambda: [0, 0])
    for k in range(lo, hi):
        s = site[ev[k][4]]
        s[0] += ev[k][2]
        s[1] += 1
    print("top stall sites (steady window):")
    for l, (x, n) in sorted(site.items(), key=lambda kv: -kv[1][0])[:8]:
        prev = text(ev[[k for k in range(lo, hi) if ev[k][4] == l][0] - 1])[:50]
        print(f"  pc_line={l:5d} stall={x:7d} hits={n:4d}  {code[l][0][:70]}   [prev: {prev}]")
    # rollback branch taken count: the v_pk_mul_f32 executions (O rescale lives only in the rollback branch)
    rb = sum(1 for e in ev if text(e).startswith("v_pk_mul_f32"))
    print(f"v_pk_mul_f32 executions (rollback O rescale, 32 per taken branch per group): {rb} -> ~{rb/32:.1f} taken rollbacks")
    for i, e in enumerate([e for e in ev if text(e).split()[0] == "s_barrier"]):
        barrier_times[i].append((wi, e[0], e[0] + e[2] + e[3]))

if len(waves) > 1:
    print("===== barrier arrival skew across waves (issue time - earliest, per occurrence):")
    sk = []
    for i, ts in sorted(barrier_times.items()):
        if len(ts) < 2:
            continue
        t0 = min(t for _, t, _ in ts)
        sk.append(sorted(t - t0 for _, t, _ in ts))
    if sk:
        print("  max-min:", fmt([s[-1] for s in sk]), " median wave:", fmt([s[len(s) // 2] for s in sk]))
        for i, s in enumerate(sk[:20]):
            print(f"  barrier {i:2d}: {s}")
