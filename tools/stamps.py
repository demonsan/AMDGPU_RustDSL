#!/usr/bin/env python3
"""stamps.py - analyse a StampBuffer dump (tests/gpu/stamp_host.h: header [workgroups, waves, slots, 0x53544d50] then u32 [workgroup][wave][slot]).

    python3 tools/stamps.py dump.u32 [--phases 0:prologue,1:loop_start,...] [--tiles FIRST:LAST] [--bubble 1.15]

Stamps are low 32 bits of s_memtime (core-clock ticks): intervals are wrapping u32 differences. A slot that is 0 was not written (the stamp is skipped).
  * per slot interval statistics (median / min / max over waves) with optional phase names (--phases slot:name, the name labels the interval ENDING at that slot)
  * --tiles A:B: slots A..B are per-tile boundary stamps; prints the steady-state tile time (median of the middle 80 %), the ramp (first / last tile vs steady) and the
    tiles whose median-over-waves interval exceeds --bubble x steady (bubble tiles), plus the wave-to-wave spread per tile
"""
import argparse
import statistics
import struct
import sys

MAGIC = 0x53544D50


def load(path):
    data = open(path, "rb").read()
    wgs, waves, slots, magic = struct.unpack_from("<4I", data, 0)
    if magic != MAGIC:
        raise SystemExit("not a StampBuffer dump (bad magic)")
    vals = struct.unpack_from(f"<{wgs * waves * slots}I", data, 16)
    return wgs, waves, slots, vals


def interval(a, b):
    return (b - a) & 0xFFFFFFFF


def per_slot(wgs, waves, slots, vals):
    out = {}
    for s in range(1, slots):
        d = []
        for r in range(wgs * waves):
            a, b = vals[r * slots + s - 1], vals[r * slots + s]
            if a and b:
                d.append(interval(a, b))
        if d:
            out[s] = d
    return out


def tile_report(wgs, waves, slots, vals, first, last, bubble):
    rows = {}
    for s in range(first + 1, last + 1):
        d = []
        for r in range(wgs * waves):
            a, b = vals[r * slots + s - 1], vals[r * slots + s]
            if a and b:
                d.append(interval(a, b))
        if d:
            rows[s] = d
    if len(rows) < 3:
        return ["fewer than 3 tile intervals"], {}
    med = {s: statistics.median(d) for s, d in rows.items()}
    keys = sorted(med)
    mid = keys[len(keys) // 10: len(keys) - len(keys) // 10] or keys
    steady = statistics.median(med[s] for s in mid)
    L = [f"tile intervals: slots {first}..{last} ({len(keys)} tiles), steady-state median {steady:.0f} cycles (middle 80 %)",
         f"first tile {med[keys[0]]:.0f} ({med[keys[0]] / steady:.2f}x), last tile {med[keys[-1]]:.0f} ({med[keys[-1]] / steady:.2f}x)"]
    bad = [s for s in keys if med[s] > bubble * steady]
    L.append(f"tiles slower than {bubble:.2f} x steady: " + (", ".join(f"{s} ({med[s]:.0f})" for s in bad[:20]) + (" ..." if len(bad) > 20 else "") if bad else "none"))
    spread = [(max(d) - min(d)) for d in rows.values()]
    L.append(f"wave-to-wave spread within a tile interval: median {statistics.median(spread):.0f}, max {max(spread):.0f} cycles")
    return L, {"steady": steady, "med": med, "bad": bad}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--phases", default="")
    ap.add_argument("--tiles")
    ap.add_argument("--bubble", type=float, default=1.15)
    a = ap.parse_args(argv)
    wgs, waves, slots, vals = load(a.dump)
    names = dict((int(k), v) for k, v in (x.split(":", 1) for x in a.phases.split(",") if x))
    print(f"{wgs} workgroups x {waves} waves x {slots} slots")
    stats = per_slot(wgs, waves, slots, vals)
    tot = 0
    for s, d in stats.items():
        if a.tiles and int(a.tiles.split(":")[0]) < s <= int(a.tiles.split(":")[1]):
            continue
        m = statistics.median(d)
        tot += m
        print(f"  slot {s - 1:2d} -> {s:2d} {names.get(s, ''):18s} median {m:10.0f}  min {min(d):10d}  max {max(d):10d}  ({len(d)} waves)")
    if a.tiles:
        f, l = (int(x) for x in a.tiles.split(":"))
        lines, _ = tile_report(wgs, waves, slots, vals, f, l, a.bubble)
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())