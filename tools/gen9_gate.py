#!/usr/bin/env python3
"""gen9 accuracy / vendor-parity gate over a `tests/gpu/run_ref.sh` log (stdin or file).

Per shape (blocks start with `=== B=...`): our mean_err <= --mean-max (default 2.6e-3), our bad count <= the vendor's bad count on the SAME shape,
no NaN/inf in any error field, and a `REF_KERNEL_OK` verdict unless the vendor itself fails that shape (then the shape is accepted when our bad <= vendor bad).
Exit status 0 = all shapes pass; prints one line per shape and a summary."""
import re, sys, argparse, math

ap = argparse.ArgumentParser()
ap.add_argument("log", nargs="?")
ap.add_argument("--mean-max", type=float, default=2.6e-3)
ap.add_argument("--shapes", type=int, default=15)
a = ap.parse_args()
text = open(a.log).read() if a.log else sys.stdin.read()
blocks = re.split(r"(?m)^=== ", text)[1:]
kv = lambda s, k: (re.search(r"\b%s=(\S+)" % k, s) or [None, None])[1]
fails, n = 0, 0
for blk in blocks:
    head = blk.splitlines()[0]
    v = re.search(r"(?m)^vendor kernel +vs CPU:.*$", blk)
    o = re.search(r"(?m)^our kernel +vs CPU:.*$", blk)
    verdict = "REF_KERNEL_OK" in blk
    n += 1
    if not o:
        print("FAIL %-34s no 'our kernel' line" % head); fails += 1; continue
    ob, om = kv(o.group(0), "bad"), kv(o.group(0), "mean_err")
    vb = kv(v.group(0), "bad") if v else None
    vm = kv(v.group(0), "mean_err") if v else None
    bad_nan = any(x is not None and (x.lower() in ("nan", "inf", "-nan") or not math.isfinite(float(x))) for x in (ob, om))
    ob_i, om_f = int(float(ob)), float(om)
    vb_i = int(float(vb)) if vb is not None else 0
    reasons = []
    if bad_nan: reasons.append("NaN/inf")
    if om_f > a.mean_max: reasons.append("mean_err %.3e > %.1e" % (om_f, a.mean_max))
    if ob_i > vb_i: reasons.append("bad %d > vendor bad %d" % (ob_i, vb_i))
    if not verdict and ob_i > 0: reasons.append("REF_KERNEL_FAIL with our bad=%d" % ob_i)
    ok = not reasons
    if not ok: fails += 1
    print("%s %-34s ours bad=%d mean=%.3e | vendor bad=%s mean=%s %s" % ("ok  " if ok else "FAIL", head, ob_i, om_f, vb, vm, ("<- " + "; ".join(reasons)) if reasons else ""))
print("GEN9_ACCURACY %s: %d shapes, %d failing (need %d shapes)" % ("PASS" if (fails == 0 and n >= a.shapes) else "FAIL", n, fails, a.shapes))
sys.exit(0 if (fails == 0 and n >= a.shapes) else 1)
