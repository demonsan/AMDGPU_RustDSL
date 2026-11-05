#!/usr/bin/env python3
"""Markdown tables from tools/perf_matrix.sh raw output: python3 tools/perf_matrix_md.py build/pm/raw_fmha.txt build/pm/raw_gemm.txt"""
import re, statistics, sys
def kv(tokens):
    return dict(t.split("=", 1) for t in tokens if "=" in t)
def fmha(path):
    runs, acc, order = {}, {}, []
    for l in open(path):
        t = l.split()
        if not t: continue
        if t[0] == "FMHA":
            key = tuple(t[1:6]); d = kv(t[6:])
            if key not in runs: runs[key] = {"ours": [], "vendor": [], "path": d.get("path", "?")}; order.append(key)
            runs[key]["ours"].append(float(d["ours"])); runs[key]["vendor"].append(float(d["vendor"])); runs[key]["path"] = d.get("path", "?")
        elif t[0] == "ACC":
            acc[tuple(t[1:6])] = l.strip()
    out = ["| B | H | S | SK | GQA | 路径 | ours TF/s | vendor TF/s | 比值 | reps | ours bad / mean_err | vendor bad / mean_err |", "|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|---|---|"]
    for key in order:
        r = runs[key]; mo = statistics.median(r["ours"]); mv = statistics.median(r["vendor"])
        a = acc.get(key, ""); m = re.search(r"ours: bad=(\d+) worst=\S+ mean=(\S+) vendor: bad=(\d+) worst=\S+ mean=(\S+)", a)
        ao = f"{m.group(1)} / {float(m.group(2)):.2e}" if m else "NA"; av = f"{m.group(3)} / {float(m.group(4)):.2e}" if m else "NA"
        b, h, s, sk, g = key
        out.append(f"| {b} | {h} | {s} | {sk} | {g} | {r['path']} | {mo:.1f} | {mv:.1f} | {mo / mv:.3f} | {len(r['ours'])} | {ao} | {av} |")
    return "\n".join(out)
def gemm(path):
    runs, order = {}, []
    for l in open(path):
        t = l.split()
        if not t or t[0] != "GEMM": continue
        key = tuple(t[1:4]); d = kv(t[5:])
        if key not in runs: runs[key] = {"ours": [], "vendor": [], "ok": set(), "vok": set(), "bad": set(), "rel": []}; order.append(key)
        r = runs[key]; r["ours"].append(float(d["ours"])); r["vendor"].append(float(d["vendor"])); r["ok"].add(d.get("ours_check", "NA")); r["vok"].add(d.get("vendor_check", "NA")); r["bad"].add(d.get("bad", "?"))
        if "worst_rel" in d: r["rel"].append(float(d["worst_rel"]))
    out = ["| M | N | K | ours TF/s | vendor TF/s | 比值 | reps | ours 检查 | vendor 检查 | worst_rel(ours) |", "|--:|--:|--:|--:|--:|--:|--:|---|---|--:|"]
    for key in order:
        r = runs[key]; mo = statistics.median(r["ours"]); mv = statistics.median(r["vendor"])
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {mo:.1f} | {mv:.1f} | {mo / mv:.3f} | {len(r['ours'])} | {','.join(sorted(r['ok']))} | {','.join(sorted(r['vok']))} | {max(r['rel']) if r['rel'] else float('nan'):.2e} |")
    return "\n".join(out)
if __name__ == "__main__":
    print("## FMHA\n\n" + fmha(sys.argv[1]) + "\n\n## GEMM\n\n" + gemm(sys.argv[2]))