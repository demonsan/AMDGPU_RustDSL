#!/usr/bin/env python3
"""Estimate the VOPD pairing potential of an emitted gfx1250 .s (before writing the pass): VALU lines of the pairable e32 forms, adjacent
pairs that meet the gfx1250 VOPD rules (public LLVM GCNVOPDUtils / AMDGPUBaseInfo: dst parity differs, src0 / src1 banks (reg & 3) differ
unless the same register, no RAW / WAW between the two, same VGPR window), and the same with a look-ahead of K VALU lines (the second moved
up across register-independent VALU lines). The s_set_vgpr_msb state is tracked to get absolute register numbers.
usage: vopd_estimate.py file.s [K]"""
import re, sys

X = {"v_mul_f32", "v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mov_b32", "v_max_num_f32", "v_min_num_f32", "v_fmac_f32", "v_mul_dx9_zero_f32"}
Y = X | {"v_add_nc_u32", "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32", "v_sub_nc_u32", "v_max_i32", "v_min_i32"}
HAZ = re.compile(r"^(v_wmma|v_exp|v_log|v_rcp|v_rsq|v_sqrt|v_sin|v_cos|v_nop|v_permlane|v_readlane|v_writelane|v_cmpx|.*_dpp)")

def regs(tok, win):
    m = re.fullmatch(r"v(\d+)", tok)
    if m:
        return [int(m.group(1)) + 256 * win]
    m = re.fullmatch(r"v\[(\d+):(\d+)\]", tok)
    if m:
        return [x + 256 * win for x in range(int(m.group(1)), int(m.group(2)) + 1)]
    return []

def parse(path):
    out = []
    msb = [0, 0, 0, 0]
    for raw in open(path):
        s = raw.strip()
        if s.startswith("s_set_vgpr_msb"):
            v = int(s.split()[1], 0)
            msb = [(v >> 6) & 3, v & 3, (v >> 2) & 3, (v >> 4) & 3]
            continue
        if not s or s.endswith(":") or s.startswith("."):
            out.append(None)
            continue
        mn = s.split()[0]
        ops = [o.strip() for o in s[len(mn):].split(",")] if " " in s else []
        base = mn[:-4] if mn.endswith("_e32") else mn
        is_v = mn.startswith("v_")
        d = regs(ops[0], msb[0]) if ops else []
        srcs = [regs(o, msb[k + 1]) if k + 1 < 4 else [] for k, o in enumerate(ops[1:])]
        lit = any(re.fullmatch(r"(0x[0-9a-f]+|-?\d+)", o) and not (-16 <= int(o, 0) <= 64) for o in ops[1:] if re.fullmatch(r"(0x[0-9a-f]+|-?\d+)", o))
        pairable = is_v and mn.endswith("_e32") and base in Y and not lit and (base == "v_mov_b32" or (len(ops) == 3 and srcs[1]))
        out.append(dict(mn=base, d=d, s=srcs, w=[x // 256 for x in d] + [x // 256 for s_ in srcs for x in s_], pair=pairable, valu=is_v, haz=bool(HAZ.match(mn)) or s.startswith("s_delay"), line=s))
    return out

def ok(a, b):
    if a["mn"] not in X or b["mn"] not in Y:
        return False
    if len(a["d"]) != 1 or len(b["d"]) != 1 or (a["d"][0] & 1) == (b["d"][0] & 1):
        return False
    # windows per slot
    if a["d"][0] // 256 != b["d"][0] // 256:
        return False
    for k in range(2):
        sa = a["s"][k] if k < len(a["s"]) else []
        sb = b["s"][k] if k < len(b["s"]) else []
        if sa and sb:
            if sa[0] // 256 != sb[0] // 256:
                return False
            if sa[0] != sb[0] and (sa[0] & 3) == (sb[0] & 3):
                return False
    # no RAW / WAW between
    reads_b = [x for s_ in b["s"] for x in s_]
    if a["d"][0] in reads_b or a["d"][0] == b["d"][0]:
        return False
    if a["mn"] == "v_fmac_f32" and b["d"][0] in [x for s_ in a["s"] for x in s_]:
        return False
    return True

def main():
    L = parse(sys.argv[1])
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    valu = sum(1 for x in L if x and x["valu"])
    cand = sum(1 for x in L if x and x["pair"])
    for look in (1, K):
        used = set()
        n = 0
        for i, a in enumerate(L):
            if not a or i in used or not a["pair"]:
                continue
            for j in range(i + 1, min(len(L), i + 1 + look)):
                b = L[j]
                if not b or not b["valu"] or b["haz"]:
                    break
                if j not in used and b["pair"] and ok(a, b):
                    # b moves up across i+1..j-1: independent of them
                    mid = L[i + 1:j]
                    def touch(x):
                        return set(x["d"]) | {r for s_ in x["s"] for r in s_}
                    if all(m and m["valu"] and not m["haz"] and not (set(b["d"]) & touch(m)) and not (set(m["d"]) & touch(b)) for m in mid):
                        used |= {i, j}
                        n += 1
                        break
        print(f"look-ahead {look}: {n} pairs -> {n} VALU instructions saved of {valu} ({100.0 * n / max(valu, 1):.1f} %); pairable-form lines {cand}")

main()
