#!/usr/bin/env python3
"""Generate the count-matched s_barrier probe kernels (rotated-phase hang check + skew timing)."""
import os

def nops(x):
    out = ["    s_nop 7"] * (x // 8)
    if x % 8:
        out.append("    s_nop %d" % (x % 8 - 1))
    return out

WORK_V = ["    v_add_u32_e32 v27, 1, v27", "    v_add_u32_e32 v27, 1, v27"]
WORK_M = ["    v_add_u32_e32 v28, 1, v28"] * 8
CHECK = [
    "    ds_read_b32 v6, v2",
    "    s_waitcnt lgkmcnt(0)",
    "    v_mov_b32_e32 v11, s14",
    "    v_cmp_ne_u32_e32 vcc, v6, v11",
    "    v_cndmask_b32_e32 v23, v24, v25, vcc",
    "    v_add_u32_e32 v14, v14, v23",
]
BAR = ["    s_barrier"]

def body_a():
    return BAR + CHECK + WORK_V + WORK_M + WORK_V + WORK_M + [
        "    v_mov_b32_e32 v10, s14",
        "    v_add_u32_e32 v10, 1, v10",
        "    ds_write_b32 v1, v10",
        "    s_waitcnt lgkmcnt(0)",
    ]

def body_b(delay=0):
    return [
        "    v_mov_b32_e32 v10, s14",
        "    ds_write_b32 v1, v10",
        "    s_waitcnt lgkmcnt(0)",
    ] + WORK_V + nops(delay) + BAR + CHECK + WORK_M + WORK_V + WORK_M

def body_a_nobar():
    return CHECK + WORK_V + WORK_M + WORK_V + WORK_M + [
        "    v_mov_b32_e32 v10, s14",
        "    v_add_u32_e32 v10, 1, v10",
        "    ds_write_b32 v1, v10",
        "    s_waitcnt lgkmcnt(0)",
    ]

def body_b_nobar():
    return [
        "    v_mov_b32_e32 v10, s14",
        "    ds_write_b32 v1, v10",
        "    s_waitcnt lgkmcnt(0)",
    ] + WORK_V + CHECK + WORK_M + WORK_V + WORK_M

SPECS = [
    ("aligned", body_a(), body_a(), 2000, 2000),
    ("rotated", body_a(), body_b(), 2000, 2000),
    ("none", body_a_nobar(), body_b_nobar(), 2000, 2000),
    ("skew100", body_a(), body_b(100), 2000, 2000),
    ("skew500", body_a(), body_b(500), 2000, 2000),
    ("skew1000", body_a(), body_b(1000), 2000, 2000),
    ("skew2000", body_a(), body_b(2000), 2000, 2000),
    ("asym", body_a(), body_b(), 1000, 2000),
    ("rot_long", body_a(), body_b(), 100000, 100000),
]

def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl = open(os.path.join(repo, "tools", "gen3_barrier2_template.s"), encoding="utf-8").read()
    out = os.path.join(repo, "build", "gen3c")
    os.makedirs(out, exist_ok=True)
    for name, ba, bb, na, nb in SPECS:
        t = (tpl.replace("__NAME__", name).replace("__BODY_A__", "\n".join(ba))
             .replace("__BODY_B__", "\n".join(bb)).replace("__N_A__", str(na)).replace("__N_B__", str(nb)))
        open(os.path.join(out, name + ".s"), "w", encoding="utf-8", newline="\n").write(t)
    print("[gen3c] generated", len(SPECS), "kernels in build/gen3c")

if __name__ == "__main__":
    main()
