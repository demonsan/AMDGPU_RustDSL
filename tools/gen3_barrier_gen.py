#!/usr/bin/env python3
"""Generate the count-matched s_barrier probe kernels (rotated phase between two wave groups)."""
import os

def nops(x):
    out = ["    s_nop 7"] * (x // 8)
    if x % 8:
        out.append("    s_nop %d" % (x % 8 - 1))
    return out

WORK_V = ["    v_add_u32_e32 v27, 1, v27", "    v_add_u32_e32 v27, 1, v27"]
WORK_M = ["    v_add_u32_e32 v28, 1, v28"] * 8
READ = [
    "    ds_read_b32 v6, v2",
    "    s_waitcnt lgkmcnt(0)",
    "    v_mov_b32_e32 v11, s14",
    "    v_cmp_ne_u32_e32 vcc, v6, v11",
    "    v_cndmask_b32_e32 v23, v24, v25, vcc",
    "    ds_add_u32 v23, v18",
    "    s_waitcnt lgkmcnt(0)",
]
WRITE = [
    "    v_mov_b32_e32 v10, s14",
    "    v_add_u32_e32 v10, 1, v10",
    "    ds_write_b32 v1, v10",
    "    s_waitcnt lgkmcnt(0)",
]
BAR = ["    s_barrier"]

def body_a(delay=0):
    return BAR + READ + WORK_V + WORK_M + WORK_V + WORK_M + WRITE

def body_b(delay=0, barrier=True):
    b = (WORK_V + nops(delay) + BAR) if barrier else (WORK_V + nops(delay))
    return b + READ + WORK_M + WORK_V + WORK_M + WRITE

def body_a_nobar():
    return READ + WORK_V + WORK_M + WORK_V + WORK_M + WRITE

def body_b_nobar():
    return WORK_V + READ + WORK_M + WORK_V + WORK_M + WRITE

SPECS = [
    # name          body_a                       body_b                       limit_a  limit_b
    ("aligned",  body_a(),                    body_a(),                    "s6", "s6"),
    ("rotated",  body_a(),                    body_b(),                    "s6", "s6"),
    ("none",     body_a_nobar(),              body_b_nobar(),              "s6", "s6"),
    ("skew100",  body_a(),                    body_b(delay=100),           "s6", "s6"),
    ("skew500",  body_a(),                    body_b(delay=500),           "s6", "s6"),
    ("skew1000", body_a(),                    body_b(delay=1000),          "s6", "s6"),
    ("skew2000", body_a(),                    body_b(delay=2000),          "s6", "s6"),
    ("asym",     body_a(),                    body_b(),                    "s6", "s7"),
]

def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl = open(os.path.join(repo, "tools", "gen3_barrier_template.s"), encoding="utf-8").read()
    out = os.path.join(repo, "build", "gen3b")
    os.makedirs(out, exist_ok=True)
    for name, ba, bb, la, lb in SPECS:
        t = (tpl.replace("__NAME__", name).replace("__BODY_A__", "\n".join(ba))
                .replace("__BODY_B__", "\n".join(bb)).replace("__LIMIT_A__", la).replace("__LIMIT_B__", lb))
        open(os.path.join(out, name + ".s"), "w", encoding="utf-8", newline="\n").write(t)
    print("[gen3b] generated", len(SPECS), "kernels in build/gen3b")

if __name__ == "__main__":
    main()
