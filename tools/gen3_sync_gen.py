#!/usr/bin/env python3
"""Generate the gen3 sync-probe kernels from tools/gen3_sync_template.s."""
import os

SYNC_BLOCK = "    s_barrier"
SYNC_NONE = "    ; no sync (data race; baseline for the error counter)"
SYNC_SOFT = """    ; software group barrier: lane0 of each wave bumps the group counter, all lanes spin until the target
    s_mov_b64 s[10:11], exec
    s_mov_b64 exec, 1
    v_mov_b32_e32 v10, 1
    ds_add_u32 v10, v3
    s_waitcnt lgkmcnt(0)
    s_mov_b64 exec, s[10:11]
    v_mov_b32_e32 v8, s12
    v_mov_b32_e32 v11, 0
Lspin___NAME__:
    ds_read_b32 v10, v3
    s_waitcnt lgkmcnt(0)
    v_cmp_ge_u32_e32 vcc, v10, v8
    s_cbranch_vccnz Lspindone___NAME__
    v_add_u32_e32 v11, 1, v11
    v_mov_b32_e32 v24, __POLLCAP__
    v_cmp_lt_u32_e32 vcc, v11, v24
    s_cbranch_vccnz Lspin___NAME__
    v_mov_b32_e32 v10, 1
    ds_write_b32 v19, v10
    s_waitcnt lgkmcnt(0)
    s_branch Ldone___NAME__
Lspindone___NAME__:"""
DELAY_B = """    ; group B only: extra scalar delay so the two groups are unbalanced
    s_cmp_eq_u32 s16, 1
    s_cbranch_scc0 Lskipd___NAME__
    s_mov_b32 s17, __DELAY_N__
Ldelay___NAME__:
    s_add_u32 s17, s17, -1
    s_cmp_lg_u32 s17, 0
    s_cbranch_scc1 Ldelay___NAME__
Lskipd___NAME__:"""

SPECS = [
    ("block_bal", SYNC_BLOCK, "", 200, 200, 4, 100000, 512),
    ("soft_bal", SYNC_SOFT, "", 200, 200, 4, 100000, 512),
    ("block_asym", SYNC_BLOCK, DELAY_B, 200, 200, 4, 100000, 512),
    ("soft_asym", SYNC_SOFT, DELAY_B, 200, 200, 4, 100000, 512),
    ("none_bal", SYNC_NONE, "", 200, 200, 4, 100000, 512),
    ("soft_badcnt", SYNC_SOFT, "", 20, 20, 5, 2000, 512),
]

def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl = open(os.path.join(repo, "tools", "gen3_sync_template.s"), encoding="utf-8").read()
    out = os.path.join(repo, "build", "gen3")
    os.makedirs(out, exist_ok=True)
    for name, sync, delay, ita, itb, target, poll, delayn in SPECS:
        t = (tpl.replace("__NAME__", name).replace("__SYNC__", sync).replace("__DELAY_B__", delay)
             .replace("__ITERS_A__", str(ita)).replace("__ITERS_B__", str(itb))
             .replace("__TARGET_MUL__", str(target)).replace("__POLLCAP__", str(poll))
             .replace("__DELAY_N__", str(delayn)))
        open(os.path.join(out, name + ".s"), "w", encoding="utf-8", newline="\n").write(t)
    print("[gen3] generated", len(SPECS), "kernel sources in build/gen3")

if __name__ == "__main__":
    main()
