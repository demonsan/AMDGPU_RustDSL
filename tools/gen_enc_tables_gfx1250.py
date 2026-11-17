#!/usr/bin/env python3
"""gfx1250 encoder tables from the llvm-mc oracle (docs/gfx1250-plan.md G1): one sample line per mnemonic the gfx1250 emitter can print is
assembled with the pinned gfx1250 assembler (backend.gfx1250.lock); format and opcode are read back from the bytes and written to
crates/amdgpu-target/src/gfx1250_co/tables.rs.

    python3 tools/gen_enc_tables_gfx1250.py [--mc /opt/rocm/llvm/bin/llvm-mc] [--check]
"""
import os, re, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "crates/amdgpu-target/src/gfx1250_co/tables.rs")

VOP1 = ["v_mov_b32", "v_exp_f32", "v_log_f32", "v_rcp_f32", "v_rsq_f32", "v_sqrt_f32", "v_sin_f32", "v_cos_f32", "v_cvt_f32_i32", "v_cvt_f32_u32",
        "v_cvt_i32_f32", "v_cvt_u32_f32", "v_cvt_f32_f16", "v_cvt_f16_f32", "v_not_b32", "v_readfirstlane_b32", "v_floor_f32", "v_ceil_f32",
        "v_trunc_f32", "v_rndne_f32", "v_fract_f32", "v_ffbl_b32", "v_ffbh_u32", "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2",
        "v_cvt_f32_ubyte3", "v_nop", "v_swap_b32"]
VOP2 = ["v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32", "v_fmac_f32", "v_mul_u32_u24", "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32",
        "v_and_b32", "v_or_b32", "v_xor_b32", "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32", "v_max_num_f32", "v_min_num_f32", "v_min_i32",
        "v_max_i32", "v_min_u32", "v_max_u32", "v_cndmask_b32", "v_add_co_ci_u32", "v_sub_co_ci_u32"]
VOP3 = ["v_mul_lo_u32", "v_mul_hi_u32", "v_mad_u32_u24", "v_fma_f32", "v_bfe_u32", "v_bfe_i32", "v_lshl_add_u32", "v_add_lshl_u32", "v_lshl_or_b32",
        "v_and_or_b32", "v_or3_b32", "v_xad_u32", "v_add3_u32", "v_med3_f32", "v_med3_i32", "v_min3_i32", "v_max3_i32", "v_perm_b32",
        "v_alignbit_b32", "v_bfi_b32", "v_cvt_pk_bf16_f32", "v_add_co_u32", "v_sub_co_u32"]
VOPC = ["v_cmp_lt_f32", "v_cmp_le_f32", "v_cmp_gt_f32", "v_cmp_ge_f32", "v_cmp_eq_f32", "v_cmp_lg_f32", "v_cmp_u_f32", "v_cmp_o_f32",
        "v_cmp_neq_f32", "v_cmp_nlt_f32", "v_cmp_nle_f32", "v_cmp_ngt_f32", "v_cmp_nge_f32", "v_cmp_lt_i32", "v_cmp_le_i32", "v_cmp_gt_i32",
        "v_cmp_ge_i32", "v_cmp_eq_i32", "v_cmp_ne_i32", "v_cmp_lt_u32", "v_cmp_le_u32", "v_cmp_gt_u32", "v_cmp_ge_u32", "v_cmp_eq_u32",
        "v_cmp_ne_u32"]
SOP1 = ["s_mov_b32", "s_mov_b64", "s_barrier_signal", "s_not_b32", "s_bitset0_b32", "s_bitset1_b32"]
SOP2 = ["s_add_co_u32", "s_add_co_ci_u32", "s_sub_co_u32", "s_sub_co_ci_u32", "s_lshl_b32", "s_lshr_b32", "s_ashr_i32", "s_and_b32", "s_or_b32",
        "s_xor_b32", "s_mul_i32", "s_mul_hi_u32", "s_min_i32", "s_max_i32", "s_min_u32", "s_max_u32", "s_cselect_b32", "s_bfe_u32", "s_bfe_i32",
        "s_lshl_b64", "s_lshr_b64", "s_add_co_i32", "s_sub_co_i32", "s_add_nc_u64"]
SOPC = ["s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32", "s_cmp_eq_i32", "s_cmp_lg_i32",
        "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32"]
SOPP = ["s_nop", "s_endpgm", "s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz", "s_cbranch_execz",
        "s_cbranch_execnz", "s_setprio", "s_sleep", "s_barrier_wait", "s_wait_loadcnt", "s_wait_storecnt", "s_wait_dscnt", "s_wait_kmcnt",
        "s_wait_xcnt", "s_wait_asynccnt", "s_wait_tensorcnt", "s_wait_loadcnt_dscnt", "s_wait_storecnt_dscnt", "s_set_vgpr_msb"]
SMEM = [("s_load_b32", "s1"), ("s_load_b64", "s[2:3]"), ("s_load_b128", "s[4:7]"), ("s_load_b256", "s[8:15]"), ("s_load_b512", "s[16:31]")]
GLOBAL_LOAD = [("global_load_b32", "v1"), ("global_load_b64", "v[2:3]"), ("global_load_b96", "v[4:6]"), ("global_load_b128", "v[4:7]"),
               ("global_load_u16", "v1"), ("global_load_i16", "v1"), ("global_load_u8", "v1"), ("global_load_i8", "v1")]
GLOBAL_STORE = [("global_store_b32", "v1"), ("global_store_b64", "v[2:3]"), ("global_store_b96", "v[4:6]"), ("global_store_b128", "v[4:7]"),
                ("global_store_b16", "v1"), ("global_store_b8", "v1")]
DS_LOAD = [("ds_load_b32", "v1"), ("ds_load_b64", "v[2:3]"), ("ds_load_b128", "v[4:7]"), ("ds_load_u16", "v1"), ("ds_load_u8", "v1")]
DS_STORE = [("ds_store_b32", "v1"), ("ds_store_b64", "v[2:3]"), ("ds_store_b128", "v[4:7]"), ("ds_store_b16", "v1"), ("ds_store_b8", "v1")]
TDM = ["tensor_load_to_lds", "tensor_store_from_lds"]
BUF_LOAD = ["buffer_load_b32", "buffer_load_b64", "buffer_load_b96", "buffer_load_b128", "buffer_load_u8", "buffer_load_i8", "buffer_load_u16", "buffer_load_i16"]
BUF_STORE = ["buffer_store_b8", "buffer_store_b16", "buffer_store_b32", "buffer_store_b64", "buffer_store_b96", "buffer_store_b128"]
BUF_ATOMIC = ["buffer_atomic_add_u32", "buffer_atomic_add_f32"]
# the kernel prologue (see the gfx1250 emitter)
WMMA = [("v_wmma_f32_16x16x32_bf16", "v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[16:23], v[0:7]"),
        ("v_wmma_f32_16x16x64_fp8_fp8", "v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[16:23], v[24:31]"),
        ("v_wmma_f32_16x16x128_fp8_fp8", "v_wmma_f32_16x16x128_fp8_fp8 v[8:15], v[16:31], v[32:47], v[48:55]"),
        ("v_wmma_scale_f32_32x16x128_f4", "v_wmma_scale_f32_32x16x128_f4 v[64:79], v[16:31], v[32:39], v[48:63], v3, v4")]
EXTRA = [("s_wait_alu", "s_wait_alu 0xfff"), ("s_barrier_signal_isfirst", "s_barrier_signal_isfirst -1"), ("s_barrier_init", "s_barrier_init m0"), ("s_barrier_join", "s_barrier_join 1"), ("s_barrier_leave", "s_barrier_leave"), ("s_setreg_imm32_b32", "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1"), ("global_prefetch_b8", "global_prefetch_b8 v0, s[6:7] scope:SCOPE_SE")]


def samples():
    s = []
    for m in VOP1:
        s.append((m, f"{m}_e32 v1, v2" if m not in ("v_nop", "v_readfirstlane_b32") else ("v_nop" if m == "v_nop" else "v_readfirstlane_b32 s1, v2")))
    for m in VOP2:
        s.append((m, f"{m}_e32 v1, v2, v3, vcc_lo" if m == "v_cndmask_b32" else (f"{m}_e32 v1, vcc_lo, v2, v3, vcc_lo" if "co_ci" in m else f"{m}_e32 v1, v2, v3")))
    for m in VOP3:
        if m in ("v_add_co_u32", "v_sub_co_u32"):
            s.append((m, f"{m} v1, vcc_lo, v2, v3"))
        elif m in ("v_mul_lo_u32", "v_mul_hi_u32", "v_cvt_pk_bf16_f32"):
            s.append((m, f"{m} v1, v2, v3"))
        else:
            s.append((m, f"{m} v1, v2, v3, v4"))
    for m in VOPC:
        s.append((m, f"{m}_e32 vcc_lo, v1, v2"))
    for m in SOP1:
        s.append((m, "s_barrier_signal -1" if m == "s_barrier_signal" else (f"{m} s[2:3], s[4:5]" if m.endswith("b64") else f"{m} s1, s2")))
    for m in SOP2:
        s.append((m, f"{m} s[2:3], s[4:5], 1" if m.endswith("b64") else (f"{m} s[2:3], s[4:5], s[6:7]" if m.endswith("u64") else f"{m} s1, s2, s3")))
    for m in SOPC:
        s.append((m, f"{m} s1, s2"))
    for m in SOPP:
        s.append((m, "s_endpgm" if m == "s_endpgm" else (f"{m} 4" if "branch" in m else f"{m} 0")))
    for m, d in SMEM:
        s.append((m, f"{m} {d}, s[0:1], 0x10"))
    for m, d in GLOBAL_LOAD:
        s.append((m, f"{m} {d}, v[8:9], off offset:16"))
    for m, d in GLOBAL_STORE:
        s.append((m, f"{m} v[8:9], {d}, off offset:16"))
    for m, d in DS_LOAD:
        s.append((m, f"{m} {d}, v9 offset:16"))
    for m, d in DS_STORE:
        s.append((m, f"{m} v9, {d} offset:16"))
    for m in TDM:
        s.append((m, f"{m} s[0:3], s[4:11]"))
    w = {"b64": "v[2:3]", "b96": "v[4:6]", "b128": "v[4:7]"}
    for m in BUF_LOAD + BUF_STORE:
        s.append((m, f"{m} {w.get(m.rsplit('_', 1)[1], 'v1')}, v9, s[4:7], s3 offen offset:16"))
    for m in BUF_ATOMIC:
        s.append((m, f"{m} v1, v9, s[4:7], s3 offen"))
    s.extend(EXTRA)
    s.extend(WMMA)
    return s


def fmt(b):
    d0 = int.from_bytes(b[:4], "little")
    # VOP3PX2: a scale prefix (VOP3P) and the matrix instruction (VOP3P): the second one's opcode
    if len(b) == 16 and d0 >> 26 == 0x33 and int.from_bytes(b[8:12], "little") >> 26 == 0x33:
        return "VOP3PX2", (int.from_bytes(b[8:12], "little") >> 16) & 0xFF
    if d0 >> 23 == 0x17D:
        return "SOP1", (d0 >> 8) & 0xFF
    if d0 >> 23 == 0x17E:
        return "SOPC", (d0 >> 16) & 0x7F
    if d0 >> 23 == 0x17F:
        return "SOPP", (d0 >> 16) & 0x7F
    if d0 >> 28 == 0xB:
        return "SOPK", (d0 >> 23) & 0x1F
    if d0 >> 30 == 2:
        return "SOP2", (d0 >> 23) & 0x7F
    if d0 >> 26 == 0x3D:
        return "SMEM", (d0 >> 13) & 0xFF
    if d0 >> 25 == 0x3F:
        return "VOP1", (d0 >> 9) & 0xFF
    if d0 >> 25 == 0x3E:
        return "VOPC", (d0 >> 17) & 0xFF
    if d0 >> 31 == 0:
        return "VOP2", (d0 >> 25) & 0x3F
    if d0 >> 26 == 0x35:
        return "VOP3", (d0 >> 16) & 0x3FF
    if d0 >> 26 == 0x33:
        return "VOP3P", (d0 >> 16) & 0xFF
    if d0 >> 26 == 0x36:
        return "VDS", (d0 >> 18) & 0xFF
    if d0 >> 24 == 0xEE:
        return "VGLOBAL", (d0 >> 14) & 0xFF
    if d0 >> 26 == 0x34:
        return "VIMAGE", (d0 >> 14) & 0xFF
    if d0 >> 26 == 0x31:
        return "VBUFFER", (d0 >> 14) & 0xFF
    return "UNK", d0 >> 24


def main():
    mc = "/opt/rocm/llvm/bin/llvm-mc"
    if "--mc" in sys.argv:
        mc = sys.argv[sys.argv.index("--mc") + 1]
    ss = samples()
    src = "\n".join(l for _, l in ss) + "\n"
    r = subprocess.run([mc, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-show-encoding"], input=src, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("llvm-mc failed:\n" + r.stderr)
    encs = [l for l in r.stdout.splitlines() if "encoding:" in l]
    if len(encs) != len(ss):
        sys.exit(f"{len(encs)} encodings for {len(ss)} samples:\n{r.stderr}")
    rows = []
    for (m, line), e in zip(ss, encs):
        b = bytes(int(x, 16) for x in re.search(r"encoding: \[(.*)\]", e).group(1).split(","))
        f, o = fmt(b)
        # the assembler may promote e32 forms with an inline constant / SGPR to e64: the e32 format is what the table records
        if f == "VOP3" and m in VOP2 + VOP1 + VOPC:
            sys.exit(f"{line}: promoted to VOP3 (choose another sample)")
        rows.append((m, f, o))
    out = ["//! GENERATED by tools/gen_enc_tables_gfx1250.py from the gfx1250 llvm-mc oracle (backend.gfx1250.lock). Do not edit.", "",
           "pub static OPS: &[(&str, &str, u16)] = &["]
    for m, f, o in sorted(rows):
        out.append(f'    ("{m}", "{f}", {o}),')
    out.append("];")
    text = "\n".join(out) + "\n"
    if "--check" in sys.argv:
        cur = open(OUT).read() if os.path.exists(OUT) else ""
        if cur != text:
            sys.exit("gfx1250 tables are stale: run tools/gen_enc_tables_gfx1250.py")
        print("gfx1250 tables up to date")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(text)
    print(f"wrote {OUT}: {len(rows)} mnemonics")


if __name__ == "__main__":
    main()
