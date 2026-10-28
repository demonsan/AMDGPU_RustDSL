#!/usr/bin/env python3
# Generates mb.s (many microbenchmark kernels) + tests.json (name -> divisor N).
import json, sys

HDR = ".text\ns_endpgm\ns_endpgm\n.p2align 8\n"
def kernel(name, setup, body, n, chk="v1", extra=0, wait="s_waitcnt vmcnt(0) lgkmcnt(0)"):
    s = []
    a = s.append
    a(".globl %s\n.p2align 8\n.type %s,@function\n%s:" % (name, name, name))
    a("  s_load_dwordx4 s[4:7], s[0:1], 0")
    a("  s_waitcnt lgkmcnt(0)")
    a("  v_readfirstlane_b32 s8, v0")
    a("  s_lshr_b32 s8, s8, 6")
    a("  s_lshl_b32 s9, s8, 9")          # wave * 512 bytes
    a("  v_mov_b32 v236, s9")
    a("  v_mbcnt_lo_u32_b32 v237, -1, 0")
    a("  v_mbcnt_hi_u32_b32 v237, -1, v237")
    a("  v_lshlrev_b32 v241, 4, v237")   # lane*16
    a("  v_lshlrev_b32 v242, 3, v237")   # lane*8
    a("  v_lshlrev_b32 v243, 2, v237")   # lane*4
    a("  v_mov_b32 v244, v241")
    a("  s_mov_b32 s30, 0xaaaaaaaa\n  s_mov_b32 s31, 0xaaaaaaaa\n  s_mov_b64 vcc, s[30:31]")
    a("  v_mov_b32 v2, 1.0\n  v_mov_b32 v3, 1.0\n  v_mov_b32 v4, 0.5\n  v_mov_b32 v5, 0.5\n  v_mov_b32 v1, 0")
    a("  v_mov_b32 v250, 127\n  v_mov_b32 v251, 127")
    s.extend("  " + x for x in setup)
    a("  s_waitcnt vmcnt(0) lgkmcnt(0)")
    a("  s_barrier")
    a("  s_nop 15\n  s_nop 15")
    a("  s_memrealtime s[18:19]\n  s_memtime s[10:11]")
    a("  s_waitcnt lgkmcnt(0)")
    s.append("  ;---- body")
    for l in body:
        if l == "@MARK":
            a("  s_memtime s[16:17]")
        else:
            a("  " + l)
    s.append("  ;---- end body")
    a("  " + wait)
    a("  s_memtime s[12:13]\n  s_memrealtime s[20:21]")
    a("  s_waitcnt lgkmcnt(0)")
    a("  s_sub_u32 s22, s12, s10\n  s_sub_u32 s23, s20, s18\n  s_sub_u32 s24, s16, s10")
    a("  v_mov_b32 v232, s22\n  v_mov_b32 v233, s23\n  v_mov_b32 v234, s24\n  v_mov_b32 v235, %s" % chk)
    a("  s_mov_b64 exec, 1")
    a("  global_store_dwordx4 v236, v[232:235], s[4:5]")
    for j in range(extra):
        a("  global_store_dword v236, v%d, s[4:5] offset:%d" % (200 + j, 32 + 4 * j))
    a("  s_waitcnt vmcnt(0)")
    a("  s_endpgm")
    a(".size %s, .-%s\n" % (name, name))
    return "\n".join(s) + "\n"

KS = []   # (name, text, n)
def add(name, setup, body, n, **kw):
    KS.append((name, kernel(name, setup, body, n, **kw), n))

add('aa_pad0', [], ['s_nop 0'], 1)
add('aa_dummy_first', [], ['s_nop 0'], 1)  # first kernel in the .co faults on this node; keep a sacrificial one
N = 256
# ---- 1. VALU ops
def D(i): return 16 + i % 32
ops = {
 "v_add_f32":   (lambda i: "v_add_f32 v1, v1, v2", lambda i: "v_add_f32 v%d, v2, v3" % D(i)),
 "v_fma_f32":   (lambda i: "v_fma_f32 v1, v1, v2, v3", lambda i: "v_fma_f32 v%d, v2, v3, v4" % D(i)),
 "v_mul_f32":   (lambda i: "v_mul_f32 v1, v1, v2", lambda i: "v_mul_f32 v%d, v2, v3" % D(i)),
 "v_max3_f32":  (lambda i: "v_max3_f32 v1, v1, v2, v3", lambda i: "v_max3_f32 v%d, v2, v3, v4" % D(i)),
 "v_cvt_pk_fp8_f32": (lambda i: "v_cvt_pk_fp8_f32 v1, v1, v2", lambda i: "v_cvt_pk_fp8_f32 v%d, v2, v3" % D(i)),
 "v_cvt_pk_bf16_f32": (lambda i: "v_cvt_pk_bf16_f32 v1, v1, v2", lambda i: "v_cvt_pk_bf16_f32 v%d, v2, v3" % D(i)),
 "v_cndmask_b32": (lambda i: "v_cndmask_b32 v1, v1, v2, vcc", lambda i: "v_cndmask_b32 v%d, v2, v3, vcc" % D(i)),
 "v_mov_b32":   (lambda i: "v_mov_b32 v%d, v%d" % (1 + (i + 1) % 2, 1 + i % 2), lambda i: "v_mov_b32 v%d, v2" % D(i)),
 "v_pk_add_f32": (lambda i: "v_pk_add_f32 v[10:11], v[10:11], v[2:3]", lambda i: "v_pk_add_f32 v[%d:%d], v[2:3], v[4:5]" % (2*(D(i)//2), 2*(D(i)//2)+1)),
 "v_pk_mul_f32": (lambda i: "v_pk_mul_f32 v[10:11], v[10:11], v[2:3]", lambda i: "v_pk_mul_f32 v[%d:%d], v[2:3], v[4:5]" % (2*(D(i)//2), 2*(D(i)//2)+1)),
 "v_permlane32_swap_b32": (lambda i: "v_permlane32_swap_b32 v1, v2", lambda i: "v_permlane32_swap_b32 v%d, v%d" % (16+2*(i%16), 17+2*(i%16))),
 "v_exp_f32":   (lambda i: "v_exp_f32 v1, v1", lambda i: "v_exp_f32 v%d, v2" % D(i)),
}
for op, (dep, ind) in ops.items():
    for kind, f in (("dep", dep), ("ind", ind)):
        add("%s_%s" % (op, kind), ["v_mov_b32 v10, 0\n  v_mov_b32 v11, 0"], [f(i) for i in range(N)], N)

# ---- 2. trans mix: 1 exp + k independent v_add
G = 32
for k in range(0, 9):
    body = []
    for g in range(G):
        body.append("v_exp_f32 v%d, v2" % (80 + g % 16))
        for j in range(k): body.append("v_add_f32 v%d, v2, v3" % (16 + (g * 9 + j) % 32))
    add("exp_mix_add%d" % k, [], body, G)
# k VALU alone for comparison
for k in (1, 4, 8):
    body = []
    for g in range(G):
        for j in range(k): body.append("v_add_f32 v%d, v2, v3" % (16 + (g * 9 + j) % 32))
    add("addonly_%d" % k, [], body, G)

# ---- 3. MFMA
mf_setup = ["v_mov_b32 v%d, 0x38383838" % r for r in range(16, 32)]
mf_setup += ["v_mov_b32 v%d, 0" % r for r in range(32, 160)]
def DS(i, w, rot):  # D base for 32x32 (16 regs) / w regs
    return 32 + w * (i % rot)
types = {
 "scale32": (16, lambda d, c: "v_mfma_scale_f32_32x32x64_f8f6f4 v[%d:%d], v[16:23], v[24:31], v[%d:%d], v250, v251" % (d, d+15, c, c+15)),
 "f32x32x16": (16, lambda d, c: "v_mfma_f32_32x32x16_fp8_fp8 v[%d:%d], v[16:17], v[24:25], v[%d:%d]" % (d, d+15, c, c+15)),
 "f16x16x32": (4,  lambda d, c: "v_mfma_f32_16x16x32_fp8_fp8 v[%d:%d], v[16:17], v[24:25], v[%d:%d]" % (d, d+3, c, c+3)),
}
NM = 32
for t, (w, f) in types.items():
    add("mfma_%s_dep" % t, mf_setup, [f(32, 32) for i in range(NM)], NM, chk="v32")
    for rot in ((2, 4, 8) if t == "scale32" else (8,)):
        add("mfma_%s_ind%d" % (t, rot), mf_setup, [f(DS(i, w, rot), DS(i, w, rot)) for i in range(NM)], NM, chk="v32")
# interleave
fS = types["scale32"][1]
for k in list(range(0, 17)):
    for kind in ("dep", "ind8"):
        body = []
        for g in range(NM):
            d = 32 if kind == "dep" else DS(g, 16, 8)
            body.append(fS(d, d))
            for j in range(k): body.append("v_add_f32 v%d, v2, v3" % (200 + (g * 17 + j) % 16))
        add("mfma_il_%s_k%d" % (kind, k), mf_setup, body, NM, chk="v32")
# interleave with 16x16x32 (shorter) and 32x32x16
for t in ("f32x32x16", "f16x16x32"):
    w, f = types[t]
    for k in (0, 1, 2, 4, 8):
        body = []
        for g in range(NM):
            d = DS(g, w, 8)
            body.append(f(d, d))
            for j in range(k): body.append("v_add_f32 v%d, v2, v3" % (200 + (g * 17 + j) % 16))
        add("mfma_il_%s_ind8_k%d" % (t, k), mf_setup, body, NM, chk="v32")
# MFMA -> dependent VALU pair (interlock cost)
for t, (w, f) in types.items():
    body = []
    for g in range(16):
        body.append(f(DS(g, w, 1), DS(g, w, 1)).replace("v250, v251", "v250, v251"))
        body.append("v_mov_b32 v%d, v32" % (200 + g % 16))
    add("mfma_%s_depvalu_pair" % t, mf_setup, body, 16, chk="v200")
    body = []
    for g in range(16):
        body.append(f(DS(g, w, 8), DS(g, w, 8)))
        body.append("v_mov_b32 v%d, v2" % (200 + g % 16))
    add("mfma_%s_indvalu_pair" % t, mf_setup, body, 16, chk="v200")
# hazard sweeps: Z-form accumulator via zeroed C, sentinel in D[0], read after m s_nops
zf = {
 "scale32": "v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[16:23], v[24:31], 0, v250, v251",
 "f32x32x16": "v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[16:17], v[24:25], 0",
 "f16x16x32": "v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[16:17], v[24:25], 0",
}
ms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 40]
for t, z in zf.items():
    body = []
    for j, m in enumerate(ms):
        body += ["s_nop 15"] * 6
        body += ["v_mov_b32 v32, 0xbf800000", "s_nop 7", z]
        # m wait states via s_nop (s_nop n = n+1 states)
        left = m
        while left > 0:
            c = min(left, 16); body.append("s_nop %d" % (c - 1)); left -= c
        body.append("v_mov_b32 v%d, v32" % (200 + j))
    add("mfma_haz_%s" % t, mf_setup, body, 1, chk="v200", extra=len(ms))

# ---- 4. LDS
LN = 64
add("lds_rd128_ind", [], ["ds_read_b128 v[%d:%d], v241 offset:%d" % (16+4*(i%16), 19+4*(i%16), (i%16)*1024) for i in range(LN)], LN)
add("lds_rd_b32_ind", [], ["ds_read_b32 v%d, v243 offset:%d" % (16+i%32, (i%16)*256) for i in range(LN)], LN)
add("lds_rd_b64_tr_b8_ind", [], ["ds_read_b64_tr_b8 v[%d:%d], v242 offset:%d" % (16+2*(i%16), 17+2*(i%16), (i%16)*512) for i in range(LN)], LN)
add("lds_wr128_ind", [], ["ds_write_b128 v241, v[16:19] offset:%d" % ((i%16)*1024) for i in range(LN)], LN)
for nm, ins in (("rd128", "ds_read_b128 v[16:19], v241"), ("rd_b32", "ds_read_b32 v16, v243"),
                ("rd_b64_tr_b8", "ds_read_b64_tr_b8 v[16:17], v242"), ("wr128", "ds_write_b128 v241, v[16:19]")):
    add("lds_%s_lat" % nm, [], [ins, "s_waitcnt lgkmcnt(0)"] * 32, 32)
# issue-only variants (mark after last issue, before wait)
for nm, f in (("rd128", lambda i: "ds_read_b128 v[%d:%d], v241 offset:%d" % (16+4*(i%16), 19+4*(i%16), (i%16)*1024)),
              ("wr128", lambda i: "ds_write_b128 v241, v[16:19] offset:%d" % ((i%16)*1024))):
    add("lds_%s_issue8" % nm, [], [f(i) for i in range(8)] + ["@MARK"], 8)
# ---- 5. global
g_setup = ["global_load_dwordx4 v[16:19], v244, s[6:7]", "global_load_dword v20, v243, s[6:7]", "s_waitcnt vmcnt(0)"]
add("gl_dwordx4_lat", g_setup, ["global_load_dwordx4 v[16:19], v244, s[6:7]", "s_waitcnt vmcnt(0)"] * 16, 16)
add("gl_dword_lat", g_setup, ["global_load_dword v16, v243, s[6:7]", "s_waitcnt vmcnt(0)"] * 16, 16)
add("gl_dwordx4_4ind", g_setup, ["global_load_dwordx4 v[%d:%d], v244, s[6:7] offset:%d" % (16+4*i, 19+4*i, i*1024) for i in range(4)] + ["@MARK"], 4)
add("gl_dword_4ind", g_setup, ["global_load_dword v%d, v243, s[6:7] offset:%d" % (16+i, i*1024) for i in range(4)] + ["@MARK"], 4)
add("gl_dwordx4_32ind", g_setup, ["global_load_dwordx4 v[%d:%d], v244, s[6:7] offset:%d" % (16+4*(i%16), 19+4*(i%16), (i%4)*1024) for i in range(32)] + ["@MARK"], 32)
# ---- 6. misc
add("s_barrier_x64", [], ["s_barrier"] * 64, 64)
add("s_setprio_x256", [], ["s_setprio %d" % (i % 4) for i in range(256)], 256)
add("s_nop15_x64", [], ["s_nop 15"] * 64, 64)
add("s_nop0_x256", [], ["s_nop 0"] * 256, 256)
add("s_mov_x256", [], ["s_mov_b32 s40, s41"] * 256, 256)
add("v_add_ind_x256_long", [], ["v_add_f32 v%d, v2, v3" % D(i) for i in range(1024)], 1024)

# ================= round 2 additions =================
# clock calibration: long s_nop loop, compare s_memtime delta vs s_memrealtime (100 MHz) delta
add("clk_cal_loop", [], ["s_mov_b32 s40, 3000", "cal_loop:"] + ["s_nop 15"] * 16 + ["s_sub_u32 s40, s40, 1", "s_cmp_lg_u32 s40, 0", "s_cbranch_scc1 cal_loop"], 1)
# cndmask variants
add("v_cndmask_e64_sgpr_ind", [], ["v_cndmask_b32_e64 v%d, v2, v3, s[30:31]" % D(i) for i in range(N)], N)
add("v_cndmask_e64_sgpr_dep", [], ["v_cndmask_b32_e64 v1, v1, v2, s[30:31]" for i in range(N)], N)
# larger-N LDS/global for slope (N=128 vs N=64)
L2 = 128
add("lds_rd128_ind128", [], ["ds_read_b128 v[%d:%d], v241 offset:%d" % (16+4*(i%16), 19+4*(i%16), (i%16)*1024) for i in range(L2)], L2)
add("lds_rd_b32_ind128", [], ["ds_read_b32 v%d, v243 offset:%d" % (16+i%32, (i%16)*256) for i in range(L2)], L2)
add("lds_rd_b64_tr_b8_ind128", [], ["ds_read_b64_tr_b8 v[%d:%d], v242 offset:%d" % (16+2*(i%16), 17+2*(i%16), (i%16)*512) for i in range(L2)], L2)
add("lds_wr128_ind128", [], ["ds_write_b128 v241, v[16:19] offset:%d" % ((i%16)*1024) for i in range(L2)], L2)
add("gl_dwordx4_ind64", g_setup, ["global_load_dwordx4 v[%d:%d], v244, s[6:7] offset:%d" % (16+4*(i%16), 19+4*(i%16), (i%4)*1024) for i in range(64)] + ["@MARK"], 64)
# MFMA hazard sweep, reading D[0] (first) and D[last]; fine sweep of s_nop count
ms2 = list(range(0, 21)) + [22, 24, 26, 28, 32]
last = {"scale32": 47, "f32x32x16": 47, "f16x16x32": 35}
for t, z in zf.items():
    for which, reg in (("first", 32), ("last", last[t])):
        body = []
        for j, m in enumerate(ms2):
            body += ["s_nop 15"] * 6
            body += ["v_mov_b32 v32, 0xbf800000", "v_mov_b32 v%d, 0xbf800000" % last[t], "s_nop 7", z]
            left = m
            while left > 0:
                c = min(left, 16); body.append("s_nop %d" % (c - 1)); left -= c
            body.append("v_mov_b32 v%d, v%d" % (200 + j, reg))
        add("mfma_haz2_%s_%s" % (t, which), mf_setup, body, 1, chk="v200", extra=len(ms2))
# VALU write of A operand then MFMA reads it after m wait states
body = []
for j, m in enumerate([0, 1, 2, 3, 4]):
    body += ["s_nop 15"] * 6
    body += ["v_mov_b32 v%d, 0" % r for r in range(16, 24)] + ["s_nop 7"]
    body += ["v_mov_b32 v%d, 0x38383838" % r for r in range(16, 24)]
    left = m
    while left > 0:
        c = min(left, 16); body.append("s_nop %d" % (c - 1)); left -= c
    body.append(zf["scale32"])
    body += ["s_nop 15"] * 6
    body.append("v_mov_b32 v%d, v32" % (200 + j))
add("mfma_haz_srcA_after_valu_scale32", mf_setup, body, 1, chk="v200", extra=5)
# ---- global latency: L1 hit vs L2 hit (L1 thrashed by touching 1 MB first)
thrash = ["s_mov_b64 s[44:45], s[6:7]"]
for i in range(256):
    thrash += ["global_load_dword v20, v243, s[6:7]", "s_add_u32 s6, s6, 4096", "s_addc_u32 s7, s7, 0"]
thrash += ["s_waitcnt vmcnt(0)", "s_mov_b64 s[6:7], s[44:45]"]
adv = ["s_add_u32 s6, s6, 4096", "s_addc_u32 s7, s7, 0"]
same = ["s_add_u32 s6, s6, 0", "s_addc_u32 s7, s7, 0"]
warm1 = ["global_load_dword v20, v243, s[6:7]", "s_waitcnt vmcnt(0)"]
add("gl_dword_lat_L2", thrash, ["global_load_dword v16, v243, s[6:7]", "s_waitcnt vmcnt(0)"] + [] , 1)
add("gl_dword_lat_L2x8", thrash, sum([["global_load_dword v16, v243, s[6:7]", "s_waitcnt vmcnt(0)"] + adv for i in range(8)], []), 8)
add("gl_dword_lat_L1x8", warm1, sum([["global_load_dword v16, v243, s[6:7]", "s_waitcnt vmcnt(0)"] + same for i in range(8)], []), 8)
add("gl_dwordx4_lat_L2x8", thrash, sum([["global_load_dwordx4 v[16:19], v244, s[6:7]", "s_waitcnt vmcnt(0)"] + adv for i in range(8)], []), 8)
add("gl_dwordx4_lat_L1x8", warm1, sum([["global_load_dwordx4 v[16:19], v244, s[6:7]", "s_waitcnt vmcnt(0)"] + same for i in range(8)], []), 8)
add("gl_dword_lat_sc0sc1x8", warm1, sum([["global_load_dword v16, v243, s[6:7] sc0 sc1", "s_waitcnt vmcnt(0)"] + same for i in range(8)], []), 8)
add("gl_dword_lat_sc1x8", warm1, sum([["global_load_dword v16, v243, s[6:7] sc1", "s_waitcnt vmcnt(0)"] + same for i in range(8)], []), 8)
out = [HDR]
meta = []
for name, text, n in KS:
    out.append(text)
    meta.append(name)
out.append("\n")
for name, text, n in KS:
    out.append("""
.amdhsa_kernel %s
  .amdhsa_group_segment_fixed_size 65536
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256
  .amdhsa_next_free_sgpr 64
  .amdhsa_accum_offset 256
.end_amdhsa_kernel
""" % name)
out.append(".amdgpu_metadata\n---\namdhsa.kernels:\n")
for name, text, n in KS:
    out.append("""  - .name: "%s"
    .symbol: %s.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .max_flat_workgroup_size: 1024
    .sgpr_count: 64
    .vgpr_count: 256
    .args:
      - {.name: "o", .offset: 0, .size: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: "i", .offset: 8, .size: 8, .value_kind: global_buffer, .address_space: global}
""" % (name, name))
out.append("amdhsa.target: amdgcn-amd-amdhsa--gfx950\namdhsa.version:\n  - 1\n  - 0\n...\n.end_amdgpu_metadata\n")
open("mb.s", "w").write("".join(out))
json.dump({n: m for n, _, m in KS}, open("tests.json", "w"))
print(len(KS), "kernels")







