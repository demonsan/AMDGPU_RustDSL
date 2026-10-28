#!/usr/bin/env python3
import json
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


KS=[]
def add(name, setup, body, n, **kw):
    KS.append((name, kernel(name, setup, body, n, **kw), n))

add('aa_pad0', [], ['s_nop 0'], 1)
add('aa_dummy_first', [], ['s_nop 0'], 1)
LSETUP = ["s_mov_b32 s40, 0", "s_mov_b32 s41, 2000", "v_mov_b32 v20, 1", "v_mov_b32 v21, 0", "v_mov_b32 v23, 2000"]
def mul(i): return "v_mul_f32 v%d, v2, v3" % (100 + i % 32)
CNT = {}
def loop(name, kind, nb=16, al=6, k=0, shift=0, iters=2000):
    L = name + "_L"; X = name + "_X"; N1 = name + "_N1"; N2 = name + "_N2"
    setup = ["s_mov_b32 s40, 0", "s_mov_b32 s41, %d" % iters, "v_mov_b32 v20, 1", "v_mov_b32 v21, 0", "v_mov_b32 v23, %d" % iters]
    b = [".p2align %d" % al] + ["s_nop 0"] * shift + [L + ":"] + [mul(i) for i in range(nb)]
    va = ["v_add_u32 v21, v21, v20", "v_cmp_lt_i32_e32 vcc, v21, v23"]
    ex = ["v_mul_f32 v%d, v2, v3" % (140 + i) for i in range(k)]
    chk = "v21"
    if kind == "salu":
        b += ["s_add_u32 s40, s40, 1", "s_cmp_lt_u32 s40, s41", "s_cbranch_scc1 " + L]; chk = "s40"
    elif kind == "salu_fwd":
        b += ["s_add_u32 s40, s40, 1", "s_cmp_lt_u32 s40, s41", "s_branch " + N1, N1 + ":", "s_cbranch_scc1 " + L]; chk = "s40"
    elif kind == "salu_fwd2":
        b += ["s_add_u32 s40, s40, 1", "s_cmp_lt_u32 s40, s41", "s_branch " + N1, N1 + ":", "s_branch " + N2, N2 + ":", "s_cbranch_scc1 " + L]; chk = "s40"
    elif kind == "salu_nop":   # fall-through reference == salu
        b += ["s_add_u32 s40, s40, 1", "s_cmp_lt_u32 s40, s41", "s_cbranch_scc1 " + L]; chk = "s40"
    elif kind == "vcc":
        b += va + ex + ["s_cbranch_vccnz " + L]
    elif kind == "vcc_fwd":
        b += va + ["s_branch " + N1, N1 + ":", "s_cbranch_vccnz " + L]
    elif kind == "full":     # unoptimised: vccz exit(not taken); s_branch next; next: s_branch loop
        b += va + ["s_cbranch_vccz " + X, "s_branch " + N1, N1 + ":", "s_branch " + L, X + ":"]
    elif kind == "vccz_back":
        b += va + ["s_cbranch_vccz " + X, "s_branch " + L, X + ":"]
    elif kind == "vccz_nt_salu":   # salu loop + a not-taken vccz branch in the loop (vcc from v_cmp done before)
        b += va + ["s_cbranch_vccz " + X, "s_add_u32 s40, s40, 1", "s_cmp_lt_u32 s40, s41", "s_cbranch_scc1 " + L, X + ":"]; chk = "s40"
    elif kind == "uncond_exit":   # salu cmp, branch scc0 exit, s_branch loop
        b += ["s_add_u32 s40, s40, 1", "s_cmp_lt_u32 s40, s41", "s_cbranch_scc0 " + X, "s_branch " + L, X + ":"]; chk = "s40"
    add(name, setup, b, iters, chk=chk)

for nw, (nb, it) in {8: (8, 2000), 16: (16, 2000), 64: (64, 2000), 256: (256, 500)}.items():
    pass
loop("t1_salu", "salu")
for k in (0, 1, 2, 4, 8):
    loop("t2_vcc_k%d" % k, "vcc", k=k)
loop("t3_full", "full"); loop("t3_vcczback", "vccz_back"); loop("t3_vccfwd", "vcc_fwd")
loop("t3_salufwd", "salu_fwd"); loop("t3_salufwd2", "salu_fwd2"); loop("t3_vcczNT_salu", "vccz_nt_salu"); loop("t3_uncondexit", "uncond_exit")
for nb in (8, 16, 64, 256):
    it = 500 if nb == 256 else 2000
    for al in (4, 5, 6, 8):
        loop("t4_salu_b%d_a%d" % (nb, al), "salu", nb=nb, al=al, iters=it)
        loop("t4_vcc_b%d_a%d" % (nb, al), "vcc", nb=nb, al=al, iters=it)
for sh in range(1, 16):
    loop("t4_salu_b16_a6_s%d" % sh, "salu", nb=16, al=6, shift=sh)
for sh in (1, 2, 3, 5, 9, 15):
    loop("t4_vcc_b16_a6_s%d" % sh, "vcc", nb=16, al=6, shift=sh)
# unrolled references (n = 64 copies of [16 mul + SALU tail variants]); warm pass + MARK
def blk(name, c, tail, reps):
    b = []
    for r in range(reps):
        b += [mul(i) for i in range(16)] + [x.replace("@r", "%s_%d_%d" % (name, c, r)).replace("EXITU", name + "_X") for x in tail]
    return b
def addu(name, tail, reps=64):
    add(name, ["s_mov_b32 s40, 0", "s_mov_b32 s41, 100000000"], blk(name, 0, tail, reps) + ["@MARK"] + blk(name, 1, tail, reps) + [name + "_X:"], reps)
addu("u_mul16_only", [])
addu("u_mul16_salu2", ["s_add_u32 s40, s40, 1", "s_cmp_lt_u32 s40, s41"])
addu("u_mul16_salu2_nt_scc0", ["s_add_u32 s40, s40, 1", "s_cmp_lt_u32 s40, s41", "s_cbranch_scc0 EXITU"])
addu("u_mul16_fwd", ["s_branch NX@r", "NX@r:"])
addu("u_mul16_fwd2", ["s_branch NX@r", "NX@r:", "s_branch NY@r", "NY@r:"])
addu("u_mul16_nt_vccz", ["s_cbranch_vccz EXITU"])
for sh in range(1, 16):
    loop("t4_salu_b8_a6_s%d" % sh, "salu", nb=8, al=6, shift=sh)
for sh in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
    loop("t4_salu_b64_a6_s%d" % sh, "salu", nb=64, al=6, shift=sh)
# ---- 5 cmp/cndmask: warm pass, @MARK, timed pass
def add5(name, setup, body, n, **kw):
    add(name, setup, body + ["@MARK"] + body, n, **kw)
add("t5_markempty", [], ["@MARK"], 1)
P = 128
def D(i): return 16 + i % 32
cm = lambda i: "v_cmp_lt_i32_e32 vcc, v250, v251"
cn = lambda i: "v_cndmask_b32 v%d, v2, v3, vcc" % D(i)
cm64 = lambda i: "v_cmp_lt_i32_e64 s[32:33], v250, v251"
cn64 = lambda i: "v_cndmask_b32_e64 v%d, v2, v3, s[32:33]" % D(i)
add5("t5_cmp_only", [], [cm(i) for i in range(P)], P)
add5("t5_cmp_e64_only", [], [cm64(i) for i in range(P)], P)
add5("t5_cn_only_vcc", [], [cn(i) for i in range(P)], P)
add5("t5_cn_only_e64", [], [cn64(i) for i in range(P)], P)
add5("t5_pair_vcc", [], sum([[cm(i), cn(i)] for i in range(P)], []), P)
add5("t5_pair_vcc_gap1", [], sum([[cm(i), mul(i), cn(i)] for i in range(P)], []), P)
add5("t5_pair_vcc_gap2", [], sum([[cm(i), mul(i), mul(i+1), cn(i)] for i in range(P)], []), P)
add5("t5_pair_e64", [], sum([[cm64(i), cn64(i)] for i in range(P)], []), P)
add5("t5_pair_e64_gap1", [], sum([[cm64(i), mul(i), cn64(i)] for i in range(P)], []), P)
add5("t5_pair_e64_gap2", [], sum([[cm64(i), mul(i), mul(i+1), cn64(i)] for i in range(P)], []), P)
add5("t5_pair_e64_cmpvcc", [], sum([[cm(i), cn64(i)] for i in range(P)], []), P)  # cmp writes vcc, cndmask reads s[32:33] (no dep)
add5("t5_pair_cmp64_cnvcc", [], sum([[cm64(i), cn(i)] for i in range(P)], []), P)  # no dep, vcc const
add5("t5_mul_only", [], [mul(i) for i in range(P)], P)
add5("t5_smov_vcc_cn", [], sum([["s_mov_b64 vcc, s[30:31]", cn(i)] for i in range(P)], []), P)
add5("t5_smov_s32_cn64", [], sum([["s_mov_b64 s[32:33], s[30:31]", cn64(i)] for i in range(P)], []), P)
add5("t5_cn_vcc_2mul", [], sum([[cn(i), mul(i), mul(i+1)] for i in range(P)], []), P)
add5("t5_cn_e64_2mul", [], sum([[cn64(i), mul(i), mul(i+1)] for i in range(P)], []), P)
add5("t5_cn_vcc_dep", [], ["v_cndmask_b32 v1, v1, v2, vcc"] * P, P)
def emit(path, kernels):
    out = [HDR]
    for name, text, n in kernels:
        out.append(text)
    out.append("\n")
    for name, text, n in kernels:
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
    for name, text, n in kernels:
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
    open(path, "w").write("".join(out))


pad = KS[:2]
rest = KS[2:]
groups = {}
h = (len(rest) + 2) // 3
for gi in range(3):
    part = rest[gi*h:(gi+1)*h]
    emit("mb2_%d.s" % gi, pad + part)
    groups["mb2_%d.co" % gi] = [k[0] for k in part]
json.dump({n: m for n, _, m in KS}, open("tests2.json", "w"))
json.dump(groups, open("groups2.json", "w"), indent=1)
print(len(KS), "kernels")




