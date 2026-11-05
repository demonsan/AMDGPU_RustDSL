#!/usr/bin/env python3
"""MFMA dependency / operand-reuse probe (gfx950): issue interval, effective clock and power of back-to-back dependent pairs vs interleaved accumulators.

Kernel name: <ins>_<pat>_v<n>   ins: s32 = v_mfma_scale_f32_32x32x64_f8f6f4 (acc 16 VGPRs), s16 = v_mfma_scale_f32_16x16x128_f8f6f4 (acc 4 VGPRs)
  pat   indep    : MFMA m accumulates into accumulator m % 8 (in place), A and B fragments both vary: no operand reuse, no dependence for 8 MFMAs
        sameB    : like indep but every MFMA of the loop reads the SAME srcB fragment (our FMHA's interleaving: different accumulators sharing srcB)
        dep2     : back-to-back DEPENDENT pairs (the vendor FMHA): MFMA 2p and 2p+1 accumulate into the same accumulator p % 8 (srcC == previous vDst), the two
                   MFMAs of a pair use two different (A, B) fragments (the two K halves); consecutive pairs use different accumulators
        chain    : every MFMA accumulates into accumulator 0 (a fully dependent chain, the dependent-issue reference)
  n     number of independent VALU instructions (v_add_u32 on scratch registers) between consecutive MFMAs: 0 or 5
Each wave times its loop with s_memtime / s_memrealtime and stores (delta s_memtime, delta s_memrealtime, iterations, wave) to out[(workgroup*4 + wave)*16];
32 MFMAs per iteration. Operands: random fp8 bytes (bit 3 of every byte cleared: no NaN pattern) or zero (name suffix _z), scales ~127.
usage: mfma_dep.py out.s [iters=2048]
"""
import sys
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
out = open(sys.argv[1], "w")
LIT = [0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09]
NACC = 8
A0, B0, SC, VT, TMP, TID = 128, 160, 192, 194, 200, 201
ACCUM_OFF = 208


def kernel(name, ins, pat, nvalu, data):
    aregs = 16 if ins == "s32" else 4
    mn = "v_mfma_scale_f32_32x32x64_f8f6f4" if ins == "s32" else "v_mfma_scale_f32_16x16x128_f8f6f4"
    o = [f".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:"]
    o.append("    s_load_dwordx2 s[4:5], s[0:1], 0x0")
    o.append("    v_readfirstlane_b32 s6, v0")
    o.append(f"    v_mov_b32 v{TID}, v0")
    for i in range(64):   # A fragments v128..159 (4 sets), B fragments v160..191 (4 sets)
        if data == "zero":
            o.append(f"    v_mov_b32 v{A0 + i}, 0")
        else:
            o.append(f"    s_mov_b32 s60, {(LIT[i % 8] ^ (i * 0x01000193)) & 0xFFFFFFFF:#x}")
            o.append(f"    v_mul_lo_u32 v{A0 + i}, v{TID}, s60")
            o.append(f"    v_xor_b32 v{A0 + i}, 0x5BD1E995, v{A0 + i}")
            o.append(f"    v_and_b32 v{A0 + i}, 0xF7F7F7F7, v{A0 + i}")
    for r in range(NACC * aregs):
        o.append(f"    v_mov_b32 v{r}, 0")
    o.append("    s_mov_b32 s61, 0x7E7F8081")
    o.append(f"    v_mov_b32 v{SC}, s61")
    o.append(f"    v_mov_b32 v{SC + 1}, s61")
    for j in range(6):
        o.append(f"    v_mov_b32 v{VT + j}, 1")
    o.append("    s_waitcnt lgkmcnt(0)")
    o += ["    s_nop 7"] * 4
    o.append(f"    s_mov_b32 s20, {iters}")
    o.append("    s_memtime s[24:25]")
    o.append("    s_memrealtime s[26:27]")
    o.append("    s_waitcnt lgkmcnt(0)")
    o.append(f"{name}_LOOP:")
    vj = 0
    for m in range(32):
        if pat == "indep":
            acc, a, b = m % NACC, m % 4, (m + 1 + m // 4) % 4
        elif pat == "sameB":
            acc, a, b = m % NACC, m % 4, 0
        elif pat == "dep2":
            p = m // 2
            acc, a, b = p % NACC, (2 * p + m % 2) % 4, (p + m % 2 * 2 + 1) % 4
        elif pat == "chain":
            acc, a, b = 0, m % 4, (m + 1) % 4
        else:
            raise SystemExit("pattern")
        c = f"v[{acc * aregs}:{acc * aregs + aregs - 1}]"
        o.append(f"    {mn} {c}, v[{A0 + 8 * a}:{A0 + 8 * a + 7}], v[{B0 + 8 * b}:{B0 + 8 * b + 7}], {c}, v{SC}, v{SC + 1}")
        for _ in range(nvalu):
            o.append(f"    v_add_u32 v{VT + vj % 5}, v{VT + vj % 5}, v{TID}")
            vj += 1
    o.append("    s_sub_u32 s20, s20, 1")
    o.append("    s_cmp_lg_u32 s20, 0")
    o.append(f"    s_cbranch_scc1 {name}_LOOP")
    o.append("    s_memtime s[28:29]")
    o.append("    s_memrealtime s[30:31]")
    o.append("    s_waitcnt lgkmcnt(0)")
    o += ["    s_nop 7"] * 4
    o += ["    s_sub_u32 s8, s28, s24", "    s_sub_u32 s9, s30, s26", "    s_lshr_b32 s10, s6, 6", "    s_lshl_b32 s11, s2, 2", "    s_add_u32 s10, s10, s11",
          "    s_lshl_b32 s10, s10, 4", "    s_add_u32 s4, s4, s10", "    s_addc_u32 s5, s5, 0", "    v_mov_b32 v0, s8", "    v_mov_b32 v1, s9",
          f"    v_mov_b32 v2, {iters}", f"    v_lshrrev_b32 v3, 6, v{TID}", "    v_mov_b32 v4, 0", "    global_store_dwordx4 v4, v[0:3], s[4:5]", "    s_waitcnt vmcnt(0)", "    s_endpgm"]
    o.append(f".size {name}, .-{name}")
    o.append("\n.p2align 6")
    o.append(f".amdhsa_kernel {name}\n    .amdhsa_group_segment_fixed_size 0\n    .amdhsa_private_segment_fixed_size 0\n    .amdhsa_kernarg_size 8\n    .amdhsa_user_sgpr_count 2\n    .amdhsa_user_sgpr_kernarg_segment_ptr 1\n    .amdhsa_system_sgpr_workgroup_id_x 1\n    .amdhsa_system_vgpr_workitem_id 0\n    .amdhsa_next_free_vgpr {ACCUM_OFF}\n    .amdhsa_next_free_sgpr 64\n    .amdhsa_accum_offset {ACCUM_OFF}\n    .amdhsa_float_denorm_mode_32 3\n    .amdhsa_float_denorm_mode_16_64 3\n    .amdhsa_ieee_mode 0\n.end_amdhsa_kernel\n")
    out.write("\n".join(o) + "\n")
    return name


names = []
for ins in ("s32", "s16"):
    for pat in ("indep", "sameB", "dep2", "chain"):
        for nv in (0, 5):
            names.append(kernel(f"{ins}_{pat}_v{nv}", ins, pat, nv, "rand"))
names.append(kernel("s32_dep2_v0_z", "s32", "dep2", 0, "zero"))
names.append(kernel("s32_indep_v0_z", "s32", "indep", 0, "zero"))
out.write("\n.amdgpu_metadata\n---\namdhsa.kernels:\n")
for n in names:
    out.write(f"  - .name: {n}\n    .symbol: {n}.kd\n    .kernarg_segment_size: 8\n    .kernarg_segment_align: 8\n    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n    .wavefront_size: 64\n    .max_flat_workgroup_size: 256\n    .sgpr_count: 64\n    .vgpr_count: {ACCUM_OFF}\n    .sgpr_spill_count: 0\n    .vgpr_spill_count: 0\n    .args:\n      - .name: \"out\"\n        .offset: 0\n        .size: 8\n        .value_kind: global_buffer\n        .address_space: global\n")
out.write("amdhsa.target: amdgcn-amd-amdhsa--gfx950\namdhsa.version:\n  - 1\n  - 0\n...\n.end_amdgpu_metadata\n")
out.close()
print("kernels:", " ".join(names))