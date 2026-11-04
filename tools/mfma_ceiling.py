#!/usr/bin/env python3
"""MFMA per-instruction ceiling probe (gfx950), extension of mfma_interval.py: every kernel measures ITS OWN cycles with s_memtime / s_memrealtime.

Each kernel: a loop of 32 independent v_mfma_scale_f32_16x16x128_f8f6f4 per iteration over 16 accumulators (4 A fragments x 4 B fragments, same A for 4 MFMAs, B
varying - the GEMM tile pattern), in-place accumulation. Variants (kernel name = acc<V|A|M>_ops<V|A>_<nosel|sel4>_<zero|rand>):
  acc   V: VGPR accumulators, A: AGPR accumulators, M: 12 VGPR + 4 AGPR (the vendor GEMM mixes 225 VGPR-form / 95 AGPR-form MFMAs)
  ops   A/B fragments in VGPRs or AGPRs (the GEMM keeps them in AGPRs)
  sel   nosel: no op_sel; sel4: the four op_sel / op_sel_hi byte-select forms the GEMM uses, cycled per MFMA
  data  zero: operands 0; rand: pseudo-random fp8 bytes (bit 3 of every byte cleared -> never the e4m3 NaN pattern) in the A/B fragments, scale bytes ~127
Scale operands: only VGPRs assemble (SGPR and inline-constant scale operands are rejected by the gfx950 assembler: 'invalid operand'), so that axis is closed.
Each wave stores (delta s_memtime, delta s_memrealtime, iterations, wave) to out[(workgroup*4 + wave)*16]; the host derives cycles/MFMA and the effective clock
(core-clock ticks per real-time tick at 100 MHz) per wave.
usage: mfma_ceiling.py out.s [iters=4096]
"""
import sys
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
out = open(sys.argv[1], "w")
SEL = ["", " op_sel:[0,1,0] op_sel_hi:[0,0,0]", " op_sel:[0,0,0] op_sel_hi:[0,1,0]", " op_sel:[0,1,0] op_sel_hi:[0,1,0]"]
LIT = [0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09]


def t(f, base, n):
    return f"{f}[{base}:{base + n - 1}]"


def kernel(name, acc, ops, sel, data):
    OPS_V, ACC_V = 100, 0
    SC, TMP, TID = 170, 174, 175
    accum_off = 176
    if ops == "V":
        A0, B0, ops_f = 100, 132, "v"
        accA_base = 0
        opsA_regs = 0
    else:
        A0, B0, ops_f = 0, 32, "a"
        accA_base = 64
        opsA_regs = 64
    accA_regs = {"V": 0, "A": 64, "M": 16}[acc]
    agprs = opsA_regs + accA_regs
    agprs = (agprs + 3) // 4 * 4
    nfv = (accum_off + agprs + 7) // 8 * 8
    o = [f".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:"]
    o.append("    s_load_dwordx2 s[4:5], s[0:1], 0x0")
    o.append("    v_readfirstlane_b32 s6, v0")
    o.append(f"    v_mov_b32 v{TID}, v0")
    # operand data
    for i in range(64):
        dst = (OPS_V + i) if ops == "V" else None
        if data == "zero":
            if ops == "V":
                o.append(f"    v_mov_b32 v{dst}, 0")
            else:
                o.append(f"    v_accvgpr_write_b32 a{i}, 0")
        else:
            o.append(f"    s_mov_b32 s60, {LIT[i % 8] ^ (i * 0x01000193) & 0xFFFFFFFF:#x}")
            tgt = f"v{dst}" if ops == "V" else f"v{TMP}"
            o.append(f"    v_mul_lo_u32 {tgt}, v{TID}, s60")
            o.append(f"    v_xor_b32 {tgt}, 0x5BD1E995, {tgt}")
            o.append(f"    v_and_b32 {tgt}, 0xF7F7F7F7, {tgt}")
            if ops == "A":
                o.append("    s_nop 4")
                o.append(f"    v_accvgpr_write_b32 a{i}, v{TMP}")
    # accumulators start at 0
    for k in range(16):
        isA = acc == "A" or (acc == "M" and k >= 12)
        for r in range(4):
            if isA:
                base = accA_base + (4 * k if acc == "A" else 4 * (k - 12))
                o.append(f"    v_accvgpr_write_b32 a{base + r}, 0")
            else:
                o.append(f"    v_mov_b32 v{ACC_V + 4 * k + r}, 0")
    # scale bytes (e8m0 ~ 127 = 1.0, neighbours to exercise different exponents)
    o.append("    s_mov_b32 s61, 0x7E7F8081")
    o.append(f"    v_mov_b32 v{SC}, s61")
    o.append(f"    v_mov_b32 v{SC + 1}, s61")
    o.append("    s_waitcnt lgkmcnt(0)")
    o += ["    s_nop 7"] * 4
    o.append(f"    s_mov_b32 s20, {iters}")
    o.append("    s_getreg_b32 s12, hwreg(29, 0, 20)")
    o.append("    s_memtime s[24:25]")
    o.append("    s_memrealtime s[26:27]")
    o.append("    s_waitcnt lgkmcnt(0)")
    o.append(f"{name}_LOOP:")
    n = 0
    for rnd in range(2):
        for i in range(4):
            for j in range(4):
                k = 4 * i + j
                isA = acc == "A" or (acc == "M" and k >= 12)
                if isA:
                    base = accA_base + (4 * k if acc == "A" else 4 * (k - 12))
                    c = t("a", base, 4)
                else:
                    c = t("v", ACC_V + 4 * k, 4)
                a = t(ops_f, A0 + 8 * i, 8)
                b = t(ops_f, B0 + 8 * j, 8)
                s = SEL[n % 4] if sel == "sel4" else ""
                o.append(f"    v_mfma_scale_f32_16x16x128_f8f6f4 {c}, {a}, {b}, {c}, v{SC}, v{SC + 1}{s}")
                n += 1
    o.append("    s_sub_u32 s20, s20, 1")
    o.append("    s_cmp_lg_u32 s20, 0")
    o.append(f"    s_cbranch_scc1 {name}_LOOP")
    o.append("    s_getreg_b32 s13, hwreg(29, 0, 20)")
    o.append("    s_memtime s[28:29]")
    o.append("    s_memrealtime s[30:31]")
    o.append("    s_waitcnt lgkmcnt(0)")
    o += ["    s_nop 7"] * 4
    o.append("    s_sub_u32 s8, s28, s24")
    o.append("    s_sub_u32 s9, s30, s26")
    o.append("    s_lshr_b32 s10, s6, 6")
    o.append("    s_lshl_b32 s11, s2, 2")
    o.append("    s_add_u32 s10, s10, s11")
    o.append("    s_lshl_b32 s10, s10, 4")
    o.append("    s_add_u32 s4, s4, s10")
    o.append("    s_addc_u32 s5, s5, 0")
    o.append("    v_mov_b32 v0, s8")
    o.append("    v_mov_b32 v1, s9")
    o.append(f"    v_mov_b32 v2, {iters}")
    o.append("    s_sub_u32 s14, s13, s12")
    o.append("    v_mov_b32 v3, s14")
    o.append("    v_mov_b32 v4, 0")
    o.append("    global_store_dwordx4 v4, v[0:3], s[4:5]")
    o.append("    s_waitcnt vmcnt(0)")
    o.append("    s_endpgm")
    o.append(f".size {name}, .-{name}")
    o.append("\n.p2align 6")
    o.append(f".amdhsa_kernel {name}\n    .amdhsa_group_segment_fixed_size 0\n    .amdhsa_private_segment_fixed_size 0\n    .amdhsa_kernarg_size 8\n    .amdhsa_user_sgpr_count 2\n    .amdhsa_user_sgpr_kernarg_segment_ptr 1\n    .amdhsa_system_sgpr_workgroup_id_x 1\n    .amdhsa_system_vgpr_workitem_id 0\n    .amdhsa_next_free_vgpr {nfv}\n    .amdhsa_next_free_sgpr 64\n    .amdhsa_accum_offset {accum_off}\n    .amdhsa_float_denorm_mode_32 3\n    .amdhsa_float_denorm_mode_16_64 3\n    .amdhsa_ieee_mode 0\n.end_amdhsa_kernel\n")
    out.write("\n".join(o) + "\n")
    return name, nfv, agprs


meta = []
for acc in "VAM":
    for ops in "VA":
        for sel in ("nosel", "sel4"):
            for data in ("zero", "rand"):
                meta.append(kernel(f"acc{acc}_ops{ops}_{sel}_{data}", acc, ops, sel, data))
out.write("\n.amdgpu_metadata\n---\namdhsa.kernels:\n")
for name, nfv, agprs in meta:
    out.write(f"  - .name: {name}\n    .symbol: {name}.kd\n    .kernarg_segment_size: 8\n    .kernarg_segment_align: 8\n    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n    .wavefront_size: 64\n    .max_flat_workgroup_size: 256\n    .sgpr_count: 64\n    .vgpr_count: {nfv}\n")
    if agprs:
        out.write(f"    .agpr_count: {agprs}\n")
    out.write("    .sgpr_spill_count: 0\n    .vgpr_spill_count: 0\n    .args:\n      - .name: \"out\"\n        .offset: 0\n        .size: 8\n        .value_kind: global_buffer\n        .address_space: global\n")
out.write("amdhsa.target: amdgcn-amd-amdhsa--gfx950\namdhsa.version:\n  - 1\n  - 0\n...\n.end_amdgpu_metadata\n")
out.close()
print("kernels:", " ".join(m[0] for m in meta))