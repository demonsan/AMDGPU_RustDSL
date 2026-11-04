#!/usr/bin/env python3
"""64-MFMA tile-block probe (gfx950): the per-tile synchronisation cost, measured in cycles per MFMA with s_memtime inside the kernel.

One loop iteration = one 'tile' of 64 v_mfma_scale_f32_16x16x128_f8f6f4 over 64 accumulators (4 rounds x 4 A fragments x 4 B fragments, same A for 4 MFMAs), with the
GEMM v4 accumulator split (40 VGPR-form + 24 AGPR-form, in place), the four op_sel/op_sel_hi forms cycled, random fp8 operands.
  ops  A: srcA and srcB in AGPRs (our GEMM)   V: both in VGPRs   M: srcA AGPR, srcB VGPR (the vendor has a few VGPR-srcA/srcB forms)
  sync none        : nothing
       bar_end     : s_barrier after the 64th MFMA (all 4 waves of the workgroup)
       bar_rd_end  : ds_read_b64 at the block start, s_waitcnt lgkmcnt(0) + s_barrier after the 64th MFMA
       bar_rd_pre  : the vendor order - ds_read_b64 after the 64th MFMA, then in the next block `s_waitcnt lgkmcnt(0); s_barrier` after 4 MFMAs (pre_sync)
       rd_nobar    : ds_read_b64 at the block start, s_waitcnt lgkmcnt(0) after the 64th MFMA, no barrier
Kernel name = blk_<ops>_<sync>. usage: mfma_block.py out.s [iters=2048]   (host: tests/gpu/mfma_ceiling_host.cpp with MPI=64)
"""
import sys
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
out = open(sys.argv[1], "w")
SEL = ["", " op_sel:[0,1,0] op_sel_hi:[0,0,0]", " op_sel:[0,0,0] op_sel_hi:[0,1,0]", " op_sel:[0,1,0] op_sel_hi:[0,1,0]"]
LIT = [0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09]


def t(f, base, n):
    return f"{f}[{base}:{base + n - 1}]"


def kernel(name, ops, sync):
    ACC_V, SC, TMP, TID, ADDR = 0, 224, 226, 227, 228
    accum_off = 232
    # AGPR layout: A frags a[0:31], B frags a[32:63] when AGPR, accumulators a[64:..] (24 accs x 4 = 96)
    aA = 0
    aB = 32
    accA_base = 64
    agprs = 64 + 96
    # VGPR layout: acc V v[0:159], VGPR copies of the operand fragments after the scales
    vA, vB = 160, 192
    nfv = accum_off + agprs
    nfv = (nfv + 7) // 8 * 8
    o = [f".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:"]
    o.append("    s_load_dwordx2 s[4:5], s[0:1], 0x0")
    o.append("    v_readfirstlane_b32 s6, v0")
    o.append(f"    v_mov_b32 v{TID}, v0")
    o.append(f"    v_and_b32 v{ADDR}, 63, v{TID}")
    o.append(f"    v_lshlrev_b32 v{ADDR}, 3, v{ADDR}")
    # operand data: random fp8 bytes; A frags in AGPR (ops A, M) or VGPR (ops V); B frags in AGPR (A) or VGPR (V, M)
    def fill(i, to_agpr, vbase):
        o.append(f"    s_mov_b32 s60, {(LIT[i % 8] ^ (i * 0x01000193)) & 0xFFFFFFFF:#x}")
        tgt = f"v{TMP}" if to_agpr else f"v{vbase + i}"
        o.append(f"    v_mul_lo_u32 {tgt}, v{TID}, s60")
        o.append(f"    v_xor_b32 {tgt}, 0x5BD1E995, {tgt}")
        o.append(f"    v_and_b32 {tgt}, 0xF7F7F7F7, {tgt}")
        if to_agpr:
            o.append("    s_nop 4")
    for i in range(32):
        fill(i, ops in ("A", "M"), vA)
        if ops in ("A", "M"):
            o.append(f"    v_accvgpr_write_b32 a{aA + i}, v{TMP}")
    for i in range(32):
        fill(32 + i, ops == "A", vB)
        if ops == "A":
            o.append(f"    v_accvgpr_write_b32 a{aB + i}, v{TMP}")
    for k in range(64):
        for r in range(4):
            if k < 40:
                o.append(f"    v_mov_b32 v{ACC_V + 4 * k + r}, 0")
            else:
                o.append(f"    v_accvgpr_write_b32 a{accA_base + 4 * (k - 40) + r}, 0")
    o.append("    s_mov_b32 s61, 0x7E7F8081")
    o.append(f"    v_mov_b32 v{SC}, s61")
    o.append(f"    v_mov_b32 v{SC + 1}, s61")
    o.append("    s_waitcnt lgkmcnt(0)")
    o += ["    s_nop 7"] * 4
    o.append(f"    s_mov_b32 s20, {iters}")
    o.append("    s_barrier")
    o.append("    s_memtime s[24:25]")
    o.append("    s_memrealtime s[26:27]")
    o.append("    s_waitcnt lgkmcnt(0)")
    o.append(f"{name}_LOOP:")
    if sync in ("bar_rd_end", "rd_nobar"):
        o.append(f"    ds_read_b64 v[{TMP}:{TMP + 1}], v{ADDR}")
    n = 0
    for rnd in range(4):
        for i in range(4):
            for j in range(4):
                k = 16 * rnd + 4 * i + j
                c = t("v", ACC_V + 4 * k, 4) if k < 40 else t("a", accA_base + 4 * (k - 40), 4)
                a = t("a", aA + 8 * i, 8) if ops in ("A", "M") else t("v", vA + 8 * i, 8)
                b = t("a", aB + 8 * j, 8) if ops == "A" else t("v", vB + 8 * j, 8)
                o.append(f"    v_mfma_scale_f32_16x16x128_f8f6f4 {c}, {a}, {b}, {c}, v{SC}, v{SC + 1}{SEL[n % 4]}")
                n += 1
                if sync == "bar_rd_pre" and n == 4:
                    o.append("    s_waitcnt lgkmcnt(0)")
                    o.append("    s_barrier")
    if sync == "bar_end":
        o.append("    s_barrier")
    elif sync == "bar_rd_end":
        o.append("    s_waitcnt lgkmcnt(0)")
        o.append("    s_barrier")
    elif sync == "bar_rd_pre":
        o.append(f"    ds_read_b64 v[{TMP}:{TMP + 1}], v{ADDR}")
    elif sync == "rd_nobar":
        o.append("    s_waitcnt lgkmcnt(0)")
    o.append("    s_sub_u32 s20, s20, 1")
    o.append("    s_cmp_lg_u32 s20, 0")
    o.append(f"    s_cbranch_scc1 {name}_LOOP")
    o.append("    s_waitcnt lgkmcnt(0)")
    o.append("    s_memtime s[28:29]")
    o.append("    s_memrealtime s[30:31]")
    o.append("    s_waitcnt lgkmcnt(0)")
    o += ["    s_nop 7"] * 4
    o += ["    s_sub_u32 s8, s28, s24", "    s_sub_u32 s9, s30, s26", "    s_lshr_b32 s10, s6, 6", "    s_lshl_b32 s11, s2, 2", "    s_add_u32 s10, s10, s11",
          "    s_lshl_b32 s10, s10, 4", "    s_add_u32 s4, s4, s10", "    s_addc_u32 s5, s5, 0", "    v_mov_b32 v0, s8", "    v_mov_b32 v1, s9",
          f"    v_mov_b32 v2, {iters}", f"    v_lshrrev_b32 v3, 6, v{TID}", "    v_mov_b32 v4, 0", "    global_store_dwordx4 v4, v[0:3], s[4:5]", "    s_waitcnt vmcnt(0)", "    s_endpgm"]
    o.append(f".size {name}, .-{name}")
    o.append("\n.p2align 6")
    o.append(f".amdhsa_kernel {name}\n    .amdhsa_group_segment_fixed_size 4096\n    .amdhsa_private_segment_fixed_size 0\n    .amdhsa_kernarg_size 8\n    .amdhsa_user_sgpr_count 2\n    .amdhsa_user_sgpr_kernarg_segment_ptr 1\n    .amdhsa_system_sgpr_workgroup_id_x 1\n    .amdhsa_system_vgpr_workitem_id 0\n    .amdhsa_next_free_vgpr {nfv}\n    .amdhsa_next_free_sgpr 64\n    .amdhsa_accum_offset {accum_off}\n    .amdhsa_float_denorm_mode_32 3\n    .amdhsa_float_denorm_mode_16_64 3\n    .amdhsa_ieee_mode 0\n.end_amdhsa_kernel\n")
    out.write("\n".join(o) + "\n")
    return name, nfv, agprs


meta = [kernel(f"blk_{ops}_{sync}", ops, sync) for ops in "AVM" for sync in ("none", "bar_end", "bar_rd_end", "bar_rd_pre", "rd_nobar")]
out.write("\n.amdgpu_metadata\n---\namdhsa.kernels:\n")
for name, nfv, agprs in meta:
    out.write(f"  - .name: {name}\n    .symbol: {name}.kd\n    .kernarg_segment_size: 8\n    .kernarg_segment_align: 8\n    .group_segment_fixed_size: 4096\n    .private_segment_fixed_size: 0\n    .wavefront_size: 64\n    .max_flat_workgroup_size: 256\n    .sgpr_count: 64\n    .vgpr_count: {nfv}\n    .agpr_count: {agprs}\n    .sgpr_spill_count: 0\n    .vgpr_spill_count: 0\n    .args:\n      - .name: \"out\"\n        .offset: 0\n        .size: 8\n        .value_kind: global_buffer\n        .address_space: global\n")
out.write("amdhsa.target: amdgcn-amd-amdhsa--gfx950\namdhsa.version:\n  - 1\n  - 0\n...\n.end_amdgpu_metadata\n")
out.close()
print("kernels:", " ".join(m[0] for m in meta))