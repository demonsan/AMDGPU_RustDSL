#!/usr/bin/env python3
"""MFMA issue-interval micro-probe (gfx950): 8 hand-written kernels, each a loop of 32 independent v_mfma_scale_f32_16x16x128_f8f6f4 per iteration over 16 accumulators
(4 A fragments x 4 B fragments: the same A fragment for 4 consecutive MFMAs, B varying, like the GEMM tile), accumulating
  inplace : vdst == srcC (what the amdc GEMM does), or
  rot     : vdst != srcC, the accumulators rotate between two register pools every round (what the vendor kernel does).
Accumulator file (V|A) x operand file (V|A): kernels acc{V,A}_ops{V,A}_{inplace,rot}.
usage: mfma_interval.py out.s [iters=4096]    (the host: tests/gpu/mfma_interval_host.cpp, same ITERS)"""
import sys
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
out = open(sys.argv[1], "w")
def tup(f, base, n): return f"{f}[{base}:{base+n-1}]"
def kernel(name, acc_f, ops_f, rot):
    # register layout
    if acc_f == "v" and ops_f == "v":
        acc1, acc2 = 0, 64; A0, B0 = 128, 160; ssc = 192; accum_off = 196; agprs = 0
    elif acc_f == "v" and ops_f == "a":
        acc1, acc2 = 0, 64; A0, B0 = 0, 32; ssc = 128; accum_off = 132; agprs = 64
    elif acc_f == "a" and ops_f == "v":
        acc1, acc2 = 0, 64; A0, B0 = 0, 32; ssc = 64; accum_off = 68; agprs = 128
    else:
        acc1, acc2 = 0, 64; A0, B0 = 128, 160; ssc = 0; accum_off = 4; agprs = 192
    total = accum_off + agprs
    nfv = (total + 7) // 8 * 8
    o = []
    o.append(f".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:")
    o.append(f"    s_mov_b32 s20, {iters}")
    o.append(f"    v_mov_b32 v{ssc}, 127")
    o.append(f"    v_mov_b32 v{ssc+1}, 127")
    o.append(f"{name}_LOOP:")
    for rnd in range(2):
        for i in range(4):
            for j in range(4):
                k = 4 * i + j
                if rot:
                    src, dst = (acc1, acc2) if rnd == 0 else (acc2, acc1)
                else:
                    src, dst = acc1, acc1
                d = tup(acc_f, dst + 4 * k, 4); c = tup(acc_f, src + 4 * k, 4)
                a = tup(ops_f, A0 + 8 * i, 8); b = tup(ops_f, B0 + 8 * j, 8)
                o.append(f"    v_mfma_scale_f32_16x16x128_f8f6f4 {d}, {a}, {b}, {c}, v{ssc}, v{ssc+1}")
    o.append("    s_sub_u32 s20, s20, 1")
    o.append("    s_cmp_lg_u32 s20, 0")
    o.append(f"    s_cbranch_scc1 {name}_LOOP")
    o.append("    s_endpgm")
    o.append(f".size {name}, .-{name}")
    o.append("\n.p2align 6")
    o.append(f".amdhsa_kernel {name}\n    .amdhsa_group_segment_fixed_size 0\n    .amdhsa_private_segment_fixed_size 0\n    .amdhsa_kernarg_size 8\n    .amdhsa_user_sgpr_count 2\n    .amdhsa_user_sgpr_kernarg_segment_ptr 1\n    .amdhsa_system_sgpr_workgroup_id_x 1\n    .amdhsa_system_vgpr_workitem_id 0\n    .amdhsa_next_free_vgpr {nfv}\n    .amdhsa_next_free_sgpr 32\n    .amdhsa_accum_offset {accum_off}\n    .amdhsa_float_denorm_mode_32 3\n    .amdhsa_float_denorm_mode_16_64 3\n    .amdhsa_ieee_mode 0\n.end_amdhsa_kernel\n")
    out.write("\n".join(o) + "\n")
    return (name, nfv, agprs)
meta = []
for acc_f in "va":
    for ops_f in "va":
        for rot in (False, True):
            meta.append(kernel(f"acc{acc_f.upper()}_ops{ops_f.upper()}_{'rot' if rot else 'inplace'}", acc_f, ops_f, rot))
out.write("\n.amdgpu_metadata\n---\namdhsa.kernels:\n")
for name, nfv, agprs in meta:
    out.write(f"  - .name: {name}\n    .symbol: {name}.kd\n    .kernarg_segment_size: 8\n    .kernarg_segment_align: 8\n    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n    .wavefront_size: 64\n    .max_flat_workgroup_size: 256\n    .sgpr_count: 32\n    .vgpr_count: {nfv}\n")
    if agprs: out.write(f"    .agpr_count: {agprs}\n")
    out.write("    .sgpr_spill_count: 0\n    .vgpr_spill_count: 0\n    .args:\n      - .name: \"out\"\n        .offset: 0\n        .size: 8\n        .value_kind: global_buffer\n        .address_space: global\n")
out.write("amdhsa.target: amdgcn-amd-amdhsa--gfx950\namdhsa.version:\n  - 1\n  - 0\n...\n.end_amdgpu_metadata\n")
out.close()
print("kernels:", " ".join(m[0] for m in meta))