// Syntax probe v4: constants are materialized into registers; all ALU ops are reg-reg.
.text
.globl k_probe
.p2align 8
.type k_probe,@function
k_probe:
    s_load_dwordx2 s[2:3], s[0:1], 0x0
    s_waitcnt lgkmcnt(0)
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, v0
    v_lshlrev_b32_e32 v3, 1, v3
    v_add_co_u32 v4, vcc, v3, 0
    v_addc_co_u32 v5, vcc, v2, 0, vcc
    global_load_ushort v6, v[4:5], off offset:128
    global_load_dword v7, v[4:5], off offset:0
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16 v11, v6
    v_exp_f32_e32 v12, v11
    v_mov_b32_e32 v22, 4
    v_mov_b32_e32 v23, 1.0
    v_mul_f32_e32 v13, v12, v12
    v_max_f32_e32 v14, v23, v13
    v_sub_f32_e32 v15, v13, v14
    v_fma_f32 v16, v13, v14, v15
    v_cndmask_b32_e32 v17, v13, v14, vcc
    v_cmp_lt_i32_e32 v6, v22
    v_and_b32_e32 v18, 3, v3
    v_lshrrev_b32_e32 v19, 2, v3
    v_mul_u32_u24 v20, v3, v22
    s_cbranch_vccnz 1f
    s_branch 2f
1:
    s_nop 0
2:
    v_cvt_f16_f32 v21, v17
    global_store_short v[4:5], v21, off offset:64
    s_mov_b32 s6, 0
    s_add_u32 s6, s6, 1
    s_lshl_b32 s7, s6, 2
    s_cmp_lt_i32 s6, s7
    s_cbranch_scc0 3f
    s_mov_b32 s8, 0
3:
    s_endpgm
.size k_probe, .-k_probe

.amdhsa_kernel k_probe
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 24
    .amdhsa_next_free_sgpr 9
    .amdhsa_accum_offset 24
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_ieee_mode 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: out
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: in
        .offset: 8
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 64
    .name: k_probe
    .private_segment_fixed_size: 0
    .sgpr_count: 9
    .sgpr_spill_count: 0
    .symbol: k_probe.kd
    .vgpr_count: 24
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
