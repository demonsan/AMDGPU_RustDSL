.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl __NAME__
.p2align 8
.type __NAME__,@function
__NAME__:
    s_load_dwordx2 s[2:3], s[0:1], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32_e32 v4, v0                 ; own value = global thread id
    v_lshrrev_b32_e32 v16, 6, v0         ; wave id 0..7
    v_lshrrev_b32_e32 v12, 2, v16        ; group 0/1 (waves 0-3 / 4-7)
    v_and_b32_e32 v9, 63, v0             ; lane
    v_lshlrev_b32_e32 v9, 2, v9          ; lane*4
    v_and_b32_e32 v13, 3, v16
    v_lshlrev_b32_e32 v13, 8, v13        ; (wave&3)*256
    v_lshlrev_b32_e32 v17, 11, v12       ; group*2048
    v_add_u32_e32 v1, v13, v9
    v_add_u32_e32 v1, v17, v1            ; own data slot
    v_mov_b32_e32 v23, 256
    v_add_u32_e32 v15, v23, v1
    v_mov_b32_e32 v14, 768
    v_sub_u32_e32 v14, v1, v14
    v_and_b32_e32 v13, 3, v16
    v_cmp_eq_u32_e32 vcc, 3, v13
    v_cndmask_b32_e32 v2, v15, v14, vcc   ; partner data slot (next wave in group, wraps)
    v_lshlrev_b32_e32 v3, 2, v12
    v_mov_b32_e32 v17, 0x1000
    v_add_u32_e32 v3, v3, v17            ; group counter address (4096 + group*4)
    v_add_u32_e32 v18, 8, v3             ; group error counter
    v_add_u32_e32 v19, 16, v3            ; group timeout flag
    v_add_u32_e32 v13, 1, v13
    v_and_b32_e32 v13, 3, v13
    v_lshlrev_b32_e32 v13, 6, v13
    v_lshlrev_b32_e32 v17, 8, v12
    v_add_u32_e32 v7, v17, v13
    v_add_u32_e32 v7, v7, v9             ; expected partner value
    v_readfirstlane_b32 s16, v12
    s_cmp_eq_u32 s16, 1
    s_cselect_b32 s15, __ITERS_B__, __ITERS_A__
    ; ---- zero the group error/timeout counters from lane 0 of the group's first wave ----
    s_mov_b64 s[10:11], exec
    s_mov_b64 exec, 1
    v_and_b32_e32 v13, 3, v16
    v_cmp_eq_u32_e32 vcc, 0, v13
    s_cbranch_vccz Linit_skip___NAME__
    v_mov_b32_e32 v10, 0
    ds_write_b32 v18, v10
    ds_write_b32 v19, v10
    s_waitcnt lgkmcnt(0)
Linit_skip___NAME__:
    s_mov_b64 exec, s[10:11]
    s_barrier
    s_memrealtime s[4:5]
    s_mov_b32 s14, 0
    s_mov_b32 s12, 0
Lloop___NAME__:
    s_add_u32 s12, s12, __TARGET_MUL__
    ds_write_b32 v1, v4
    s_waitcnt lgkmcnt(0)
__DELAY_B__
__SYNC__
    ds_read_b32 v6, v2
    s_waitcnt lgkmcnt(0)
    v_cmp_ne_u32_e32 vcc, v6, v7
    v_cndmask_b32_e32 v23, 0, 1, vcc
    ds_add_u32 v23, v18
    s_waitcnt lgkmcnt(0)
    s_add_u32 s14, s14, 1
    s_cmp_lt_u32 s14, s15
    s_cbranch_scc1 Lloop___NAME__
Ldone___NAME__:
    s_memrealtime s[6:7]
    s_sub_u32 s8, s6, s4
    s_subb_u32 s9, s7, s5
    s_mov_b64 s[10:11], exec
    s_mov_b64 exec, 1
    v_mov_b32_e32 v20, s2
    v_mov_b32_e32 v21, s3
    v_lshlrev_b32_e32 v17, 4, v16
    v_add_co_u32 v20, vcc, v20, v17
    v_addc_co_u32 v21, vcc, v21, 0, vcc
    v_mov_b32_e32 v22, s8
    v_mov_b32_e32 v23, s9
    global_store_dword v[20:21], v22, off
    global_store_dword v[20:21], v23, off offset:4
    ds_read_b32 v25, v18
    s_waitcnt lgkmcnt(0)
    global_store_dword v[20:21], v25, off offset:8
    ds_read_b32 v26, v19
    s_waitcnt lgkmcnt(0)
    global_store_dword v[20:21], v26, off offset:12
    s_mov_b64 exec, s[10:11]
    s_endpgm
.size __NAME__, .-__NAME__
.p2align 6
.amdhsa_kernel __NAME__
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 40
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 40
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .name: __NAME__
    .symbol: __NAME__.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .max_flat_workgroup_size: 512
    .sgpr_count: 24
    .vgpr_count: 40
    .args:
      - .name: "o"
        .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...
.end_amdgpu_metadata
