// Hand-written control kernel for the multi-wave LDS question.
//
// Same operation as the generated `probe_waves2`: each lane writes its id into
// its own LDS slot, the workgroup synchronises, the lane reads the slot back
// and stores it to global memory. No DSL, no generated code: if this fails on
// hardware the problem is not in the compiler's instruction selection.
.text
.globl hw_roles
.p2align 8
.type hw_roles,@function
hw_roles:
    s_load_dwordx2 s[8:9], s[0:1], 0
    v_mov_b32_e32 v4, v0
    v_lshlrev_b32_e32 v4, 2, v4          // off = tid * 4
    ds_write_b32 v4, v0                  // LDS[off] = tid
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_waitcnt lgkmcnt(0)
    ds_read_b32 v5, v4                   // v5 = LDS[off]
    s_waitcnt lgkmcnt(0)
    v_mov_b32_e32 v2, s8
    v_mov_b32_e32 v3, s9
    v_add_co_u32 v2, vcc, v2, v4
    v_addc_co_u32 v3, vcc, v3, 0, vcc
    global_store_dword v[2:3], v5, off   // o[tid] = v5
    s_endpgm
.size hw_roles, .-hw_roles

.p2align 6
.amdhsa_kernel hw_roles
    .amdhsa_group_segment_fixed_size 512
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: "hw_roles"
    .symbol: hw_roles.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 512
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .max_flat_workgroup_size: 128
    .sgpr_count: 16
    .vgpr_count: 8
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
