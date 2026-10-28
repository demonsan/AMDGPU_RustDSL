// Derived from the emitter's output for probe_lds (which runs correctly), with
// the body replaced by a plain global store. Used to isolate what makes a
// hand-written file behave differently from generated assembly.
.text
.globl hw_store2
.p2align 8
.type hw_store2,@function
hw_store2:
BB0:
    s_load_dwordx2 s[8:9], s[0:1], 0
    v_mov_b32_e32 v1, v0
    v_mov_b32_e32 v3, 4
    v_mul_lo_u32 v4, v1, v3
    v_mov_b32_e32 v2, s8
    v_mov_b32_e32 v3, s9
    v_add_co_u32 v6, vcc, v2, v4
    v_addc_co_u32 v7, vcc, v3, 0, vcc
    global_store_dword v[6:7], v1, off
    s_endpgm
.size hw_store2, .-hw_store2

.p2align 6
.amdhsa_kernel hw_store2
    .amdhsa_group_segment_fixed_size 256
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_ieee_mode 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: hw_store2
    .symbol: hw_store2.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .max_flat_workgroup_size: 64
    .sgpr_count: 16
    .vgpr_count: 8
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
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
