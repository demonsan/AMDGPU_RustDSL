// Assembler probe for the LDS instructions the FMHA loop needs.
.text
.globl lds_probe
.p2align 8
.type lds_probe,@function
lds_probe:
    ds_write_b32 v1, v2
    ds_write_b32 v3, v4 offset:16
    ds_read_b32 v5, v6
    ds_read_b32 v7, v8 offset:32
    ds_write_b128 v9, v[10:13]
    ds_read_b128 v[14:17], v18
    s_waitcnt lgkmcnt(0)
    s_waitcnt lgkmcnt(4)
    s_barrier
    s_endpgm
.size lds_probe, .-lds_probe

.p2align 6
.amdhsa_kernel lds_probe
    .amdhsa_group_segment_fixed_size 1024
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 32
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: "lds_probe"
    .symbol: lds_probe.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .max_flat_workgroup_size: 64
    .sgpr_count: 16
    .vgpr_count: 32
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
