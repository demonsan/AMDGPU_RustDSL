// Assembler probe for MFMA operand shapes on gfx950.
// Each variant is commented out except the one under test; the driver script
// rewrites this file to try them one at a time.
.text
.globl mfma_probe
.p2align 8
.type mfma_probe,@function
mfma_probe:
    // 16x16x16 f16: D = 4 dwords, A = 2, B = 2, C = 4.
    v_mfma_f32_16x16x16_f16 v[4:7], v[8:9], v[10:11], v[12:15]
    // 32x32x8 f16: D = 16 dwords, A = 2, B = 2, C = 16.
    v_mfma_f32_32x32x8_f16 v[16:31], v[32:33], v[34:35], v[36:51]
    s_endpgm
.size mfma_probe, .-mfma_probe

.amdhsa_kernel mfma_probe
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 64
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: "mfma_probe"
    .symbol: mfma_probe.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .max_flat_workgroup_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
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
