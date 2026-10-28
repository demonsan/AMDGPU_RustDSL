// Assembler probe for the instruction family used by the FMHA reference core
// loop. Every line is intentional; clang reports the exact operand shape when
// a form is wrong.
.text
.globl fmha_isa_probe
.p2align 8
.type fmha_isa_probe,@function
fmha_isa_probe:
    // Wider f16 MFMA: D = 16 dwords.
    v_mfma_f32_32x32x8_f16 v[16:31], v[32:33], v[34:35], v[36:51]
    // FP8 / FP6 / FP4 MFMA (gfx950 shapes).
    v_mfma_f32_16x16x32_f8f6f4 v[4:7], v[8:11], v[12:15], v[16:19]
    v_mfma_f32_32x32x16_f8f6f4 v[16:31], v[32:39], v[40:47], v[48:63]
    v_mfma_f32_32x32x64_f8f6f4 v[16:31], v[32:39], v[40:47], v[48:63]
    // Scaled variant: the 16-byte encoding used by the reference kernel.
    v_mfma_scale_f32_32x32x64_f8f6f4 v[16:31], v[32:39], v[40:47], v[48:63], s0, s1
    // Scaled fused multiply-add-accumulate for the PV product.
    v_smfmac_f32_16x16x128_fp8_fp8 v[4:7], v[8:11], v[12:15], v16
    // Softmax helpers.
    v_max3_f32 v1, v2, v3, v4
    v_cvt_pknorm_u16_f32 v5, v6, v7
    v_cvt_scalef32_pk_fp8_bf16 v8, v9, v10
    // LDS variants used by the reference (transposed byte read included).
    ds_read_b64 v[4:5], v6
    ds_read_b64_tr_b8 v[8:9], v10
    ds_read_b128 v[12:15], v16
    s_waitcnt lgkmcnt(4)
    s_waitcnt lgkmcnt(2)
    s_barrier
    s_endpgm
.size fmha_isa_probe, .-fmha_isa_probe

.p2align 6
.amdhsa_kernel fmha_isa_probe
    .amdhsa_group_segment_fixed_size 1024
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
  - .name: "fmha_isa_probe"
    .symbol: fmha_isa_probe.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 1024
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
