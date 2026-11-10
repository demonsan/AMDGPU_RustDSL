# WMMA / SWMMAC (wave32) forms
# Source of the forms: LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s, gfx12_asm_*.s) at llvm-project 274ab686 (public).
.text
v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] matrix_a_reuse
v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19]
v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23]
v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11]
v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23]
v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[0:15], v[8:23], v[16:23]
v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23] clamp
v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47]
v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[24:39], v[40:47] matrix_a_fmt:MATRIX_FMT_FP4
v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[40:47], v1, v2 matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP6 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
v_wmma_ld_scale_paired_b32 v1, v2
v_wmma_ld_scale16_paired_b64 v[2:3], v[4:5]
v_wmma_f32_32x16x128_f4 v[4:19], v[0:15], v[2:9], v[4:19]
v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 index_key:1
v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32
v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v[32:33] index_key:1
