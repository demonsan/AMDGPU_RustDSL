# VOPD / VOPD3 dual issue
# Source of the forms: LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s, gfx12_asm_*.s) at llvm-project 274ab686 (public).
.text
v_dual_fmac_f32 v255, v4, v2 :: v_dual_add_f32 v6, v1, v3
v_dual_mul_f32 v255, v4, v2 :: v_dual_add_f32 v6, v1, v3
v_dual_mov_b32 v255, v4 :: v_dual_mov_b32 v6, v1
v_dual_add_f32 v255, v4, v2 :: v_dual_max_num_f32 v6, v1, v3
v_dual_fma_f32 v1, v4, v2, v10 :: v_dual_fma_f32 v7, v1, v3, v4
v_dual_add_f64 v[252:253], v[6:7], v[4:5] :: v_dual_add_f32 v8, v1, v3
