# probe: accepted by the rocm-10.1 clang (AMD LLVM d6f6cb69) although upstream LLVM rejects both for gfx1250
# (llvm/test/MC/AMDGPU/gfx1250-strict_err.s and gfx1250_asm_unsupported.s @274ab686). The description follows upstream.
.text
v_swmmac_f32_16x16x32_f16 v[10:17], v[0:3], v[4:11], v18
v_dual_dot2acc_f32_f16 v255, v4, v2 :: v_dual_add_f32 v6, v1, v3
