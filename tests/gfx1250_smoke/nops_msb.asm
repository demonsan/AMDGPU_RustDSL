# padding instructions and the high-VGPR window (s_set_vgpr_msb)
# Source of the forms: LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s, gfx12_asm_*.s) at llvm-project 274ab686 (public).
.text
s_nop 0
v_nop_e64
s_set_vgpr_msb 0
s_set_vgpr_msb 0x55
