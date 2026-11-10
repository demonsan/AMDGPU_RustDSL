# EXPECT-FAIL: the combined s_waitcnt (the gfx950 WaitForm::Combined) is not a gfx1250 instruction
# (upstream: llvm/test/MC/AMDGPU/gfx1250_asm_unsupported.s @274ab686)
.text
s_waitcnt vmcnt(0) lgkmcnt(0)
