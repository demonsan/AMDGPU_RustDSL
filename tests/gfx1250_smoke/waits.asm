# per-counter waits (Target WaitForm::PerCounter), ALU dependency waits, idle
# Source of the forms: LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s, gfx12_asm_*.s) at llvm-project 274ab686 (public).
.text
s_wait_loadcnt 0x0
s_wait_loadcnt 0x3f
s_wait_storecnt 0x0
s_wait_dscnt 0x3f
s_wait_kmcnt 0x0
s_wait_loadcnt_dscnt 0x0
s_wait_storecnt_dscnt 0x0
s_wait_xcnt 0x0
s_wait_asynccnt 0x0
s_wait_tensorcnt 0x0
s_wait_tensorcnt 0x3
s_wait_alu 0
s_wait_alu depctr_va_vdst(0)
s_wait_alu depctr_va_sdst(0) depctr_vm_vsrc(0)
s_delay_alu instid0(VALU_DEP_1)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
s_wait_idle
