# tensor data mover: descriptor in 2 or 4 SGPR groups; waited with s_wait_tensorcnt
# Source of the forms: LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s, gfx12_asm_*.s) at llvm-project 274ab686 (public).
.text
tensor_load_to_lds s[0:3], s[4:11]
tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_BYPASS scope:SCOPE_SYS
tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19]
tensor_store_from_lds s[0:3], s[4:11]
tensor_store_from_lds s[0:3], s[4:11], null, null th:TH_STORE_NT_HT scope:SCOPE_DEV
s_wait_tensorcnt 0x0
