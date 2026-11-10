# global / buffer / scalar memory with temporal hints and scopes; cache control
# Source of the forms: LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s, gfx12_asm_*.s) at llvm-project 274ab686 (public).
.text
global_load_b128 v[2:5], v0, s[0:1] offset:-64
global_store_b128 v0, v[2:5], s[0:1] offset:64
global_load_b32 v0, v[2:3], off th:TH_LOAD_NT scope:SCOPE_SE nv
global_load_b32 v5, v1, s[2:3] offset:32 scale_offset
buffer_load_b128 v[6:9], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
buffer_store_b128 v[2:5], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT scope:SCOPE_SE nv
global_load_tr_b128 v[2:5], v0, s[0:1]
global_inv scope:SCOPE_DEV
global_wb scope:SCOPE_SYS
global_wbinv scope:SCOPE_DEV
