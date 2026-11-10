# async loads / stores between global memory and LDS, cluster loads, async barrier arrive
# Source of the forms: LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s, gfx12_asm_*.s) at llvm-project 274ab686 (public).
.text
global_load_async_to_lds_b32 v1, v2, s[2:3] offset:64
global_load_async_to_lds_b64 v1, v2, s[2:3] th:TH_LOAD_NT_HT scope:SCOPE_DEV
global_load_async_to_lds_b128 v1, v[2:3], off offset:-64
global_load_async_to_lds_b128 v1, v2, s[2:3]
global_store_async_from_lds_b128 v2, v1, s[2:3] offset:64
cluster_load_async_to_lds_b128 v1, v2, s[2:3] th:TH_LOAD_NT_HT scope:SCOPE_DEV
cluster_load_b128 v[0:3], v4, s[0:1] th:TH_LOAD_NT_HT scope:SCOPE_DEV
ds_atomic_async_barrier_arrive_b64 v5
s_wait_asynccnt 0x0
