# barriers: work-group (-1), cluster user (-3), named (m0); state, wakeup
# Source of the forms: LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s, gfx12_asm_*.s) at llvm-project 274ab686 (public).
.text
s_barrier_signal -1
s_barrier_wait -1
s_barrier_signal -3
s_barrier_wait -3
s_barrier_signal m0
s_barrier_signal_isfirst -1
s_get_barrier_state s3, -3
s_wakeup_barrier m0
s_barrier_leave
