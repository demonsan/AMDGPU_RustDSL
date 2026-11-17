# gfx1250 software hazards (G2 of docs/gfx1250-plan.md)

What the public upstream LLVM AMDGPU backend inserts for correctness after register allocation for gfx1250 (llvm-project `main` @
`0855aa8e`, `GCNHazardRecognizer.cpp` / `SIInsertWaitcnts.cpp` / `AMDGPULowerVGPREncoding.cpp`; feature set `FeatureISAVersion12_50`,
GCNFeatures.td 2504-2597 / 2636-2674). gfx1250 has `FeatureNoDataDepHazard`: no `s_nop` wait states at all (the hazard recognizer's wait-state
checks, the MFMA/XDL tables included, return early); `AMDGPUWaitSGPRHazards` does nothing for gfx1250 (no `s_wait_alu` for VALU -> SGPR).
Distances count intervening VALU instructions (`v_nop` included unless noted; SALU / memory / `s_nop` do not count), along every path.

| rule | trigger | fix | own tail |
|---|---|---|---|
| TRANS co-execution | a TRANS VALU, then the next VALU (0 VALUs between) is a non-TRANS that reads its result or overwrites one of its sources | 1 `v_nop` | `target/gfx1250/hazards.rs` |
| WMMA -> WMMA | a WMMA's D read as A / B (C may overlap D) by the next WMMA with 0 VALUs between | 1 `v_nop` | covered by the co-execution rule |
| WMMA co-execution | an XDL WMMA, then a WMMA reading its D as A / B, or any VALU reading D or writing D / A / B / C | `v_nop` x (window - VALUs between); windows (next WMMA / next VALU) by category: 16x16x64 FP8 (latency 4): 2 / 1; F16 / BF16, 16x16x128 FP8, 32x16x128 F4 (latency 8): 5 / 4; IU8 / IU4 (16): 9 / 8; SWMMAC 8: 3 / 2, 16: 5 / 4 | `hazards.rs` |
| v_cmpx -> permlane | a v_cmpx (EXEC write), then a permlane with no non-`v_nop` VALU between | `v_mov_b32 vS, vS` (src0) before the permlane | not emitted (no v_cmpx / permlane in the gfx1250 tail yet): compile error if it would be |
| v_perm_pk16_* | not followed immediately by a safe VALU | 1 `v_nop` after | not emitted |
| s_getreg of STATUS / STATE_PRIV / EXCP_FLAG_* | always | `s_wait_alu 0` before | not emitted (exact replays: the author's) |
| ds_atomic_async_barrier_arrive_b64 | always | `s_wait_alu vm_vsrc(0)` before and after | G3 (with the async barrier) |
| s102 / s103 write -> flat-scratch-base read | < 10 SGPR writes between | `s_wait_alu sa_sdst(0) va_sdst(0)` | not emitted (the allocator does not use s102 / s103) |
| s_setreg MODE | always | 2 `v_nop` before; an `s_set_vgpr_msb` right after it is dropped (`s_nop 0` between) | not emitted outside the prologue |
| kernel entry | every kernel | `s_mov_b64 s[64:65], 0; v_nop; global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE` (the first VMEM not in a clause) | the emitter's prologue (s[6:7]) |
| TDM drain | a tensor op reachable from an earlier one without `s_wait_tensorcnt <= 10` | `s_wait_tensorcnt 10` before it | `hazards.rs` |

VOPD pairing constraints are applied when LLVM forms pairs (no post-fix); the own tail emits no VOPD. VGPR MSB (`s_set_vgpr_msb`): reset to 0
at every block end / before branches, folded where possible; see G2 VGPR windowing.
