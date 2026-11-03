# AGPR (accumulation VGPR) support

gfx950 has one 512-register-per-lane file per wave split into **arch VGPRs** (`v0..accum_offset-1`, at most 256) and **AGPRs** (`a0..`, at most 256);
`.amdhsa_next_free_vgpr` = `accum_offset + #AGPR` rounded to 8, `.amdhsa_accum_offset` = arch VGPRs rounded to 4, metadata `.agpr_count`.
The aiter 4-wave GEMM (244 arch + 212 AGPR, one wave per SIMD) needs it; this page is the compiler's AGPR contract. Evidence grades:
**A** = assembler-verified (clang of ROCm 10.1.0, `tests/agpr.rs`), **H** = hardware-verified (GPU 6, `tools/agpr_probe.sh`), **L** = LLVM-tail oracle (`tests/hazard_oracle.rs`), **I** = ISA table.

## Legality rules (A)
* AGPR operands are accepted by `ds_read_b32/b64/b128`, `ds_read_b64_tr_b8`, `ds_write_*` (data), `global_load/store_*` (data), `buffer_load/store_*` (data), `v_accvgpr_*`, and every MFMA operand.
* **MFMA: `vdst` and `srcC` must be in the same file** (the `acc_cd` bit); `srcA`/`srcB` may be in either file independently; scale operands are VGPR/SGPR only.
* VALU (including `v_mov`, converts, `v_cmp`) cannot read an AGPR. `v_accvgpr_write_b32` takes VGPR/SGPR/inline constant (VOP3: no literal); `v_accvgpr_read_b32 v, a`; `v_accvgpr_mov_b32 a, a`.
* AGPR tuples start at **even** indices (`a[1:4]`, `a[1:2]`, `a[1:8]` rejected, `a[2:5]` accepted); indices 0..255.

## Compiler representation
* `RClass::A1/A2/A4/A8/A16` parallel to V*; `is_vector()` is the arch VGPR file only, `is_agpr()`, `is_scalar()`. `to_agpr()/to_vgpr()`.
* `Alloc::phys` holds `AGPR_TAG (0x4000_0000) + index` for AGPR vregs, so ranges of the two files never overlap numerically.
* New MOps `VAccvgprRead/Write/Mov`. MFMA ops are unchanged: the file of `dst` selects the form (`v_mfma ... a[..], ..., a[..]` = acc_cd).
* `regalloc.rs`: AGPR values are allocated first-fit over per-register occupancy intervals in their own file; exact pins (`MFunction::agpr_pin`) first, misaligned or
  overlapping pins are errors; `unified_budget(max_v, agprs) -> (accum_offset, next_free_vgpr, waves/SIMD)`; arch + AGPR > 512 is "out of AGPRs".
* `agpr.rs` (runs right after lowering): moves the requested vregs (and their alias views) to the AGPR file and repairs instructions that cannot touch it: `v_mov/VWriteSub/VReadSub`
  become `v_accvgpr_*`; other VALU readers/writers go through a VGPR temp with `v_accvgpr_read/write`; an MFMA with srcC in the other file gets a copy; read-modify-write converts
  to an AGPR are rejected. `mir::verify` and `gpu_verify` (GV-REG-ALIGN/LIMIT/CLASS/MFMA-FILE, unified budget) enforce the rules on the final MIR.

## DSL
`b.pin_agpr(var)`, `b.pin_agpr_at(var, base)` (loop variables), `b.agpr(value)`, `b.agpr_at(value, base)` (loads, MFMA results, fragments). In-place MFMA accumulation on a pinned
variable is the acc_cd form. MFMA fragment operands may be loop variables (`Opnd::Var`).
**In-place reload**: `b.set_var(av, b.load_lds_b128x2(lo, hi))` (and `load_lds_b128`) is coalesced into the variable when the load is the value's only use and nothing reads the variable after the
first `ds_read` was issued: the `ds_read_b128`s write the variable's own registers (no `v_accvgpr_mov`, no second fragment). Programs must put the reload after the variable's last read.

## Hazards (own tail)
Evidence: CDNA4 ISA table 38 (does not distinguish VGPR/AGPR) + LLVM oracle (L) + hardware probes (H).
* MFMA result (A or V) read by `v_accvgpr_read`, a store, or an MFMA: result latency (20/21 slots scaled, 12 others) - same model as VGPR; unit-tested. In-place same-opcode accumulation: 0.
* `v_accvgpr_read/write/mov` are VALU writes: `VALU write -> MFMA read` = 2 wait states (I).
* `v_accvgpr_read` -> `v_add_co/v_addc` (pointer math): 2 wait states (L, the only AGPR row where LLVM needed more than we padded).
* Post-RA: SrcC write-after-read, MFMA-destination write-after-write and wide-store data hazards now cover both files (tagged ranges).
* A `ds_read` into the registers a running MFMA reads as srcA/srcB (the vendor's in-place reload) needs no padding: the LLVM tail inserts none and the hardware probe `probe_agpr_reload` is bit-exact (H).
* Known generic gaps of the own tail (not AGPR-specific, shown by the oracle matrix): opsel/SDWA-style producers (`v_cvt_pk_fp8 ... op_sel`, `v_pk_sub_u16`) -> next VALU, 1 wait state; this now also
  applies to `v_accvgpr_write` as the consumer.

## LLVM tail
`Op@a` rows in `llvm20.tbl`/`llvm24.tbl` (acc_cd form `V_MFMA_*_e64`, chosen when the destination is an AGPR; the VGPR form is `*_vgprcd_e64`), rows for the three `v_accvgpr_*` ops, `$agprN` names,
AGPR physical slots in the exporter's liveness, scaled-MFMA byte selects (`op_sel` immediates = 4 x byte, llc dump). `tests/llvm_tail.rs` (opcode probes through `llc -verify-machineinstrs`) and the
hazard oracle include the new rows.

## Tests / hardware
`crates/amdgpu-compiler/tests/agpr.rs` (11 tests), `register_accounting.rs` (AGPR index space), `tools/agpr_probe.sh`: probes `probe_agpr_{ref,acc,full,pinned}` (32x32x64 scaled MFMA, accumulator and
fragments in AGPRs, exact pin a[64:79]), `probe_agpr_m16_{ref,mix}` (four 16x16x128 accumulators, two acc_cd + two vgpr form, fragments `ds_read`-ed into `a[0:15]`), `probe_agpr_reload{,_ref}` (in-place
reload of loop-carried fragments): outputs bit-identical to the all-VGPR versions on GPU 6 for trip counts 1,2,3,7,8,9. All 91 pre-existing kernels emit byte-identical assembly (`tools/md5_kernels.sh`).

## Not done
Unknown/ungraded: the exact minimum wait states for `v_accvgpr_write` -> MFMA srcC and MFMA -> `v_accvgpr_read` on the hardware (modelled as VGPR; ISA table + oracle, hardware probes only show correctness
of the generated schedules). `b64` LDS-transpose reloads (`ds_read_b64_tr_b8`) are not coalesced into variables yet. Scheduler register-pressure accounting does not model AGPRs (RA checks the budget).
## Wait placement: LDS-DMA operands (silent-miscompile postmortem)
`sched/waits.rs` (scoreboard of `s_waitcnt`) treated every `Fence`-class instruction as "ordering only" and returned before checking its register operands. The LDS-DMA loads
(`buffer_load ... lds`, `global_load_lds`) are fences for the list scheduler *and* read registers (voffset / soffset), so a `ds_read` result feeding a DMA address was consumed without
`s_waitcnt lgkmcnt` (`insert_waits` added none, `verify_waits` did not look). It surfaced only when the register-pressure ladder (`lower_and_allocate`) reached the **LDS spill** (`ldsspillN`:
spilled loop-invariant addresses are reloaded by a `ds_read_b32` right before the DMA): GEMM v4 with 16 of 64 accumulators in AGPRs (VGPR-bound) returned wrong values for 75% of the outputs.
Fix: fence-class instructions that read registers go through the normal operand check; `verify_waits` checks them too; the unscheduled path now runs `verify_waits` as well (it used to trust the
lowering-time waits). Regression: `tests/wait_operands.rs` (MIR) and the GPU probe `probe_dma_voff_spill` (`AMDC_RA_LDS_SPILL=8 AMDC_RA_LDS_SPILL_TARGET=0` forces the spill; with
`AMDC_WAITS_SKIP_FENCE_OPERANDS=1` the old behaviour is restored and the probe's output changes).
Loop-entry loads (`begin_loop` init values defined by `load_lds_b128(x2)` with no other use) are coalesced into the loop variable like `set_var` reloads, which removes the second fragment
copy and makes 32 pinned accumulators + 16 pinned fragments (256 AGPRs) fit.