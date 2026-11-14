# Peephole audit: semantic preconditions

Prompted by a cmpfold bug that matched on SHAPE only (a `v_cmp_ne` of a select with an inline 0 arm taken for a 0/1 materialisation;
fixed by kernel-opt: `fold_select_cmp` / `dce_cmp_pairs` act only on registered materialisations, test `cmpfold_select_zero.rs`). Every other
lowering / MIR fold was audited for the same class of bug: for each, the precondition that makes it correct, and the check in the code.

| Fold | Precondition | Finding | Fix | Test |
|---|---|---|---|---|
| `lds_addr` (x + k into the DS offset) | the folded operand has the same value at the access as at the add | **bug**: a variable operand was read at the access (after a `set_var`) | stop at variables and values in a variable's register (`stable_at_use`) | `an_lds_address_snapshot_of_a_variable_is_not_folded_past_its_update` |
| LDS-DMA M0 readfirstlane cache | the cached register is not rewritten in between | **bug**: a variable's register was cached across its `set_var` | no caching of variable registers | `a_variable_dma_base_is_read_again_after_its_update` |
| `m0::reuse` (one M0 write for consecutive DMAs) | same register AND same word; nothing writes M0 in between | **bugs**: two words of one tuple compared equal; `set_m0` / M0 adds did not reset | compare (base, offset); reset on M0 writers and on any write to the tuple | `two_words_of_one_tuple_are_two_m0_sources`, `an_m0_write_in_between_resets_the_reuse` |
| `frag_cat` producer retargeting | the part's value is read once in total | **bug**: `frag_cat([x, x])` (read count de-duplicated): slot 1 copied an unwritten register | only a part that occurs once | `a_part_given_twice_keeps_its_producer` |
| "single use" folds (loop-entry load coalescing, producer retargeting, MFMA in-place, loopreuse) | the use count is complete | **gap**: asm inputs, variable MFMA / pk-fma / set-lane forms, scope inits, LDS-DMA, `set_m0` were not counted | `value_use_counts` counts every reading statement; `stmt_operands` now includes `set_m0` | `use_counts_see_every_reading_statement` |
| `fuse_minmax3` (min(max) -> med3, max(max) -> max3) | the inner operands are unchanged up to the outer op | **gap**: a write through a view of the operand's tuple was not seen | compare register bases | `med3_is_not_formed_across_a_write_through_a_view` |
| `select_minmax` (cmp + cndmask -> min / max) | f32: no NaN, no signed zero | **unsound for f32** (a NaN second operand, +0 / -0): changed argmax / sample kernels | f32 (and the f32 med3) only with the opt-in `fminmax` | `an_f32_select_is_not_a_min_max_by_default` |
| exact: NotI1 -> negated compare, m0 add -> `s_add m0` | same operands and EXEC at the reader | **gap**: an `exec_set`, a `set_var` of an operand or a control boundary in between | `def_movable` | `a_compare_is_not_moved_across_exec_or_a_variable_write` |
| `remat_consts` / `remat_exprs` / LDS-spill reloads (opt-in remat1-4) | the clone runs under the EXEC its readers run under | **bug**: a clone in a block that narrows EXEC was read under the full EXEC | no clone / reload in an EXEC-changing block | `remat_does_not_clone_under_a_narrow_exec` |
| `batch_kernarg_loads`, `retarget_def` (itself), `retarget_producer`, `lazy_add` / `split_const`, `coalesce`, `dce_cmp_pairs`, `m0_source`, `m0::explicit`, `const_of` / `const_bits`, `lower_condition` VCC folds, `VCndmaskM` | (as stated in the code) | sound | - | - |

The DSL-level and M0 tests fail on the pre-audit compiler and pass after it; on hardware, `ops_check.py select_f32`
(`probe_select_f32`: NaN and signed-zero pairs) fails on the pre-audit compiler (48 of 128 values) and passes after it.

The only shipped code the f32 fold reached were the per-thread running maxima of argmax / sample (`argmax_bf16_v128256`,
`sample_bf16_v128256`, `sample_topkp_bf16_v128256`). Keeping the select there cost mb/argmax v128256 5 % (0.951, 9 reps). The kernels now
state the hardware max themselves (`Op::MaxF32`): it equals the select there because the running maximum is never a NaN (it starts at -inf
and only takes values that compared greater) and an equal zero's sign does not reach the result (the index keeps the compare's choice; the
value only feeds `best == block max`). Their code is identical to before the audit (same md5), so no performance change. Both kernels
ignore a NaN logit (it never compares greater); torch.argmax returns the first NaN's index - a pre-existing kernel-level deviation, not
changed here.