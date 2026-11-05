//! Verifier test cases: kernels the GPU-semantics verifier must reject (`gv_attack_*`, `probe_lds_oob`) or must list as trusted (`gv_audit_*`).

use amdgpu_dsl::*;

/// ATTACK: false contract. The kernel claims a 1 MiB extent for a pointer the host may allocate much
/// smaller; the verifier should not treat the declaration as host truth beyond checking it.
pub fn gv_attack_false_contract() -> Function {
    let mut b = FnBuilder::new("gv_attack_false_contract", 64);
    b.arg_tensor("q", Ext::c(1 << 20));
    b.arg("q", DType::Ptr, 0, 8);
    b.arg("o", DType::Ptr, 8, 8);
    b.arg_tensor("o", Ext::c(64));
    let q = b.let_(Op::KernargPtr { offset: 0 });
    let o = b.let_(Op::KernargPtr { offset: 8 });
    let c4096 = b.c_i32(4096);
    let p = b.ptr_add(q, c4096);
    let v = b.load_f32(p, 0);
    b.store_f32(o, 0, v);
    b.finish()
}

/// ATTACK: `assume` on a loop variable that is violated by `set_var` (only Info in v1), then an OOB
/// global access justified by the assumed range.
pub fn gv_attack_assume_loop() -> Function {
    let mut b = FnBuilder::new("gv_attack_assume_loop", 64);
    b.arg_tensor("q", Ext::c(64));
    b.arg("q", DType::Ptr, 0, 8);
    b.arg("o", DType::Ptr, 8, 8);
    b.arg_tensor("o", Ext::c(64));
    let q = b.let_(Op::KernargPtr { offset: 0 });
    let o = b.let_(Op::KernargPtr { offset: 8 });
    b.allow_var_rereads();
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0))]);
    let i = vars[0];
    b.assume(Opnd::Var(i), Ext::c(0), Ext::c(0));
    let ni = b.add_i32(Opnd::Var(i), c1);
    b.set_var(i, ni);
    let off = b.mul_i32(Opnd::Var(i), c4);
    let p = b.ptr_add(q, off);
    let v = b.load_f32(p, 0);
    b.store_f32(o, 0, v);
    b.end_loop();
    b.finish()
}

/// ATTACK: constant LDS access past the declared 256-byte segment.
pub fn gv_attack_lds_const() -> Function {
    let mut b = FnBuilder::new("gv_attack_lds_const", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg_tensor("o", Ext::c(64));
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let c512 = b.c_i32(512);
    let raw = b.load_lds_b32(Opnd::Val(c512));
    let v = b.bitcast_f32(Opnd::Val(raw));
    b.store_f32(o, 0, v);
    b.finish()
}

/// ATTACK: descriptor num_records (4096) overstates a 64-byte declared extent.
pub fn gv_attack_desc_over() -> Function {
    let mut b = FnBuilder::new("gv_attack_desc_over", 64);
    b.arg_tensor("q", Ext::c(64));
    b.arg("q", DType::Ptr, 0, 8);
    b.arg("o", DType::Ptr, 8, 8);
    b.arg_tensor("o", Ext::c(64));
    let q = b.let_(Op::KernargPtr { offset: 0 });
    let o = b.let_(Op::KernargPtr { offset: 8 });
    let c0 = b.c_i32(0);
    let r = b.buffer_desc(q, 4096);
    let v = b.buffer_load_f32(r, c0, c0, 0);
    b.store_f32(o, 0, v);
    b.finish()
}

/// ATTACK: min-clamp is applied only on one branch; the merged pointer bound must not become trusted
/// if the other branch is unclamped.
pub fn gv_attack_clamp_path() -> Function {
    let mut b = FnBuilder::new("gv_attack_clamp_path", 64);
    b.arg_tensor("q", Ext::c(512));
    b.arg("q", DType::Ptr, 0, 8);
    b.arg("o", DType::Ptr, 8, 8);
    b.arg_tensor("o", Ext::c(64));
    let q = b.let_(Op::KernargPtr { offset: 0 });
    let o = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c0 = b.c_i32(0);
    let c127 = b.c_i32(127);
    let c4 = b.c_i32(4);
    let c16 = b.c_i32(16);
    let off = b.mul_i32(tid, c16);
    let clamped = b.min_i32(off, c127);
    let c32 = b.c_i32(32);
    let pick = b.cmp_lt_i32(tid, c32);
    b.begin_if(Opnd::Val(pick));
    let p1 = b.ptr_add(q, clamped);
    let v1 = b.load_f32(p1, 0);
    b.store_f32(o, 0, v1);
    b.else_();
    let p2 = b.ptr_add(q, off);
    let v2 = b.load_f32(p2, 0);
    b.store_f32(o, 0, v2);
    b.end_if();
    b.finish()
}

/// ATTACK (trust boundary): `assume_trusted` can certify an access the loop actually violates.
pub fn gv_attack_assume_trusted() -> Function {
    let mut b = FnBuilder::new("gv_attack_assume_trusted", 64);
    b.arg_tensor("q", Ext::c(64));
    b.arg("q", DType::Ptr, 0, 8);
    b.arg("o", DType::Ptr, 8, 8);
    b.arg_tensor("o", Ext::c(64));
    let q = b.let_(Op::KernargPtr { offset: 0 });
    let o = b.let_(Op::KernargPtr { offset: 8 });
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0))]);
    let i = vars[0];
    b.assume_trusted(Opnd::Var(i), Ext::c(0), Ext::c(0), "loop exits before i reaches 1");
    let ni = b.add_i32(Opnd::Var(i), c1);
    b.set_var(i, ni);
    let off = b.mul_i32(Opnd::Var(i), c4);
    let p = b.ptr_add(q, off);
    let v = b.load_f32(p, 0);
    b.store_f32(o, 0, v);
    b.end_loop();
    b.finish()
}

/// AUDIT: unsafe global-region protocol made "certified" by the counted `buf_trust` escape. There is
/// no completion/release/acquire; the region check is skipped and the ledger must expose the trust.
pub fn gv_audit_region_trust() -> Function {
    let mut b = FnBuilder::new("gv_audit_region_trust", 128);
    b.arg("buf", DType::Ptr, 0, 8);
    b.arg_tensor("buf", Ext::c(8192));
    b.buf_region("r", "buf", Ext::c(0), Ext::c(8192));
    b.buf_trust("r", "host launch order / cross-kernel visibility argued outside the verifier");
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    b.begin_if(Opnd::Val(is_p));
    b.buf_produce("r", Opnd::I32(0));
    b.store_f32(p, 0, Opnd::F32(1.0));
    b.tile_end();
    b.buf_consume("r", Opnd::I32(0));
    let _ = b.load_f32(p, 0);
    b.tile_end();
    b.end_if();
    b.finish()
}

/// LDS OOB probe (gpu-semantics audit): declares a 256-byte LDS segment, writes/reads sentinels
/// inside it, then reads and writes at offsets past the declared segment. There is no LDS bounds
/// checking on gfx950; the host records whether OOB returns 0, stale data or the written sentinel.
pub fn probe_lds_oob() -> Function {
    let mut b = FnBuilder::new("probe_lds_oob", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(tid, c4);
    let s_valid = b.c_i32(0x1111_1111u32 as i32);
    let s_oob = b.c_i32(0x2222_2222u32 as i32);
    b.store_lds_b32(Opnd::Val(off), Opnd::Val(s_valid));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let valid = b.load_lds_b32(Opnd::Val(off));
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c4096 = b.c_i32(4096);
    let a256 = b.add_i32(off, c256);
    let a1024 = b.add_i32(off, c1024);
    let a4096 = b.add_i32(off, c4096);
    let r256 = b.load_lds_b32(Opnd::Val(a256));
    let r1024 = b.load_lds_b32(Opnd::Val(a1024));
    b.store_lds_b32(Opnd::Val(a4096), Opnd::Val(s_oob));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let r4096 = b.load_lds_b32(Opnd::Val(a4096));
    let f_valid = b.bitcast_f32(Opnd::Val(valid));
    let f_256 = b.bitcast_f32(Opnd::Val(r256));
    let f_1024 = b.bitcast_f32(Opnd::Val(r1024));
    let f_4096 = b.bitcast_f32(Opnd::Val(r4096));
    let c16 = b.c_i32(16);
    let row = b.mul_i32(tid, c16);
    let p = b.ptr_add(o, row);
    b.store_f32(p, 0, Opnd::Val(f_valid));
    b.store_f32(p, 4, Opnd::Val(f_256));
    b.store_f32(p, 8, Opnd::Val(f_1024));
    b.store_f32(p, 12, Opnd::Val(f_4096));
    b.finish()
}
