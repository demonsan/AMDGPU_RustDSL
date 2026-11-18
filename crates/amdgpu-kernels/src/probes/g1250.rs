//! gfx1250 G1 probes (docs/gfx1250-plan.md): small wave32 kernels for the gfx1250 emitter / encoder / LLVM tail. They compile for gfx950 too.

use amdgpu_dsl::{DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};

/// `out[i] = a[i] + b[i]` (f32), one element per work-item, 256 work-items per workgroup: global loads / stores, the workgroup id, the
/// workitem id, 64-bit address arithmetic.
pub fn probe_g1250_add() -> Function {
    let mut b = FnBuilder::new("probe_g1250_add", 256);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("out", DType::Ptr, 16, 8);
    b.grid_is(0, Ext::sym("N"));
    for n in ["a", "b", "out"] {
        b.arg_tensor(n, Ext::sym("N") * Ext::c(256 * 4));
    }
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    let po = b.let_(Op::KernargPtr { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let wg = b.let_(Op::WorkgroupIdX);
    let w8 = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(8)));
    let i = b.let_(Op::AddI32(Opnd::Val(w8), Opnd::Val(tid)));
    let off = b.let_(Op::ShlI32(Opnd::Val(i), Opnd::I32(2)));
    let qa = b.let_(Op::PtrAdd { base: Opnd::Val(pa), byte_off: Opnd::Val(off) });
    let qb = b.let_(Op::PtrAdd { base: Opnd::Val(pb), byte_off: Opnd::Val(off) });
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
    let x = b.load_global(qa, 0, MemTy::B32);
    let y = b.load_global(qb, 0, MemTy::B32);
    let xf = b.let_(Op::BitcastF32(Opnd::Val(x)));
    let yf = b.let_(Op::BitcastF32(Opnd::Val(y)));
    let s = b.let_(Op::AddF32(Opnd::Val(xf), Opnd::Val(yf)));
    b.store_global(qo, 0, s, MemTy::B32, false);
    b.finish()
}

/// Transcendentals and scalar arithmetic: `out[tid] = exp2(x) * rsq(y) + sqrt(z)` from three kernarg floats, 64 work-items.
pub fn probe_g1250_math() -> Function {
    let mut b = FnBuilder::new("probe_g1250_math", 64);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg("x", DType::F32, 8, 4);
    b.arg("y", DType::F32, 12, 4);
    b.arg("z", DType::F32, 16, 4);
    b.arg_tensor("out", Ext::c(64 * 4));
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let x = b.let_(Op::KernargF32 { offset: 8 });
    let y = b.let_(Op::KernargF32 { offset: 12 });
    let z = b.let_(Op::KernargF32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let e = b.exp2_f32(x);
    let r = b.un(amdgpu_dsl::UnOp::RsqF32, y);
    let q = b.un(amdgpu_dsl::UnOp::SqrtF32, z);
    let t = b.mul_f32(e, r);
    let v = b.add_f32(t, q);
    let tf = b.un(amdgpu_dsl::UnOp::CvtF32I32, tid);
    let w = b.add_f32(v, tf);
    let off = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
    b.store_global(qo, 0, w, MemTy::B32, false);
    b.finish()
}

/// TDM: the 12 descriptor words (groups 0 and 1) come from the kernarg segment (offset 16..64, built by the host), one
/// `tensor_load_to_lds`, its completion wait, the workgroup barrier, then every work-item copies one LDS dword to `out`. `raw`: the same
/// load as three verbatim instruction words (exact-replay form).
pub fn probe_g1250_tdm_with(raw: Option<[u32; 3]>) -> Function {
    let mut b = FnBuilder::new(if raw.is_some() { "probe_g1250_tdm_raw" } else { "probe_g1250_tdm" }, 64);
    b.arg("out", DType::Ptr, 0, 8);
    for k in 0..12u32 {
        b.arg(&format!("d{k}"), DType::I32, 16 + 4 * k, 4);
    }
    b.arg_tensor("out", Ext::c(64 * 4));
    // host contract: group 1 word 0 (kernarg offset 32) carries no multicast mask (low 16 bits zero): the data-size field only (docs/gfx1250-hang-rules.md GV-1250-MCAST)
    b.ka_one_of(32, &[0, 0x10000, 0x20000, 0x30000]);
    b.lds(4096);
    b.tdm_host_descriptor();   // the evidence kernel of raw / host-built descriptors: the host owns the global address
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let w: Vec<Opnd> = (0..12u32).map(|k| Opnd::Val(b.let_(Op::KernargI32 { offset: 16 + 4 * k }))).collect();
    let g0 = b.let_(Op::RsrcCat { words: [w[0], w[1], w[2], w[3]] });
    let g1 = b.sgpr_cat8([w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11]]);
    match raw {
        Some(r) => b.tensor_op_raw(true, Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default(), r),
        None => b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default()),
    }
    b.wait_tensor(0);
    b.barrier();
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let x = b.load_lds(a, MemTy::B32);
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(a) });
    b.store_global(qo, 0, x, MemTy::B32, false);
    b.finish()
}

pub fn probe_g1250_tdm() -> Function {
    probe_g1250_tdm_with(None)
}

/// One wave32 WMMA (amdgpu-isa `gfx1250` atoms): every lane loads its A / B (/ C) fragment dwords from per-lane-contiguous buffers
/// (`a[lane * na + j]`, the host arranges the operand tiles by the atom layouts), runs the WMMA and stores D the same way. The scaled kind also
/// loads one scale dword per lane for A and for B. `with_c: false`: the zero-accumulator form.
pub fn probe_g1250_wmma_with(name: &str, kind: amdgpu_dsl::WmmaKind, with_c: bool) -> Function {
    let mut b = FnBuilder::new(name, 32);
    let (na, nb, nc) = (kind.a_dwords(), kind.b_dwords(), kind.c_dwords());
    let names = ["a", "b", "c", "d", "sa", "sb"];
    for (k, n) in names.iter().enumerate() {
        b.arg(n, DType::Ptr, 8 * k as u32, 8);
    }
    for (n, dw) in [("a", na), ("b", nb), ("c", nc), ("d", nc), ("sa", 1), ("sb", 1)] {
        b.arg_tensor(n, Ext::c(32 * 4 * dw as i64));
    }
    let lane = b.let_(Op::WorkitemId);
    let ptr = |b: &mut FnBuilder, k: u32, dw: u32| {
        let p = b.let_(Op::KernargPtr { offset: 8 * k });
        let off = b.let_(Op::MulI32(Opnd::Val(lane), Opnd::I32(4 * dw as i32)));
        b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) })
    };
    let frag = |b: &mut FnBuilder, k: u32, dw: u32| {
        let q = ptr(b, k, dw);
        let parts: Vec<Opnd> = (0..dw / 4).map(|i| Opnd::Val(b.load_global(q, 16 * i, MemTy::B128))).collect();
        b.let_(Op::FragCat { parts, dwords: dw as u8 })
    };
    let fa = frag(&mut b, 0, na);
    let fb = frag(&mut b, 1, nb);
    let fc = if with_c { Some(Opnd::Val(frag(&mut b, 2, nc))) } else { None };
    let d = if kind.scaled() {
        let qa = ptr(&mut b, 4, 1);
        let qb = ptr(&mut b, 5, 1);
        let sa = b.load_global(qa, 0, MemTy::B32);
        let sb = b.load_global(qb, 0, MemTy::B32);
        b.wmma_scaled(kind, fa, fb, fc, amdgpu_dsl::WmmaScale { sa: Opnd::Val(sa), sb: Opnd::Val(sb), b_row1: false })
    } else {
        b.wmma(kind, fa, fb, fc)
    };
    let qd = ptr(&mut b, 3, nc);
    for i in 0..nc / 4 {
        let s = b.let_(Op::FragSlice { frag: Opnd::Val(d), start: (4 * i) as u8, n: 4 });
        b.store_global(qd, 16 * i, s, MemTy::B128, false);
    }
    b.finish()
}

/// `v_wmma_f32_16x16x64_fp8_fp8` with an accumulator.
pub fn probe_g1250_wmma_fp8() -> Function {
    probe_g1250_wmma_with("probe_g1250_wmma_fp8", amdgpu_dsl::WmmaKind::Fp8_16x16x64, true)
}

/// `v_wmma_f32_16x16x128_fp8_fp8`, zero accumulator.
pub fn probe_g1250_wmma_fp8_k128() -> Function {
    probe_g1250_wmma_with("probe_g1250_wmma_fp8_k128", amdgpu_dsl::WmmaKind::Fp8_16x16x128, false)
}

/// `v_wmma_scale_f32_32x16x128_f4` (E8M0 block 32) with an accumulator.
pub fn probe_g1250_wmma_f4() -> Function {
    probe_g1250_wmma_with("probe_g1250_wmma_f4", amdgpu_dsl::WmmaKind::ScaleF4_32x16x128, true)
}

/// Lane masks on wave32: `out[i] = a[i] > b[i] ? a[i] : 2 b[i]` and `out[i + 256] = (i & 1) ? 7 : -i` (compares into VCC and into an SGPR,
/// `v_cndmask`), 256 work-items.
pub fn probe_g1250_select() -> Function {
    use amdgpu_dsl::{CmpPred, CmpTy};
    let mut b = FnBuilder::new("probe_g1250_select", 256);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("out", DType::Ptr, 16, 8);
    b.arg_tensor("a", Ext::c(256 * 4));
    b.arg_tensor("b", Ext::c(256 * 4));
    b.arg_tensor("out", Ext::c(512 * 4));
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    let po = b.let_(Op::KernargPtr { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let off = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let qa = b.let_(Op::PtrAdd { base: Opnd::Val(pa), byte_off: Opnd::Val(off) });
    let qb = b.let_(Op::PtrAdd { base: Opnd::Val(pb), byte_off: Opnd::Val(off) });
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
    let x = b.load_global(qa, 0, MemTy::B32);
    let y = b.load_global(qb, 0, MemTy::B32);
    let xf = b.let_(Op::BitcastF32(Opnd::Val(x)));
    let yf = b.let_(Op::BitcastF32(Opnd::Val(y)));
    let y2 = b.add_f32(yf, yf);
    let c = b.cmp(CmpPred::Gt, CmpTy::F32, xf, yf);
    let s = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(c), a: Opnd::Val(xf), b: Opnd::Val(y2) });
    b.store_global(qo, 0, s, MemTy::B32, false);
    let odd = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(1)));
    let c2 = b.cmp(CmpPred::Ne, CmpTy::I32, odd, Opnd::I32(0));
    let neg = b.bin(amdgpu_dsl::BinOp::SubU32, Opnd::I32(0), tid);
    let s2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c2), a: Opnd::I32(7), b: Opnd::Val(neg) });
    b.store_global(qo, 1024, s2, MemTy::B32, false);
    b.finish()
}

/// Buffers on gfx1250 (the 57-bit-base / 45-bit-num_records V#): `out[i] = 3 * a[i] + b[i]` through buffer loads / stores with a
/// constant and a kernarg-sized `num_records`, plus a returning buffer atomic add into `out[256]`; 256 work-items.
pub fn probe_g1250_buffer() -> Function {
    let mut b = FnBuilder::new("probe_g1250_buffer", 256);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("out", DType::Ptr, 16, 8);
    b.arg("nb", DType::I32, 24, 4);
    b.arg_tensor("a", Ext::c(256 * 4));
    b.arg_tensor("b", Ext::c(256 * 4));
    b.arg_tensor("out", Ext::c(257 * 4));
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    let po = b.let_(Op::KernargPtr { offset: 16 });
    let nb = b.let_(Op::KernargI32 { offset: 24 });
    let ra = b.buffer_desc(pa, 1024);
    let rb = b.buffer_desc_dyn(pb, nb);
    let ro = b.buffer_desc(po, 1028);
    let tid = b.let_(Op::WorkitemId);
    let off = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let x = b.buffer_load_f32(ra, off, Opnd::I32(0), 0);
    let y = b.buffer_load_f32(rb, off, Opnd::I32(0), 0);
    let x3 = b.mul_f32(x, Opnd::F32(3.0));
    let s = b.add_f32(x3, y);
    b.buffer_store_f32(ro, off, Opnd::I32(0), 0, s);
    let one = b.let_(Op::ConstI32(1));
    let _r = b.buffer_atomic_add_ret(ro, Opnd::I32(0), Opnd::I32(0), 1024, one);
    b.finish()
}

/// A 2-D TDM tile from the typed descriptor builder (`FnBuilder::tensor_desc_2d`): a 16 x 8 f32 tile of a row-major tensor, starting
/// `off` bytes into `src`, with `dim0` / `dim1` the extents from the tile start (elements outside read as 0) and `pitch` elements per
/// row; then every work-item copies one LDS dword to `out[tid]`. 128 work-items. Kernargs: src @0, out @8, dim0 @16, dim1 @20,
/// pitch @24, off @28.
pub fn probe_g1250_tdm2d() -> Function {
    let mut b = FnBuilder::new("probe_g1250_tdm2d", 128);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    for (k, n) in ["dim0", "dim1", "pitch", "off"].iter().enumerate() {
        b.arg(n, DType::I32, 16 + 4 * k as u32, 4);
    }
    b.arg_tensor("out", Ext::c(128 * 4));
    // host contract: `src` holds the tile (16 x 8 f32, `pitch` elements per row) `off` bytes in
    b.arg_tensor("src", Ext::ka(28) + Ext::c(7 * 4) * Ext::ka(24) + Ext::c(64));
    b.lds(128 * 4);
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let d0 = b.let_(Op::KernargI32 { offset: 16 });
    let d1 = b.let_(Op::KernargI32 { offset: 20 });
    let pitch = b.let_(Op::KernargI32 { offset: 24 });
    let off = b.let_(Op::KernargI32 { offset: 28 });
    let t = amdgpu_dsl::TensorTile { data_size: 4, tensor_dim: [Opnd::Val(d0), Opnd::Val(d1)], stride0: Opnd::Val(pitch), tile: [16, 8], pad: None, wg_mask: 0 };
    let (g0, g1) = b.tensor_desc_2d(&t, Opnd::Val(ps), Opnd::Val(off), Opnd::I32(0));
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default());
    b.wait_tensor(0);
    b.barrier();
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let x = b.load_lds(a, MemTy::B32);
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(a) });
    b.store_global(qo, 0, x, MemTy::B32, false);
    b.finish()
}

/// Named and cluster barriers (compile / encode coverage; the cluster barrier needs a cluster launch): every wave initialises and joins
/// named barrier 1 (all 4 waves), signals / waits it, leaves it, then the cluster barrier; out[tid] = tid.
pub fn probe_g1250_barriers() -> Function {
    let mut b = FnBuilder::new("probe_g1250_barriers", 128);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::c(128 * 4));
    let po = b.let_(Op::KernargPtr { offset: 0 });
    b.barrier_init(1, 4);
    b.barrier(); // orders the initialization before any wave uses the barrier (GV-1250-BAR-INIT)
    b.barrier_join(1);
    b.barrier_signal(1);
    b.barrier_wait(1);
    b.barrier_leave();
    b.cluster_barrier();
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(a) });
    b.store_global(q, 0, tid, MemTy::B32, false);
    let mut f = b.finish();
    f.kd.push(("cluster_dims".into(), 2));
    f.kd.push(("named_barrier_count".into(), 1));
    f
}

/// A 2-slot TDM ring (unrolled, K = 4 tiles of 16 x 8 f32 from a 32 x 16 row-major tensor): tile i+2 is issued into the slot tile i was
/// read from once every wave has read it (barrier), tile i is consumed after `wait_tensor(1)` (the in-order TENSORcnt: only tile i+1 may
/// still be in flight) and a barrier. out[tid] = sum over the 4 tiles of element tid. 128 work-items. Kernargs: src @0, out @8.
pub fn probe_g1250_tdm_ring() -> Function {
    tdm_ring_with(true, true)
}

/// The ring of `probe_g1250_tdm_ring` with its protocol markers (an `lds_ring` of 2 slots, `tile_produce` around each tensor load,
/// `tile_consume` around each read), so the protocol verifier checks the tensor completion (`wait_tensor`) and the slot reuse (barriers).
/// `wait`: the consumer's tensor wait; `reuse_barrier`: the barrier between the last read of a slot and the tensor load overwriting it
/// (both true in the shipped probe; the mutation tests clear them).
pub fn tdm_ring_with(wait: bool, reuse_barrier: bool) -> Function {
    const K: usize = 4;
    let mut b = FnBuilder::new("probe_g1250_tdm_ring", 128);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("out", Ext::c(128 * 4));
    b.arg_tensor("src", Ext::c(K as i64 * 128 * 4));
    b.lds(2 * 128 * 4);
    b.lds_ring("ring", 0, 512, 2);
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let tile = amdgpu_dsl::TensorTile { data_size: 4, tensor_dim: [Opnd::I32(16), Opnd::I32(8)], stride0: Opnd::I32(16), tile: [16, 8], pad: None, wg_mask: 0 };
    let issue = |b: &mut FnBuilder, i: usize| {
        let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), Opnd::I32((i * 128 * 4) as i32), Opnd::I32(((i % 2) * 512) as i32));
        b.tile_produce("ring", Opnd::I32((i % 2) as i32), Opnd::I32((i / 2) as i32));
        b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default());
        b.tile_end();
    };
    issue(&mut b, 0);
    issue(&mut b, 1);
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let mut acc = b.let_(Op::ConstF32(0.0));
    for i in 0..K {
        if wait {
            b.wait_tensor(if i + 1 < K { 1 } else { 0 });
        }
        b.barrier();
        let la = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(((i % 2) * 512) as i32)));
        b.tile_consume("ring", Opnd::I32((i % 2) as i32), Opnd::I32((i / 2) as i32));
        let x = b.load_lds(la, MemTy::B32);
        b.tile_end();
        let xf = b.let_(Op::BitcastF32(Opnd::Val(x)));
        acc = b.add_f32(acc, xf);
        if i + 2 < K {
            b.wait_lds(0);
            if reuse_barrier {
                b.barrier();
            }
            issue(&mut b, i + 2);
        }
    }
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(a) });
    b.store_global(qo, 0, acc, MemTy::B32, false);
    b.finish()
}

/// A named barrier as the workgroup's synchronisation (hardware probe): 4 waves initialise / join named barrier 1 (4 members), every
/// work-item writes `3 * tid` to LDS, releases its LDS writes (`wait_lds(0)`), signals / waits barrier 1, then reads the value of the
/// work-item one wave further: out[tid] = 3 * ((tid + 32) % 128). Then it leaves the barrier. 128 work-items.
pub fn probe_g1250_named() -> Function {
    let mut b = FnBuilder::new("probe_g1250_named", 128);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::c(128 * 4));
    b.lds(128 * 4);
    let po = b.let_(Op::KernargPtr { offset: 0 });
    b.barrier_init(1, 4);
    b.barrier(); // orders the initialization before any wave uses the barrier (GV-1250-BAR-INIT)
    b.barrier_join(1);
    // (the ISA's recommended order: the member count is set before any wave uses the barrier)
    b.barrier();
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let v = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(3)));
    b.store_lds_b32(a, v);
    b.wait_lds(0);
    b.barrier_signal(1);
    b.barrier_wait(1);
    let a2 = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(128)));
    let a3 = b.let_(Op::AndI32(Opnd::Val(a2), Opnd::I32(511)));
    let x = b.load_lds(a3, MemTy::B32);
    b.barrier_leave();
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(a) });
    b.store_global(q, 0, x, MemTy::B32, false);
    let mut f = b.finish();
    f.kd.push(("named_barrier_count".into(), 1));
    f
}

/// More than 256 VGPRs (the MSB window): one wave loads `N` values per lane (`src[k * 32 + lane]`), all live until a sum in reverse order
/// at the end: out[lane] = (((x[N-1] + x[N-2]) + ...) + x[0]). Kernargs: src @0, out @8.
pub fn probe_g1250_wide() -> Function {
    const N: usize = 320;
    let mut b = FnBuilder::new("probe_g1250_wide", 32);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("src", Ext::c((N * 32 * 4) as i64));
    b.arg_tensor("out", Ext::c(32 * 4));
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(ps), byte_off: Opnd::Val(a) });
    let xs: Vec<ValueId> = (0..N)
        .map(|k| {
            let qk = b.let_(Op::PtrAdd { base: Opnd::Val(q), byte_off: Opnd::I32((k / 32 * 4096) as i32) });
            b.load_global(qk, ((k % 32) * 128) as u32, MemTy::B32)
        })
        .collect();
    let mut acc = b.let_(Op::BitcastF32(Opnd::Val(xs[N - 1])));
    for k in (0..N - 1).rev() {
        let f = b.let_(Op::BitcastF32(Opnd::Val(xs[k])));
        acc = b.add_f32(acc, f);
    }
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(a) });
    b.store_global(qo, 0, acc, MemTy::B32, false);
    b.finish()
}

/// P1 (LDS above 64 KiB): every work-item of every workgroup writes `(wg << 24) ^ (k << 12) ^ tid` to LDS byte `k * 1024 + 4 tid` for every
/// 1 KiB row k of an allocation of `bytes`, then (after a barrier) reads the row words of work-item (tid + 37) % 256 and sums the XOR
/// against the expected value: out[wg * 256 + tid] = 0 when every row came back. 256 work-items; many workgroups run at once (allocation
/// granularity / overlap). Kernarg: out @0.
pub fn probe_g1250_lds_big(bytes: u32) -> Function {
    let rows = bytes / 1024;
    let mut b = FnBuilder::new(&format!("probe_g1250_lds_{bytes}"), 256);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::grid_x() * Ext::c(1024));   // one KiB per workgroup
    b.lds(bytes);
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let wg = b.let_(Op::WorkgroupIdX);
    let wgs = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(24)));
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let base = b.let_(Op::XorI32(Opnd::Val(wgs), Opnd::Val(tid)));
    for k in 0..rows {
        let la = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32((k * 1024) as i32)));
        let v = b.let_(Op::XorI32(Opnd::Val(base), Opnd::I32((k << 12) as i32)));
        b.store_lds_b32(la, v);
    }
    b.wait_lds(0);
    b.barrier();
    let t37 = b.let_(Op::AddI32(Opnd::Val(tid), Opnd::I32(37)));
    let t2 = b.let_(Op::AndI32(Opnd::Val(t37), Opnd::I32(255)));
    let a2 = b.let_(Op::ShlI32(Opnd::Val(t2), Opnd::I32(2)));
    let base2 = b.let_(Op::XorI32(Opnd::Val(wgs), Opnd::Val(t2)));
    let mut acc = b.let_(Op::ConstI32(0));
    for k in 0..rows {
        let la = b.let_(Op::AddI32(Opnd::Val(a2), Opnd::I32((k * 1024) as i32)));
        let x = b.load_lds(la, MemTy::B32);
        let e = b.let_(Op::XorI32(Opnd::Val(base2), Opnd::I32((k << 12) as i32)));
        let d = b.let_(Op::XorI32(Opnd::Val(x), Opnd::Val(e)));
        acc = b.let_(Op::AddI32(Opnd::Val(acc), Opnd::Val(d)));
    }
    let g = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(10)));
    let o = b.let_(Op::AddI32(Opnd::Val(g), Opnd::Val(a)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o) });
    b.store_global(q, 0, acc, MemTy::B32, false);
    let mut f = b.finish();
    f
}

/// P2 (TDM above 64 KiB of LDS): one tensor load of a 1024-dword row (4 KiB) to LDS byte `lds_addr` (LDS declared 163 840 B), read back
/// with ds_load: out[t * 4 + j] = LDS[lds_addr + 16 t + 4 j] (256 work-items x 4 dwords = the row). Kernargs: src @0, out @8.
pub fn probe_g1250_tdm_hi(lds_addr: i32) -> Function {
    let mut b = FnBuilder::new(&format!("probe_g1250_tdm_hi_{lds_addr}"), 256);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("out", Ext::c(4096));
    b.arg_tensor("src", Ext::c(4096));
    b.lds(163_840);
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let tile = amdgpu_dsl::TensorTile { data_size: 4, tensor_dim: [Opnd::I32(1024), Opnd::I32(1)], stride0: Opnd::I32(1024), tile: [1024, 1], pad: None, wg_mask: 0 };
    let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), Opnd::I32(0), Opnd::I32(lds_addr));
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default());
    b.wait_tensor(0);
    b.barrier();
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(4)));
    for j in 0..4 {
        let la = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(lds_addr + 4 * j)));
        let x = b.load_lds(la, MemTy::B32);
        let ga = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(4 * j)));
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(ga) });
        b.store_global(q, 0, x, MemTy::B32, false);
    }
    let mut f = b.finish();
    f
}

/// TRANS co-execution (hardware evidence for the gfx1250 TRANS -> VALU rule): per lane, 8 independent chains x 16 steps; each step is a
/// TRANS op (rsq / sqrt / sin / cos rotating) whose result the next VALU (add / mul / fma / pk_add pattern) reads right away, so the
/// compiled code has a TRANS directly followed by its reader at many sites (the emitter puts one v_nop there; the test removes them in a
/// copy and compares bit for bit). out[tid * 8 + c] = chain c. 512 work-items. Kernargs: out @0.
pub fn probe_g1250_trans() -> Function {
    use amdgpu_dsl::UnOp;
    let mut b = FnBuilder::new("probe_g1250_trans", 512);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::grid_x() * Ext::c(16384));   // 16 KiB per workgroup
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let tf = b.un(UnOp::CvtF32I32, tid);
    let ops = [UnOp::RsqF32, UnOp::SqrtF32, UnOp::SinF32, UnOp::CosF32, UnOp::SqrtF32];
    let mut ch: Vec<ValueId> = (0..8).map(|c| b.add_f32(tf, Opnd::F32(1.5 + c as f32 * 0.37))).collect();
    for s in 0..16 {
        for c in 0..8 {
            let x = ch[c];
            // keep the arguments in a safe range: |x| small and positive before the transcendental
            let xa = b.mul_f32(x, Opnd::F32(0.001));
            let xa = b.add_f32(xa, Opnd::F32(1.25));
            let tr = b.un(ops[(s + c) % ops.len()], xa);
            let y = if (s + c) % 2 == 0 { b.add_f32(tr, x) } else { b.mul_f32(tr, Opnd::F32(1.0001)) };
            ch[c] = y;
        }
    }
    // out[(wg * 512 + tid) * 8 + c]
    let wg = b.let_(Op::WorkgroupIdX);
    let wg9 = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(9)));
    let gt = b.let_(Op::AddI32(Opnd::Val(wg9), Opnd::Val(tid)));
    let a = b.let_(Op::ShlI32(Opnd::Val(gt), Opnd::I32(5)));
    for c in 0..8 {
        let o = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(4 * c as i32)));
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o) });
        b.store_global(q, 0, ch[c], MemTy::B32, false);
    }
    b.finish()
}

/// TRANS co-execution, all combinations: every TRANS op (exp2 / log2 / rcp / rsq / sqrt / sin / cos) read right away by every reader class
/// and operand position: VOP2 add (src0 / src1), mul, VOP3 fma (src0 / src1 / src2), max, VOP3P packed f32 add (the TRANS result one
/// half of a packed pair). 8 chains x 21 steps per lane; combination k = (step * 8 + chain) mod 49 (each about 3 times). The script removes
/// the emitter's v_nop per reader class and compares bit for bit. out[(wg * 512 + tid) * 8 + chain]. 512 work-items. Kernargs: out @0.
pub fn probe_g1250_trans_all() -> Function {
    use amdgpu_dsl::UnOp;
    let mut b = FnBuilder::new("probe_g1250_trans_all", 512);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::grid_x() * Ext::c(16384));   // 16 KiB per workgroup (tensor-bounds contract)
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let tf = b.un(UnOp::CvtF32I32, tid);
    let tf = b.mul_f32(tf, Opnd::F32(1.0 / 512.0));
    let mut ch: Vec<ValueId> = (0..8).map(|c| b.add_f32(tf, Opnd::F32(0.5 + c as f32 * 0.0625))).collect();
    for s in 0..21 {
        for c in 0..8 {
            let k = (s * 8 + c) % 49;
            let (tr_i, rd) = (k % 7, k / 7);
            let x = ch[c];
            // the TRANS argument in (1, 2): safe for every op
            let xa = b.mul_f32(x, Opnd::F32(0.03125));
            let xa = b.add_f32(xa, Opnd::F32(1.125));
            let trans = |b: &mut FnBuilder, i: usize, v: ValueId| match i {
                0 => b.let_(Op::Exp2F32(Opnd::Val(v))),
                1 => b.let_(Op::Log2F32(Opnd::Val(v))),
                2 => b.let_(Op::RcpF32(Opnd::Val(v))),
                3 => b.un(UnOp::RsqF32, v),
                4 => b.un(UnOp::SqrtF32, v),
                5 => b.un(UnOp::SinF32, v),
                _ => b.un(UnOp::CosF32, v),
            };
            let tr = if rd == 6 { xa } else { trans(&mut b, tr_i, xa) };
            // keep |y| bounded: every reader mixes the TRANS result with x scaled down
            let y = match rd {
                0 => b.add_f32(tr, x),
                1 => b.add_f32(x, tr),
                2 => b.mul_f32(tr, Opnd::F32(0.75)),
                3 => b.let_(Op::FmaF32(Opnd::Val(tr), Opnd::F32(0.5), Opnd::Val(x))),
                4 => b.let_(Op::FmaF32(Opnd::F32(0.5), Opnd::Val(tr), Opnd::Val(x))),
                5 => b.let_(Op::FmaF32(Opnd::Val(x), Opnd::F32(0.5), Opnd::Val(tr))),
                _ => {
                    // both halves written by TRANS ops (the second one right before the packed reader), read as a packed pair
                    let t0 = b.un(UnOp::SqrtF32, x);
                    let t1 = trans(&mut b, tr_i, xa);
                    let pr = b.frag_cat(&[Opnd::Val(t0), Opnd::Val(t1)]);
                    let sum = b.let_(Op::PkAddF32(Opnd::Val(pr), Opnd::Val(pr)));
                    let lo = b.frag_slice(sum, 0, 1);
                    let lo = b.bitcast_f32(lo);
                    let hi = b.frag_slice(sum, 1, 1);
                    let hi = b.bitcast_f32(hi);
                    b.add_f32(lo, hi)
                }
            };
            // the max reader: on odd steps the chain value also goes through max(tr', y) with a fresh TRANS
            let y = if s % 3 == 1 {
                let t2 = trans(&mut b, (tr_i + 3) % 7, xa);
                b.let_(Op::MaxF32(Opnd::Val(t2), Opnd::Val(y)))
            } else {
                y
            };
            // renormalise: y / 4 + 0.5 (stays in a small range over 21 steps)
            ch[c] = b.let_(Op::FmaF32(Opnd::Val(y), Opnd::F32(0.25), Opnd::F32(0.5)));
        }
    }
    let wg = b.let_(Op::WorkgroupIdX);
    let wg9 = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(9)));
    let gt = b.let_(Op::AddI32(Opnd::Val(wg9), Opnd::Val(tid)));
    let a = b.let_(Op::ShlI32(Opnd::Val(gt), Opnd::I32(5)));
    for c in 0..8 {
        let o = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(4 * c as i32)));
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o) });
        b.store_global(q, 0, ch[c], MemTy::B32, false);
    }
    b.finish()
}

/// Multicast probe (cluster of 2 workgroups in x): every workgroup loads ITS slice of `src` (2048 bytes, slice = its position in the cluster) with the group-1 multicast mask `mask` into LDS at
/// byte 2048 x slice (wave 0 issues, cache policy `th` / `scope`), waits for its own tensor completion, a workgroup barrier, the cluster barrier; then every work-item copies the whole 4096-byte LDS
/// to `out[position]`. With mask 0b11 each workgroup must see both slices; mask 0 (no multicast) only its own. Kernargs: src @0 (4096 bytes), out @8 (8192 bytes). 64 work-items.
pub fn probe_g1250_mcast_with(name: &str, th: u8, scope: u8, mask: u16) -> Function {
    probe_g1250_mcast_lds(name, th, scope, mask, 4096)
}

/// `probe_g1250_mcast_with` declaring `lds_bytes` of LDS (hypothesis: a multicast lands at the SAME LDS address in every destination, which is only the same location when every workgroup's LDS
/// base is equal, e.g. when each workgroup owns the whole LDS).
pub fn probe_g1250_mcast_lds(name: &str, th: u8, scope: u8, mask: u16, lds_bytes: u32) -> Function {
    probe_g1250_mcast_full(name, th, scope, mask, lds_bytes, false, 0)
}

/// The full multicast probe. `wm24`: the issuing wave first sets WAVE_MODE bit 24 (`s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 24, 1), 1`), which the vendor's GEMM kernel does in its wave 0 only
/// (the wave that issues the tensor loads of its workgroup), before any tensor operation.
pub fn probe_g1250_mcast_full(name: &str, th: u8, scope: u8, mask: u16, lds_bytes: u32, wm24: bool, delay: u32) -> Function {
    probe_g1250_mcast_same(name, th, scope, mask, lds_bytes, wm24, delay, false)
}

/// `same`: every workgroup issues the SAME request (the slice 0 of `src` to LDS byte 0, the same mask): the cluster multicast is a rendezvous of identical requests (the ISA: one wave per
/// workgroup issues the same request with the same mask; the data goes to every requesting workgroup once all have asked). Expected with mask 0b11: both workgroups hold slice 0 at byte 0.
pub fn probe_g1250_mcast_same(name: &str, th: u8, scope: u8, mask: u16, lds_bytes: u32, wm24: bool, delay: u32, same: bool) -> Function {
    use amdgpu_dsl::ev;
    let mut b = FnBuilder::new(name, 64);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("src", Ext::c(4096));
    b.arg_tensor("out", Ext::c(8192));
    b.lds(lds_bytes);
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let t6 = b.let_(Op::ReadTtmp { idx: 6 });
    let pos = b.let_(Op::AndI32(Opnd::Val(t6), Opnd::I32(15)));
    b.assume_trusted(pos, Ext::c(0), Ext::c(1), "cluster position x < the cluster's 2 workgroups (TTMP6[3:0])");
    let off = if same { b.let_(Op::AddI32(Opnd::I32(0), Opnd::I32(0))) } else { b.let_(Op::ShlI32(Opnd::Val(pos), Opnd::I32(11))) };
    let tile = amdgpu_dsl::TensorTile { data_size: 4, tensor_dim: [Opnd::I32(512), Opnd::I32(1)], stride0: Opnd::I32(512), tile: [512, 1], pad: None, wg_mask: mask };
    let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), Opnd::Val(off), Opnd::Val(off));
    let tid = b.let_(Op::WorkitemId);
    let w5 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let w = b.let_(Op::ReadFirstLane(Opnd::Val(w5)));
    let lead = b.bind_cond(ev(w).eq(0));
    b.begin_if(Opnd::Val(lead));
    if wm24 {
        b.set_reg_imm(1, 24, 1, 1);   // HW_REG_WAVE_MODE (id 1), bit 24
    }
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol { th, scope, nv: false });
    b.end_if();
    b.wait_tensor(0);
    b.barrier();
    b.cluster_barrier();
    if delay > 0 {
        // a delay of about `delay` workgroup barriers before the copy: separates "the peer's slice arrives late" from "never"
        let v = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
        b.barrier();
        let nx = b.add_i32(v[0], Opnd::I32(1));
        b.set_var(v[0], nx);
        let c = b.cmp(amdgpu_dsl::CmpPred::Ge, amdgpu_dsl::CmpTy::I32, nx, Opnd::I32(delay as i32));
        b.break_if(c);
        b.end_loop();
    }
    let pbase = b.let_(Op::ShlI32(Opnd::Val(pos), Opnd::I32(12)));
    for k in 0..16i32 {
        let a = b.let_(Op::AddI32(Opnd::Val(tid), Opnd::I32(64 * k)));
        let la = b.let_(Op::ShlI32(Opnd::Val(a), Opnd::I32(2)));
        let x = b.load_lds(la, MemTy::B32);
        let go = b.let_(Op::AddI32(Opnd::Val(pbase), Opnd::Val(la)));
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(go) });
        b.store_global(q, 0, x, MemTy::B32, false);
    }
    let mut f = b.finish();
    f.kd.push(("cluster_dims".into(), 2));
    f
}

/// Multicast landing-place scan (cluster of 2): both workgroups fill 64 KiB of LDS with a marker, join the cluster barrier, wave 0 of each loads its 2048-byte slice of `src` with mask 0b11 at LDS byte
/// 2048 x position, then own wait, workgroup barrier, cluster barrier, and every work-item dumps the whole 64 KiB to `out[position]`. The host looks for the slices' dwords everywhere in both dumps, to
/// find where a multicast lands (if anywhere). Kernargs: src @0 (4096 bytes), out @8 (131072 bytes). 64 work-items.
pub fn probe_g1250_mcast_scan(name: &str, mask: u16) -> Function {
    use amdgpu_dsl::ev;
    let mut b = FnBuilder::new(name, 64);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("src", Ext::c(4096));
    b.arg_tensor("out", Ext::c(131072));
    b.lds(65536);
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let t6 = b.let_(Op::ReadTtmp { idx: 6 });
    let pos = b.let_(Op::AndI32(Opnd::Val(t6), Opnd::I32(15)));
    b.assume_trusted(pos, Ext::c(0), Ext::c(1), "cluster position x < the cluster's 2 workgroups (TTMP6[3:0])");
    let tid = b.let_(Op::WorkitemId);
    for k in 0..256i32 {
        let a = b.let_(Op::AddI32(Opnd::Val(tid), Opnd::I32(64 * k)));
        let la = b.let_(Op::ShlI32(Opnd::Val(a), Opnd::I32(2)));
        let v = b.let_(Op::XorI32(Opnd::I32(0x7EED_0000u32 as i32), Opnd::Val(la)));
        b.store_lds_b32(la, v);
    }
    b.wait_lds(0);
    b.barrier();
    b.cluster_barrier();
    let off = b.let_(Op::ShlI32(Opnd::Val(pos), Opnd::I32(11)));
    let tile = amdgpu_dsl::TensorTile { data_size: 4, tensor_dim: [Opnd::I32(512), Opnd::I32(1)], stride0: Opnd::I32(512), tile: [512, 1], pad: None, wg_mask: mask };
    let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), Opnd::Val(off), Opnd::Val(off));
    let w5 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let w = b.let_(Op::ReadFirstLane(Opnd::Val(w5)));
    let lead = b.bind_cond(ev(w).eq(0));
    b.begin_if(Opnd::Val(lead));
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default());
    b.end_if();
    b.wait_tensor(0);
    b.barrier();
    b.cluster_barrier();
    let pbase = b.let_(Op::ShlI32(Opnd::Val(pos), Opnd::I32(16)));
    for k in 0..256i32 {
        let a = b.let_(Op::AddI32(Opnd::Val(tid), Opnd::I32(64 * k)));
        let la = b.let_(Op::ShlI32(Opnd::Val(a), Opnd::I32(2)));
        let x = b.load_lds(la, MemTy::B32);
        let go = b.let_(Op::AddI32(Opnd::Val(pbase), Opnd::Val(la)));
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(go) });
        b.store_global(q, 0, x, MemTy::B32, false);
    }
    let mut f = b.finish();
    f.kd.push(("cluster_dims".into(), 2));
    f
}
