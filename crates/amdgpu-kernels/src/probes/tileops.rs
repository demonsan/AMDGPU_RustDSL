//! Hardware probes of the P1 base ops (docs/op-coverage-tileops.md §6.5): every op on 64 lanes of generated inputs, one result dword
//! per (op, lane); `tools/tileop_probe.sh <probe>` runs it and checks the bytes against the Python reference (`tools/tileop_probe_check.py`).
use amdgpu_dsl::{BinOp, CmpPred, CmpTy, DType, FnBuilder, Function, Op, Opnd, TriOp, UnOp, ValueId};

/// Input planes (64 lanes each, dwords): 0 a:f32, 1 b:f32, 2 c:f32, 3 ia:i32, 4 ib:i32, 5 ic:i32.
pub const VALU_P1_INPUT_PLANES: u32 = 6;

/// The ops of `probe_valu_p1` in result order (the checker uses the same names).
pub fn valu_p1_ops() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for u in UN {
        v.push(format!("{u:?}"));
    }
    for b in BIN {
        v.push(format!("{b:?}"));
    }
    for t in TRI {
        v.push(format!("{t:?}"));
    }
    for (p, ty) in CMPS {
        v.push(format!("Cmp{p:?}{ty:?}"));
    }
    v
}

const UN: [UnOp; 17] = [
    UnOp::FloorF32,
    UnOp::CeilF32,
    UnOp::TruncF32,
    UnOp::RndneF32,
    UnOp::FractF32,
    UnOp::RsqF32,
    UnOp::SqrtF32,
    UnOp::SinF32,
    UnOp::CosF32,
    UnOp::CvtI32F32,
    UnOp::CvtU32F32,
    UnOp::CvtF32I32,
    UnOp::CvtF32U32,
    UnOp::CvtF32Ubyte0,
    UnOp::CvtF32Ubyte1,
    UnOp::CvtF32Ubyte2,
    UnOp::CvtF32Ubyte3,
];
const BIN: [BinOp; 14] = [
    BinOp::MinF32,
    BinOp::MaxI32,
    BinOp::MaxU32,
    BinOp::MinU32,
    BinOp::SubU32,
    BinOp::MulU32U24,
    BinOp::AshrI32,
    BinOp::MulHiU32,
    BinOp::MulHiI32,
    BinOp::OrB32,
    BinOp::PkAddF16,
    BinOp::PkMulF16,
    BinOp::PkMaxF16,
    BinOp::PkMinF16,
];
const TRI: [TriOp; 15] = [
    TriOp::Min3F32,
    TriOp::Med3F32,
    TriOp::Min3I32,
    TriOp::Max3I32,
    TriOp::Med3I32,
    TriOp::Min3U32,
    TriOp::Max3U32,
    TriOp::Med3U32,
    TriOp::MadU32U24,
    TriOp::LshlAddU32,
    TriOp::Add3U32,
    TriOp::BfeU32,
    TriOp::BfeI32,
    TriOp::BfiB32,
    TriOp::PkFmaF16,
];
const CMPS: [(CmpPred, CmpTy); 22] = [
    (CmpPred::Lt, CmpTy::F32),
    (CmpPred::Le, CmpTy::F32),
    (CmpPred::Gt, CmpTy::F32),
    (CmpPred::Ge, CmpTy::F32),
    (CmpPred::Eq, CmpTy::F32),
    (CmpPred::Ne, CmpTy::F32),
    (CmpPred::Lg, CmpTy::F32),
    (CmpPred::Uno, CmpTy::F32),
    (CmpPred::Ord, CmpTy::F32),
    (CmpPred::Lt, CmpTy::I32),
    (CmpPred::Le, CmpTy::I32),
    (CmpPred::Gt, CmpTy::I32),
    (CmpPred::Ge, CmpTy::I32),
    (CmpPred::Eq, CmpTy::I32),
    (CmpPred::Ne, CmpTy::I32),
    (CmpPred::Lt, CmpTy::U32),
    (CmpPred::Le, CmpTy::U32),
    (CmpPred::Gt, CmpTy::U32),
    (CmpPred::Ge, CmpTy::U32),
    (CmpPred::Eq, CmpTy::U32),
    (CmpPred::Ne, CmpTy::U32),
    (CmpPred::Ne, CmpTy::F32),
];

/// `probe_valu_p1(in, out)`: out[(k * 64 + lane)] = op_k(inputs of lane) as a raw dword (compares: 0 / 1).
pub fn probe_valu_p1() -> Function {
    let mut b = FnBuilder::new("probe_valu_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    let n_ops = (UN.len() + BIN.len() + TRI.len() + CMPS.len()) as u32;
    b.arg_tensor("in", amdgpu_dsl::Ext::Const((VALU_P1_INPUT_PLANES * 256) as i64));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const((n_ops * 256) as i64));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(c4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off) });
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off) });
    let ld = |b: &mut FnBuilder, k: u32| b.let_(Op::LoadF32 { base: Opnd::Val(pin), byte_off: 256 * k });  // 256 * 5 < 4096
    let (fa, fb, fc) = (ld(&mut b, 0), ld(&mut b, 1), ld(&mut b, 2));
    let ia = { let x = ld(&mut b, 3); b.let_(Op::BitcastI32(Opnd::Val(x))) };
    let ib = { let x = ld(&mut b, 4); b.let_(Op::BitcastI32(Opnd::Val(x))) };
    let ic = { let x = ld(&mut b, 5); b.let_(Op::BitcastI32(Opnd::Val(x))) };
    let mut k = 0u32;
    // one pointer per 16 ops: global offsets are 13-bit signed (at most 4095)
    let mut store = |b: &mut FnBuilder, v: ValueId, float: bool| {
        let v = if float { v } else { b.let_(Op::BitcastF32(Opnd::Val(v))) };
        let base = b.let_(Op::PtrAdd { base: Opnd::Val(pout), byte_off: Opnd::I32((256 * (k / 8 * 8)) as i32) });
        b.store_f32(base, 256 * (k % 8), v);
        k += 1;
    };
    for u in UN {
        let x = if u.float_in() { fa } else { ia };
        let r = b.un(u, x);
        store(&mut b, r, u.result_ty() == DType::F32);
    }
    for o in BIN {
        let (x, y) = if o == BinOp::MinF32 { (fa, fb) } else { (ia, ib) };
        let r = b.bin(o, x, y);
        store(&mut b, r, o.result_ty() == DType::F32);
    }
    for o in TRI {
        let (x, y, z) = if matches!(o, TriOp::Min3F32 | TriOp::Med3F32) { (fa, fb, fc) } else { (ia, ib, ic) };
        let r = b.tri(o, x, y, z);
        store(&mut b, r, o.result_ty() == DType::F32);
    }
    for (p, ty) in CMPS {
        let (x, y) = if ty == CmpTy::F32 { (fa, fb) } else { (ia, ib) };
        let c = b.cmp(p, ty, x, y);
        let r = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::I32(1), b: Opnd::I32(0) });
        store(&mut b, r, false);
    }
    b.finish()
}

/// Output dword planes of `probe_mem_p1` (64 lanes each) before the two 64 x 32-byte store regions.
pub const MEM_P1_PLANES: u32 = 64;

/// `probe_mem_p1(in, out)`: every P1 load width (global, buffer, LDS incl. d16 and read2) as raw dwords, then the P1 stores
/// (LDS -> read back; global and buffer -> byte regions of `out`). `in` is 4096 generated bytes; lane l works on bytes 16 l ..
pub fn probe_mem_p1() -> Function {
    use amdgpu_dsl::{D16Ty, MemTy};
    let mut b = FnBuilder::new("probe_mem_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(4096));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const((MEM_P1_PLANES * 256 + 2 * 64 * 32) as i64));
    b.lds(4096);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c16 = b.c_i32(16);
    let c4 = b.c_i32(4);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(c16)));
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(c4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off16) });
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    let mut k = 0u32;
    let mut put = |b: &mut FnBuilder, v: ValueId| {
        let f = b.let_(Op::BitcastF32(Opnd::Val(v)));
        let base = b.let_(Op::PtrAdd { base: Opnd::Val(pout), byte_off: Opnd::I32((256 * (k / 8 * 8)) as i32) });
        b.store_f32(base, 256 * (k % 8), f);
        k += 1;
    };
    let lanes = |b: &mut FnBuilder, v: ValueId, n: u8| -> Vec<ValueId> {
        (0..n).map(|i| match n {
            2 => b.let_(Op::Frag2Lane { frag: Opnd::Val(v), index: i }),
            _ => b.let_(Op::Frag4Lane { frag: Opnd::Val(v), index: i }),
        }).collect()
    };
    // ---- global loads (planes 0..19)
    for (ty, off) in [(MemTy::U8, 1), (MemTy::I8, 3), (MemTy::U16, 2), (MemTy::I16, 6), (MemTy::B32, 4)] {
        let v = b.load_global(pin, off, ty);
        put(&mut b, v);
    }
    let v = b.load_global(pin, 8, MemTy::B64);
    for x in lanes(&mut b, v, 2) { put(&mut b, x); }
    let v = b.load_global(pin, 0, MemTy::B96);
    for x in lanes(&mut b, v, 4).into_iter().take(3) { put(&mut b, x); }
    let v128 = b.load_global(pin, 0, MemTy::B128);
    for x in lanes(&mut b, v128, 4) { put(&mut b, x); }
    for (ty, hi, off) in [(D16Ty::U8, false, 1), (D16Ty::U8, true, 1), (D16Ty::I8, false, 3), (D16Ty::I8, true, 3), (D16Ty::B16, false, 2), (D16Ty::B16, true, 2)] {
        let v = b.load_global_d16(pin, off, ty, hi);
        put(&mut b, v);
    }
    // ---- buffer loads (planes 20..26)
    let rin = b.buffer_desc(inp, 4096);
    for (ty, imm) in [(MemTy::U8, 1), (MemTy::I8, 3), (MemTy::U16, 2), (MemTy::I16, 6)] {
        let v = b.buffer_load_n(rin, Some(Opnd::Val(off16)), 0, imm, ty);
        put(&mut b, v);
    }
    let v = b.buffer_load_n(rin, Some(Opnd::Val(off16)), 0, 0, MemTy::B96);
    for x in lanes(&mut b, v, 4).into_iter().take(3) { put(&mut b, x); }
    // ---- LDS: region 0 = the lane's 16 input bytes; regions 1..3 zeroed
    let zero = b.c_i32(0);
    let z4 = b.frag4(zero, zero, zero, zero);
    b.store_lds_b128(off16, v128);
    for r in 1..4 {
        let a = b.add_i32(off16, 1024 * r);
        b.store_lds_b128(a, z4);
    }
    b.wait_lds(0);
    b.barrier();
    // planes 27..51
    for (ty, off) in [(MemTy::U8, 1), (MemTy::I8, 3), (MemTy::U16, 2), (MemTy::I16, 6)] {
        let a = b.add_i32(off16, off);
        let v = b.load_lds(a, ty);
        put(&mut b, v);
    }
    let v = b.load_lds(off16, MemTy::B96);
    for x in lanes(&mut b, v, 4).into_iter().take(3) { put(&mut b, x); }
    for (ty, hi, off) in [(D16Ty::U8, false, 1), (D16Ty::U8, true, 1), (D16Ty::I8, false, 3), (D16Ty::I8, true, 3), (D16Ty::B16, false, 2), (D16Ty::B16, true, 2)] {
        let a = b.add_i32(off16, off);
        let v = b.load_lds_d16(a, ty, hi);
        put(&mut b, v);
    }
    let v = b.load_lds2(off16, 1, 3, false, false);
    for x in lanes(&mut b, v, 2) { put(&mut b, x); }
    let v = b.load_lds2(off16, 0, 1, true, false);
    for x in lanes(&mut b, v, 4) { put(&mut b, x); }
    let v = b.load_lds2(off4, 0, 1, false, true);
    for x in lanes(&mut b, v, 2) { put(&mut b, x); }
    let off8 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(8)));
    let v = b.load_lds2(off8, 0, 1, true, true);
    for x in lanes(&mut b, v, 4) { put(&mut b, x); }
    // ---- LDS stores (regions 1..3), read back as planes 52..63
    let x = b.let_(Op::Frag4Lane { frag: Opnd::Val(v128), index: 0 });
    let y = b.let_(Op::Frag4Lane { frag: Opnd::Val(v128), index: 1 });
    let r1 = b.add_i32(off16, 1024);
    b.store_lds(r1, x, MemTy::U8, false);
    let a = b.add_i32(off16, 1025);
    b.store_lds(a, x, MemTy::U8, true);
    let a = b.add_i32(off16, 1026);
    b.store_lds(a, y, MemTy::U16, false);
    let a = b.add_i32(off16, 1028);
    b.store_lds(a, y, MemTy::U16, true);
    let r2 = b.add_i32(off16, 2048);
    b.store_lds(r2, v128, MemTy::B96, false);
    let r3 = b.add_i32(off16, 3072);
    b.store_lds2(r3, x, y, 0, 2, false, false);
    b.wait_lds(0);
    for r in [r1, r2, r3] {
        let v = b.load_lds_b128(r);
        for x in lanes(&mut b, v, 4) { put(&mut b, x); }
    }
    // ---- global / buffer stores into the byte regions (lane l: 32 bytes each)
    let c32 = b.c_i32(32);
    let off32 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(c32)));
    let g0 = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::I32((MEM_P1_PLANES * 256) as i32) });
    let g = b.let_(Op::PtrAdd { base: Opnd::Val(g0), byte_off: Opnd::Val(off32) });
    b.store_global(g, 0, x, MemTy::U8, false);
    b.store_global(g, 1, x, MemTy::U8, true);
    b.store_global(g, 2, y, MemTy::U16, false);
    b.store_global(g, 4, y, MemTy::U16, true);
    b.store_global(g, 16, v128, MemTy::B96, false);
    let rout = b.buffer_desc(out, MEM_P1_PLANES * 256 + 2 * 64 * 32);
    let bo = b.add_i32(off32, (MEM_P1_PLANES * 256 + 64 * 32) as i32);
    b.buffer_store_n(rout, Some(Opnd::Val(bo)), 0, 0, x, MemTy::U8, false);
    b.buffer_store_n(rout, Some(Opnd::Val(bo)), 0, 1, x, MemTy::U8, true);
    b.buffer_store_n(rout, Some(Opnd::Val(bo)), 0, 2, y, MemTy::U16, false);
    b.buffer_store_n(rout, Some(Opnd::Val(bo)), 0, 4, y, MemTy::U16, true);
    b.buffer_store_n(rout, Some(Opnd::Val(bo)), 0, 16, v128, MemTy::B96, false);
    assert_eq!(k, MEM_P1_PLANES);
    b.finish()
}


/// MFMA kinds of `probe_mfma_p1`, in input / output order.
pub fn mfma_p1_kinds() -> Vec<amdgpu_dsl::MmaKind> {
    use amdgpu_dsl::{MmaKind, MmaShape as S, MmaTy as T};
    vec![
        MmaKind::new(S::M16N16K32, T::Bf16),
        MmaKind::new(S::M32N32K16, T::Bf16),
        MmaKind::new(S::M16N16K32, T::F16),
        MmaKind::new(S::M32N32K16, T::F16),
        MmaKind::new(S::M16N16K64, T::I8),
        MmaKind::new(S::M32N32K32, T::I8),
    ]
}

/// `probe_mfma_p1(in, out)`: for every kind, A and B fragments (16 bytes per lane each, lane-major) from `in`; `out` gets per kind
/// D = A x B (zero-C form), D2 = A x B + D (C form) and D3 (loop-variable form: var = A x B, then var += A x B), each lane-major.
pub fn probe_mfma_p1() -> Function {
    let kinds = mfma_p1_kinds();
    let mut b = FnBuilder::new("probe_mfma_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    let out_bytes: u32 = kinds.iter().map(|k| 3 * 64 * 4 * k.acc_dwords()).sum::<u32>() + 64 * 16;
    b.arg_tensor("in", amdgpu_dsl::Ext::Const((kinds.len() * 2048) as i64));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(out_bytes as i64));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c16 = b.c_i32(16);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(c16)));
    let mut obase = 0u32;
    for (i, k) in kinds.iter().enumerate() {
        let pa = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::I32((i * 2048) as i32) });
        let pa = b.let_(Op::PtrAdd { base: Opnd::Val(pa), byte_off: Opnd::Val(off16) });
        let fa = b.load_global(pa, 0, amdgpu_dsl::MemTy::B128);
        let fb = b.load_global(pa, 1024, amdgpu_dsl::MemTy::B128);
        let d1 = b.mfma(*k, fa, fb, None);
        let d2 = b.mfma(*k, fa, fb, Some(Opnd::Val(d1)));
        let vars = b.begin_loop(&[(k.acc_ty(), Opnd::Val(d1))]);
        b.pin_agpr(vars[0]); // the var form accumulates in AGPRs (acc_cd)
        b.var_mfma(vars[0], *k, fa, fb, true);
        b.var_mfma(vars[0], *k, fa, fb, false);
        let always = b.cmp(CmpPred::Ge, CmpTy::I32, 0, 0); // a uniform (constant) exit: one trip
        b.break_if(always);
        b.end_loop();
        let n = k.acc_dwords();
        let lane_off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32((4 * n) as i32)));
        for (j, v) in [Opnd::Val(d1), Opnd::Val(d2), Opnd::Var(vars[0])].into_iter().enumerate() {
            let p = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::I32((obase + j as u32 * 64 * 4 * n) as i32) });
            let p = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(lane_off) });
            for e in 0..n as u8 {
                let x = if n == 4 { b.frag4_lane(v, e) } else { b.frag16_lane(v, e) };
                b.store_global(p, 4 * e as u32, x, amdgpu_dsl::MemTy::B32, false);
            }
        }
        obase += 3 * 64 * 4 * n;
        if i == 0 {
            // `var_slice4` views of a Frag8 loop variable holding A ++ B: the same product without copies (extra block at the end of `out`)
            let ab = b.cat_frag8(fa, fb);
            let vs = b.begin_loop(&[(DType::Frag8, Opnd::Val(ab))]);
            let s0 = b.var_slice4(vs[0], 0);
            let s1 = b.var_slice4(vs[0], 1);
            let d4 = b.mfma(*k, s0, s1, None);
            let always = b.cmp(CmpPred::Ge, CmpTy::I32, 0, 0); // a uniform (constant) exit: one trip
            b.break_if(always);
            b.end_loop();
            let p = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::I32(out_bytes as i32 - 64 * 16) });
            let p = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off16) });
            b.store_global(p, 0, d4, amdgpu_dsl::MemTy::B128, false);
        }
    }
    b.finish()
}

/// (shape, A type, B type) combinations of `probe_f8f6f4_p1`, in input / output order.
pub fn f8f6f4_p1_kinds() -> Vec<amdgpu_dsl::MmaKind> {
    use amdgpu_dsl::{MmaKind, MmaShape as S, MmaTy as T};
    let mut v = Vec::new();
    for (a, b) in [(T::Fp4, T::Fp4), (T::Fp8, T::Fp4), (T::Fp4, T::Fp8), (T::Fp8, T::Fp8), (T::Bf8, T::Bf8), (T::Fp6, T::Fp6), (T::Bf6, T::Bf6), (T::Bf8, T::Fp6), (T::Bf6, T::Fp8), (T::Fp4, T::Bf8)] {
        v.push(MmaKind::scaled_f8f6f4(S::M16N16K128, a, b));
    }
    for (a, b) in [(T::Fp4, T::Fp4), (T::Fp8, T::Fp4), (T::Fp6, T::Bf8)] {
        v.push(MmaKind::scaled_f8f6f4(S::M32N32K64, a, b));
    }
    v
}

/// `probe_f8f6f4_p1(in, out)`: per kind, A and B (32 bytes per lane each, lane-major; fp4 uses the first 16, fp6 the first 24) with unit
/// E8M0 scales (0x7F); `out` gets D = A x B (zero-C) and D2 = A x B + D, lane-major.
/// The unscaled kinds of `probe_f8f6f4_plain` (`MmaKind::f8f6f4`, no scale operands).
pub fn f8f6f4_plain_kinds() -> Vec<amdgpu_dsl::MmaKind> {
    use amdgpu_dsl::{MmaKind, MmaShape as S, MmaTy as T};
    vec![
        MmaKind::f8f6f4(S::M16N16K128, T::Fp4, T::Fp4),
        MmaKind::f8f6f4(S::M16N16K128, T::Fp8, T::Fp4),
        MmaKind::f8f6f4(S::M16N16K128, T::Fp8, T::Fp8),
        MmaKind::f8f6f4(S::M16N16K128, T::Fp6, T::Bf8),
        MmaKind::f8f6f4(S::M32N32K64, T::Fp4, T::Fp4),
        MmaKind::f8f6f4(S::M32N32K64, T::Bf8, T::Fp8),
    ]
}

/// `probe_f8f6f4_plain(in, out)`: as `probe_f8f6f4_p1` for the unscaled kinds (D = A x B, D2 = A x B + D).
pub fn probe_f8f6f4_plain() -> Function {
    f8f6f4_probe("probe_f8f6f4_plain", f8f6f4_plain_kinds())
}

pub fn probe_f8f6f4_p1() -> Function {
    f8f6f4_probe("probe_f8f6f4_p1", f8f6f4_p1_kinds())
}

fn f8f6f4_probe(name: &str, kinds: Vec<amdgpu_dsl::MmaKind>) -> Function {
    let mut b = FnBuilder::new(name, 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    let out_bytes: u32 = kinds.iter().map(|k| 2 * 64 * 4 * k.acc_dwords()).sum();
    b.arg_tensor("in", amdgpu_dsl::Ext::Const((kinds.len() * 4096) as i64));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(out_bytes as i64));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off32 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(32)));
    let one = b.c_i32(0x7F7F7F7F);
    let s = amdgpu_dsl::MmaScale { sa: Opnd::Val(one), sb: Opnd::Val(one), opsel: 0 };
    let mut obase = 0u32;
    for (i, k) in kinds.iter().enumerate() {
        let p = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::I32((i * 4096) as i32) });
        let p = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off32) });
        let mut opnd = |b: &mut FnBuilder, off: u32, t: amdgpu_dsl::MmaTy| {
            let lo = b.load_global(p, off, amdgpu_dsl::MemTy::B128);
            if k.operand_dwords(t) == 4 {
                lo
            } else {
                let hi = b.load_global(p, off + 16, amdgpu_dsl::MemTy::B128);
                b.cat_frag8(lo, hi)
            }
        };
        let fa = opnd(&mut b, 0, k.a);
        let fb = opnd(&mut b, 2048, k.b);
        let (d1, d2) = if k.scaled {
            let d1 = b.mfma_scaled(*k, fa, fb, None, s);
            (d1, b.mfma_scaled(*k, fa, fb, Some(Opnd::Val(d1)), s))
        } else {
            let d1 = b.mfma(*k, fa, fb, None);
            (d1, b.mfma(*k, fa, fb, Some(Opnd::Val(d1))))
        };
        let n = k.acc_dwords();
        let lane_off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32((4 * n) as i32)));
        for (j, v) in [d1, d2].into_iter().enumerate() {
            let q = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::I32((obase + j as u32 * 64 * 4 * n) as i32) });
            let q = b.let_(Op::PtrAdd { base: Opnd::Val(q), byte_off: Opnd::Val(lane_off) });
            for e in 0..n as u8 {
                let x = if n == 4 { b.frag4_lane(v, e) } else { b.frag16_lane(v, e) };
                b.store_global(q, 4 * e as u32, x, amdgpu_dsl::MemTy::B32, false);
            }
        }
        obase += 2 * 64 * 4 * n;
    }
    b.finish()
}

/// `probe_cpol_p1(in, out)`: cache-policy modifiers on global / buffer loads and stores (sc0, sc1, nt): out = in through four
/// differently-flagged copy paths, lane l copying 16 bytes per path.
pub fn probe_cpol_p1() -> Function {
    use amdgpu_dsl::{CachePolicy as C, MemTy};
    let mut b = FnBuilder::new("probe_cpol_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(4096));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(4096));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off16) });
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off16) });
    let all = C { sc0: true, sc1: true, nt: true };
    let sc0 = C { sc0: true, ..C::NONE };
    let sc1 = C { sc1: true, ..C::NONE };
    // path 0: global nt load -> global sc0 sc1 nt store
    let v = b.load_global_c(pin, 0, MemTy::B128, C::NT);
    b.store_global_c(pout, 0, v, MemTy::B128, false, all);
    // path 1: global sc1 dword loads -> global sc0 stores (bytes 1024..)
    for k in 0..4u32 {
        let x = b.load_global_c(pin, 1024 + 4 * k, MemTy::B32, sc1);
        b.store_global_c(pout, 1024 + 4 * k, x, MemTy::B32, false, sc0);
    }
    // paths 2, 3: buffer loads / stores with policies (bytes 2048.., 3072..)
    let rin = b.buffer_desc(inp, 4096);
    let rout = b.buffer_desc(out, 4096);
    let v = b.buffer_load_c(rin, Some(Opnd::Val(off16)), 0, 2048, MemTy::B128, sc0);
    b.buffer_store_c(rout, Some(Opnd::Val(off16)), 0, 2048, v, MemTy::B128, false, C::NT);
    let v = b.buffer_load_c(rin, Some(Opnd::Val(off16)), 0, 3072, MemTy::B96, all);
    b.buffer_store_c(rout, Some(Opnd::Val(off16)), 0, 3072, v, MemTy::B96, false, sc1);
    let w = b.buffer_load_c(rin, Some(Opnd::Val(off16)), 0, 3084, MemTy::B32, C::NT);
    b.buffer_store_c(rout, Some(Opnd::Val(off16)), 0, 3084, w, MemTy::B32, false, all);
    b.finish()
}

/// Conversions of `probe_cvt_p1`: (op, input planes of the used operands, sel). Input planes (64 lanes each): 0 fp8 / bf8 bytes, 1 fp4 nibbles,
/// 2 f32 a, 3 f32 b, 4 f32 c (u8 range), 5 i32 i, 6 i32 j, 7 scale (2.0), 8 old (0x5A5A5A5A), 9 scale 1.63, 10 scale 3.0.
pub fn cvt_p1_cases() -> Vec<(amdgpu_dsl::CvtOp, Vec<usize>, u8)> {
    use amdgpu_dsl::CvtOp as C;
    vec![
        (C::PkF32Fp8, vec![0], 0),
        (C::PkF32Fp8, vec![0], 1),
        (C::PkF32Bf8, vec![0], 1),
        (C::F32Fp8, vec![0], 2),
        (C::F32Bf8, vec![0], 3),
        (C::PkBf8F32, vec![2, 3, 8], 0),
        (C::PkBf8F32, vec![2, 3, 8], 1),
        (C::ScalePkBf16Fp8, vec![0, 7], 1),
        (C::ScalePkF16Fp8, vec![0, 7], 0),
        (C::ScalePkBf16Bf8, vec![0, 7], 1),
        (C::ScalePkF16Bf8, vec![0, 7], 0),
        (C::ScalePkF32Fp8, vec![0, 7], 1),
        (C::ScalePkF32Bf8, vec![0, 7], 0),
        (C::ScalePkF32Fp4, vec![1, 7], 2),
        (C::ScalePkBf16Fp4, vec![1, 7], 3),
        (C::ScalePkF16Fp4, vec![1, 7], 1),
        (C::ScalePkFp8F32, vec![2, 3, 7, 8], 1),
        (C::ScalePkBf8F32, vec![2, 3, 7, 8], 0),
        (C::ScalePkFp4F32, vec![2, 3, 7, 8], 2),
        // non-power-of-two scales (plane 9: 1.63, plane 10: 3.0): only the exponent counts (E8M0)
        (C::ScalePkF32Fp8, vec![0, 9], 0),
        (C::ScalePkBf16Fp4, vec![1, 10], 1),
        (C::ScalePkFp8F32, vec![2, 3, 10, 8], 0),
        // every destination byte of the fp4 pack (kernel-opt: sel 1..3)
        (C::ScalePkFp4F32, vec![2, 3, 7, 8], 0),
        (C::ScalePkFp4F32, vec![2, 3, 7, 8], 1),
        (C::ScalePkFp4F32, vec![2, 3, 7, 8], 3),
        (C::PkU8F32, vec![4, 8], 2),
        (C::PkI16I32, vec![5, 6], 0),
        (C::PkU16U32, vec![5, 6], 0),
    ]
}

/// `probe_cvt_p1(in, out)`: every case of `cvt_p1_cases` per lane; results (1 or 2 dwords) go to consecutive 64-lane output planes.
pub fn probe_cvt_p1() -> Function {
    let cases = cvt_p1_cases();
    let planes: u32 = cases.iter().map(|(op, _, _)| if op.sig().1 == DType::Frag2 { 2 } else { 1 }).sum();
    let mut b = FnBuilder::new("probe_cvt_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(11 * 256));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const((planes * 256) as i64));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off4) });
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    let ins: Vec<ValueId> = (0..11u32).map(|k| b.load_global(pin, 256 * k, amdgpu_dsl::MemTy::B32)).collect();
    let mut k = 0u32;
    let mut put = |b: &mut FnBuilder, v: ValueId| {
        let base = b.let_(Op::PtrAdd { base: Opnd::Val(pout), byte_off: Opnd::I32((256 * (k / 8 * 8)) as i32) });
        b.store_global(base, 256 * (k % 8), v, amdgpu_dsl::MemTy::B32, false);
        k += 1;
    };
    for (op, srcs, sel) in cases {
        let ops: Vec<Opnd> = srcs.iter().map(|p| Opnd::Val(ins[*p])).collect();
        let r = b.cvt(op, &ops, sel);
        if op.sig().1 == DType::Frag2 {
            for i in 0..2 {
                let x = b.let_(Op::Frag2Lane { frag: Opnd::Val(r), index: i });
                put(&mut b, x);
            }
        } else {
            put(&mut b, r);
        }
    }
    b.finish()
}

/// DPP cases of `probe_dpp_p1`: (name, ctrl, row_mask, bank_mask, bound_zero, ALU op or None for v_mov_b32_dpp).
pub fn dpp_p1_cases() -> Vec<(amdgpu_dsl::Dpp, Option<amdgpu_dsl::DppAluOp>)> {
    use amdgpu_dsl::{Dpp, DppAluOp as A, DppCtrl as C};
    let d = Dpp::new;
    let mut v: Vec<(Dpp, Option<A>)> = [
        C::QuadPerm([3, 2, 1, 0]),
        C::RowShl(1),
        C::RowShr(3),
        C::RowRor(5),
        C::WaveShl1,
        C::WaveRol1,
        C::WaveShr1,
        C::WaveRor1,
        C::RowMirror,
        C::RowHalfMirror,
        C::RowBcast15,
        C::RowBcast31,
    ]
    .into_iter()
    .map(|c| (d(c), None))
    .collect();
    v.push((Dpp { bound_zero: true, ..d(C::RowShr(1)) }, None));
    v.push((Dpp { row_mask: 0xa, ..d(C::RowBcast15) }, None));
    v.push((Dpp { bank_mask: 0x5, ..d(C::RowShr(2)) }, None));
    v.push((d(C::RowShr(1)), Some(A::AddU32)));
    v.push((Dpp { row_mask: 0xc, ..d(C::RowBcast31) }, Some(A::MaxI32)));
    v.push((Dpp { bound_zero: true, ..d(C::RowShr(4)) }, Some(A::AddF32)));
    v.push((d(C::QuadPerm([1, 0, 3, 2])), Some(A::MinU32)));
    v.push((d(C::RowHalfMirror), Some(A::XorB32)));
    v
}

/// `probe_dpp_p1(in, out)`: every case of `dpp_p1_cases` on per-lane x (plane 0), y (plane 1), floats fx (plane 2), fy (plane 3), with
/// old = 0xDEAD0000 + lane; then permlane16_swap(x, y) (2 planes).
pub fn probe_dpp_p1() -> Function {
    let cases = dpp_p1_cases();
    let mut b = FnBuilder::new("probe_dpp_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(4 * 256));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(((cases.len() + 2) * 256) as i64));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off4) });
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    let x = b.load_global(pin, 0, amdgpu_dsl::MemTy::B32);
    let y = b.load_global(pin, 256, amdgpu_dsl::MemTy::B32);
    let fx = { let t = b.load_global(pin, 512, amdgpu_dsl::MemTy::B32); b.let_(Op::BitcastF32(Opnd::Val(t))) };
    let fy = { let t = b.load_global(pin, 768, amdgpu_dsl::MemTy::B32); b.let_(Op::BitcastF32(Opnd::Val(t))) };
    let old = b.add_i32(tid, 0xDEAD0000u32 as i32);
    let mut k = 0u32;
    let mut put = |b: &mut FnBuilder, v: ValueId| {
        let base = b.let_(Op::PtrAdd { base: Opnd::Val(pout), byte_off: Opnd::I32((256 * (k / 8 * 8)) as i32) });
        b.store_global(base, 256 * (k % 8), v, amdgpu_dsl::MemTy::B32, false);
        k += 1;
    };
    for (dpp, op) in cases {
        let r = match op {
            None => b.dpp(dpp, x, old),
            Some(o @ amdgpu_dsl::DppAluOp::AddF32) => {
                let of = b.let_(Op::BitcastF32(Opnd::Val(old)));
                let r = b.dpp_alu(o, dpp, fx, fy, of);
                b.let_(Op::BitcastI32(Opnd::Val(r)))
            }
            Some(o) => b.dpp_alu(o, dpp, x, y, old),
        };
        put(&mut b, r);
    }
    let s = b.permlane16_swap(x, y);
    for i in 0..2 {
        let v = b.let_(Op::Frag2Lane { frag: Opnd::Val(s), index: i });
        put(&mut b, v);
    }
    b.finish()
}

/// `probe_tr_b16_p1(in, out)`: LDS holds u16 i at byte 2 i (i < 2048, from `in`); lane l reads `ds_read_b64_tr_b16` at byte 8 l (+ 512 for the
/// second read); out = the 4 returned 16-bit values per lane and read (2 dwords each), so value v names (source lane v / 4, element v % 4).
pub fn probe_tr_b16_p1() -> Function {
    let mut b = FnBuilder::new("probe_tr_b16_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(4096));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(1024));
    b.lds(4096);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off16) });
    for k in 0..4u32 {
        let v = b.load_global(pin, 1024 * k, amdgpu_dsl::MemTy::B128);
        let a = b.add_i32(off16, (1024 * k) as i32);
        b.store_lds_b128(a, v);
    }
    b.wait_lds(0);
    b.barrier();
    let off8 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(8)));
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off8) });
    for (j, extra) in [0i32, 512].into_iter().enumerate() {
        let a = b.add_i32(off8, extra);
        let r = b.load_lds_tr_b16(a);
        b.store_global(pout, 512 * j as u32, r, amdgpu_dsl::MemTy::B64, false);
    }
    b.finish()
}

/// `probe_tr_b16_mfma(in, out)`: the bf16 16x16x32 A operand built from `ds_read_b64_tr_b16` of a row-major LDS tile T[k][m] (32 x 16 bf16,
/// bytes 0..1024 of `in`), B lane-major fragments from bytes 1024..2048 (lane l: column l % 16, k = 8 (l / 16) + 0..7); out = D (lane-major, 4 dwords).
pub fn probe_tr_b16_mfma() -> Function {
    use amdgpu_dsl::{MemTy, MmaKind, MmaShape, MmaTy};
    let mut b = FnBuilder::new("probe_tr_b16_mfma", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(2048));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(1024));
    b.lds(1024);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off16) });
    let t = b.load_global(pin, 0, MemTy::B128);
    b.store_lds_b128(off16, t);
    let fb = b.load_global(pin, 1024, MemTy::B128);
    b.wait_lds(0);
    b.barrier();
    // lane l = 16 G + s reads &T[8 G + 4 r + s / 4][4 (s % 4)]  (row = 16 elements = 32 bytes)
    let s = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(15)));
    let g = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(4)));
    let q = b.let_(Op::LshrI32(Opnd::Val(s), Opnd::I32(2)));
    let c = b.let_(Op::AndI32(Opnd::Val(s), Opnd::I32(3)));
    let row = b.tri(amdgpu_dsl::TriOp::LshlAddU32, g, 3, q); // 8 G + q
    let rb = b.let_(Op::MulI32(Opnd::Val(row), Opnd::I32(32)));
    let addr = b.tri(amdgpu_dsl::TriOp::LshlAddU32, c, 3, rb); // + 8 c bytes
    let r0 = b.load_lds_tr_b16(addr);
    let a1 = b.add_i32(addr, 128); // + 4 rows
    let r1 = b.load_lds_tr_b16(a1);
    let parts: Vec<ValueId> = [(r0, 0u8), (r0, 1), (r1, 0), (r1, 1)].iter().map(|(v, i)| b.let_(Op::Frag2Lane { frag: Opnd::Val(*v), index: *i })).collect();
    let fa = b.frag4(parts[0], parts[1], parts[2], parts[3]);
    let d = b.mfma(MmaKind::new(MmaShape::M16N16K32, MmaTy::Bf16), fa, fb, None);
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off16) });
    b.store_global(pout, 0, d, MemTy::B128, false);
    b.finish()
}

/// `probe_atomic_p1(in, out)`: LDS and global atomics from all 64 lanes. in: plane 0 u32 x, plane 1 f32 y (small integers). Counters (dword index
/// c): add c = l % 4 (+1, returning) at 0..3, add c = 4 + l % 4 (+lane, no return), max_u32 c = 8 + l % 2 (x), min_i32 c = 10 + l % 2 (x,
/// returning, init 0x7fffffff), inc c = 12 (limit 15, returning), add_f32 c = 13 (y). out: planes 0..2 LDS returns (add, min, inc), plane 3 the
/// final LDS counters (lane l reads dword l), planes 4..5 global returns (add, inc), then 64 global counter dwords (same layout).
pub fn probe_atomic_p1() -> Function {
    use amdgpu_dsl::{AtomicOp as A, MemTy};
    let mut b = FnBuilder::new("probe_atomic_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(512));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(7 * 256));
    b.lds(256);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off4) });
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    let x = b.load_global(pin, 0, MemTy::B32);
    let y = { let t = b.load_global(pin, 256, MemTy::B32); b.let_(Op::BitcastF32(Opnd::Val(t))) };
    // init: dwords 10, 11 = 0x7fffffff, all others 0 (LDS and the global counter block)
    let half = b_shr1(&mut b, tid);
    let is_min = b.cmp(CmpPred::Eq, CmpTy::I32, half, 5);
    let init = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is_min), a: Opnd::I32(0x7fffffff), b: Opnd::I32(0) });
    b.store_lds(off4, init, MemTy::B32, false);
    let gctr = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::I32(6 * 256) });
    let gl = b.let_(Op::PtrAdd { base: Opnd::Val(gctr), byte_off: Opnd::Val(off4) });
    b.store_global(gl, 0, init, MemTy::B32, false);
    b.wait_lds(0);
    b.wait_vm(0);
    b.barrier();
    let l4 = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(3)));
    let l2 = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(1)));
    let a_add = b.let_(Op::MulI32(Opnd::Val(l4), Opnd::I32(4)));
    let a_add2 = b.add_i32(a_add, 16);
    let a2 = b.let_(Op::MulI32(Opnd::Val(l2), Opnd::I32(4)));
    let a_max = b.add_i32(a2, 32);
    let a_min = b.add_i32(a2, 40);
    let one = b.c_i32(1);
    let r_add = b.atomic_lds(A::AddU32, a_add, one, true).unwrap();
    b.atomic_lds(A::AddU32, a_add2, tid, false);
    b.atomic_lds(A::MaxU32, a_max, x, false);
    let r_min = b.atomic_lds(A::MinI32, a_min, x, true).unwrap();
    let r_inc = b.atomic_lds(A::IncU32, 48, 15, true).unwrap();
    b.atomic_lds(A::AddF32, 52, y, false);
    // global: the same counters in the block at out + 6 * 256
    let g = |b: &mut FnBuilder, a: ValueId| b.let_(Op::PtrAdd { base: Opnd::Val(gctr), byte_off: Opnd::Val(a) });
    let (ga, ga2, gmx, gmn) = (g(&mut b, a_add), g(&mut b, a_add2), g(&mut b, a_max), g(&mut b, a_min));
    let rg_add = b.atomic_global(A::AddU32, ga, 0, one, true).unwrap();
    b.atomic_global(A::AddU32, ga2, 0, tid, false);
    b.atomic_global(A::MaxU32, gmx, 0, x, false);
    b.atomic_global(A::MinI32, gmn, 0, x, false);
    let rg_inc = b.atomic_global(A::IncU32, gctr, 48, 15, true).unwrap();
    b.atomic_global(A::AddF32, gctr, 52, y, false);
    b.wait_lds(0);
    b.barrier();
    let fin = b.load_lds(off4, MemTy::B32);
    let mut k = 0u32;
    for v in [r_add, r_min, r_inc, fin, rg_add, rg_inc] {
        let v = if b.f.value_tys[v as usize] == DType::F32 { b.let_(Op::BitcastI32(Opnd::Val(v))) } else { v };
        let base = b.let_(Op::PtrAdd { base: Opnd::Val(pout), byte_off: Opnd::I32((256 * k) as i32) });
        b.store_global(base, 0, v, MemTy::B32, false);
        k += 1;
    }
    b.finish()
}

fn b_shr1(b: &mut FnBuilder, x: ValueId) -> ValueId {
    b.let_(Op::LshrI32(Opnd::Val(x), Opnd::I32(1)))
}

/// `probe_lane_p1(in, out)`: cross-lane / wave-mask ops on x (plane 0 of `in`); out planes: 0 read_lane(x, 5), 1 read_lane(x, 63), 2 write_lane(read_lane(x, 7),
/// 3, old = lane), 3 / 4 ballot(x & 1) lo / hi, 5 mbcnt(-1, -1), 6 mbcnt(ballot), 7 popcount(x), 8 ffbl(x), 9 ffbh(x), 10 bpermute from lane 7 l % 64,
/// 11 permute to lane l + 5, 12 swizzle bit-mask xor 1, 13 swizzle quad_perm [3,2,1,0].
pub fn probe_lane_p1() -> Function {
    use amdgpu_dsl::{BinOp, MemTy, UnOp};
    let mut b = FnBuilder::new("probe_lane_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(256));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(14 * 256));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off4) });
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    let x = b.load_global(pin, 0, MemTy::B32);
    let mut vals = Vec::new();
    vals.push(b.read_lane(x, 5));
    vals.push(b.read_lane(x, 63));
    let x7 = b.read_lane(x, 7);
    vals.push(b.write_lane(x7, 3, tid));
    let bit = b.let_(Op::AndI32(Opnd::Val(x), Opnd::I32(1)));
    let cond = b.cmp(CmpPred::Ne, CmpTy::I32, bit, 0);
    let (lo, hi) = b.ballot(cond);
    vals.push(lo);
    vals.push(hi);
    vals.push(b.mbcnt(-1, -1));
    vals.push(b.mbcnt(lo, hi));
    vals.push(b.bin(BinOp::BcntAdd, x, 0));
    vals.push(b.un(UnOp::Ffbl, x));
    vals.push(b.un(UnOp::Ffbh, x));
    let src = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(7)));
    let src = b.let_(Op::AndI32(Opnd::Val(src), Opnd::I32(63)));
    let a1 = b.let_(Op::MulI32(Opnd::Val(src), Opnd::I32(4)));
    vals.push(b.bpermute(a1, x));
    let dstl = b.add_i32(tid, 5);
    let dstl = b.let_(Op::AndI32(Opnd::Val(dstl), Opnd::I32(63)));
    let a2 = b.let_(Op::MulI32(Opnd::Val(dstl), Opnd::I32(4)));
    vals.push(b.permute(a2, x));
    vals.push(b.swizzle(x, 0x1f | (1 << 10)));
    vals.push(b.swizzle(x, 0x8000 | 0x1b));
    for (k, v) in vals.into_iter().enumerate() {
        let base = b.let_(Op::PtrAdd { base: Opnd::Val(pout), byte_off: Opnd::I32((256 * (k / 8 * 8)) as i32) });
        b.store_global(base, 256 * (k as u32 % 8), v, MemTy::B32, false);
    }
    b.finish()
}

/// `probe_tileop_p1(in, out)` (own backend: DPP): the tileop builders on x, y = 4 f32 per lane (planes 0..3 / 4..7 of `in`). out per lane (15
/// dwords, lane-major): ewise(x, y: 2 x + y) [4], apply2 AddF32 (packed) [4], apply2 MaxF32 [4], row_allreduce16 AddF32 (x0) [1],
/// wave_allreduce MaxF32 (x1) [1], wave_allreduce AddU32 (lane id) [1].
pub fn probe_tileop_p1() -> Function {
    use amdgpu_dsl::{DppAluOp, MemTy};
    use amdgpu_tile::tileop::{apply2, ewise, row_allreduce16, wave_allreduce, Ew2, RegTile};
    let mut b = FnBuilder::new("probe_tileop_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(2048));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(64 * 60));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off16) });
    let fx = b.load_global(pin, 0, MemTy::B128);
    let fy = b.load_global(pin, 1024, MemTy::B128);
    let x = RegTile::new(fx, 4, DType::F32);
    let y = RegTile::new(fy, 4, DType::F32);
    let e = ewise(&mut b, &[&x, &y], DType::F32, |b, v| b.let_(Op::FmaF32(Opnd::F32(2.0), Opnd::Val(v[0]), Opnd::Val(v[1]))));
    let s = apply2(&mut b, Ew2::AddF32, &x, &y);
    let m = apply2(&mut b, Ew2::MaxF32, &x, &y);
    let x0 = x.elem(&mut b, 0);
    let x1 = x.elem(&mut b, 1);
    let r = row_allreduce16(&mut b, DppAluOp::AddF32, x0);
    let w = wave_allreduce(&mut b, DppAluOp::MaxF32, x1);
    let c = wave_allreduce(&mut b, DppAluOp::AddU32, tid);
    let off60 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(60)));
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off60) });
    for (k, t) in [e, s, m].iter().enumerate() {
        b.store_global(pout, 16 * k as u32, t.frag, MemTy::B128, false);
    }
    for (k, v) in [r, w, c].into_iter().enumerate() {
        let v = if b.f.value_tys[v as usize] == DType::F32 { b.let_(Op::BitcastI32(Opnd::Val(v))) } else { v };
        b.store_global(pout, 48 + 4 * k as u32, v, MemTy::B32, false);
    }
    b.finish()
}

/// `probe_atomic2_p1(in, out)`: buffer atomics (no return) and packed 16-bit global atomics from 64 lanes. in: plane 0 f32 y, plane 1 packed bf16
/// pair p, plane 2 packed f16 pair h (small integers). out dwords (counter c = l % 4): 0..3 buffer add_f32 (y), 4..7 buffer pk_add_bf16 (p),
/// 8..11 buffer pk_add_f16 (h), 12..15 buffer add_u32 (lane), 16..19 global pk_add_bf16 (p, no return), 20..23 global pk_add_f16 (h,
/// returning; the returns go to plane 1 of out: dwords 64..127).
pub fn probe_atomic2_p1() -> Function {
    use amdgpu_dsl::{AtomicOp as A, MemTy};
    let mut b = FnBuilder::new("probe_atomic2_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(768));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(512));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off4) });
    let y = { let t = b.load_global(pin, 0, MemTy::B32); b.let_(Op::BitcastF32(Opnd::Val(t))) };
    let p = b.load_global(pin, 256, MemTy::B32);
    let h = b.load_global(pin, 512, MemTy::B32);
    // zero the 24 counters (lanes 0..23) and wait
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    b.store_global(pout, 0, 0, MemTy::B32, false);
    b.wait_vm(0);
    b.barrier();
    let c4 = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(3)));
    let co = b.let_(Op::MulI32(Opnd::Val(c4), Opnd::I32(4)));
    let r = b.buffer_desc(out, 512);
    b.buffer_atomic(A::AddF32, r, Some(Opnd::Val(co)), 0, 0, y);
    b.buffer_atomic(A::PkAddBf16, r, Some(Opnd::Val(co)), 0, 16, p);
    b.buffer_atomic(A::PkAddF16, r, Some(Opnd::Val(co)), 0, 32, h);
    b.buffer_atomic(A::AddU32, r, Some(Opnd::Val(co)), 0, 48, tid);
    let g = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(co) });
    b.atomic_global(A::PkAddBf16, g, 64, p, false);
    let ret = b.atomic_global(A::PkAddF16, g, 80, h, true).unwrap();
    b.store_global(pout, 256, ret, MemTy::B32, false);
    b.finish()
}

/// `probe_tr_b16_mfma2`: `probe_tr_b16_mfma` with the A operand from `load_lds_tr_b16x2` (both reads write their half of the Frag4 in place) and
/// the B operand from four single-dword loads packed with `pack4` (each load writes its dword in place): no operand copies.
pub fn probe_tr_b16_mfma2() -> Function {
    use amdgpu_dsl::{MemTy, MmaKind, MmaShape, MmaTy};
    let mut b = FnBuilder::new("probe_tr_b16_mfma2", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(2048));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(1024));
    b.lds(1024);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off16) });
    let t = b.load_global(pin, 0, MemTy::B128);
    b.store_lds_b128(off16, t);
    let w: Vec<ValueId> = (0..4u32).map(|i| b.load_global(pin, 1024 + 4 * i, MemTy::B32)).collect();
    let fb = b.pack4(w[0], w[1], w[2], w[3]);
    b.wait_lds(0);
    b.barrier();
    let s = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(15)));
    let g = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(4)));
    let q = b.let_(Op::LshrI32(Opnd::Val(s), Opnd::I32(2)));
    let c = b.let_(Op::AndI32(Opnd::Val(s), Opnd::I32(3)));
    let row = b.tri(amdgpu_dsl::TriOp::LshlAddU32, g, 3, q);
    let rb = b.let_(Op::MulI32(Opnd::Val(row), Opnd::I32(32)));
    let addr = b.tri(amdgpu_dsl::TriOp::LshlAddU32, c, 3, rb);
    let a1 = b.add_i32(addr, 128);
    let fa = b.load_lds_tr_b16x2(addr, a1);
    let d = b.mfma(MmaKind::new(MmaShape::M16N16K32, MmaTy::Bf16), fa, fb, None);
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off16) });
    b.store_global(pout, 0, d, MemTy::B128, false);
    b.finish()
}

/// `probe_ptr64_p1(in, out)`: 64-bit address arithmetic. in: plane 0 x, plane 1 y (u32). out planes (64 dwords each): 0 / 1 the out pointer
/// (lo / hi), 2 / 3 `ptr_mad(out, 0x12345678, 0x9abc)` (scalar path, constants), 4 / 5 `ptr_mad(out, x, y)` (vector path), 6 / 7
/// `ptr_mad(out, rfl(x), rfl(y))` (scalar path, SGPR factors), 8 lane + 1 stored through a descriptor whose base was moved by `ptr_mad(out, 64, 32)`.
pub fn probe_ptr64_p1() -> Function {
    use amdgpu_dsl::MemTy;
    let mut b = FnBuilder::new("probe_ptr64_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(512));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(9 * 256));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off4) });
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    let x = b.load_global(pin, 0, MemTy::B32);
    let y = b.load_global(pin, 256, MemTy::B32);
    let s = b.ptr_mad(out, 0x12345678, 0x9abc);
    let v = b.ptr_mad(out, x, y);
    let xr = b.let_(Op::ReadFirstLane(Opnd::Val(x)));
    let yr = b.let_(Op::ReadFirstLane(Opnd::Val(y)));
    let r = b.ptr_mad(out, xr, yr);
    let mut k = 0u32;
    for p in [out, s, v, r] {
        for h in 0..2u8 {
            let w = b.frag_slice(p, h, 1);
            b.store_global(pout, 256 * k, w, MemTy::B32, false);
            k += 1;
        }
    }
    let moved = b.ptr_mad(out, 64, 32);
    let rs = b.buffer_desc(moved, 256);
    let one = b.add_i32(tid, 1);
    b.buffer_store_n(rs, Some(Opnd::Val(off4)), 0, 0, one, MemTy::B32, false);
    b.finish()
}

/// The byte offsets of `probe_big_addr`'s windows (just below / above 4 GiB and 8 GiB) as (offset >> 8, offset & 255).
/// Window k covers A = [w, w + 256) and B = [w + 256, w + 512): windows 0 / 2 put B across the 4 GiB / 8 GiB line, windows 1 / 3 sit just above it.
pub const BIG_ADDR_WINDOWS: [(u32, u32); 4] = [(0x00ff_fffe, 0x80), (0x0100_0001, 0x00), (0x01ff_fffe, 0x80), (0x0200_0001, 0x00)];

/// `probe_big_addr(big, res)` (host tests/gpu/big_addr_host.cpp, a 9 GiB `big` buffer): for every window k, lane l writes pattern A
/// (0xA000_0000 | k << 16 | l) at `window + 4 l` through a descriptor whose base was moved with a SCALAR `ptr_mad`, and pattern B
/// (0xB000_0000 | k << 16 | l) at `window + 256 + 4 l` through a VECTOR `ptr_mad` pointer; after a wait each lane reads A back through the
/// vector pointer and B through the descriptor. res[k][dir][l] = 1 when the read matches (512 passes expected).
pub fn probe_big_addr() -> Function {
    use amdgpu_dsl::MemTy;
    let mut b = FnBuilder::new("probe_big_addr", 64);
    b.arg("big", DType::Ptr, 0, 8);
    b.arg("res", DType::Ptr, 8, 8);
    b.arg_tensor("big", amdgpu_dsl::Ext::Const(9i64 << 30));
    b.arg_tensor("res", amdgpu_dsl::Ext::Const(8 * 256));
    let big = b.let_(Op::KernargPtr { offset: 0 });
    let res = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let zero_v = b.let_(Op::MinI32(Opnd::Val(tid), Opnd::I32(0))); // 0, but per-lane: forces the vector path
    let mut views = Vec::new();
    for (k, (hi8, rem)) in BIG_ADDR_WINDOWS.iter().enumerate() {
        let m = b.ptr_mad(big, *hi8 as i32, 256);
        let r = b.buffer_desc(m, 1024);
        let va = b.add_i32(tid, (0xA000_0000u32 | (k as u32) << 16) as i32);
        b.buffer_store_n(r, Some(Opnd::Val(off4)), 0, *rem, va, MemTy::B32, false);
        let hv = b.add_i32(zero_v, *hi8 as i32);
        let p = b.ptr_mad(big, hv, 256);
        let pl = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off4) });
        let vb = b.add_i32(tid, (0xB000_0000u32 | (k as u32) << 16) as i32);
        b.store_global(pl, rem + 256, vb, MemTy::B32, false);
        views.push((r, pl, va, vb, *rem));
    }
    b.wait_vm(0);
    let pres = b.let_(Op::PtrAdd { base: Opnd::Val(res), byte_off: Opnd::Val(off4) });
    for (k, (r, pl, va, vb, rem)) in views.into_iter().enumerate() {
        let ga = b.load_global(pl, rem, MemTy::B32);
        let gb = b.buffer_load_n(r, Some(Opnd::Val(off4)), 0, rem + 256, MemTy::B32);
        for (dir, (got, want)) in [(ga, va), (gb, vb)].into_iter().enumerate() {
            let eq = b.cmp(CmpPred::Eq, CmpTy::I32, got, want);
            let f = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(eq), a: Opnd::I32(1), b: Opnd::I32(0) });
            let base = b.let_(Op::PtrAdd { base: Opnd::Val(pres), byte_off: Opnd::I32((256 * (2 * k + dir)) as i32) });
            b.store_global(base, 0, f, MemTy::B32, false);
        }
    }
    b.finish()
}

/// `probe_anylane(in, out)`: wave-level branches on negated per-lane conditions (`any_lane(!c)` must be taken when SOME lane has !c). out plane
/// k (64 dwords) = 1 in every lane when branch k was entered, else 0: 0 any_lane(!(tid != 0)) -> taken (lane 0); 1 any_lane(!(tid >= 0)) -> not
/// taken; 2 any_lane(tid == 5) (a == compare) -> taken; 3 a uniform !(1 < 0) -> taken; 4 any_lane(tid >= 0) (all lanes) -> taken; 5
/// any_lane(tid < 0) (no lane) -> not taken; 6 any_lane(!(tid >= 64)) (all) -> taken; 7 any_lane(CmpLtI32(tid, 3)) (3 lanes, the folded
/// v_cmp path) -> taken; 8 any_lane(!CmpLtI32(tid, 3)) (61 lanes) -> taken.
pub fn probe_anylane() -> Function {
    use amdgpu_dsl::MemTy;
    let mut b = FnBuilder::new("probe_anylane", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(256));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(9 * 256));
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    for k in 0..9u32 {
        b.store_global(pout, 256 * k, 0, MemTy::B32, false);
    }
    b.wait_vm(0);
    let ne0 = b.cmp(CmpPred::Ne, CmpTy::I32, tid, 0);
    let ge0 = b.cmp(CmpPred::Ge, CmpTy::I32, tid, 0);
    let eq5 = b.cmp(CmpPred::Eq, CmpTy::I32, tid, 5);
    let ulit = b.cmp(CmpPred::Lt, CmpTy::I32, 1, 0);
    let lt0 = b.cmp(CmpPred::Lt, CmpTy::I32, tid, 0);
    let ge64 = b.cmp(CmpPred::Ge, CmpTy::I32, tid, 64);
    let ltl = b.let_(Op::CmpLtI32(Opnd::Val(tid), Opnd::I32(3)));
    let conds = [
        { let n = b.let_(Op::NotI1(Opnd::Val(ne0))); b.any_lane(n) },
        { let n = b.let_(Op::NotI1(Opnd::Val(ge0))); b.any_lane(n) },
        b.any_lane(eq5),
        b.let_(Op::NotI1(Opnd::Val(ulit))),
        b.any_lane(ge0),
        b.any_lane(lt0),
        { let n = b.let_(Op::NotI1(Opnd::Val(ge64))); b.any_lane(n) },
        b.any_lane(ltl),
        { let n = b.let_(Op::NotI1(Opnd::Val(ltl))); b.any_lane(n) },
    ];
    for (k, c) in conds.into_iter().enumerate() {
        b.begin_if(Opnd::Val(c));
        b.store_global(pout, 256 * k as u32, 1, MemTy::B32, false);
        b.end_if();
    }
    b.finish()
}

/// `probe_reload_p1(in, out)`: a loop-carried Frag4 variable reloaded IN PLACE from global memory each iteration (`set_var(v, cat_frag4(load
/// b64, load b64))`, AGPR-resident): iteration i stores the variable (row i of `in`) and reloads row i + 1. out row i = in row i (i < 4).
pub fn probe_reload_p1() -> Function {
    use amdgpu_dsl::MemTy;
    let mut b = FnBuilder::new("probe_reload_p1", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(5 * 1024));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(4 * 1024));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
    let r_in = b.buffer_desc(inp, 5 * 1024);
    let lo0 = b.buffer_load_n(r_in, Some(Opnd::Val(off16)), 0, 0, MemTy::B64);
    let hi0 = b.buffer_load_n(r_in, Some(Opnd::Val(off16)), 0, 8, MemTy::B64);
    let f0 = b.cat_frag4(lo0, hi0);
    let vars = b.begin_loop(&[(DType::Frag4, Opnd::Val(f0)), (DType::I32, Opnd::I32(0))]);
    let (v, i) = (vars[0], vars[1]);
    b.pin_agpr(v);
    // store the current row i
    let row = b.let_(Op::MulI32(Opnd::Var(i), Opnd::I32(1024)));
    let oo = b.add_i32(off16, row);
    let r_out = b.buffer_desc(out, 4 * 1024);
    let cur = b.let_(Op::FragCat { parts: vec![Opnd::Var(v)], dwords: 4 });
    b.buffer_store_n(r_out, Some(Opnd::Val(oo)), 0, 0, cur, MemTy::B128, false);
    // reload row i + 1 in place
    let nrow = b.add_i32(row, 1024);
    let no = b.add_i32(off16, nrow);
    let lo = b.buffer_load_n(r_in, Some(Opnd::Val(no)), 0, 0, MemTy::B64);
    let hi = b.buffer_load_n(r_in, Some(Opnd::Val(no)), 0, 8, MemTy::B64);
    let f = b.cat_frag4(lo, hi);
    b.set_var(v, f);
    let i2 = b.add_i32(Opnd::Var(i), 1);
    b.set_var(i, i2);
    let done = b.cmp(CmpPred::Ge, CmpTy::I32, i2, 4);
    b.break_if(done);
    b.end_loop();
    b.finish()
}

/// LDS byte offsets of `probe_lds_high` (DMA destinations; the last three are at / past 128 KiB).
pub const LDS_HIGH_M0: [u32; 5] = [0x1_c000, 0x1_f000, 0x2_0000, 0x2_4000, 0x2_7000];

/// `probe_lds_high(in, out)`: does the M0-based LDS-DMA (`buffer_load_dwordx4 ... lds`) reach LDS bytes >= 128 KiB on gfx950 (160 KiB per
/// workgroup)? Lane l first ds_writes a marker (0xC0DE0000 | region << 8 | l) at `M0 - 128 KiB + 16 l` for the high regions, then the DMA of
/// region k writes in[1 KiB k + 16 l] to `M0_k + 16 l`; after a wait + barrier every lane reads both back with ds_read_b128. out plane pair k:
/// [the DMA'd 16 bytes as read at M0_k] then [the marker location's first dword] (64 lanes x 5 dwords per region).
pub fn probe_lds_high() -> Function {
    use amdgpu_dsl::MemTy;
    let mut b = FnBuilder::new("probe_lds_high", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(5 * 1024));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(5 * 64 * 20));
    b.lds(160 * 1024);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off16 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
    let r_in = b.buffer_desc(inp, 5 * 1024);
    for (k, m0) in LDS_HIGH_M0.iter().enumerate() {
        if *m0 >= 0x2_0000 {
            let a = b.add_i32(off16, (*m0 - 0x2_0000) as i32);
            let mk = b.add_i32(tid, (0xC0DE_0000u32 | (k as u32) << 8) as i32);
            b.store_lds(a, mk, MemTy::B32, false);
        }
    }
    b.wait_lds(0);
    b.barrier();
    for (k, m0) in LDS_HIGH_M0.iter().enumerate() {
        let m0v = b.c_i32(*m0 as i32);
        let m0s = b.read_first_lane(m0v);
        let vo = b.add_i32(off16, (1024 * k) as i32);
        b.load_buffer_lds(r_in, vo, 0i32, m0s, 0, 4);
    }
    b.wait_vm(0);
    b.barrier();
    let off20 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(20)));
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off20) });
    for (k, m0) in LDS_HIGH_M0.iter().enumerate() {
        let a = b.add_i32(off16, *m0 as i32);
        let v = b.load_lds(a, MemTy::B128);
        let base = b.let_(Op::PtrAdd { base: Opnd::Val(pout), byte_off: Opnd::I32((64 * 20 * k) as i32) });
        b.store_global(base, 0, v, MemTy::B128, false);
        let low = b.add_i32(off16, (*m0 as i32) - 0x2_0000);
        let mk = if *m0 >= 0x2_0000 { b.load_lds(low, MemTy::B32) } else { b.c_i32(0) };
        b.store_global(base, 16, mk, MemTy::B32, false);
    }
    b.finish()
}

/// `probe_ifelse(in, out)`: the MLA-decode pattern (kernel-opt): values defined BEFORE a wave-uniform if / else (a zero, the per-lane
/// `live` select, an out-of-range sentinel) used only in the ELSE arm, while the THEN arm runs a register-hungry loop. in[0] = n (uniform):
/// n > 0 runs the then arm, else the else arm stores -inf to out[l] for live lanes (l < 40) through a buffer descriptor (dead lanes: the
/// sentinel offset, dropped) and 0 to out[64 + l] for all lanes. The host runs it with n = 0 (else arm) and n = 3 (then arm: out[l] = sum).
pub fn probe_ifelse() -> Function {
    use amdgpu_dsl::MemTy;
    let mut b = FnBuilder::new("probe_ifelse", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(256 * 4));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(128 * 4));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let r_in = b.buffer_desc(inp, 1024);
    let r_out = b.buffer_desc(out, 512);
    let c0 = b.c_i32(0);
    let n = { let v = b.buffer_load_n(r_in, Some(Opnd::Val(c0)), 0, 0, MemTy::B32); b.read_first_lane(v) };
    let zero = b.c_i32(0);
    let big = b.c_i32(0x7fff_fff0);
    let live = b.cmp(CmpPred::Lt, CmpTy::I32, tid, 40);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(off4), b: Opnd::Val(big) });
    let ninf = b.c_f32(f32::NEG_INFINITY);
    let go = b.cmp(CmpPred::Gt, CmpTy::I32, n, 0);
    b.begin_if(Opnd::Val(go));
    {
        // register-hungry loop: 48 loaded values accumulated per iteration
        let init: Vec<(DType, Opnd)> = (0..48).map(|_| (DType::I32, Opnd::I32(0))).chain([(DType::I32, Opnd::I32(0))]).collect();
        let vars = b.begin_loop(&init);
        let it = vars[48];
        let mut s = b.c_i32(0);
        for j in 0..48usize {
            let o = b.add_i32(off4, (4 * (j % 8) as i32) + 256);
            let x = b.buffer_load_n(r_in, Some(Opnd::Val(o)), 0, 0, MemTy::B32);
            let y = b.add_i32(Opnd::Var(vars[j]), x);
            b.set_var(vars[j], y);
        }
        for j in 0..48usize {
            s = b.add_i32(s, Opnd::Var(vars[j]));
        }
        let _ = s;
        let i2 = b.add_i32(Opnd::Var(it), 1);
        b.set_var(it, i2);
        let done = b.cmp(CmpPred::Ge, CmpTy::I32, i2, n);
        b.break_if(done);
        b.end_loop();
        let mut s = b.c_i32(0);
        for j in 0..48usize {
            s = b.add_i32(s, Opnd::Var(vars[j]));
        }
        b.buffer_store_n(r_out, Some(Opnd::Val(off4)), 0, 0, s, MemTy::B32, false);
    }
    b.else_();
    {
        let nb = b.let_(Op::BitcastI32(Opnd::Val(ninf)));
        b.buffer_store_n(r_out, Some(Opnd::Val(vo)), 0, 0, nb, MemTy::B32, false);
        b.buffer_store_n(r_out, Some(Opnd::Val(off4)), 0, 256, zero, MemTy::B32, false);
    }
    b.end_if();
    b.finish()
}

/// `probe_fp4_chain(in, out)` (kernel-opt's mxquant pattern): one word built from four chained `ScalePkFp4F32` conversions, sel 0..3, each
/// with the previous result as `old`. in planes 0..7: f32 pairs (x_s, y_s) for byte s (values representable after / 2), plane 8: scale 2.0.
pub fn probe_fp4_chain() -> Function {
    use amdgpu_dsl::{CvtOp, MemTy};
    let mut b = FnBuilder::new("probe_fp4_chain", 64);
    b.arg("in", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("in", amdgpu_dsl::Ext::Const(9 * 256));
    b.arg_tensor("out", amdgpu_dsl::Ext::Const(256));
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let out = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let off4 = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let pin = b.let_(Op::PtrAdd { base: Opnd::Val(inp), byte_off: Opnd::Val(off4) });
    let pl: Vec<ValueId> = (0..9u32).map(|k| b.load_global(pin, 256 * k, MemTy::B32)).collect();
    let mut w = b.c_i32(0);
    for s in 0..4u8 {
        let (x, y) = (pl[2 * s as usize], pl[2 * s as usize + 1]);
        w = b.cvt(CvtOp::ScalePkFp4F32, &[Opnd::Val(x), Opnd::Val(y), Opnd::Val(pl[8]), Opnd::Val(w)], s);
    }
    let pout = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off4) });
    b.store_global(pout, 0, w, MemTy::B32, false);
    b.finish()
}
