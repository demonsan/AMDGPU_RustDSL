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

