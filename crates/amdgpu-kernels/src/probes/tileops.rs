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
