//! TIRx-style tile operations on register tiles (docs/op-coverage-tileops.md §6.3): `ewise` applies a scalar function to every element of
//! one or more register tiles, `apply2` applies one VALU instruction family over a tile and picks the packed `v_pk_*` form for element pairs
//! when one exists, and the cross-lane `row_allreduce16` / `wave_allreduce` reduce over lanes with DPP-modified instructions. They are
//! builders: everything they emit is ordinary DSL IR (capture / weave, IR text and the verifiers apply unchanged).
//!
//! Copies and MMAs keep their tile layer (`TiledCopy`, `TiledMma`); a `RegTile` is the per-lane register side those produce and consume.
use amdgpu_dsl::{DType, Dpp, DppAluOp, DppCtrl, FnBuilder, Op, Opnd, ValueId};

/// A per-lane register tile: `dwords` consecutive dwords of the fragment value `frag` (1, 2, 4, 8 or 16), elements of type `elem`
/// (`F32` / `I32`: one per dword).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegTile {
    pub frag: ValueId,
    pub dwords: u8,
    pub elem: DType,
}

impl RegTile {
    pub fn new(frag: ValueId, dwords: u8, elem: DType) -> RegTile {
        assert!(matches!(dwords, 1 | 2 | 4 | 8 | 16));
        RegTile { frag, dwords, elem }
    }
    /// Element `i` as a value of type `elem` (a register view: no instruction).
    pub fn elem(&self, b: &mut FnBuilder, i: u8) -> ValueId {
        assert!(i < self.dwords);
        let raw = if self.dwords == 1 { self.frag } else { b.frag_slice(self.frag, i, 1) };
        match self.elem {
            DType::F32 => b.let_(Op::BitcastF32(Opnd::Val(raw))),
            _ => raw,
        }
    }
    /// Elements `2 p, 2 p + 1` as a Frag2 view.
    pub fn pair(&self, b: &mut FnBuilder, p: u8) -> ValueId {
        assert!(2 * p + 1 < self.dwords);
        if self.dwords == 2 { self.frag } else { b.frag_slice(self.frag, 2 * p, 2) }
    }
    /// A tile from per-element values (copied into one fragment).
    pub fn from_elems(b: &mut FnBuilder, elems: &[ValueId], elem: DType) -> RegTile {
        if elems.len() == 1 {
            return RegTile::new(elems[0], 1, elem);
        }
        let parts: Vec<Opnd> = elems.iter().map(|v| Opnd::Val(*v)).collect();
        let f = b.frag_cat(&parts);
        RegTile::new(f, elems.len() as u8, elem)
    }
}

/// `ewise`: `out[i] = f(x0[i], x1[i], ...)` for every element (TIRx `tile.ewise`); `f` gets the elements as values of their `elem` type and
/// returns a value of type `out`.
pub fn ewise(b: &mut FnBuilder, srcs: &[&RegTile], out: DType, f: impl Fn(&mut FnBuilder, &[ValueId]) -> ValueId) -> RegTile {
    let n = srcs[0].dwords;
    assert!(srcs.iter().all(|s| s.dwords == n), "ewise: tiles of equal size");
    let mut res = Vec::with_capacity(n as usize);
    for i in 0..n {
        let xs: Vec<ValueId> = srcs.iter().map(|s| s.elem(b, i)).collect();
        let r = f(b, &xs);
        let r = if out == DType::F32 { b.let_(Op::BitcastI32(Opnd::Val(r))) } else { r };
        res.push(r);
    }
    RegTile::from_elems(b, &res, out)
}

/// Element-wise binary VALU ops `apply2` knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ew2 {
    AddF32,
    MulF32,
    SubF32,
    MaxF32,
    MinF32,
    AddI32,
}

/// `apply2`: `out[i] = x[i] op y[i]` (TIRx `tile.apply` with a VALU instruction); f32 add / mul use `v_pk_add_f32` / `v_pk_mul_f32` on element
/// pairs (half the instructions), the rest one instruction per element.
pub fn apply2(b: &mut FnBuilder, op: Ew2, x: &RegTile, y: &RegTile) -> RegTile {
    assert_eq!(x.dwords, y.dwords, "apply2: tiles of equal size");
    if matches!(op, Ew2::AddF32 | Ew2::MulF32) && x.dwords >= 2 {
        let mut parts = Vec::new();
        for p in 0..x.dwords / 2 {
            let (a, c) = (x.pair(b, p), y.pair(b, p));
            parts.push(Opnd::Val(if op == Ew2::AddF32 { b.pk_add_f32(a, c) } else { b.pk_mul_f32(a, c) }));
        }
        let f = if parts.len() == 1 { match parts[0] { Opnd::Val(v) => v, _ => unreachable!() } } else { b.frag_cat(&parts) };
        return RegTile::new(f, x.dwords, DType::F32);
    }
    let out = if op == Ew2::AddI32 { DType::I32 } else { DType::F32 };
    ewise(b, &[x, y], out, move |b, v| {
        let (a, c) = (Opnd::Val(v[0]), Opnd::Val(v[1]));
        match op {
            Ew2::AddF32 => b.let_(Op::AddF32(a, c)),
            Ew2::MulF32 => b.let_(Op::MulF32(a, c)),
            Ew2::SubF32 => b.let_(Op::SubF32(a, c)),
            Ew2::MaxF32 => b.let_(Op::MaxF32(a, c)),
            Ew2::MinF32 => b.let_(Op::Bin(amdgpu_dsl::BinOp::MinF32, a, c)),
            Ew2::AddI32 => b.let_(Op::AddI32(a, c)),
        }
    })
}

/// All-reduce of `x` over each 16-lane row: every lane of the row gets `op` over its 16 values, in four DPP-modified instructions
/// (quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_ror:4, row_ror:8; no lane is ever invalid, so no identity element is needed).
pub fn row_allreduce16(b: &mut FnBuilder, op: DppAluOp, x: ValueId) -> ValueId {
    let mut v = x;
    for c in [DppCtrl::QuadPerm([1, 0, 3, 2]), DppCtrl::QuadPerm([2, 3, 0, 1]), DppCtrl::RowRor(4), DppCtrl::RowRor(8)] {
        v = b.dpp_alu(op, Dpp::new(c), v, v, v);
    }
    v
}

/// All-reduce of `x` over the 64-lane wave: the row all-reduce, then the four row results (read through `v_readlane`) combined. Returns a
/// value of `x`'s type (F32 for the float ops), wave-uniform.
pub fn wave_allreduce(b: &mut FnBuilder, op: DppAluOp, x: ValueId) -> ValueId {
    let r = row_allreduce16(b, op, x);
    let float = matches!(op, DppAluOp::AddF32 | DppAluOp::SubF32 | DppAluOp::MulF32 | DppAluOp::MaxF32 | DppAluOp::MinF32);
    let rows: Vec<ValueId> = [0u8, 16, 32, 48]
        .iter()
        .map(|l| {
            let v = b.read_lane(r, *l);
            if float { b.let_(Op::BitcastF32(Opnd::Val(v))) } else { v }
        })
        .collect();
    let combine = |b: &mut FnBuilder, p: ValueId, q: ValueId| -> ValueId {
        let (a, c) = (Opnd::Val(p), Opnd::Val(q));
        match op {
            DppAluOp::AddF32 => b.let_(Op::AddF32(a, c)),
            DppAluOp::MulF32 => b.let_(Op::MulF32(a, c)),
            DppAluOp::MaxF32 => b.let_(Op::MaxF32(a, c)),
            DppAluOp::MinF32 => b.let_(Op::Bin(amdgpu_dsl::BinOp::MinF32, a, c)),
            DppAluOp::AddU32 => b.let_(Op::AddI32(a, c)),
            DppAluOp::MaxI32 => b.let_(Op::Bin(amdgpu_dsl::BinOp::MaxI32, a, c)),
            DppAluOp::MinI32 => b.let_(Op::MinI32(a, c)),
            DppAluOp::MaxU32 => b.let_(Op::Bin(amdgpu_dsl::BinOp::MaxU32, a, c)),
            DppAluOp::MinU32 => b.let_(Op::Bin(amdgpu_dsl::BinOp::MinU32, a, c)),
            DppAluOp::AndB32 => b.let_(Op::AndI32(a, c)),
            DppAluOp::OrB32 => b.let_(Op::Bin(amdgpu_dsl::BinOp::OrB32, a, c)),
            DppAluOp::XorB32 => b.let_(Op::XorI32(a, c)),
            DppAluOp::SubF32 => panic!("wave_allreduce: subtraction is not associative"),
        }
    };
    let s0 = combine(b, rows[0], rows[1]);
    let s1 = combine(b, rows[2], rows[3]);
    combine(b, s0, s1)
}
