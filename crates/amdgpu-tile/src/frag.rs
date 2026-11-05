//! L4: register fragments with layouts and tile-level MFMA.
//!
//! The FMHA computes `S^T = K Q^T` and `O^T = V^T P^T` with `v_mfma_scale_f32_32x32x64_f8f6f4`: the A operand is K (rows of the K tile, read with
//! `ds_read_b128 x2`) or `V^T` (columns of the V tile, read with `ds_read_b64_tr_b8 x4`), the B operand is Q (resident registers) or P (converted words), the
//! accumulator is a 32x32 f32 tile whose lane l holds the 16 values `r` at row `8 (r >> 2) + 4 (l >> 5) + (r & 3)`, column `l & 31` (`C32x32F32`).
//! The K order inside a 64-byte k-group is PERMUTED identically for A and B (lane half h reads bytes `16 h ..` and `32 + 16 h ..`): `kperm`. The layout types
//! make a mismatched operand a rustc type error; the readers compute the lane addresses once (the manual `kfx`/`trx`) and add only the slot and the constant
//! row/column offsets per read.

use std::marker::PhantomData;

use crate::shared::{Layout, Shared, SlotRef, Swz};
use amdgpu_dsl::{ev, fmha_swz, Lane};
use crate::{add0};
use amdgpu_dsl::{DType, FnBuilder, Opnd, ValueId, VarId, I32};

/// A register fragment layout.
pub trait FragLayout {
    const TY: DType;
}
/// A operand of the scaled 32x32x64 fp8 MFMA (8 dwords, kperm order).
#[derive(Clone, Copy, Debug)]
pub struct A32x32x64Fp8;
/// B operand of the scaled 32x32x64 fp8 MFMA (8 dwords, kperm order).
#[derive(Clone, Copy, Debug)]
pub struct B32x32x64Fp8;
/// 32x32 f32 accumulator (16 dwords).
#[derive(Clone, Copy, Debug)]
pub struct C32x32F32;
/// 16x16 f32 accumulator of the smfmac row sums (4 dwords).
#[derive(Clone, Copy, Debug)]
pub struct C16x16F32;
impl FragLayout for A32x32x64Fp8 {
    const TY: DType = DType::Frag8;
}
impl FragLayout for B32x32x64Fp8 {
    const TY: DType = DType::Frag8;
}
impl FragLayout for C32x32F32 {
    const TY: DType = DType::Frag16;
}
impl FragLayout for C16x16F32 {
    const TY: DType = DType::Frag4;
}

impl C32x32F32 {
    /// Row of value `r` of a lane = `half_base + row_const(r)` with `half_base = 4 (lane >> 5)`.
    pub fn row_const(r: u8) -> i32 {
        (8 * (r / 4) + (r % 4)) as i32
    }
    /// `4 (lane >> 5)`.
    pub fn half_base(lane: &Lane) -> I32 {
        lane.half() << 2
    }
}

/// A fragment value.
#[derive(Debug)]
pub struct Frag<L> {
    pub v: ValueId,
    _l: PhantomData<L>,
}
impl<L> Clone for Frag<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for Frag<L> {}
impl<L: FragLayout> Frag<L> {
    /// Wraps a value (the caller asserts the layout).
    pub fn wrap(v: ValueId) -> Frag<L> {
        Frag { v, _l: PhantomData }
    }
}
impl Frag<B32x32x64Fp8> {
    /// 8 packed words (4 fp8 each) in kperm order.
    pub fn from_words(b: &mut FnBuilder, w: &[ValueId]) -> Frag<B32x32x64Fp8> {
        assert_eq!(w.len(), 8);
        Frag::wrap(b.frag8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]))
    }
}

/// A loop-carried accumulator.
#[derive(Debug)]
pub struct Acc<L> {
    pub var: VarId,
    _l: PhantomData<L>,
}
impl<L> Clone for Acc<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for Acc<L> {}
impl<L: FragLayout> Acc<L> {
    pub fn wrap(var: VarId) -> Acc<L> {
        Acc { var, _l: PhantomData }
    }
}

/// N accumulators that form one logical tile (S of 128 keys = 4 x 32 keys; O of 128 dims = 4 x 32 dims).
#[derive(Clone, Copy, Debug)]
pub struct AccTile<L, const N: usize> {
    pub accs: [Acc<L>; N],
}
impl<L: FragLayout, const N: usize> AccTile<L, N> {
    pub fn from_vars(v: &[VarId]) -> AccTile<L, N> {
        assert_eq!(v.len(), N);
        AccTile { accs: std::array::from_fn(|i| Acc::wrap(v[i])) }
    }
    pub fn vars(&self) -> [VarId; N] {
        std::array::from_fn(|i| self.accs[i].var)
    }
}
impl<const N: usize> AccTile<C32x32F32, N> {
    /// The 16 f32 values of accumulator `i` of this lane (`frag16_lane` + bitcast).
    pub fn values(&self, b: &mut FnBuilder, i: usize) -> Vec<ValueId> {
        (0..16u8)
            .map(|r| {
                let l = b.frag16_lane(Opnd::Var(self.accs[i].var), r);
                b.bitcast_f32(l)
            })
            .collect()
    }
}

/// `acc += mfma_scale(a, b; sa, sb)` (in place).
pub fn mma_scaled(b: &mut FnBuilder, acc: &Acc<C32x32F32>, a: &Frag<A32x32x64Fp8>, bf: &Frag<B32x32x64Fp8>, sa: impl Into<Opnd>, sb: impl Into<Opnd>) {
    b.var_mfma_scale(acc.var, a.v, bf.v, sa, sb);
}
/// Value form: `mfma_scale(a, b, c; sa, sb)`.
pub fn mma_scaled_v(b: &mut FnBuilder, a: &Frag<A32x32x64Fp8>, bf: &Frag<B32x32x64Fp8>, c: ValueId, sa: impl Into<Opnd>, sb: impl Into<Opnd>) -> ValueId {
    b.mfma_scale_fp8(a.v, bf.v, c, sa, sb)
}

/// Selects the per-lane-half E8M0 scale bytes from a scale word: `word >> 8h` (k-group 0) and `word >> (8h + 16)` (k-group 1), `h` = lane half.
#[derive(Clone, Copy, Debug)]
pub struct ScaleSel {
    /// `8 * (lane >> 5)`
    pub g8: ValueId,
}
impl ScaleSel {
    pub fn new(b: &mut FnBuilder, lane: &Lane) -> ScaleSel {
        ScaleSel { g8: b.bind_i32(lane.half() << 3) }
    }
    pub fn pair(&self, b: &mut FnBuilder, word: ValueId) -> (ValueId, ValueId) {
        let lo = b.bind_i32(ev(word).lshr(ev(self.g8)));
        let hi = b.bind_i32(ev(word).lshr(ev(self.g8) + 16));
        (lo, hi)
    }
}

/// The two resident B fragments of a 128-wide row (Q) loaded from global memory: 2 x `global_load_dwordx4` per 64-byte k-group (lane half h: bytes
/// `64 c + 16 h` and `64 c + 32 + 16 h`), plus the raw words (for an LDS copy).
#[derive(Clone, Debug)]
pub struct RowFrags {
    pub frags: [Frag<B32x32x64Fp8>; 2],
    pub words: [[ValueId; 8]; 2],
}
impl RowFrags {
    /// `row_ptr_off` = byte offset of the lane's row INCLUDING the lane-half term `16 (lane >> 5)`.
    pub fn load(b: &mut FnBuilder, ptr: ValueId, row_off: ValueId) -> RowFrags {
        let mut frags = Vec::with_capacity(2);
        let mut words = [[0 as ValueId; 8]; 2];
        for c in 0..2u32 {
            let qp = b.ptr_add(ptr, row_off);
            let mut w: Vec<ValueId> = Vec::with_capacity(8);
            for half in 0..2u32 {
                let f4 = b.load_b128(qp, 64 * c + 32 * half);
                for k in 0..4u8 {
                    let l = b.frag4_lane(f4, k);
                    w.push(b.bitcast_i32(l));
                }
            }
            words[c as usize].copy_from_slice(&w);
            frags.push(Frag::<B32x32x64Fp8>::from_words(b, &w));
        }
        RowFrags { frags: [frags[0], frags[1]], words }
    }
}

/// Reader of A fragments from the ROWS of a 128-byte-row swizzled tile (the K tile): lane (j, h) reads row `j + 32 s`, chunks `(h ^ s(j)) ^ 2k`, k = 0..3
/// (two `ds_read_b128` per 64-byte k-group).
#[derive(Clone, Debug)]
pub struct RowsReader {
    sh: Shared,
    kfx: [I32; 4],
}
impl RowsReader {
    pub fn new(b: &mut FnBuilder, sh: &Shared, lane: &Lane) -> RowsReader {
        assert!(matches!(sh.layout, Layout::Rows { row_bytes: 128, .. }), "RowsReader: 128-byte rows");
        let x0 = match sh.layout {
            Layout::Rows { swz: Swz::Fmha, .. } => {
                let s = fmha_swz(b, &lane.j());
                b.let_i32(lane.half() ^ s)
            }
            _ => lane.half(),
        };
        let jrow = b.let_i32(lane.j() * 128);
        let base = I32::c(sh.base as i32);
        let kfx = std::array::from_fn(|k| b.let_i32(add0(&jrow + ((&x0 ^ (2 * k as i32)) << 4), &base)));
        RowsReader { sh: sh.clone(), kfx }
    }
    /// Per-slot addresses (4 adds, once per tile).
    pub fn at<'a>(&self, b: &mut FnBuilder, slot: &SlotRef<'a>) -> RowsSlot<'a> {
        assert_eq!(slot.sh.name, self.sh.name);
        let off = slot.off_v(b);
        let kfb = std::array::from_fn(|k| b.let_i32(&self.kfx[k] + &off));
        RowsSlot { slot: slot.clone(), kfb }
    }
}
/// A `RowsReader` bound to one slot.
#[derive(Clone, Debug)]
pub struct RowsSlot<'a> {
    pub slot: SlotRef<'a>,
    kfb: [I32; 4],
}
impl RowsSlot<'_> {
    /// The two A fragments (k-groups 0 and 1) of row group `s` (rows 32 s ..), under one read marker.
    pub fn read_pair(&self, b: &mut FnBuilder, s: usize) -> [Frag<A32x32x64Fp8>; 2] {
        let ko = 4096 * s as i32;
        let a: [ValueId; 4] = std::array::from_fn(|k| b.bind_i32(&self.kfb[k] + ko));
        self.slot.read(b, |b| {
            let f0 = b.load_lds_b128x2(a[0], a[1]);
            let f1 = b.load_lds_b128x2(a[2], a[3]);
            [Frag::wrap(f0), Frag::wrap(f1)]
        })
    }
}

/// Reader of `V^T` A fragments from a 128-byte-row swizzled tile with `ds_read_b64_tr_b8` (the manual `trx`): lane l covers key
/// `8 ((l & 15) >> 3) + 4 (l >> 5) + ((l >> 1) & 3)` (+ 16 per read), column chunk `((l >> 4) & 1) ^ s(key) ^ 2e`, byte `8 (l & 1)`.
#[derive(Clone, Debug)]
pub struct TrReader {
    sh: Shared,
    trx: [I32; 4],
}
impl TrReader {
    pub fn new(b: &mut FnBuilder, sh: &Shared, lane: &Lane) -> TrReader {
        assert!(matches!(sh.layout, Layout::Rows { row_bytes: 128, .. }), "TrReader: 128-byte rows");
        let l = lane.lane();
        let key_a = b.let_i32(((&l & 15).lshr(3)) << 3);
        let key_b = b.let_i32(lane.half() << 2);
        let key_c = b.let_i32(l.lshr(1) & 3);
        let key_part = b.let_i32((key_a + key_b) + key_c);
        let m_gl = b.let_i32(l.lshr(4) & 1);
        let m_h8 = b.let_i32((&l & 1) << 3);
        let z = match sh.layout {
            Layout::Rows { swz: Swz::Fmha, .. } => {
                let s = fmha_swz(b, &key_part);
                b.let_i32(&m_gl ^ s)
            }
            _ => m_gl,
        };
        let trow = b.let_i32(add0((&key_part * 128) + m_h8, &I32::c(sh.base as i32)));
        let trx = std::array::from_fn(|e| b.let_i32(&trow + ((&z ^ (2 * e as i32)) << 4)));
        TrReader { sh: sh.clone(), trx }
    }
    pub fn at<'a>(&self, b: &mut FnBuilder, slot: &SlotRef<'a>) -> TrSlot<'a> {
        assert_eq!(slot.sh.name, self.sh.name);
        let off = slot.off_v(b);
        let trb = std::array::from_fn(|e| b.let_i32(&self.trx[e] + &off));
        TrSlot { slot: slot.clone(), trb }
    }
}
#[derive(Clone, Debug)]
pub struct TrSlot<'a> {
    pub slot: SlotRef<'a>,
    trb: [I32; 4],
}
impl TrSlot<'_> {
    /// A fragment of keys `64 hh ..` (4 x 16 rows), dim chunk group `e`: four `ds_read_b64_tr_b8` under one read marker.
    pub fn read(&self, b: &mut FnBuilder, hh: usize, e: usize) -> Frag<A32x32x64Fp8> {
        let a: [ValueId; 4] = std::array::from_fn(|r| b.bind_i32(&self.trb[e] + ((hh as i32) * 8192 + 2048 * r as i32)));
        self.slot.read(b, |b| Frag::wrap(b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3])))
    }
}

/// Reader of the per-row scale words of a 512-byte scale slot (row `j + 32 s` of the tile at `4 (j + 32 s)`).
#[derive(Clone, Debug)]
pub struct RowScaleReader {
    sh: Shared,
    cur: I32,
}
impl RowScaleReader {
    pub fn new(b: &mut FnBuilder, sh: &Shared, lane: &Lane) -> RowScaleReader {
        let cur = b.let_i32((lane.j() << 2) + sh.base as i32);
        RowScaleReader { sh: sh.clone(), cur }
    }
    /// Base address in `slot` (one add per tile).
    pub fn at<'a>(&self, b: &mut FnBuilder, slot: &SlotRef<'a>) -> (SlotRef<'a>, I32) {
        assert_eq!(slot.sh.name, self.sh.name);
        let off = slot.off_v(b);
        let a = b.let_i32(&self.cur + off);
        (slot.clone(), a)
    }
    /// The scale word of row group `s`.
    pub fn read(b: &mut FnBuilder, at: &(SlotRef, I32), s: usize) -> ValueId {
        let a = b.bind_i32(&at.1 + 128 * s as i32);
        at.0.read(b, |b| b.load_lds_b32(a))
    }
}

/// Where the QK accumulators go.
pub enum QkOut<'t> {
    /// in place into the loop variables (the steady loop), each first set to `ones x bias` (f16 MFMA) when a bias is given
    Vars(&'t AccTile<C32x32F32, 4>),
    /// as fresh values (the prologue), each started from `ones x bias` when a bias is given, else from zero; carries the (shared) zero Frag16
    Values(ValueId),
}

/// `S[s] (+)= K(rows 32 s ..) . Q^T` for s = 0..3 (`qk_block` for each s). Returns the 4 S values in the `Values` mode.
#[allow(clippy::too_many_arguments)]
pub fn qk_tile(b: &mut FnBuilder, out: QkOut, bias: Option<(ValueId, ValueId)>, k: &RowsSlot, kd: &(SlotRef, I32), q: &[Frag<B32x32x64Fp8>; 2], sq: [ValueId; 2], sel: &ScaleSel) -> Vec<ValueId> {
    (0..4usize).filter_map(|s| qk_block(b, &out, bias, k, kd, q, sq, sel, s)).collect()
}

/// One 32-key block `s` of the QK tile: the optional f16 bias MFMA, the two K fragments, the K-scale word, the scale bytes, two scaled MFMAs (the manual
/// `qk` closure body, statement for statement). The slot addresses (`k`, `kd`) are computed once by the caller (`RowsReader::at` / `RowScaleReader::at`)
/// and shared by the four blocks, which may be interleaved with other work (`_t128`: convert P(s), then QK(t+1, s)). Returns the S value in `Values` mode.
#[allow(clippy::too_many_arguments)]
pub fn qk_block(b: &mut FnBuilder, out: &QkOut, bias: Option<(ValueId, ValueId)>, k: &RowsSlot, kd: &(SlotRef, I32), q: &[Frag<B32x32x64Fp8>; 2], sq: [ValueId; 2], sel: &ScaleSel, s: usize) -> Option<ValueId> {
    let start: Option<ValueId> = match (out, bias) {
        (QkOut::Vars(t), Some((ones, w))) => {
            b.var_mfma_f16_z(t.accs[s].var, ones, w);
            None
        }
        (QkOut::Values(z), Some((ones, w))) => Some(b.mfma_f16_wide(ones, w, *z)),
        (QkOut::Values(z), None) => Some(*z),
        (QkOut::Vars(_), None) => None,
    };
    let [k0, k1] = k.read_pair(b, s);
    let kw = RowScaleReader::read(b, kd, s);
    let (sk0, sk1) = sel.pair(b, kw);
    match out {
        QkOut::Vars(t) => {
            mma_scaled(b, &t.accs[s], &k0, &q[0], sk0, sq[0]);
            mma_scaled(b, &t.accs[s], &k1, &q[1], sk1, sq[1]);
            None
        }
        QkOut::Values(_) => {
            let s0 = mma_scaled_v(b, &k0, &q[0], start.expect("start"), sk0, sq[0]);
            Some(mma_scaled_v(b, &k1, &q[1], s0, sk1, sq[1]))
        }
    }
}
/// `O[e] += V^T(keys 64 hh ..) . P[hh]` for hh = 0..1, e = 0..3 (8 scaled MFMAs, the V fragments read with the transposing reads).
pub fn pv_tile(b: &mut FnBuilder, o: &AccTile<C32x32F32, 4>, v: &TrSlot, p: &[Frag<B32x32x64Fp8>; 2], sa: impl Into<Opnd> + Copy, sb: impl Into<Opnd> + Copy) {
    for (hh, pf) in p.iter().enumerate() {
        for e in 0..4usize {
            let vf = v.read(b, hh, e);
            mma_scaled(b, &o.accs[e], &vf, pf, sa, sb);
        }
    }
}
