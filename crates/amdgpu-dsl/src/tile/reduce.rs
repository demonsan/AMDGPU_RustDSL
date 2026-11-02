//! L5: elementwise operations and reductions on accumulator tiles (the VALU phase vocabulary): key masks derived from the C32x32 layout, the row maximum
//! (`v_max3_f32` chain + partner exchange), the fp8 P conversion (pknorm + scalef32, optionally with the LUT correction) and the smfmac row sums.

use super::frag::{AccTile, C32x32F32};
use super::pipeline::TileIt;
use super::{ev, Lane};
use crate::{DType, FnBuilder, Op, Opnd, ValueId, VarId};

/// Validity of the keys of an S tile (rows of the C32x32 accumulators) against the number of valid keys: key of value r of accumulator s =
/// `key0 + 4 (lane >> 5) + 32 s + 8 (r >> 2) + (r & 3)`.
#[derive(Clone, Debug)]
pub struct KeyMask {
    kb: [ValueId; 4],
    limit: ValueId,
}

impl KeyMask {
    /// The mask of the current tile of `t` (`key0 = it * tile_rows`) against `limit` valid keys; `half_base` = `C32x32F32::half_base(lane)` bound once per kernel.
    pub fn new(b: &mut FnBuilder, t: &TileIt, half_base: ValueId, limit: ValueId) -> KeyMask {
        let kb = std::array::from_fn(|s| {
            let k0 = b.let_i32(t.key0());
            let k1 = b.let_i32(k0 + ev(half_base));
            b.bind_i32(k1 + 32 * s as i32)
        });
        KeyMask { kb, limit }
    }
    /// `key(s, r) < limit` (one add + one compare per call, like the manual `valid`).
    pub fn valid(&self, b: &mut FnBuilder, s: usize, r: u8) -> ValueId {
        b.bind_cond((ev(self.kb[s]) + C32x32F32::row_const(r)).lt(ev(self.limit)))
    }
}

/// The 16 values of S accumulator `s`, the invalid keys replaced by `fill` when a mask is given.
pub fn masked_values(b: &mut FnBuilder, t: &AccTile<C32x32F32, 4>, s: usize, mask: Option<&KeyMask>, fill: ValueId) -> Vec<ValueId> {
    (0..16u8)
        .map(|r| {
            let l = b.frag16_lane(Opnd::Var(t.accs[s].var), r);
            let v = b.bitcast_f32(l);
            match mask {
                Some(m) => {
                    let ok = m.valid(b, s, r);
                    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(fill) })
                }
                None => v,
            }
        })
        .collect()
}

/// Maximum over the lane's values and its partner lane (`l ^ 32`, the other half of the same query column): `v_max3_f32` chain + one exchange.
pub fn row_max(b: &mut FnBuilder, flat: &[ValueId]) -> ValueId {
    assert!(flat.len() >= 3);
    let mut m = b.max3_f32(flat[0], flat[1], flat[2]);
    let mut k = 3usize;
    while k + 1 < flat.len() {
        m = b.max3_f32(m, flat[k], flat[k + 1]);
        k += 2;
    }
    if k < flat.len() {
        m = b.max_f32(m, flat[k]);
    }
    b.partner_max_f32(m)
}

/// fp8 conversion of P.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum P8Conv {
    /// `v_cvt_pknorm_u16_f32` pairs + scalef32 conversion (one word per 4 values)
    PkNorm,
    /// with the LUT correction of the pknorm exponent rounding (short sequences): pknorm pair -> lshr/and -> perm (LUT) -> pk_sub_u16 -> scalef32
    PkNormLut,
}

/// Constants of the LUT correction (materialised once per call site, like the manual VALU phase).
#[derive(Clone, Copy, Debug)]
pub struct LutConsts {
    c4: ValueId,
    mask: ValueId,
    lo: ValueId,
    hi: ValueId,
}
impl LutConsts {
    pub fn new(b: &mut FnBuilder) -> LutConsts {
        let c4 = b.c_i32(4);
        let mask = b.c_i32(0x0007_0007);
        let lo = b.c_i32(i32::from_le_bytes([0, 6, 9, 11]));
        let hi = b.c_i32(i32::from_le_bytes([11, 10, 7, 3]));
        LutConsts { c4, mask, lo, hi }
    }
}

/// Converts 16 f32 values to 4 fp8 words (scale = the scalef32 operand).
pub fn to_p8(b: &mut FnBuilder, xs: &[ValueId], scale: ValueId, conv: P8Conv, lut: Option<&LutConsts>) -> Vec<ValueId> {
    assert_eq!(xs.len(), 16);
    (0..4usize)
        .map(|m| match conv {
            P8Conv::PkNorm => b.cvt_pknorm_p8x4(xs[4 * m], xs[4 * m + 1], xs[4 * m + 2], xs[4 * m + 3], scale),
            P8Conv::PkNormLut => {
                let c = lut.expect("P8Conv::PkNormLut needs LutConsts");
                let mut us: Vec<ValueId> = Vec::with_capacity(2);
                for h in 0..2usize {
                    let u = b.cvt_pknorm_pair(xs[4 * m + 2 * h], xs[4 * m + 2 * h + 1]);
                    let s1 = b.let_(Op::LshrI32(Opnd::Val(u), Opnd::Val(c.c4)));
                    let s2 = b.let_(Op::AndI32(Opnd::Val(s1), Opnd::Val(c.mask)));
                    let pm = b.perm_b32(c.hi, c.lo, s2);
                    us.push(b.pk_sub_u16(u, pm));
                }
                b.cvt_p8x4_from_u16(us[0], us[1], scale)
            }
        })
        .collect()
}

/// The per-lane A operand of the smfmac row sums (E4M3 ones in the pattern that sums the lane's own query: vendor-spec 4.5.3) and the two index words.
#[derive(Clone, Copy, Debug)]
pub struct SmfmacOnes {
    pub a: ValueId,
    pub idx01: ValueId,
    pub idx23: ValueId,
}
impl SmfmacOnes {
    pub fn new(b: &mut FnBuilder, lane: &Lane) -> SmfmacOnes {
        let ones = b.c_i32(0x3838_3838);
        let z = b.c_i32(0);
        let b0 = b.let_i32(lane.lane() & 3);
        let b1 = b.let_i32(lane.lane() & 4);
        let lt2 = b.bind_cond(b0.lt(2));
        let b1z = b.bind_cond(b1.lt(1));
        let v0 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lt2), a: Opnd::Val(z), b: Opnd::Val(ones) });
        let a01 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(b1z), a: Opnd::Val(z), b: Opnd::Val(v0) });
        let a23 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(b1z), a: Opnd::Val(v0), b: Opnd::Val(z) });
        let a = b.frag4(a23, a23, a01, a01);
        let idx01 = b.c_i32(0x4444_4444_u32 as i32);
        let idx23 = b.c_i32(0xeeee_eeee_u32 as i32);
        SmfmacOnes { a, idx01, idx23 }
    }
    /// `la += rowsum(P)`: four in-place `v_smfmac_f32_16x16x128_fp8_fp8` over the two P fragments (dword 2 of `la` = 16 x the row sum).
    pub fn row_sum(&self, b: &mut FnBuilder, la: VarId, pa: ValueId, pb: ValueId) {
        b.var_smfmac_fp8(la, self.a, pa, self.idx01);
        b.var_smfmac_fp8(la, self.a, pb, self.idx01);
        b.var_smfmac_fp8(la, self.a, pa, self.idx23);
        b.var_smfmac_fp8(la, self.a, pb, self.idx23);
    }
}
