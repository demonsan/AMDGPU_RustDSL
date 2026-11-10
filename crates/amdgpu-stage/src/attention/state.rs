//! The attention family's per-block runtime state (docs/l2-family-bodies.md §2.1): `FmhaBlock` (block / lane ids, clamped query row, resident Q fragments and scales, GQA head, KV-split key range, the
//! per-tensor FP8 scales), the KV-split kernarg data (`KvsArgs`) and the small accessors the bodies use. The ABI itself (kernarg offsets, host contract) stays in the kernel crate (`FmhaIo`), which
//! builds these through `block3` and exposes its operands to the bodies through the `Operands` trait.

use amdgpu_dsl::{FnBuilder, ValueId, I32};
use amdgpu_tile::frag::*;
use amdgpu_tile::*;

/// KV-split kernarg extension (0x270 Ohat, 0x278 lse, 0x280 splits, 0x284 rows_total = B*S*H; grid z = B * splits).
pub struct KvsArgs {
    pub ohat: ValueId,
    pub lse: ValueId,
    pub splits: ValueId,
    pub rows_total: ValueId,
    /// kernarg offset of `splits` (the contract symbol of the split count)
    pub splits_off: u32,
}

pub struct FmhaBlock {
    pub lane: Lane,
    pub bidx: ValueId,
    pub hidx: ValueId,
    pub kvh: ValueId,
    /// first query row of this wave
    pub row0: I32,
    /// this lane's query row clamped to S - 1, and S - 1
    pub row_c: I32,
    pub s_last: I32,
    /// keys of this block (relative to its first key `key_lo`) / number of 128-key tiles; tiles of the whole sequence
    pub kv_len: ValueId,
    /// the K / V copies clamp their rows to `kv_clamp - 1` (= `kv_len`; the paired causal form: all keys, its ring serves both passes)
    pub kv_clamp: ValueId,
    pub key_lo: I32,
    pub nkv: ValueId,
    pub ntiles_total: ValueId,
    /// KV split index of this block
    pub split: Option<ValueId>,
    /// causal KV split: this split lies past the block's key range (1 / 0); its partial is discarded (log-sum-exp -inf)
    pub split_empty: Option<ValueId>,
    pub q: RowFrags,
    pub sq: [ValueId; 2],
    pub sel: ScaleSel,
    pub vd: ValueId,
    /// `4 (lane >> 5)`: key offset of this lane half in a C32x32 accumulator
    pub g4: ValueId,
    /// per-tensor FP8 scales (`fp8pt`)
    pub pt: Option<PtScale>,
}

/// Per-tensor FP8 attention (`fmha_fwd_hd128_fp8pt_v3`): the softmax exponent factor `c = scale log2(e) q_ds k_ds = 2^e m` (m in [1, 2)). 2^e rides
/// the E8M0 Q scale of the scaled QK MFMA (the K scale is the unit word), m multiplies S in fp32 before the softmax; the reference words entering
/// the bias MFMA encode `b / m`, so `m S` carries the reference `b` of the recipe.
#[derive(Clone, Copy, Debug)]
pub struct PtScale {
    /// (m, m)
    pub m2: ValueId,
    /// 1 / m
    pub inv_m: ValueId,
    /// the unit E8M0 K-scale word 0x7F7F7F7F
    pub kunit: ValueId,
}

impl FmhaBlock {
    /// A K/V-shaped tensor (rows = keys) as a tile of this block's KV head, rows clamped to the last key.
    pub fn kv_tile(&self, b: &mut FnBuilder, t: &Tensor) -> TileView {
        let last = b.let_i32(ev(self.kv_clamp) - 1);
        t.tile(vec![ev(self.bidx), self.key_lo.clone(), ev(self.kvh), I32::c(0)]).clamp(1, ev(self.kv_clamp), last)
    }
}


impl FmhaBlock {
    /// The recipe copy: sequences of fewer than 4 tiles of 128 keys (the WHOLE sequence, also for a KV split) keep the exact LUT-corrected recipe
    /// (from 2 tiles on, the vendor recipe missed parity at S=300 SK=129: 1.31x the vendor's mean error).
    pub fn lut_copy(&self, b: &mut FnBuilder) -> ValueId {
        b.bind_cond(ev(self.ntiles_total).lt(4))
    }
    /// The output scale `V descale / l`.
    pub fn out_scale(&self, b: &mut FnBuilder, l: ValueId) -> ValueId {
        let inv_l = b.rcp_f32(l);
        b.mul_f32(inv_l, self.vd)
    }
}

