//! L6a: online softmax with the reference-shift ("bias") + rollback recipe of the shipped FMHA (`_vs128` vsx default, kernels.rs `valu` closure).
//!
//! The scores carry a reference: every S tile starts as `ones x bias` (one f16 MFMA per 32 keys, `bias_mfma` / `qk_tile`), with the reference stored as
//! TWO f16 words `b = 4 (hi + lo)` (exact shift). Per 128-key tile ONE decision: `m = rowmax(S')` (max3 chain + partner exchange); when `m > thr` (any lane:
//! the branch has no exec mask, so non-rolling lanes see `delta = 0`) the reference moves to `b_q = quant(b + (b131 - m))` and the state is rescaled:
//! `S += delta` (`v_pk_add_f32`), `O *= 2^(512 delta)` (`v_pk_mul_f32`), `l *= 2^(512 delta)`. P = fp8(S') via pknorm (+ LUT correction in the short copy);
//! the row sum `l` accumulates with four smfmacs (dword 2 = 16 x l).

use super::frag::{AccTile, C32x32F32, C16x16F32};
use super::reduce::{masked_values, row_max, to_p8, KeyMask, LutConsts, P8Conv, SmfmacOnes};
use super::Lane;
use crate::{DType, FnBuilder, Op, Opnd, ValueId, VarId};

/// Recipe parameters (the shipped default: `Recipe::vsx(corr)`).
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    /// rollback threshold on the scaled max (120/512)
    pub thr: f32,
    /// target reference offset (131/512)
    pub b131: f32,
    /// LUT correction of the fp8 conversion (short sequences)
    pub lut: bool,
}
impl Recipe {
    pub fn vsx(lut: bool) -> Recipe {
        Recipe { thr: 120.0 / 512.0, b131: 131.0 / 512.0, lut }
    }
}

/// Constants + loop state of the online softmax. Create it outside the loop (`new`), register `state()` with the pipeline and `bind` the variables.
pub struct OnlineSoftmax {
    pub recipe: Recipe,
    pub smf: SmfmacOnes,
    pub ones_f16: ValueId,
    pub bias_init: ValueId,
    neg_big: ValueId,
    zero_f: ValueId,
    thr: ValueId,
    b131: ValueId,
    c16: ValueId,
    c512: ValueId,
    one_f: ValueId,
    zero4: ValueId,
    /// loop variables: the two-word bias (Frag2) and the smfmac row-sum accumulator (Frag4)
    pub bias: Option<VarId>,
    pub la: Option<VarId>,
}

impl OnlineSoftmax {
    pub fn new(b: &mut FnBuilder, lane: &Lane, recipe: Recipe) -> OnlineSoftmax {
        let neg_big = b.c_f32(-1.0e30);
        let thr = b.c_f32(recipe.thr);
        let b131 = b.c_f32(recipe.b131);
        let c512 = b.c_f32(512.0);
        let one_f = b.c_f32(1.0);
        let zero_f = b.c_f32(0.0);
        let c16 = b.c_i32(16);
        let smf = SmfmacOnes::new(b, lane);
        let oh = b.c_i32(0x3C00_3C00);
        let ones_f16 = b.frag2(oh, oh);
        let wi = b.c_i32(0x4800_4800);
        let bias_init = b.frag2(wi, wi);
        let zero4 = b.frag_zero(DType::Frag4);
        OnlineSoftmax { recipe, smf, ones_f16, bias_init, neg_big, zero_f, thr, b131, c16, c512, one_f, zero4, bias: None, la: None }
    }
    /// Loop-state initial values: [bias (Frag2), row sums (Frag4)].
    pub fn state(&self) -> Vec<(DType, Opnd)> {
        vec![(DType::Frag2, Opnd::Val(self.bias_init)), (DType::Frag4, Opnd::Val(self.zero4))]
    }
    pub fn bind(&mut self, vars: &[VarId]) {
        self.bias = Some(vars[0]);
        self.la = Some(vars[1]);
    }
    /// The row-sum accumulator as a typed variable.
    pub fn l_acc(&self) -> VarId {
        self.la.expect("OnlineSoftmax: state not bound")
    }

    /// The decision of one tile: row max, the new reference word (an SSA value, also written to the bias variable exactly once), and the rollback branch
    /// rescaling `s`, `o` and the row sum. Returns the new bias word (the C input of the next tile's bias MFMAs).
    pub fn decide(&self, b: &mut FnBuilder, s: &AccTile<C32x32F32, 4>, o: &AccTile<C32x32F32, 4>, mask: Option<&KeyMask>) -> ValueId {
        let bias2 = self.bias.expect("OnlineSoftmax: state not bound");
        let mut flat: Vec<ValueId> = Vec::with_capacity(64);
        for i in 0..4usize {
            flat.extend(masked_values(b, s, i, mask, self.neg_big));
        }
        let mt = row_max(b, &flat);
        let grow = b.cmp_lt_f32(self.thr, mt);
        // two-word reference: b_old = 4 (hi + lo); b_new = b_old + (b131 - m); quantised again into two f16 words
        let w_old_u = b.frag2_lane(Opnd::Var(bias2), 0);
        let w_old = b.bitcast_i32(w_old_u);
        let hi_o = b.let_(Op::CvtF16ToF32(Opnd::Val(w_old)));
        let lo_s = b.let_(Op::LshrI32(Opnd::Val(w_old), Opnd::Val(self.c16)));
        let lo_o = b.let_(Op::CvtF16ToF32(Opnd::Val(lo_s)));
        let c4f = b.c_f32(4.0);
        let c025 = b.c_f32(0.25);
        let sum_o = b.add_f32(hi_o, lo_o);
        let b_old = b.mul_f32(sum_o, c4f);
        let tt = b.sub_f32(self.b131, mt);
        let b_new = b.add_f32(b_old, tt);
        let q = b.mul_f32(b_new, c025);
        let h_raw = b.let_(Op::CvtF32ToF16(Opnd::Val(q)));
        let hq = b.let_(Op::CvtF16ToF32(Opnd::Val(h_raw)));
        let r = b.sub_f32(q, hq);
        let l_raw = b.let_(Op::CvtF32ToF16(Opnd::Val(r)));
        let lq = b.let_(Op::CvtF16ToF32(Opnd::Val(l_raw)));
        let sum_q = b.add_f32(hq, lq);
        let b_q = b.mul_f32(sum_q, c4f);
        let d_full = b.sub_f32(b_q, b_old);
        let delta = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(grow), a: Opnd::Val(d_full), b: Opnd::Val(self.zero_f) });
        let l_sh = b.shl_i32(l_raw, self.c16);
        let w_q = b.let_(Op::AddI32(Opnd::Val(h_raw), Opnd::Val(l_sh)));
        let w_new = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(grow), a: Opnd::Val(w_q), b: Opnd::Val(w_old) });
        let nb = b.frag2(w_new, w_new);
        b.set_var(bias2, nb);
        b.begin_if(Opnd::Val(grow));
        {
            let dupd = b.frag2_dup(delta);
            for a in &s.accs {
                b.var_add_pk(a.var, dupd);
            }
            let ex = b.mul_f32(delta, self.c512);
            let corr = b.exp2_f32(ex);
            let cd = b.frag2_dup(corr);
            for a in &o.accs {
                b.var_scale_pk(a.var, cd);
            }
            b.var_scale_lane(self.l_acc(), 2, corr);
        }
        b.end_if();
        nb
    }

    /// P words of S accumulators `t0..t1` (4 words each; invalid keys contribute 0).
    pub fn convert(&self, b: &mut FnBuilder, s: &AccTile<C32x32F32, 4>, mask: Option<&KeyMask>, t0: usize, t1: usize) -> Vec<ValueId> {
        let lut = if self.recipe.lut { Some(LutConsts::new(b)) } else { None };
        let conv = if self.recipe.lut { P8Conv::PkNormLut } else { P8Conv::PkNorm };
        let mut words = Vec::with_capacity(4 * (t1 - t0));
        for i in t0..t1 {
            let xs = masked_values(b, s, i, mask, self.zero_f);
            words.extend(to_p8(b, &xs, self.one_f, conv, lut.as_ref()));
        }
        words
    }

    /// The row sums of this tile's P (two P fragments of 8 words).
    pub fn row_sum(&self, b: &mut FnBuilder, pa: ValueId, pb: ValueId) {
        self.smf.row_sum(b, self.l_acc(), pa, pb);
    }

    /// The final row sum `l` (after the loop): dword 2 of the smfmac accumulator / 16.
    pub fn l_total(&self, b: &mut FnBuilder) -> ValueId {
        let d2 = b.frag4_lane(Opnd::Var(self.l_acc()), 2);
        let lf = b.bitcast_f32(d2);
        let c = b.c_f32(0.0625);
        b.mul_f32(lf, c)
    }

    /// `log2(l) - 512 b` (the split-independent log-sum-exp of the KV-split partials: P = 2^(x + 512 b - 127)).
    pub fn lse(&self, b: &mut FnBuilder, l_tot: ValueId) -> ValueId {
        let bias2 = self.bias.expect("OnlineSoftmax: state not bound");
        let bw_u = b.frag2_lane(Opnd::Var(bias2), 0);
        let bw = b.bitcast_i32(bw_u);
        let hi_f = b.let_(Op::CvtF16ToF32(Opnd::Val(bw)));
        let lo_s = b.let_(Op::LshrI32(Opnd::Val(bw), Opnd::Val(self.c16)));
        let lo_f = b.let_(Op::CvtF16ToF32(Opnd::Val(lo_s)));
        let hl = b.add_f32(hi_f, lo_f);
        let c2048 = b.c_f32(2048.0);
        let b512 = b.mul_f32(hl, c2048);
        let lg = b.log2_f32(l_tot);
        let cneg = b.c_f32(-1.0);
        let nb = b.mul_f32(b512, cneg);
        b.add_f32(lg, nb)
    }
}

/// Marker type use (keeps the smfmac accumulator layout visible in signatures that take `la`).
pub type RowSumAcc = super::frag::Acc<C16x16F32>;
