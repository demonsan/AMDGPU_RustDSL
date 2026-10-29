//! Real sequences written in the safe DSL.

use crate::*;

/// The `_vs128` PV + row-sum inner phase for one 32-key step: two scaled MFMAs accumulating P x V into one output fragment (the second
/// is a hardware-interlocked in-place chain), then the four-SMFMAC row-sum chain over the two P fragments (`idx01`, `idx23`), then the
/// running sum `l += row_sum[lane 2]`. Operand fragments are defined by `zeroed` here; in a kernel they come from DS reads / the S tile.
pub fn pv_rowsum_phase(s: &mut Seq) {
    // fragments and scales
    let pa = s.zeroed::<8>();
    let pb = s.zeroed::<8>();
    let v0 = s.zeroed::<8>();
    let v1 = s.zeroed::<8>();
    let sa = s.splat(1);
    let sb = s.splat(2);
    let a_rs = s.zeroed::<4>();
    let idx01 = s.splat(3);
    let idx23 = s.splat(4);
    let l_old = s.splat(0);

    // PV: o = pa x v0, then o += pb x v1 (in place, same opcode: no wait between)
    let mut o = s.vgpr::<16>();
    let (rpa, rv0, rsa, rsb) = (s.ready(&pa), s.ready(&v0), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(&mut o, rpa, rv0, rsa, rsb);
    let (rpb, rv1) = (s.ready(&pb), s.ready(&v1));
    let t = t.chain(s, rpb, rv1, rsa, rsb);
    t.retire(s);

    // row sums: r = sum over the four sparse index patterns (in place chain of four SMFMACs)
    let mut r = s.zeroed::<4>();
    let (ra, ri01, ri23) = (s.ready(&a_rs), s.ready(&idx01), s.ready(&idx23));
    let (rpa, rpb) = (s.ready(&pa), s.ready(&pb));
    let t = s.v_smfmac_f32_16x16x128_fp8_fp8(&mut r, ra, rpa, ri01);
    let t = t.chain(s, ra, rpb, ri01);
    let t = t.chain(s, ra, rpa, ri23);
    let t = t.chain(s, ra, rpb, ri23);
    t.retire(s);

    // l_new = l_old + r[2]
    let lane = s.lane::<4, 2>(&r);
    let mut l_new = s.vgpr::<1>();
    s.v_add_f32(&mut l_new, &l_old, lane);
}