//! Real sequences written in the safe DSL.

use crate::*;

/// The `_vs128` PV + row-sum inner phase for one 32-key step (standalone: operands defined by `zeroed`): two scaled MFMAs accumulating
/// P x V into one output fragment (the second is a hardware-interlocked in-place chain), then the four-SMFMAC row-sum chain over the two P
/// fragments (`idx01`, `idx23`), then the running sum `l += row_sum[lane 2]`.
pub fn pv_rowsum_phase(s: &mut Seq) {
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
    // the `_zero` form is a different opcode from the accumulate form: its result needs the full latency before the dependent MFMA (retire)
    let o = s.vgpr::<16>();
    let (rpa, rv0, rsa, rsb) = (s.ready(&pa), s.ready(&v0), s.ready(&sa), s.ready(&sb));
    let o = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, rpa, rv0, rsa, rsb).retire(s);
    let (rpb, rv1) = (s.ready(&pb), s.ready(&v1));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4(o, rpb, rv1, rsa, rsb);
    let _o = t.retire(s);

    // row sums: r = sum over the four sparse index patterns (in place chain of four SMFMACs)
    let r = s.zeroed::<4>();
    let (ra, ri01, ri23) = (s.ready(&a_rs), s.ready(&idx01), s.ready(&idx23));
    let (rpa, rpb) = (s.ready(&pa), s.ready(&pb));
    let t = s.v_smfmac_f32_16x16x128_fp8_fp8(r, ra, rpa, ri01);
    let t = t.chain(s, ra, rpb, ri01);
    let t = t.chain(s, ra, rpa, ri23);
    let t = t.chain(s, ra, rpb, ri23);
    let r = t.retire(s);

    // l_new = l_old + r[2]
    let lane = s.lane::<4, 2>(&r);
    let mut l_new = s.vgpr::<1>();
    s.v_add_f32(&mut l_new, &l_old, lane);
}

/// The kernel-embedded form of the row-sum chain (what `_vs128` runs per tile): inputs are DSL values (sparse A fragment, the two P
/// fragments, the two index words); the output is the 4-dword accumulator, handed back with its last SMFMAC still in flight so the
/// compiler's hazard pass pads/fills the consumer exactly as it does for the DSL-built chain.
pub fn rowsum_chain_block() -> AsmBlock {
    let mut s = Seq::new();
    let a_rs = s.input::<4>();
    let pa = s.input::<8>();
    let pb = s.input::<8>();
    let idx01 = s.input::<1>();
    let idx23 = s.input::<1>();
    let r = s.zeroed::<4>();
    let (ra, rpa, rpb, ri01, ri23) = (s.ready(&a_rs), s.ready(&pa), s.ready(&pb), s.ready(&idx01), s.ready(&idx23));
    let t = s.v_smfmac_f32_16x16x128_fp8_fp8(r, ra, rpa, ri01);
    let t = t.chain(&mut s, ra, rpb, ri01);
    let t = t.chain(&mut s, ra, rpa, ri23);
    let t = t.chain(&mut s, ra, rpb, ri23);
    let ex = t.exit();
    s.output_exit(ex);
    s.finish().expect("rowsum_chain_block is a verified sequence")
}