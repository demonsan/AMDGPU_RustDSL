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
/// The `_vs128` row-sum chain exactly as the kernel runs it (`smfv`): the four SMFMACs accumulate IN PLACE into the Frag4 loop variable `la`
/// (dword 2 = 16 x row sum), ports = [inout la, a_rs, pa, pb, idx01, idx23]. The last SMFMAC stays in flight at exit.
pub fn rowsum_var_block() -> AsmBlock {
    let mut s = Seq::new();
    let acc = s.inout::<4>();
    let a_rs = s.input::<4>();
    let pa = s.input::<8>();
    let pb = s.input::<8>();
    let idx01 = s.input::<1>();
    let idx23 = s.input::<1>();
    let (ra, rpa, rpb, ri01, ri23) = (s.ready(&a_rs), s.ready(&pa), s.ready(&pb), s.ready(&idx01), s.ready(&idx23));
    let t = s.v_smfmac_f32_16x16x128_fp8_fp8(acc, ra, rpa, ri01);
    let t = t.chain(&mut s, ra, rpb, ri01);
    let t = t.chain(&mut s, ra, rpa, ri23);
    let t = t.chain(&mut s, ra, rpb, ri23);
    let ex = t.exit();
    s.output_exit(ex);
    s.finish().expect("rowsum_var_block is a verified sequence")
}

/// End-to-end runnable fragment (`probe_safeasm_e2e`): ports = [input pointer (V2), output pointer (V2)]. Per lane it loads a float4 from global
/// memory (vmcnt), runs a scaled MFMA with all-ones fp8 operands while the load is in flight (every D element = 64.0), waits for the load with the
/// partial count the scoreboard derives, sums the four floats, adds D[0] and stores the result.
pub fn global_probe_block() -> AsmBlock {
    let mut s = Seq::new();
    let p_in = s.input::<2>();
    let p_out = s.input::<2>();
    let d = s.vgpr::<4>();
    let pending = s.global_load_dwordx4(d, &p_in, 0);

    // independent work while the load is in flight: 0x38 is fp8 e4m3 1.0, scale 127 is 2^0
    let a = s.filled::<8>(0x3838_3838);
    let b = s.filled::<8>(0x3838_3838);
    let sa = s.splat(127);
    let sb = s.splat(127);
    let acc = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let acc = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(acc, ra, rb, rsa, rsb).retire(&mut s);

    let d = pending.wait(&mut s); // s_waitcnt vmcnt(0)
    let (l0, l1, l2, l3) = (s.lane::<4, 0>(&d), s.lane::<4, 1>(&d), s.lane::<4, 2>(&d), s.lane::<4, 3>(&d));
    let (mut x, mut y, mut t, mut r) = (s.vgpr::<1>(), s.vgpr::<1>(), s.vgpr::<1>(), s.vgpr::<1>());
    s.v_add_f32(&mut x, l0, l1);
    s.v_add_f32(&mut y, l2, l3);
    s.v_add_f32(&mut t, &x, &y);
    let c = s.lane::<16, 0>(&acc);
    s.v_add_f32(&mut r, c, &t);
    s.global_store_dword(&p_out, &r, 0);
    s.wait_vm_all();
    s.output(r);
    s.finish().expect("global_probe_block is a verified sequence")
}

/// The softmax conversion chain of one 8-score group, hand-authored: per pair `v_pk_add_f32` (shift), `v_pk_mul_f32` (scale), `v_cvt_pknorm_u16_f32`,
/// the svcorr correction (`v_lshrrev_b32 4`, `v_and_b32 mask`, `v_perm_b32`, `v_pk_sub_u16`), then two fp8 P words from the half-write converts
/// (`v_cvt_scalef32_pk_fp8_bf16` lo/hi on `Pw`), written to LDS with `ds_write_b32`, drained, and read back transposed with `ds_read_b64_tr_b8`.
/// Ports: inputs = [scores0 (V4), scores1 (V4), delta2 (V2), scale2 (V2), lds address (V1)]; output = the transposed pair (V2). Pinned.
pub fn cvt_chain_block() -> AsmBlock {
    let mut s = Seq::new();
    s.pin();
    let sc0 = s.input::<4>();
    let sc1 = s.input::<4>();
    let delta2 = s.input::<2>();
    let scale2 = s.input::<2>();
    let addr = s.input::<1>();
    let cmask = s.splat(0x0007_0007);
    let csel = s.splat(0x0c0c_0400);
    let cscale = s.splat(127);
    let mut words: Vec<Vgpr<1>> = Vec::new();
    for sc in [&sc0, &sc1] {
        let mut ys: Vec<Vgpr<1>> = Vec::new();
        for j in 0..2u8 {
            // pair (2j, 2j+1) of the score vector
            let mut p = s.vgpr::<2>();
            for (k, lane) in [2 * j, 2 * j + 1].into_iter().enumerate() {
                let src = match lane {
                    0 => s.lane::<4, 0>(sc),
                    1 => s.lane::<4, 1>(sc),
                    2 => s.lane::<4, 2>(sc),
                    _ => s.lane::<4, 3>(sc),
                };
                let dstv = if k == 0 { s.lane_mut::<2, 0>(&mut p) } else { s.lane_mut::<2, 1>(&mut p) };
                s.v_mov_b32(dstv, src);
            }
            let mut q = s.vgpr::<2>();
            s.v_pk_add_f32(&mut q, &p, &delta2);
            let mut r = s.vgpr::<2>();
            s.v_pk_mul_f32(&mut r, &q, &scale2);
            let mut u = s.vgpr::<1>();
            {
                let (lo, hi) = (s.lane::<2, 0>(&r), s.lane::<2, 1>(&r));
                s.v_cvt_pknorm_u16_f32(&mut u, lo, hi);
            }
            let mut v = s.vgpr::<1>();
            s.v_lshrrev_b32(&mut v, Imm::<4>::new(), &u);
            let mut m = s.vgpr::<1>();
            s.v_and_b32(&mut m, &cmask, &v);
            let mut x = s.vgpr::<1>();
            s.v_perm_b32(&mut x, &m, &m, &csel);
            let mut y = s.vgpr::<1>();
            s.v_pk_sub_u16(&mut y, &u, &x);
            ys.push(y);
        }
        let d = s.vgpr::<1>();
        let pw = s.partial(d);
        let pw = s.v_cvt_scalef32_pk_fp8_bf16_lo(pw, &ys[0], &cscale);
        let pw = s.v_cvt_scalef32_pk_fp8_bf16_hi(pw, &ys[1], &cscale);
        words.push(pw.done());
    }
    s.ds_write_b32::<0>(&addr, &words[0]);
    s.ds_write_b32::<4>(&addr, &words[1]);
    s.wait_lgkm_all();
    let t = s.vgpr::<2>();
    let t = s.ds_read_b64_tr_b8::<0, _>(t, &addr).wait(&mut s);
    s.output(t);
    s.finish().expect("cvt_chain_block is a verified sequence")
}
