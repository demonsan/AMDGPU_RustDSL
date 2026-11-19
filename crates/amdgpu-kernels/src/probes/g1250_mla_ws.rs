//! gfx1250 MLA decode, wave-split (`mla_g1250_qh16_ws`): the 4 waves of a work-group split the sequence's KV pages (wave w: pages w, w + 4, ..),
//! each with its own private 2-slot page ring, its own online softmax and its own O accumulator over all 512 V dims - no barrier and no
//! cross-wave LDS traffic inside the loop. One LDS merge of the four (max, sum, O) partials at the end (flash-decoding inside the work-group).
//! Same ABI, layouts and host contract as `mla_g1250_qh16` (crates/amdgpu-kernels/src/probes/g1250_mla.rs).
//!
//! Per page and wave: S^T for the page's 64 tokens as 4 WMMA tiles of 16 tokens (9 fp8 WMMA 16x16x64 each: lane (m, g) holds head m, tokens
//! 16 j + 8 g + r); those 4 x 8 values per lane are exactly the A fragment of P (16 heads x 64 tokens), so P never leaves the registers; PV
//! over the 32 tiles of 16 V dims (32 WMMA, V by `ds_load_tr8_b64`). The row maximum combines the two lane halves with `v_permlane16_swap`;
//! the rescale factor of the O rows (heads 8 g + r) is transposed through a wave-private LDS hop.

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, TdmCpol, TensorTile, ValueId, VarId, WmmaKind};
use amdgpu_stage::attention::mla::{PageLayout, DV, PAGE_BYTES, PAGE_TOKENS, ROPE_OFF};
use amdgpu_stage::gemm::tdm::tdm_group0;
use amdgpu_stage::weave::{Units, Weaver};
use amdgpu_tile::shared::{Layout, LdsPlan, SlotIdx};

pub const MLA_WS_NAME: &str = "mla_g1250_qh16_ws";
const MAX_PAGES: i32 = 1024;
/// V dim tiles of 16 (the whole 512)
const NT: usize = 32;
/// workspace bytes per (sequence, split) of the KV-split form: O 16 x 512 f32, then the 16 maxima and the 16 sums
pub const WS_BYTES: i64 = (16 * 512 + 32) * 4;

fn branch_bits_lean(bits: u32) -> bool {
    let lean = bits & 65536 != 0;
    assert!(!lean || bits & 256 != 0, "the lean softmax needs the tail form (no fully masked pages)");
    lean
}

fn f32_of(b: &mut FnBuilder, v: ValueId) -> ValueId {
    b.let_(Op::BitcastF32(Opnd::Val(v)))
}

pub fn mla_g1250_qh16_ws() -> Function {
    // the default: scalar loads for the sequence bounds / scales / page ids, full page pairs + an odd tail page
    mla_ws_with(MLA_WS_NAME, false, 64 | 256 | 16384 | 65536, false)
}

/// `_ws` with the per-page mask and the O rescale branch-free (a select on every page; the rescale on every page).
pub fn mla_g1250_qh16_wsb() -> Function {
    mla_ws_with("mla_g1250_qh16_wsb", false, 3, false)
}

/// bisect aids: only the mask (`_wsbm`) / only the rescale (`_wsbr`) branch-free
/// The cross-page form: page i's softmax and page i - 1's PV in one Free plan phase (interleave hint for the back-end Weaver).
pub fn mla_g1250_qh16_wsp() -> Function {
    mla_ws_with("mla_g1250_qh16_wsp", false, 1, true)
}

/// The intra-page form: per page, the rescale of each group of 4 accumulators interleaved with the previous group's PV WMMAs (branch-free
/// rescale with a select-free row factor; one Free plan phase per page with interleave hints).
pub fn mla_g1250_qh16_wsi() -> Function {
    mla_ws_with("mla_g1250_qh16_wsi", false, 1 | 16, false)
}

/// The KV split across work-groups (grid (S, B)): partials to the workspace; `mla_g1250_qh16_cmb` combines them.
pub fn mla_g1250_qh16_wsk() -> Function {
    mla_ws_with("mla_g1250_qh16_wsk", false, 1 | 32, false)
}

/// The combine of the KV-split partials: grid (1, B), thread t = head t / 8, dims 64 (t % 8) ..: max, rescaled sums, bf16 R and the LSE.
pub fn mla_g1250_qh16_cmb() -> Function {
    let mut b = FnBuilder::new("mla_g1250_qh16_cmb", 128);
    let names = ["R", "LSE", "Q", "KV", "LTP", "LTD", "LTL", "QTP", "QS", "KS", "VSC"];
    for (i, n) in names.iter().enumerate() {
        b.arg(n, DType::Ptr, 8 * i as u32, 8);
    }
    for (i, n) in ["out16", "scalar", "qlens", "passes", "stride_page", "log2_page", "stride_q", "flags"].iter().enumerate() {
        b.arg(n, DType::I32, 88 + 4 * i as u32, 4);
    }
    b.arg_tensor("R", Ext::grid_y() * Ext::c(16 * 512 * 2));
    b.arg_tensor("LSE", Ext::grid_y() * Ext::c(16 * 4));
    b.arg_tensor("KS", Ext::c(4));
    b.ka_one_of(100, &[1, 2, 4, 8]);
    b.arg_tensor("VSC", Ext::grid_y() * Ext::ka(100) * Ext::c(WS_BYTES));
    let pr = b.let_(Op::KernargPtr { offset: 0 });
    let plse = b.let_(Op::KernargPtr { offset: 8 });
    let pks = b.let_(Op::KernargPtr { offset: 72 });
    let pws = b.let_(Op::KernargPtr { offset: 80 });
    let s_count = b.let_(Op::KernargI32 { offset: 100 });
    let tid = b.let_(Op::WorkitemId);
    let bidx = b.let_(Op::WorkgroupIdY);
    let z = b.c_i32(0);
    let kq = b.let_(Op::PtrAdd { base: Opnd::Val(pks), byte_off: Opnd::Val(z) });
    let ksv = b.load_global(kq, 0, MemTy::B32);
    let ks = b.read_first_lane(ksv);
    let ksf = f32_of(&mut b, ks);
    let h = b.bind_i32(ev(tid) >> 3);
    let d0 = b.bind_i32((ev(tid) & 7) << 6);
    // pass 1: the maximum over the splits
    let neg = b.c_f32(-1.0e30);
    let zf = b.c_f32(0.0);
    let mut inits = vec![(DType::F32, Opnd::Val(neg)), (DType::F32, Opnd::Val(zf))];
    for _ in 0..64 {
        inits.push((DType::F32, Opnd::Val(zf)));
    }
    let st = b.declare_vars(&inits);
    let (vmx, vl) = (st[0], st[1]);
    let ov: Vec<VarId> = st[2..].to_vec();
    let z0 = b.c_i32(0);
    let lv = b.begin_loop(&[(DType::I32, Opnd::Val(z0))]);
    let si = lv[0];
    b.assume(si, Ext::c(0), Ext::c(7));
    let smax = b.bind_i32(ev(s_count) - 1);
    let sic = b.let_(Op::MinI32(Opnd::Var(si), Opnd::Val(smax)));
    let wsb = b.bind_i32((ev(bidx) * ev(s_count) + ev(sic)) * WS_BYTES as i32);
    let wm = b.bind_i32(ev(wsb) + (8192 + ev(h)) * 4);
    let pwm = b.let_(Op::PtrAdd { base: Opnd::Val(pws), byte_off: Opnd::Val(wm) });
    let msv = b.load_global(pwm, 0, MemTy::B32);
    let msf = f32_of(&mut b, msv);
    let m2 = b.max_f32(Opnd::Var(vmx), msf);
    b.set_var(vmx, Opnd::Val(m2));
    let s1 = b.bind_i32(amdgpu_dsl::varx(si) + 1);
    b.set_var(si, Opnd::Val(s1));
    let f1 = b.cmp(CmpPred::Ge, CmpTy::I32, s1, s_count);
    b.break_if(f1);
    b.end_loop();
    // pass 2: the rescaled sums and O
    let z1 = b.c_i32(0);
    let lv2 = b.begin_loop(&[(DType::I32, Opnd::Val(z1))]);
    let sj = lv2[0];
    b.assume(sj, Ext::c(0), Ext::c(7));
    let sjc = b.let_(Op::MinI32(Opnd::Var(sj), Opnd::Val(smax)));
    let wsb2 = b.bind_i32((ev(bidx) * ev(s_count) + ev(sjc)) * WS_BYTES as i32);
    let wm2 = b.bind_i32(ev(wsb2) + (8192 + ev(h)) * 4);
    let pwm2 = b.let_(Op::PtrAdd { base: Opnd::Val(pws), byte_off: Opnd::Val(wm2) });
    let msv2 = b.load_global(pwm2, 0, MemTy::B32);
    let lsv2 = b.load_global(pwm2, 64, MemTy::B32);
    let msf2 = f32_of(&mut b, msv2);
    let lsf2 = f32_of(&mut b, lsv2);
    let dd = b.sub_f32(msf2, Opnd::Var(vmx));
    let sc = b.exp2_f32(dd);
    let lt = b.mul_f32(lsf2, sc);
    let l2 = b.add_f32(Opnd::Var(vl), lt);
    b.set_var(vl, Opnd::Val(l2));
    let wo = b.bind_i32(ev(wsb2) + (ev(h) * 512 + ev(d0)) * 4);
    let pwo = b.let_(Op::PtrAdd { base: Opnd::Val(pws), byte_off: Opnd::Val(wo) });
    for k in 0..16u32 {
        let v4 = b.load_global(pwo, 16 * k, MemTy::B128);
        for e in 0..4u8 {
            let x = b.frag_slice(Opnd::Val(v4), e, 1);
            let xf = f32_of(&mut b, x);
            let t = b.mul_f32(xf, sc);
            let i = 4 * k as usize + e as usize;
            let y = b.add_f32(Opnd::Var(ov[i]), t);
            b.set_var(ov[i], Opnd::Val(y));
        }
    }
    let s2 = b.bind_i32(amdgpu_dsl::varx(sj) + 1);
    b.set_var(sj, Opnd::Val(s2));
    let f2 = b.cmp(CmpPred::Ge, CmpTy::I32, s2, s_count);
    b.break_if(f2);
    b.end_loop();
    let rl = b.rcp_f32(Opnd::Var(vl));
    let inv = b.mul_f32(rl, ksf);
    let rb_off = b.bind_i32((ev(bidx) * 16 + ev(h)) * 1024 + ev(d0) * 2);
    let prb = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(rb_off) });
    for k in 0..8u32 {
        let parts: Vec<Opnd> = (0..4)
            .map(|i| {
                let a = b.mul_f32(Opnd::Var(ov[8 * k as usize + 2 * i]), inv);
                let c = b.mul_f32(Opnd::Var(ov[8 * k as usize + 2 * i + 1]), inv);
                Opnd::Val(b.cvt_pk_bf16_f32(a, c))
            })
            .collect();
        let v = b.frag_cat(&parts);
        b.store_global(prb, 16 * k, v, MemTy::B128, false);
    }
    let lg = b.log2_f32(Opnd::Var(vl));
    let s3 = b.add_f32(Opnd::Var(vmx), lg);
    let ln2 = b.c_f32(std::f32::consts::LN_2);
    let lse = b.mul_f32(s3, ln2);
    let lo = b.bind_i32((ev(bidx) * 16 + ev(h)) * 4);
    let pl = b.let_(Op::PtrAdd { base: Opnd::Val(plse), byte_off: Opnd::Val(lo) });
    b.store_global(pl, 0, lse, MemTy::B32, false);
    b.finish()
}

/// `_ws` with scalar loads for the sequence bounds, scales and page ids (no LDS page table, no prologue barrier).
pub fn mla_g1250_qh16_wss() -> Function {
    mla_ws_with("mla_g1250_qh16_wss", false, 64, false)
}

/// `_wss` with the instruction prefetch at entry.
pub fn mla_g1250_qh16_wssp() -> Function {
    mla_ws_with("mla_g1250_qh16_wssp", false, 64 | 128, false)
}

/// `_wssp` with full pairs in the loop and the odd last page after it.
pub fn mla_g1250_qh16_wsst() -> Function {
    mla_ws_with("mla_g1250_qh16_wsst", false, 64 | 128 | 256, false)
}

/// `_wsst` with debug timestamps over the LSE (host PRINT_LSE=1).
pub fn mla_g1250_qh16_wsstd() -> Function {
    mla_ws_with("mla_g1250_qh16_wsstd", false, 64 | 128 | 256 | 512, false)
}

/// timing experiments (not for use): `_ws` exiting after the prologue / after the first page's tensor load
pub fn mla_g1250_qh16_wsx0() -> Function {
    mla_ws_with("mla_g1250_qh16_wsx0", false, 64 | 256 | 1024, false)
}

pub fn mla_g1250_qh16_wsx1() -> Function {
    mla_ws_with("mla_g1250_qh16_wsx1", false, 64 | 256 | 2048, false)
}

/// `_ws` with every wave taking every page and owning 128 of the V dims (S computed by all four): a 4x shorter PV chain per page, for
/// sequences with few pages.
pub fn mla_g1250_qh16_ws4() -> Function {
    mla_ws_with("mla_g1250_qh16_ws4", false, 64 | 256 | 4096 | 32768 | 65536, false)
}

/// `_ws` with wave pairs sharing a page (256 dims each), scalar loads and the odd tail page.
pub fn mla_g1250_qh16_ws2() -> Function {
    mla_ws_with("mla_g1250_qh16_ws2", true, 64 | 256 | 16384 | 65536, false)
}

/// Repro for the back end (not in the registry): `_wsstd` without the `wait_vm(0)` before the loop: the wait self-check rejects the output.
pub fn mla_g1250_qh16_wsstd_nowait() -> Function {
    mla_ws_with("mla_g1250_qh16_wsstd_nowait", false, 64 | 128 | 256 | 512 | 8192, false)
}

/// `_ws` compiled with the GEMM compile policy.
pub fn mla_g1250_qh16_wsg() -> Function {
    mla_ws_with("mla_g1250_qh16_wsg", false, 64 | 256 | 16384, false)
}

/// ablation aids (not registered): `_ws4` without the GEMM policy; and also with the cross-wave merge instead of the direct epilogue
pub fn mla_g1250_qh16_ws4b() -> Function {
    mla_ws_with("mla_g1250_qh16_ws4b", false, 64 | 256 | 4096, false)
}

pub fn mla_g1250_qh16_ws4c() -> Function {
    mla_ws_with("mla_g1250_qh16_ws4c", false, 64 | 256 | 4096 | 32768, false)
}

/// `_ws` with the lean softmax.
pub fn mla_g1250_qh16_wsl() -> Function {
    mla_ws_with("mla_g1250_qh16_wsl", false, 64 | 256 | 16384 | 65536, false)
}

/// `_ws` with the sequence's last page computed apart (the only masked page).
pub fn mla_g1250_qh16_wsm() -> Function {
    mla_ws_with("mla_g1250_qh16_wsm", false, 64 | 256 | 16384 | 65536 | 131072, false)
}

pub fn mla_g1250_qh16_wsbm() -> Function {
    mla_ws_with("mla_g1250_qh16_wsbm", false, 1, false)
}

pub fn mla_g1250_qh16_wsbr1() -> Function {
    mla_ws_with("mla_g1250_qh16_wsbr1", false, 2 | 4, false)
}

pub fn mla_g1250_qh16_wsbrg() -> Function {
    mla_ws_with("mla_g1250_qh16_wsbrg", false, 2 | 8, false)
}

pub fn mla_g1250_qh16_wsbr() -> Function {
    mla_ws_with("mla_g1250_qh16_wsbr", false, 2, false)
}

/// The wave-split MLA with the V dims split as well: wave pair q = w / 2 takes pages q, q + 2, ..; the pair's two waves compute the same
/// S / softmax and own 256 V dims each (16 accumulators: O fits in 128 VGPRs).
pub fn mla_g1250_qh16_wsd() -> Function {
    mla_ws_with("mla_g1250_qh16_wsd", true, 0, false)
}

fn mla_ws_with(name: &str, dsplit: bool, branchless: u32, pipe: bool) -> Function {
    // bit 4 of `branchless`: the intra-page form (`_wsi`); bit 5: the KV split across work-groups (`_wsk`: partials to a workspace, `_cmb` combines)
    let intra = branchless & 16 != 0;
    let split = branchless & 32 != 0;
    // bit 6: scalar loads (s_load) for the sequence bounds, the scales and the page ids; no LDS page table and no prologue barrier
    let sload = branchless & 64 != 0;
    // bit 7: prefetch the kernel's code into the instruction cache at entry (s_prefetch_inst, 12 x 4 KB)
    let ipf = branchless & 128 != 0;
    // bit 8: full page pairs in the loop, an odd last page after it (no masked re-read of a pad page)
    let tail = branchless & 256 != 0;
    // bit 16: lean softmax (only with `tail`: no fully masked page exists, so a masked logit's exp2 underflows to 0 and needs no select):
    // the mask as a branch-free select on the logits, no I1 loop variables
    let lean = branch_bits_lean(branchless);
    // bit 17 (with `lean`): the sequence's last page (the only one that can be partial) is computed apart, with the mask; every other page
    // without any mask code
    let lastsep = branchless & 131072 != 0;
    assert!(!lastsep || (lean && tail), "lastsep needs the lean tail form");
    let masked_now = std::cell::Cell::new(true);
    // bit 9: debug timestamps (SHADER_CYCLES, 20 bits) of wave 0 at checkpoints, written over LSE[b][0..6] (the LSE is then garbage)
    let tstamp = branchless & 512 != 0;
    // bits 10 / 11 (timing experiments): exit right after the prologue / right after the first page's tensor load completes
    let (exit_pro, exit_tdm) = (branchless & 1024 != 0, branchless & 2048 != 0);
    // bit 0: the mask branch-free; bit 1: the rescale branch-free
    let (bl_mask, bl_rescale) = (branchless & 1 != 0, branchless & 2 != 0);
    // bisect aids for the branch-free rescale: bit 2 = scale by the constant 1.0; bit 3 = scale by alpha only where some lane grew (select)
    let (dbg_one, dbg_gate) = (branchless & 4 != 0, branchless & 8 != 0);
    // dim groups: the waves that share a page and split its V dims (1: none; 2: `_wsd`; 4 (bit 12): every wave takes every page, 128 dims each)
    let dg: i32 = if branchless & 4096 != 0 { 4 } else if dsplit { 2 } else { 1 };
    let lg_dg: i32 = dg.trailing_zeros() as i32;
    let ntw: usize = NT / dg as usize;
    let ngrp: i32 = 4 / dg;
    let layout = PageLayout::plan();
    let slot = layout.slot_bytes as i64;
    let mut b = FnBuilder::new(name, 128);
    if branchless & 16384 != 0 {
        // bit 14: the GEMM compile policy (its lowering passes and the scheduler preset)
        b.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
    }
    let names = ["R", "LSE", "Q", "KV", "LTP", "LTD", "LTL", "QTP", "QS", "KS", "VSC"];
    for (i, n) in names.iter().enumerate() {
        b.arg(n, DType::Ptr, 8 * i as u32, 8);
    }
    for (i, n) in ["out16", "scalar", "qlens", "passes", "stride_page", "log2_page", "stride_q", "flags"].iter().enumerate() {
        b.arg(n, DType::I32, 88 + 4 * i as u32, 4);
    }
    b.arg_tensor("R", Ext::grid_y() * Ext::c(16 * 512 * 2));
    b.arg_tensor("LSE", Ext::grid_y() * Ext::c(16 * 4));
    b.arg_tensor("Q", Ext::grid_y() * Ext::c(16 * 768));
    b.arg_tensor("LTP", (Ext::grid_y() + Ext::c(1)) * Ext::c(4));
    b.arg_tensor("LTL", Ext::grid_y() * Ext::c(4));
    b.arg_tensor("QS", Ext::c(4));
    b.arg_tensor("KS", Ext::c(4));
    if split {
        // KV split: grid (S, B), S = passes (kernarg 100) in {1, 2, 4, 8}, flags (116) = log2 S; the workspace (VSC slot) holds per (b, s) the
        // un-normalised O (16 x 512 f32), then the max and the sum of the 16 heads
        b.ka_one_of(100, &[1, 2, 4, 8]);
        b.ka_pow2(116, Ext::ka(100));
        b.grid_is(0, Ext::ka(100));
        b.arg_tensor("VSC", Ext::grid_y() * Ext::ka(100) * Ext::c(WS_BYTES));
    }
    b.sym_range("P", 1, 1 << 24);
    b.sym_range("NPG", 1, 1 << 26);
    b.arg_tensor("KV", Ext::sym("P") * Ext::c(PAGE_BYTES as i64));
    b.arg_tensor("LTD", Ext::sym("NPG") * Ext::c(4));
    // LDS: two private page slots per wave, the alpha transposition hop per wave, the page table; the merge buffers over the dead rings
    let mut lds = LdsPlan::new();
    let kw = [lds.private("KVW0", slot, 4, &[]), lds.private("KVW1", slot, 4, &[])];
    let atr = lds.private("ATR", 64, 4, &[]);
    let ptab = lds.tile("PTAB", 4 * MAX_PAGES as i64, Layout::Linear);
    let mo = lds.tile_over("MO", 4 * 16 * 512 * 4, &[&kw[0]]);
    let ml = lds.tile_over("ML", 4 * 16 * 4 * 3, &[&kw[1]]);
    lds.commit(&mut b);
    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 8 * i });
    let (pr, plse, pq, pkv, pltp, pltd, pltl) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4), kp(&mut b, 5), kp(&mut b, 6));
    let (pqs, pks) = (kp(&mut b, 8), kp(&mut b, 9));
    let stamps: std::cell::RefCell<Vec<ValueId>> = std::cell::RefCell::new(Vec::new());
    let cyc = |b: &mut FnBuilder| {
        let c = b.shader_cycles();
        b.let_(Op::PtrLo(Opnd::Val(c)))
    };
    // instruction prefetch of the code that follows (n x 4 KB from here)
    let pf_here = |b: &mut FnBuilder, n: u32| {
        if ipf {
            let pc = b.get_pc();
            let nl = b.let_(Op::MovS { ty: DType::I32, a: Opnd::I32(31) });
            for k in 0..n {
                b.prefetch_inst(pc, 0x1000 * k, nl);
            }
        }
    };
    // a checkpoint of the debug variant: the stamp is stored at once over R[b] (word 64 w + checkpoint index; all lanes write the same word;
    // the output is garbage in this variant, and the R stores of the epilogue are skipped)
    let stamp_n = std::cell::Cell::new(0u32);
    let stamp = |b: &mut FnBuilder| {
        if tstamp {
            let c = cyc(b);
            let i = stamp_n.get();
            stamp_n.set(i + 1);
            let wb = b.let_(Op::WorkgroupIdY);
            let tidx = b.let_(Op::WorkitemId);
            let o = b.bind_i32(ev(wb) * (16 * 512 * 2) + ((ev(tidx) >> 5) * 64 + i as i32) * 4);
            let pr0 = b.let_(Op::KernargPtr { offset: 0 });
            let pa = b.let_(Op::PtrAdd { base: Opnd::Val(pr0), byte_off: Opnd::Val(o) });
            b.store_global(pa, 0, c, MemTy::B32, false);
            stamps.borrow_mut().push(c);
        }
    };
    if ipf {
        // MODE.SCALAR_PREFETCH_EN first: the prefetches are NOPs without it (the vendor kernels set it the same way)
        b.enable_scalar_prefetch();
        // the first 16 KB (prologue, the page loop); later windows are fetched where their code starts. A prefetch past the end of the code
        // can fault (an unmapped page): only windows that are known to lie inside the code are requested
        let pc = b.get_pc();
        let nl = b.let_(Op::MovS { ty: DType::I32, a: Opnd::I32(31) });
        for k in 0..4u32 {
            b.prefetch_inst(pc, 0x1000 * k, nl);
        }
    }
    stamp(&mut b);
    let tid = b.let_(Op::WorkitemId);
    let bidx = b.let_(Op::WorkgroupIdY);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let wv = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let w = b.read_first_lane(wv);
    let m = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(15)));
    let g = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::I32(4)));
    // the page group of the wave (its pages: grp, grp + ngrp, ..) and, with the dim split, its half of the V dims
    let grp = if dg > 1 { b.bind_i32(ev(w) >> lg_dg) } else { w };
    // the page sequence of the wave: pages gbase + gstride k (split: the 4 S waves of the S work-groups interleave)
    let (sid, lgs) = if split { (b.let_(Op::WorkgroupIdX), b.let_(Op::KernargI32 { offset: 116 })) } else { (b.c_i32(0), b.c_i32(0)) };
    let gbase = if split { b.bind_i32(ev(sid) * 4 + ev(w)) } else { grp };
    let gshift = if split { b.bind_i32(ev(lgs) + 2) } else { b.c_i32(2 - lg_dg) };
    let gstride = if split {
        let sc_ = b.let_(Op::KernargI32 { offset: 100 });
        b.bind_i32(ev(sc_) * 4)
    } else {
        b.c_i32(ngrp)
    };
    let hd = b.bind_i32(ev(w) & (dg - 1));
    let db = if dg > 1 { b.bind_i32(ev(hd) * (512 / dg)) } else { b.c_i32(0) };
    let ld = |b: &mut FnBuilder, p: ValueId, off: ValueId| {
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
        if sload {
            b.let_(Op::LoadScalar { base: Opnd::Val(q), byte_off: 0 })
        } else {
            let v = b.load_global(q, 0, MemTy::B32);
            b.read_first_lane(v)
        }
    };
    let bo = b.let_(Op::ShlI32(Opnd::Val(bidx), Opnd::I32(2)));
    let kv0 = ld(&mut b, pltp, bo);
    let bo4 = b.add_i32(bo, Opnd::I32(4));
    let kv1 = ld(&mut b, pltp, bo4);
    let last = ld(&mut b, pltl, bo);
    let npages = b.bind_i32(ev(kv1) - ev(kv0));
    b.assume_trusted(npages, Ext::c(1), Ext::c(MAX_PAGES as i64), "host contract: every sequence has 1 .. 1024 pages (ctx <= 65536)");
    b.assume_trusted(kv0, Ext::c(0), Ext::sym("NPG") - Ext::c(1), "host contract: LTP is monotone and LTP[B] <= NPG, the page table's entry count");
    let kv_end = b.bind_i32(ev(kv0) + ev(npages));
    b.assume_trusted(kv_end, Ext::c(1), Ext::sym("NPG"), "host contract: LTP[b + 1] <= NPG (a sequence's entries lie inside the page table)");
    b.assume_trusted(last, Ext::c(1), Ext::c(PAGE_TOKENS as i64), "host contract: the last page holds 1 .. 64 tokens");
    let ctx = b.bind_i32(((ev(npages) - 1) << 6) + ev(last));
    let zero = b.c_i32(0);
    let qs = ld(&mut b, pqs, zero);
    let ks = ld(&mut b, pks, zero);
    let qsf = b.let_(Op::BitcastF32(Opnd::Val(qs)));
    let ksf = b.let_(Op::BitcastF32(Opnd::Val(ks)));
    let sc = b.let_(Op::KernargI32 { offset: 92 });
    let scf = b.let_(Op::BitcastF32(Opnd::Val(sc)));
    let c1 = b.mul_f32(qsf, ksf);
    let c2 = b.mul_f32(c1, scf);
    let log2e = b.c_f32(std::f32::consts::LOG2_E);
    let cscale = b.mul_f32(c2, log2e);
    // Q fragments (B of S^T): head m, dims 64 c + 32 g .. + 32 (the K order inside a WMMA is free: the K reads use the same one)
    let qrow = b.bind_i32((ev(bidx) * 16 + ev(m)) * 768 + ev(g) * 32);
    let pqr = b.let_(Op::PtrAdd { base: Opnd::Val(pq), byte_off: Opnd::Val(qrow) });
    let q: Vec<ValueId> = (0..9u32)
        .map(|c| {
            let parts: Vec<Opnd> = (0..2u32).map(|j| Opnd::Val(b.load_global(pqr, 64 * c + 16 * j, MemTy::B128))).collect();
            b.frag_cat(&parts)
        })
        .collect();
    // the timing experiments keep the Q loads alive (one word to R[b]: the output is garbage in these variants)
    let keep_q = |b: &mut FnBuilder| {
        let o = b.bind_i32(ev(bidx) * (16 * 512 * 2));
        let pa = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(o) });
        for c in 0..9usize {
            for e8 in 0..8u8 {
                let e = b.frag_slice(Opnd::Val(q[c]), e8, 1);
                b.store_global(pa, 4 * (8 * c as u32 + e8 as u32), e, MemTy::B32, false);
            }
        }
    };
    if exit_pro {
        keep_q(&mut b);
        return b.finish();
    }
    if tstamp && branchless & 8192 == 0 {
        // the debug stamps store to global memory inside the loop: settle the Q loads first (bit 13 drops this: the wait-pass repro)
        b.wait_vm(0);
    }
    // one page = one tensor op (4608 x 8 bytes), padded by the plan
    let tile = TensorTile { data_size: 8, tensor_dim: [Opnd::I32(PAGE_BYTES / 8), Opnd::I32(1)], stride0: Opnd::I32(PAGE_BYTES / 8), tile: [(PAGE_BYTES / 8) as u16, 1], pad: layout.pad, wg_mask: 0 };
    let (_, g1) = b.tensor_desc_2d(&tile, Opnd::Val(pkv), Opnd::I32(0), Opnd::I32(0));
    // lane offsets in a page slot (tile j = tokens 16 j ..): K nope / rope rows, the V tr8 fetch address; the constant parts as checked deltas
    let kofs = layout.at_expr(&mut b, ev(m) * DV + ev(g) * 32);
    let rofs = layout.at_expr(&mut b, ev(m) * 64 + ev(g) * 32 + ROPE_OFF);
    let vofs = layout.at_expr(&mut b, (((ev(lane) >> 4) << 3) + (ev(lane) & 3) + (((ev(lane) >> 3) & 1) << 2)) * DV + (((ev(lane) >> 2) & 1) << 3) + ev(db));
    let kb: Vec<i32> = (0..32).map(|l| (l & 15) * DV + (l >> 4) * 32).collect();
    let rb: Vec<i32> = (0..32).map(|l| (l & 15) * 64 + (l >> 4) * 32 + ROPE_OFF).collect();
    let vb: Vec<i32> = (0..32).map(|f| (8 * (f >> 4) + (f & 3) + 4 * ((f >> 3) & 1)) * DV + 8 * ((f >> 2) & 1)).collect();
    // kd[j][c]: chunk c (< 8 nope, 8 rope) of S tile j; vd[jj][nt]: token chunk jj, dim tile nt
    let kd: Vec<Vec<i32>> = (0..4).map(|j| (0..9).map(|c| if c < 8 { layout.delta(&kb, 16 * j * DV + 64 * c) } else { layout.delta(&rb, 16 * j * 64) }).collect()).collect();
    let vd: Vec<Vec<i32>> = (0..4).map(|jj| (0..NT as i32).map(|nt| layout.delta(&vb, 16 * jj * DV + 16 * nt)).collect()).collect();
    // the page table into LDS (shared, read-only after the barrier); with scalar loads the page ids are read straight from the table
    if !sload {
    let nch = b.bind_i32((ev(npages) + 127) >> 7);
    let k0 = b.c_i32(0);
    let lv = b.begin_loop(&[(DType::I32, Opnd::Val(k0))]);
    let kk = lv[0];
    b.assume(kk, Ext::c(0), Ext::c((MAX_PAGES / 128 - 1) as i64));
    let idx = b.bind_i32(amdgpu_dsl::varx(kk) * 128 + ev(tid));
    let e_raw = b.bind_i32(ev(kv0) + ev(idx));
    let e_last = b.bind_i32(ev(kv_end) - 1);
    let ci = b.let_(Op::MinI32(Opnd::Val(e_raw), Opnd::Val(e_last)));
    let eo = b.bind_i32(ev(ci) * 4);
    let pe = b.let_(Op::PtrAdd { base: Opnd::Val(pltd), byte_off: Opnd::Val(eo) });
    let pv = b.load_global(pe, 0, MemTy::B32);
    let sa = b.bind_i32(ev(idx) * 4 + ptab.base as i32);
    ptab.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).write(&mut b, |b| b.store_lds(sa, pv, MemTy::B32, false));
    let k1 = b.bind_i32(amdgpu_dsl::varx(kk) + 1);
    b.set_var(kk, Opnd::Val(k1));
    let done = b.cmp(CmpPred::Ge, CmpTy::I32, k1, nch);
    b.break_if(done);
    b.end_loop();
    b.wait_lds(0);
    b.barrier();
    }

    stamp(&mut b);
    // ---- the wave's pages: w, w + 4, ..: nw of them (0 for a short sequence) ----
    let neg = b.c_f32(-1.0e30);
    let zf = b.c_f32(0.0);
    let mut inits = vec![(DType::F32, Opnd::Val(neg)), (DType::F32, Opnd::Val(zf))];
    for _ in 0..ntw {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    let st = b.declare_vars(&inits);
    let (vm, vl) = (st[0], st[1]);
    let ov: Vec<VarId> = st[2..].to_vec();
    let ptab_rd = |b: &mut FnBuilder, local: amdgpu_dsl::I32| -> ValueId {
        // past the sequence (the waves run the same trip count): the last page again, all its tokens masked
        let pg = b.bind_i32(local * ev(gstride) + ev(gbase));
        let lastp = b.bind_i32(ev(npages) - 1);
        let pc = b.let_(Op::MinI32(Opnd::Val(pg), Opnd::Val(lastp)));
        let pid = if sload {
            // the entry kv0 + page, clamped to the sequence's last entry (kv_end - 1 < NPG)
            let er = b.bind_i32(ev(kv0) + ev(pg));
            let el = b.bind_i32(ev(kv_end) - 1);
            let ec = b.let_(Op::MinI32(Opnd::Val(er), Opnd::Val(el)));
            let eo = b.bind_i32(ev(ec) * 4);
            let pe = b.let_(Op::PtrAdd { base: Opnd::Val(pltd), byte_off: Opnd::Val(eo) });
            b.let_(Op::LoadScalar { base: Opnd::Val(pe), byte_off: 0 })
        } else {
            let a = b.bind_i32((ev(pc) << 2) + ptab.base as i32);
            let v = ptab.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).read(b, |b| b.load_lds(a, MemTy::B32));
            b.read_first_lane(v)
        };
        b.assume_trusted(pid, Ext::c(0), Ext::sym("P") - Ext::c(1), "host contract: every page-table entry < P, the KV pool's page count (KV holds P x 36864 bytes)");
        pid
    };
    // stage local page `local` (generation) into private slot s
    let stage = |b: &mut FnBuilder, s: usize, local: amdgpu_dsl::I32| {
        let pid = ptab_rd(b, local.clone());
        let off = b.bind_i32(ev(pid) * PAGE_BYTES);
        let mine = kw[s].mine(w);
        let lds_a = b.bind_i32(mine.base());
        let g0 = tdm_group0(b, pkv, off, Opnd::Val(lds_a));
        let gen = b.bind_i32(local);
        mine.write(b, gen, |b| b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default()));
    };
    // the pieces of one page (private slot s, generation `local`): S; the softmax (logits, mask, max, P); the O rescale; PV
    let page_of = |b: &mut FnBuilder, local: amdgpu_dsl::I32| b.bind_i32(local * ev(gstride) + ev(gbase));
    let s_part = |b: &mut FnBuilder, s: usize, gen: ValueId| -> Vec<ValueId> {
        let mine = kw[s].mine(w);
        let sb = b.bind_i32(mine.base());
        let nrow = b.bind_i32(ev(sb) + ev(kofs));
        let rrow = b.bind_i32(ev(sb) + ev(rofs));
        // the 4 S tiles interleaved chunk by chunk: the 4 K fragments of chunk c are read before the 4 WMMAs of chunk c - 1 complete their
        // chains (4 independent accumulators: WMMA latency and LDS latency overlap)
        let svals: Vec<ValueId> = mine.read_back(b, gen, |b| {
            let frag = |b: &mut FnBuilder, j: usize, c: usize| {
                let lo = if c < 8 { b.bind_i32(ev(nrow) + kd[j][c]) } else { b.bind_i32(ev(rrow) + kd[j][c]) };
                let hi = b.bind_i32(ev(lo) + 16);
                b.load_lds_b128x2(lo, hi)
            };
            let mut acc: Vec<Option<Opnd>> = vec![None; 4];
            let mut cur: Vec<ValueId> = (0..4).map(|j| frag(b, j, 0)).collect();
            for c in 0..9usize {
                let next: Vec<ValueId> = if c + 1 < 9 { (0..4).map(|j| frag(b, j, c + 1)).collect() } else { Vec::new() };
                for j in 0..4 {
                    let d = b.wmma(WmmaKind::Fp8_16x16x64, cur[j], q[c], acc[j]);
                    acc[j] = Some(Opnd::Val(d));
                }
                cur = next;
            }
            acc.into_iter()
                .map(|a| match a {
                    Some(Opnd::Val(v)) => v,
                    _ => unreachable!(),
                })
                .collect()
        });
        svals
    };
    let soft = |b: &mut FnBuilder, svals: &[ValueId], page: ValueId| -> (ValueId, ValueId, ValueId, ValueId) {
        // valid tokens of this page (>= 64 except on the sequence's last page)
        let tok_lim = b.bind_i32(ev(ctx) - (ev(page) << 6));
        if lean {
            let mut x: Vec<ValueId> = Vec::with_capacity(32);
            for j in 0..4 {
                for r in 0..8u8 {
                    let si = b.frag_slice(Opnd::Val(svals[j]), r, 1);
                    let sf = f32_of(b, si);
                    let v = b.mul_f32(sf, cscale);
                    if masked_now.get() {
                        let tk = b.bind_i32(ev(g) * 8 + (16 * j as i32 + r as i32));
                        let ok = b.cmp(CmpPred::Lt, CmpTy::I32, tk, tok_lim);
                        x.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(neg) }));
                    } else {
                        x.push(v);
                    }
                }
            }
            let mut mx = b.max_f32(x[0], x[1]);
            for r in 2..32 {
                mx = b.max_f32(mx, x[r]);
            }
            let sw = b.let_(Op::Permlane16Swap { a: Opnd::Val(mx), b: Opnd::Val(mx) });
            let s0 = b.frag_slice(Opnd::Val(sw), 0, 1);
            let s1 = b.frag_slice(Opnd::Val(sw), 1, 1);
            let s0f = f32_of(b, s0);
            let s1f = f32_of(b, s1);
            let pm = b.max_f32(s0f, s1f);
            let m_new = b.max_f32(Opnd::Var(vm), pm);
            let dm = b.sub_f32(Opnd::Var(vm), m_new);
            let alpha = b.exp2_f32(dm);
            let mut lsum = b.mul_f32(Opnd::Var(vl), alpha);
            // all the exponents first (distinct registers: no exp source is overwritten right behind its TRANS op), then the exps, then the sums
            let d: Vec<ValueId> = (0..32).map(|r| b.sub_f32(x[r], m_new)).collect();
            let p: Vec<ValueId> = (0..32).map(|r| b.exp2_f32(d[r])).collect();
            for r in 0..32 {
                lsum = b.add_f32(lsum, p[r]);
            }
            let parts: Vec<Opnd> = (0..8).map(|k| Opnd::Val(b.cvt_pk_fp8x4_f32(p[4 * k], p[4 * k + 1], p[4 * k + 2], p[4 * k + 3]))).collect();
            let pa = b.frag_cat(&parts);
            return (pa, alpha, m_new, lsum);
        }
        // logits; the mask (tokens at or beyond ctx) only on a page that reaches past ctx: a uniform branch around VALU only
        let mut x: Vec<ValueId> = Vec::with_capacity(32);
        for j in 0..4 {
            for r in 0..8u8 {
                let si = b.frag_slice(Opnd::Val(svals[j]), r, 1);
                let sf = f32_of(b, si);
                x.push(b.mul_f32(sf, cscale));
            }
        }
        let xv = b.declare_vars(&x.iter().map(|&v| (DType::F32, Opnd::Val(v))).collect::<Vec<_>>());
        let one = b.c_i32(1);
        let okv0: Vec<ValueId> = (0..32).map(|_| b.cmp(CmpPred::Eq, CmpTy::I32, one, Opnd::I32(1))).collect();
        let okvars = b.declare_vars(&okv0.iter().map(|&v| (DType::I1, Opnd::Val(v))).collect::<Vec<_>>());
        // the mask: a uniform branch around it (only a page reaching past ctx), or branch-free (select on every page)
        let partial = b.cmp(CmpPred::Lt, CmpTy::I32, tok_lim, Opnd::I32(64));
        if !bl_mask {
            b.begin_if(partial);
        }
        for j in 0..4 {
            for r in 0..8u8 {
                let k = 8 * j + r as usize;
                let tk = b.bind_i32(ev(g) * 8 + (16 * j as i32 + r as i32));
                let ok = b.cmp(CmpPred::Lt, CmpTy::I32, tk, tok_lim);
                let xm = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Var(xv[k]), b: Opnd::Val(neg) });
                b.set_var(xv[k], Opnd::Val(xm));
                b.set_var(okvars[k], Opnd::Val(ok));
            }
        }
        if !bl_mask {
            b.end_if();
        }
        let x: Vec<Opnd> = xv.iter().map(|&v| Opnd::Var(v)).collect();
        let mut mx = b.max_f32(x[0], x[1]);
        for r in 2..32 {
            mx = b.max_f32(mx, x[r]);
        }
        // the other lane half (same head m): v_permlane16_swap of (mx, mx)
        let sw = b.let_(Op::Permlane16Swap { a: Opnd::Val(mx), b: Opnd::Val(mx) });
        let s0 = b.frag_slice(Opnd::Val(sw), 0, 1);
        let s1 = b.frag_slice(Opnd::Val(sw), 1, 1);
        let s0f = f32_of(b, s0);
        let s1f = f32_of(b, s1);
        let pm = b.max_f32(s0f, s1f);
        let m_new = b.max_f32(Opnd::Var(vm), pm);
        let dm = b.sub_f32(Opnd::Var(vm), m_new);
        let alpha = b.exp2_f32(dm);
        let mut lsum = b.mul_f32(Opnd::Var(vl), alpha);
        let mut p = Vec::with_capacity(32);
        for r in 0..32 {
            let d = b.sub_f32(x[r], m_new);
            let e0 = b.exp2_f32(d);
            // a masked token contributes 0 (also when the whole page is masked: then x = m_new = -1e30)
            let e = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Var(okvars[r]), a: Opnd::Val(e0), b: Opnd::Val(zf) });
            lsum = b.add_f32(lsum, e);
            p.push(e);
        }
        // the A fragment of P: tile j's 8 values at bytes 8 j .. (tokens 16 j + 8 g + r)
        let parts: Vec<Opnd> = (0..8).map(|k| Opnd::Val(b.cvt_pk_fp8x4_f32(p[4 * k], p[4 * k + 1], p[4 * k + 2], p[4 * k + 3]))).collect();
        let pa = b.frag_cat(&parts);
        (pa, alpha, m_new, lsum)
    };
    let rescale = |b: &mut FnBuilder, alpha: ValueId| {
        let grew = b.cmp(CmpPred::Lt, CmpTy::F32, alpha, Opnd::F32(1.0));
        let any = b.let_(Op::AnyLane(Opnd::Val(grew)));
        // lazy (only when some head's max grew: a uniform branch) or branch-free (every page)
        if !bl_rescale {
            b.begin_if(any);
        }
        // the alpha of row 8 g + r: lane 8 g + r holds head 8 g + r (v_readlane, then the lane half's choice)
        let alpha = if dbg_one {
            b.c_f32(1.0)
        } else if dbg_gate {
            let anyf = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(any), a: Opnd::I32(1), b: Opnd::I32(0) });
            let ab = b.cmp(CmpPred::Ne, CmpTy::I32, anyf, Opnd::I32(0));
            let one = b.c_f32(1.0);
            b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ab), a: Opnd::Val(alpha), b: Opnd::Val(one) })
        } else {
            alpha
        };
        let sa: Vec<ValueId> = (0..16u8).map(|h| b.read_lane(alpha, h)).collect();
        let g0c = b.cmp(CmpPred::Eq, CmpTy::I32, g, Opnd::I32(0));
        let al: Vec<ValueId> = (0..8usize)
            .map(|r| {
                let a0 = b.let_(Op::BitcastF32(Opnd::Val(sa[r])));
                let a1 = b.let_(Op::BitcastF32(Opnd::Val(sa[8 + r])));
                b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(g0c), a: Opnd::Val(a0), b: Opnd::Val(a1) })
            })
            .collect();
        for nt in 0..ntw {
            for r in 0..8u8 {
                b.var_scale_lane(ov[nt], r, al[r as usize]);
            }
        }
        if !bl_rescale {
            b.end_if();
        }
    };
    let pv = |b: &mut FnBuilder, s: usize, gen: ValueId, pa: ValueId| {
        let mine = kw[s].mine(w);
        let sb = b.bind_i32(mine.base());
        let vrow = b.bind_i32(ev(sb) + ev(vofs));
        // PV: 32 dim tiles, V fragments in groups of 4
        mine.read_back(b, gen, |b| {
            // V fragments one group of 4 tiles ahead of the WMMAs that use them
            let vgrp = |b: &mut FnBuilder, grp: usize| -> Vec<ValueId> {
                (0..4)
                    .map(|k| {
                        let nt = 4 * grp + k;
                        let a: Vec<ValueId> = (0..4).map(|jj| b.bind_i32(ev(vrow) + vd[jj][nt])).collect();
                        b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3])
                    })
                    .collect()
            };
            let mut cur = vgrp(b, 0);
            for grp in 0..ntw / 4 {
                let next = if grp + 1 < ntw / 4 { vgrp(b, grp + 1) } else { Vec::new() };
                for k in 0..4 {
                    let o = ov[4 * grp + k];
                    let d = b.wmma(WmmaKind::Fp8_16x16x64, pa, cur[k], Some(Opnd::Var(o)));
                    b.set_var(o, Opnd::Val(d));
                }
                cur = next;
            }
        });
    };
    // intra-page form: the alpha of row 8 g + r without an I1 select (al = a0 (1 - g) + a1 g: exact for g in {0, 1}); the rescale of a
    // group of 4 accumulators and that group's PV WMMAs as units of one Free plan phase (PV the skeleton, the next group's rescale its filler)
    let rows_of = |b: &mut FnBuilder, alpha: ValueId| -> Vec<ValueId> {
        let sa: Vec<ValueId> = (0..16u8).map(|h| b.read_lane(alpha, h)).collect();
        // g in {0, 1} as f32: the bits of 1.0 times g
        let gbits = b.bind_i32(ev(g) * 0x3F80_0000);
        let gi = b.let_(Op::BitcastF32(Opnd::Val(gbits)));
        let one = b.c_f32(1.0);
        let hf = b.sub_f32(one, gi);
        (0..8usize)
            .map(|r| {
                let a0 = b.let_(Op::BitcastF32(Opnd::Val(sa[r])));
                let a1 = b.let_(Op::BitcastF32(Opnd::Val(sa[8 + r])));
                let t0 = b.mul_f32(a0, hf);
                let t1 = b.mul_f32(a1, gi);
                b.add_f32(t0, t1)
            })
            .collect()
    };
    let pv_intra = |b: &mut FnBuilder, s: usize, gen: ValueId, pa: ValueId, al: &[ValueId], h: usize| {
        let mine = kw[s].mine(w);
        let sb = b.bind_i32(mine.base());
        let vrow = b.bind_i32(ev(sb) + ev(vofs));
        let mut u = Units::new();
        let mut pvs = Vec::new();
        let mut rss = Vec::new();
        let vgrp = |b: &mut FnBuilder, grp: usize| -> Vec<ValueId> {
            (0..4)
                .map(|k| {
                    let nt = 4 * grp + k;
                    let a: Vec<ValueId> = (0..4).map(|jj| b.bind_i32(ev(vrow) + vd[jj][nt])).collect();
                    b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3])
                })
                .collect()
        };
        // the V fragments of group 0 first; each PV unit loads the next group's before its WMMAs (one group ahead, as in `_ws`)
        let (_, mut cur) = u.capture(b, format!("v{h}.0"), |b| mine.read_back(b, gen, |b| vgrp(b, 0)));
        for grp in 0..ntw / 4 {
            let (rid, _) = u.capture(b, format!("rs{h}.{grp}"), |b| {
                for k in 0..4 {
                    for r in 0..8u8 {
                        b.var_scale_lane(ov[4 * grp + k], r, al[r as usize]);
                    }
                }
            });
            let fv = cur.clone();
            let (pid, next) = u.capture(b, format!("pv{h}.{grp}"), |b| {
                mine.read_back(b, gen, |b| {
                    let next = if grp + 1 < ntw / 4 { vgrp(b, grp + 1) } else { Vec::new() };
                    for k in 0..4 {
                        let o = ov[4 * grp + k];
                        let d = b.wmma(WmmaKind::Fp8_16x16x64, pa, fv[k], Some(Opnd::Var(o)));
                        b.set_var(o, Opnd::Val(d));
                    }
                    next
                })
            });
            cur = next;
            rss.push(rid);
            pvs.push(pid);
        }
        let mut wv = Weaver::new(u).phase(format!("mla.pv{h}")).free();
        for gi in 0..pvs.len() {
            if gi + 1 < rss.len() {
                wv = wv.interleave(pvs[gi], vec![rss[gi + 1]]);
            }
        }
        wv.emit(b);
    };
    // one page, in order: S, softmax, rescale, PV, the running max / sum; the slot's reads complete before the next stage overwrites it
    let compute = |b: &mut FnBuilder, s: usize, local: amdgpu_dsl::I32| {
        let gen = b.bind_i32(local.clone());
        let half = (s & 1) as usize;
        let page = page_of(b, local);
        stamp(b);
        let sv = s_part(b, s, gen);
        stamp(b);
        let (pa, alpha, m_new, lsum) = soft(b, &sv, page);
        stamp(b);
        if intra {
            let al = rows_of(b, alpha);
            pv_intra(b, s, gen, pa, &al, half);
        } else {
            rescale(b, alpha);
            stamp(b);
            pv(b, s, gen, pa);
        }
        stamp(b);
        b.set_var(vm, Opnd::Val(m_new));
        b.set_var(vl, Opnd::Val(lsum));
        b.wait_lds(0);
    };
    // the wave's own pages: nw = ceil((npages - w) / 4) (the page group's), in pairs (an odd last pair re-reads the last page, fully masked);
    // a wave without a page skips the loop. The trip count is the wave's own (wave-uniform): no barrier inside the loop needs the waves in step
    let nw = b.bind_i32((ev(npages) - ev(gbase) + ev(gstride) - 1) >> ev(gshift));
    let has = b.cmp(CmpPred::Gt, CmpTy::I32, nw, Opnd::I32(0));
    b.begin_if(has);
    stage(&mut b, 0, amdgpu_dsl::I32::c(0));
    if exit_tdm {
        b.wait_tensor(0);
        keep_q(&mut b);
    }
    if !pipe && tail && !exit_tdm && lastsep {
        // the wave's full pages: all but the sequence's last page when this wave owns it
        let lastp = b.bind_i32(ev(npages) - 1);
        let own = b.bind_i32((ev(lastp) - ev(gbase)) & (ev(gstride) - 1));
        let ownb = b.cmp(CmpPred::Eq, CmpTy::I32, own, Opnd::I32(0));
        let ownl = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ownb), a: Opnd::I32(1), b: Opnd::I32(0) });
        let nf = b.bind_i32(ev(nw) - ev(ownl));
        let npairs = b.bind_i32(ev(nf) >> 1);
        let z0 = b.c_i32(0);
        let gl = b.declare_vars(&[(DType::I32, Opnd::Val(z0))])[0];
        b.assume(gl, Ext::c(0), Ext::c(MAX_PAGES as i64));
        masked_now.set(false);
        let any_pair = b.cmp(CmpPred::Gt, CmpTy::I32, npairs, Opnd::I32(0));
        b.begin_if(any_pair);
        let z = b.c_i32(0);
        let pv2 = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let kp2 = pv2[0];
        b.assume(kp2, Ext::c(0), Ext::c((MAX_PAGES / 8) as i64));
        let l0 = b.bind_i32(amdgpu_dsl::varx(kp2) * 2);
        let l1 = b.bind_i32(ev(l0) + 1);
        let l2 = b.bind_i32(ev(l0) + 2);
        stage(&mut b, 1, ev(l1));
        b.wait_tensor(1);
        compute(&mut b, 0, ev(l0));
        stage(&mut b, 0, ev(l2));
        b.wait_tensor(1);
        compute(&mut b, 1, ev(l1));
        let g2 = b.bind_i32(amdgpu_dsl::varx(gl) + 2);
        b.set_var(gl, Opnd::Val(g2));
        let kn = b.bind_i32(amdgpu_dsl::varx(kp2) + 1);
        b.set_var(kp2, Opnd::Val(kn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, kn, npairs);
        b.break_if(fin);
        b.end_loop();
        b.end_if();
        // page 2 npairs (slot 0, already staged) is an odd full page, or this wave's masked last page
        let odd = b.bind_i32(ev(nf) & 1);
        let is_odd = b.cmp(CmpPred::Ne, CmpTy::I32, odd, Opnd::I32(0));
        b.begin_if(is_odd);
        b.wait_tensor(0);
        compute(&mut b, 0, amdgpu_dsl::varx(gl) + 0);
        // and then the last page (if this wave owns it) from slot 1
        masked_now.set(true);
        b.begin_if(ownb);
        let gl1 = b.bind_i32(amdgpu_dsl::varx(gl) + 1);
        stage(&mut b, 1, ev(gl1));
        b.wait_tensor(0);
        compute(&mut b, 1, ev(gl1));
        b.end_if();
        b.else_();
        b.begin_if(ownb);
        b.wait_tensor(0);
        compute(&mut b, 0, amdgpu_dsl::varx(gl) + 0);
        b.end_if();
        b.end_if();
        masked_now.set(true);
    } else if !pipe && tail && !exit_tdm {
        let npairs = b.bind_i32(ev(nw) >> 1);
        let z0 = b.c_i32(0);
        // the generation (local index) of the odd last page: a counter stepped by + 2 per pass
        let gl = b.declare_vars(&[(DType::I32, Opnd::Val(z0))])[0];
        let any_pair = b.cmp(CmpPred::Gt, CmpTy::I32, npairs, Opnd::I32(0));
        b.begin_if(any_pair);
        let z = b.c_i32(0);
        let pv2 = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let kp2 = pv2[0];
        b.assume(kp2, Ext::c(0), Ext::c((MAX_PAGES / 8) as i64));
        let l0 = b.bind_i32(amdgpu_dsl::varx(kp2) * 2);
        let l1 = b.bind_i32(ev(l0) + 1);
        let l2 = b.bind_i32(ev(l0) + 2);
        stage(&mut b, 1, ev(l1));
        b.wait_tensor(1);
        compute(&mut b, 0, ev(l0));
        stage(&mut b, 0, ev(l2));
        b.wait_tensor(1);
        compute(&mut b, 1, ev(l1));
        let g2 = b.bind_i32(amdgpu_dsl::varx(gl) + 2);
        b.set_var(gl, Opnd::Val(g2));
        let kn = b.bind_i32(amdgpu_dsl::varx(kp2) + 1);
        b.set_var(kp2, Opnd::Val(kn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, kn, npairs);
        b.break_if(fin);
        b.end_loop();
        b.end_if();
        // an odd count: the last page (staged into slot 0 by the prologue or the last pass)
        let odd = b.bind_i32(ev(nw) & 1);
        let is_odd = b.cmp(CmpPred::Ne, CmpTy::I32, odd, Opnd::I32(0));
        b.begin_if(is_odd);
        pf_here(&mut b, 2);
        b.wait_tensor(0);
        compute(&mut b, 0, amdgpu_dsl::varx(gl) + 0);
        if tstamp {
            // timing experiment: the same page again (warm instruction cache; the output is garbage in this variant anyway)
            compute(&mut b, 0, amdgpu_dsl::varx(gl) + 0);
        }
        b.end_if();
    } else if !pipe && !exit_tdm {
        let npairs = b.bind_i32((ev(nw) + 1) >> 1);
        let z = b.c_i32(0);
        let pv2 = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let kp2 = pv2[0];
        b.assume(kp2, Ext::c(0), Ext::c((MAX_PAGES / 8) as i64));
        let l0 = b.bind_i32(amdgpu_dsl::varx(kp2) * 2);
        let l1 = b.bind_i32(ev(l0) + 1);
        let l2 = b.bind_i32(ev(l0) + 2);
        stage(&mut b, 1, ev(l1));
        b.wait_tensor(1);
        compute(&mut b, 0, ev(l0));
        stage(&mut b, 0, ev(l2));
        b.wait_tensor(1);
        compute(&mut b, 1, ev(l1));
        let kn = b.bind_i32(amdgpu_dsl::varx(kp2) + 1);
        b.set_var(kp2, Opnd::Val(kn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, kn, npairs);
        b.break_if(fin);
        b.end_loop();
    } else if !exit_tdm {
        // cross-page form: step i = S(i); [softmax(i) || PV(i - 1)] (one Free plan phase, PV WMMAs the skeleton, the softmax the fillers);
        // then the stage of page i + 1 into page i - 1's slot, the rescale by alpha(i). P(i) is carried to the next step.
        b.wait_tensor(0);
        let g0v = b.c_i32(0);
        let sv0 = s_part(&mut b, 0, g0v);
        let pg0 = page_of(&mut b, amdgpu_dsl::I32::c(0));
        let (pa0, al0, m0, ls0) = soft(&mut b, &sv0, pg0);
        rescale(&mut b, al0);
        b.set_var(vm, Opnd::Val(m0));
        b.set_var(vl, Opnd::Val(ls0));
        stage(&mut b, 1, amdgpu_dsl::I32::c(1));
        let z0 = b.c_i32(0);
        let pv_ = b.declare_vars(&[(WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(pa0)), (DType::I32, Opnd::Val(z0))]);
        // the carried P, and the generation (local page index) of the last page: a counter stepped by + 2 per loop pass
        let (pvar, glast) = (pv_[0], pv_[1]);
        // steps i = 1 .. 2 npairs (page i in slot i % 2); npairs = floor(nw / 2): pages past nw - 1 are masked re-reads
        let npairs = b.bind_i32(ev(nw) >> 1);
        let more = b.cmp(CmpPred::Gt, CmpTy::I32, npairs, Opnd::I32(0));
        b.begin_if(more);
        let z = b.c_i32(0);
        let lv2 = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let kp2 = lv2[0];
        b.assume(kp2, Ext::c(0), Ext::c((MAX_PAGES / 8) as i64));
        for h in 0..2usize {
            // page i = 2 k + 1 + h in slot s = 1 - h; page i - 1 in slot 1 - s
            let s = 1 - h;
            let li = b.bind_i32(amdgpu_dsl::varx(kp2) * 2 + (1 + h as i32));
            let gi = li;
            let gp = b.bind_i32(ev(li) - 1);
            b.wait_tensor(0);
            let sv = s_part(&mut b, s, gi);
            let page = page_of(&mut b, ev(li));
            let pprev = b.let_(Op::FragCat { parts: vec![Opnd::Var(pvar)], dwords: 8 });
            let mut u = Units::new();
            let mut skel = Vec::new();
            let (fid, (pa, alpha, m_new, lsum)) = u.capture(&mut b, format!("softmax{h}"), |b| soft(b, &sv, page));
            let (pid, _) = u.capture(&mut b, format!("pv{h}"), |b| pv(b, 1 - s, gp, pprev));
            skel.push(pid);
            Weaver::new(u).phase(format!("mla.step{h}")).free().interleave(pid, vec![fid]).emit(&mut b);
            let _ = skel;
            b.wait_lds(0);
            let ln = b.bind_i32(ev(li) + 1);
            stage(&mut b, 1 - s, ev(ln));
            rescale(&mut b, alpha);
            b.set_var(vm, Opnd::Val(m_new));
            b.set_var(vl, Opnd::Val(lsum));
            b.set_var(pvar, Opnd::Val(pa));
        }
        let g2 = b.bind_i32(amdgpu_dsl::varx(glast) + 2);
        b.set_var(glast, Opnd::Val(g2));
        let kn = b.bind_i32(amdgpu_dsl::varx(kp2) + 1);
        b.set_var(kp2, Opnd::Val(kn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, kn, npairs);
        b.break_if(fin);
        b.end_loop();
        b.end_if();
        // the last page's PV: page 2 npairs, in slot (2 npairs) % 2 = 0
        let gl = b.bind_i32(amdgpu_dsl::varx(glast) + 0);
        let plast = b.let_(Op::FragCat { parts: vec![Opnd::Var(pvar)], dwords: 8 });
        pv(&mut b, 0, gl, plast);
        b.wait_lds(0);
    }
    b.end_if();
    b.wait_tensor(0);

    stamp(&mut b);
    pf_here(&mut b, 1);
    let t0 = SlotIdx::new(Opnd::I32(0), Opnd::I32(0));
    // (measured slower than the merge path below on e08 - 5.1 vs 4.3 us at 1 x 64 - and kept off: bit 15 set in `_ws4`)
    if dg == 4 && !split && !tstamp && branchless & 32768 == 0 {
        // every wave saw every page: its O (128 dims) is complete and its max / sums are the work-group's. No cross-wave merge: the
        // row sum of head m = the two lane halves' partials (v_permlane16_swap), 1 / L of row 8 g + r by v_readlane, then the scaled O through
        // the wave's quarter of the (dead) merge buffer for 16-byte stores
        let sw = b.let_(Op::Permlane16Swap { a: Opnd::Var(vl), b: Opnd::Var(vl) });
        let s0 = b.frag_slice(Opnd::Val(sw), 0, 1);
        let s1 = b.frag_slice(Opnd::Val(sw), 1, 1);
        let s0f = f32_of(&mut b, s0);
        let s1f = f32_of(&mut b, s1);
        let lh = b.add_f32(s0f, s1f);
        let rl = b.rcp_f32(lh);
        let invh = b.mul_f32(rl, ksf);
        let si: Vec<ValueId> = (0..16u8).map(|hh| b.read_lane(invh, hh)).collect();
        let gbits = b.bind_i32(ev(g) * 0x3F80_0000);
        let gf = b.let_(Op::BitcastF32(Opnd::Val(gbits)));
        let one = b.c_f32(1.0);
        let hf = b.sub_f32(one, gf);
        let inv_r: Vec<ValueId> = (0..8usize)
            .map(|r| {
                let a0 = b.let_(Op::BitcastF32(Opnd::Val(si[r])));
                let a1 = b.let_(Op::BitcastF32(Opnd::Val(si[8 + r])));
                let t0_ = b.mul_f32(a0, hf);
                let t1_ = b.mul_f32(a1, gf);
                b.add_f32(t0_, t1_)
            })
            .collect();
        // the wave's staging area: [16 heads][128 dims] f32 at w * 8 KB of the merge buffer
        let sbase = b.bind_i32(ev(w) * 8192 + (ev(g) * 8 * 128 + ev(m)) * 4 + mo.base as i32);
        b.wait_lds(0);
        b.barrier();
        mo.at(&t0).write(&mut b, |b| {
            for nt in 0..ntw {
                for r in 0..8u8 {
                    let e = b.frag_slice(Opnd::Var(ov[nt]), r, 1);
                    let ef = f32_of(b, e);
                    let y = b.mul_f32(ef, inv_r[r as usize]);
                    let a = b.bind_i32(ev(sbase) + ((r as i32) * 512 + (64 * nt as i32)));
                    b.store_lds(a, y, MemTy::B32, false);
                }
            }
        });
        b.wait_lds(0);
        b.barrier();
        // lane l: head l / 2, dims 64 (l % 2) .. + 64 of the wave's 128
        let hh = b.bind_i32(ev(lane) >> 1);
        let dd = b.bind_i32((ev(lane) & 1) * 64);
        let rbase = b.bind_i32(ev(w) * 8192 + (ev(hh) * 128 + ev(dd)) * 4 + mo.base as i32);
        let vals: Vec<ValueId> = mo.at(&t0).read(&mut b, |b| {
            (0..16).map(|k| { let a = b.bind_i32(ev(rbase) + 16 * k); b.load_lds_b128(a) }).collect()
        });
        let ro = b.bind_i32((ev(bidx) * 16 + ev(hh)) * 1024 + (ev(db) + ev(dd)) * 2);
        let prr = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(ro) });
        for k in 0..8u32 {
            let parts: Vec<Opnd> = (0..4u32)
                .map(|i| {
                    let j = 8 * k + 2 * i;
                    let x0 = b.frag_slice(Opnd::Val(vals[(j / 4) as usize]), (j % 4) as u8, 1);
                    let x1 = b.frag_slice(Opnd::Val(vals[((j + 1) / 4) as usize]), ((j + 1) % 4) as u8, 1);
                    let f0 = f32_of(&mut b, x0);
                    let f1 = f32_of(&mut b, x1);
                    Opnd::Val(b.cvt_pk_bf16_f32(f0, f1))
                })
                .collect();
            let v = b.frag_cat(&parts);
            b.store_global(prr, 16 * k, v, MemTy::B128, false);
        }
        // the LSE of head m (both lane halves and every wave write the same value)
        let lg = b.log2_f32(lh);
        let s2 = b.add_f32(Opnd::Var(vm), lg);
        let ln2 = b.c_f32(std::f32::consts::LN_2);
        let lse = b.mul_f32(s2, ln2);
        let lo = b.bind_i32((ev(bidx) * 16 + ev(m)) * 4);
        let pl = b.let_(Op::PtrAdd { base: Opnd::Val(plse), byte_off: Opnd::Val(lo) });
        b.store_global(pl, 0, lse, MemTy::B32, false);
        return b.finish();
    }
    // ---- merge the four partials (the rings are dead: every wave is past its loop at the barrier) ----
    b.wait_lds(0);
    b.barrier();
    let mlw = |b: &mut FnBuilder, a: ValueId, v: Opnd| b.store_lds(a, v, MemTy::B32, false);
    ml.at(&t0).write(&mut b, |b| {
        // max of head m (both lane halves write the same value), the half-sum of lane (m, g)
        let am = b.bind_i32(ev(w) * 64 + ev(m) * 4 + ml.base as i32);
        mlw(b, am, Opnd::Var(vm));
        let al = b.bind_i32(256 + (ev(w) * 2 + ev(g)) * 64 + ev(m) * 4 + ml.base as i32);
        mlw(b, al, Opnd::Var(vl));
    });
    let mob = b.bind_i32(ev(w) * 32768 + (ev(g) * 4096 + ev(db) + ev(m)) * 4 + mo.base as i32);
    mo.at(&t0).write(&mut b, |b| {
        for nt in 0..ntw {
            for r in 0..8u8 {
                let e = b.frag_slice(Opnd::Var(ov[nt]), r, 1);
                // O_w [head 8 g + r][dim db + 16 nt + m]
                // the lane base once, the (row, tile) part as the DS offset
                let a = b.bind_i32(ev(mob) + ((r as i32) * 2048 + (64 * nt as i32)));
                b.store_lds(a, e, MemTy::B32, false);
            }
        }
    });
    b.wait_lds(0);
    b.barrier();
    stamp(&mut b);
    pf_here(&mut b, 1);
    // thread t: head h = t / 8, dims 64 (t % 8) .. + 64
    let h = b.bind_i32(ev(tid) >> 3);
    let d0 = b.bind_i32((ev(tid) & 7) << 6);
    // the waves holding this thread's dims: all four, or (dim split) the two of the dim half d0 / 256 (one per page group)
    let wl: Vec<ValueId> = if dg > 1 {
        // the dim group of d0, one wave per page group
        let hh = b.bind_i32(ev(d0) >> (9 - lg_dg));
        (0..ngrp).map(|k| b.bind_i32(ev(hh) + k * dg)).collect()
    } else {
        (0..4).map(|k| b.c_i32(k)).collect()
    };
    let nwl = wl.len();
    let (mx, lt) = ml.at(&t0).read(&mut b, |b| {
        let mw: Vec<ValueId> = (0..nwl)
            .map(|wi| {
                let a = b.bind_i32(ev(h) * 4 + ev(wl[wi]) * 64 + ml.base as i32);
                let v = b.load_lds(a, MemTy::B32);
                f32_of(b, v)
            })
            .collect();
        let mut mx = mw[0];
        for wi in 1..nwl {
            mx = b.max_f32(mx, mw[wi]);
        }
        let mut sc = Vec::new();
        let mut lt: Option<ValueId> = None;
        for wi in 0..nwl {
            let d = b.sub_f32(mw[wi], mx);
            let s = b.exp2_f32(d);
            sc.push(s);
            for gg in 0..2 {
                let a = b.bind_i32(ev(h) * 4 + (ev(wl[wi]) * 2 + gg) * 64 + (256 + ml.base as i32));
                let v = b.load_lds(a, MemTy::B32);
                let vf = f32_of(b, v);
                let t = b.mul_f32(vf, s);
                lt = Some(match lt {
                    None => t,
                    Some(x) => b.add_f32(x, t),
                });
            }
        }
        ((mx, sc), lt.unwrap())
    });
    let (mx, sc) = mx;
    stamp(&mut b);
    if split {
        // the partial of this work-group: O un-normalised (scaled to the merged max), the max and the sum of head h
        let fb: Vec<ValueId> = (0..nwl).map(|wi| b.bind_i32(ev(h) * 2048 + ev(d0) * 4 + ev(wl[wi]) * 32768 + mo.base as i32)).collect();
    let outs: Vec<ValueId> = mo.at(&t0).read(&mut b, |b| {
            let mut acc: Vec<Option<ValueId>> = vec![None; 64];
            for wi in 0..nwl {
                let v4s: Vec<ValueId> = (0..16).map(|k| { let a = b.bind_i32(ev(fb[wi]) + 16 * k); b.load_lds_b128(a) }).collect();
                for k in 0..16 {
                    let v4 = v4s[k as usize];
                    for e in 0..4u8 {
                        let x = b.frag_slice(Opnd::Val(v4), e, 1);
                        let xf = f32_of(b, x);
                        let t = b.mul_f32(xf, sc[wi as usize]);
                        let i = 4 * k as usize + e as usize;
                        acc[i] = Some(match acc[i] {
                            None => t,
                            Some(y) => b.add_f32(y, t),
                        });
                    }
                }
            }
            acc.into_iter().map(|x| x.unwrap()).collect()
        });
        let pws = b.let_(Op::KernargPtr { offset: 80 });
        let s_count = b.let_(Op::KernargI32 { offset: 100 });
        let wsb = b.bind_i32((ev(bidx) * ev(s_count) + ev(sid)) * WS_BYTES as i32);
        let wo = b.bind_i32(ev(wsb) + (ev(h) * 512 + ev(d0)) * 4);
        let pwo = b.let_(Op::PtrAdd { base: Opnd::Val(pws), byte_off: Opnd::Val(wo) });
        for k in 0..16u32 {
            let parts: Vec<Opnd> = (0..4).map(|i| Opnd::Val(b.let_(Op::BitcastI32(Opnd::Val(outs[4 * k as usize + i]))))).collect();
            let v = b.frag_cat(&parts);
            b.store_global(pwo, 16 * k, v, MemTy::B128, false);
        }
        let wm = b.bind_i32(ev(wsb) + (8192 + ev(h)) * 4);
        let pwm = b.let_(Op::PtrAdd { base: Opnd::Val(pws), byte_off: Opnd::Val(wm) });
        b.store_global(pwm, 0, mx, MemTy::B32, false);
        b.store_global(pwm, 64, lt, MemTy::B32, false);
        return b.finish();
    }
    let rl = b.rcp_f32(lt);
    let inv = b.mul_f32(rl, ksf);
    let scaled: Vec<ValueId> = sc.iter().map(|&s| b.mul_f32(s, inv)).collect();
    let fb: Vec<ValueId> = (0..nwl).map(|wi| b.bind_i32(ev(h) * 2048 + ev(d0) * 4 + ev(wl[wi]) * 32768 + mo.base as i32)).collect();
    let outs: Vec<ValueId> = mo.at(&t0).read(&mut b, |b| {
        let mut acc: Vec<Option<ValueId>> = vec![None; 64];
        for wi in 0..nwl {
            let v4s: Vec<ValueId> = (0..16).map(|k| { let a = b.bind_i32(ev(fb[wi]) + 16 * k); b.load_lds_b128(a) }).collect();
            for k in 0..16 {
                let v4 = v4s[k as usize];
                for e in 0..4u8 {
                    let x = b.frag_slice(Opnd::Val(v4), e, 1);
                    let xf = f32_of(b, x);
                    let t = b.mul_f32(xf, scaled[wi as usize]);
                    let i = 4 * k as usize + e as usize;
                    acc[i] = Some(match acc[i] {
                        None => t,
                        Some(y) => b.add_f32(y, t),
                    });
                }
            }
        }
        acc.into_iter().map(|x| x.unwrap()).collect()
    });
    stamp(&mut b);
    // bf16 pairs -> 8 x 16-byte stores of R[b][h][d0 ..]
    let rb_off = b.bind_i32((ev(bidx) * 16 + ev(h)) * 1024 + ev(d0) * 2);
    let prb = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(rb_off) });
    for k in 0..8u32 {
        let parts: Vec<Opnd> = (0..4).map(|i| Opnd::Val(b.cvt_pk_bf16_f32(outs[8 * k as usize + 2 * i], outs[8 * k as usize + 2 * i + 1]))).collect();
        let v = b.frag_cat(&parts);
        if !tstamp {
            b.store_global(prb, 16 * k, v, MemTy::B128, false);
        }
    }
    // LSE of head h (the 8 threads of the head write the same value)
    let lg = b.log2_f32(lt);
    let s2 = b.add_f32(mx, lg);
    let ln2 = b.c_f32(std::f32::consts::LN_2);
    let lse = b.mul_f32(s2, ln2);
    let lo = b.bind_i32((ev(bidx) * 16 + ev(h)) * 4);
    let pl = b.let_(Op::PtrAdd { base: Opnd::Val(plse), byte_off: Opnd::Val(lo) });
    if tstamp {
        stamp(&mut b);
        return b.finish();
    }
    b.store_global(pl, 0, lse, MemTy::B32, false);
    b.finish()
}

/// Minimal repro for the branch-free rescale: `nv` loop-carried Frag8 accumulators; per iteration each is scaled in place per row
/// (`var_scale_lane`, alpha[r] = in[(i * 8 + r) * 32 + lane]), unconditionally or under `any_lane(alpha < 1)` (`branch`), then accumulated
/// in place by a WMMA of fp8 ones x twos (+128 per element). Output out[(v * 8 + r) * 32 + lane]. Host: `g1250_probe_host varscale`.
pub fn probe_g1250_varscale(name: &str, nv: usize, branch: bool) -> Function {
    let mut b = FnBuilder::new(name, 32);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg("inp", DType::Ptr, 8, 8);
    b.arg("n", DType::I32, 16, 4);
    b.arg_tensor("out", Ext::c((nv * 8 * 32 * 4) as i64));
    // the host allocates the input for the 64 iterations the loop counter is bounded by
    b.arg_tensor("inp", Ext::c(64 * 8 * 32 * 4));
    b.ka_range(16, 1, 64);
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let pi = b.let_(Op::KernargPtr { offset: 8 });
    let n = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let ones = b.c_i32(0x3838_3838);
    let fa = b.frag_cat(&[Opnd::Val(ones); 8]);
    // B = fp8 2.0 (distinct registers from A): +128 per element per WMMA
    let twos = b.c_i32(0x4040_4040);
    let fb = b.frag_cat(&[Opnd::Val(twos); 8]);
    let mut inits = Vec::new();
    for _ in 0..nv {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    let ov = b.declare_vars(&inits);
    let z = b.c_i32(0);
    let lv = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let i = lv[0];
    b.assume(i, Ext::c(0), Ext::c(63));
    let al: Vec<ValueId> = (0..8)
        .map(|r| {
            let off = b.bind_i32(((amdgpu_dsl::varx(i) * 8 + r) * 32 + ev(tid)) * 4);
            let pa = b.let_(Op::PtrAdd { base: Opnd::Val(pi), byte_off: Opnd::Val(off) });
            let v = b.load_global(pa, 0, MemTy::B32);
            f32_of(&mut b, v)
        })
        .collect();
    if branch {
        let lt = b.cmp(CmpPred::Lt, CmpTy::F32, al[0], Opnd::F32(1.0));
        let any = b.let_(Op::AnyLane(Opnd::Val(lt)));
        let one = b.c_i32(1);
        let anyb = b.cmp(CmpPred::Eq, CmpTy::I32, one, Opnd::I32(1));
        let _ = (any, anyb);
        b.begin_if(any);
    }
    for v in 0..nv {
        for r in 0..8u8 {
            b.var_scale_lane(ov[v], r, al[r as usize]);
        }
    }
    if branch {
        b.end_if();
    }
    for v in 0..nv {
        let d = b.wmma(WmmaKind::Fp8_16x16x64, fa, fb, Some(Opnd::Var(ov[v])));
        b.set_var(ov[v], Opnd::Val(d));
    }
    let i1 = b.bind_i32(amdgpu_dsl::varx(i) + 1);
    b.set_var(i, Opnd::Val(i1));
    let fin = b.cmp(CmpPred::Ge, CmpTy::I32, i1, n);
    b.break_if(fin);
    b.end_loop();
    for v in 0..nv {
        for r in 0..8u8 {
            let e = b.frag_slice(Opnd::Var(ov[v]), r, 1);
            let off = b.bind_i32(((v as i32 * 8 + r as i32) * 32 + ev(tid)) * 4);
            let pa = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
            b.store_global(pa, 0, e, MemTy::B32, false);
        }
    }
    b.finish()
}
