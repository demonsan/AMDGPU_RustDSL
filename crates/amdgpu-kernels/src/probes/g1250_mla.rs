//! gfx1250 MLA decode pieces: the `ds_load_tr8_b64` lane-mapping probe (the ISA gives the mapping only as diagrams).

use amdgpu_dsl::{DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};

/// `ds_load_tr8_b64` mapping probe (one wave): LDS bytes 0..512 hold their own address (byte i = i mod 256 + 256 * (i >= 256) is not
/// needed: the probe reads within the first 256 bytes), lane l reads at `(l % 16) * 16 + (l / 16) * 8` (8 contiguous bytes of memory row l % 16,
/// half l / 16 of a 16 x 16 byte tile) and stores its two result dwords to out[2 l], out[2 l + 1]. The host prints which memory byte landed
/// in which (lane, byte) of the result.
pub fn probe_g1250_tr8() -> Function {
    let mut b = FnBuilder::new("probe_g1250_tr8", 32);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::c(64 * 4));
    b.lds(256);
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    // fill: lane l writes dwords 2 l and 2 l + 1 (bytes 8 l .. 8 l + 7), byte value = its address
    for h in 0..2 {
        let k = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(1)));
        let k = b.add_i32(k, Opnd::I32(h));
        let mul = b.mul_i32(k, Opnd::I32(0x0404_0404));
        let v = b.add_i32(mul, Opnd::I32(0x0302_0100));
        let a = b.let_(Op::ShlI32(Opnd::Val(k), Opnd::I32(2)));
        b.store_lds(a, v, MemTy::B32, false);
    }
    b.barrier();
    let row = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(15)));
    let half = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(4)));
    let ra = b.let_(Op::ShlI32(Opnd::Val(row), Opnd::I32(4)));
    let ha = b.let_(Op::ShlI32(Opnd::Val(half), Opnd::I32(3)));
    let addr = b.add_i32(ra, ha);
    let x = b.load_lds_tr_b8(addr);
    let off = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(3)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
    let x0 = b.frag_slice(Opnd::Val(x), 0, 1);
    let x1 = b.frag_slice(Opnd::Val(x), 1, 1);
    b.store_global(q, 0, x0, MemTy::B32, false);
    b.store_global(q, 4, x1, MemTy::B32, false);
    b.finish()
}

/// The MLA decode name (qh16, one split, the aiter gfx1250 ABI: the 120-byte kernarg of tests/gpu/g1250_mla_host.cpp).
pub const MLA_NAME: &str = "mla_g1250_qh16";
/// pages per sequence the LDS page table holds (ctx <= 65536)
const MAX_PAGES: i32 = 1024;

/// MLA decode, 16 query heads, fp8 Q / KV with per-tensor scales, page size 64, one KV split: grid (1, B, 1), block 128 (4 waves), one
/// sequence per work-group. The stage body is `amdgpu_stage::attention::mla::MlaBody` on the `OneWindow` pipeline (3 pages in flight through
/// the tensor data mover). Output: bf16 R [B][16][512], the natural-log LSE [B][16].
pub fn mla_g1250_qh16() -> Function {
    mla_g1250_qh16_with(MLA_NAME, false, 3, false)
}

/// The window kernel with the wave-split kernel's later improvements that apply here: the GEMM compile policy and scalar loads for the
/// sequence bounds / scales (the exponent batching is in the shared body).
pub fn mla_g1250_qh16_g() -> Function {
    G_MODE.with(|c| c.set(true));
    let f = mla_g1250_qh16_with("mla_g1250_qh16_g", false, 3, false);
    G_MODE.with(|c| c.set(false));
    f
}

thread_local! {
    static G_MODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The MLA decode with the KV pages loaded by per-lane async 16-byte copies instead of the tensor data mover (`async_kv`).
pub fn mla_g1250_qh16_async() -> Function {
    mla_g1250_qh16_with("mla_g1250_qh16_async", true, 3, false)
}

/// The MLA decode with a 4-slot ring (the producer runs 3 tiles ahead).
pub fn mla_g1250_qh16_s4() -> Function {
    mla_g1250_qh16_with("mla_g1250_qh16_s4", false, 4, false)
}

/// The MLA decode with the cross-tile prefetch window: S of tile t + 1 is computed in window t (carried), overlapping tile t's PV.
pub fn mla_g1250_qh16_pf() -> Function {
    mla_g1250_qh16_with("mla_g1250_qh16_pf", false, 4, true)
}

fn mla_g1250_qh16_with(name: &str, async_kv: bool, slots: u8, prefetch: bool) -> Function {
    use amdgpu_dsl::{ev, TensorTile, WmmaKind};
    use amdgpu_stage::attention::mla::{MlaBody, PageLayout, DV as DVB, PAGES, PAGE_BYTES, PAGE_TOKENS, QUARTER, ROPE_OFF};
    let layout = PageLayout::plan();
    use amdgpu_stage::{Cadence, Pipeline};
    use amdgpu_tile::lane::Role;
    use amdgpu_tile::shared::{Layout, LdsPlan};
    let mut b = FnBuilder::new(name, 128);
    let gmode = G_MODE.with(|c| c.get());
    if gmode {
        b.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
    }
    let names = ["R", "LSE", "Q", "KV", "LTP", "LTD", "LTL", "QTP", "QS", "KS", "VSC"];
    for (i, n) in names.iter().enumerate() {
        b.arg(n, DType::Ptr, 8 * i as u32, 8);
    }
    for (i, n) in ["out16", "scalar", "qlens", "passes", "stride_page", "log2_page", "stride_q", "flags"].iter().enumerate() {
        b.arg(n, DType::I32, 88 + 4 * i as u32, 4);
    }
    // the buffers of grid (1, B, 1): one sequence of 16 heads per work-group row
    b.arg_tensor("R", Ext::grid_y() * Ext::c(16 * 512 * 2));
    b.arg_tensor("LSE", Ext::grid_y() * Ext::c(16 * 4));
    b.arg_tensor("Q", Ext::grid_y() * Ext::c(16 * 768));
    b.arg_tensor("LTP", (Ext::grid_y() + Ext::c(1)) * Ext::c(4));
    b.arg_tensor("LTL", Ext::grid_y() * Ext::c(4));
    b.arg_tensor("QS", Ext::c(4));
    b.arg_tensor("KS", Ext::c(4));
    // the paged KV pool (P pages) and the page table (NPG entries): free symbols, the host contract on the loaded indices below
    b.sym_range("P", 1, 1 << 24);
    b.sym_range("NPG", 1, 1 << 26);
    b.arg_tensor("KV", Ext::sym("P") * Ext::c(PAGE_BYTES as i64));
    b.arg_tensor("LTD", Ext::sym("NPG") * Ext::c(4));
    let mut lds = LdsPlan::new();
    let kv = lds.ring("KV", (PAGES * layout.slot_bytes) as i64, slots as u32, Layout::Linear);
    let red = lds.tile("RED", 512, Layout::Linear);
    let pbuf = lds.tile("PBUF", 1024 * PAGES as i64, Layout::Linear);
    let abuf = lds.tile("ABUF", 256, Layout::Linear);
    let ptab = lds.tile("PTAB", 4 * MAX_PAGES as i64, Layout::Linear);
    let out = lds.tile_over("OUT", 16 * 512 * 2, &[&kv]);
    lds.commit(&mut b);
    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 8 * i });
    let (pr, plse, pq, pkv, pltp, pltd, pltl) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4), kp(&mut b, 5), kp(&mut b, 6));
    let (pqs, pks) = (kp(&mut b, 8), kp(&mut b, 9));
    let tid = b.let_(Op::WorkitemId);
    let bidx = b.let_(Op::WorkgroupIdY);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let wv = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let w = b.read_first_lane(wv);
    let m = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(15)));
    let g = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::I32(4)));
    // uniform scalars of the sequence
    let ld = |b: &mut FnBuilder, p: ValueId, off: ValueId| {
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
        if gmode {
            b.let_(Op::LoadScalar { base: Opnd::Val(q), byte_off: 0 })
        } else {
            let v = b.load_global(q, 0, MemTy::B32);
            b.read_first_lane(v)
        }
    };
    let ld_v = |b: &mut FnBuilder, p: ValueId, off: ValueId| {
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
        b.load_global(q, 0, MemTy::B32)
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
    let full = b.add_i32(npages, Opnd::I32(-1));
    let ctx0 = b.let_(Op::ShlI32(Opnd::Val(full), Opnd::I32(6)));
    let ctx = b.add_i32(ctx0, last);
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
    // Q fragments: head m (row 768 bytes, sequence stride 16 rows), dims 64 c + 8 g + 16 j
    let qrow = b.bind_i32((ev(bidx) * 16 + ev(m)) * 768 + ev(g) * 32);
    let pqr = b.let_(Op::PtrAdd { base: Opnd::Val(pq), byte_off: Opnd::Val(qrow) });
    let q: Vec<ValueId> = (0..9u32)
        .map(|c| {
            let parts: Vec<Opnd> = (0..2u32).map(|j| Opnd::Val(b.load_global(pqr, 64 * c + 16 * j, MemTy::B128))).collect();
            b.frag_cat(&parts)
        })
        .collect();
    // one page = 576 rows x 64 bytes (contiguous); each wave loads a quarter (144 rows)
    let tile = TensorTile { data_size: 8, tensor_dim: [Opnd::I32(1152), Opnd::I32(1)], stride0: Opnd::I32(1152), tile: [1152, 1], pad: layout.pad, wg_mask: 0 };
    let (_, g1) = b.tensor_desc_2d(&tile, Opnd::Val(pkv), Opnd::I32(0), Opnd::I32(0));
    // loop-invariant lane offsets inside a page slot
    // lane bases as logical page offsets, mapped to the planned LDS layout (the quarter split: a quarter is a whole number of pad intervals)
    let kofs = layout.at_expr(&mut b, (ev(w) * 16 + ev(m)) * DVB + ev(g) * 32);
    let rofs = layout.at_expr(&mut b, (ev(w) * 16 + ev(m)) * 64 + ev(g) * 32 + ROPE_OFF);
    // V tr8: fetch lane f reads 8 dims at token 8 (f >> 4) + (f & 3) + 4 ((f >> 3) & 1) of a 16-token chunk, dims + 8 ((f >> 2) & 1)
    let vofs = layout.at_expr(&mut b, (((ev(lane) >> 4) << 3) + (ev(lane) & 3) + (((ev(lane) >> 3) & 1) << 2)) * DVB + (((ev(lane) >> 2) & 1) << 3) + ev(w) * 128);
    // the constant parts' LDS deltas, checked uniform over every lane and wave base
    let kbases: Vec<i32> = (0..128).map(|t| (t / 32 * 16 + (t & 15)) * DVB + ((t & 31) >> 4) * 32).collect();
    let kdelta: Vec<i32> = (0..8).map(|c| layout.delta(&kbases, 64 * c)).collect();
    let vbases: Vec<i32> = (0..128).map(|t| { let f = t & 31; (8 * (f >> 4) + (f & 3) + 4 * ((f >> 3) & 1)) * DVB + 8 * ((f >> 2) & 1) + (t / 32) * 128 }).collect();
    let vdelta: Vec<i32> = (0..32).map(|i| layout.delta(&vbases, 16 * (i / 8) * DVB + 16 * (i % 8))).collect();
    let _ = QUARTER;
    // the sequence's page ids into LDS: 128 per pass (indices clamped to the last page)
    let nch = b.bind_i32((ev(npages) + 127) >> 7);
    let k0 = b.c_i32(0);
    let lv = b.begin_loop(&[(DType::I32, Opnd::Val(k0))]);
    let kk = lv[0];
    b.assume(kk, Ext::c(0), Ext::c((MAX_PAGES / 128 - 1) as i64));
    let idx = b.bind_i32(amdgpu_dsl::varx(kk) * 128 + ev(tid));
    // the entry kv0 + idx, clamped to the sequence's last entry (kv_end - 1 < NPG)
    let e_raw = b.bind_i32(ev(kv0) + ev(idx));
    let e_last = b.bind_i32(ev(kv_end) - 1);
    let ci = b.let_(Op::MinI32(Opnd::Val(e_raw), Opnd::Val(e_last)));
    let eo = b.bind_i32(ev(ci) * 4);
    let pv = ld_v(&mut b, pltd, eo);
    let sa = b.bind_i32(ev(idx) * 4 + ptab.base as i32);
    ptab.at(&amdgpu_tile::shared::SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).write(&mut b, |b| b.store_lds(sa, pv, MemTy::B32, false));
    let k1 = b.bind_i32(amdgpu_dsl::varx(kk) + 1);
    b.set_var(kk, Opnd::Val(k1));
    let done = b.cmp(amdgpu_dsl::CmpPred::Ge, amdgpu_dsl::CmpTy::I32, k1, nch);
    b.break_if(done);
    b.end_loop();
    b.wait_lds(0);
    b.barrier();
    let mut body = MlaBody { kv, red: red.clone(), pbuf, abuf, pkv, ptab, ctx, g1, w, m, g, lane, q, kofs, rofs, vofs, kdelta, vdelta, layout, cscale, npages, async_kv, tid, depth: slots, prefetch, s0: Vec::new(), vars: Vec::new() };
    // one pipeline tile = PAGES pages (the last one possibly partial: its clamped pages are masked by ctx)
    let ntiles = b.bind_i32((ev(npages) + (PAGES - 1)) >> PAGES.trailing_zeros() as i32);
    let mut pipe = Pipeline::new(Cadence::OneWindow, ntiles, ctx, PAGE_TOKENS * PAGES, Ext::c(1 << 16));
    pipe.slots = slots;
    pipe.run(&mut b, Role::All, &mut body);
    // drain the tensor data mover: the last steady iteration leaves one (clamped, re-staged) page load in flight. A tensor write that lands after
    // s_endpgm writes LDS the next work-group on this CU may already own (its page table: a wild page id, an illegal access). The vendor ends with s_wait_idle.
    b.wait_tensor(0);
    b.wait_counter(amdgpu_dsl::WaitCtr::Async, 0);
    let vars = body.vars.clone();
    let _ = WmmaKind::Fp8_16x16x64;
    // ---- epilogue: the row sums (8 partials per head) through LDS, O / L * ks to bf16, LSE ----
    b.wait_lds(0);
    b.barrier();
    let rw = b.bind_i32(ev(w) * 128 + ev(g) * 64 + ev(m) * 4 + red.base as i32);
    b.store_lds(rw, Opnd::Var(vars[1]), MemTy::B32, false);
    b.wait_lds(0);
    b.barrier();
    let sum_head = |b: &mut FnBuilder, head: amdgpu_dsl::I32| -> ValueId {
        let mut acc: Option<ValueId> = None;
        for k in 0..8 {
            let a = b.bind_i32(head.clone() * 4 + (k * 64 + red.base as i32));
            let v = b.load_lds(a, MemTy::B32);
            let vf = b.let_(Op::BitcastF32(Opnd::Val(v)));
            acc = Some(match acc {
                None => vf,
                Some(s) => b.add_f32(s, vf),
            });
        }
        acc.unwrap()
    };
    let inv: Vec<ValueId> = (0..8)
        .map(|r| {
            let l = sum_head(&mut b, ev(g) * 8 + r);
            let rl = b.rcp_f32(l);
            b.mul_f32(rl, ksf)
        })
        .collect();
    // the bf16 output through LDS (the KV ring is dead after the drain): every lane writes its 64 values at [head][dim], then each thread
    // stores 128 contiguous bytes of the sequence's 16 x 512 block with 16-byte stores
    let ob = out.base as i32;
    out.at(&amdgpu_tile::shared::SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).write(&mut b, |b| {
        for nt in 0..8u32 {
            let o = vars[2 + nt as usize];
            for r in 0..8u8 {
                let e = b.frag_slice(Opnd::Var(o), r, 1);
                let ef = b.let_(Op::BitcastF32(Opnd::Val(e)));
                let y = b.mul_f32(ef, inv[r as usize]);
                let h = b.cvt_pk_bf16_f32(y, y);
                // [8 g + r][128 w + 16 nt + m]
                let a = b.bind_i32(((ev(g) * 8 + r as i32) * 512 + ev(w) * 128 + (16 * nt as i32) + ev(m)) * 2 + ob);
                b.store_lds(a, h, MemTy::U16, false);
            }
        }
    });
    b.wait_lds(0);
    b.barrier();
    let rb = b.bind_i32(ev(bidx) * (16 * 512 * 2) + ev(tid) * 128);
    let prb = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(rb) });
    out.at(&amdgpu_tile::shared::SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).read(&mut b, |b| {
        let rows: Vec<ValueId> = (0..8).map(|k| {
            let a = b.bind_i32(ev(tid) * 128 + (16 * k + ob));
            b.load_lds_b128(a)
        }).collect();
        for k in 0..8u32 {
            b.store_global(prb, 16 * k, rows[k as usize], MemTy::B128, false);
        }
    });
    // LSE (natural log) of head m
    // (wave 0: both lane halves write the same value of head m)
    let lt = b.cmp(amdgpu_dsl::CmpPred::Eq, amdgpu_dsl::CmpTy::I32, w, Opnd::I32(0));
    b.begin_if(lt);
    let l = sum_head(&mut b, ev(m));
    let lg = b.log2_f32(l);
    let s2 = b.add_f32(Opnd::Var(vars[0]), lg);
    let ln2 = b.c_f32(std::f32::consts::LN_2);
    let lse = b.mul_f32(s2, ln2);
    let off = b.bind_i32((ev(bidx) * 16 + ev(m)) * 4);
    let pl = b.let_(Op::PtrAdd { base: Opnd::Val(plse), byte_off: Opnd::Val(off) });
    b.store_global(pl, 0, lse, MemTy::B32, false);
    b.end_if();
    b.finish()
}
