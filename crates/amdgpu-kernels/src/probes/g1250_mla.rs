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
    use amdgpu_dsl::{ev, TensorTile, WmmaKind};
    use amdgpu_stage::attention::mla::{MlaBody, LDS_ROPE, LDS_ROW, PAGE_TOKENS, SLOT_BYTES};
    use amdgpu_stage::{Cadence, Pipeline};
    use amdgpu_tile::lane::Role;
    use amdgpu_tile::shared::{Layout, LdsPlan};
    let mut b = FnBuilder::new(MLA_NAME, 128);
    let names = ["R", "LSE", "Q", "KV", "LTP", "LTD", "LTL", "QTP", "QS", "KS", "VSC"];
    for (i, n) in names.iter().enumerate() {
        b.arg(n, DType::Ptr, 8 * i as u32, 8);
    }
    for (i, n) in ["out16", "scalar", "qlens", "passes", "stride_page", "log2_page", "stride_q", "flags"].iter().enumerate() {
        b.arg(n, DType::I32, 88 + 4 * i as u32, 4);
    }
    b.arg_tensor("R", Ext::c(1 << 30));
    b.arg_tensor("LSE", Ext::c(1 << 28));
    let mut lds = LdsPlan::new();
    let kv = lds.ring("KV", SLOT_BYTES as i64, 3, Layout::Linear);
    let red = lds.tile("RED", 512, Layout::Linear);
    let pbuf = lds.tile("PBUF", 1024, Layout::Linear);
    let abuf = lds.tile("ABUF", 256, Layout::Linear);
    let ptab = lds.tile("PTAB", 4 * MAX_PAGES as i64, Layout::Linear);
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
        let v = b.load_global(q, 0, MemTy::B32);
        b.read_first_lane(v)
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
    let tile = TensorTile { data_size: 8, tensor_dim: [Opnd::I32(1152), Opnd::I32(1)], stride0: Opnd::I32(1152), tile: [1152, 1], pad: Some((128, 4)), wg_mask: 0 };
    let (_, g1) = b.tensor_desc_2d(&tile, Opnd::Val(pkv), Opnd::I32(0), Opnd::I32(0));
    // loop-invariant lane offsets inside a page slot
    let kofs = b.bind_i32((ev(w) * 16 + ev(m)) * LDS_ROW + ev(g) * 32);
    let rrow = b.bind_i32(ev(w) * 16 + ev(m));
    let rofs = b.bind_i32(ev(rrow) * 64 + ((ev(rrow) >> 3) << 4) + ev(g) * 32 + LDS_ROPE);
    // V tr8: fetch lane f reads 8 dims at token 8 (f >> 4) + (f & 3) + 4 ((f >> 3) & 1) of a 16-token chunk, dims + 8 ((f >> 2) & 1)
    let vofs = b.bind_i32((((ev(lane) >> 4) << 3) + (ev(lane) & 3) + (((ev(lane) >> 3) & 1) << 2)) * LDS_ROW + (((ev(lane) >> 2) & 1) << 3) + ev(w) * 128);
    // the sequence's page ids into LDS: 128 per pass (indices clamped to the last page)
    let nch = b.bind_i32((ev(npages) + 127) >> 7);
    let k0 = b.c_i32(0);
    let lv = b.begin_loop(&[(DType::I32, Opnd::Val(k0))]);
    let kk = lv[0];
    let idx = b.bind_i32(amdgpu_dsl::varx(kk) * 128 + ev(tid));
    let last_i = b.bind_i32(ev(npages) - 1);
    let lt = b.cmp(amdgpu_dsl::CmpPred::Lt, amdgpu_dsl::CmpTy::I32, idx, last_i);
    let ci = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lt), a: Opnd::Val(idx), b: Opnd::Val(last_i) });
    let eo = b.bind_i32((ev(kv0) + ev(ci)) * 4);
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
    let mut body = MlaBody { kv, red: red.clone(), pbuf, abuf, pkv, ptab, ctx, g1, w, m, g, lane, q, kofs, rofs, vofs, cscale, vars: Vec::new() };
    let pipe = Pipeline::new(Cadence::OneWindow, npages, ctx, PAGE_TOKENS, Ext::c(1 << 16));
    pipe.run(&mut b, Role::All, &mut body);
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
    for nt in 0..8u32 {
        let o = vars[2 + nt as usize];
        for r in 0..8u8 {
            let e = b.frag_slice(Opnd::Var(o), r, 1);
            let ef = b.let_(Op::BitcastF32(Opnd::Val(e)));
            let y = b.mul_f32(ef, inv[r as usize]);
            let h = b.cvt_pk_bf16_f32(y, y);
            // R[b][8 g + r][128 w + 16 nt + m]
            let off = b.bind_i32(((ev(bidx) * 16 + ev(g) * 8 + r as i32) * 512 + ev(w) * 128 + (16 * nt as i32) + ev(m)) * 2);
            let pa = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(off) });
            b.store_u16(pa, 0, h);
        }
    }
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
