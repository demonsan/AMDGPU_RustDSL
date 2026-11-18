//! The MLA decode stage body (gfx1250): one work-group = one sequence of 16 query heads, 4 waves, KV in pages of 64 tokens x 576 fp8 bytes
//! (the 64 x 512 "nope" plane, then the 64 x 64 "rope" plane) streamed by the tensor data mover through a 3-slot ring (`Cadence::OneWindow`).
//!
//! Per page (one window):
//! * S^T = K Q^T: wave w computes the 16 tokens 16 w .. 16 w + 15 of the page against the 16 heads: 9 fp8 WMMA 16x16x64 over the 576 dims
//!   (A = 16 K rows from the slot, B = the wave's Q fragments held in registers). Lane (m, g) holds head m, tokens 16 w + 8 g + r (r = 0..8).
//! * online softmax (base 2): the per-head page maximum through LDS (8 partial maxima per head: 4 waves x 2 lane halves), the rescale factor
//!   alpha of the running maximum, p = exp2(x - max) and the per-lane partial row sum (reduced once at the end).
//! * P to fp8 through LDS: lane (m, g) of wave w writes its 8 probabilities; the 4 x 8 bytes lane (m, g) reads back from the 4 waves are
//!   exactly the A fragment of P (16 heads x 64 tokens) for the PV WMMA.
//! * O += P V: wave w owns the V dims 128 w .. 128 w + 127 (8 tiles of 16); the B fragments (16 dims x 64 tokens) come from the slot's nope
//!   plane with `ds_load_tr8_b64` (the transposing LDS read; lane mapping measured by `probe_g1250_tr8`).

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, FnBuilder, MemTy, Op, Opnd, TdmCpol, ValueId, VarId, WmmaKind};
use amdgpu_tile::lane::{Role, TileIt};
use amdgpu_tile::shared::{Shared, SlotIdx};

use crate::gemm::tdm::tdm_group0;
use crate::PipeBody;

/// Page geometry.
pub const PAGE_TOKENS: i32 = 64;
pub const D: i32 = 576;
pub const DV: i32 = 512;
pub const PAGE_BYTES: i32 = PAGE_TOKENS * D;
/// byte offset of the rope plane inside a page
pub const ROPE_OFF: i32 = PAGE_TOKENS * DV;

pub struct MlaBody {
    /// the KV ring (one slot = one page)
    pub kv: Shared,
    /// cross-wave exchange buffers: page maxima (8 x 16 f32), P (4 waves x 32 lanes x 8 bytes), alpha per wave (4 x 16 f32)
    pub red: Shared,
    pub pbuf: Shared,
    pub abuf: Shared,
    /// kernarg pointers of KV and of the page table (`LTD`), the sequence's first page-table entry, its page count
    pub pkv: ValueId,
    pub pltd: ValueId,
    pub kv0: ValueId,
    /// valid tokens of the sequence
    pub ctx: ValueId,
    /// group 1 of the page descriptor (one 64 x 576 byte tile)
    pub g1: ValueId,
    /// lane roles
    pub w: ValueId,
    pub m: ValueId,
    pub g: ValueId,
    pub lane: ValueId,
    /// Q fragments (B operand of S^T): 9 x Frag8, head m, dims 64 c + 8 g + 16 j
    pub q: Vec<ValueId>,
    /// qs * ks * softmax scale * log2(e)
    pub cscale: ValueId,
    /// loop state: running max (head m), partial row sum (this lane's tokens), the 8 O tiles (Frag8 each)
    pub vars: Vec<VarId>,
}

fn f32_of(b: &mut FnBuilder, v: ValueId) -> ValueId {
    b.let_(Op::BitcastF32(Opnd::Val(v)))
}

fn bits_of(b: &mut FnBuilder, v: ValueId) -> Opnd {
    Opnd::Val(b.let_(Op::BitcastI32(Opnd::Val(v))))
}

impl PipeBody for MlaBody {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &amdgpu_dsl::I32, slot: &SlotIdx) {
        // page t = key0 / 64: its id from the page table, the descriptor of the whole page into the slot
        let t = b.bind_i32(key0.clone() >> 6);
        let e = b.add_i32(self.kv0, t);
        let eo = b.let_(Op::ShlI32(Opnd::Val(e), Opnd::I32(2)));
        let pe = b.let_(Op::PtrAdd { base: Opnd::Val(self.pltd), byte_off: Opnd::Val(eo) });
        let pidv = b.load_global(pe, 0, MemTy::B32);
        let pid = b.read_first_lane(pidv);
        let off = b.mul_i32(pid, Opnd::I32(PAGE_BYTES));
        let r = self.kv.at(slot);
        let lds = b.bind_i32(r.base());
        let g0 = tdm_group0(b, self.pkv, off, Opnd::Val(lds));
        let g1 = self.g1;
        r.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default()));
    }

    fn stage_complete(&mut self, b: &mut FnBuilder, _role: Role, _slot: &SlotIdx) {
        b.wait_tensor(0);
    }

    fn prologue_compute(&mut self, _b: &mut FnBuilder, _role: Role, _t0: &TileIt) {}

    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let neg = b.c_f32(-1.0e30);
        let zf = b.c_f32(0.0);
        let mut v = vec![(DType::F32, Opnd::Val(neg)), (DType::F32, Opnd::Val(zf))];
        for _ in 0..8 {
            let z = b.frag_cat(&[Opnd::I32(0); 8]);
            v.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
        }
        v
    }

    fn bind_state(&mut self, vars: &[VarId]) {
        self.vars = vars.to_vec();
    }

    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (vm, vl) = (self.vars[0], self.vars[1]);
        let (w, m, g, lane) = (self.w, self.m, self.g, self.lane);
        let kv = self.kv.at(&t.cur);
        // the single-slot exchange tiles: generation = the tile counter
        let tix = SlotIdx::new(Opnd::I32(0), t.it);
        let sb = b.bind_i32(kv.base());
        // ---- S^T: A = K rows (token 16 w + m), K bytes 8 g + 16 j of each 64-byte chunk; B = Q ----
        let row = b.bind_i32(ev(w) * 16 + ev(m));
        let nope_row = b.bind_i32(ev(sb) + ev(row) * DV + ev(g) * 8);
        let rope_row = b.bind_i32(ev(sb) + ROPE_OFF + ev(row) * 64 + ev(g) * 8);
        let q = self.q.clone();
        let s = kv.read(b, |b| {
            let mut acc: Option<Opnd> = None;
            for c in 0..9 {
                let parts: Vec<Opnd> = (0..4)
                    .map(|j| {
                        let a = if c < 8 { b.bind_i32(ev(nope_row) + (64 * c + 16 * j)) } else { b.bind_i32(ev(rope_row) + 16 * j) };
                        Opnd::Val(b.load_lds_b64(a))
                    })
                    .collect();
                let fa = b.frag_cat(&parts);
                let d = b.wmma(WmmaKind::Fp8_16x16x64, fa, q[c as usize], acc);
                acc = Some(Opnd::Val(d));
            }
            match acc {
                Some(Opnd::Val(v)) => v,
                _ => unreachable!(),
            }
        });
        // ---- logits (base 2), masked beyond the sequence ----
        let tok0 = b.bind_i32(t.key0() + ev(w) * 16 + ev(g) * 8);
        let neg = b.c_f32(-1.0e30);
        let mut x = Vec::with_capacity(8);
        for r in 0..8u8 {
            let si = b.frag_slice(Opnd::Val(s), r, 1);
            let sf = f32_of(b, si);
            let v = b.mul_f32(sf, self.cscale);
            let tok = b.add_i32(tok0, Opnd::I32(r as i32));
            let ok = b.cmp(CmpPred::Lt, CmpTy::I32, tok, self.ctx);
            x.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(neg) }));
        }
        let mut mx = x[0];
        for r in 1..8 {
            mx = b.max_f32(mx, x[r]);
        }
        // ---- page maximum per head: 8 partials through LDS ----
        let red_w = b.bind_i32(ev(w) * 128 + ev(g) * 64 + ev(m) * 4 + self.red.base as i32);
        self.red.at(&tix).write(b, |b| b.store_lds(red_w, mx, MemTy::B32, false));
        b.wait_lds(0);
        b.barrier();
        let pm = self.red.at(&tix).read(b, |b| {
            let mut pm: Option<ValueId> = None;
            for k in 0..8 {
                let a = b.bind_i32(ev(m) * 4 + (k * 64 + self.red.base as i32));
                let v = b.load_lds(a, MemTy::B32);
                let vf = f32_of(b, v);
                pm = Some(match pm {
                    None => vf,
                    Some(p) => b.max_f32(p, vf),
                });
            }
            pm.unwrap()
        });
        let m_new = b.max_f32(Opnd::Var(vm), pm);
        let dm = b.sub_f32(Opnd::Var(vm), m_new);
        let alpha = b.exp2_f32(dm);
        let mut p = Vec::with_capacity(8);
        let mut lsum = b.mul_f32(Opnd::Var(vl), alpha);
        for r in 0..8 {
            let d = b.sub_f32(x[r], m_new);
            let e = b.exp2_f32(d);
            lsum = b.add_f32(lsum, e);
            p.push(e);
        }
        // ---- P (fp8) and alpha through LDS ----
        let p0 = b.cvt_pk_fp8x4_f32(p[0], p[1], p[2], p[3]);
        let p1 = b.cvt_pk_fp8x4_f32(p[4], p[5], p[6], p[7]);
        let pw = b.bind_i32((ev(w) * 32 + ev(lane)) * 8 + self.pbuf.base as i32);
        let aw = b.bind_i32(ev(w) * 64 + ev(m) * 4 + self.abuf.base as i32);
        self.pbuf.at(&tix).write(b, |b| {
            b.store_lds(pw, p0, MemTy::B32, false);
            let pw4 = b.add_i32(pw, Opnd::I32(4));
            b.store_lds(pw4, p1, MemTy::B32, false);
        });
        self.abuf.at(&tix).write(b, |b| b.store_lds(aw, alpha, MemTy::B32, false));
        b.wait_lds(0);
        b.barrier();
        // the A fragment of P: wave j's 8 bytes of lane (m, g) = tokens 16 j + 8 g .. + 8
        let pa = self.pbuf.at(&tix).read(b, |b| {
            let parts: Vec<Opnd> = (0..4)
                .map(|j| {
                    let a = b.bind_i32(ev(lane) * 8 + (j * 256 + self.pbuf.base as i32));
                    Opnd::Val(b.load_lds_b64(a))
                })
                .collect();
            b.frag_cat(&parts)
        });
        // alpha of the O rows: heads 8 g + r
        let al: Vec<ValueId> = self.abuf.at(&tix).read(b, |b| {
            (0..8)
                .map(|r| {
                    let a = b.bind_i32(ev(w) * 64 + ev(g) * 32 + (4 * r + self.abuf.base as i32));
                    let v = b.load_lds(a, MemTy::B32);
                    f32_of(b, v)
                })
                .collect()
        });
        // ---- O = alpha O + P V; wave w owns dims 128 w .. ; the V fragment by ds_load_tr8_b64 ----
        // fetch lane f reads 8 dims at token 8 (f >> 4) + (f & 3) + 4 ((f >> 3) & 1) of a 16-token chunk, dims + 8 ((f >> 2) & 1)
        let ktok = b.bind_i32(((ev(lane) >> 4) << 3) + (ev(lane) & 3) + (((ev(lane) >> 3) & 1) << 2));
        let kdim = b.bind_i32(((ev(lane) >> 2) & 1) << 3);
        let vrow = b.bind_i32(ev(sb) + ev(ktok) * DV + ev(kdim) + ev(w) * 128);
        let vars = self.vars.clone();
        kv.read(b, |b| {
            for nt in 0..8 {
                let o = vars[2 + nt];
                let scaled: Vec<Opnd> = (0..8u8)
                    .map(|r| {
                        let e = b.frag_slice(Opnd::Var(o), r, 1);
                        let ef = f32_of(b, e);
                        let y = b.mul_f32(ef, al[r as usize]);
                        bits_of(b, y)
                    })
                    .collect();
                let oc = b.frag_cat(&scaled);
                let parts: Vec<Opnd> = (0..4)
                    .map(|jj| {
                        let a = b.bind_i32(ev(vrow) + (16 * jj * DV + 16 * nt as i32));
                        Opnd::Val(b.load_lds_tr_b8(a))
                    })
                    .collect();
                let fv = b.frag_cat(&parts);
                let d = b.wmma(WmmaKind::Fp8_16x16x64, pa, fv, Some(Opnd::Val(oc)));
                b.set_var(o, Opnd::Val(d));
            }
        });
        b.set_var(vm, Opnd::Val(m_new));
        b.set_var(vl, Opnd::Val(lsum));
    }
}
