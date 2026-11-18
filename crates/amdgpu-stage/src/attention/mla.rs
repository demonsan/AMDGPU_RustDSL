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
/// byte offset of the rope plane inside a page (global layout)
pub const ROPE_OFF: i32 = PAGE_TOKENS * DV;
/// bytes of a page one wave loads (a quarter)
pub const QUARTER: i32 = PAGE_BYTES / 4;

/// The LDS layout of a page slot: the TDM padding (`amdgpu_dsl::tdm_pad`) planned from the window's read patterns, so the K reads and the
/// transposing V reads are bank-conflict free. Each wave's quarter is padded on its own (the padding restarts per tensor op), so the plan's
/// interval must divide a quarter: then a logical page offset maps with one formula.
#[derive(Clone, Copy, Debug)]
pub struct PageLayout {
    pub pad: Option<(u16, u16)>,
    /// LDS bytes of a quarter / a slot
    pub quarter_lds: i32,
    pub slot_bytes: i32,
}

impl PageLayout {
    pub fn plan() -> PageLayout {
        use amdgpu_dsl::tdm_pad::{footprint, plan, LaneRead};
        let mut reads = Vec::new();
        // K (wave 0; the other waves differ by whole 512-byte rows): lane (m, g) reads 2 x 16 bytes at row m, 64 c + 32 g
        for c in 0..8u32 {
            for h in 0..2u32 {
                reads.push(LaneRead { bytes: 16, lanes: (0..32u32).map(|l| Some((l & 15) * DV as u32 + 64 * c + 32 * (l >> 4) + 16 * h)).collect() });
            }
        }
        // V: ds_load_tr8_b64, fetch lane f at token 8 (f >> 4) + (f & 3) + 4 ((f >> 3) & 1), dims 8 ((f >> 2) & 1) (+ 16 nt)
        for nt in 0..8u32 {
            reads.push(LaneRead {
                bytes: 8,
                lanes: (0..32u32).map(|f| Some((8 * (f >> 4) + (f & 3) + 4 * ((f >> 3) & 1)) * DV as u32 + 8 * ((f >> 2) & 1) + 16 * nt)).collect(),
            });
        }
        let p = plan(QUARTER as u32, &reads);
        if let Some((iv, _)) = p.pad {
            assert!(QUARTER as u32 % (4 * iv as u32) == 0, "MLA page layout: pad interval {iv} dwords does not divide a quarter page");
        }
        let q = footprint(QUARTER as u32, p.pad) as i32;
        PageLayout { pad: p.pad, quarter_lds: q, slot_bytes: 4 * q }
    }
    /// LDS offset (in the slot) of logical page offset `off` (a constant)
    pub fn at(&self, off: i32) -> i32 {
        amdgpu_dsl::tdm_pad::padded(off as u32, self.pad) as i32
    }
    /// LDS offset of a logical page offset expression: off + (off >> log2(4 interval)) * 4 amount
    pub fn at_expr(&self, b: &mut FnBuilder, off: amdgpu_dsl::I32) -> ValueId {
        match self.pad {
            None => b.bind_i32(off),
            Some((iv, am)) => {
                let s = (4 * iv as u32).trailing_zeros() as i32;
                b.bind_i32(off.clone() + ((off >> s) * (4 * am as i32)))
            }
        }
    }
    /// The LDS delta of adding the constant `k` to every lane base `bases` (logical offsets): the same for all lanes, or panics (the read
    /// pattern would cross a padding boundary differently per lane).
    pub fn delta(&self, bases: &[i32], k: i32) -> i32 {
        let d = self.at(bases[0] + k) - self.at(bases[0]);
        for &x in bases {
            assert_eq!(self.at(x + k) - self.at(x), d, "MLA page layout: offset {k} crosses a padding boundary unevenly (base {x})");
        }
        d
    }
}

pub struct MlaBody {
    /// the KV ring (one slot = one page)
    pub kv: Shared,
    /// cross-wave exchange buffers: page maxima (8 x 16 f32), P (4 waves x 32 lanes x 8 bytes), alpha per wave (4 x 16 f32)
    pub red: Shared,
    pub pbuf: Shared,
    pub abuf: Shared,
    /// the kernarg pointer of KV; the sequence's page ids, staged once in LDS (`ptab[t]` = the page of tile t; a global read per page would put
    /// the memory latency on every window's critical path)
    pub pkv: ValueId,
    pub ptab: Shared,
    /// valid tokens of the sequence
    pub ctx: ValueId,
    /// group 1 of the descriptor of a quarter page (a 64 x 144 byte tile; wave w loads rows 144 w ..)
    pub g1: ValueId,
    /// lane roles
    pub w: ValueId,
    pub m: ValueId,
    pub g: ValueId,
    pub lane: ValueId,
    /// Q fragments (B operand of S^T): 9 x Frag8, head m, dims 64 c + 32 g .. + 32 (the K order inside a WMMA is free as long as A and B
    /// agree: lane half g holds 32 contiguous bytes of each 64-byte chunk, two b128 reads)
    pub q: Vec<ValueId>,
    /// lane offsets inside a slot (loop-invariant, LDS layout): K nope row (token 16 w + m, + 32 g), K rope row, the V tr8 fetch address of
    /// the wave's dims; the LDS deltas of the constant parts (K chunk c; V chunk jj, tile nt: index 8 jj + nt), the page layout
    pub kofs: ValueId,
    pub rofs: ValueId,
    pub vofs: ValueId,
    pub kdelta: Vec<i32>,
    pub vdelta: Vec<i32>,
    pub layout: PageLayout,
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
        let ta = b.bind_i32(((key0.clone() >> 6) << 2) + self.ptab.base as i32);
        let pidv = self.ptab.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).read(b, |b| b.load_lds(ta, MemTy::B32));
        let pid = b.read_first_lane(pidv);
        b.assume_trusted(pid, amdgpu_dsl::Ext::c(0), amdgpu_dsl::Ext::sym("P") - amdgpu_dsl::Ext::c(1), "host contract: every page-table entry < P, the KV pool's page count (KV holds P x 36864 bytes)");
        // each wave loads its quarter of the page (144 of the 576 rows of 64 bytes): one copy of the page per work-group
        let off = b.bind_i32(ev(pid) * PAGE_BYTES + ev(self.w) * QUARTER);
        let r = self.kv.at(slot);
        let lds = b.bind_i32(r.base() + ev(self.w) * self.layout.quarter_lds);
        let g0 = tdm_group0(b, self.pkv, off, Opnd::Val(lds));
        let g1 = self.g1;
        r.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default()));
    }

    fn stage_complete(&mut self, b: &mut FnBuilder, _role: Role, _slot: &SlotIdx) {
        // TENSORcnt is in order: at most the page just staged (t + 2) stays in flight; t + 1, read by the next window, is complete
        b.wait_tensor(1);
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
        let nope_row = b.bind_i32(ev(sb) + ev(self.kofs));
        let rope_row = b.bind_i32(ev(sb) + ev(self.rofs));
        let q = self.q.clone();
        let s = kv.read(b, |b| {
            // all nine K fragments first (distinct registers: the reads overlap), then the WMMA chain
            let fa: Vec<ValueId> = (0..9)
                .map(|c| {
                    let lo = if c < 8 { b.bind_i32(ev(nope_row) + self.kdelta[c as usize]) } else { b.bind_i32(ev(rope_row)) };
                    let hi = b.bind_i32(ev(lo) + 16);
                    b.load_lds_b128x2(lo, hi)
                })
                .collect();
            let mut acc: Option<Opnd> = None;
            for c in 0..9 {
                let d = b.wmma(WmmaKind::Fp8_16x16x64, fa[c], q[c], acc);
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
        // P rows: lane l's 32 bytes = the 4 waves' 8 bytes (wave w at byte 8 w): one b128 pair per lane when read back
        let pw = b.bind_i32(ev(lane) * 32 + ev(w) * 8 + self.pbuf.base as i32);
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
            let lo = b.bind_i32(ev(lane) * 32 + self.pbuf.base as i32);
            let hi = b.bind_i32(ev(lo) + 16);
            b.load_lds_b128x2(lo, hi)
        });
        // alpha of the O rows: heads 8 g + r
        let al: Vec<ValueId> = self.abuf.at(&tix).read(b, |b| {
            let lo = b.bind_i32(ev(w) * 64 + ev(g) * 32 + self.abuf.base as i32);
            let hi = b.bind_i32(ev(lo) + 16);
            let a8 = b.load_lds_b128x2(lo, hi);
            (0..8u8)
                .map(|r| {
                    let v = b.frag_slice(Opnd::Val(a8), r, 1);
                    f32_of(b, v)
                })
                .collect()
        });
        // ---- O = alpha O + P V; wave w owns dims 128 w .. ; the V fragment by ds_load_tr8_b64 ----
        // fetch lane f reads 8 dims at token 8 (f >> 4) + (f & 3) + 4 ((f >> 3) & 1) of a 16-token chunk, dims + 8 ((f >> 2) & 1)
        let vrow = b.bind_i32(ev(sb) + ev(self.vofs));
        let vd = self.vdelta.clone();
        let vars = self.vars.clone();
        kv.read(b, |b| {
            // all eight V fragments first, then the rescale + WMMA per tile
            let fvs: Vec<ValueId> = (0..8)
                .map(|nt| {
                    let a: Vec<ValueId> = (0..4).map(|jj| b.bind_i32(ev(vrow) + vd[8 * jj as usize + nt as usize])).collect();
                    b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3])
                })
                .collect();
            for nt in 0..8 {
                let o = vars[2 + nt];
                // the rescale in place (element r of the accumulator times alpha of its row) and the WMMA accumulating into the
                // variable itself (D = C = O): no rebuilt fragment, no loop-carried copy
                for r in 0..8u8 {
                    b.var_scale_lane(o, r, al[r as usize]);
                }
                let fv = fvs[nt];
                let d = b.wmma(WmmaKind::Fp8_16x16x64, pa, fv, Some(Opnd::Var(o)));
                b.set_var(o, Opnd::Val(d));
            }
        });
        b.set_var(vm, Opnd::Val(m_new));
        b.set_var(vl, Opnd::Val(lsum));
    }
}
