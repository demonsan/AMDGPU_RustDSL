//! The GEMM main-loop body (`PipeBody` of the register-staged cadence): scale panel, MXFP4 shuffled scales, operand reads and MMAs.

use super::*;

/// The scale panel: every K group's scales of the CTA in LDS, written once in the prologue (`prologue_stores`).
pub(crate) struct Panel {
    pub(crate) sh: Shared,
    /// LDS offset of the B part inside the panel
    pub(crate) b_base: i64,
    /// A part transposed: [kb][wm][lane % 16][m block i] (8 bytes per lane: 2 dwords, MFMA byte selects); this lane's offset
    pub(crate) tr: ValueId,
    /// B part expanded to dwords [wn][PANEL_KB] (byte 0 = the scale): this wave's dword offset (wn 4 PANEL_KB)
    pub(crate) b_dw: ValueId,
    /// the VGPR scale path emitted in `prologue_stores`
    pub(crate) sfa: SfaJob,
}

/// The VGPR scale path: A scales by 4 passes of buffer_load_dwordx4 (lane = 16 M rows of one K group), a 4x4 byte transpose per dword across each
/// lane quad (dpp quad_perm [1,0,3,2] + v_perm, dpp quad_perm [2,3,0,1] + v_perm) straight into the transposed panel layout, one ds_write per dword;
/// B scales by one dword load per thread. Layout proof: `sfa_transpose_proof`.
pub(crate) struct SfaJob {
    pub(crate) voff: ValueId,
    pub(crate) pso: Vec<ValueId>,
    pub(crate) sel1: ValueId,
    pub(crate) sel2: ValueId,
    pub(crate) dst: ValueId,
    pub(crate) bvo: ValueId,
    pub(crate) bsh: ValueId,
    pub(crate) bdst: ValueId,
}

/// Build-time proof of the VGPR scale path: simulates every lane of every pass byte by byte (the global load, both dpp/perm steps) and checks that
/// the byte written at panel offset 256 kb + 128 wm + 8 r + i is sfa[kb][128 wm + 16 i + r] (the transposed layout `scale_issue` reads).
/// Public wrapper for the VGPR/DPP scale-panel proof (item E1 calls it explicitly; the v4 builder
/// calls it build-time through `sfa_transpose_proof`).
pub fn verify_sfa_transpose() { sfa_transpose_proof() }

pub(crate) fn sfa_transpose_proof() {
    // a byte is identified by (kb, m)
    let perm = |s0: [(i64, i64); 4], s1: [(i64, i64); 4], sel: u32| -> [(i64, i64); 4] {
        let mut o = [(0, 0); 4];
        for (k, ok) in o.iter_mut().enumerate() {
            let sb = (sel >> (8 * k)) & 0xff;
            *ok = if sb < 4 { s1[sb as usize] } else { s0[(sb - 4) as usize] };
        }
        o
    };
    let mut seen = std::collections::HashMap::new();
    for p in 0..4i64 {
        for d in 0..4i64 {
            // x of every thread (256 threads, one pass): bytes of rows 16 c + 4 d + k of K group g / 16
            let x: Vec<[(i64, i64); 4]> = (0..256i64).map(|tid| { let g = tid + 256 * p; let (kb, c) = (g / 16, g % 16); [0, 1, 2, 3].map(|k| (kb, 16 * c + 4 * d + k)) }).collect();
            let qp = |v: &Vec<[(i64, i64); 4]>, pm: [usize; 4]| -> Vec<[(i64, i64); 4]> { (0..256usize).map(|l| v[4 * (l / 4) + pm[l % 4]]).collect() };
            let y = qp(&x, [1, 0, 3, 2]);
            let n1: Vec<[(i64, i64); 4]> = (0..256usize).map(|l| perm(y[l], x[l], if l % 2 == 0 { 0x0602_0400 } else { 0x0307_0105 })).collect();
            let z = qp(&n1, [2, 3, 0, 1]);
            for tid in 0..256i64 {
                let l = tid as usize;
                let out = perm(z[l], n1[l], if l % 4 < 2 { 0x0504_0100 } else { 0x0302_0706 });
                let g = tid + 256 * p;
                let (kb, c, j) = (g / 16, g % 16, tid % 4);
                let addr = 256 * kb + 128 * (c / 8) + 8 * (4 * d + j) + 4 * ((c / 4) % 2);
                for (k, byte) in out.iter().enumerate() {
                    let a = addr + k as i64;
                    let (pk, r, i, wm) = (a / 256, (a % 128) / 8, a % 8, (a % 256) / 128);
                    assert_eq!(*byte, (pk, 128 * wm + 16 * i + r), "sfa transpose: tid {tid} pass {p} dword {d} byte {k}");
                    seen.insert(a, ());
                }
            }
        }
    }
    assert_eq!(seen.len(), 64 * 256, "sfa transpose: every panel byte written once");
}

pub(crate) struct Body {
    pub(crate) cfg: Cfg,
    /// K tiles of this CTA's loop (K / 128, or KS with split-K)
    pub(crate) kb_v: ValueId,
    /// the E8M0 scale panel (Scales::Mx128)
    pub(crate) panel: Option<Panel>,
    pub(crate) mma: TiledMma,
    pub(crate) a_s: Shared,
    pub(crate) b_s: Shared,
    pub(crate) a_dma: Dma,
    pub(crate) b_dma: Dma,
    pub(crate) a_org: ValueId,
    pub(crate) b_org: ValueId,
    pub(crate) a_wave: ValueId,
    pub(crate) b_wave: ValueId,
    pub(crate) rsrc_a: ValueId,
    pub(crate) rsrc_b: ValueId,
    pub(crate) rsrc_sfa: ValueId,
    pub(crate) rsrc_sfb: ValueId,
    pub(crate) a_rd: LdsPart,
    pub(crate) b_rd: LdsPart,
    /// mixed precision: the A images 1.. (the image-0 partition moved by k `a_hi`) and the image stride in the slot
    pub(crate) a_rd_x: Vec<LdsPart>,
    pub(crate) a_hi: i32,
    /// A16W4: the lane's B-scale byte shift (8 (g / 2), g = lane / 16)
    pub(crate) w4_shift: Option<ValueId>,
    pub(crate) zero4: ValueId,
    pub(crate) acc: Vec<VarId>,
    /// MMAs of the register tile already issued by `pre_sync` / `post_wait`
    pub(crate) npre_done: usize,
    /// timing stamps (GEMM_STAMP=ends|tiles): debug buffer, this wave's record offset, per-tile stamps
    pub(crate) stamp: Option<(ValueId, ValueId, bool)>,
    /// `Scales::RowCol`: the unit E8M0 scale words as VGPR values (lane / 64 = 0 keeps them vector: a wave-uniform constant would be held in an
    /// SGPR and copied to a VGPR before every scaled MFMA, with a VALU -> MFMA wait)
    pub(crate) unit: Option<(ValueId, ValueId)>,
    /// `Scales::Mx32`: the per-tile scale loads
    pub(crate) mx32: Option<Mx32>,
    /// `Scales::Block128`: the per-tile fp32 scale loads (voffsets; `shift` unused)
    pub(crate) bs128: Option<Mx32>,
    /// implicit-GEMM convolution: the per-tile A addressing
    pub(crate) conv: Option<ConvRt>,
    /// split-K: this split's first K tile
    pub(crate) zb: Option<ValueId>,
    /// `Elem::Fp8Bs`: the block products of the tile waiting for their scaled accumulation (`bs_fma`)
    pub(crate) pending: std::cell::RefCell<Vec<(VarId, ValueId, VarId, VarId)>>,
    /// register-staged tile: initial values (tile 0) and the loop variables: B fragments, A fragments, A scales, B scale
    pub(crate) staged: Option<(Vec<ValueId>, Vec<ValueId>, Vec<ValueId>, Vec<ValueId>)>,
    pub(crate) rv: Option<(Vec<VarId>, Vec<VarId>, Vec<VarId>, Vec<VarId>)>,
    /// shuffled 1 x 32 scales: the MFMA scale-byte offset of the tile being computed (0 even tile, 2 odd tile of a pair: the pair's dword is
    /// loaded once, for the even tile, and kept)
    pub(crate) odd_by: std::cell::Cell<u8>,
    /// the tail's one-time shift of the pair dwords has been emitted (per tail branch: reset by `iter` hooks below)
    pub(crate) tail_shifted: std::cell::Cell<bool>,
    /// depth3 + Block128: the per-tile fp32 scales go through the DMA stage into the SC ring (two dword DMAs per wave: A rows of the CTA,
    /// B block; only wave 0's are real, the others land in their own 512-byte part of the slot): (ring, A voffset, B voffset)
    pub(crate) sc_dma: Option<(Shared, ValueId, ValueId, ValueId, ValueId)>,
}

/// The 1 x 32 E8M0 scale loads of `Scales::Mx32`: per m block (n block) the lane's loop-invariant voffset of its row's 8 groups of the tile,
/// and the per-lane v_perm selector (`shift`) that picks the bytes of K groups g and 4 + g (g = lane / 16).
pub(crate) struct Mx32 {
    pub(crate) voff_a: Vec<ValueId>,
    pub(crate) voff_b: Vec<ValueId>,
    pub(crate) shift: ValueId,
    /// `scale_shuf`: [rows / 16][KT / 2][lane][4 bytes] with lane 16 g + r holding the bytes of K groups g, 4 + g of the even
    /// tile, then of the odd tile (one coalesced dword per lane, block and tile pair); the bit shift of the last `load_scales` (0 / 16)
    pub(crate) shuf: Option<std::cell::RefCell<Option<ValueId>>>,
}

impl Body {
    /// MMAs `from..to` (in row-major order over the first half of the B fragments) of the register tile.
    fn early_mmas(&mut self, b: &mut FnBuilder, from: usize, to: usize) {
        let (bv, av, sav, sbv) = self.rv.clone().expect("state bound");
        let nn = bv.len();
        let half = nn / 2;
        let mut n = 0;
        let mut blocks = Vec::new();
        let nma = self.n_a();
        'o: for i in 0..nma {
            for j in 0..half {
                if n >= to {
                    break 'o;
                }
                if n >= from {
                    let (w, by) = self.sa_sel(i);
                    blocks.push((self.acc[nn * i + j], av[i], bv[j], sav[w], sbv[self.sb_sel(j)], by));
                }
                n += 1;
            }
        }
        let p = mma_blocks(b, &self.mma, self.cfg.elem, self.unit.map(|u| u.0), &blocks, nma as u32, self.w4_shift);
        self.pending.borrow_mut().extend(p);
        self.npre_done = n;
    }
    /// The scale words of tile `kb` from the transposed panel: the 8 A-scale bytes of this lane (one ds_read_b64; the MMA selects byte i % 4 of
    /// word i / 4: `sa_sel`) and the B scale (the dword-expanded panel: byte 0, no shift).
    fn load_scales(&self, b: &mut FnBuilder, kb: I32, slot: &SlotIdx) -> (Vec<ValueId>, Vec<ValueId>) {
        if let Some((sc, _, _, _, rd)) = &self.sc_dma {
            // depth3 + Block128: the tile's scales from its SC slot (A row of the lane per m block, the B block's float at +256)
            let nm = self.n_a();
            let p = sc.at(slot);
            let base = { let o = p.off_v(b); b.bind_i32(o + sc.base as i32) };
            return p.read(b, |b| {
                let sa: Vec<ValueId> = (0..nm).map(|i| { let a = b.bind_i32(ev(base) + ev(*rd) + 64 * i as i32); let v = b.load_lds_b32(a); b.bitcast_f32(v) }).collect();
                let sb = { let a = b.bind_i32(ev(base) + 256); let v = b.load_lds_b32(a); b.bitcast_f32(v) };
                (sa, vec![sb])
            });
        }
        if self.cfg.scales == Scales::RowCol || self.cfg.scales == Scales::None {
            // unit E8M0 scales (127 in every byte): the fp32 row / column scales are applied in the epilogue
            let (u, v) = self.unit.expect("unit scales");
            return (vec![u, u], vec![v]);
        }
        // K group of a real tile (t + 1 is read only when it exists): the clamp is a no-op that bounds it for the verifier's loop-exit-blind intervals
        let kb = b.bind_i32(kb.lt(ev(self.kb_v)).select_i32(kb.clone(), ev(self.kb_v) - 1));
        if let Some(bs) = &self.bs128 {
            // fp32 [rows][KT]: tile kb (+ the split's first tile) at byte 4 kb (SGPR soffset), per m block one float; the B block float (raw:
            // installed as words)
            let kb = match self.zb {
                Some(zb) => b.bind_i32(ev(kb) + ev(zb)),
                None => kb,
            };
            let so = { let v = b.bind_i32(ev(kb) << 2); b.read_first_lane(v) };
            let sa = bs.voff_a.iter().map(|v| b.buffer_load_f32(self.rsrc_sfa, *v, so, 0)).collect();
            let sb = bs.voff_b.iter().map(|v| b.buffer_load_f32(self.rsrc_sfb, *v, so, 0)).collect();
            return (sa, sb);
        }
        if let (Some(mx), Some(sh)) = (&self.mx32, self.mx32.as_ref().and_then(|m| m.shuf.as_ref())) {
            // shuffled: the tile pair kb / 2 (256 bytes per block), the tile's half selected by a shift at the install
            let so = { let v = b.bind_i32(ev(kb).lshr(1) << 8); b.read_first_lane(v) };
            // (the raw pair dword is installed: the tile's bytes are picked by the MFMA byte select, `odd_by`)
            let s = b.c_i32(0);
            *sh.borrow_mut() = Some(s);
            let ld = |b: &mut FnBuilder, rs: ValueId, vo: &[ValueId]| -> Vec<ValueId> { vo.iter().map(|v| b.buffer_load_f32(rs, *v, so, 0)).collect() };
            return (ld(b, self.rsrc_sfa, &mx.voff_a), ld(b, self.rsrc_sfb, &mx.voff_b));
        }
        if let (Some(mx), Elem::Bf16W4) = (&self.mx32, self.cfg.elem) {
            // A16W4: no A scales (zeros, unused); B [N][K / 32]: the 8 group bytes of the tile per n block, as two dwords (words 2 j, 2 j + 1)
            let so = { let v = b.bind_i32(ev(kb) << 3); b.read_first_lane(v) };
            let sa = (0..mx.voff_a.len()).map(|_| b.c_i32(0)).collect();
            // the tile's 8 group bytes of the block's row as two words (groups 0..3, 4..7)
            let mut sb = Vec::new();
            for v in &mx.voff_b {
                let f = b.buffer_load_b64(self.rsrc_sfb, *v, so, 0);
                sb.push(b.frag2_lane(f, 0));
                sb.push(b.frag2_lane(f, 1));
            }
            return (sa, sb);
        }
        if let Some(mx) = &self.mx32 {
            // 1 x 32 E8M0 [rows][K / 32] bytes: the 8 groups of the tile (one 8-byte load per m / n block); the lane group g = lane / 16 of
            // MFMA h (fragment bytes 64 h + 16 g ..) covers K group 4 h + g: v_perm puts byte g of the low word in byte 0 and byte g of the high
            // word in byte 1 (byte h = the scale of MFMA h)
            let so = { let v = b.bind_i32(ev(kb) << 3); b.read_first_lane(v) };
            // the raw 8-byte loads: the byte selection (`scale_word`) is applied where the words are installed, so the loads stay in flight
            // behind the MMAs instead of being waited for at once
            let ld = |b: &mut FnBuilder, rs: ValueId, vo: &[ValueId]| -> Vec<ValueId> { vo.iter().map(|v| b.buffer_load_b64(rs, *v, so, 0)).collect() };
            let sa = ld(b, self.rsrc_sfa, &mx.voff_a);
            let sb = ld(b, self.rsrc_sfb, &mx.voff_b);
            return (sa, sb);
        }
        let kb = ev(kb);
        let p = self.panel.as_ref().expect("scale panel");
        let (base, pr) = (p.sh.base, p.sh.at(&SlotIdx::new(0, 0)));
        let sab = b.bind_i32((kb.clone() << 8) + (base as i32) + ev(p.tr));
        let sbb = b.bind_i32((kb << 2) + ev(p.b_dw) + (base + p.b_base) as i32);
        pr.read(b, |b| {
            let w2 = b.load_lds_b64(sab);
            let wa = vec![b.frag2_lane(w2, 0), b.frag2_lane(w2, 1)];
            (wa, vec![b.load_lds_b32(sbb)])
        })
    }
    /// The scale word of a `load_scales` value: `Scales::Mx32` selects the lane's two group bytes of the 8-byte load (v_perm); identity otherwise.
    fn scale_word(&self, b: &mut FnBuilder, v: ValueId) -> ValueId {
        if self.w4_shift.is_some() {
            // A16W4: the words are selected at the load
            return b.bitcast_i32(v);
        }
        match &self.mx32 {
            Some(mx) if mx.shuf.is_some() => {
                let s = mx.shuf.as_ref().unwrap().borrow().expect("shift of the last scale load");
                let w = b.bitcast_i32(v);
                b.bind_i32(ev(w).lshr(ev(s)))
            }
            Some(mx) => {
                let lo = b.frag2_lane(v, 0);
                let hi = b.frag2_lane(v, 1);
                let (lo, hi) = (b.bitcast_i32(lo), b.bitcast_i32(hi));
                b.perm_b32(hi, lo, mx.shift)
            }
            None if self.bs128.is_some() => b.bitcast_i32(v),
            None => v,
        }
    }
    /// The (word, byte) of the A scale of m block i in the `load_scales` words.
    fn sa_sel(&self, i: usize) -> (usize, u8) {
        if self.mx32.as_ref().is_some_and(|m| m.shuf.is_some()) {
            return (i, self.odd_by.get());
        }
        if self.mx32.is_some() || self.bs128.is_some() {
            return (i, 0);
        }
        (i / 4, (i % 4) as u8)
    }
    /// The B scale word of n block j.
    fn sb_sel(&self, j: usize) -> usize {
        if self.cfg.elem == Elem::Bf16W4 {
            2 * j
        } else if self.mx32.is_some() {
            j
        } else if self.bs128.as_ref().is_some_and(|bs| bs.voff_b.len() == 2) {
            // MoE stage 1 blockscale: gate / up blocks alternate
            j % 2
        } else {
            0
        }
    }
    /// An operand fragment read into AGPRs.
    /// A fragments (m blocks) of the wave (A8W4 keeps lo + hi fragments: the A variables are twice as many)
    fn n_a(&self) -> usize {
        (self.cfg.wave_m() / 16) as usize
    }
    fn frag(&self, b: &mut FnBuilder, part: &amdgpu_tile::LdsSlotPart, o: usize) -> ValueId {
        let v = part.read(b, o);
        b.agpr(v)
    }
    /// The whole tile `t` into registers (prologue: tile 0).
    fn load_tile(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let (nm, nn) = ((self.cfg.wave_m() / 16) as usize, (self.cfg.wave_n() / 16) as usize);
        let (sa, sb) = self.load_scales(b, ox(t.it), &t.cur);
        let sa: Vec<ValueId> = sa.into_iter().map(|v| self.scale_word(b, v)).collect();
        let sb: Vec<ValueId> = sb.into_iter().map(|v| self.scale_word(b, v)).collect();
        let ar = self.a_rd.at(b, &self.a_s.at(&t.cur));
        let br = self.b_rd.at(b, &self.b_s.at(&t.cur));
        let bf: Vec<ValueId> = (0..nn).map(|j| self.frag(b, &br, j)).collect();
        let mut af: Vec<ValueId> = (0..nm).map(|i| self.frag(b, &ar, i)).collect();
        // mixed precision: the fragments of A image k right after those of image k - 1 (mma_blocks finds image k at lo + k nm)
        for part in &self.a_rd_x {
            let arh = part.at(b, &self.a_s.at(&t.cur));
            let hi: Vec<ValueId> = (0..nm).map(|i| self.frag(b, &arh, i)).collect();
            af.extend(hi);
        }
        self.staged = Some((bf, af, sa, sb));
    }
    /// The MMAs of the register tile; when `t.staged`, each fragment is reloaded from tile t+1 right after its last use: first half of the B
    /// fragments with every A fragment, then the B halves are reloaded; second half, after which every A fragment is reloaded; the scales of t+1
    /// are read before the MMAs.
    fn compute_tile(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let roles = vec!["all".to_string()];
        // pipelined loop: the linear barrier count is not a stable property (StageContract::ANY_PHASE)
        let phase = amdgpu_dsl::StageContract::ANY_PHASE;
        let me: &Body = self;
        // the slot offsets of t + 1 are bound before the region (they are shared with the staging instance through the SlotIdx cache)
        if t.staged {
            let _ = me.a_s.at(&t.next).off_v(b);
            let _ = me.b_s.at(&t.next).off_v(b);
        }
        me.tile_stage(b, t, phase, roles).emit(b);
    }
    /// Stage `mma.blockscale_tile`: the MMAs of the register tile; when `t.staged`, each fragment is reloaded from tile t+1 right after its last
    /// use (first half of the B fragments with every A fragment, then the B halves are reloaded; second half, after which every A fragment is
    /// reloaded); the scales of t+1 are read before the MMAs and installed at the end (partial lgkm waits). Captured units.
    fn tile_stage(&self, b: &mut FnBuilder, t: &TileIt, phase: u32, roles: Vec<String>) -> amdgpu_tile::stage::StageInst<'static> {
        use amdgpu_tile::weave::Units;
        let (bv, av, sav, sbv) = self.rv.clone().expect("state bound");
        let (nm, nn) = (self.n_a(), bv.len());
        let half = nn / 2;
        let mut u = Units::new();
        let (_, (next, early)) = u.capture(b, "tile.setup", |b| {
            let next = t.staged.then(|| (self.a_rd.at(b, &self.a_s.at(&t.next)), self.b_rd.at(b, &self.b_s.at(&t.next))));
            // mixed precision: the other A images of t + 1
            let next_hi: Vec<amdgpu_tile::LdsSlotPart> = if t.staged { self.a_rd_x.iter().map(|p| p.at(b, &self.a_s.at(&t.next))).collect() } else { Vec::new() };
            // the scales of t+1 are read FIRST (before the fragment reloads queue up behind them) and installed at the end (partial lgkm waits)
            // shuffled scales: t + 1 odd shares t's pair dword (already installed): no load
            let same_pair = self.mx32.as_ref().is_some_and(|m| m.shuf.is_some()) && Self::slot_of(t) == 0;
            let early = (t.staged && !same_pair).then(|| self.load_scales(b, ox(t.it) + 1, &t.next));
            ((next, next_hi), early)
        });
        let (next, next_hi) = next;
        // MMAs of the register tile with the in-place reloads of t+1 (into `u`)
        let body = |b: &mut FnBuilder, u: &mut Units| {
            // bf16 (two dependent K-half MFMAs per block): the blocks of a PAIR of rows form one unit (first halves of all, then second
            // halves), so a block's second MFMA issues 8 MFMAs after its first (4 per row left 120 s_nops of MFMA dependency stalls)
            let rows_per_unit = if self.cfg.elem == Elem::Bf16 && nm % 2 == 0 { 2 } else if self.cfg.elem == Elem::Bf16W4 { nm } else { 1 };
            for (lo, hi) in [(0, half), (half, nn)] {
                for i0 in (0..nm).step_by(rows_per_unit) {
                    let rows = i0..i0 + rows_per_unit;
                    let mut blocks = Vec::new();
                    for i in rows.clone() {
                        let todo: Vec<usize> = (lo..hi).filter(|&j| !(lo == 0 && i * half + j < self.npre_done)).collect(); // the rest were issued before the barrier
                        let (w, by) = self.sa_sel(i);
                        if self.cfg.elem == Elem::Fp8 {  // one unit per block
                            for &j in &todo {
                                u.capture(b, format!("mma.i{i}j{j}"), |b| mma_var(b, &self.mma, self.cfg.elem, self.unit.map(|u| u.0), self.acc[nn * i + j], av[i], bv[j], sav[w], sbv[self.sb_sel(j)], by));
                            }
                        } else {
                            blocks.extend(todo.iter().map(|&j| (self.acc[nn * i + j], av[i], bv[j], sav[w], sbv[self.sb_sel(j)], by)));
                        }
                    }
                    if !blocks.is_empty() {
                        u.capture(b, format!("mma.i{i0}+{}h{lo}", rows_per_unit), |b| {
                            let p = mma_blocks(b, &self.mma, self.cfg.elem, self.unit.map(|u| u.0), &blocks, nm as u32, self.w4_shift);
                            self.pending.borrow_mut().extend(p);
                        });
                    }
                    for i in rows {
                        if let (true, Some((ar, _))) = (lo == half, &next) {
                            // in-place reload: the ds_reads write the variable's own AGPRs (coalesced set_var)
                            u.capture(b, format!("reload.a{i}"), |b| {
                                let v = ar.read(b, i);
                                b.set_var(av[i], v);
                            });
                            for (k, ah) in next_hi.iter().enumerate() {
                                u.capture(b, format!("reload.ah{k}.{i}"), |b| {
                                    let v = ah.read(b, i);
                                    b.set_var(av[(k + 1) * nm + i], v);
                                });
                            }
                        }
                    }
                }
                if let Some((_, br)) = &next {
                    for j in lo..hi {
                        u.capture(b, format!("reload.b{j}"), |b| {
                            let v = br.read(b, j);
                            b.set_var(bv[j], v);
                        });
                    }
                }
            }
        };
        // E3 proof hook for the reg-staged compute path: a real IR branch drops one tile's MMAs, reproducing the historical "tile 1 swallowed"
        // regression in emitted kernel code (tests/functional_gemm_ir.rs); the branch region is one captured unit
        match std::env::var("GEMM_PROOF_DROP_KB").ok().and_then(|v| v.parse::<i32>().ok()) {
            Some(kb) => {
                u.capture(b, "tile.proof_drop", |b| {
                    let keep = b.bind_cond(ox(t.it).ne(kb));
                    b.begin_if(keep);
                    let mut inner = Units::new();
                    body(b, &mut inner);
                    inner.emit_all(b);
                    b.end_if();
                });
            }
            None => body(b, &mut u),
        }
        // the deferred block-scale updates of this tile, before the scales of t + 1 replace the scale variables
        let pend: Vec<_> = self.pending.borrow_mut().drain(..).collect();
        if !pend.is_empty() {
            u.capture(b, "bs.fma", |b| bs_fma(b, &pend));
        }
        if let Some((sa, sb)) = early {
            u.capture(b, "scales.install", |b| {
                for (i, v) in sa.iter().enumerate() {
                    let w = self.scale_word(b, *v);
                    b.set_var(sav[i], w);
                }
                for (j, v) in sb.iter().enumerate() {
                    let w = self.scale_word(b, *v);
                    b.set_var(sbv[j], w);
                }
            });
        }
        let contract = amdgpu_dsl::StageContract { barrier_phase: phase, roles, ..Default::default() };
        amdgpu_tile::stage::StageInst::new("mma.blockscale_tile", amdgpu_tile::stage::Prim::A, "gemm.tile", contract, u)
            .param("atom", "mfma_scale_16x16x128_f8")
            .param("wave_tile", format!("{}x{}", 16 * nm, 16 * nn))
            .param("staged", t.staged)
    }    /// The DMA of one tile (A issues, then B issues): soffset = the wave-uniform tile origin, M0 = the first 1 KB block of each group of 4 issues.
    fn stage_now(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotIdx) {
        let phase = amdgpu_dsl::StageContract::ANY_PHASE;
        // the slot offsets are bound before the region (shared with the readers through the SlotIdx cache)
        let _ = self.a_s.at(slot).off_v(b);
        let _ = self.b_s.at(slot).off_v(b);
        self.ab_stage(b, key0, slot, phase).emit(b);
    }
    /// Stage `move.ab_stage`: the DMA of one tile (A issues, then B issues); soffset = the wave-uniform tile origin, M0 = the first 1 KB block of
    /// each group of 4 issues; one captured unit per DMA instruction (each under the slot's write marker).
    fn ab_stage(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotIdx, phase: u32) -> amdgpu_tile::stage::StageInst<'static> {
        let mut u = amdgpu_tile::weave::Units::new();
        let (_, orgs) = u.capture(b, "ab.setup", |b| {
            // A advances by key0 bytes, the preshuffled B by (key0 / 16) 256 = 16 key0; the tile origins are wave uniform: SGPR soffsets
            // mixed precision: A advances (images) key0 bytes (128 A bytes per image per 128-byte fp4 B tile)
            let nimg = self.cfg.elem.a_images();
            let ka = if nimg > 1 {
                let v = b.bind_i32((key0.clone() << (nimg.trailing_zeros() as i32)) + ev(self.a_org)); b.read_first_lane(v)
            } else {
                let v = b.bind_i32(key0.clone() + ev(self.a_org)); b.read_first_lane(v)
            };
            let kb = { let v = b.bind_i32((key0.clone() << 4) + ev(self.b_org)); b.read_first_lane(v) };
            // (mixed precision only: the other images' origins; other kernels keep their exact code)
            let mut o = vec![ka, kb];
            for k in 1..nimg {
                let v = b.bind_i32(ev(ka) + 128 * k as i32);
                o.push(b.read_first_lane(v));
            }
            o
        });
        // implicit-GEMM conv: this tile's channel block and tap (wave uniform), then per A issue the input pixel's voffset (or out of range)
        let conv_vo: Option<Vec<ValueId>> = self.conv.as_ref().map(|cv| {
            let (_, vo) = u.capture(b, "ab.conv", |b| {
                let cc = cv.cfg;
                let trs = cc.t * cc.r * cc.s;
                let rs = cc.r * cc.s;
                // t / trs, rs / s by multiply-shift (exact for t < 2^12 = 4096 K tiles, checked)
                let magic = |d: i32| -> i64 { ((1i64 << 24) + d as i64 - 1) / d as i64 };
                for d in [trs, rs, cc.s] {
                    let m = magic(d);
                    for x in 0..(1i64 << 12) {
                        assert_eq!((x * m) >> 24, x / d as i64, "conv: multiply-shift by {d}");
                    }
                }
                let t = b.bind_i32(key0.clone().lshr(7));
                let cb = b.bind_i32((ev(t) * magic(trs) as i32).lshr(24));
                let tap = b.bind_i32(ev(t) - ev(cb) * trs);
                let dz = b.bind_i32((ev(tap) * magic(rs) as i32).lshr(24));
                let tr = b.bind_i32(ev(tap) - ev(dz) * rs);
                let rr = b.bind_i32((ev(tr) * magic(cc.s) as i32).lshr(24));
                let ss = b.bind_i32(ev(tr) - ev(rr) * cc.s);
                let toff = b.bind_i32((((ev(dz) * cc.dil) * ev(cv.h) + ev(rr) * cc.dil) * ev(cv.w) + ev(ss) * cc.dil) * ev(cv.c) * 2 + (ev(cb) << 7));
                let big = b.c_i32(0x7fff_fff0);
                cv.rows
                    .iter()
                    .map(|&(base, id0, ih0, iw0)| {
                        let (zi, yi, xi) = (b.bind_i32(ev(id0) + ev(dz) * cc.dil), b.bind_i32(ev(ih0) + ev(rr) * cc.dil), b.bind_i32(ev(iw0) + ev(ss) * cc.dil));
                        let ok = b.bind_i32(
                            ev(zi).ge(0).select_i32(1, 0) & ev(zi).lt(ev(cv.di)).select_i32(1, 0) & ev(yi).ge(0).select_i32(1, 0) & ev(yi).lt(ev(cv.h)).select_i32(1, 0)
                                & ev(xi).ge(0).select_i32(1, 0) & ev(xi).lt(ev(cv.w)).select_i32(1, 0),
                        );
                        let ok = b.bind_cond(ev(ok).eq(1));
                        let v = b.bind_i32(ev(base) + ev(toff));
                        b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(big) })
                    })
                    .collect::<Vec<_>>()
            });
            vo
        });
        // (A8W4: a third pass, the hi A image: LDS + a_hi, global + 128 bytes)
        let mut passes = vec![(&self.a_s, &self.a_dma, self.rsrc_a, self.a_wave, 0usize, 0i32), (&self.b_s, &self.b_dma, self.rsrc_b, self.b_wave, 1, 0)];
        for k in 1..self.cfg.elem.a_images() {
            passes.push((&self.a_s, &self.a_dma, self.rsrc_a, self.a_wave, k + 1, k as i32 * self.a_hi));
        }
        for (bi, (sh, dma, rsrc, wave, oi, lds_off)) in passes.into_iter().enumerate() {
            let uname = if bi < 2 { sh.name.to_string() } else if bi == 2 { format!("{}h", sh.name) } else { format!("{}h{bi}", sh.name) };
            let (_, mine) = u.capture(b, format!("ab.{uname}.setup"), |b| {
                let o = sh.at(slot).off_v(b);
                let sbase = if lds_off == 0 { b.bind_i32(o + sh.base as i32 + ev(wave)) } else { b.bind_i32(o + (sh.base as i32 + lds_off) + ev(wave)) };
                let mut m0s: Vec<(i32, ValueId)> = Vec::new();
                let mut mine: Vec<(ValueId, i32, ValueId)> = Vec::new();
                for (g, lds) in dma.issues.iter() {
                    let grp = dma.group(*lds);
                    if !m0s.iter().any(|x| x.0 == grp) {
                        let v = b.bind_i32(ev(sbase) + dma.act(*lds).0);
                        let v = b.read_first_lane(v);
                        m0s.push((grp, v));
                    }
                    mine.push((*g, *lds, m0s.iter().find(|x| x.0 == grp).unwrap().1));
                }
                mine
            });
            for (q, &(g, lds, m0)) in mine.iter().enumerate() {
                // buffer_load_dwordx4 ... lds: voffset = the lane's loop-invariant offset, soffset = the tile origin (wave uniform), M0 = the first 1 KB
                // block of each group of 4 issues, instruction offset 1024 (q % 4) moves both the LDS and the global address
                let (g, so) = match (&conv_vo, bi) {
                    (Some(vo), 0) => (vo[q], self.conv.as_ref().map(|c| c.zero).unwrap()),
                    _ => (g, orgs[oi]),
                };
                u.capture(b, format!("ab.{uname}.{q}"), |b| sh.at(slot).write(b, |b| b.load_buffer_lds(rsrc, g, so, m0, dma.act(lds).1, 4)));
            }
        }
        if let Some((sc, va, vb, wofs, _)) = &self.sc_dma {
            // the tile's scales: soffset = 4 kb (+ the split's first tile), M0 = the wave's 512-byte part of the slot (A at +0, B at +256)
            let (_, (so, m0a, m0b)) = u.capture(b, "ab.SC.setup", |b| {
                let kb = key0.clone().lshr(7);
                let kb = match self.zb { Some(zb) => kb + ev(zb), None => kb };
                let so = { let v = b.bind_i32(kb << 2); b.read_first_lane(v) };
                let o = sc.at(slot).off_v(b);
                let m0a = { let v = b.bind_i32(o + sc.base as i32 + ev(*wofs)); b.read_first_lane(v) };
                let m0b = { let v = b.bind_i32(ev(m0a) + 256); b.read_first_lane(v) };
                (so, m0a, m0b)
            });
            u.capture(b, "ab.SC.a", |b| sc.at(slot).write(b, |b| b.load_buffer_lds(self.rsrc_sfa, *va, so, m0a, 0, 1)));
            u.capture(b, "ab.SC.b", |b| sc.at(slot).write(b, |b| b.load_buffer_lds(self.rsrc_sfb, *vb, so, m0b, 0, 1)));
        }
        let contract = amdgpu_dsl::StageContract { barrier_phase: phase, roles: vec!["all".to_string()], ..Default::default() };
        amdgpu_tile::stage::StageInst::new("move.ab_stage", amdgpu_tile::stage::Prim::M, "gemm.ab_stage", contract, u).param("slots", 2)
    }}

impl PipeBody for Body {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &I32, slot: &SlotIdx) {
        self.stage_now(b, key0, slot);
    }
    fn prologue_stores(&mut self, b: &mut FnBuilder, _role: Role) {
        if self.cfg.scales == Scales::Mx128 {
            let p = self.panel.as_ref().expect("scale panel");
            let sj = &p.sfa;
            let pr = p.sh.at(&SlotIdx::new(0, 0));
            // loads (A: 4 x 16 bytes, B: one dword), then the transposes and the panel writes; published by the pipeline's prologue barrier
            let fr: Vec<ValueId> = sj.pso.iter().map(|so| b.buffer_load_b128(self.rsrc_sfa, sj.voff, *so, 0)).collect();
            let bw = b.buffer_load_f32(self.rsrc_sfb, sj.bvo, 0i32, 0);
            let bwi = b.bitcast_i32(bw);
            let bval = b.bind_i32(ev(bwi).lshr(ev(sj.bsh)) & 255);
            pr.write(b, |b| {
                for (pi, f) in fr.iter().enumerate() {
                    for d in 0..4u8 {
                        let xl = b.frag4_lane(Opnd::Val(*f), d);
                        let x = b.bitcast_i32(xl);
                        let y = b.dpp_quad_perm(x, [1, 0, 3, 2]);
                        let y = b.bitcast_i32(y);
                        let n1 = b.perm_b32(y, x, sj.sel1);
                        let z = b.dpp_quad_perm(n1, [2, 3, 0, 1]);
                        let z = b.bitcast_i32(z);
                        let out = b.perm_b32(z, n1, sj.sel2);
                        let a = b.bind_i32(ev(sj.dst) + (4096 * pi as i32 + 32 * d as i32));
                        b.store_lds_b32(a, out);
                    }
                }
                b.store_lds_b32(sj.bdst, bval);
            });
        }
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        if let Some((buf, rec, _)) = self.stamp {
            let st = b.memtime_lo();
            b.stamp_store(buf, rec, 9, st); // slot 9: prologue wait + barrier passed (tiles 0/1 + transposed panel visible)
        }
        self.load_tile(b, t0);
        if let Some((buf, rec, _)) = self.stamp {
            let st = b.memtime_lo();
            b.stamp_store(buf, rec, 2, st); // slot 2: prologue done (tile 0 fragments requested), about to enter the tile loop
        }
    }
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let mut v: Vec<(DType, Opnd)> = (0..(self.cfg.wave_m() / 16) * (self.cfg.wave_n() / 16)).map(|_| (DType::Frag4, Opnd::Val(self.zero4))).collect();
        if let Some((bf, af, sa, sb)) = &self.staged {
            v.extend(bf.iter().chain(af.iter()).map(|x| (DType::Frag8, Opnd::Val(*x))));
            v.extend(sa.iter().map(|x| (DType::I32, Opnd::Val(*x))));
            v.extend(sb.iter().map(|x| (DType::I32, Opnd::Val(*x))));
        }
        v
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        let na = ((self.cfg.wave_m() / 16) * (self.cfg.wave_n() / 16)) as usize;
        self.acc = vars[..na].to_vec();
        if let Some((bf, af, sa, sb)) = self.staged.take() {
            let (nb, nf, ns, nsb) = (bf.len(), af.len(), sa.len(), sb.len());
            let r = &vars[na..];
            self.rv = Some((r[..nb].to_vec(), r[nb..nb + nf].to_vec(), r[nb + nf..nb + nf + ns].to_vec(), r[nb + nf + ns..nb + nf + ns + nsb].to_vec()));
            let nimg = self.cfg.elem.a_images();
            if nimg > 1 {
                let av = &r[nb..nb + nf];
                let nm = nf / nimg;
                assert!((1..nimg).all(|k| (0..nm).all(|i| av[k * nm + i].0 == av[i].0 + (k * nm) as u32)), "mixed precision: the A image variables are consecutive");
            }
        }
    }
    /// The next MMAs of the register tile between the waits and the barrier (the vendor issues 3 there): the barrier latency hides behind them.
    fn post_wait(&mut self, b: &mut FnBuilder, _role: Role, _t: &TileIt) {
        self.set_parity(b, _t);
        if self.cfg.mid_sync_mmas > 0 {
            let from = self.npre_done;
            self.early_mmas(b, from, from + self.cfg.mid_sync_mmas);
        }
    }
    /// The first MMAs of the register tile run while the wave waits for the DMA and the barrier (the vendor issues 4 there).
    fn pre_sync(&mut self, b: &mut FnBuilder, _role: Role, _t: &TileIt) {
        self.set_parity(b, _t);
        let npre = self.cfg.pre_sync_mmas;
        self.early_mmas(b, 0, npre);
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        if let Some((buf, rec, true)) = self.stamp {
            // slot 16 + tile: the wave reached this tile's matrix phase (after the tile's top wait + barrier and its DMA issue)
            let st = b.memtime_lo();
            let four = b.c_i32(4);
            let ti = b.mul_i32(t.it, four);
            let off = b.add_i32(rec, ti);
            b.stamp_store(buf, off, 16, st);
        }
        self.set_parity(b, t);
        self.compute_tile(b, t);
        self.tail_shifted.set(false);
    }
}

impl Body {
    /// The static ring slot of the tile (two tiles per pass: slot = tile parity); -1 in the peeled tail (runtime slot).
    fn slot_of(t: &TileIt) -> i32 {
        match t.cur.slot {
            Opnd::I32(s) => s,
            _ => -1,
        }
    }
    /// shuffled scales: the byte select of the tile; in the tail (runtime parity) the pair dwords are shifted down once instead (`shifted`)
    fn set_parity(&self, b: &mut FnBuilder, t: &TileIt) {
        if !self.mx32.as_ref().is_some_and(|m| m.shuf.is_some()) {
            return;
        }
        match Self::slot_of(t) {
            -1 => {
                self.odd_by.set(0);
                if !self.tail_shifted.get() {
                    self.tail_shifted.set(true);
                    let (_, _, sav, sbv) = self.rv.clone().expect("state bound");
                    let sh = b.bind_i32(amdgpu_dsl::ox(t.cur.slot.clone()) << 4);
                    for v in sav.iter().chain(sbv.iter()) {
                        let w = b.bind_i32(amdgpu_dsl::varx(*v).lshr(ev(sh)));
                        b.set_var(*v, w);
                    }
                }
            }
            s => self.odd_by.set(2 * (s & 1) as u8),
        }
    }
}
