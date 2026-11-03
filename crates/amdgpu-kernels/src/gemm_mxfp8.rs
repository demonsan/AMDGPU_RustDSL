//! MXFP8 blockscale GEMM on the route-B layers (P4; target = aiter PR #6696 4-wave pin-AGPR kernel, docs/mxfp8-gemm-4wave-analysis.md), same 96-byte ABI:
//!   C[m, n] (bf16) = sum_k A[m, k] 2^(sa[k/128][m] - 127) * B[n, k] 2^(sb[n/128][k/128] - 127)
//!   A [M, K] fp8 e4m3 row-major; B [N, K] fp8 preshuffled (aiter shuffle_weight 16x16); sa E8M0 [K/128][M]; sb E8M0 [N/128][K/128].
//!
//! v2 (VGPR-only until AGPR support lands): CTA 256 (M) x 128 (N), 4 waves 2x2 of 128 x 64; `v_mfma_scale_f32_16x16x128_f8f6f4` through
//! `TiledMma(swap_ab)` (hardware srcA = the B fragment: lane % 16 = M row of C, 4 consecutive N per lane); fragments read with `TiledCopy`
//! (`DS_READ_B128X2`) partitioned over the LDS view; global -> LDS DMA derived from the inverse of the LDS layout; scales staged through LDS by DMA;
//! `tile::Pipeline` OneWindow (3-slot ring, tile t+2 in flight).

use amdgpu_dsl::tile::{ev, ox, Cadence, LdsPlan, Lane, PipeBody, Pipeline, Role, Shared, SlotIdx, TileIt};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};
use amdgpu_isa::{MmaAtom, DS_READ_B128X2};
use amdgpu_layout::{composition, layout_to_offsets, make_layout, right_inverse, tup, ComposedLayout, ImmRange, IssuePlan, Ix, Layout, Swizzle};
use amdgpu_tile::{gemm, make_tiled_copy_a, make_tiled_copy_b, make_tiled_mma, Ctx, LdsPart, TiledMma};

/// Build configuration (4 waves 2 x 2 in every variant).
#[derive(Clone, Copy)]
struct Cfg {
    cta_m: i64,
    cta_n: i64,
    /// A/B MMA operand fragments in AGPRs (ds_read straight into a[..])
    agpr_ops: bool,
    /// accumulators (16x16 blocks, m-block-major) pinned in AGPRs: the first `acc_agpr`
    acc_agpr: usize,
    /// 3-slot OneWindow ring (prefetch distance 2) or the 2-slot cadence (distance 1, for tiles whose 3 slots do not fit the LDS)
    three_slots: bool,
    /// 2-slot `RegStaged` cadence: the whole tile is read into registers first, its slot then receives tile t+2 (distance 2 with two slots)
    reg_staged: bool,
    /// spread the DMA of the next tile over the MMA rows of the window (instead of issuing it all before the window)
    defer_dma: bool,
}
impl Cfg {
    fn wave_m(&self) -> i64 {
        self.cta_m / 2
    }
    fn wave_n(&self) -> i64 {
        self.cta_n / 2
    }
}

/// LDS image of a [rows x 128 bytes] K-tile: 16-row x 16-byte interleave (the aiter B preshuffle pattern with K = 128):
/// (r, k) -> (r / 16) 2048 + (k / 16) 256 + (r % 16) 16 + k % 16. A 16-lane group of every operand read covers 256 consecutive bytes (all banks once).
fn smem_tile(rows: i64) -> Layout {
    make_layout(tup![tup![16, rows / 16], tup![16, 8]], tup![tup![16, 2048], tup![1, 256]])
}
/// The same image restricted to one 16-row MMA operand block (element index r + 16 k, 16 x 128).
fn smem_block() -> Layout {
    make_layout(tup![16, tup![16, 8]], tup![16, tup![1, 256]])
}

/// `gemm` on loop variables (the operands of the register-staged tile are loop-carried): same operand / scale exchange as `amdgpu_tile::gemm`.
fn mma_var(b: &mut FnBuilder, mma: &TiledMma, acc: VarId, a: VarId, bf: VarId, sa: VarId, sb: VarId) {
    assert_eq!(mma.atom().name, "MfmaScale16x16x128F8");
    let (x, y, sx, sy) = if mma.swap_ab { (bf, a, sb, sa) } else { (a, bf, sa, sb) };
    b.var_mfma_scale16(acc, Opnd::Var(x), Opnd::Var(y), Opnd::Var(sx), Opnd::Var(sy), 0, 0);
}

/// The 96-byte ABI (opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950), declared in offset order.
fn declare_abi(b: &mut FnBuilder) {
    let ptrs = [("ptr_a", 0u32), ("ptr_b", 8), ("ptr_c", 16), ("ptr_sfa", 64), ("ptr_sfb", 72)];
    let ints = [("m", 24u32), ("n", 28), ("k", 32), ("batch", 36), ("stride_a", 40), ("stride_b", 44), ("stride_c", 48), ("stride_a_batch", 52), ("stride_b_batch", 56), ("stride_c_batch", 60), ("stride_sfa", 80), ("stride_sfb", 84), ("stride_sfa_batch", 88), ("stride_sfb_batch", 92)];
    let mut all: Vec<(&str, u32, bool)> = ptrs.iter().map(|&(n, o)| (n, o, true)).chain(ints.iter().map(|&(n, o)| (n, o, false))).collect();
    all.sort_by_key(|x| x.1);
    for (name, off, is_ptr) in all {
        if is_ptr {
            b.arg(name, DType::Ptr, off, 8)
        } else {
            b.arg(name, DType::I32, off, 4)
        }
    }
}

/// One DMA stage of a [rows x 128 B] tile: per issue the global offset (relative to the tile origin) and the M0 offset inside the slot.
struct Dma {
    /// per issue: the lane's global offset (bound once, loop invariant) and the LDS offset of its 1 KB block in the wave's slice
    issues: Vec<(ValueId, i32)>,
}

/// LDS byte `d = 1024 (issues (tid / 64) + q) + 16 (tid % 64)` holds tile element `smem^-1(d)`, so the source offset is `glob o right_inverse(smem)`;
/// proved here for sampled threads against a direct search of the LDS layout.
fn dma_plan(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64) -> Dma {
    let issues = rows * 128 / 1024 / 4;
    let sm = smem_tile(rows);
    let tv = make_layout(tup![tup![64, 4], tup![16, issues]], tup![tup![16, 1024 * issues], tup![1, 1024]]);
    let g = composition(glob, &right_inverse(&sm));
    let gofs = IssuePlan::new(tv, 16).offsets(&g, s_tid, &Ix::C(0), &ctx.dyns);
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    for th in [0i64, 1, 15, 16, 63, 64, 100, 255] {
        let env = |s: u32| if s == tid_id { th } else { proof_k };
        for q in 0..issues {
            let lds = 1024 * (issues * (th / 64) + q) + 16 * (th % 64);
            let (r, c) = (0..rows).flat_map(|r| (0..8).map(move |c| (r, 16 * c))).find(|&(r, c)| sm.at(r + rows * c) == lds).expect("LDS byte covered");
            let want = glob.crd2idx(&tup![r, c], &ctx.dyns).eval(&env);
            assert_eq!(gofs[q as usize].eval(&env), want, "DMA rows {rows} thread {th} issue {q}");
        }
    }
    // MUBUF form: voffset (loop invariant, bound here) + the instruction offset 1024 (q % 4) = the global offset; M0 = the group's first 1 KB block
    Dma { issues: gofs.iter().enumerate().map(|(q, g)| (b.bind_i32(ctx.lower(g) - 1024 * (q as i32 % 4)), 1024 * q as i32)).collect() }
}

/// A-tile DMA for the row-major LDS image with the FMHA 128-byte-row swizzle (`swz o (r, k) -> 128 r + k`): one instruction moves 8 whole
/// 128-byte rows (coalesced 128-byte segments, like the vendor), the swizzle is applied on the source side (the LDS landing is linear in the lane).
/// LDS byte d -> element (r, k) = swz(d) (the swizzle is an involution) -> global offset `glob(r, k)`; proved for sampled threads.
fn dma_plan_rows_swz(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64) -> Dma {
    let issues = rows * 128 / 1024 / 4;
    let swz = Swizzle::fmha128();
    let d = &ctx.dyns;
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    let mut out = Vec::new();
    for q in 0..issues {
        let lds = s_tid.mod_c(64, d).mul_c(16).add(&s_tid.div_c(64, d).mul_c(1024 * issues)).add_c(1024 * q);
        let e = swz.apply_ix(&lds, d);
        let g = glob.crd2idx(&tup![e.div_c(128, d), e.mod_c(128, d)], d);
        for th in [0i64, 1, 7, 8, 63, 64, 100, 255] {
            let env = |s: u32| if s == tid_id { th } else { proof_k };
            let lb = 1024 * (issues * (th / 64) + q) + 16 * (th % 64);
            let lin = swz.apply(lb);
            let want = glob.crd2idx(&tup![lin / 128, lin % 128], d).eval(&env);
            assert_eq!(g.eval(&env), want, "A DMA thread {th} issue {q}");
        }
        out.push((b.bind_i32(ctx.lower(&g) - 1024 * (q as i32 % 4)), 1024 * q as i32));
    }
    Dma { issues: out }
}

struct Body {
    cfg: Cfg,
    mma: TiledMma,
    a_s: Shared,
    b_s: Shared,
    sc_s: Shared,
    a_dma: Dma,
    b_dma: Dma,
    a_org: ValueId,
    b_org: ValueId,
    a_wave: ValueId,
    b_wave: ValueId,
    sc_wave: ValueId,
    rsrc_a: ValueId,
    rsrc_b: ValueId,
    ptr_sfa: ValueId,
    ptr_sfb: ValueId,
    m_v: ValueId,
    sa_src: ValueId,
    sb_src: ValueId,
    sa_rd: Vec<ValueId>,
    a_rd: LdsPart,
    b_rd: LdsPart,
    zero4: ValueId,
    acc: Vec<VarId>,
    /// deferred stage (key0, slot, units issued) and whether the pipeline loop has started
    deferred: Option<(I32, SlotIdx, usize)>,
    in_loop: bool,
    /// register-staged tile (`RegStaged`): initial values (tile 0) and the loop variables: B fragments, A fragments, A scales, B scale
    staged: Option<(Vec<ValueId>, Vec<ValueId>, Vec<ValueId>, ValueId)>,
    rv: Option<(Vec<VarId>, Vec<VarId>, Vec<VarId>, VarId)>,
}

impl Body {
    /// Loads of tile `t`'s operands from `slot`: scale words (one round trip) + byte selects, then the fragments in the order of `order`
    /// (`(is_b, index)`), each handed to `sink` right after its load (in place of the dead fragment).
    #[allow(clippy::type_complexity)]
    fn load_scales(&self, b: &mut FnBuilder, slot: &SlotIdx, kb: I32) -> (Vec<ValueId>, ValueId) {
        let nm = (self.cfg.wave_m() / 16) as usize;
        let scr = self.sc_s.at(slot);
        let so = scr.off_v(b);
        let scb = b.bind_i32(so + self.sc_s.base as i32 + ev(self.sc_wave));
        let (wa, wb) = scr.read(b, |b| {
            let wa: Vec<ValueId> = (0..nm).map(|i| {
                let a = b.bind_i32(ev(scb) + (ev(self.sa_rd[i]) & -4));
                b.load_lds_b32(a)
            }).collect();
            let a = b.bind_i32(ev(scb) + 256);
            (wa, b.load_lds_b32(a))
        });
        let sa: Vec<ValueId> = (0..nm).map(|i| b.bind_i32(ev(wa[i]).lshr((ev(self.sa_rd[i]) & 3) << 3))).collect();
        let sb = b.bind_i32(ev(wb).lshr(((ev(self.sb_src) + kb) & 3) << 3));
        (sa, sb)
    }
    fn frag(&self, b: &mut FnBuilder, part: &amdgpu_tile::LdsSlotPart, o: usize) -> ValueId {
        let v = part.read(b, o);
        if self.cfg.agpr_ops {
            b.agpr(v)
        } else {
            v
        }
    }
    /// The whole tile `t` into registers (prologue: tile 0).
    fn load_tile(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let (nm, nn) = ((self.cfg.wave_m() / 16) as usize, (self.cfg.wave_n() / 16) as usize);
        let (sa, sb) = self.load_scales(b, &t.cur, ox(t.it));
        let mut deferred = self.deferred.take();
        let ar = self.a_rd.at(b, &self.a_s.at(&t.cur));
        let br = self.b_rd.at(b, &self.b_s.at(&t.cur));
        let bf: Vec<ValueId> = (0..nn).map(|j| self.frag(b, &br, j)).collect();
        let af: Vec<ValueId> = (0..nm).map(|i| self.frag(b, &ar, i)).collect();
        self.staged = Some((bf, af, sa, sb));
    }
    /// The MMAs of the register tile; when `t.staged`, each fragment is reloaded from tile t+1 right after its last use: first half of the B
    /// fragments with every A fragment, then the B halves are reloaded; second half, after which every A fragment is reloaded; the scales of t+1
    /// are read before the MMAs.
    fn compute_tile(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let (bv, av, sav, sbv) = self.rv.clone().expect("state bound");
        let (nm, nn) = (av.len(), bv.len());
        let next = if t.staged { Some((self.a_rd.at(b, &self.a_s.at(&t.next)), self.b_rd.at(b, &self.b_s.at(&t.next)), (), ())) } else { None };
        let mut deferred = self.deferred.take();
        let total = self.a_dma.issues.len() + self.b_dma.issues.len() + 1;
        let per = (total + 2 * nm - 1) / (2 * nm);
        let half = nn / 2;
        for (lo, hi) in [(0, half), (half, nn)] {
            for i in 0..nm {
                if let Some((k, s, done)) = deferred.take() {
                    self.stage_now(b, &k, &s, done..done + per);
                    if done + per < total {
                        deferred = Some((k, s, done + per));
                    }
                }
                for j in lo..hi {
                    mma_var(b, &self.mma, self.acc[nn * i + j], av[i], bv[j], sav[i], sbv);
                }
                if lo == half {
                    if let Some((ar, _, _, _)) = &next {
                        // in-place reload: the ds_reads write the variable's own AGPRs (coalesced set_var)
                        let v = ar.read(b, i);
                        b.set_var(av[i], v);
                    }
                }
            }
            if let Some((_, br, _, _)) = &next {
                for j in lo..hi {
                    let v = br.read(b, j);
                    b.set_var(bv[j], v);
                }
            }
        }
        if t.staged {
            // the scales of t+1 last (their byte selects would otherwise wait at the top of the MMAs)
            let (sa, sb) = self.load_scales(b, &t.next, ox(t.it) + 1);
            for i in 0..nm {
                b.set_var(sav[i], sa[i]);
            }
            b.set_var(sbv, sb);
        }
    }
}
impl Body {
    /// The DMA of one tile (A issues, B issues, the 2 scale loads); `range` selects which of them to issue now (the deferred form spreads them
    /// over the MMA rows).
    fn stage_now(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotIdx, range: std::ops::Range<usize>) {
        // diagnostic ablations (wrong results): FMHA_ABLATE=gemm_nodma skips the copies
        if std::env::var("FMHA_ABLATE").map_or(false, |v| v.contains("gemm_nodma")) {
            return;
        }
        // A advances by key0 bytes, the preshuffled B by (key0 / 16) 256 = 16 key0
        // the tile origins are wave uniform: SGPR soffsets
        let ka = { let v = b.bind_i32(key0.clone() + ev(self.a_org)); b.read_first_lane(v) };
        let kb = { let v = b.bind_i32((key0.clone() << 4) + ev(self.b_org)); b.read_first_lane(v) };
        let mut unit = 0usize;
        for (sh, dma, org, rsrc, wave) in [(&self.a_s, &self.a_dma, ka, self.rsrc_a, self.a_wave), (&self.b_s, &self.b_dma, kb, self.rsrc_b, self.b_wave)] {
            let mine: Vec<(ValueId, i32)> = dma.issues.iter().enumerate().filter(|(q, _)| range.contains(&(unit + q))).map(|(_, x)| *x).collect();
            unit += dma.issues.len();
            if mine.is_empty() {
                continue;
            }
            let r = sh.at(slot);
            let o = r.off_v(b);
            let sbase = b.bind_i32(o + sh.base as i32 + ev(wave));
            r.write(b, |b| {
                // buffer_load_dwordx4 ... lds: voffset = the lane's loop-invariant offset, soffset = the tile origin (wave uniform), M0 = the first 1 KB
                // block of each group of 4 issues, instruction offset 1024 (q % 4) moves both the LDS and the global address
                let mut m0: Option<(i32, ValueId)> = None;
                for (g, lds) in mine.iter() {
                    let grp = *lds - *lds % 4096;
                    if m0.map(|x| x.0) != Some(grp) {
                        m0 = Some((grp, b.bind_i32(ev(sbase) + grp)));
                    }
                    b.load_buffer_lds(rsrc, *g, org, m0.unwrap().1, (*lds % 4096) as u32, 4);
                }
            });
        }
        // scales of tile kb: the CTA's 256 A-scale bytes (contiguous in [K/128][M]) and the dword holding the B-scale byte, into the wave's 512 bytes
        if !range.contains(&unit) {
            return;
        }
        let kbi = b.bind_i32(key0.clone().lshr(7));
        let r = self.sc_s.at(slot);
        let o = r.off_v(b);
        let sbase = b.bind_i32(o + self.sc_s.base as i32 + ev(self.sc_wave));
        let pa = {
            let o = b.bind_i32(ev(kbi) * ev(self.m_v) + ev(self.sa_src));
            b.ptr_add(self.ptr_sfa, o)
        };
        let pb = {
            let o = b.bind_i32((ev(self.sb_src) + ev(kbi)) & -4);
            b.ptr_add(self.ptr_sfb, o)
        };
        let sb_m0 = b.bind_i32(ev(sbase) + 256);
        r.write(b, |b| {
            b.load_global_lds(pa, sbase, 0, 1);
            b.load_global_lds(pb, sb_m0, 0, 1);
        });
    }
}

impl PipeBody for Body {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &I32, slot: &SlotIdx) {
        if self.cfg.defer_dma && self.in_loop {
            // issued inside the window, spread over the MMA rows
            self.deferred = Some((key0.clone(), slot.clone(), 0));
            return;
        }
        self.stage_now(b, key0, slot, 0..usize::MAX);
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        self.in_loop = true;
        if self.cfg.reg_staged {
            self.load_tile(b, t0);
        }
    }
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let mut v: Vec<(DType, Opnd)> = (0..(self.cfg.wave_m() / 16) * (self.cfg.wave_n() / 16)).map(|_| (DType::Frag4, Opnd::Val(self.zero4))).collect();
        if let Some((bf, af, sa, sb)) = &self.staged {
            v.extend(bf.iter().chain(af.iter()).map(|x| (DType::Frag8, Opnd::Val(*x))));
            v.extend(sa.iter().map(|x| (DType::I32, Opnd::Val(*x))));
            v.push((DType::I32, Opnd::Val(*sb)));
        }
        v
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        let na = ((self.cfg.wave_m() / 16) * (self.cfg.wave_n() / 16)) as usize;
        self.acc = vars[..na].to_vec();
        if let Some((bf, af, sa, _)) = self.staged.take() {
            let (nb, nf, ns) = (bf.len(), af.len(), sa.len());
            let r = &vars[na..];
            self.rv = Some((r[..nb].to_vec(), r[nb..nb + nf].to_vec(), r[nb + nf..nb + nf + ns].to_vec(), r[nb + nf + ns]));
        }
    }
    fn v_phase(&mut self, b: &mut FnBuilder, role: Role, t: &TileIt) {
        if !self.cfg.reg_staged {
            // 2-slot TwoPhaseOffset: the whole tile is computed between the stage of t+1 and its wait (the matrix phase stays empty)
            self.window(b, role, t);
        }
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        if self.cfg.reg_staged {
            self.compute_tile(b, t);
        }
    }
    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (nm, nn) = ((self.cfg.wave_m() / 16) as usize, (self.cfg.wave_n() / 16) as usize);
        let agpr = self.cfg.agpr_ops;
        let nords = std::env::var("FMHA_ABLATE").map_or(false, |v| v.contains("gemm_nords"));
        let zf = if nords { let z = b.frag_zero(DType::Frag4); Some(b.cat_frag8(z, z)) } else { None };
        let zfb = if nords { let z = b.frag_zero(DType::Frag4); let o = b.frag_zero(DType::Frag4); Some(b.cat_frag8(o, z)) } else { None };
        let mark = |b: &mut FnBuilder, v: ValueId| if agpr { b.agpr(v) } else { v };
        let kbx = ox(t.it);
        // scale words (one LDS round trip), then the byte selects (shifted to byte 0)
        let scr = self.sc_s.at(&t.cur);
        let so = scr.off_v(b);
        let scb = b.bind_i32(so + self.sc_s.base as i32 + ev(self.sc_wave));
        let (wa, wb) = scr.read(b, |b| {
            let wa: Vec<ValueId> = (0..nm).map(|i| {
                let a = b.bind_i32(ev(scb) + (ev(self.sa_rd[i]) & -4));
                b.load_lds_b32(a)
            }).collect();
            let a = b.bind_i32(ev(scb) + 256);
            (wa, b.load_lds_b32(a))
        });
        let mut deferred = self.deferred.take();
        let ar = self.a_rd.at(b, &self.a_s.at(&t.cur));
        let br = self.b_rd.at(b, &self.b_s.at(&t.cur));
        // A0 and B0 first; the other B fragments are issued one ahead of the first row of MMAs (partial lgkm waits)
        let mut af = { let v = if let Some(z) = zf { z } else { ar.read(b, 0) }; mark(b, v) };
        let bpre = std::env::var("FMHA_ABLATE").ok().and_then(|v| v.split(",").find_map(|t| t.strip_prefix("bpre").and_then(|n| n.parse::<usize>().ok()))).unwrap_or(2).max(1).min(nn);
        let mut bf: Vec<ValueId> = (0..bpre).map(|j| { let v = if let Some(z) = zfb { z } else { br.read(b, j) }; mark(b, v) }).collect();
        let sa: Vec<ValueId> = (0..nm).map(|i| b.bind_i32(ev(wa[i]).lshr((ev(self.sa_rd[i]) & 3) << 3))).collect();
        let sb = b.bind_i32(ev(wb).lshr(((ev(self.sb_src) + kbx) & 3) << 3));
        let prio = std::env::var("FMHA_ABLATE").map_or(false, |v| v.contains("prio"));
        if prio {
            b.set_prio(1);
        }
        for i in 0..nm {
            // prefetch the next A fragment while this one feeds 4 MMAs
            let next = if i + 1 < nm { let v = if let Some(z) = zf { z } else { ar.read(b, i + 1) }; Some(mark(b, v)) } else { None };
            if let Some((k, s, done)) = deferred.take() {
                let total = self.a_dma.issues.len() + self.b_dma.issues.len() + 1;
                let per = (total + nm - 1) / nm;
                self.stage_now(b, &k, &s, done..done + per);
                if done + per < total {
                    deferred = Some((k, s, done + per));
                }
            }
            for j in 0..nn {
                if i == 0 && j + bpre < nn {
                    let v = if let Some(z) = zfb { z } else { br.read(b, j + bpre) };
                    bf.push(mark(b, v));
                }
                if !std::env::var("FMHA_ABLATE").map_or(false, |v| v.contains("gemm_nomma")) {
                    gemm(b, &self.mma, self.acc[nn * i + j], af, bf[j], sa[i], sb, 0, 0);
                }
            }
            if let Some(n) = next {
                af = n;
            }
        }
        if prio {
            b.set_prio(0);
        }
    }
}

pub fn gemm_mxfp8_bs_v2() -> Function {
    build("gemm_mxfp8_bs_v2", Cfg { cta_m: 256, cta_n: 128, agpr_ops: false, acc_agpr: 0, three_slots: true, reg_staged: false, defer_dma: false })
}

/// v3: the vendor tile (CTA 256 x 256, 128 x 128 per wave, 64 accumulators): operand fragments in AGPRs, half of the accumulators pinned in AGPRs,
/// 2-slot LDS ring (2 x 64 KB).
pub fn gemm_mxfp8_bs_v3() -> Function {
    build("gemm_mxfp8_bs_v3", Cfg { cta_m: 256, cta_n: 256, agpr_ops: true, acc_agpr: 32, three_slots: false, reg_staged: false, defer_dma: !std::env::var("FMHA_ABLATE").map_or(false, |v| v.contains("nodefer")) })
}

/// v4: v3 with the register-staged 2-slot cadence (prefetch distance 2): all 16 operand fragments of a tile are read into AGPRs, the slot is refilled
/// with tile t+2 while the 64 MMAs run.
pub fn gemm_mxfp8_bs_v4() -> Function {
    build("gemm_mxfp8_bs_v4", Cfg { cta_m: 256, cta_n: 256, agpr_ops: true, acc_agpr: 24, three_slots: false, reg_staged: true, defer_dma: std::env::var("FMHA_ABLATE").map_or(false, |v| v.contains("defer4")) })
}

fn build(name: &str, cfg: Cfg) -> Function {
    let (cta_m, cta_n, wave_m, wave_n) = (cfg.cta_m, cfg.cta_n, cfg.wave_m(), cfg.wave_n());
    let mut fb = FnBuilder::new(name, 256);
    let b = &mut fb;
    declare_abi(b);
    let ptr_a = b.let_(Op::KernargPtr { offset: 0 });
    let ptr_b = b.let_(Op::KernargPtr { offset: 8 });
    let ptr_c = b.let_(Op::KernargPtr { offset: 16 });
    let ptr_sfa = b.let_(Op::KernargPtr { offset: 64 });
    let ptr_sfb = b.let_(Op::KernargPtr { offset: 72 });
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let n_v = b.let_(Op::KernargI32 { offset: 28 });
    let k_v = b.let_(Op::KernargI32 { offset: 32 });
    let kb_v = b.let_(Op::KernargI32 { offset: 84 });
    let tid = b.let_(Op::WorkitemId);
    let lane = Lane::from_tid(b, tid);
    let bx = b.let_(Op::WorkgroupIdX); // N tile
    let by = b.let_(Op::WorkgroupIdY); // M tile
    let (lg_m, lg_n) = (cta_m.trailing_zeros() as i32, cta_n.trailing_zeros() as i32);
    let wm = b.bind_i32(lane.wave() & 1);
    let wn = b.bind_i32(lane.wave().lshr(1));

    // ---- layout symbols ----
    let mut ctx = Ctx::new();
    let s_lane = ctx.sym("lane", 0, 63, lane.lane());
    let s_tid = ctx.sym("tid", 0, 255, ev(tid));
    let s_k = ctx.sym("K", 128, 1 << 20, ev(k_v));
    let s_n = ctx.sym("N", 128, 1 << 20, ev(n_v));
    let s_wm = ctx.sym("wm", 0, 1, ev(wm));
    let s_wn = ctx.sym("wn", 0, 1, ev(wn));

    // ---- MMA + operand copies ----
    let mma = make_tiled_mma(MmaAtom::mfma_scale_16x16x128_f8(), true);
    let copy_a = make_tiled_copy_a(DS_READ_B128X2, &mma);
    let copy_b = make_tiled_copy_b(DS_READ_B128X2, &mma);

    // ---- LDS ring: 3 slots of A 32 KB + B 16 KB + scales 2 KB ----
    let mut lds = LdsPlan::new();
    let slots = if cfg.three_slots { 3 } else { 2 };
    let a_s = lds.ring("A", cta_m * 128, slots, amdgpu_dsl::tile::Layout::Linear);
    let b_s = lds.ring("B", cta_n * 128, slots, amdgpu_dsl::tile::Layout::Linear);
    let sc_s = lds.ring("SC", 4 * 512, slots, amdgpu_dsl::tile::Layout::Linear);
    lds.commit(b);

    // ---- operand reads: the 16-row blocks of the wave (placements (row block) 2048), the wave offset added to the bases ----
    let view = ComposedLayout::new(Swizzle::none(), smem_block());
    // A: 128-byte rows with the FMHA swizzle (a 16-row block = 2048 bytes; the swizzle only depends on row bits 0..3)
    let a_view = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![16, 128], tup![128, 1]));
    let a_place: Vec<i64> = (0..wave_m / 16).map(|i| 2048 * i).collect();
    let b_place: Vec<i64> = (0..wave_n / 16).map(|j| 2048 * j).collect();
    let mut a_rd = copy_a.partition_lds(b, &ctx, &s_lane, &a_s, &a_view, &a_place);
    let mut b_rd = copy_b.partition_lds(b, &ctx, &s_lane, &b_s, &view, &b_place);
    let a_wo = ctx.lower(&s_wm.mul_c(wave_m / 16 * 2048));
    let b_wo = ctx.lower(&s_wn.mul_c(wave_n / 16 * 2048));
    a_rd.bases = a_rd.bases.iter().map(|x| b.let_i32(x + &a_wo)).collect();
    b_rd.bases = b_rd.bases.iter().map(|x| b.let_i32(x + &b_wo)).collect();

    // ---- DMA (A row-major, B preshuffled), wave parts of M0, CTA origins ----
    let a_glob = make_layout(tup![cta_m, 128], tup![s_k.clone(), 1]);
    let b_glob = make_layout(tup![tup![16, cta_n / 16], tup![16, 8]], tup![tup![16, s_k.mul_c(16)], tup![1, 256]]);
    let a_dma = dma_plan_rows_swz(b, &ctx, &s_tid, cta_m, &a_glob, 8192);
    let b_dma = dma_plan(b, &ctx, &s_tid, cta_n, &b_glob, 8192);
    let a_wave = b.bind_i32(lane.wave() * (cta_m * 128 / 4) as i32);
    let b_wave = b.bind_i32(lane.wave() * (cta_n * 128 / 4) as i32);
    // buffer descriptors over A (M K bytes) and the preshuffled B (N K bytes)
    let rsrc_a = { let n = b.bind_i32(ev(m_v) * ev(k_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_a, n) };
    let rsrc_b = { let n = b.bind_i32(ev(n_v) * ev(k_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_b, n) };
    let a_org = b.bind_i32((ev(by) << lg_m) * ev(k_v));
    let b_org = b.bind_i32((ev(bx) << lg_n) * ev(k_v));

    // ---- scales: A row of this lane in m-block i = 128 wm + 16 i + lane % 16; B-scale row bx (CTA_N = 128 = one block) ----
    let sc_wave = b.bind_i32(lane.wave() << 9);
    // the A-scale DMA covers 256 rows (64 lanes x 4 bytes): the CTA has at most 256
    assert!(cta_m <= 256);
    let sa_src = b.bind_i32((ev(by) << lg_m) + (lane.lane() << 2));
    let sa_rd: Vec<ValueId> = (0..wave_m / 16).map(|i| b.bind_i32(ctx.lower(&s_wm.mul_c(wave_m).add_c(16 * i).add(&s_lane.mod_c(16, &ctx.dyns))))).collect();
    // B-scale row of this wave's 128-column block (wave_n <= 128): (n_cta + wn wave_n) / 128
    assert!(wave_n <= 128 && 128 % wave_n == 0);
    let sb_row = b.bind_i32(((ev(bx) << lg_n) + (ev(wn) * wave_n as i32)).lshr(7));
    let sb_src = b.bind_i32(ev(sb_row) * ev(kb_v));

    let cadence = if cfg.reg_staged { Cadence::RegStaged } else if cfg.three_slots { Cadence::OneWindow } else { Cadence::TwoSlot };
    let mut pipe = Pipeline::new(cadence, kb_v, k_v, 128, Ext::ka(84));
    if cfg.three_slots || cfg.reg_staged {
        pipe.vm_inflight = (a_dma.issues.len() + b_dma.issues.len() + 2) as u8;
    }
    let zero4 = b.frag_zero(DType::Frag4);
    let mut body = Body {
        cfg, mma: mma.clone(), a_s, b_s, sc_s, a_dma, b_dma, a_org, b_org, a_wave, b_wave, sc_wave, rsrc_a, rsrc_b, ptr_sfa, ptr_sfb, m_v, sa_src, sb_src, sa_rd,
        a_rd, b_rd, zero4, acc: Vec::new(), staged: None, rv: None, deferred: None, in_loop: false,
    };
    pipe.run(b, Role::All, &mut body);
    for v in body.acc.iter().take(cfg.acc_agpr) {
        b.pin_agpr(*v);
    }
    if let (Some((bv, av, _, _)), true) = (&body.rv, cfg.agpr_ops) {
        for v in bv.iter().chain(av.iter()) {
            b.pin_agpr(*v);
        }
    }

    // ---- epilogue: the (swapped) C layout of the atom x C row-major (bf16); 4 consecutive N per lane = one 8-byte store ----
    let c_blk = make_layout(tup![16, 16], tup![s_n.mul_c(2), 2]); // element m + 16 n
    let c_plan = IssuePlan::new(mma.c_tv(), 4);
    let c_org = b.bind_i32((((ev(by) << lg_m) * ev(n_v)) + (ev(bx) << lg_n)) << 1);
    let cb = b.ptr_add(ptr_c, c_org);
    let mut exprs = Vec::new();
    for i in 0..wave_m / 16 {
        for j in 0..wave_n / 16 {
            let org = s_wm.mul_c(wave_m).add_c(16 * i).mul(&s_n).mul_c(2).add(&s_wn.mul_c(wave_n).add_c(16 * j).mul_c(2));
            exprs.extend(c_plan.offsets(&c_blk, &s_lane, &org, &ctx.dyns));
        }
    }
    let plan = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::GLOBAL, &ctx.dyns);
    let bases: Vec<ValueId> = plan.bases.iter().map(|e| {
        let o = b.bind_i32(ctx.lower(e));
        b.ptr_add(cb, o)
    }).collect();
    for (n, acc) in body.acc.iter().enumerate() {
        let v: Vec<ValueId> = (0..4u8).map(|r| {
            let l = b.frag4_lane(Opnd::Var(*acc), r);
            b.bitcast_f32(l)
        }).collect();
        let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
        let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
        let f2 = b.frag2(w0, w1);
        let (bi, imm) = plan.issues[n];
        b.store_f32x2(bases[bi], imm as u32, f2);
    }
    fb.finish()
}
