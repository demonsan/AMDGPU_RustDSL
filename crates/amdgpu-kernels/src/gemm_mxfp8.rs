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

/// Diagnostic ablations (wrong results, timing only): remove one stream to measure the others.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ablate {
    /// no global -> LDS copies
    pub no_dma: bool,
    /// no MMAs
    pub no_mma: bool,
    /// no operand LDS reads (zero fragments)
    pub no_lds_reads: bool,
}

/// Build configuration of the MXFP8 blockscale GEMM (4 waves 2 x 2 in every variant). `GemmConfig::v3()` is the best measured one.
#[derive(Clone, Copy, Debug)]
pub struct GemmConfig {
    pub cta_m: i64,
    pub cta_n: i64,
    /// A/B MMA operand fragments in AGPRs (ds_read straight into a[..])
    pub agpr_ops: bool,
    /// accumulators (16x16 blocks, m-block-major) pinned in AGPRs: the first `acc_agpr`
    pub acc_agpr: usize,
    /// 3-slot OneWindow ring (prefetch distance 2) or the 2-slot cadence (distance 1, for tiles whose 3 slots do not fit the LDS)
    pub three_slots: bool,
    /// 2-slot `RegStaged` cadence: operands are loop-carried registers reloaded in place from tile t+1 (the vendor structure)
    pub reg_staged: bool,
    /// spread the DMA of the next tile over the MMAs of the window (instead of issuing it as one burst)
    pub defer_dma: bool,
    /// one DMA instruction every `dma_gap` MMAs (with `defer_dma`)
    pub dma_gap: usize,
    /// B fragments read ahead of the first row of MMAs (window cadences)
    pub b_prefetch: usize,
    /// `s_setprio 1` around the MMAs of the window
    pub setprio: bool,
    /// all scales of the CTA in one LDS panel loaded in the prologue (K <= 128 x PANEL_KB), no per-tile scale DMA
    pub scale_panel: bool,
    /// panel A part transposed in the prologue: 2 dword reads per tile + MFMA byte selects (window cadences)
    pub panel_transposed: bool,
    /// XCD-aware 4 x 4 tile raster (16 x 16 tile grids)
    pub xcd_raster: bool,
    /// `RegStaged`: MMAs issued before the wait + barrier of the iteration
    pub pre_sync_mmas: usize,
    /// DMA issues share voffsets, the wave-uniform source deltas go to the soffset (pays off once soffset adds are SALU)
    pub shared_voffsets: bool,
    /// epilogue through LDS: bf16 C block transposed in a wave-private staging area, then 16-byte row-contiguous stores (the vendor form)
    pub lds_epilogue: bool,
    /// `RegStaged`: the scale reads of t+1 issued before the fragment reloads (otherwise after them: an lgkmcnt(0) drain)
    pub early_scales: bool,
    /// extra scheduler tokens for this kernel (Function::sched_opts, on top of the Gemm profile's wait pass), e.g. "hoistchain"
    pub sched: &'static str,
    /// `RegStaged`: two tiles per loop pass with static ring slots (Pipeline::unroll2, the vendor u2)
    pub unroll2: bool,
    /// XCD raster: each XCD's two 4 x 4 blocks stacked vertically (8 x 4 tiles) instead of side by side
    pub raster_tall: bool,
    pub ablate: Ablate,
}

impl GemmConfig {
    /// VGPR-only, CTA 256 x 128, 3-slot OneWindow ring.
    pub fn v2() -> GemmConfig {
        GemmConfig {
            cta_m: 256, cta_n: 128, agpr_ops: false, acc_agpr: 0, three_slots: true, reg_staged: false, defer_dma: false, dma_gap: 2, b_prefetch: 2,
            setprio: false, scale_panel: false, panel_transposed: false, xcd_raster: false, pre_sync_mmas: 0, shared_voffsets: false, lds_epilogue: false, early_scales: false, sched: "", unroll2: false, raster_tall: false, ablate: Ablate::default(),
        }
    }
    /// The best measured: CTA 256 x 256, AGPR operands + 32 AGPR accumulators, 2-slot ring with one barrier per tile, DMA every 3 MMAs (2 before the SALU address chains), transposed
    /// scale panel, XCD raster.
    pub fn v3() -> GemmConfig {
        GemmConfig {
            cta_m: 256, cta_n: 256, agpr_ops: true, acc_agpr: 32, three_slots: false, reg_staged: false, defer_dma: true, dma_gap: 3, b_prefetch: 2,
            setprio: false, scale_panel: true, panel_transposed: true, xcd_raster: true, pre_sync_mmas: 0, shared_voffsets: false, lds_epilogue: true, early_scales: true, sched: "", unroll2: false, raster_tall: false, ablate: Ablate::default(),
        }
    }
    /// The vendor structure: loop-carried AGPR operands reloaded in place (RegStaged), 24 AGPR accumulators.
    pub fn v4() -> GemmConfig {
        GemmConfig { acc_agpr: 24, reg_staged: true, pre_sync_mmas: 4, sched: "rewait,hoistchain", unroll2: true, ..GemmConfig::v3() }
    }
}

type Cfg = GemmConfig;
/// Epilogue staging row pitch (bytes): 256-byte bf16 rows + 16 bytes of padding (the 16 rows a write instruction touches spread over the banks).
const EPI_PITCH: i64 = 272;

/// Scale panel capacity in K groups of 128 (the vendor `sp64`): K <= 8192.
const PANEL_KB: i64 = 64;
impl GemmConfig {
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
#[allow(clippy::too_many_arguments)]
fn mma_var(b: &mut FnBuilder, mma: &TiledMma, acc: VarId, a: VarId, bf: VarId, sa: VarId, sb: VarId, byte_a: u8) {
    assert_eq!(mma.atom().name, "MfmaScale16x16x128F8");
    let (x, y, sx, sy, bx, by) = if mma.swap_ab { (bf, a, sb, sa, 0, byte_a) } else { (a, bf, sa, sb, byte_a, 0) };
    b.var_mfma_scale16(acc, Opnd::Var(x), Opnd::Var(y), Opnd::Var(sx), Opnd::Var(sy), bx, by);
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

/// The host contract of the ABI for a `cta_m x cta_n` CTA (what `gpu-verify` proves the accesses against): M = MT cta_m, N = NT cta_n, K = KT 128
/// (the kernel has no partial tiles), the grid (NT, MT), dense strides (stride_a = stride_b = K, stride_c = N, stride_sfa = M, stride_sfb = K / 128),
/// batch 1, and the tensor extents A M K, B N K (preshuffled, same size), C 2 M N, sa (K / 128) M, sb (N / 128) (K / 128) bytes.
fn declare_contract(b: &mut FnBuilder, cta_m: i64, cta_n: i64) {
    let (mt, nt, kt) = (Ext::sym("MT"), Ext::sym("NT"), Ext::sym("KT"));
    // int32 offsets: M K, N K and 2 M N stay below 2^31 (16384 x 65536 tiles of bytes at most)
    b.sym_range("MT", 1, (1 << 14) / cta_m);
    b.sym_range("NT", 1, (1 << 14) / cta_n);
    // K <= 128 PANEL_KB: the scale panel holds every K group of the CTA (the vendor sp64 limit)
    b.sym_range("KT", 1, PANEL_KB);
    let m = mt.clone() * Ext::c(cta_m);
    let n = nt.clone() * Ext::c(cta_n);
    let k = kt.clone() * Ext::c(128);
    b.ka_is(24, m.clone());
    b.ka_is(28, n.clone());
    b.ka_is(32, k.clone());
    b.ka_is(36, Ext::c(1));
    b.ka_is(40, k.clone());
    b.ka_is(44, k.clone());
    b.ka_is(48, n.clone());
    b.ka_is(80, m.clone());
    b.ka_is(84, kt.clone());
    b.grid_is(0, nt.clone());
    b.grid_is(1, mt.clone());
    b.arg_tensor("ptr_a", m.clone() * k.clone());
    b.arg_tensor("ptr_b", n.clone() * k.clone());
    b.arg_tensor("ptr_c", Ext::c(2) * m.clone() * n.clone());
    b.arg_tensor("ptr_sfa", kt.clone() * m);
    b.arg_tensor("ptr_sfb", nt * Ext::c(cta_n / 128) * kt);
}

/// One DMA stage of a [rows x 128 B] tile: per issue the global offset (relative to the tile origin) and the M0 offset inside the slot.
struct Dma {
    /// per issue: the lane's voffset (bound once, loop invariant, shared by the issues whose sources differ by a wave-uniform amount), that
    /// wave-uniform source delta (an SGPR added to the soffset, like the vendor's one `s_add` per DMA) and the LDS offset of its 1 KB block
    issues: Vec<(ValueId, i32)>,
    delta: Vec<Option<ValueId>>,
}

/// Turns the per-issue source offsets (minus the instruction offset 1024 (q % 4)) into voffsets: an issue whose offset differs from an earlier
/// issue's by an amount independent of the thread (checked for all 256 threads at two K values) reuses that issue's voffset and adds the delta
/// to the soffset (SGPR, computed once).
fn finish_dma(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, gofs: Vec<Ix>, share: bool) -> Dma {
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    // `GemmConfig::shared_voffsets` (today a v_add + v_readfirstlane per issue costs more than the VGPRs it saves)
    let adj: Vec<Ix> = gofs.iter().enumerate().map(|(q, g)| g.add_c(-1024 * (q as i64 % 4))).collect();
    let mut voff: Vec<ValueId> = Vec::new();
    let mut base_of: Vec<usize> = Vec::new(); // issue index whose voffset is shared
    let mut issues: Vec<(ValueId, i32)> = Vec::new();
    let mut delta: Vec<Option<ValueId>> = Vec::new();
    for (q, g) in adj.iter().enumerate() {
        let found = if share {
            base_of.iter().copied().find(|&r| {
                let d = g.sub(&adj[r]);
                [4096i64, 8192].iter().all(|&k| {
                    let v0 = d.eval(&|s| if s == tid_id { 0 } else { k });
                    (0..256).all(|th| d.eval(&|s| if s == tid_id { th } else { k }) == v0)
                })
            })
        } else {
            None
        };
        match found {
            Some(r) => {
                let d = g.sub(&adj[r]);
                let dv = b.bind_i32(ctx.lower(&d));
                let ds = b.read_first_lane(dv);
                issues.push((issues[r].0, 1024 * q as i32));
                delta.push(Some(ds));
                let _ = &voff;
            }
            None => {
                let v = b.bind_i32(ctx.lower(g));
                voff.push(v);
                base_of.push(q);
                issues.push((v, 1024 * q as i32));
                delta.push(None);
            }
        }
    }
    Dma { issues, delta }
}

/// LDS byte `d = 1024 (issues (tid / 64) + q) + 16 (tid % 64)` holds tile element `smem^-1(d)`, so the source offset is `glob o right_inverse(smem)`;
/// proved here for sampled threads against a direct search of the LDS layout.
fn dma_plan(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64, share: bool) -> Dma {
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
    // MUBUF form: voffset (loop invariant) + soffset + the instruction offset 1024 (q % 4) = the global offset; M0 = the group's first 1 KB block
    finish_dma(b, ctx, s_tid, gofs, share)
}

/// A-tile DMA for the row-major LDS image with the FMHA 128-byte-row swizzle (`swz o (r, k) -> 128 r + k`): one instruction moves 8 whole
/// 128-byte rows (coalesced 128-byte segments, like the vendor), the swizzle is applied on the source side (the LDS landing is linear in the lane).
/// LDS byte d -> element (r, k) = swz(d) (the swizzle is an involution) -> global offset `glob(r, k)`; proved for sampled threads.
fn dma_plan_rows_swz(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64, share: bool) -> Dma {
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
        out.push(g);
    }
    finish_dma(b, ctx, s_tid, out, share)
}

/// The scale panel: every K group's scales of the CTA, loaded once in the prologue (`Cfg::scale_panel`).
struct Panel {
    sh: Shared,
    /// LDS offset of the B part inside the panel
    b_base: i64,
    /// byte offset of this wave's B scales inside the B part (wave column block x KB + the source alignment delta)
    b_off: ValueId,
    /// A part transposed in the prologue: [kb][wm][lane % 16][m block i] (8 bytes per lane: 2 dwords, MFMA byte selects instead of shifts)
    tr: Option<ValueId>,
    /// the transposition still to emit (prologue_stores)
    job: Option<TrJob>,
}

/// The in-place panel transposition: thread (r = tid % 16, wm = tid / 16 % 2, k = tid / 32) gathers the bytes of rows 128 wm + 16 i + r (i < 8) of
/// K group 8 round + k (one of the rows its own wave loaded) and writes them as 8 consecutive bytes at 128 wm + 8 r.
#[derive(Clone, Copy)]
struct TrJob {
    rd0: ValueId,
    wr0: ValueId,
    sh_r: ValueId,
}

struct Body {
    cfg: Cfg,
    /// K / 128 (kernarg)
    kb_v: ValueId,
    panel: Option<Panel>,
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
    rsrc_sfa: ValueId,
    rsrc_sfb: ValueId,
    ptr_sfa: ValueId,
    ptr_sfb: ValueId,
    m_v: ValueId,
    sa_src: ValueId,
    sb_src: ValueId,
    sa_rd: Vec<ValueId>,
    sa_sh: Vec<ValueId>,
    sa_dw: Vec<ValueId>,
    a_rd: LdsPart,
    b_rd: LdsPart,
    zero4: ValueId,
    acc: Vec<VarId>,
    /// deferred stage (key0, slot, units issued) and whether the pipeline loop has started
    deferred: Option<(I32, SlotIdx, usize, StageCache)>,
    /// MMAs of the register tile already issued by `pre_sync`
    npre_done: usize,
    in_loop: bool,
    /// register-staged tile (`RegStaged`): initial values (tile 0) and the loop variables: B fragments, A fragments, A scales, B scale
    staged: Option<(Vec<ValueId>, Vec<ValueId>, Vec<ValueId>, ValueId)>,
    rv: Option<(Vec<VarId>, Vec<VarId>, Vec<VarId>, VarId)>,
}

impl Body {
    /// Loads of tile `t`'s operands from `slot`: scale words (one round trip) + byte selects, then the fragments in the order of `order`
    /// (`(is_b, index)`), each handed to `sink` right after its load (in place of the dead fragment).
    #[allow(clippy::type_complexity)]
    /// The scale words of tile `kb` (one LDS round trip; ring slot or the panel) and the B byte shift: `(A words, B word, B shift)`.
    fn scale_words(&self, b: &mut FnBuilder, slot: &SlotIdx, kb: I32) -> (Vec<ValueId>, ValueId, I32) {
        let nm = (self.cfg.wave_m() / 16) as usize;
        if let Some(p) = &self.panel {
            // scale panel (loaded once): A rows [kb][256], B bytes [wn][KB] (+ the alignment delta of the B source)
            let pr = p.sh.at(&SlotIdx::new(0, 0));
            let sab = b.bind_i32((kb.clone() << 8) + (p.sh.base as i32) + ev(self.sa_dw[0]));
            let bo = b.bind_i32(ev(p.b_off) + kb);
            let sbb = b.bind_i32((ev(bo).lshr(2) << 2) + (p.sh.base as i32 + p.b_base as i32));
            let (wa, wb) = pr.read(b, |b| {
                let wa: Vec<ValueId> = (0..nm).map(|i| {
                    let a = b.bind_i32(ev(sab) + 16 * i as i32);
                    b.load_lds_b32(a)
                }).collect();
                (wa, b.load_lds_b32(sbb))
            });
            return (wa, wb, (ev(bo) & 3) << 3);
        }
        let scr = self.sc_s.at(slot);
        let so = scr.off_v(b);
        let scb = b.bind_i32(so + self.sc_s.base as i32 + ev(self.sc_wave));
        let sab = b.bind_i32(ev(scb) + ev(self.sa_dw[0]));
        let (wa, wb) = scr.read(b, |b| {
            let wa: Vec<ValueId> = (0..nm).map(|i| {
                // rows 16 i apart: the same lane dword + 16 i (an immediate)
                let a = b.bind_i32(ev(sab) + 16 * i as i32);
                b.load_lds_b32(a)
            }).collect();
            let a = b.bind_i32(ev(scb) + 256);
            (wa, b.load_lds_b32(a))
        });
        (wa, wb, ((ev(self.sb_src) + kb) & 3) << 3)
    }
    /// Scale words + byte selects (shifted to byte 0).
    /// The scale reads of tile `kb` (issue only): the A words (transposed panel: final, 2 words; otherwise to be shifted), the B word and its shift.
    fn scale_issue(&self, b: &mut FnBuilder, slot: &SlotIdx, kb: I32) -> (Vec<ValueId>, ValueId, I32, bool) {
        // K group of a real tile (t + 1 is read only when it exists): the clamp is a no-op that bounds it for the verifier's loop-exit-blind intervals
        let kb = b.bind_i32(kb.lt(ev(self.kb_v)).select_i32(kb.clone(), ev(self.kb_v) - 1));
        let kb = ev(kb);
        if let Some((base, b_base, b_off, lo, pr)) = self.panel.as_ref().and_then(|p| p.tr.map(|o| (p.sh.base, p.b_base, p.b_off, o, p.sh.at(&SlotIdx::new(0, 0))))) {
            let sab = b.bind_i32((kb.clone() << 8) + (base as i32) + ev(lo));
            let bo = b.bind_i32(ev(b_off) + kb);
            let sbb = b.bind_i32((ev(bo).lshr(2) << 2) + (base + b_base) as i32);
            let (wa, wb) = pr.read(b, |b| {
                // the 8 A-scale bytes of this lane: one ds_read_b64
                let w2 = b.load_lds_b64(sab);
                let wa = vec![b.frag2_lane(w2, 0), b.frag2_lane(w2, 1)];
                (wa, b.load_lds_b32(sbb))
            });
            return (wa, wb, (ev(bo) & 3) << 3, true);
        }
        let (wa, wb, sh) = self.scale_words(b, slot, kb);
        (wa, wb, sh, false)
    }
    /// The byte selects of `scale_issue`'s words (shifts to byte 0 where the MMA cannot select).
    fn scale_finish(&self, b: &mut FnBuilder, w: (Vec<ValueId>, ValueId, I32, bool)) -> (Vec<ValueId>, ValueId) {
        let (wa, wb, sh, fin) = w;
        let sa = if fin { wa } else { wa.iter().map(|x| b.bind_i32(ev(*x).lshr(ev(self.sa_sh[0])))).collect() };
        (sa, b.bind_i32(ev(wb).lshr(sh)))
    }
    /// With the transposed panel the A words are NOT shifted (2 words, the MMA selects byte i % 4 of word i / 4: `sa_sel`).
    fn load_scales(&self, b: &mut FnBuilder, slot: &SlotIdx, kb: I32) -> (Vec<ValueId>, ValueId) {
        let w = self.scale_issue(b, slot, kb);
        self.scale_finish(b, w)
    }    /// The (word, byte) of the A scale of m block i in the `load_scales` words.
    fn sa_sel(&self, i: usize) -> (usize, u8) {
        if self.panel.as_ref().map_or(false, |p| p.tr.is_some()) {
            (i / 4, (i % 4) as u8)
        } else {
            (i, 0)
        }
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
        // the scales of t+1 are read FIRST (before the fragment reloads queue up behind them) and selected at the end (partial lgkm waits)
        let early = if t.staged && self.cfg.early_scales { Some(self.scale_issue(b, &t.next, ox(t.it) + 1)) } else { None };
        let mut deferred = self.deferred.take();
        let total = self.a_dma.issues.len() + self.b_dma.issues.len() + 1;
        let per = (total + 2 * nm - 1) / (2 * nm);
        let prio = self.cfg.setprio;
        if prio {
            b.set_prio(1);
        }
        let half = nn / 2;
        let gap = self.cfg.dma_gap.max(1);
        let mut nmma = 0usize;
        let _ = per;
        for (lo, hi) in [(0, half), (half, nn)] {
            for i in 0..nm {
                for j in lo..hi {
                    if lo == 0 && i * half + j < self.npre_done {
                        continue; // issued before the barrier (pre_sync)
                    }
                    // the stage of t+2: one DMA every `gap` MMAs
                    if nmma % gap == 0 {
                        if let Some((k, s, done, mut cache)) = deferred.take() {
                            self.stage_now(b, &k, &s, done..done + 1, &mut cache);
                            if done + 1 < total {
                                deferred = Some((k, s, done + 1, cache));
                            }
                        }
                    }
                    nmma += 1;
                    let (w, by) = self.sa_sel(i);
                    mma_var(b, &self.mma, self.acc[nn * i + j], av[i], bv[j], sav[w], sbv, by);
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
        if prio {
            b.set_prio(0);
        }
        if t.staged {
            // the scales of t+1 (read early, or here)
            let (sa, sb) = match early {
                Some(w) => self.scale_finish(b, w),
                None => self.load_scales(b, &t.next, ox(t.it) + 1),
            };
            for (i, v) in sa.iter().enumerate() {
                b.set_var(sav[i], *v);
            }
            b.set_var(sbv, sb);
        }
    }
}
impl Body {
    /// The DMA of one tile (A issues, B issues, the 2 scale loads); `range` selects which of them to issue now (the deferred form spreads them
    /// over the MMA rows).
    fn stage_now(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotIdx, range: std::ops::Range<usize>, cache: &mut StageCache) {
        // diagnostic ablations (wrong results): FMHA_ABLATE=gemm_nodma skips the copies
        if self.cfg.ablate.no_dma {
            return;
        }
        // A advances by key0 bytes, the preshuffled B by (key0 / 16) 256 = 16 key0
        // the tile origins are wave uniform: SGPR soffsets
        // computed once per stage (the deferred form issues the stage in chunks)
        let (ka, kb) = *cache.org.get_or_insert_with(|| {
            let ka = { let v = b.bind_i32(key0.clone() + ev(self.a_org)); b.read_first_lane(v) };
            let kb = { let v = b.bind_i32((key0.clone() << 4) + ev(self.b_org)); b.read_first_lane(v) };
            (ka, kb)
        });
        let mut unit = 0usize;
        for (bi, (sh, dma, org, rsrc, wave)) in [(&self.a_s, &self.a_dma, ka, self.rsrc_a, self.a_wave), (&self.b_s, &self.b_dma, kb, self.rsrc_b, self.b_wave)].into_iter().enumerate() {
            let mine: Vec<(ValueId, i32, Option<ValueId>)> =
                dma.issues.iter().zip(dma.delta.iter()).enumerate().filter(|(q, _)| range.contains(&(unit + q))).map(|(_, (x, d))| (x.0, x.1, *d)).collect();
            unit += dma.issues.len();
            if mine.is_empty() {
                continue;
            }
            let r = sh.at(slot);
            if cache.sbase[bi].is_none() {
                let o = r.off_v(b);
                cache.sbase[bi] = Some(b.bind_i32(o + sh.base as i32 + ev(wave)));
            }
            let sbase = cache.sbase[bi].unwrap();
            let mut m0s: Vec<(i32, ValueId)> = Vec::new();
            for (_, lds, _) in mine.iter() {
                let grp = *lds - *lds % 4096;
                let key = (bi, grp);
                if !cache.m0.iter().any(|x| x.0 == key) {
                    let v = b.bind_i32(ev(sbase) + grp);
                    let v = b.read_first_lane(v);
                    cache.m0.push((key, v));
                }
                m0s.push((grp, cache.m0.iter().find(|x| x.0 == key).unwrap().1));
            }
            r.write(b, |b| {
                // buffer_load_dwordx4 ... lds: voffset = the lane's loop-invariant offset, soffset = the tile origin (wave uniform), M0 = the first 1 KB
                // block of each group of 4 issues (SGPR, once per stage), instruction offset 1024 (q % 4) moves both the LDS and the global address
                for ((g, lds, d), (_, m0)) in mine.iter().zip(m0s.iter()) {
                    // shared voffset: the issue's wave-uniform source delta goes into the soffset (one scalar add)
                    let so = match d {
                        Some(d) => {
                            let v = b.bind_i32(ev(org) + ev(*d));
                            b.read_first_lane(v)
                        }
                        None => org,
                    };
                    b.load_buffer_lds(rsrc, *g, so, *m0, (*lds % 4096) as u32, 4);
                }
            });
        }
        if self.panel.is_some() {
            return; // the scales come from the panel
        }
        // scales of tile kb: the CTA's 256 A-scale bytes (contiguous in [K/128][M]) and the dword holding the B-scale byte, into the wave's 512 bytes
        if !range.contains(&unit) {
            return;
        }
        let kbi = b.bind_i32(key0.clone().lshr(7));
        let r = self.sc_s.at(slot);
        let o = r.off_v(b);
        let sbase = b.bind_i32(o + self.sc_s.base as i32 + ev(self.sc_wave));
        // MUBUF: A scales voffset = the lane's 4 bytes of the CTA rows, soffset = kb M; B scale: every lane the same dword, soffset = its offset
        let soa = { let v = b.bind_i32(ev(kbi) * ev(self.m_v)); b.read_first_lane(v) };
        let sob = { let v = b.bind_i32((ev(self.sb_src) + ev(kbi)) & -4); b.read_first_lane(v) };
        let m0a = b.read_first_lane(sbase);
        let m0b = { let v = b.bind_i32(ev(sbase) + 256); b.read_first_lane(v) };
        let z = b.c_i32(0);
        r.write(b, |b| {
            b.buffer_load_lds_b32(self.rsrc_sfa, self.sa_src, soa, m0a, 0);
            b.buffer_load_lds_b32(self.rsrc_sfb, z, sob, m0b, 0);
        });
    }
}

/// Per-stage values shared by the chunks of a deferred stage: SGPR tile origins, slot bases, M0 per (buffer, 4 KB group).
#[derive(Default, Clone)]
struct StageCache {
    org: Option<(ValueId, ValueId)>,
    sbase: [Option<ValueId>; 2],
    m0: Vec<((usize, i32), ValueId)>,
}

impl PipeBody for Body {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &I32, slot: &SlotIdx) {
        if self.cfg.defer_dma && self.in_loop {
            // issued inside the window, spread over the MMA rows
            self.deferred = Some((key0.clone(), slot.clone(), 0, StageCache::default()));
            return;
        }
        self.stage_now(b, key0, slot, 0..usize::MAX, &mut StageCache::default());
    }
    fn prologue_stores(&mut self, b: &mut FnBuilder, _role: Role) {
        let Some(p) = &self.panel else { return };
        let Some(job) = p.job else { return };
        // the panel DMA was issued before the two tile stages: wait for it only (in-order VMEM returns)
        let after = 2 * (self.a_dma.issues.len() + self.b_dma.issues.len());
        b.wait_vm(after.min(63) as u8);
        // publish the panel (the protocol is cross-wave), gather every round, one barrier, write every round (the pipeline prologue barrier
        // publishes the result): 2 barriers in all, the tiles' DMA keeps running behind
        b.barrier();
        let pr = p.sh.at(&SlotIdx::new(0, 0));
        let mut packed: Vec<(ValueId, ValueId)> = Vec::new();
        for round in 0..(PANEL_KB / 8) as i32 {
            let words: Vec<ValueId> = pr.read(b, |b| (0..8).map(|i| { let a = b.bind_i32(ev(job.rd0) + (2048 * round + 16 * i)); b.load_lds_b32(a) }).collect());
            let bs: Vec<ValueId> = words.iter().map(|w| b.bind_i32(ev(*w).lshr(ev(job.sh_r)) & 255)).collect();
            let d0 = b.bind_i32(ev(bs[0]) | (ev(bs[1]) << 8) | (ev(bs[2]) << 16) | (ev(bs[3]) << 24));
            let d1 = b.bind_i32(ev(bs[4]) | (ev(bs[5]) << 8) | (ev(bs[6]) << 16) | (ev(bs[7]) << 24));
            packed.push((d0, d1));
        }
        b.wait_lds(0);
        b.barrier();
        pr.write(b, |b| {
            for (round, (d0, d1)) in packed.iter().enumerate() {
                let a0 = b.bind_i32(ev(job.wr0) + 2048 * round as i32);
                let a1 = b.bind_i32(ev(job.wr0) + 2048 * round as i32 + 4);
                b.store_lds_b32(a0, *d0);
                b.store_lds_b32(a1, *d1);
            }
        });
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
    fn pre_sync(&mut self, b: &mut FnBuilder, _role: Role, _t: &TileIt) {
        if !self.cfg.reg_staged {
            return;
        }
        // the first MMAs of the register tile run while the wave waits for the DMA and the barrier (the vendor issues 4 there)
        let npre = self.cfg.pre_sync_mmas;
        let (bv, av, sav, sbv) = self.rv.clone().expect("state bound");
        let nn = bv.len();
        let half = nn / 2;
        let mut n = 0;
        'o: for i in 0..av.len() {
            for j in 0..half {
                if n == npre {
                    break 'o;
                }
                let (w, by) = self.sa_sel(i);
                mma_var(b, &self.mma, self.acc[nn * i + j], av[i], bv[j], sav[w], sbv, by);
                n += 1;
            }
        }
        self.npre_done = n;
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        if self.cfg.reg_staged {
            self.compute_tile(b, t);
        }
    }
    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (nm, nn) = ((self.cfg.wave_m() / 16) as usize, (self.cfg.wave_n() / 16) as usize);
        let agpr = self.cfg.agpr_ops;
        let nords = self.cfg.ablate.no_lds_reads;
        let zf = if nords { let z = b.frag_zero(DType::Frag4); Some(b.cat_frag8(z, z)) } else { None };
        let zfb = if nords { let z = b.frag_zero(DType::Frag4); let o = b.frag_zero(DType::Frag4); Some(b.cat_frag8(o, z)) } else { None };
        let mark = |b: &mut FnBuilder, v: ValueId| if agpr { b.agpr(v) } else { v };
        let kbx = ox(t.it);
        // scale words (one LDS round trip), then the byte selects (shifted to byte 0) after the first fragment reads
        let tr = self.panel.as_ref().and_then(|p| p.tr.map(|o| (p.sh.base, p.b_base, p.b_off, o, p.sh.at(&SlotIdx::new(0, 0)))));
        let (wa, wb, sbsh) = if let Some((base, b_base, b_off, lo, ref pr)) = tr {
            // transposed panel: 2 dwords hold the 8 A-scale bytes of this lane; the B byte as before
            let sab = b.bind_i32((kbx.clone() << 8) + (base as i32) + ev(lo));
            let bo = b.bind_i32(ev(b_off) + kbx.clone());
            let sbb = b.bind_i32((ev(bo).lshr(2) << 2) + (base + b_base) as i32);
            let (wa, wb) = pr.read(b, |b| {
                // the 8 A-scale bytes of this lane: one ds_read_b64
                let w2 = b.load_lds_b64(sab);
                let wa = vec![b.frag2_lane(w2, 0), b.frag2_lane(w2, 1)];
                (wa, b.load_lds_b32(sbb))
            });
            (wa, wb, (ev(bo) & 3) << 3)
        } else {
            self.scale_words(b, &t.cur, kbx.clone())
        };
        let mut deferred = self.deferred.take();
        let ar = self.a_rd.at(b, &self.a_s.at(&t.cur));
        let br = self.b_rd.at(b, &self.b_s.at(&t.cur));
        // A0 and B0 first; the other B fragments are issued one ahead of the first row of MMAs (partial lgkm waits)
        let mut af = { let v = if let Some(z) = zf { z } else { ar.read(b, 0) }; mark(b, v) };
        let bpre = self.cfg.b_prefetch.max(1).min(nn);
        let mut bf: Vec<ValueId> = (0..bpre).map(|j| { let v = if let Some(z) = zfb { z } else { br.read(b, j) }; mark(b, v) }).collect();
        // A scale of m block i: (word, byte select)
        let sa: Vec<(ValueId, u8)> = if tr.is_some() {
            (0..nm).map(|i| (wa[i / 4], (i % 4) as u8)).collect()
        } else {
            (0..nm).map(|i| (b.bind_i32(ev(wa[i]).lshr(ev(self.sa_sh[0]))), 0u8)).collect()
        };
        let sb = b.bind_i32(ev(wb).lshr(sbsh));
        let prio = self.cfg.setprio;
        if prio {
            b.set_prio(1);
        }
        for i in 0..nm {
            // prefetch the next A fragment while this one feeds 4 MMAs
            let next = if i + 1 < nm { let v = if let Some(z) = zf { z } else { ar.read(b, i + 1) }; Some(mark(b, v)) } else { None };
            for j in 0..nn {
                // the next tile's DMA: one issue every `gap` MMAs (FMHA_ABLATE=dmagapN), from the first row on
                let gap = self.cfg.dma_gap.max(1);
                if (nn * i + j) % gap == 0 {
                    if let Some((k, s, done, mut cache)) = deferred.take() {
                        let total = self.a_dma.issues.len() + self.b_dma.issues.len() + 1;
                        self.stage_now(b, &k, &s, done..done + 1, &mut cache);
                        if done + 1 < total {
                            deferred = Some((k, s, done + 1, cache));
                        }
                    }
                }
                if i == 0 && j + bpre < nn {
                    let v = if let Some(z) = zfb { z } else { br.read(b, j + bpre) };
                    bf.push(mark(b, v));
                }
                if !self.cfg.ablate.no_mma {
                    gemm(b, &self.mma, self.acc[nn * i + j], af, bf[j], sa[i].0, sb, sa[i].1, 0);
                }
            }
            if let Some(n) = next {
                af = n;
            }
        }
        // whatever is left of the stage (when gap x issues exceeds the MMAs)
        if let Some((k, s, done, mut cache)) = deferred.take() {
            self.stage_now(b, &k, &s, done..usize::MAX, &mut cache);
        }
        if prio {
            b.set_prio(0);
        }
        // scheduler-side interleave (CompileProfile::Gemm): FMHA_ABLATE=phase[mN][rM][dK]: N MFMAs, then K ds_reads, every M-th step one DMA issue
        if let Some(spec) = std::env::var("FMHA_ABLATE").ok().and_then(|v| v.split(',').find(|t| t.starts_with("phase")).map(|t| t.to_string())) {
            let num = |c: char, d: u32| spec.split(c).nth(1).and_then(|s| s.chars().take_while(|x| x.is_ascii_digit()).collect::<String>().parse::<u32>().ok()).unwrap_or(d);
            let (nmf, every, nds) = (num('m', 2), num('r', 2), num('d', 1));
            let steps = (nm * nn) as u32 / nmf;
            for s in 0..steps {
                b.sched_group_barrier(amdgpu_dsl::SCHED_MFMA, nmf, 0);
                b.sched_group_barrier(amdgpu_dsl::SCHED_DS_READ, nds, 0);
                if s % every == 0 {
                    b.sched_group_barrier(amdgpu_dsl::SCHED_VMEM_READ, 1, 0);
                }
            }
        }
    }
}

pub fn gemm_mxfp8_bs_v2() -> Function {
    gemm_mxfp8_bs("gemm_mxfp8_bs_v2", GemmConfig::v2())
}
pub fn gemm_mxfp8_bs_v3() -> Function {
    gemm_mxfp8_bs("gemm_mxfp8_bs_v3", GemmConfig::v3())
}
pub fn gemm_mxfp8_bs_v4() -> Function {
    gemm_mxfp8_bs("gemm_mxfp8_bs_v4", GemmConfig::v4())
}

/// The registered names: the variants and the v3 ablations (`_nodma`, `_nomma`, `_nords`, `_nodma_nords`: timing diagnostics, wrong results).
pub fn by_name(name: &str) -> Option<Function> {
    let abl = |a: Ablate| GemmConfig { ablate: a, ..GemmConfig::v3() };
    let cfg = match name {
        "gemm_mxfp8_bs_v2" => GemmConfig::v2(),
        "gemm_mxfp8_bs_v3" => GemmConfig::v3(),
        "gemm_mxfp8_bs_v4" => GemmConfig::v4(),
        // experiments
        "gemm_mxfp8_bs_v3_vshare" => GemmConfig { shared_voffsets: true, ..GemmConfig::v3() },
        "gemm_mxfp8_bs_v4_vshare" => GemmConfig { shared_voffsets: true, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_acc32" => GemmConfig { acc_agpr: 32, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_pre0" => GemmConfig { pre_sync_mmas: 0, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_pre8" => GemmConfig { pre_sync_mmas: 8, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_gap2" => GemmConfig { dma_gap: 2, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_late" => GemmConfig { early_scales: false, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_noxcd" => GemmConfig { xcd_raster: false, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_u1" => GemmConfig { unroll2: false, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_tall" => GemmConfig { raster_tall: true, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_prio" => GemmConfig { setprio: true, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_gap4" => GemmConfig { dma_gap: 4, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_acc32h" => GemmConfig { acc_agpr: 32, sched: "rewait,hoistchain", ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_pre0h" => GemmConfig { pre_sync_mmas: 0, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v4_pre8h" => GemmConfig { pre_sync_mmas: 8, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v3_bpre4" => GemmConfig { b_prefetch: 4, ..GemmConfig::v3() },
        "gemm_mxfp8_bs_v3_bpre6" => GemmConfig { b_prefetch: 6, ..GemmConfig::v3() },
        "gemm_mxfp8_bs_v3_prio" => GemmConfig { setprio: true, ..GemmConfig::v3() },
        "gemm_mxfp8_bs_v3_gap3" => GemmConfig { dma_gap: 3, ..GemmConfig::v3() },
        "gemm_mxfp8_bs_v3_directc" => GemmConfig { lds_epilogue: false, ..GemmConfig::v3() },
        "gemm_mxfp8_bs_v4_directc" => GemmConfig { lds_epilogue: false, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v3_gap4" => GemmConfig { dma_gap: 4, ..GemmConfig::v3() },
        "gemm_mxfp8_bs_v3_gap3_bpre4" => GemmConfig { dma_gap: 3, b_prefetch: 4, ..GemmConfig::v3() },
        "gemm_mxfp8_bs_v4_gap3" => GemmConfig { dma_gap: 3, ..GemmConfig::v4() },
        "gemm_mxfp8_bs_v3_nodma" => abl(Ablate { no_dma: true, ..Ablate::default() }),
        "gemm_mxfp8_bs_v3_nomma" => abl(Ablate { no_mma: true, ..Ablate::default() }),
        "gemm_mxfp8_bs_v3_nords" => abl(Ablate { no_lds_reads: true, ..Ablate::default() }),
        "gemm_mxfp8_bs_v3_nodma_nords" => abl(Ablate { no_dma: true, no_lds_reads: true, ..Ablate::default() }),
        _ => return None,
    };
    Some(gemm_mxfp8_bs(name, cfg))
}

/// Builds the GEMM for `cfg` under `name`.
pub fn gemm_mxfp8_bs(name: &str, cfg: GemmConfig) -> Function {
    build(name, cfg)
}
fn build(name: &str, cfg: Cfg) -> Function {
    let (cta_m, cta_n, wave_m, wave_n) = (cfg.cta_m, cfg.cta_n, cfg.wave_m(), cfg.wave_n());
    let mut fb = FnBuilder::new(name, 256);
    // compile policy: lowering `lgkmn,zeroinit` + the scoreboard wait pass (partial waits across the loop back edge, ds_read hoisting); nothing name-keyed
    fb.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
    if !cfg.sched.is_empty() {
        fb.sched_opts(cfg.sched);
    }
    let b = &mut fb;
    declare_abi(b);
    declare_contract(b, cta_m, cta_n);
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
    let bx0 = b.let_(Op::WorkgroupIdX); // N tile
    let by0 = b.let_(Op::WorkgroupIdY); // M tile
    // XCD-aware raster (FMHA_ABLATE=xcd, experiment): workgroups are dispatched round robin over the 8 XCDs (linear id % 8), each with its own L2;
    // give every XCD whole 4 x 4 blocks of tiles (4 A + 4 B panels instead of all A panels). Only for a 16 x 16 tile grid here; identity otherwise.
    let (bx, by) = if cfg.xcd_raster {
        let okv = b.bind_i32((ev(n_v) ^ (16 * cfg.cta_n) as i32) | (ev(m_v) ^ (16 * cfg.cta_m) as i32));
        let id = b.bind_i32(ev(bx0) + (ev(by0) << 4));
        let loc = b.bind_i32(ev(id).lshr(3));
        // block (of 16) = 2 (id % 8) + loc / 16 -> (row, col) = (block / 4, block % 4); inside the block: loc % 16 -> (n4, m4)
        let blk = b.bind_i32(((ev(id) & 7) << 1) + ev(loc).lshr(4));
        // raster_tall: an XCD's two blocks stacked (8 x 4 tiles) instead of side by side (4 x 8)
        let (bc, br) = if cfg.raster_tall { (ev(blk).lshr(2), ev(blk) & 3) } else { (ev(blk) & 3, ev(blk).lshr(2)) };
        let nx = b.bind_i32((bc << 2) + (ev(loc) & 3));
        let ny = b.bind_i32((br << 2) + (ev(loc).lshr(2) & 3));
        let s = b.bind_i32(ev(okv).eq(0).select_i32(1, 0));
        let px = b.bind_i32(ev(bx0) + ev(s) * (ev(nx) - ev(bx0)));
        let py = b.bind_i32(ev(by0) + ev(s) * (ev(ny) - ev(by0)));
        b.assume_trusted(px, Ext::c(0), Ext::sym("NT") - Ext::c(1), "XCD raster: (px, py) is a permutation of the 16 x 16 tile grid when the select applies (N = 16 cta_n, M = 16 cta_m: 4 x 4 blocks, 2 per XCD), the identity (workgroup ids) otherwise");
        b.assume_trusted(py, Ext::c(0), Ext::sym("MT") - Ext::c(1), "XCD raster: see px");
        (px, py)
    } else {
        (bx0, by0)
    };
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
    // scale panel: A [PANEL_KB][256 rows] + B [2 blocks][PANEL_KB] (+ 4 bytes of source alignment slack, padded to 256 for the dword DMA)
    let panel_sh = if cfg.scale_panel { Some(lds.tile("SCP", PANEL_KB * 256 + 512, amdgpu_dsl::tile::Layout::Linear)) } else { None };
    // epilogue staging (wave-private, reusing the dead ring / panel storage): the wave's C block, bf16 rows of EPI_PITCH bytes
    let epi = if cfg.lds_epilogue {
        let mut ov: Vec<&Shared> = vec![&a_s, &b_s, &sc_s];
        if let Some(p) = &panel_sh {
            ov.push(p);
        }
        let per_wave = wave_m * EPI_PITCH;
        assert!(4 * per_wave <= 160 * 1024 - 16, "epilogue staging does not fit the LDS");
        Some(lds.private("EPI", per_wave, 4, &ov))
    } else {
        None
    };
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
    // the view bases lie inside one 16-row block (< 2048 bytes): the mask is a no-op that gives the verifier the bound the XOR swizzle hides
    // (bases include the buffer base; 16-byte reads: the low 4 bits are zero)
    let (ab, bb) = (a_s.base as i32, b_s.base as i32);
    let _ = (ab, bb);
    a_rd.bases = a_rd.bases.iter().map(|x| b.let_i32(x + &a_wo)).collect();
    b_rd.bases = b_rd.bases.iter().map(|x| b.let_i32(x + &b_wo)).collect();

    // ---- DMA (A row-major, B preshuffled), wave parts of M0, CTA origins ----
    let a_glob = make_layout(tup![cta_m, 128], tup![s_k.clone(), 1]);
    let b_glob = make_layout(tup![tup![16, cta_n / 16], tup![16, 8]], tup![tup![16, s_k.mul_c(16)], tup![1, 256]]);
    let a_dma = dma_plan_rows_swz(b, &ctx, &s_tid, cta_m, &a_glob, 8192, cfg.shared_voffsets);
    let b_dma = dma_plan(b, &ctx, &s_tid, cta_n, &b_glob, 8192, cfg.shared_voffsets);
    let a_wave = b.bind_i32(lane.wave() * (cta_m * 128 / 4) as i32);
    let b_wave = b.bind_i32(lane.wave() * (cta_n * 128 / 4) as i32);
    // buffer descriptors over A (M K bytes) and the preshuffled B (N K bytes)
    let rsrc_a = { let n = b.bind_i32(ev(m_v) * ev(k_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_a, n) };
    let rsrc_b = { let n = b.bind_i32(ev(n_v) * ev(k_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_b, n) };
    let rsrc_sfa = { let n = b.bind_i32(ev(m_v) * ev(kb_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfa, n) };
    let rsrc_sfb = { let n = b.bind_i32((ev(n_v) >> 7) * ev(kb_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfb, n) };
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
    // loop-invariant byte-select shifts of the A scales and the dword offsets of their rows
    let sa_sh: Vec<ValueId> = sa_rd.iter().map(|r| b.bind_i32((ev(*r) & 3) << 3)).collect();
    let sa_dw: Vec<ValueId> = sa_rd.iter().map(|r| b.bind_i32(ev(*r).lshr(2) << 2)).collect();

    let cadence = if cfg.reg_staged { Cadence::RegStaged } else if cfg.three_slots { Cadence::OneWindow } else { Cadence::TwoSlot };
    // the panel DMA (before the pipeline prologue, whose wait_vm(0) + barrier publish it): wave w loads K groups 16 w .. 16 w + 15 (zeros past KB:
    // the descriptor bound), and wave 0 the CTA's B-scale rows (2 x KB bytes from the dword-aligned start)
    let panel = panel_sh.map(|sh| {
        assert!(cta_m == 256 && cta_n == 256, "scale panel: 256 x 256 CTA");
        let per_wave = PANEL_KB / 4;
        let pr = sh.at(&SlotIdx::new(0, 0));

        let b_src = b.bind_i32(((ev(bx) << 1) * ev(kb_v)) & -4);
        let b_delta = b.bind_i32(((ev(bx) << 1) * ev(kb_v)) & 3);
        // wave w loads the K groups 8 r + 2 w + h (r < 8, h < 2): exactly the rows its own threads transpose below (thread group tid / 32 = 2 w + h
        // owns K groups 8 round + tid / 32), so the transposition needs no barrier
        let _ = per_wave;
        let m0s: Vec<ValueId> = (0..(PANEL_KB / 8) as i32).map(|r| { let v = b.bind_i32((lane.wave() << 9) + (2048 * r + sh.base as i32)); b.read_first_lane(v) }).collect();
        pr.write(b, |b| {
            for r in 0..(PANEL_KB / 8) as i32 {
                for h in 0..2i32 {
                    // soffset = row (8 r + 2 w + h) M; voffset = the CTA rows + 4 lane (sa_src); M0 + imm = panel + 256 (8 r + 2 w) + 256 h (the
                    // instruction offset also moves the source: compensated in soffset)
                    let so = { let v = b.bind_i32(((lane.wave() << 1) + (8 * r + h)) * ev(m_v) - 256 * h); b.read_first_lane(v) };
                    b.buffer_load_lds_b32(rsrc_sfa, sa_src, so, m0s[r as usize], (256 * h) as u32);
                }
            }
        });
        let w0 = b.bind_cond(lane.wave().eq(0));
        b.begin_if(w0);
        let m0b = b.c_i32((sh.base + PANEL_KB * 256) as i32);
        let zv = b.bind_i32(lane.lane() << 2);
        let sob = b.read_first_lane(b_src);
        pr.write(b, |b| b.buffer_load_lds_b32(rsrc_sfb, zv, sob, m0b, 0));
        b.end_if();
        let b_off = b.bind_i32((ev(wn) * ev(kb_v)) + ev(b_delta));
        // transpose the A part in place, 8 K groups per round: thread (r = tid % 16, wm = tid / 16 % 2, k = tid / 32) gathers the bytes of rows
        // 128 wm + 16 i + r (i < 8) of K group 8 round + k and writes them as 8 consecutive bytes at 128 wm + 8 r
        let tr = if !cfg.panel_transposed { None } else {
            let r = b.bind_i32(ev(tid) & 15);
            let wmq = b.bind_i32((ev(tid).lshr(4)) & 1);
            let kq = b.bind_i32(ev(tid).lshr(5));
            let sh_r = b.bind_i32((ev(r) & 3) << 3);
            let rd0 = b.bind_i32((ev(wmq) << 7) + (ev(r).lshr(2) << 2) + (ev(kq) << 8) + sh.base as i32);
            let wr0 = b.bind_i32((ev(wmq) << 7) + (ev(r) << 3) + (ev(kq) << 8) + sh.base as i32);
            // the rounds run in the pipeline prologue (`prologue_stores`), after the first tiles' DMA is issued
            let job = TrJob { rd0, wr0, sh_r };
            Some((b.bind_i32((ev(wm) << 7) + ((lane.lane() & 15) << 3)), job))
        };
        let (tr, job) = match tr { Some((o, j)) => (Some(o), Some(j)), None => (None, None) };
        Panel { sh, b_base: PANEL_KB * 256, b_off, tr, job }
    });
    let mut pipe = Pipeline::new(cadence, kb_v, k_v, 128, Ext::ka(84));
    pipe.unroll2 = cfg.unroll2;
    if cfg.three_slots || cfg.reg_staged {
        pipe.vm_inflight = (a_dma.issues.len() + b_dma.issues.len() + 2) as u8;
    }
    let zero4 = b.frag_zero(DType::Frag4);
    let mut body = Body {
        cfg, kb_v, panel, mma: mma.clone(), a_s, b_s, sc_s, a_dma, b_dma, a_org, b_org, a_wave, b_wave, sc_wave, rsrc_a, rsrc_b, rsrc_sfa, rsrc_sfb, ptr_sfa, ptr_sfb, m_v, sa_src, sb_src, sa_rd, sa_sh, sa_dw,
        a_rd, b_rd, zero4, acc: Vec::new(), staged: None, rv: None, deferred: None, in_loop: false, npre_done: 0,
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

    if let Some(epi) = &epi {
        lds_epilogue(b, &ctx, &mma, epi, &body.acc, &lane, &s_lane, &s_n, ptr_c, m_v, n_v, bx, by, wm, wn, lg_m, lg_n, wave_m, wave_n);
        return fb.finish();
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

/// The LDS-staged epilogue: every accumulator (16 x 16, lane = row m, 4 consecutive n) is converted to bf16 and written to the wave's staging
/// block E[m][n] (row pitch EPI_PITCH); then each lane reads 16 bytes = 8 consecutive n of one row and stores them with one `buffer_store_dwordx4`
/// (16 lanes cover a 256-byte row segment). All addresses from layouts: the C TV layout of the atom over E, a row-contiguous read TV layout over E
/// and the same TV layout over the global C (voffset per lane, the wave-uniform per-issue row step in the SGPR soffset).
#[allow(clippy::too_many_arguments)]
fn lds_epilogue(
    b: &mut FnBuilder, ctx: &Ctx, mma: &TiledMma, epi: &Shared, acc: &[VarId], lane: &Lane, s_lane: &Ix, s_n: &Ix, ptr_c: ValueId, m_v: ValueId,
    n_v: ValueId, bx: ValueId, by: ValueId, wm: ValueId, wn: ValueId, lg_m: i32, lg_n: i32, wave_m: i64, wave_n: i64,
) {
    let d = &ctx.dyns;
    let (nm, nn) = ((wave_m / 16) as usize, (wave_n / 16) as usize);
    assert_eq!(wave_n * 2, 256, "lds_epilogue: 128-column wave blocks (256-byte rows)");
    // every wave is done with the ring / panel storage the staging reuses
    b.wait_lds(0);
    b.barrier();
    let wid = b.bind_i32(lane.wave());
    let me = epi.mine(wid);
    let base = b.bind_i32(me.base());
    // staging layout over the C block element index m + 16 n of one 16 x 16 block: (m, n) -> m EPI_PITCH + 2 n
    let e_blk = make_layout(tup![16, 16], tup![EPI_PITCH, 2]);
    let w_plan = IssuePlan::new(mma.c_tv(), 4); // 4 bf16 = one ds_write_b64
    let mut exprs = Vec::new();
    for i in 0..nm as i64 {
        for j in 0..nn as i64 {
            exprs.extend(w_plan.offsets(&e_blk, s_lane, &Ix::C(16 * i * EPI_PITCH + 32 * j), d));
        }
    }
    let wp = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::DS, d);
    let wbases: Vec<ValueId> = wp.bases.iter().map(|e| b.bind_i32(ev(base) + ctx.lower(e))).collect();
    me.write(b, 0, |b| {
        for (n, a) in acc.iter().enumerate() {
            let v: Vec<ValueId> = (0..4u8).map(|r| {
                let l = b.frag4_lane(Opnd::Var(*a), r);
                b.bitcast_f32(l)
            }).collect();
            // 4 bf16 = one ds_write_b64
            let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
            let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
            let f2 = b.frag2(w0, w1);
            let (bi, imm) = wp.issues[n];
            let ad = b.bind_i32(ev(wbases[bi]) + imm as i32);
            b.store_lds_b64(ad, f2);
        }
    });
    b.wait_lds(0);
    // read / store TV over the wave block element index n + wave_n m: lane -> (chunk = lane % 16: 8 n, row = lane / 16), value -> (8 n, row step 4)
    let rows_per = 64 / (wave_n / 8); // 4
    let issues = wave_m / rows_per; // 32
    let tv = make_layout(tup![tup![wave_n / 8, rows_per], tup![8, issues]], tup![tup![8, wave_n], tup![1, wave_n * rows_per]]);
    let e_blk2 = make_layout(tup![wave_n, wave_m], tup![2, EPI_PITCH]);
    let rp = IssuePlan::new(tv.clone(), 8);
    let rexprs = rp.offsets(&e_blk2, s_lane, &Ix::C(0), d);
    let rpl = layout_to_offsets(&rexprs, &Swizzle::none(), ImmRange::DS, d);
    let rbases: Vec<ValueId> = rpl.bases.iter().map(|e| b.bind_i32(ev(base) + ctx.lower(e))).collect();
    let vals: Vec<ValueId> = me.read_back(b, 0, |b| {
        rpl.issues.iter().map(|&(bi, imm)| {
            let ad = b.bind_i32(ev(rbases[bi]) + imm as i32);
            b.load_lds_b128(ad)
        }).collect()
    });
    // global: C row-major bf16, the wave block at rows m_cta + wave_m wm, cols n_cta + wave_n wn
    let g_blk = make_layout(tup![wave_n, wave_m], tup![2, s_n.mul_c(2)]);
    let gofs = rp.offsets(&g_blk, s_lane, &Ix::C(0), d);
    let voff = b.bind_i32(ctx.lower(&gofs[0]) + (ev(wm) * (wave_m as i32) * ev(n_v) + ev(wn) * (wave_n as i32)) * 2);
    let rsrc_c = {
        let nr = b.bind_i32(ev(m_v) * ev(n_v) * 2);
        let nr = b.read_first_lane(nr);
        b.buffer_desc_dyn(ptr_c, nr)
    };
    let c_org = { let v = b.bind_i32((((ev(by) << lg_m) * ev(n_v)) + (ev(bx) << lg_n)) << 1); b.read_first_lane(v) };
    for (q, v) in vals.iter().enumerate() {
        // the per-issue step is wave-uniform (rows_per rows): checked on the layout expressions
        let dq = gofs[q].sub(&gofs[0]);
        let dv = b.bind_i32(ev(c_org) + ctx.lower(&dq));
        let so = b.read_first_lane(dv);
        b.buffer_store_b128(rsrc_c, voff, so, 0, *v);
    }
}