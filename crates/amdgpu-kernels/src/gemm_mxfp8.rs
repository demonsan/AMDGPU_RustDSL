//! MXFP8 blockscale GEMM on the route-B layers (P4; target = aiter PR #6696 4-wave pin-AGPR kernel, docs/mxfp8-gemm-4wave-analysis.md), same 96-byte ABI:
//!   C[m, n] (bf16) = sum_k A[m, k] 2^(sa[k/128][m] - 127) * B[n, k] 2^(sb[n/128][k/128] - 127)
//!   A [M, K] fp8 e4m3 row-major; B [N, K] fp8 preshuffled (aiter shuffle_weight 16x16); sa E8M0 [K/128][M]; sb E8M0 [N/128][K/128].
//!
//!
//! The shipped kernel (`gemm_mxfp8_bs_v4`): CTA 256 x 256, 4 waves 2 x 2 of 128 x 128, `v_mfma_scale_f32_16x16x128_f8f6f4` through
//! `TiledMma(swap_ab)`; the vendor's register-staged structure (`Cadence::RegStaged`, u2 unroll): operands in AGPRs reloaded in place from tile t+1,
//! accumulators partly pinned in AGPRs; A in the padded LDS image (1056-byte blocks), B preshuffled, both by MUBUF LDS-DMA with layout-derived
//! voffsets; the scales once per CTA into a transposed panel (VGPR loads + in-register byte transposes); the LDS-staged epilogue; the slot-filling
//! scheduler places the DMA issues.

use amdgpu_dsl::tile::{ev, ox, Cadence, LdsPlan, Lane, PipeBody, Pipeline, Role, Shared, SlotIdx, TileIt};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};
use amdgpu_isa::{MmaAtom, DS_READ_B128X2};
use amdgpu_layout::{composition, layout_to_offsets, make_layout, right_inverse, tup, ComposedLayout, ImmRange, IssuePlan, Ix, Layout, Swizzle};
use amdgpu_tile::{make_tiled_copy_a, make_tiled_copy_b, make_tiled_mma, Ctx, LdsPart, TiledMma};

/// Slots per wave record of the optional timing stamps (`GEMM_STAMP`): 0 kernel start, 1 setup done, 2 prologue done, 3 loop done, 4 epilogue stores issued, 16 + tile = the tile matrix phase.
const STAMP_SLOTS: usize = 96;


/// Build configuration of the shipped MXFP8 blockscale GEMM (`GemmConfig::v4`).
#[derive(Clone, Copy, Debug)]
pub struct GemmConfig {
    pub cta_m: i64,
    pub cta_n: i64,
    /// accumulators (16x16 blocks, m-block-major) pinned in AGPRs: the first `acc_agpr`
    pub acc_agpr: usize,
    /// MMAs of the register tile issued before the wait + barrier of the iteration
    pub pre_sync_mmas: usize,
    /// MMAs of the register tile issued between the top-of-tile waits and the barrier (after `pre_sync_mmas`)
    pub mid_sync_mmas: usize,
    /// LDS epilogue in this many row phases (conversion / LDS writes of the next phase overlap the stores of this one)
    pub epi_phases: usize,
    /// extra scheduler tokens (Function::sched_opts, on top of the Gemm profile's wait pass)
    pub sched: &'static str,
    /// epilogue through LDS (bf16 C block transposed in a wave-private staging area, 16-byte row stores; the shipped form). `false` = the direct-C
    /// epilogue (8-byte stores from the accumulator layout), used by the E3 IR proof (tests/functional_gemm_ir.rs)
    pub lds_epilogue: bool,
}

impl GemmConfig {
    /// The shipped configuration.
    pub fn v4() -> GemmConfig {
        GemmConfig { cta_m: 256, cta_n: 256, acc_agpr: 24, pre_sync_mmas: 5, mid_sync_mmas: 3, epi_phases: 8, sched: "rewait,slotfill", lds_epilogue: true }
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

/// One DMA stage of a [rows x 128 B] tile: per issue the lane's voffset (bound once, loop invariant) and the LDS offset of its 1 KB block.
struct Dma {
    issues: Vec<(ValueId, i32)>,
    /// LDS bytes per 1 KB issue block (1024, or 1056 for the padded A image); the issue LDS offsets above are in 1 KB units (`act`)
    pitch: i32,
}

impl Dma {
    /// (M0 offset of the issue's group of 4, instruction offset) for the virtual 1 KB offset `lds`
    fn act(&self, lds: i32) -> (i32, u32) {
        ((lds - lds % 4096) / 1024 * self.pitch, ((lds % 4096) / 1024 * self.pitch) as u32)
    }
}

/// Turns the per-issue source offsets (minus the instruction offset `pitch (q % 4)`) into voffsets, lowered through the GF(2) bit-form normal form
/// (amdgpu_layout::BitForm: shift groups, CSE across plans by Ctx::bind).
fn finish_dma_p(b: &mut FnBuilder, ctx: &Ctx, gofs: Vec<Ix>, pitch: i32) -> Dma {
    let adj: Vec<Ix> = gofs.iter().enumerate().map(|(q, g)| g.add_c(-(pitch as i64) * (q as i64 % 4))).collect();
    ctx.set_bitfield(true);
    let issues = adj.iter().enumerate().map(|(q, g)| { let e = ctx.bind(b, g); (b.bind_i32(e), 1024 * q as i32) }).collect();
    ctx.set_bitfield(false);
    Dma { issues, pitch }
}
/// LDS byte `d = 1024 (issues (tid / 64) + q) + 16 (tid % 64)` holds tile element `smem^-1(d)`, so the source offset is `glob o right_inverse(smem)`;
/// proved here for sampled threads against a direct search of the LDS layout.
fn dma_plan(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64, waves: i64) -> Dma {
    let issues = rows * 128 / 1024 / waves;
    let sm = smem_tile(rows);
    let tv = make_layout(tup![tup![64, waves], tup![16, issues]], tup![tup![16, 1024 * issues], tup![1, 1024]]);
    let g = composition(glob, &right_inverse(&sm));
    let gofs = IssuePlan::new(tv, 16).offsets(&g, s_tid, &Ix::C(0), &ctx.dyns);
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    for th in [0i64, 1, 15, 16, 63, 64, 100, 127, 255].into_iter().filter(|t| *t < 64 * waves) {
        let env = |s: u32| if s == tid_id { th } else { proof_k };
        for q in 0..issues {
            let lds = 1024 * (issues * (th / 64) + q) + 16 * (th % 64);
            let (r, c) = (0..rows).flat_map(|r| (0..8).map(move |c| (r, 16 * c))).find(|&(r, c)| sm.at(r + rows * c) == lds).expect("LDS byte covered");
            let want = glob.crd2idx(&tup![r, c], &ctx.dyns).eval(&env);
            assert_eq!(gofs[q as usize].eval(&env), want, "DMA rows {rows} thread {th} issue {q}");
        }
    }
    // MUBUF form: voffset (loop invariant) + soffset + the instruction offset 1024 (q % 4) = the global offset; M0 = the group's first 1 KB block
    finish_dma_p(b, ctx, gofs, 1024)
}

/// A-tile DMA for the PADDED row-major LDS image (the vendor form): 1 KB blocks of 8 whole 128-byte rows at a pitch of 1056 bytes (32 bytes of
/// padding per block spread the banks of the fragment reads), lanes linear in the source (lane -> row lane / 8, bytes 16 (lane % 8)): fully
/// monotone 128-byte segments, no source-side permutation. LDS image: (r, k) -> 1056 (r / 8) + 128 (r % 8) + k.
fn dma_plan_rows_pad(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64, waves: i64) -> Dma {
    let issues = rows * 128 / 1024 / waves;
    let d = &ctx.dyns;
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    let mut out = Vec::new();
    for q in 0..issues {
        // block index (issues (tid / 64) + q), row inside the block lane / 8, bytes 16 (lane % 8)
        let blk = s_tid.div_c(64, d).mul_c(issues).add_c(q);
        let lane = s_tid.mod_c(64, d);
        let r = blk.mul_c(8).add(&lane.div_c(8, d));
        let k = lane.mod_c(8, d).mul_c(16);
        let g = glob.crd2idx(&tup![r.clone(), k.clone()], d);
        for th in [0i64, 1, 7, 8, 63, 64, 100, 127, 255].into_iter().filter(|t| *t < 64 * waves) {
            let env = |s: u32| if s == tid_id { th } else { proof_k };
            let (bb, ll) = (issues * (th / 64) + q, th % 64);
            let want = glob.crd2idx(&tup![8 * bb + ll / 8, 16 * (ll % 8)], d).eval(&env);
            assert_eq!(g.eval(&env), want, "padded A DMA thread {th} issue {q}");
        }
        out.push(g);
    }
    finish_dma_p(b, ctx, out, 1056)
}

/// The scale panel: every K group's scales of the CTA in LDS, written once in the prologue (`prologue_stores`).
struct Panel {
    sh: Shared,
    /// LDS offset of the B part inside the panel
    b_base: i64,
    /// A part transposed: [kb][wm][lane % 16][m block i] (8 bytes per lane: 2 dwords, MFMA byte selects); this lane's offset
    tr: ValueId,
    /// B part expanded to dwords [wn][PANEL_KB] (byte 0 = the scale): this wave's dword offset (wn 4 PANEL_KB)
    b_dw: ValueId,
    /// the VGPR scale path emitted in `prologue_stores`
    sfa: SfaJob,
}

/// The VGPR scale path: A scales by 4 passes of buffer_load_dwordx4 (lane = 16 M rows of one K group), a 4x4 byte transpose per dword across each
/// lane quad (dpp quad_perm [1,0,3,2] + v_perm, dpp quad_perm [2,3,0,1] + v_perm) straight into the transposed panel layout, one ds_write per dword;
/// B scales by one dword load per thread. Layout proof: `sfa_transpose_proof`.
struct SfaJob {
    voff: ValueId,
    pso: Vec<ValueId>,
    sel1: ValueId,
    sel2: ValueId,
    dst: ValueId,
    bvo: ValueId,
    bsh: ValueId,
    bdst: ValueId,
}

/// Build-time proof of the VGPR scale path: simulates every lane of every pass byte by byte (the global load, both dpp/perm steps) and checks that
/// the byte written at panel offset 256 kb + 128 wm + 8 r + i is sfa[kb][128 wm + 16 i + r] (the transposed layout `scale_issue` reads).
/// Public wrapper for the VGPR/DPP scale-panel proof (item E1 calls it explicitly; the v4 builder
/// calls it build-time through `sfa_transpose_proof`).
pub fn verify_sfa_transpose() { sfa_transpose_proof() }

fn sfa_transpose_proof() {
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

struct Body {
    cfg: Cfg,
    /// K / 128 (kernarg)
    kb_v: ValueId,
    panel: Panel,
    mma: TiledMma,
    a_s: Shared,
    b_s: Shared,
    a_dma: Dma,
    b_dma: Dma,
    a_org: ValueId,
    b_org: ValueId,
    a_wave: ValueId,
    b_wave: ValueId,
    rsrc_a: ValueId,
    rsrc_b: ValueId,
    rsrc_sfa: ValueId,
    rsrc_sfb: ValueId,
    a_rd: LdsPart,
    b_rd: LdsPart,
    zero4: ValueId,
    acc: Vec<VarId>,
    /// MMAs of the register tile already issued by `pre_sync` / `post_wait`
    npre_done: usize,
    /// timing stamps (GEMM_STAMP=ends|tiles): debug buffer, this wave's record offset, per-tile stamps
    stamp: Option<(ValueId, ValueId, bool)>,
    /// register-staged tile: initial values (tile 0) and the loop variables: B fragments, A fragments, A scales, B scale
    staged: Option<(Vec<ValueId>, Vec<ValueId>, Vec<ValueId>, ValueId)>,
    rv: Option<(Vec<VarId>, Vec<VarId>, Vec<VarId>, VarId)>,
}

impl Body {
    /// The scale words of tile `kb` from the transposed panel: the 8 A-scale bytes of this lane (one ds_read_b64; the MMA selects byte i % 4 of
    /// word i / 4: `sa_sel`) and the B scale (the dword-expanded panel: byte 0, no shift).
    fn load_scales(&self, b: &mut FnBuilder, kb: I32) -> (Vec<ValueId>, ValueId) {
        // K group of a real tile (t + 1 is read only when it exists): the clamp is a no-op that bounds it for the verifier's loop-exit-blind intervals
        let kb = b.bind_i32(kb.lt(ev(self.kb_v)).select_i32(kb.clone(), ev(self.kb_v) - 1));
        let kb = ev(kb);
        let p = &self.panel;
        let (base, pr) = (p.sh.base, p.sh.at(&SlotIdx::new(0, 0)));
        let sab = b.bind_i32((kb.clone() << 8) + (base as i32) + ev(p.tr));
        let sbb = b.bind_i32((kb << 2) + ev(p.b_dw) + (base + p.b_base) as i32);
        pr.read(b, |b| {
            let w2 = b.load_lds_b64(sab);
            let wa = vec![b.frag2_lane(w2, 0), b.frag2_lane(w2, 1)];
            (wa, b.load_lds_b32(sbb))
        })
    }
    /// The (word, byte) of the A scale of m block i in the `load_scales` words.
    fn sa_sel(&self, i: usize) -> (usize, u8) {
        (i / 4, (i % 4) as u8)
    }
    /// An operand fragment read into AGPRs.
    fn frag(&self, b: &mut FnBuilder, part: &amdgpu_tile::LdsSlotPart, o: usize) -> ValueId {
        let v = part.read(b, o);
        b.agpr(v)
    }
    /// The whole tile `t` into registers (prologue: tile 0).
    fn load_tile(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let (nm, nn) = ((self.cfg.wave_m() / 16) as usize, (self.cfg.wave_n() / 16) as usize);
        let (sa, sb) = self.load_scales(b, ox(t.it));
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
        let next = if t.staged { Some((self.a_rd.at(b, &self.a_s.at(&t.next)), self.b_rd.at(b, &self.b_s.at(&t.next)))) } else { None };
        // the scales of t+1 are read FIRST (before the fragment reloads queue up behind them) and installed at the end (partial lgkm waits)
        let early = if t.staged { Some(self.load_scales(b, ox(t.it) + 1)) } else { None };
        let half = nn / 2;
        // E3 proof hook for the reg-staged compute path: a real IR branch drops one tile's MMAs, reproducing the historical "tile 1 swallowed"
        // regression in emitted kernel code (tests/functional_gemm_ir.rs).
        let proof_drop = std::env::var("GEMM_PROOF_DROP_KB").ok().and_then(|v| v.parse::<i32>().ok());
        if let Some(kb) = proof_drop {
            let keep = b.bind_cond(ox(t.it).ne(kb));
            b.begin_if(keep);
        }
        for (lo, hi) in [(0, half), (half, nn)] {
            for i in 0..nm {
                for j in lo..hi {
                    if lo == 0 && i * half + j < self.npre_done {
                        continue; // issued before the barrier (pre_sync / post_wait)
                    }
                    let (w, by) = self.sa_sel(i);
                    mma_var(b, &self.mma, self.acc[nn * i + j], av[i], bv[j], sav[w], sbv, by);
                }
                if lo == half {
                    if let Some((ar, _)) = &next {
                        // in-place reload: the ds_reads write the variable's own AGPRs (coalesced set_var)
                        let v = ar.read(b, i);
                        b.set_var(av[i], v);
                    }
                }
            }
            if let Some((_, br)) = &next {
                for j in lo..hi {
                    let v = br.read(b, j);
                    b.set_var(bv[j], v);
                }
            }
        }
        if proof_drop.is_some() {
            b.end_if();
        }
        if let Some((sa, sb)) = early {
            for (i, v) in sa.iter().enumerate() {
                b.set_var(sav[i], *v);
            }
            b.set_var(sbv, sb);
        }
    }
    /// The DMA of one tile (A issues, then B issues): soffset = the wave-uniform tile origin, M0 = the first 1 KB block of each group of 4 issues.
    fn stage_now(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotIdx) {
        // A advances by key0 bytes, the preshuffled B by (key0 / 16) 256 = 16 key0; the tile origins are wave uniform: SGPR soffsets
        let ka = { let v = b.bind_i32(key0.clone() + ev(self.a_org)); b.read_first_lane(v) };
        let kb = { let v = b.bind_i32((key0.clone() << 4) + ev(self.b_org)); b.read_first_lane(v) };
        let bufs = [(&self.a_s, &self.a_dma, ka, self.rsrc_a, self.a_wave), (&self.b_s, &self.b_dma, kb, self.rsrc_b, self.b_wave)];
        for (sh, dma, org, rsrc, wave) in bufs {
            let r = sh.at(slot);
            let o = r.off_v(b);
            let sbase = b.bind_i32(o + sh.base as i32 + ev(wave));
            let mut m0s: Vec<(i32, ValueId)> = Vec::new();
            let mut mine: Vec<(ValueId, i32, ValueId)> = Vec::new();
            for (g, lds) in dma.issues.iter() {
                let grp = *lds - *lds % 4096;
                if !m0s.iter().any(|x| x.0 == grp) {
                    let v = b.bind_i32(ev(sbase) + dma.act(*lds).0);
                    let v = b.read_first_lane(v);
                    m0s.push((grp, v));
                }
                mine.push((*g, *lds, m0s.iter().find(|x| x.0 == grp).unwrap().1));
            }
            r.write(b, |b| {
                // buffer_load_dwordx4 ... lds: voffset = the lane's loop-invariant offset, soffset = the tile origin (wave uniform), M0 = the first 1 KB
                // block of each group of 4 issues, instruction offset 1024 (q % 4) moves both the LDS and the global address
                for (g, lds, m0) in mine.iter() {
                    b.load_buffer_lds(rsrc, *g, org, *m0, dma.act(*lds).1, 4);
                }
            });
        }
    }
    /// MMAs `from..to` (in row-major order over the first half of the B fragments) of the register tile.
    fn early_mmas(&mut self, b: &mut FnBuilder, from: usize, to: usize) {
        let (bv, av, sav, sbv) = self.rv.clone().expect("state bound");
        let nn = bv.len();
        let half = nn / 2;
        let mut n = 0;
        'o: for i in 0..av.len() {
            for j in 0..half {
                if n >= to {
                    break 'o;
                }
                if n >= from {
                    let (w, by) = self.sa_sel(i);
                    mma_var(b, &self.mma, self.acc[nn * i + j], av[i], bv[j], sav[w], sbv, by);
                }
                n += 1;
            }
        }
        self.npre_done = n;
    }
}

impl PipeBody for Body {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &I32, slot: &SlotIdx) {
        self.stage_now(b, key0, slot);
    }
    fn prologue_stores(&mut self, b: &mut FnBuilder, _role: Role) {
        {
            let (p, sj) = (&self.panel, &self.panel.sfa);
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
    /// The next MMAs of the register tile between the waits and the barrier (the vendor issues 3 there): the barrier latency hides behind them.
    fn post_wait(&mut self, b: &mut FnBuilder, _role: Role, _t: &TileIt) {
        if self.cfg.mid_sync_mmas > 0 {
            let from = self.npre_done;
            self.early_mmas(b, from, from + self.cfg.mid_sync_mmas);
        }
    }
    /// The first MMAs of the register tile run while the wave waits for the DMA and the barrier (the vendor issues 4 there).
    fn pre_sync(&mut self, b: &mut FnBuilder, _role: Role, _t: &TileIt) {
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
        self.compute_tile(b, t);
    }
}

/// The shipped MXFP8 blockscale GEMM.
pub fn gemm_mxfp8_bs_v4() -> Function {
    gemm_mxfp8_bs("gemm_mxfp8_bs_v4", GemmConfig::v4())
}

pub fn by_name(name: &str) -> Option<Function> {
    match name {
        "gemm_mxfp8_bs_v4" => Some(gemm_mxfp8_bs_v4()),
        _ => None,
    }
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
    // timing stamps (opt-in debug build, GEMM_STAMP=ends|tiles): one extra pointer argument after the ABI (offset 96); see docs/stamps.md
    let stamp_mode = std::env::var("GEMM_STAMP").ok().filter(|v| v == "ends" || v == "tiles");
    if stamp_mode.is_some() {
        b.arg("dbg_stamps", DType::Ptr, 96, 8);
        b.arg_tensor("dbg_stamps", Ext::sym("MT") * Ext::sym("NT") * Ext::c((4 * STAMP_SLOTS * 4) as i64));
    }
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
    let stamp_ctx: Option<(ValueId, ValueId, bool)> = stamp_mode.as_ref().map(|mode| {
        let buf = b.let_(Op::KernargPtr { offset: 96 });
        let nt = b.bind_i32(ev(n_v).lshr(cta_n.trailing_zeros() as i32));
        let lin = b.bind_i32(ev(bx0) + ev(by0) * ev(nt));
        let rec = b.stamp_rec_off(lin, STAMP_SLOTS as u32, 4);
        b.assume_trusted(rec, Ext::c(0), Ext::sym("MT") * Ext::sym("NT") * Ext::c((4 * STAMP_SLOTS * 4) as i64) - Ext::c(4 * STAMP_SLOTS as i64), "debug stamps: the record offset (workgroup * 4 + wave) * slots * 4 is inside dbg_stamps by construction (grid = NT x MT, 4 waves)");
        let t0 = b.memtime_lo();
        b.stamp_store(buf, rec, 0, t0); // slot 0: kernel start
        (buf, rec, mode == "tiles")
    });
    // XCD-aware raster (FMHA_ABLATE=xcd, experiment): workgroups are dispatched round robin over the 8 XCDs (linear id % 8), each with its own L2;
    // give every XCD whole 4 x 4 blocks of tiles (4 A + 4 B panels instead of all A panels). Only for a 16 x 16 tile grid here; identity otherwise.
    let (bx, by) = {
        let okv = b.bind_i32((ev(n_v) ^ (16 * cfg.cta_n) as i32) | (ev(m_v) ^ (16 * cfg.cta_m) as i32));
        let id = b.bind_i32(ev(bx0) + (ev(by0) << 4));
        let loc = b.bind_i32(ev(id).lshr(3));
        // block (of 16) = 2 (id % 8) + loc / 16 -> (row, col) = (block / 4, block % 4); inside the block: loc % 16 -> (n4, m4)
        let blk = b.bind_i32(((ev(id) & 7) << 1) + ev(loc).lshr(4));
        let (bc, br) = (ev(blk) & 3, ev(blk).lshr(2));
        let nx = b.bind_i32((bc << 2) + (ev(loc) & 3));
        let ny = b.bind_i32((br << 2) + (ev(loc).lshr(2) & 3));
        let s = b.bind_i32(ev(okv).eq(0).select_i32(1, 0));
        let px = b.bind_i32(ev(bx0) + ev(s) * (ev(nx) - ev(bx0)));
        let py = b.bind_i32(ev(by0) + ev(s) * (ev(ny) - ev(by0)));
        b.assume_trusted(px, Ext::c(0), Ext::sym("NT") - Ext::c(1), "XCD raster: (px, py) is a permutation of the 16 x 16 tile grid when the select applies (N = 16 cta_n, M = 16 cta_m: 4 x 4 blocks, 2 per XCD), the identity (workgroup ids) otherwise");
        b.assume_trusted(py, Ext::c(0), Ext::sym("MT") - Ext::c(1), "XCD raster: see px");
        (px, py)
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

    // ---- LDS ring: 2 slots of the padded A image + B 32 KB (+ an unused per-tile scale ring kept for the shipped LDS layout) ----
    let mut lds = LdsPlan::new();
    let slots = 2;
    let a_bytes = cta_m / 8 * 1056;
    let a_s = lds.ring("A", a_bytes, slots, amdgpu_dsl::tile::Layout::Linear);
    let b_s = lds.ring("B", cta_n * 128, slots, amdgpu_dsl::tile::Layout::Linear);
    let sc_s = lds.ring("SC", 4 * 512, slots, amdgpu_dsl::tile::Layout::Linear);
    // scale panel: A [PANEL_KB][256 rows] + B [2 blocks][PANEL_KB] (+ 4 bytes of source alignment slack, padded to 256 for the dword DMA)
    let panel_sh = lds.tile("SCP", PANEL_KB * 256 + 512, amdgpu_dsl::tile::Layout::Linear);
    // epilogue staging (wave-private, reusing the dead ring / panel storage): the wave's C block, bf16 rows of EPI_PITCH bytes
    let epi = if cfg.lds_epilogue {
        let ov: Vec<&Shared> = vec![&a_s, &b_s, &sc_s, &panel_sh];
        let per_wave = wave_m * EPI_PITCH;
        assert!(4 * per_wave <= 160 * 1024 - 16, "epilogue staging does not fit the LDS");
        Some(lds.private("EPI", per_wave, 4, &ov))
    } else {
        None
    };
    lds.commit(b);

    // ---- operand reads: the 16-row blocks of the wave (padded A: a 16-row block = two 1056-byte blocks of 8 rows; B 2048 bytes) ----
    let view = ComposedLayout::new(Swizzle::none(), smem_block());
    let a_view = ComposedLayout::new(Swizzle::none(), make_layout(tup![tup![8, 2], 128], tup![tup![128, 1056], 1]));
    let a_blk = 2112;
    let a_place: Vec<i64> = (0..wave_m / 16).map(|i| a_blk * i).collect();
    let b_place: Vec<i64> = (0..wave_n / 16).map(|j| 2048 * j).collect();
    // ---- DMA (A row-major into the padded image, B preshuffled), wave parts of M0, CTA origins ----
    let a_glob = make_layout(tup![cta_m, 128], tup![s_k.clone(), 1]);
    let b_glob = make_layout(tup![tup![16, cta_n / 16], tup![16, 8]], tup![tup![16, s_k.mul_c(16)], tup![1, 256]]);
    let a_dma = dma_plan_rows_pad(b, &ctx, &s_tid, cta_m, &a_glob, 8192, 4);
    let b_dma = dma_plan(b, &ctx, &s_tid, cta_n, &b_glob, 8192, 4);
    let a_wave = b.bind_i32(lane.wave() * (a_bytes / 4) as i32);
    let b_wave = b.bind_i32(lane.wave() * (cta_n * 128 / 4) as i32);
    // buffer descriptors over A (M K bytes) and the preshuffled B (N K bytes)
    let rsrc_a = { let n = b.bind_i32(ev(m_v) * ev(k_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_a, n) };
    let rsrc_b = { let n = b.bind_i32(ev(n_v) * ev(k_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_b, n) };
    let rsrc_sfa = { let n = b.bind_i32(ev(m_v) * ev(kb_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfa, n) };
    let rsrc_sfb = { let n = b.bind_i32((ev(n_v) >> 7) * ev(kb_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfb, n) };
    let a_org = b.bind_i32((ev(by) << lg_m) * ev(k_v));
    let b_org = b.bind_i32((ev(bx) << lg_n) * ev(k_v));

    // RETAINED SCALAR VALUES: the shipped v4 binary still computes these (the per-tile scale DMA, the panel-DMA / LDS-gather scale path and the
    // non-dword B scales, all replaced by the VGPR scale path below) - scalar code the lowering does not dead-code-eliminate. They are kept so the
    // shipped binary stays byte-identical in this cleanup; dropping them is a separate change (md5 change + gemm_gate).
    let _sc_wave = b.bind_i32(lane.wave() << 9);
    let _sa_src = b.bind_i32((ev(by) << lg_m) + (lane.lane() << 2));
    let sa_rd: Vec<ValueId> = (0..wave_m / 16).map(|i| b.bind_i32(ctx.lower(&s_wm.mul_c(wave_m).add_c(16 * i).add(&s_lane.mod_c(16, &ctx.dyns))))).collect();
    let sb_row = b.bind_i32(((ev(bx) << lg_n) + (ev(wn) * wave_n as i32)).lshr(7));
    let _sb_src = b.bind_i32(ev(sb_row) * ev(kb_v));
    let _sa_sh: Vec<ValueId> = sa_rd.iter().map(|r| b.bind_i32((ev(*r) & 3) << 3)).collect();
    let _sa_dw: Vec<ValueId> = sa_rd.iter().map(|r| b.bind_i32(ev(*r).lshr(2) << 2)).collect();

    // ---- the scale panel (written in the pipeline prologue, `prologue_stores`) ----
    assert!(cta_m == 256 && cta_n == 256, "scale panel: 256 x 256 CTA");
    let sh = panel_sh;
    let _b_src = b.bind_i32(((ev(bx) << 1) * ev(kb_v)) & -4);
    let b_delta = b.bind_i32(((ev(bx) << 1) * ev(kb_v)) & 3);
    let _m0s: Vec<ValueId> = (0..(PANEL_KB / 8) as i32).map(|r| { let v = b.bind_i32((lane.wave() << 9) + (2048 * r + sh.base as i32)); b.read_first_lane(v) }).collect();
    let _b_off = b.bind_i32((ev(wn) * ev(kb_v)) + ev(b_delta));
    {
        let r = b.bind_i32(ev(tid) & 15);
        let wmq = b.bind_i32((ev(tid).lshr(4)) & 1);
        let kq = b.bind_i32(ev(tid).lshr(5));
        let _sh_r = b.bind_i32((ev(r) & 3) << 3);
        let _rd0 = b.bind_i32((ev(wmq) << 7) + (ev(r).lshr(2) << 2) + (ev(kq) << 8) + sh.base as i32);
        let _wr0 = b.bind_i32((ev(wmq) << 7) + (ev(r) << 3) + (ev(kq) << 8) + sh.base as i32);
        let s = b.bind_i32(ev(r) & 3);
        let _sel_pair = b.bind_i32(ev(s) | ((ev(s) + 4) << 8) | I32::c(0x0c0c_0000));
        let _sel_merge = b.c_i32(0x0504_0100);
    }
    // this lane's A offset in the transposed panel: 128 wm + 8 (lane % 16)
    let tr = b.bind_i32((ev(wm) << 7) + ((lane.lane() & 15) << 3));
    {
        let _b_dw = b.bind_i32(ev(wn) * (4 * PANEL_KB as i32));
        let blk = b.bind_i32((ev(tid).lshr(6)) & 1);
        let kbt = b.bind_i32(ev(tid) & 63);
        let off = b.bind_i32(ev(blk) * ev(kb_v) + ev(b_delta) + ev(kbt));
        let _src = b.bind_i32((ev(off).lshr(2) << 2) + (sh.base + PANEL_KB * 256) as i32);
        let _shv = b.bind_i32((ev(off) & 3) << 3);
        let _dst = b.bind_i32(((ev(tid) & 127) << 2) + (sh.base + PANEL_KB * 256) as i32);
    }
    // the VGPR scale path: A by 4 passes of buffer_load_dwordx4 + in-register 4x4 byte transposes (dpp quad_perm + v_perm) + ds_write, B one dword
    sfa_transpose_proof();
    // A: lane g = 64 w + lane (+ 256 p per pass) holds 16 M rows (chunk c = g % 16) of K group g / 16: voffset (K group) M + m_cta + 16 c
    let g0 = b.bind_i32(lane.tid());
    let voff = b.bind_i32((ev(g0).lshr(4)) * ev(m_v) + (ev(by) << 8) + ((ev(g0) & 15) << 4));
    let j = b.bind_i32(ev(g0) & 3);
    // v_perm selectors (byte k of the result = sel byte k: 0..3 S1, 4..7 S0): step 1 [0,4,2,6] / [5,1,7,3], step 2 [0,1,4,5] / [6,7,2,3]
    let sel1 = b.bind_i32((ev(j) & 1).eq(0).select_i32(0x0602_0400, 0x0307_0105));
    let sel2 = b.bind_i32(ev(j).lt(2).select_i32(0x0504_0100, 0x0302_0706));
    // destination of dword d of pass p: panel + 256 kb + 128 wm + 8 (4 d + j) + 4 a  (wm = c / 8, a = c / 4 % 2) = base + 4096 p + 32 d
    let c = b.bind_i32(ev(g0) & 15);
    let dst = b.bind_i32(((ev(g0).lshr(4)) << 8) + ((ev(c).lshr(3)) << 7) + (ev(j) << 3) + (((ev(c).lshr(2)) & 1) << 2) + sh.base as i32);
    // B: thread tid % 128 -> (blk, kb) = (tid / 64 % 2, tid % 64): the dword holding byte (2 bx + blk) KB + kb, shifted to byte 0
    let blk = b.bind_i32((ev(tid).lshr(6)) & 1);
    let kbt = b.bind_i32(ev(tid) & 63);
    let boff = b.bind_i32(((ev(bx) << 1) + ev(blk)) * ev(kb_v) + ev(kbt));
    let bvo = b.bind_i32(ev(boff).lshr(2) << 2);
    let bsh = b.bind_i32((ev(boff) & 3) << 3);
    let bdst = b.bind_i32(((ev(tid) & 127) << 2) + (sh.base + PANEL_KB * 256) as i32);
    let pso: Vec<ValueId> = (0..4).map(|p| { let v = b.bind_i32(ev(m_v) * (16 * p)); b.read_first_lane(v) }).collect();
    let b_dw = b.bind_i32(ev(wn) * (4 * PANEL_KB as i32));
    let panel = Panel { sh, b_base: PANEL_KB * 256, tr, b_dw, sfa: SfaJob { voff, pso, sel1, sel2, dst, bvo, bsh, bdst } };

    let mut pipe = Pipeline::new(Cadence::RegStaged, kb_v, k_v, 128, Ext::ka(84));
    pipe.unroll2 = true;
    pipe.late_tile1 = true;
    pipe.late_tile1_vm = (a_dma.issues.len() + b_dma.issues.len()) as u8;
    pipe.scalar_exit = true;
    pipe.vm_inflight = (a_dma.issues.len() + b_dma.issues.len() + 2) as u8;
    // operand read partitions (the wave offset added to the bases)
    let mut a_rd = copy_a.partition_lds(b, &ctx, &s_lane, &a_s, &a_view, &a_place);
    let mut b_rd = copy_b.partition_lds(b, &ctx, &s_lane, &b_s, &view, &b_place);
    let a_wo = ctx.lower(&s_wm.mul_c(wave_m / 16 * a_blk));
    let b_wo = ctx.lower(&s_wn.mul_c(wave_n / 16 * 2048));
    a_rd.bases = a_rd.bases.iter().map(|x| b.let_i32(x + &a_wo)).collect();
    b_rd.bases = b_rd.bases.iter().map(|x| b.let_i32(x + &b_wo)).collect();
    let zero4 = b.frag_zero(DType::Frag4);
    let mut body = Body {
        cfg, kb_v, panel, mma: mma.clone(), a_s, b_s, a_dma, b_dma, a_org, b_org, a_wave, b_wave, rsrc_a, rsrc_b, rsrc_sfa, rsrc_sfb,
        a_rd, b_rd, zero4, acc: Vec::new(), staged: None, rv: None, npre_done: 0, stamp: stamp_ctx,
    };
    if let Some((buf, rec, _)) = stamp_ctx {
        let st = b.memtime_lo();
        b.stamp_store(buf, rec, 1, st); // slot 1: setup done, entering the pipeline (prologue)
    }
    pipe.run(b, Role::All, &mut body);
    if let Some((buf, rec, _)) = stamp_ctx {
        let st = b.memtime_lo();
        b.stamp_store(buf, rec, 3, st); // slot 3: the tile loop and the peeled last tile are done (epilogue next)
    }
    for v in body.acc.iter().take(cfg.acc_agpr) {
        b.pin_agpr(*v);
    }
    if let Some((bv, av, _, _)) = &body.rv {
        for v in bv.iter().chain(av.iter()) {
            b.pin_agpr(*v);
        }
    }
    if let Some(epi) = &epi {
        lds_epilogue(b, &ctx, &mma, epi, &body.acc, &lane, &s_lane, &s_n, ptr_c, m_v, n_v, bx, by, wm, wn, lg_m, lg_n, wave_m, wave_n, cfg.epi_phases);
        if let Some((buf, rec, _)) = stamp_ctx {
            let st = b.memtime_lo();
            b.stamp_store(buf, rec, 4, st); // slot 4: epilogue stores issued
        }
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
    if let Some((buf, rec, _)) = stamp_ctx {
        let st = b.memtime_lo();
        b.stamp_store(buf, rec, 4, st);
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
    n_v: ValueId, bx: ValueId, by: ValueId, wm: ValueId, wn: ValueId, lg_m: i32, lg_n: i32, wave_m: i64, wave_n: i64, phases: usize,
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
    // read / store TV over the wave block element index n + wave_n m: lane -> (chunk = lane % 16: 8 n, row = lane / 16), value -> (8 n, row step 4)
    let rows_per = 64 / (wave_n / 8); // 4
    let issues = wave_m / rows_per; // 32
    let tv = make_layout(tup![tup![wave_n / 8, rows_per], tup![8, issues]], tup![tup![8, wave_n], tup![1, wave_n * rows_per]]);
    let e_blk2 = make_layout(tup![wave_n, wave_m], tup![2, EPI_PITCH]);
    let rp = IssuePlan::new(tv.clone(), 8);
    let rexprs = rp.offsets(&e_blk2, s_lane, &Ix::C(0), d);
    let rpl = layout_to_offsets(&rexprs, &Swizzle::none(), ImmRange::DS, d);
    let rbases: Vec<ValueId> = rpl.bases.iter().map(|e| b.bind_i32(ev(base) + ctx.lower(e))).collect();
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
    // phases of wave_m / phases rows: the conversion + LDS writes of phase p+1 are issued before the stores of phase p (which only wait for
    // phase p's reads), so conversion, LDS traffic and global stores overlap (the vendor stages quadrant by quadrant)
    let phases = phases.max(1);
    let accs_per = acc.len() / phases;
    let iss_per = issues as usize / phases;
    let write_phase = |b: &mut FnBuilder, p: usize| {
        me.write(b, p as i32, |b| {
            for n in p * accs_per..(p + 1) * accs_per {
                let a = acc[n];
                let v: Vec<ValueId> = (0..4u8).map(|r| {
                    let l = b.frag4_lane(Opnd::Var(a), r);
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
    };
    write_phase(b, 0);
    for p in 0..phases {
        b.wait_lds(0);
        let vals: Vec<ValueId> = me.read_back(b, p as i32, |b| {
            rpl.issues[p * iss_per..(p + 1) * iss_per].iter().map(|&(bi, imm)| {
                let ad = b.bind_i32(ev(rbases[bi]) + imm as i32);
                b.load_lds_b128(ad)
            }).collect()
        });
        if p + 1 < phases {
            write_phase(b, p + 1);
        }
        for (k, v) in vals.iter().enumerate() {
            let q = p * iss_per + k;
            // the per-issue step is wave-uniform (rows_per rows): checked on the layout expressions
            let dq = gofs[q].sub(&gofs[0]);
            let dv = b.bind_i32(ev(c_org) + ctx.lower(&dq));
            let so = b.read_first_lane(dv);
            b.buffer_store_b128(rsrc_c, voff, so, 0, *v);
        }
    }
}