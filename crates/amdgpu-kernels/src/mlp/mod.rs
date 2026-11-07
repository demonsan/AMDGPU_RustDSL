//! Fused MLP-like chain `Y = fp8(relu(alpha X W1)) W2` (program item G step 3, docs/fusion-layout-handoff.md §3): GEMM 1 -> elementwise -> GEMM 2
//! with the intermediate kept in registers, against the unfused pair (two launches through an fp8 `H` in HBM).
//!
//! Shapes: X [M][256] fp8, W1 stored [N1 = 256][K1 = 256] (n1 rows, k contiguous), W2 stored [N2 = 256][N1 = 256] (n2 rows, n1 contiguous; the fused
//! kernel's copy is key-permuted within every 64-wide n1 block, see `sigma`), H [M][256] fp8, Y [M][256] bf16. All MFMAs are the scaled 32x32x64 fp8
//! atom with unit scales (E8M0 127). One CTA = 4 waves x 32 rows of M.
//!
//! The orientation is the one `amdgpu_tile::orient` derives for a producer -> consumer hand-off (the FMHA S -> P choice): GEMM 1 computes
//! `H^T = W1 X^T` (A = W1, B = X: every lane holds one row m), the fp8 packing in accumulator order makes `H^T` the B operand of GEMM 2
//! (`Y^T = W2 H^T`) up to the key permutation `sigma` - absorbed by the pre-permuted W2 - so the hand-off costs no instruction. The boundary is a
//! `StageBoundary` checked at build time (`check_boundary`).

use amdgpu_tile::{ev, Lane};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};
use amdgpu_isa::MmaAtom;
use amdgpu_tile::contract_stage::{check_boundary, LiveFrag, StageBoundary, Storage};
use amdgpu_tile::orient::{consumer_pairs, POperand};
use amdgpu_layout::Movement;

pub mod graph;
mod skeleton_fused;
mod skeleton_g1;
mod skeleton_g2;

pub const K1: i64 = 256;
pub const N1: i64 = 256;
pub const N2: i64 = 256;
pub const ALPHA: f32 = 0.125;

/// The key permutation of one 64-wide k block: consumer k slot -> n1 offset (the FMHA P packing: 32 (k / 32) + 8 ((k % 16) / 4) + 4 ((k % 32) / 16) + k % 4).
pub fn sigma(k: i64) -> i64 {
    32 * (k / 32) + 8 * ((k % 16) / 4) + 4 * ((k % 32) / 16) + k % 4
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Fused,
    /// GEMM 1 + elementwise, H to HBM
    G1,
    /// GEMM 2 from H
    G2,
    /// GEMM 1 with the generic epilogue chain (P2): `act(alpha H + bias[n1])`, per-row fp8 quantization, H and the row scales to HBM
    G1q(amdgpu_tile::prims::Act),
}

fn g1_like(mode: Mode) -> bool {
    matches!(mode, Mode::G1 | Mode::G1q(_))
}

pub fn by_name(name: &str) -> Option<Function> {
    let mode = match name {
        "mlp_fp8_ws_fused" => Mode::Fused,
        "mlp_fp8_ws_g1" => Mode::G1,
        "mlp_fp8_ws_g2" => Mode::G2,
        "mlp_fp8_ws_g1q_relu" => Mode::G1q(amdgpu_tile::prims::Act::Relu),
        "mlp_fp8_ws_g1q_silu" => Mode::G1q(amdgpu_tile::prims::Act::Silu),
        _ => return None,
    };
    Some(build_ws(name, mode))
}

/// The P hand-off as a stage boundary: producer = the packed words of GEMM 1's accumulators, consumer = GEMM 2's B operand (32 rows m x 64 keys per
/// block, keys through `sigma`). Element id = m + 32 n1 (n1 inside the 256-wide row of H).
fn p_boundary(jb: i64) -> (StageBoundary, StageBoundary) {
    let atom = MmaAtom::mfma_scale_32x32x64_f8();
    // producer: lane l, byte j = 4 w + i of the block's 8 words -> accumulator tile 2 jb + j / 16, value r = j % 16 -> n1 = 32 tile + 8 (r / 4) + 4 h + r % 4
    let prod: Vec<Vec<i64>> = (0..64i64).map(|l| (0..32i64).map(|j| { let (t, r, h) = (2 * jb + j / 16, j % 16, l / 32); (l % 32) + 32 * (32 * t + 8 * (r / 4) + 4 * h + r % 4) }).collect()).collect();
    // consumer: the B operand layout of the atom, k slot -> n1 = 64 jb + sigma(k)
    let cons: Vec<Vec<i64>> = consumer_pairs(&atom, POperand::B).into_iter().map(|lane| lane.into_iter().map(|(m, k)| m + 32 * (64 * jb + sigma(k))).collect()).collect();
    let roles = vec!["all 4 waves: same program".to_string()];
    let p = StageBoundary { live: vec![LiveFrag::from_pairs(&format!("P{jb}"), &prod, 8, Storage::Vgpr)], vm: 0, lgkm: 0, barrier_phase: 0, roles: roles.clone(), ..Default::default() };
    // the consumer tolerates outstanding loads of its own operands only (none of the producer's)
    let c = StageBoundary { live: vec![LiveFrag::from_pairs(&format!("P{jb}"), &cons, 8, Storage::Vgpr)], vm: 0, lgkm: 0, barrier_phase: 0, roles, ..Default::default() };
    (p, c)
}

/// LDS image row pitch of the weight matrices (256 bytes + 16 bytes of padding: the 16 rows a `ds_read_b128` phase touches fall on 16 distinct
/// 16-byte bank windows, and every read address is `lane base + immediate`).
pub const W_PITCH: i64 = 272;
/// Bytes of one weight image (256 rows).
pub const W_IMG: i64 = 256 * W_PITCH;

/// Weight-stationary, persistent form of the three MLP kernels (tuned building blocks): the weight images (host-prepared, `W_PITCH`-padded rows;
/// the fused kernel's W2 key-permuted) are DMA'd once per CTA into LDS, then every CTA loops over 128-row blocks (block c, c + G, ...): X rows by
/// direct 16-byte loads (the next block's X prefetched during GEMM 2), W fragments by `ds_read_b128` with immediate offsets, accumulators in
/// registers, the same zero-cost P hand-off as `build`. ABI: in @0, W1 image @8, W2 image @16, out @24, M @32, G (CTAs) @36.
/// The IR stage contract of the P hand-off (block `jb`): the producer / consumer element tables of `p_boundary` on the IR values.
fn p_stage_val(jb: i64, v: ValueId, producer: bool) -> amdgpu_dsl::StageVal {
    let (p, c) = p_boundary(jb);
    let lf = if producer { &p.live[0] } else { &c.live[0] };
    amdgpu_dsl::StageVal { name: lf.name.clone(), values: vec![v], table: lf.table.clone(), elem_bits: lf.elem_bits, storage: "vgpr".into(), layout: String::new() }
}

/// Values shared between the slots of the MLP kernels (what the generated skeleton's `Cx` carries).
pub(crate) struct Cx {
    p_in: ValueId,
    p_w1: ValueId,
    p_w2: ValueId,
    m_v: ValueId,
    g_v: ValueId,
    lane: Lane,
    bx: ValueId,
    w1s: Option<amdgpu_tile::Shared>,
    w2s: Option<amdgpu_tile::Shared>,
    r_in: ValueId,
    r_out: ValueId,
    epi: Option<(ValueId, ValueId)>,
    l16: ValueId,
    // set by `x_first`
    c127: Option<ValueId>,
    h16: Option<ValueId>,
    wlane: Option<ValueId>,
    nb: Option<ValueId>,
    x0: Vec<ValueId>,
    // set by `loop_open`
    vars: Vec<amdgpu_dsl::VarId>,
    blk: Option<amdgpu_dsl::I32>,
    xs: Vec<Opnd>,
    next: Option<ValueId>,
    row: Option<ValueId>,
    roles: Vec<String>,
    // set by the loop stages
    p_words: Vec<Opnd>,
    xn: Vec<ValueId>,
}

impl Cx {
    /// W fragment of 32-row tile t, k block kb (A operand): rows 32 t + j, bytes 64 kb + 16 h and + 32
    fn wfrag(&self, b: &mut FnBuilder, sh: &amdgpu_tile::Shared, t: i64, kb: i64) -> ValueId {
        use amdgpu_tile::SlotIdx;
        let wlane = self.wlane.expect("x_first");
        let pr = sh.at(&SlotIdx::new(0, 0));
        let base = sh.base as i32 + (t * 32 * W_PITCH + 64 * kb) as i32;
        pr.read(b, |b| {
            let a0 = b.bind_i32(ev(wlane) + base);
            let a1 = b.bind_i32(ev(wlane) + base + 32);
            b.load_lds_b128x2(a0, a1)
        })
    }
    /// row operand (X or H): 4 fragments of 64 k, 16 bytes at 64 kb + 16 h and + 32 of row m
    fn rowfrags(&self, b: &mut FnBuilder, blk: amdgpu_dsl::I32) -> Vec<ValueId> {
        let h16 = self.h16.expect("x_first");
        let lane = &self.lane;
        let r_in = self.r_in;
        let ro = b.bind_i32((((blk << 7) + (lane.wave() << 5) + lane.j()) << 8) + ev(h16));
        (0..K1 / 64).map(|kb| b.buffer_load_b128x2(r_in, ro, 0i32, (64 * kb) as u32, (64 * kb + 32) as u32)).collect()
    }
}

/// The slots of the MLP kernels: the orchestration the op-graph skeleton leaves to the kernel (the three modes share the pieces; `mode` selects what a slot emits).
pub(crate) struct MlpSlots {
    mode: Mode,
}

impl MlpSlots {
    fn s_prelude(&mut self, name: &str, b: &mut FnBuilder) -> Cx {
        use amdgpu_tile::{LdsPlan, Layout as TLayout};
        let mode = self.mode;
        if mode == Mode::Fused {
            for jb in 0..N1 / 64 {
                let (p, c) = p_boundary(jb);
                let rep = check_boundary(&p, &c).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert!(rep.frags.iter().all(|f| f.1 <= Movement::Register && f.2 == 0), "{name}: the P hand-off is not free: {:?}", rep.frags);
            }
        }
        b.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
        for (n, o) in [("ptr_in", 0u32), ("ptr_w1", 8), ("ptr_w2", 16), ("ptr_out", 24)] {
            b.arg(n, DType::Ptr, o, 8);
        }
        b.arg("m", DType::I32, 32, 4);
        b.arg("g", DType::I32, 36, 4);
        if let Mode::G1q(_) = mode {
            b.arg("ptr_bias", DType::Ptr, 40, 8);
            b.arg("ptr_scale", DType::Ptr, 48, 8);
        }
        let mt = Ext::sym("MT");
        let gs = Ext::sym("G");
        b.sym_range("MT", 1, 4096);
        b.sym_range("G", 1, 1024);
        let m = mt.clone() * Ext::c(128);
        b.ka_is(32, m.clone());
        b.ka_is(36, gs.clone());
        b.grid_is(0, gs);
        b.arg_tensor("ptr_in", m.clone() * Ext::c(K1));
        b.arg_tensor("ptr_w1", Ext::c(W_IMG));
        b.arg_tensor("ptr_w2", Ext::c(W_IMG));
        if let Mode::G1q(_) = mode {
            b.arg_tensor("ptr_bias", Ext::c(4 * N1));
            b.arg_tensor("ptr_scale", m.clone() * Ext::c(4));
        }
        b.arg_tensor("ptr_out", m.clone() * Ext::c(if g1_like(mode) { N1 } else { 2 * N2 }));
        let p_in = b.let_(Op::KernargPtr { offset: 0 });
        let p_w1 = b.let_(Op::KernargPtr { offset: 8 });
        let p_w2 = b.let_(Op::KernargPtr { offset: 16 });
        let p_out = b.let_(Op::KernargPtr { offset: 24 });
        let m_v = b.let_(Op::KernargI32 { offset: 32 });
        let g_v = b.let_(Op::KernargI32 { offset: 36 });
        let epi = if let Mode::G1q(_) = mode {
            let p_bias = b.let_(Op::KernargPtr { offset: 40 });
            let p_scale = b.let_(Op::KernargPtr { offset: 48 });
            Some((p_bias, p_scale))
        } else {
            None
        };
        let tid = b.let_(Op::WorkitemId);
        let lane = Lane::from_tid(b, tid);
        let bx = b.let_(Op::WorkgroupIdX);
        let mut lds = LdsPlan::new();
        let need_w1 = mode != Mode::G2;
        let need_w2 = !g1_like(mode);
        let w1s = need_w1.then(|| lds.tile("W1", W_IMG, TLayout::Linear));
        let w2s = need_w2.then(|| lds.tile("W2", W_IMG, TLayout::Linear));
        lds.commit(b);
        let r_in = { let n = b.bind_i32(ev(m_v) * (K1 as i32)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_in, n) };
        let r_out = { let n = b.bind_i32(ev(m_v) * (if g1_like(mode) { N1 } else { 2 * N2 }) as i32); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_out, n) };
        let epi = epi.map(|(pb, ps)| {
            let rb = b.buffer_desc(pb, (4 * N1) as u32);
            let rs = { let n = b.bind_i32(ev(m_v) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(ps, n) };
            (rb, rs)
        });
        // ---- prologue: the weight images into LDS (1 KB per DMA, wave w issues blocks w, w + 4, ...) ----
        let l16 = b.bind_i32(lane.lane() << 4);
        Cx { p_in, p_w1, p_w2, m_v, g_v, lane, bx, w1s, w2s, r_in, r_out, epi, l16, c127: None, h16: None, wlane: None, nb: None, x0: Vec::new(), vars: Vec::new(), blk: None, xs: Vec::new(), next: None, row: None, roles: Vec::new(), p_words: Vec::new(), xn: Vec::new() }
    }

    fn image(&mut self, b: &mut FnBuilder, cx: &Cx, w: usize) {
        use amdgpu_tile::SlotIdx;
        let (sh, p) = if w == 1 { (cx.w1s.as_ref(), cx.p_w1) } else { (cx.w2s.as_ref(), cx.p_w2) };
        let Some(sh) = sh else { return };
        let lane = cx.lane;
        let l16 = cx.l16;
        let r = b.buffer_desc(p, W_IMG as u32);
        let pr = sh.at(&SlotIdx::new(0, 0));
        let base = sh.base as i32;
        pr.write(b, |b| {
            for i in 0..(W_IMG / 1024 / 4) as i32 {
                // block q = 4 i + wave: source / destination byte 1024 q (+ 16 lane)
                let so = { let v = b.bind_i32((lane.wave() + 4 * i) << 10); b.read_first_lane(v) };
                let m0 = b.bind_i32(ev(so) + base);
                b.buffer_load_lds_b128(r, l16, so, m0, 0);
            }
        });
    }

    /// after the images: wait for the DMAs, one barrier
    fn s_images_done(&mut self, b: &mut FnBuilder) {
        b.wait_vm(0);
        b.barrier();
    }

    fn s_x_first(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        let c127 = b.c_i32(127);
        let h16 = b.bind_i32(cx.lane.half() << 4);
        let wlane = b.bind_i32(ev(cx.lane.j) * (W_PITCH as i32) + ev(h16));
        cx.c127 = Some(c127);
        cx.h16 = Some(h16);
        cx.wlane = Some(wlane);
        let nb = b.bind_i32(ev(cx.m_v).lshr(7));
        cx.nb = Some(nb);
        cx.x0 = cx.rowfrags(b, ev(cx.bx));
    }

    fn s_loop_open(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::Val(cx.bx))];
        inits.extend(cx.x0.iter().map(|x| (DType::Frag8, Opnd::Val(*x))));
        let vars = b.begin_loop(&inits);
        let blk = amdgpu_dsl::varx(vars[0]);
        cx.xs = vars[1..].iter().map(|v| Opnd::Var(*v)).collect();
        let next = b.bind_i32(blk.clone() + ev(cx.g_v));
        let row = b.bind_i32((blk.clone() << 7) + (cx.lane.wave() << 5) + cx.lane.j());
        cx.vars = vars;
        cx.blk = Some(blk);
        cx.next = Some(next);
        cx.row = Some(row);
        cx.roles = vec!["all 4 waves: same program".to_string()];
    }

    fn s_gemm1_act(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        let mode = self.mode;
        let fused = mode == Mode::Fused;
        let p_words: Vec<Opnd> = if mode == Mode::G2 {
            cx.xs.clone()
        } else {
            let wf = |b: &mut FnBuilder, sh: &amdgpu_tile::Shared, t: i64, kb: i64| -> ValueId { cx.wfrag(b, sh, t, kb) };
            let st = gemm1_act_stage(GemmCtx { wfrag: &wf, c127: cx.c127.expect("x_first") }, cx.w1s.as_ref().expect("W1"), &cx.xs, mode, cx.r_out, cx.row.expect("loop_open"), &cx.lane, &cx.roles, cx.epi);
            let out = st.out.clone();
            st.inst.plain(!fused).emit(b);
            out.get()
        };
        cx.p_words = p_words;
    }

    fn s_x_next(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        // the next block's row operand (clamped: past the end it reloads this block, unused)
        let blk = cx.blk.clone().expect("loop_open");
        let nclamp = b.bind_i32(ev(cx.next.expect("loop_open")).lt(ev(cx.nb.expect("x_first"))).select_i32(ev(cx.next.unwrap()), blk));
        cx.xn = cx.rowfrags(b, ev(nclamp));
    }

    fn s_gemm2(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        let fused = self.mode == Mode::Fused;
        let p_vals: Vec<ValueId> = cx.p_words.iter().filter_map(|o| if let Opnd::Val(v) = o { Some(*v) } else { None }).collect();
        let contract = amdgpu_dsl::StageContract {
            live_in: p_vals.iter().enumerate().map(|(jb, v)| p_stage_val(jb as i64, *v, false)).collect(),
            barrier_phase: 1,
            roles: cx.roles.clone(),
            ..Default::default()
        };
        let wf = |b: &mut FnBuilder, sh: &amdgpu_tile::Shared, t: i64, kb: i64| -> ValueId { cx.wfrag(b, sh, t, kb) };
        gemm2_stage(GemmCtx { wfrag: &wf, c127: cx.c127.expect("x_first") }, cx.w2s.as_ref().expect("W2"), &cx.p_words, cx.r_out, cx.row.expect("loop_open"), &cx.lane, contract).plain(!fused).emit(b);
    }

    fn s_loop_close(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        let next = cx.next.expect("loop_open");
        b.set_var(cx.vars[0], next);
        for (v, x) in cx.vars[1..].iter().zip(cx.xn.iter()) {
            b.set_var(*v, *x);
        }
        let done = b.bind_cond(ev(next).ge(ev(cx.nb.expect("x_first"))));
        b.break_if(done);
        b.end_loop();
    }
}

/// The fused kernel through the generated skeleton (`skeleton_fused.rs`, from the op graph `graph::graph()`): the slots call the pieces above in the graph's dependence order.
impl skeleton_fused::FusedSlots for MlpSlots {
    type Cx = Cx;
    fn prelude(&mut self, b: &mut FnBuilder) -> Cx {
        self.s_prelude("mlp_fp8_ws_fused", b)
    }
    fn w1_image(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.image(b, cx, 1);
    }
    fn w2_image(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.image(b, cx, 2);
        self.s_images_done(b);
    }
    fn x_first(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_x_first(b, cx);
    }
    fn loop_open(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_loop_open(b, cx);
    }
    fn gemm1_act(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_gemm1_act(b, cx);
    }
    fn x_next(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_x_next(b, cx);
    }
    fn gemm2(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_gemm2(b, cx);
    }
    fn loop_close(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_loop_close(b, cx);
    }
}

/// `mlp_fp8_ws_g1` through its generated cut skeleton (`skeleton_g1.rs`, kernel `g1` of `graph::graph_unfused()`): GEMM 1 + the activation, H stored by the stage itself (`store_h` has nothing left to do).
impl skeleton_g1::G1Slots for MlpSlots {
    type Cx = Cx;
    fn prelude(&mut self, b: &mut FnBuilder) -> Cx {
        self.s_prelude("mlp_fp8_ws_g1", b)
    }
    fn w1_image(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.image(b, cx, 1);
        self.s_images_done(b);
    }
    fn x_first(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_x_first(b, cx);
    }
    fn loop_open(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_loop_open(b, cx);
    }
    fn gemm1_act(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_gemm1_act(b, cx);
    }
    fn store_h(&mut self, _b: &mut FnBuilder, _cx: &mut Cx) {
        // `gemm1_act` in G1 mode stores H itself (its `map.relu_pack_fp8 -> store` epilogue)
    }
    fn x_next(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_x_next(b, cx);
    }
    fn loop_close(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_loop_close(b, cx);
    }
}

/// `mlp_fp8_ws_g2` through its generated cut skeleton (`skeleton_g2.rs`): H rows are the row operand, loaded by the row-fragment pieces (`h_first`, `h_next`); `load_h` has nothing left to do.
impl skeleton_g2::G2Slots for MlpSlots {
    type Cx = Cx;
    fn prelude(&mut self, b: &mut FnBuilder) -> Cx {
        self.s_prelude("mlp_fp8_ws_g2", b)
    }
    fn w2_image(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.image(b, cx, 2);
        self.s_images_done(b);
    }
    fn load_h(&mut self, _b: &mut FnBuilder, _cx: &mut Cx) {
        // the loads of H are the row-fragment loads of `h_first` / `h_next`
    }
    fn h_first(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_x_first(b, cx);
    }
    fn loop_open(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_loop_open(b, cx);
    }
    fn h_next(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_x_next(b, cx);
    }
    fn gemm2(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        // GEMM 2 reads the loaded H rows (the loop-carried fragments) as its B operand
        cx.p_words = cx.xs.clone();
        self.s_gemm2(b, cx);
    }
    fn loop_close(&mut self, b: &mut FnBuilder, cx: &mut Cx) {
        self.s_loop_close(b, cx);
    }
}

fn build_ws(name: &str, mode: Mode) -> Function {
    let mut slots = MlpSlots { mode };
    match mode {
        Mode::Fused => return skeleton_fused::build_fused(name, &mut slots),
        Mode::G1 => return skeleton_g1::build_g1(name, &mut slots),
        Mode::G2 => return skeleton_g2::build_g2(name, &mut slots),
        _ => {}
    }
    // the GEMM-1 + epilogue-chain variants (`g1q_*`): the same pieces in the order of g1's cut (their own graph with the bias / scale tensors is not generated yet)
    let mut fb = FnBuilder::new(name, 256);
    let b = &mut fb;
    let mut cx = slots.s_prelude(name, b);
    if g1_like(mode) {
        slots.image(b, &cx, 1);
    } else {
        slots.image(b, &cx, 2);
    }
    slots.s_images_done(b);
    slots.s_x_first(b, &mut cx);
    slots.s_loop_open(b, &mut cx);
    slots.s_gemm1_act(b, &mut cx);
    slots.s_x_next(b, &mut cx);
    if !g1_like(mode) {
        slots.s_gemm2(b, &mut cx);
    }
    slots.s_loop_close(b, &mut cx);
    fb.finish()
}

/// What the GEMM stages share with the kernel (the W fragment reader and the unit E8M0 scale).
struct GemmCtx<'r> {
    wfrag: &'r dyn Fn(&mut FnBuilder, &amdgpu_tile::Shared, i64, i64) -> ValueId,
    c127: ValueId,
}

struct Gemm1<'s> {
    inst: amdgpu_tile::stage::StageInst<'s>,
    /// the P words (fused: one Frag8 per 64-wide n1 block; G1: none, H went to HBM)
    out: amdgpu_tile::weave::Out<Vec<Opnd>>,
}

/// Stage `gemm1_act` (keys `mma.rows_x_lds` + `map.relu_pack_fp8`): `H^T = W1 X^T`, `fp8(relu(alpha H))` packed in accumulator order (fused: the
/// zero-cost B-operand words of GEMM 2; G1: stored to HBM).
#[allow(clippy::too_many_arguments)]
fn gemm1_act_stage<'s>(g: GemmCtx<'s>, w1: &'s amdgpu_tile::Shared, xs: &'s [Opnd], mode: Mode, r_out: ValueId, row: ValueId, lane: &'s Lane, roles: &[String], epi: Option<(ValueId, ValueId)>) -> Gemm1<'s> {
    use amdgpu_tile::weave::{Hint, Out, UnitClass, Units};
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut u = Units::new();
    let acc: Rc<RefCell<Vec<ValueId>>> = Rc::new(RefCell::new(Vec::new()));
    let a0 = acc.clone();
    let setup = u.add("gemm1.setup", UnitClass::Setup, Hint::Anchor, &[], move |b| {
        let zero = b.frag_zero(DType::Frag16);
        *a0.borrow_mut() = vec![zero; (N1 / 32) as usize];
    });
    let c127 = g.c127;
    let wfrag = g.wfrag;
    let mut last = setup;
    for kb in 0..K1 / 64 {
        for t in 0..(N1 / 32) as usize {
            let a1 = acc.clone();
            last = u.add(format!("gemm1.k{kb}t{t}"), UnitClass::Mfma, Hint::Skeleton, &[last], move |b| {
                let wf = wfrag(b, w1, t as i64, kb);
                let mut a = a1.borrow_mut();
                a[t] = b.mfma_scale_fp8(wf, xs[kb as usize], a[t], c127, c127);
            });
        }
    }
    let words: Rc<RefCell<Vec<Vec<ValueId>>>> = Rc::new(RefCell::new(Vec::new()));
    if let (Mode::G1q(act), Some((r_bias, r_scale))) = (mode, epi) {
        // ---- the generic epilogue chain (map.epilogue_chain): y = act(alpha acc + bias[n1]) (E), row amax (R: lane-local over the lane's 128
        // values, then the partner half of the row via permlane32_swap), per-row fp8 (Q); words in accumulator order, scale per row ----
        use amdgpu_tile::prims::{abs_f32, act as act_f, affine, fp8_pack, fp8_scale};
        let st: Out<(ValueId, ValueId)> = Out::new();
        let s1 = st.clone();
        last = u.add("epi.setup", UnitClass::Setup, Hint::Anchor, &[last], move |b| {
            let alpha = b.c_f32(ALPHA);
            let vb = b.bind_i32(lane.half() << 4);
            s1.set((alpha, vb));
        });
        let ys: Rc<RefCell<Vec<Vec<ValueId>>>> = Rc::new(RefCell::new(Vec::new()));
        let amax: Rc<RefCell<Option<ValueId>>> = Rc::new(RefCell::new(None));
        for t in 0..(N1 / 32) as usize {
            let (a2, s2, y2, m2) = (acc.clone(), st.clone(), ys.clone(), amax.clone());
            last = u.add(format!("epi.t{t}"), UnitClass::Valu, Hint::Filler, &[last], move |b| {
                let (alpha, vb) = s2.get();
                // bias of value r (n1 = 32 t + 8 (r / 4) + 4 h + r % 4): one 16-byte load per q = r / 4
                let bq: Vec<ValueId> = (0..4u32).map(|q| b.buffer_load_b128(r_bias, vb, 0i32, 128 * t as u32 + 32 * q)).collect();
                b.wait_vm(0);
                let a = a2.borrow()[t];
                let mut y = Vec::with_capacity(16);
                for r in 0..16u8 {
                    let v = b.frag16_lane(a, r);
                    let f = b.bitcast_f32(v);
                    let bl = b.frag4_lane(Opnd::Val(bq[(r / 4) as usize]), r % 4);
                    let bias = b.bitcast_f32(bl);
                    let z = affine(b, f, Some(alpha), Some(bias));
                    y.push(act_f(b, z, act));
                }
                let mut m = m2.borrow_mut();
                for &v in &y {
                    let av = abs_f32(b, v);
                    *m = Some(match *m { None => av, Some(p) => b.max_f32(p, av) });
                }
                y2.borrow_mut().push(y);
            });
        }
        let (y3, m3, w5) = (ys.clone(), amax.clone(), words.clone());
        last = u.add("epi.quant", UnitClass::Valu, Hint::Anchor, &[last], move |b| {
            let m = m3.borrow().expect("amax");
            // lanes l and l + 32 hold the two halves of row m: the partner value through permlane32_swap
            let (pa, pb) = b.permlane32_swap(m, m);
            let (fa, fb) = (b.bitcast_f32(pa), b.bitcast_f32(pb));
            let row_max = b.max_f32(fa, fb);
            let (scale, inv) = fp8_scale(b, row_max);
            for y in y3.borrow().iter() {
                w5.borrow_mut().push(fp8_pack(b, y, inv));
            }
            let vs = b.bind_i32(ev(row) << 2);
            b.buffer_store_f32(r_scale, vs, 0i32, 0, scale);
        });
    }
    let consts: Out<(ValueId, ValueId)> = Out::new();
    let c1 = consts.clone();
    if !matches!(mode, Mode::G1q(_)) {
        last = u.add("act.setup", UnitClass::Setup, Hint::Anchor, &[last], move |b| c1.set((b.c_f32(ALPHA), b.c_f32(0.0))));
    }
    for t in 0..(N1 / 32) as usize {
        if matches!(mode, Mode::G1q(_)) {
            break;
        }
        let (a2, c2, w2) = (acc.clone(), consts.clone(), words.clone());
        last = u.add(format!("act.t{t}"), UnitClass::Valu, Hint::Filler, &[last], move |b| {
            let (alpha, zf) = c2.get();
            let a = a2.borrow()[t];
            let ws: Vec<ValueId> = (0..4u8).map(|q| {
                let v: Vec<ValueId> = (0..4u8).map(|i| { let u = b.frag16_lane(a, 4 * q + i); let f = b.bitcast_f32(u); let s = b.mul_f32(f, alpha); b.max_f32(s, zf) }).collect();
                b.cvt_pk_fp8x4_f32(v[0], v[1], v[2], v[3])
            }).collect();
            w2.borrow_mut().push(ws);
        });
    }
    let out: Out<Vec<Opnd>> = Out::new();
    if g1_like(mode) {
        let vo: Out<ValueId> = Out::new();
        let v1 = vo.clone();
        last = u.add("store.setup", UnitClass::Setup, Hint::Anchor, &[last], move |b| {
            let h4 = b.bind_i32(lane.half() << 2);
            v1.set(b.bind_i32((ev(row) << 8) + ev(h4)));
        });
        for t in 0..(N1 / 32) as usize {
            let (w3, v2) = (words.clone(), vo.clone());
            last = u.add(format!("store.t{t}"), UnitClass::Dma, Hint::Filler, &[last], move |b| {
                let ws = w3.borrow()[t].clone();
                let vo = v2.get();
                for (q, w) in ws.iter().enumerate() {
                    let f = b.bitcast_f32(*w);
                    b.buffer_store_f32(r_out, vo, 0i32, (32 * t + 8 * q) as u32, f);
                }
            });
        }
        let o2 = out.clone();
        u.add("gemm1.out", UnitClass::Setup, Hint::Anchor, &[last], move |_b| o2.set(Vec::new()));
    } else {
        let (w4, o3) = (words.clone(), out.clone());
        u.add("pack.p", UnitClass::Valu, Hint::Filler, &[last], move |b| {
            let words = w4.borrow();
            let p: Vec<Opnd> = (0..N1 / 64).map(|jb| { let w: Vec<ValueId> = words[2 * jb as usize].iter().chain(words[2 * jb as usize + 1].iter()).copied().collect(); Opnd::Val(b.frag8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])) }).collect();
            o3.set(p);
        });
    }
    let contract = amdgpu_dsl::StageContract { barrier_phase: 1, roles: roles.to_vec(), ..Default::default() };
    let o4 = out.clone();
    let inst = amdgpu_tile::stage::StageInst::new("mma.rows_x_lds", amdgpu_tile::stage::Prim::A, "gemm1_act", contract, u)
        .param("atom", "mfma_scale_32x32x64_f8")
        .param("n", N1)
        .param("k", K1)
        .param("epilogue", match mode { Mode::G1 => "map.relu_pack_fp8 -> store", Mode::G1q(_) => "map.epilogue_chain(alpha, bias, act, row amax, fp8) -> store", _ => "map.relu_pack_fp8" })
        .handover(move |_b| {
            // the hand-over is recorded on the stage that produced it: the P words of GEMM 2's B operand
            let p = o4.get();
            let vals = p.iter().enumerate().filter_map(|(jb, o)| if let Opnd::Val(v) = o { Some(p_stage_val(jb as i64, *v, true)) } else { None }).collect();
            (vals, Vec::new())
        });
    Gemm1 { inst, out }
}

/// Stage `gemm2` (keys `mma.rows_x_lds` + `map.store_bf16`): `Y^T = W2 H^T` from the P words, bf16 stores.
#[allow(clippy::too_many_arguments)]
fn gemm2_stage<'s>(g: GemmCtx<'s>, w2: &'s amdgpu_tile::Shared, p_words: &'s [Opnd], r_out: ValueId, row: ValueId, lane: &'s Lane, contract: amdgpu_dsl::StageContract) -> amdgpu_tile::stage::StageInst<'s> {
    use amdgpu_tile::weave::{Hint, Out, UnitClass, Units};
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut u = Units::new();
    let acc: Rc<RefCell<Vec<ValueId>>> = Rc::new(RefCell::new(Vec::new()));
    let a0 = acc.clone();
    let mut last = u.add("gemm2.setup", UnitClass::Setup, Hint::Anchor, &[], move |b| {
        let zero = b.frag_zero(DType::Frag16);
        *a0.borrow_mut() = vec![zero; (N2 / 32) as usize];
    });
    let (c127, wfrag) = (g.c127, g.wfrag);
    for (jb, pf) in p_words.iter().enumerate() {
        for uu in 0..(N2 / 32) as usize {
            let a1 = acc.clone();
            last = u.add(format!("gemm2.j{jb}u{uu}"), UnitClass::Mfma, Hint::Skeleton, &[last], move |b| {
                let wf = wfrag(b, w2, uu as i64, jb as i64);
                let mut a = a1.borrow_mut();
                a[uu] = b.mfma_scale_fp8(wf, *pf, a[uu], c127, c127);
            });
        }
    }
    let vo: Out<ValueId> = Out::new();
    let v1 = vo.clone();
    last = u.add("store.setup", UnitClass::Setup, Hint::Anchor, &[last], move |b| {
        let h8 = b.bind_i32(lane.half() << 3);
        v1.set(b.bind_i32((ev(row) << 9) + ev(h8)));
    });
    for uu in 0..(N2 / 32) as usize {
        let (a2, v2) = (acc.clone(), vo.clone());
        last = u.add(format!("store.u{uu}"), UnitClass::Valu, Hint::Filler, &[last], move |b| {
            let a = a2.borrow()[uu];
            let vo = v2.get();
            for q in 0..4u8 {
                let x: Vec<ValueId> = (0..4u8).map(|i| { let v = b.frag16_lane(a, 4 * q + i); b.bitcast_f32(v) }).collect();
                let w0 = b.cvt_pk_bf16_f32(x[0], x[1]);
                let w1 = b.cvt_pk_bf16_f32(x[2], x[3]);
                let f2 = b.frag2(w0, w1);
                b.buffer_store_b64(r_out, vo, 0i32, (64 * uu + 16 * q as usize) as u32, f2);
            }
        });
    }
    amdgpu_tile::stage::StageInst::new("mma.rows_x_lds", amdgpu_tile::stage::Prim::A, "gemm2", contract, u)
        .param("atom", "mfma_scale_32x32x64_f8")
        .param("n", N2)
        .param("k", N1)
        .param("epilogue", "map.store_bf16")
}