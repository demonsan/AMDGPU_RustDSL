//! Tile library layers 1-3 (docs/dsl-tile-abstractions.md): tensor views generate the FMHA host contract, the LDS plan reproduces `fmha_vs_tiles`,
//! and a two-role DMA ring pipeline written with the library is accepted by `gpu_verify` with zero errors (and is rejected when the protocol is broken).
use amdgpu_codegen::gpu_verify::{analyse, Severity};
use amdgpu_tile::*;
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Opnd, VarId, I32};

fn fmha_tensors(b: &mut FnBuilder, ka: &mut KArgs) -> (Tensor, Tensor, Tensor, Tensor) {
    let s = ka.dim(b, "s_seq_len", 0x060, 1, i64::MAX);
    let skv = ka.dim(b, "s_kv_seq_len", 0x110, 1, i64::MAX);
    let h = Dim::Ka { off: 0x140, val: ka.i32(b, "s_q_head_num", 0x140) };
    let hk = Dim::sym("HK");
    let bat = Dim::grid(2);
    let d = Dim::c(128);
    let q = ka
        .tensor("ptr_q", 0x010, 1, vec![bat.clone(), s.clone(), h.clone(), d.clone()], vec![Stride::Ka("s_Bs", 0x0A0), Stride::Ka("s_Seqs", 0x070), Stride::Ka("s_Hs", 0x090), Stride::Unit])
        .packed()
        .declare(ka, b);
    let k = ka
        .tensor("ptr_k", 0x020, 1, vec![bat.clone(), skv.clone(), hk.clone(), d.clone()], vec![Stride::Ka("s_k_Bs", 0x0E0), Stride::Ka("s_k_Seqs", 0x0C0), Stride::Ka("s_k_Hs", 0x0D0), Stride::Unit])
        .packed()
        .declare(ka, b);
    let v = ka
        .tensor("ptr_v", 0x030, 1, vec![bat.clone(), skv, hk, d.clone()], vec![Stride::Ka("s_v_Bs", 0x170), Stride::Ka("s_v_Seqs", 0x150), Stride::Ka("s_v_Hs", 0x160), Stride::Unit])
        .same_strides_as(&k)
        .declare(ka, b);
    let o = ka
        .tensor("ptr_o", 0x000, 2, vec![bat, s, h, d], vec![Stride::Ka("s_o_Bs", 0x1A0), Stride::Ka("s_o_Seqs", 0x180), Stride::Ka("s_o_Hs", 0x190), Stride::Unit])
        .packed()
        .declare(ka, b);
    (q, k, v, o)
}

#[test]
fn tensor_declarations_generate_the_fmha_contract() {
    let mut b = FnBuilder::new("t", 512);
    let mut ka = KArgs::new();
    let _ = fmha_tensors(&mut b, &mut ka);
    let mine = b.finish().contract;
    let shipped = amdgpu_kernels::kernel_by_name("fmha_fwd_hd128_mxfp8_v3_vs128").unwrap().contract;
    for e in &mine.extents {
        assert!(shipped.extents.iter().any(|s| s.arg == e.arg && s.alloc == e.alloc && s.logical == e.logical), "extent {e:?} is not the shipped one");
    }
    for d in &mine.ka_defs {
        assert!(shipped.ka_defs.contains(d), "ka_is {d:?} is not in the shipped contract");
    }
    for r in &mine.ka_ranges {
        assert!(shipped.ka_ranges.contains(r), "ka_range {r:?} is not in the shipped contract");
    }
    // every stride relation of Q/K/V/O is generated (13 kernarg strides: 4 + 4 + 3 + 4 - the pinned head strides are ranges)
    let defs: Vec<u32> = mine.ka_defs.iter().map(|d| d.0).collect();
    for off in [0x070u32, 0x0A0, 0x0C0, 0x0E0, 0x150, 0x170, 0x180, 0x1A0] {
        assert!(defs.contains(&off), "missing ka_is for 0x{off:x}");
    }
    for off in [0x090u32, 0x0D0, 0x160, 0x190] {
        assert!(mine.ka_ranges.iter().any(|r| r.0 == off), "missing ka_range for 0x{off:x}");
    }
    assert_eq!(mine.extents.len(), 4);
}

#[test]
fn tile_view_offsets_clamp_the_relative_row() {
    let mut b = FnBuilder::new("t", 64);
    let mut ka = KArgs::new();
    let (q, ..) = fmha_tensors(&mut b, &mut ka);
    let bidx = b.c_i32(0);
    let s_last = b.let_i32(q.dims[1].val().unwrap() - 1);
    let qt = q.tile(vec![ev(bidx), I32::c(0), I32::c(0), I32::c(0)]).clamp(1, q.dims[1].val().unwrap(), s_last);
    let before = b.stmt_count();
    let r = b.c_i32(5);
    let _p = qt.ptr(&mut b, &[I32::c(0), ev(r), I32::c(0), I32::c(16)]);
    let f = b.finish();
    let text = amdgpu_dsl::print::print_dsl(&f);
    // clamp = cmp + select, then the offset terms and one ptr_add
    assert!(f.stmts().len() - before >= 6, "{text}");
    assert!(text.to_lowercase().contains("select"), "the row index must be clamped:\n{text}");
}

#[test]
fn lds_plan_reproduces_the_vs128_layout() {
    let mut b = FnBuilder::new("t", 512);
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(2, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
    let kd = lds.ring("KD", 512, 2, Layout::Linear);
    lds.at(67584);
    let ones = lds.tile("ones", 2048, Layout::Linear);
    let q1 = lds.tile("Q1", 16384, Layout::Linear);
    let q0 = lds.tile("Q0", 16384, Layout::Linear);
    let _epi = lds.private("epi", 4096, 8, &[&q1, &q0]);
    assert_eq!((ks.base, vs.base, ks.slot_stride, kd.base, ones.base, q1.base, q0.base), (0, 16384, 32768, 65536, 67584, 69632, 86016));
    assert_eq!(lds.total(), 102400);
    let mut aliases = lds.commit(&mut b);
    aliases.sort();
    let f = b.finish();
    assert_eq!(f.lds_bytes, 102400);
    let mut want: Vec<(String, String)> = [("K", "V"), ("V", "KD"), ("V", "ones"), ("V", "Q1"), ("Q1", "epi"), ("Q0", "epi"), ("V", "epi")].iter().map(|(a, c)| (a.to_string(), c.to_string())).collect();
    want.sort();
    assert_eq!(aliases, want, "the justified aliases are exactly the manual fmha_vs_tiles list");
}

#[test]
#[should_panic(expected = "really overlap")]
fn lds_plan_refuses_an_undeclared_real_overlap() {
    let mut b = FnBuilder::new("t", 64);
    let mut lds = LdsPlan::new();
    let _a = lds.tile("A", 1024, Layout::Linear);
    lds.at(512);
    let _c = lds.tile("C", 1024, Layout::Linear);
    lds.commit(&mut b);
}

// ---- a two-role DMA ring written with the library ------------------------------------------------------------------------------------------------------------------

struct Demo {
    ks: Shared,
    vs: Shared,
    kd: Shared,
    kcp: RowsCopy,
    vcp: RowsCopy,
    kdcp: DwordRowsCopy,
    lane: Lane,
    /// break the protocol: 0 = none, 1 = read K of the wrong generation, 2 = read V before the first barrier
    bug: u8,
    acc: Option<VarId>,
}

impl Demo {
    fn read_k(&self, b: &mut FnBuilder, slot: &SlotIdx) -> amdgpu_dsl::ValueId {
        let r = self.ks.at(slot);
        let a = b.bind_i32(r.base() + (self.lane.lane() << 4));
        r.read(b, |b| b.load_lds_b128(a))
    }
}

impl PipeBody for Demo {
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
        match role {
            Role::A => {
                self.kdcp.issue(b, key0, &self.kd.at(slot));
                self.kcp.issue(b, key0, &self.ks.at(slot));
            }
            _ => self.vcp.issue(b, key0, &self.vs.at(slot)),
        }
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        let _ = self.read_k(b, &t0.cur);
    }
    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let z = b.c_i32(0);
        vec![(DType::I32, Opnd::Val(z))]
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        self.acc = Some(vars[0]);
    }
    fn v_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        if self.bug == 2 {
            let r = self.vs.at(&t.next);
            let a = b.bind_i32(r.base() + (self.lane.lane() << 4));
            let _ = r.read(b, |b| b.load_lds_b128(a));
        }
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let r = self.vs.at(&t.cur);
        let a = b.bind_i32(r.base() + (self.lane.lane() << 4));
        let _v = r.read(b, |b| b.load_lds_b128(a));
        if t.staged {
            // the K scales are consumed with their K tile (tile t+1, read in this M phase): reading KD(t) here would race with group A's DMA of
            // KD(t+2) into the same slot while group B (one phase behind) is still in this phase - the verifier reports exactly that (GV-TILE-RACE)
            let nx = if self.bug == 1 { SlotIdx::new(t.next.slot, t.cur.gen) } else { t.next.clone() };
            let kd = self.kd.at(&nx);
            let ka = b.bind_i32(kd.base() + (self.lane.lane() << 2));
            let s = kd.read(b, |b| b.load_lds_b32(ka));
            let acc = self.acc.unwrap();
            let n = b.bind_i32(varx(acc) + ev(s));
            b.set_var(acc, n);
            let _ = self.read_k(b, &nx);
        }
    }
}

fn demo(bug: u8) -> Function {
    let mut b = FnBuilder::new("tile_demo_ring", 512);
    b.compile_profile(CompileProfile::Default);
    let mut ka = KArgs::new();
    let skv = ka.dim(&mut b, "kv_len", 0x10, 1, i64::MAX);
    let kv_len = ka.i32(&mut b, "kv_len", 0x10);
    let k = ka.tensor("ptr_k", 0x00, 1, vec![skv.clone(), Dim::c(128)], vec![Stride::Derived(I32::c(128)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let v = ka.tensor("ptr_v", 0x08, 1, vec![skv.clone(), Dim::c(128)], vec![Stride::Derived(I32::c(128)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let kd = ka.tensor("ptr_kd", 0x18, 4, vec![skv, Dim::c(1)], vec![Stride::Derived(I32::c(4)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(2, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
    let kds = lds.ring("KD", 512, 2, Layout::Linear);
    lds.commit(&mut b);
    let lane = Lane::new(&mut b);
    let roles = Roles::split(&mut b, &lane, 4);
    let kv_last = b.let_i32(ev(kv_len) - 1);
    let nkv = b.bind_i32((ev(kv_len) + 127).lshr(7));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x10).ceil_div(128), "nkv = ceil(kv_len / 128) >= 1");
    let kt = k.tile(vec![I32::c(0), I32::c(0)]).clamp(0, ev(kv_len), kv_last.clone());
    let vt = v.tile(vec![I32::c(0), I32::c(0)]).clamp(0, ev(kv_len), kv_last.clone());
    let kdt = kd.tile(vec![I32::c(0), I32::c(0)]).clamp(0, ev(kv_len), kv_last);
    let (ksrc, vsrc, kdsrc) = (kt.row_source(&mut b, 0), vt.row_source(&mut b, 0), kdt.row_source(&mut b, 0));
    let kcp = RowsCopy::new(&mut b, ksrc, &ks, roles.wave_in_group, &lane, 4);
    let vcp = RowsCopy::new(&mut b, vsrc, &vs, roles.wave_in_group, &lane, 4);
    let kdcp = DwordRowsCopy::new(&mut b, kdsrc, &kds, roles.wave_in_group, &lane);
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, nkv, kv_len, 128, Ext::ka(0x10).ceil_div(128));
    let mut body = Demo { ks, vs, kd: kds, kcp, vcp, kdcp, lane, bug, acc: None };
    roles.each(&mut b, |b, role| pipe.run(b, role, &mut body));
    b.finish()
}

fn errs(f: &Function) -> Vec<String> {
    let r = analyse(f).expect("analyse");
    r.findings.iter().filter(|x| x.severity == Severity::Error).map(|x| format!("[{}] {}", x.code, x.msg)).collect()
}

#[test]
fn library_ring_pipeline_verifies_strict_clean_and_compiles() {
    let f = demo(0);
    let e = errs(&f);
    assert!(e.is_empty(), "{}", e.join("\n"));
    let c = amdgpu_codegen::compile(&f).expect("compile");
    let n = |m: &str| c.asm.lines().filter(|l| l.trim_start().starts_with(m)).count();
    // A: K tile 4 + K scale 1 per stage, B: V tile 4; each role stages in the prologue and once per loop iteration
    assert_eq!(n("global_load_lds_dwordx4"), 16, "{}", c.asm);
    assert_eq!(n("global_load_lds_dword "), 2);
    // barriers: A = P0 + 1 + 2 (loop) + 2 x 2 (tail copies, last barrier in A) = 8; B = P0 + P1 + 1 + 2 + 2 x 1 = 7  -> 15 static
    assert_eq!(n("s_barrier"), 15, "{}", c.asm);
}

#[test]
fn library_ring_pipeline_catches_a_wrong_generation() {
    let e = errs(&demo(1));
    assert!(e.iter().any(|x| x.contains("GV-TILE")), "a stale-generation read must be rejected: {e:?}");
}

#[test]
fn library_ring_pipeline_catches_a_read_before_publication() {
    let e = errs(&demo(2));
    assert!(e.iter().any(|x| x.contains("GV-TILE")), "a read before the publishing barrier must be rejected: {e:?}");
}

// ---- the `_t128` cadence: one group, 3-slot ring, one barrier per tile ------------------------------------------------------------------------------------------

struct Demo3 {
    ks: Shared,
    vs: Shared,
    kd: Shared,
    kcp: RowsCopy,
    vcp: RowsCopy,
    kdcp: DwordRowsCopy,
    lane: Lane,
    /// the window also reads tile t+1 (the FMHA QK(t+1)); false = a GEMM-like window that consumes only tile t
    next_reads: bool,
}

impl Demo3 {
    fn rd(&self, b: &mut FnBuilder, sh: &Shared, slot: &SlotIdx, shift: i32) {
        let r = sh.at(slot);
        let a = b.bind_i32(r.base() + (self.lane.lane() << shift));
        let _ = r.read(b, |b| if shift == 4 { b.load_lds_b128(a) } else { b.load_lds_b32(a) });
    }
}

impl PipeBody for Demo3 {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &I32, slot: &SlotIdx) {
        self.kdcp.issue(b, key0, &self.kd.at(slot));
        self.kcp.issue(b, key0, &self.ks.at(slot));
        self.vcp.issue(b, key0, &self.vs.at(slot));
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        let (ks, kd) = (self.ks.clone(), self.kd.clone());
        self.rd(b, &ks, &t0.cur, 4);
        self.rd(b, &kd, &t0.cur, 2);
    }
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        vec![]
    }
    fn bind_state(&mut self, _vars: &[VarId]) {}
    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (ks, vs, kd) = (self.ks.clone(), self.vs.clone(), self.kd.clone());
        self.rd(b, &vs, &t.cur, 4);
        if !self.next_reads {
            self.rd(b, &ks, &t.cur, 4);
            self.rd(b, &kd, &t.cur, 2);
        }
        if t.staged && self.next_reads {
            self.rd(b, &ks, &t.next, 4);
            self.rd(b, &kd, &t.next, 2);
        }
    }
}

fn demo3() -> Function {
    demo3_vm(0, true)
}

fn demo3_vm(vm_inflight: u8, next_reads: bool) -> Function {
    let mut b = FnBuilder::new("tile_demo_ring3", 256);
    b.compile_profile(CompileProfile::Default);
    let mut ka = KArgs::new();
    let skv = ka.dim(&mut b, "kv_len", 0x10, 1, i64::MAX);
    let kv_len = ka.i32(&mut b, "kv_len", 0x10);
    let k = ka.tensor("ptr_k", 0x00, 1, vec![skv.clone(), Dim::c(128)], vec![Stride::Derived(I32::c(128)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let v = ka.tensor("ptr_v", 0x08, 1, vec![skv.clone(), Dim::c(128)], vec![Stride::Derived(I32::c(128)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let kd = ka.tensor("ptr_kd", 0x18, 4, vec![skv, Dim::c(1)], vec![Stride::Derived(I32::c(4)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(3, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
    let kds = lds.ring("KD", 512, 3, Layout::Linear);
    lds.commit(&mut b);
    let lane = Lane::new(&mut b);
    let roles = Roles::single(&lane, 4);
    let kv_last = b.let_i32(ev(kv_len) - 1);
    let nkv = b.bind_i32((ev(kv_len) + 127).lshr(7));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x10).ceil_div(128), "nkv = ceil(kv_len / 128) >= 1");
    let src = |t: &Tensor, b: &mut FnBuilder| t.tile(vec![I32::c(0), I32::c(0)]).clamp(0, ev(kv_len), kv_last.clone()).row_source(b, 0);
    let (ksrc, vsrc, kdsrc) = (src(&k, &mut b), src(&v, &mut b), src(&kd, &mut b));
    let kcp = RowsCopy::new(&mut b, ksrc, &ks, roles.wave_in_group, &lane, 4);
    let vcp = RowsCopy::new(&mut b, vsrc, &vs, roles.wave_in_group, &lane, 4);
    let kdcp = DwordRowsCopy::new(&mut b, kdsrc, &kds, roles.wave_in_group, &lane);
    let mut pipe = Pipeline::new(Cadence::OneWindow, nkv, kv_len, 128, Ext::ka(0x10).ceil_div(128));
    pipe.vm_inflight = vm_inflight;
    let mut body = Demo3 { ks, vs, kd: kds, kcp, vcp, kdcp, lane, next_reads };
    pipe.run(&mut b, Role::All, &mut body);
    b.finish()
}

#[test]
fn one_window_three_slot_ring_verifies() {
    let f = demo3();
    let r = analyse(&f).expect("analyse");
    let e: Vec<String> = r.findings.iter().filter(|x| x.severity != Severity::Info).map(|x| format!("[{}] {:?} {}", x.code, x.severity, x.msg)).collect();
    eprintln!("ring3 findings:\n{}\nproven {:?}", e.join("\n"), r.proven);
    assert!(errs(&f).is_empty(), "{}", e.join("\n"));
    let c = amdgpu_codegen::compile(&f).expect("compile");
    let n = |m: &str| c.asm.lines().filter(|l| l.trim_start().starts_with(m)).count();
    assert_eq!(n("s_barrier"), 2, "P0 + one per steady tile (the tail has none)");
}

#[test]
fn kv_split_and_gqa_components_verify() {
    use amdgpu_tile::components::*;
    let mut b = FnBuilder::new("tile_demo_kvs", 64);
    let mut ka = KArgs::new();
    let _ = ka.dim(&mut b, "skv", 0x00, 1, i64::MAX);
    let kv_len = ka.i32(&mut b, "skv", 0x00);
    ka.arg(&mut b, "gqa", DType::I32, 0x04, 4);
    b.ka_range(0x04, 1, 16);
    let gqa = ka.i32(&mut b, "gqa", 0x04);
    let _ = ka.dim(&mut b, "splits", 0x08, 1, i64::MAX);
    let splits = ka.i32(&mut b, "splits", 0x08);
    b.grid_is(2, Ext::sym("B") * Ext::ka(0x08));
    let z = b.let_(amdgpu_dsl::Op::WorkgroupIdZ);
    let h = b.let_(amdgpu_dsl::Op::WorkgroupIdY);
    let (_bi, sp, lg) = kv_split_decode(&mut b, z, splits, 0x08, "B");
    let (_klo, _keff) = kv_split_range(&mut b, kv_len, sp, lg, 0x00);
    let _kvh = gqa_kv_head(&mut b, h, gqa, "HK");
    let f = b.finish();
    assert!(errs(&f).is_empty(), "{:?}", errs(&f));
    let r = analyse(&f).unwrap();
    assert_eq!(r.ledger().assume_trusted, 8, "the components' trusted facts (7 ranges + the split bijection) are counted in the certificate");
}

// ---- `_t128` staging: K and V of the same rows with one shared row clamp, K scales through a VGPR written after the DMA wait ----------------------------------------

struct Demo3v {
    base: Demo3,
    kdv: DwordRowsVgpr,
    pending: Option<amdgpu_dsl::ValueId>,
}

impl PipeBody for Demo3v {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &I32, slot: &SlotIdx) {
        self.pending = Some(self.kdv.issue(b, key0));
        let (ks, vs) = (self.base.ks.clone(), self.base.vs.clone());
        issue_rows_shared(b, key0, &[(&self.base.kcp, &ks.at(slot)), (&self.base.vcp, &vs.at(slot))]);
    }
    fn stage_complete(&mut self, b: &mut FnBuilder, _role: Role, slot: &SlotIdx) {
        let kd = self.base.kd.clone();
        self.kdv.complete(b, self.pending.take().expect("pending K scale"), &kd.at(slot));
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, role: Role, t0: &TileIt) {
        self.base.prologue_compute(b, role, t0)
    }
    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        self.base.loop_state(b)
    }
    fn bind_state(&mut self, v: &[VarId]) {
        self.base.bind_state(v)
    }
    fn window(&mut self, b: &mut FnBuilder, role: Role, t: &TileIt) {
        self.base.window(b, role, t)
    }
}

#[test]
fn t128_style_staging_verifies() {
    // rebuild with the t128 staging (same declarations)
    let mut b = FnBuilder::new("tile_demo_ring3v", 256);
    b.compile_profile(CompileProfile::Default);
    let mut ka = KArgs::new();
    let skv = ka.dim(&mut b, "kv_len", 0x10, 1, i64::MAX);
    let kv_len = ka.i32(&mut b, "kv_len", 0x10);
    let k = ka.tensor("ptr_k", 0x00, 1, vec![skv.clone(), Dim::c(128)], vec![Stride::Derived(I32::c(128)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let v = ka.tensor("ptr_v", 0x08, 1, vec![skv.clone(), Dim::c(128)], vec![Stride::Derived(I32::c(128)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let kd = ka.tensor("ptr_kd", 0x18, 4, vec![skv, Dim::c(1)], vec![Stride::Derived(I32::c(4)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(3, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
    let kds = lds.ring("KD", 512, 3, Layout::Linear);
    lds.commit(&mut b);
    let lane = Lane::new(&mut b);
    let roles = Roles::single(&lane, 4);
    let kv_last = b.let_i32(ev(kv_len) - 1);
    let nkv = b.bind_i32((ev(kv_len) + 127).lshr(7));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x10).ceil_div(128), "nkv = ceil(kv_len / 128) >= 1");
    let src = |t: &Tensor, b: &mut FnBuilder| t.tile(vec![I32::c(0), I32::c(0)]).clamp(0, ev(kv_len), kv_last.clone()).row_source(b, 0);
    let (ksrc, vsrc, kdsrc) = (src(&k, &mut b), src(&v, &mut b), src(&kd, &mut b));
    let kcp = RowsCopy::new(&mut b, ksrc, &ks, roles.wave_in_group, &lane, 4);
    let vcp = RowsCopy::new(&mut b, vsrc.clone(), &vs, roles.wave_in_group, &lane, 4);
    let kdcp = DwordRowsCopy::new(&mut b, vsrc, &kds, roles.wave_in_group, &lane); // unused placeholder of Demo3
    let rt_row = b.bind_i32((lane.tid() & 255).lshr(1));
    let kdv = DwordRowsVgpr::new(&mut b, kdsrc, &kds, rt_row);
    let pipe = Pipeline::new(Cadence::OneWindow, nkv, kv_len, 128, Ext::ka(0x10).ceil_div(128));
    let mut body = Demo3v { base: Demo3 { ks, vs, kd: kds, kcp, vcp, kdcp, lane, next_reads: true }, kdv, pending: None };
    pipe.run(&mut b, Role::All, &mut body);
    let f = b.finish();
    let e = errs(&f);
    assert!(e.is_empty(), "{}", e.join("\n"));
    let c = amdgpu_codegen::compile(&f).expect("compile");
    let n = |m: &str| c.asm.lines().filter(|l| l.trim_start().starts_with(m)).count();
    assert_eq!(n("global_load_lds_dwordx4"), 3 * 8, "K + V per stage, 3 stages (2 prologue + loop)");
    assert_eq!(n("ds_write_b32"), 3, "the K-scale dword of each stage");
}

#[test]
fn counted_vmcnt_keeps_the_next_stage_in_flight() {
    // one stage = 1 (K scales) + 4 (K) + 4 (V) DMAs per wave: vmcnt(9) at the window end completes tile t+1 and leaves tile t+2 in flight
    // (a window that consumes only its own tile t: tile t+1 is published by this window's barrier and read in the next window)
    let e = errs(&demo3_vm(9, false));
    assert!(e.is_empty(), "vmcnt(9) completes the published stage: {}", e.join("\n"));
    // vmcnt(10) does not prove tile t+1 complete before its publishing barrier
    let e = errs(&demo3_vm(10, false));
    assert!(e.iter().any(|x| x.contains("GV-TILE") || x.contains("GV-PROTO")), "vmcnt(10) must be rejected: {e:?}");
    // the FMHA window reads tile t+1 in window t: tile t+1 must be complete at the end of window t-1, so vm_inflight > 0 is rejected there
    let e = errs(&demo3_vm(9, true));
    assert!(e.iter().any(|x| x.contains("GV-TILE") || x.contains("GV-PROTO")), "vmcnt(10) must be rejected: {e:?}");
}