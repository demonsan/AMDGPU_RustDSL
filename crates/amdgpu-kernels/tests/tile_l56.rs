//! Tile library layers 5-6: a complete one-group attention kernel written ONLY with library components (tensors, LDS plan, DMA copies, pipeline, QK/PV
//! tiles, online softmax with rollback, fp8 P conversion with and without the LUT correction, smfmac row sums, bf16 / KV-split partial epilogues), and
//! the KV-split combine rewritten with `combine_splits`. Every kernel: gpu_verify 0 errors, compiles, and has the FMHA instruction signature.
use amdgpu_codegen::gpu_verify::{analyse, Severity};
use amdgpu_tile::epilogue::*;
use amdgpu_tile::frag::*;
use amdgpu_tile::reduce::KeyMask;
use amdgpu_tile::softmax::*;
use amdgpu_tile::*;
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};

struct Body {
    ks: Shared,
    vs: Shared,
    kd: Shared,
    kcp: RowsCopy,
    vcp: RowsCopy,
    kdcp: DwordRowsCopy,
    kr: RowsReader,
    vr: TrReader,
    kdr: RowScaleReader,
    sel: ScaleSel,
    q: RowFrags,
    sq: [ValueId; 2],
    sm: OnlineSoftmax,
    zero: ValueId,
    g4: ValueId,
    kv_len: ValueId,
    c127: ValueId,
    c123: ValueId,
    s_init: Vec<ValueId>,
    s: Option<AccTile<C32x32F32, 4>>,
    o: Option<AccTile<C32x32F32, 4>>,
    p: Vec<ValueId>,
}

impl PipeBody for Body {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &I32, slot: &SlotIdx) {
        self.kdcp.issue(b, key0, &self.kd.at(slot));
        self.kcp.issue(b, key0, &self.ks.at(slot));
        self.vcp.issue(b, key0, &self.vs.at(slot));
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        let k = self.kr.at(b, &self.ks.at(&t0.cur));
        let kd = self.kdr.at(b, &self.kd.at(&t0.cur));
        self.s_init = qk_tile(b, QkOut::Values(self.zero), Some((self.sm.ones_f16, self.sm.bias_init)), &k, &kd, &self.q.frags, self.sq, &self.sel);
    }
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let mut v: Vec<(DType, Opnd)> = self.s_init.iter().map(|x| (DType::Frag16, Opnd::Val(*x))).collect();
        v.extend((0..4).map(|_| (DType::Frag16, Opnd::Val(self.zero))));
        v.extend(self.sm.state());
        v
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        self.s = Some(AccTile::from_vars(&vars[0..4]));
        self.o = Some(AccTile::from_vars(&vars[4..8]));
        self.sm.bind(&vars[8..10]);
    }
    fn v_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (s, o) = (self.s.unwrap(), self.o.unwrap());
        let mask = if t.masked { Some(KeyMask::new(b, t, self.g4, self.kv_len)) } else { None };
        let nb = self.sm.decide(b, &s, &o, mask.as_ref());
        self.p = self.sm.convert(b, &s, mask.as_ref(), 0, 4);
        self.p.push(nb);
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (s, o) = (self.s.unwrap(), self.o.unwrap());
        let nb = self.p[16];
        let pa = Frag::<B32x32x64Fp8>::from_words(b, &self.p[0..8]);
        let pb = Frag::<B32x32x64Fp8>::from_words(b, &self.p[8..16]);
        self.sm.row_sum(b, pa.v, pb.v);
        let v = self.vr.at(b, &self.vs.at(&t.cur));
        pv_tile(b, &o, &v, &[pa, pb], self.c127, self.c123);
        if t.staged {
            let k = self.kr.at(b, &self.ks.at(&t.next));
            let kd = self.kdr.at(b, &self.kd.at(&t.next));
            qk_tile(b, QkOut::Vars(&s), Some((self.sm.ones_f16, nb)), &k, &kd, &self.q.frags, self.sq, &self.sel);
        }
    }
}

/// `split`: KV-split partial epilogue (fp16 Ohat through the private LDS scratch + lse) instead of the bf16 output.
fn attention(lut: bool, split: bool) -> Function {
    let mut b = FnBuilder::new("tile_demo_attention", 256);
    b.compile_profile(CompileProfile::Default);
    let mut ka = KArgs::new();
    let skv = ka.dim(&mut b, "kv_len", 0x40, 1, i64::MAX);
    let kv_len = ka.i32(&mut b, "kv_len", 0x40);
    let rows = |n: &str, off: u32, ka: &mut KArgs, b: &mut FnBuilder, d: Dim, eb: i64, w: i64| {
        ka.tensor(n, off, eb, vec![d, Dim::c(w)], vec![Stride::Derived(I32::c((w * eb) as i32)), Stride::Unit]).packed().declare(ka, b)
    };
    let k = rows("ptr_k", 0x00, &mut ka, &mut b, skv.clone(), 1, 128);
    let v = rows("ptr_v", 0x08, &mut ka, &mut b, skv.clone(), 1, 128);
    let kd = rows("ptr_kd", 0x10, &mut ka, &mut b, skv, 4, 1);
    let q = rows("ptr_q", 0x18, &mut ka, &mut b, Dim::c(128), 1, 128);
    let o = rows("ptr_o", 0x20, &mut ka, &mut b, Dim::c(128), 2, 128);
    let lse_t = rows("ptr_lse", 0x28, &mut ka, &mut b, Dim::c(128), 4, 1);
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(2, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
    let kds = lds.ring("KD", 512, 2, Layout::Linear);
    // the epilogue scratch reuses V slot 0 (dead after the last PV and the barrier below)
    let epi = lds.private("epi", 4096, 4, &[&vs]);
    lds.commit(&mut b);
    let lane = Lane::new(&mut b);
    let roles = Roles::single(&lane, 4);
    let qrow = b.bind_i32((lane.wave() << 5) + lane.j());
    let qoff = b.bind_i32(q.tile(vec![I32::c(0), I32::c(0)]).offset(&[ev(qrow), lane.half() << 4]));
    let qf = RowFrags::load(&mut b, q.ptr, qoff);
    let sq = [b.c_i32(118), b.c_i32(118)];
    let kv_last = b.let_i32(ev(kv_len) - 1);
    let nkv = b.bind_i32((ev(kv_len) + 127).lshr(7));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x40).ceil_div(128), "nkv = ceil(kv_len / 128) >= 1");
    let src = |t: &Tensor, b: &mut FnBuilder| t.tile(vec![I32::c(0), I32::c(0)]).clamp(0, ev(kv_len), kv_last.clone()).row_source(b, 0);
    let (ksrc, vsrc, kdsrc) = (src(&k, &mut b), src(&v, &mut b), src(&kd, &mut b));
    let kcp = RowsCopy::new(&mut b, ksrc, &ks, roles.wave_in_group, &lane, 4);
    let vcp = RowsCopy::new(&mut b, vsrc, &vs, roles.wave_in_group, &lane, 4);
    let kdcp = DwordRowsCopy::new(&mut b, kdsrc, &kds, roles.wave_in_group, &lane);
    let kr = RowsReader::new(&mut b, &ks, &lane);
    let vr = TrReader::new(&mut b, &vs, &lane);
    let kdr = RowScaleReader::new(&mut b, &kds, &lane);
    let sel = ScaleSel::new(&mut b, &lane);
    let sm = OnlineSoftmax::new(&mut b, &lane, Recipe::vsx(lut));
    let zero = b.frag_zero(DType::Frag16);
    let g4 = b.bind_i32(C32x32F32::half_base(&lane));
    let (c127, c123) = (b.c_i32(127), b.c_i32(123));
    let mut body = Body { ks, vs, kd: kds, kcp, vcp, kdcp, kr, vr, kdr, sel, q: qf, sq, sm, zero, g4, kv_len, c127, c123, s_init: vec![], s: None, o: None, p: vec![] };
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, nkv, kv_len, 128, Ext::ka(0x40).ceil_div(128));
    pipe.run(&mut b, Role::All, &mut body);
    // ---- epilogue ----
    let l = body.sm.l_total(&mut b);
    let fin = b.rcp_f32(l);
    let o_acc = body.o.unwrap();
    let live = b.bind_cond(lane.wave().lt(4));
    if split {
        b.barrier(); // every wave is past its last V read before the scratch (V slot 0) is reused
        let lse = body.sm.lse(&mut b, l);
        let lse_ptr = lse_t.tile(vec![I32::c(0), I32::c(0)]).ptr(&mut b, &[ev(qrow), I32::c(0)]);
        let me = epi.mine(lane.wave);
        let (ot, lanev) = (o.tile(vec![I32::c(0), I32::c(0)]), lane);
        kvsplit_partial_f16(&mut b, &o_acc, fin, &lane, &me, lse_ptr, lse, live, &mut |b, i| {
            let r = b.bind_i32((lanev.wave() << 5) + (8 * i as i32) + lanev.lane().lshr(3));
            ot.ptr(b, &[ev(r), (lanev.lane() & 7) << 3])
        });
    } else {
        let optr = o.tile(vec![I32::c(0), I32::c(0)]).ptr(&mut b, &[ev(qrow), lane.half() << 2]);
        store_bf16(&mut b, &o_acc, fin, optr, live);
    }
    b.finish()
}

fn check(f: &Function) -> String {
    let r = analyse(f).expect("analyse");
    let e: Vec<String> = r.findings.iter().filter(|x| x.severity == Severity::Error).map(|x| format!("[{}] {}", x.code, x.msg)).collect();
    assert!(e.is_empty(), "{}: {}", f.name, e.join("\n"));
    amdgpu_codegen::compile(f).expect("compile").asm
}

fn n(asm: &str, m: &str) -> usize {
    asm.lines().filter(|l| l.trim_start().starts_with(m)).count()
}

#[test]
fn library_attention_bf16_verifies_and_has_the_fmha_signature() {
    for lut in [false, true] {
        let asm = check(&attention(lut, false));
        assert_eq!(n(&asm, "v_smfmac_f32_16x16x128_fp8_fp8"), 3 * 4, "row sums: loop + 2 tail copies");
        assert!(n(&asm, "v_max3_f32") >= 3 * 31, "row max chain");
        assert!(n(&asm, "v_pk_mul_f32") > 0 && n(&asm, "v_pk_add_f32") > 0, "rollback rescale");
        assert_eq!(n(&asm, "global_store_dwordx2"), 16, "4 x 4 eight-byte stores per lane (the shipped _vs128 has 4 code copies = 64)");
        assert_eq!(n(&asm, "v_perm_b32") > 0, lut, "the LUT correction only in the lut copy");
    }
}

#[test]
fn library_attention_kvsplit_epilogue_verifies() {
    let asm = check(&attention(false, true));
    assert_eq!(n(&asm, "global_store_dwordx4"), 8);
    assert_eq!(n(&asm, "ds_write_b32"), 32);
}

fn combine8h_lib() -> Function {
    let mut b = FnBuilder::new("tile_combine8h", 256);
    b.compile_profile(CompileProfile::Combine);
    let mut ka = KArgs::new();
    let ohat = ka.ptr(&mut b, "ptr_ohat", 0x00);
    let lse = ka.ptr(&mut b, "ptr_lse", 0x08);
    let out = ka.ptr(&mut b, "ptr_out", 0x10);
    let rows = ka.dim(&mut b, "rows_total", 0x18, 1, i64::MAX);
    ka.arg(&mut b, "d", DType::I32, 0x1C, 4);
    b.ka_range(0x1C, 128, 128);
    let splits = ka.dim(&mut b, "splits", 0x20, 1, i64::MAX);
    let (rext, sext) = (rows.ext(), splits.ext());
    b.arg_tensor("ptr_ohat", sext.clone() * rext.clone() * Ext::c(128 * 4));
    b.arg_tensor("ptr_lse", sext.clone() * rext.clone() * Ext::c(4));
    b.arg_tensor("ptr_out", rext * Ext::c(128 * 2));
    let (rv, sv) = (ka.i32(&mut b, "rows_total", 0x18), ka.i32(&mut b, "splits", 0x20));
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let t_raw = b.let_i32((ev(bx) << 8) + ev(tid));
    let total = b.let_i32(ev(rv) << 4);
    let t = b.bind_i32(t_raw.lt(&total).select_i32(&t_raw, &total - 1));
    let row = b.bind_i32(ev(t).lshr(4));
    b.assume_trusted(row, Ext::c(0), Ext::ka(0x18) - Ext::c(1), "row = clamped thread index >> 4 <= rows - 1");
    combine_splits(&mut b, ohat, lse, out, rv, sv, t, row, 8, Ext::ka(0x20) - Ext::c(1));
    b.finish()
}

#[test]
fn combine8h_rewritten_with_the_library_matches_the_shipped_instruction_mix() {
    let mine = check(&combine8h_lib());
    let shipped = amdgpu_codegen::compile(&amdgpu_kernels::kernel_by_name("fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h").unwrap()).unwrap().asm;
    for m in ["global_load_dwordx4", "global_load_dword ", "global_store_dwordx4", "v_exp_f32", "v_cvt_f32_f16", "v_cvt_pk_bf16_f32", "v_rcp_f32", "v_max_f32", "s_cbranch"] {
        assert_eq!(n(&mine, m), n(&shipped, m), "{m}\n--- library:\n{mine}\n--- shipped:\n{shipped}");
    }
    let (lm, ls) = (mine.lines().count() as i64, shipped.lines().count() as i64);
    assert!((lm - ls).abs() * 10 <= ls, "total size {lm} vs shipped {ls}");
}
