//! Tile library layer 4: fragments with layouts and tile-level MFMA. A one-group pipeline (4 waves stage K, K scales and V; QK(0) in the prologue, PV(t) +
//! QK(t+1) in the matrix phase) written with the library: gpu_verify 0 errors, and the emitted instruction mix is the FMHA one (8 scaled MFMAs + 4 f16 bias
//! MFMAs per QK tile, 8 per PV tile, 2 x ds_read_b128 per K fragment pair, 4 x ds_read_b64_tr_b8 per V fragment).
use amdgpu_compiler::gpu_verify::{analyse, Severity};
use amdgpu_tile::frag::*;
use amdgpu_tile::*;
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Opnd, ValueId, VarId, I32};

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
    ones: ValueId,
    bias: ValueId,
    zero: ValueId,
    s_init: Vec<ValueId>,
    s: Option<AccTile<C32x32F32, 4>>,
    o: Option<AccTile<C32x32F32, 4>>,
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
        self.s_init = qk_tile(b, QkOut::Values(self.zero), Some((self.ones, self.bias)), &k, &kd, &self.q.frags, self.sq, &self.sel);
    }
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let mut v: Vec<(DType, Opnd)> = self.s_init.iter().map(|x| (DType::Frag16, Opnd::Val(*x))).collect();
        v.extend((0..4).map(|_| (DType::Frag16, Opnd::Val(self.zero))));
        v
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        self.s = Some(AccTile::from_vars(&vars[0..4]));
        self.o = Some(AccTile::from_vars(&vars[4..8]));
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        // P = the first 8 words of S (a stand-in for the softmax of L5): the PV MFMAs and their V reads are what is measured here
        let s = self.s.unwrap();
        let w: Vec<ValueId> = (0..8u8).map(|r| b.frag16_lane(Opnd::Var(s.accs[0].var), r)).collect();
        let p = Frag::<B32x32x64Fp8>::from_words(b, &w);
        let v = self.vr.at(b, &self.vs.at(&t.cur));
        let c127 = b.c_i32(127);
        pv_tile(b, &self.o.unwrap(), &v, &[p, p], c127, c127);
        if t.staged {
            let k = self.kr.at(b, &self.ks.at(&t.next));
            let kd = self.kdr.at(b, &self.kd.at(&t.next));
            qk_tile(b, QkOut::Vars(&s), Some((self.ones, self.bias)), &k, &kd, &self.q.frags, self.sq, &self.sel);
        }
    }
}

fn demo() -> Function {
    let mut b = FnBuilder::new("tile_demo_qkpv", 256);
    b.compile_profile(CompileProfile::Default);
    let mut ka = KArgs::new();
    let skv = ka.dim(&mut b, "kv_len", 0x20, 1, i64::MAX);
    let kv_len = ka.i32(&mut b, "kv_len", 0x20);
    let row_t = |n: &str, off: u32, ka: &mut KArgs, b: &mut FnBuilder, eb: i64, d: i64| {
        ka.tensor(n, off, eb, vec![skv.clone(), Dim::c(d)], vec![Stride::Derived(I32::c((d * eb) as i32)), Stride::Unit]).packed().declare(ka, b)
    };
    let k = row_t("ptr_k", 0x00, &mut ka, &mut b, 1, 128);
    let v = row_t("ptr_v", 0x08, &mut ka, &mut b, 1, 128);
    let kd = row_t("ptr_kd", 0x10, &mut ka, &mut b, 4, 1);
    // 128 query rows (4 waves x 32) of Q and their scale words
    let q = ka.tensor("ptr_q", 0x18, 1, vec![Dim::c(128), Dim::c(128)], vec![Stride::Derived(I32::c(128)), Stride::Unit]).packed().declare(&mut ka, &mut b);
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(2, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
    let kds = lds.ring("KD", 512, 2, Layout::Linear);
    lds.commit(&mut b);
    let lane = Lane::new(&mut b);
    let roles = Roles::single(&lane, 4);
    let qrow = b.bind_i32((lane.wave() << 5) + lane.j());
    let qoff = b.bind_i32(q.tile(vec![I32::c(0), I32::c(0)]).offset(&[ev(qrow), lane.half() << 4]));
    let qf = RowFrags::load(&mut b, q.ptr, qoff);
    let sq = [b.c_i32(127), b.c_i32(127)];
    let kv_last = b.let_i32(ev(kv_len) - 1);
    let nkv = b.bind_i32((ev(kv_len) + 127).lshr(7));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x20).ceil_div(128), "nkv = ceil(kv_len / 128) >= 1");
    let src = |t: &Tensor, b: &mut FnBuilder| t.tile(vec![I32::c(0), I32::c(0)]).clamp(0, ev(kv_len), kv_last.clone()).row_source(b, 0);
    let (ksrc, vsrc, kdsrc) = (src(&k, &mut b), src(&v, &mut b), src(&kd, &mut b));
    let kcp = RowsCopy::new(&mut b, ksrc, &ks, roles.wave_in_group, &lane, 4);
    let vcp = RowsCopy::new(&mut b, vsrc, &vs, roles.wave_in_group, &lane, 4);
    let kdcp = DwordRowsCopy::new(&mut b, kdsrc, &kds, roles.wave_in_group, &lane);
    let kr = RowsReader::new(&mut b, &ks, &lane);
    let vr = TrReader::new(&mut b, &vs, &lane);
    let kdr = RowScaleReader::new(&mut b, &kds, &lane);
    let sel = ScaleSel::new(&mut b, &lane);
    let oh = b.c_i32(0x3C003C00);
    let ones = b.frag2(oh, oh);
    let wi = b.c_i32(0x48004800);
    let bias = b.frag2(wi, wi);
    let zero = b.frag_zero(DType::Frag16);
    let mut body = Body { ks, vs, kd: kds, kcp, vcp, kdcp, kr, vr, kdr, sel, q: qf, sq, ones, bias, zero, s_init: vec![], s: None, o: None };
    let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, nkv, kv_len, 128, Ext::ka(0x20).ceil_div(128));
    pipe.prebind = vec![32768, 512];
    pipe.run(&mut b, Role::All, &mut body);
    b.finish()
}

#[test]
fn qk_pv_tiles_verify_and_emit_the_fmha_instruction_mix() {
    let f = demo();
    let r = analyse(&f).expect("analyse");
    let e: Vec<String> = r.findings.iter().filter(|x| x.severity == Severity::Error).map(|x| format!("[{}] {}", x.code, x.msg)).collect();
    assert!(e.is_empty(), "{}", e.join("\n"));
    let c = amdgpu_compiler::compile(&f).expect("compile");
    let n = |m: &str| c.asm.lines().filter(|l| l.trim_start().starts_with(m)).count();
    // QK: prologue + loop = 2 x (8 scaled + 4 f16); PV: loop + 2 tail copies = 3 x 8 scaled
    assert_eq!(n("v_mfma_scale_f32_32x32x64_f8f6f4"), 2 * 8 + 3 * 8, "{}", c.asm);
    assert_eq!(n("v_mfma_f32_32x32x8_f16"), 2 * 4);
    assert_eq!(n("ds_read_b64_tr_b8"), 3 * 2 * 4 * 4);
    assert_eq!(n("ds_read_b128"), 2 * 4 * 4);
    assert_eq!(n("ds_read_b32"), 2 * 4);
}
