//! Fused RMSNorm + MXFP4 quantisation (aiter `fused_rms_mxfp4_quant`): `rmsnorm_quant_mxfp4_bf16_k<K>` = `quant_mxfp4_bf16_k<K>` (kernel-opt's rule, layouts and conversion, `rows/mxquant.rs`) applied to
//! `y = x * rsqrt(mean(x^2) + eps) * gamma` without the bf16 round trip of the normalised row. The group step (amax rule, E8M0, fp4 conversion) is `mxquant::mxfp4_group`, shared, not copied.
//!
//! One block (256 threads = 4 waves) per 16-row block and the WHOLE row (the wave's lane `16 g + r` owns row `16 rb + r` and the groups `{16 p + g + 4 j}` of every tile pair `p = wave + 4 i`, as in
//! `quant_mxfp4_bf16_k`): pass A sums the squares of the lane's groups (a runtime loop, 32 elements per step), the four lanes of a row (`l ^ 16`, `l ^ 32`) combine through two lane swaps, the four
//! waves through 256 bytes of LDS; pass B re-reads the row (L2), scales with `inv_rms * gamma`, and quantises exactly like `quant_mxfp4_bf16_k` (amax rounded at mantissa bit 21, e8m0 = exponent - 2,
//! `v_cvt_scalef32_pk_fp4_f32`, the SHUFFLED a4w4 scale layout). K % 512 == 0, 512 <= K <= 16384, M % 16 == 0.
//! ABI: x @0 (bf16 [M][K]), gamma @8 (bf16 [K]), q @16 (fp4 [M][K / 2]), scale @24 (e8m0, shuffled [M / 16][K / 512][64][4]), M @32 (i32), eps @36 (f32). Grid M / 16.

use super::mxquant::mxfp4_group;
use amdgpu_dsl::{CompileProfile, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::prims::{bf16x8_to_f32, rsqrt_f32};
use amdgpu_tile::{ev, Lane};

pub fn by_name(name: &str) -> Option<Function> {
    let r = name.strip_prefix("rmsnorm_quant_mxfp4_bf16_k")?;
    let k: i64 = r.parse().ok()?;
    (k % 512 == 0 && (512..=16384).contains(&k)).then(|| rmsnorm_mxfp4(name, k))
}

/// sum over the four lanes `{r, r + 16, r + 32, r + 48}` of a row (steps 1 and 2 of `tileop::wave_allreduce`: `l ^ 32`, then `l ^ 16`); every lane gets the sum.
fn rows_sum(b: &mut FnBuilder, x: ValueId, lane: ValueId) -> ValueId {
    let rb = b.bitcast_i32(x);
    let (na, nb) = b.permlane32_swap(rb, rb);
    let (na, nb) = (b.let_(Op::BitcastI32(Opnd::Val(na))), b.let_(Op::BitcastI32(Opnd::Val(nb))));
    let low = b.cmp(CmpPred::Lt, CmpTy::I32, lane, 32);
    let p1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(low), a: Opnd::Val(nb), b: Opnd::Val(na) });
    let p1 = b.bitcast_f32(p1);
    let v = b.add_f32(x, p1);
    let vb = b.bitcast_i32(v);
    let f2 = b.permlane16_swap(vb, vb);
    let (xa, xb) = (b.frag2_lane(f2, 0), b.frag2_lane(f2, 1));
    let (xa, xb) = (b.let_(Op::BitcastI32(Opnd::Val(xa))), b.let_(Op::BitcastI32(Opnd::Val(xb))));
    let l16 = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(16)));
    let even = b.cmp(CmpPred::Eq, CmpTy::I32, l16, 0);
    let p2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(even), a: Opnd::Val(xb), b: Opnd::Val(xa) });
    let p2 = b.bitcast_f32(p2);
    b.add_f32(v, p2)
}

fn rmsnorm_mxfp4(name: &str, k: i64) -> Function {
    let kt = k / 256;
    let tp = kt / 2;                                   // tile pairs per row
    let ni = (tp + 3) / 4;                             // tile pairs per wave
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, o) in [("x", 0u32), ("gamma", 8), ("q", 16), ("scale", 24)] {
        b.arg(n, DType::Ptr, o, 8);
    }
    b.arg("M", DType::I32, 32, 4);
    b.arg("eps", DType::F32, 36, 4);
    b.sym_range("MB", 1, 1 << 14);
    let m = Ext::sym("MB") * Ext::c(16);
    b.ka_is(32, m.clone());
    b.grid_is(0, Ext::sym("MB"));
    b.arg_tensor("x", m.clone() * Ext::c(2 * k));
    b.arg_tensor("gamma", Ext::c(2 * k));
    b.arg_tensor("q", m.clone() * Ext::c(k / 2));
    b.arg_tensor("scale", m * Ext::c(k / 32));
    // the four-wave row-sum exchange (16 rows x 4 waves x f32) is a declared tile: written once per wave slice, read after the barrier
    let mut lds = amdgpu_tile::LdsPlan::new();
    let rs_tile = lds.tile("RS", 256, amdgpu_tile::Layout::Linear);
    lds.commit(&mut b);
    let rs_slot = amdgpu_tile::SlotIdx::new(0, 0);
    let kp = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargPtr { offset: o });
    let (p_x, p_g, p_q, p_s) = (kp(&mut b, 0), kp(&mut b, 8), kp(&mut b, 16), kp(&mut b, 24));
    let m_v = b.let_(Op::KernargI32 { offset: 32 });
    let eps = b.let_(Op::KernargF32 { offset: 36 });
    let rb = b.let_(Op::WorkgroupIdX);
    let lane = Lane::new(&mut b);
    let lane_v = b.bind_i32(lane.lane());
    let r = b.bind_i32(ev(lane_v) & 15);
    let g = b.bind_i32(ev(lane_v).lshr(4));
    let wave = b.bind_i32(lane.wave());
    let row = b.bind_i32((ev(rb) << 4) + ev(r));
    let nx = { let v = b.bind_i32(ev(m_v) * (2 * k as i32)); b.read_first_lane(v) };
    let nq = { let v = b.bind_i32(ev(m_v) * (k as i32 / 2)); b.read_first_lane(v) };
    let ns = { let v = b.bind_i32(ev(m_v) * (k as i32 / 32)); b.read_first_lane(v) };
    let (r_x, r_q, r_s) = (b.buffer_desc_dyn(p_x, nx), b.buffer_desc_dyn(p_q, nq), b.buffer_desc_dyn(p_s, ns));
    let r_g = b.buffer_desc(p_g, (2 * k) as u32);
    let c0 = b.c_i32(0);
    let tpc = b.c_i32(tp as i32);
    // ---- pass A: this lane's partial sum of squares (a runtime loop over its tile pairs, the 4 groups of a pair unrolled: 16 loads in flight) ----
    let ss = b.loop_(&[(DType::F32, Opnd::F32(0.0)), (DType::I32, Opnd::I32(0))], |b, vv| {
        let (vs, vit) = (vv[0], vv[1]);
        let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::I32(ni as i32)));
        b.break_if(done);
        let i = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
        let acc0 = b.let_(Op::AddF32(Opnd::Var(vs), Opnd::F32(0.0)));
        let p = b.bind_i32(ev(wave) + 4 * ev(i));
        let valid = b.cmp(CmpPred::Lt, CmpTy::I32, p, tp as i32);
        let pc = b.bind_i32(ev(p).min(tp as i32 - 1));
        let mut loaded: Vec<ValueId> = Vec::new();
        for j in 0..4i32 {
            let grp = b.bind_i32((ev(pc) << 4) + ev(g) + 4 * j);
            let xo = b.bind_i32((ev(row) * (k as i32) + (ev(grp) << 5)) << 1);
            for c in 0..4u32 {
                loaded.push(b.buffer_load_b128(r_x, xo, c0, 16 * c));
            }
        }
        let mut sq: Option<ValueId> = None;
        for f in loaded {
            for x in bf16x8_to_f32(b, f) {
                let t = b.mul_f32(x, x);
                sq = Some(match sq { None => t, Some(a) => b.add_f32(a, t) });
            }
        }
        let zero = b.c_f32(0.0);
        let add = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(valid), a: Opnd::Val(sq.unwrap()), b: Opnd::Val(zero) });
        let acc = b.add_f32(acc0, add);
        b.set_var(vs, acc);
        let nx2 = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
        b.set_var(vit, nx2);
        vs
    });
    let part = b.let_(Op::AddF32(Opnd::Var(ss), Opnd::F32(0.0)));
    let row_sum = rows_sum(&mut b, part, lane_v);
    // the four waves: 16 rows x 4 partial sums through LDS
    let slot = b.bind_i32(((ev(wave) << 4) + ev(r)) << 2);
    let rs_i = b.bitcast_i32(row_sum);
    rs_tile.at(&rs_slot).write(&mut b, |b| b.store_lds(slot, rs_i, MemTy::B32, false));
    b.wait_lds(0);
    b.barrier();
    let mut tot = b.c_f32(0.0);
    for w in 0..4i32 {
        let a = b.bind_i32(((ev(r)) << 2) + 64 * w);
        let v = rs_tile.at(&rs_slot).read(&mut b, |b| b.load_lds(a, MemTy::B32));
        b.wait_lds(0);
        let vf = b.bitcast_f32(v);
        tot = b.add_f32(tot, vf);
    }
    let inv_k = b.c_f32(1.0 / k as f32);
    let mean = b.mul_f32(tot, inv_k);
    let me = b.add_f32(mean, eps);
    let inv = rsqrt_f32(&mut b, me);
    // ---- pass B: scale, quantise (as quant_mxfp4_bf16_k), per tile pair of this wave ----
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vv| {
        let vit = vv[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::I32(ni as i32)));
        b.break_if(done);
        let i = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
        let p = b.bind_i32((ev(wave) + 4 * ev(i)).min(tp as i32 - 1));   // waves past the last pair redo it: same values
        let mut sbytes: Vec<ValueId> = Vec::with_capacity(4);
        for j in 0..4i32 {
            let grp = b.bind_i32((ev(p) << 4) + ev(g) + 4 * j);
            let xo = b.bind_i32((ev(row) * (k as i32) + (ev(grp) << 5)) << 1);
            let go = b.bind_i32(ev(grp) << 6);
            let mut ys: Vec<ValueId> = Vec::with_capacity(32);
            for c in 0..4u32 {
                let f = b.buffer_load_b128(r_x, xo, c0, 16 * c);
                let gf = b.buffer_load_b128(r_g, go, c0, 16 * c);
                let (xv, gv) = (bf16x8_to_f32(b, f), bf16x8_to_f32(b, gf));
                for q in 0..8 {
                    let t = b.mul_f32(xv[q], inv);
                    ys.push(b.mul_f32(t, gv[q]));
                }
            }
            let (e, f4) = mxfp4_group(b, &ys);
            sbytes.push(e);
            let qo = b.bind_i32(ev(row) * (k as i32 / 2) + (ev(grp) << 4));
            b.buffer_store_b128(r_q, qo, c0, 0, f4);
        }
        let w = b.bind_i32(ev(sbytes[0]) + (ev(sbytes[1]) << 8) + (ev(sbytes[2]) << 16) + (ev(sbytes[3]) << 24));
        let so = b.bind_i32((((ev(rb) * tp as i32) + ev(p)) * 64 + ev(lane_v)) << 2);
        let wf = b.bitcast_f32(w);
        b.buffer_store_f32(r_s, so, c0, 0, wf);
        let nx2 = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
        b.set_var(vit, nx2);
    });
    let _ = tpc;
    b.finish()
}