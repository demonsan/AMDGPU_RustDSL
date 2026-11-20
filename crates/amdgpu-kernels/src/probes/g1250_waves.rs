//! gfx1250 two-waves-per-SIMD microbenchmark: 256 work-items = 8 waves, waves w and w + 4 share a SIMD. Mask `<hex g1 g0>`: the blocks each wave group runs, group 0 = waves 0 - 3, group 1 = waves 4 - 7:
//! bit 0 = a WMMA block (2 A fragments + 16 B fragments from LDS, 16 fp8 16x16x64 WMMAs into 8 accumulators), bit 1 = a VALU block (8 independent mul + exp2 chains, 8 rounds = 128
//! instructions). `01` = one wave per SIMD doing WMMA; `02` VALU alone; `03` one wave doing both (the single-wave baseline); `21` = WMMA wave + VALU wave on each SIMD. Kernargs: out @0 (u32 per
//! wave: loop cycles, then a digest), iters @8. Host `tests/gpu/g1250_waves_host.cpp`.

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId, WmmaKind};

/// `probe_g1250_waves_<hex mask>` (two hex digits: group 1 blocks, group 0 blocks).
pub fn by_mask(hex: &str) -> Option<Function> {
    let mask = u32::from_str_radix(hex, 16).ok()?;
    if mask > 0xff {
        return None;
    }
    Some(waves(&format!("probe_g1250_waves_{hex}"), mask))
}

fn waves(name: &str, mask: u32) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg("iters", DType::I32, 8, 4);
    b.sym_range("ITERS", 1, 1 << 20);
    b.ka_is(8, Ext::sym("ITERS"));
    b.arg_tensor("out", Ext::grid_x() * Ext::c(2048));
    b.lds(65536);
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let iters = b.let_(Op::KernargI32 { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let wg = b.let_(Op::WorkgroupIdX);
    let wv = b.bind_i32(ev(tid) >> 5);
    let grp = b.bind_i32(ev(wv) >> 2);
    let lane = b.bind_i32(ev(tid) & 31);
    // some LDS content (every work-item writes its 32 bytes of both fragment areas)
    let t255 = b.bind_i32(ev(tid) & 255);
    let a0 = b.bind_i32(ev(t255) * 32);
    let zero = b.frag_cat(&[Opnd::I32(0); 4]);
    for k in 0..2i32 {
        let ad = b.bind_i32(ev(a0) + 16 * k);
        b.store_lds_b128(ad, zero);
        let ad2 = b.bind_i32(ev(a0) + (8192 + 16 * k));
        b.store_lds_b128(ad2, zero);
    }
    b.wait_lds(0);
    b.barrier();
    let zf = b.c_f32(0.0);
    let cs = b.c_f32(0.9990234);
    let sel = |g: u32| -> u32 { (mask >> (4 * g)) & 3 };
    for g in 0..2u32 {
        let blocks = sel(g);
        if blocks == 0 {
            continue;
        }
        let gs = b.read_first_lane(grp);
        let c = b.cmp(CmpPred::Eq, CmpTy::I32, gs, Opnd::I32(g as i32));
        let c = b.let_(Op::AnyLane(Opnd::Val(c)));
        b.begin_if(c);
        // loop state: 8 accumulators, 8 VALU chains
        let mut inits: Vec<(DType, Opnd)> = Vec::new();
        for _ in 0..8 {
            let z = b.frag_cat(&[Opnd::I32(0); 8]);
            inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
        }
        for _ in 0..8 {
            inits.push((DType::F32, Opnd::Val(zf)));
        }
        let st = b.declare_vars(&inits);
        let acc: Vec<_> = st[0..8].to_vec();
        let uv: Vec<_> = st[8..16].to_vec();
        let t0c = b.shader_cycles();
        let t0 = b.let_(Op::PtrLo(Opnd::Val(t0c)));
        let z0 = b.c_i32(0);
        let lv = b.begin_loop(&[(DType::I32, Opnd::Val(z0))]);
        let it = lv[0];
        b.assume(it, Ext::c(0), Ext::sym("ITERS"));
        if blocks & 1 != 0 {
            let afr: Vec<ValueId> = (0..2)
                .map(|p| {
                    let lo = b.bind_i32(ev(a0) + 4096 * p);
                    let hi = b.bind_i32(ev(lo) + 16);
                    b.load_lds_b128x2(lo, hi)
                })
                .collect();
            let bfr: Vec<Vec<ValueId>> = (0..2)
                .map(|p| {
                    (0..8)
                        .map(|nt| {
                            let lo = b.bind_i32(ev(a0) + (8192 + 512 * nt + 16384 * p));
                            let hi = b.bind_i32(ev(lo) + 16);
                            b.load_lds_b128x2(lo, hi)
                        })
                        .collect()
                })
                .collect();
            for nt in 0..8usize {
                for p in 0..2usize {
                    let d = b.wmma(WmmaKind::Fp8_16x16x64, afr[p], bfr[p][nt], Some(Opnd::Var(acc[nt])));
                    b.set_var(acc[nt], Opnd::Val(d));
                }
            }
        }
        if blocks & 2 != 0 {
            for _ in 0..8 {
                for i in 0..8usize {
                    let x = b.mul_f32(Opnd::Var(uv[i]), cs);
                    let y = b.exp2_f32(x);
                    b.set_var(uv[i], Opnd::Val(y));
                }
            }
        }
        let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        b.set_var(it, Opnd::Val(tn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, iters);
        b.break_if(fin);
        b.end_loop();
        let t1c = b.shader_cycles();
        let t1 = b.let_(Op::PtrLo(Opnd::Val(t1c)));
        let dt = b.bind_i32(ev(t1) - ev(t0));
        // per wave: cycles (word 0) and a digest so that nothing is dead
        let o = b.bind_i32(ev(wg) * 2048 + ev(wv) * 256);
        let pa = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o) });
        b.store_global(pa, 0, dt, MemTy::B32, false);
        let lo = b.bind_i32(ev(lane) * 4 + 4);
        let pb = b.let_(Op::PtrAdd { base: Opnd::Val(pa), byte_off: Opnd::Val(lo) });
        let e0 = b.frag_slice(Opnd::Var(acc[0]), 0, 1);
        b.store_global(pb, 0, e0, MemTy::B32, false);
        let mut sum = b.add_f32(Opnd::Var(uv[0]), Opnd::Var(uv[1]));
        for i in 2..8usize {
            sum = b.add_f32(sum, Opnd::Var(uv[i]));
        }
        b.store_global(pb, 124, sum, MemTy::B32, false);
        b.end_if();
    }
    b.finish()
}
