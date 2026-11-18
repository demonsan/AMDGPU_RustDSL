//! gfx1250 GEMM through the tensor data mover (G3 step 5 of docs/gfx1250-plan.md): C[M][N] (f32) = A[M][K] x B[N][K]^T with fp8 e4m3 operands, `v_wmma_f32_16x16x64_fp8_fp8`.
//!
//! The first form is hand-built and fully unrolled over the K tiles (the structure the stage template will generate): per workgroup a 32 x 32 tile of C
//! (4 waves, 2 x 2, each wave one 16 x 16 block), K tiles of 64 bytes, a 3-slot K ring in LDS. Per K tile, in program order:
//! ```text
//! prologue: issue(0) issue(1)
//! tile t:   wait_tensor(<= the next tile's loads)   (the tile t loads of THIS wave are done; TENSORcnt is in order)
//!           barrier                                  (publishes tile t to every wave; every wave has finished reading the slot of tile t - 1)
//!           issue(t + 2) into the slot of tile t - 1 (ring marker produce / end)
//!           consume(t): the fragments of tile t from its slot (marker consume / end) + the WMMA
//! ```
//! Every wave issues both tile loads (the same addresses): no cross-wave tensor completion is needed, only the barrier. A / B are row-major with K contiguous
//! (B is [N][K]); the lane layout of the fp8 16x16x64 operands is the `A8` rule of docs/target-gfx1250.md section 3.

use amdgpu_codegen::DEFAULT_SCALAR;
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, TdmCpol, TensorTile, ValueId, WmmaKind};
use amdgpu_stage::gemm::tdm_f4::F4Shape;


/// C tile rows / columns per workgroup, K bytes per tile, ring slots.
pub const BM: i32 = 32;
pub const BN: i32 = 32;
pub const KB: i32 = 64;
pub const SLOTS: usize = 3;
const SLOT_BYTES: i32 = (BM + BN) * KB;
const B_OFF: i32 = BM * KB;

/// The kernel name of the `kt`-tile form.
pub fn gemm_tdm_fp8_name(kt: usize) -> String {
    format!("gemm_g1250_tdm_fp8_k{}", kt as i32 * KB)
}

/// Kernargs: a @0, b @8, c @16 (pointers), k @24 (K elements = bytes per row of A and B), n @28 (columns of C). Grid (N / 32, M / 32), 128 work-items.
pub fn gemm_tdm_fp8(kt: usize) -> Function {
    assert!(kt >= 2, "the 3-slot ring needs two tiles in flight");
    let mut b = FnBuilder::new(&gemm_tdm_fp8_name(kt), 128);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("c", DType::Ptr, 16, 8);
    b.arg("k", DType::I32, 24, 4);
    b.arg("n", DType::I32, 28, 4);
    // the host contract (tensor bounds, docs/gfx1250-hang-rules.md): the grid covers C exactly (M = grid y x BM, N = grid x x BN); A / B hold M x K / N x K fp8 bytes
    b.ka_is(28, Ext::grid_x() * Ext::c(BN as i64));
    let (m_rows, n_cols) = (Ext::grid_y() * Ext::c(BM as i64), Ext::grid_x() * Ext::c(BN as i64));
    b.ka_is(24, Ext::c(kt as i64 * KB as i64));
    b.arg_tensor("a", m_rows.clone() * Ext::c(kt as i64 * KB as i64));
    b.arg_tensor("b", n_cols.clone() * Ext::c(kt as i64 * KB as i64));
    b.arg_tensor("c", m_rows * n_cols * Ext::c(4));
    b.lds((SLOTS as i32 * SLOT_BYTES) as u32);
    b.lds_ring("ring", 0, SLOT_BYTES as i64, SLOTS as u32);
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    let pc = b.let_(Op::KernargPtr { offset: 16 });
    let kk = b.let_(Op::KernargI32 { offset: 24 });
    let nn = b.let_(Op::KernargI32 { offset: 28 });
    let tid = b.let_(Op::WorkitemId);
    let wgx = b.let_(Op::WorkgroupIdX);
    let wgy = b.let_(Op::WorkgroupIdY);

    // lane roles: wave w = tid / 32 (2 x 2 waves: wm = w / 2 rows of 16, wn = w % 2 columns of 16), lane l: m = l % 16, g = l / 16
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let w = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let wm = b.let_(Op::LshrI32(Opnd::Val(w), Opnd::I32(1)));
    let wn = b.let_(Op::AndI32(Opnd::Val(w), Opnd::I32(1)));
    let m = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(15)));
    let g = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::I32(4)));

    // global tile origins (bytes): the first row of this workgroup's A rows / B rows
    let row0 = b.mul_i32(wgy, Opnd::I32(BM));
    let col0 = b.mul_i32(wgx, Opnd::I32(BN));
    let a_org = b.mul_i32(row0, kk);
    let b_org = b.mul_i32(col0, kk);

    // LDS fragment bases of this lane: row (16 w? + m) of the tile, K bytes 8 g + 16 j (j = 0..4) of the 64-byte row
    let rowa = b.let_(Op::ShlI32(Opnd::Val(wm), Opnd::I32(4)));
    let rowa = b.add_i32(rowa, m);
    let abase = b.let_(Op::ShlI32(Opnd::Val(rowa), Opnd::I32(6)));
    let g8 = b.let_(Op::ShlI32(Opnd::Val(g), Opnd::I32(3)));
    let abase = b.add_i32(abase, g8);
    let rowb = b.let_(Op::ShlI32(Opnd::Val(wn), Opnd::I32(4)));
    let rowb = b.add_i32(rowb, m);
    let bbase = b.let_(Op::ShlI32(Opnd::Val(rowb), Opnd::I32(6)));
    let bbase = b.add_i32(bbase, g8);
    let bbase = b.add_i32(bbase, Opnd::I32(B_OFF));

    let tile = TensorTile { data_size: 1, tensor_dim: [Opnd::I32(KB), Opnd::I32(BM)], stride0: Opnd::Val(kk), tile: [KB as u16, BM as u16], pad: None, wg_mask: 0 };
    let issue = |b: &mut FnBuilder, t: usize| {
        let slot = (t % SLOTS) as i32;
        let koff = t as i32 * KB;
        let oa = b.add_i32(a_org, Opnd::I32(koff));
        let ob = b.add_i32(b_org, Opnd::I32(koff));
        let (a0, a1) = b.tensor_desc_2d(&tile, Opnd::Val(pa), Opnd::Val(oa), Opnd::I32(slot * SLOT_BYTES));
        let (b0, b1) = b.tensor_desc_2d(&tile, Opnd::Val(pb), Opnd::Val(ob), Opnd::I32(slot * SLOT_BYTES + B_OFF));
        b.tile_produce("ring", Opnd::I32(slot), Opnd::I32((t / SLOTS) as i32));
        b.tensor_load_to_lds(Opnd::Val(a0), Opnd::Val(a1), None, TdmCpol::default());
        b.tensor_load_to_lds(Opnd::Val(b0), Opnd::Val(b1), None, TdmCpol::default());
        b.tile_end();
    };

    issue(&mut b, 0);
    issue(&mut b, 1);
    let mut acc: Option<Opnd> = None;
    for t in 0..kt {
        // this wave's loads of tile t are done once at most the next tile's two loads are outstanding
        b.wait_tensor(if t + 1 < kt { 2 } else { 0 });
        b.barrier();
        if t + 2 < kt {
            issue(&mut b, t + 2);
        }
        let slot = (t % SLOTS) as i32;
        b.tile_consume("ring", Opnd::I32(slot), Opnd::I32((t / SLOTS) as i32));
        let mut pa_parts: Vec<Opnd> = Vec::new();
        let mut pb_parts: Vec<Opnd> = Vec::new();
        for j in 0..4 {
            let aa = b.add_i32(abase, Opnd::I32(slot * SLOT_BYTES + 16 * j));
            pa_parts.push(Opnd::Val(b.load_lds_b64(aa)));
            let ba = b.add_i32(bbase, Opnd::I32(slot * SLOT_BYTES + 16 * j));
            pb_parts.push(Opnd::Val(b.load_lds_b64(ba)));
        }
        b.tile_end();
        let fa = b.frag_cat(&pa_parts);
        let fb = b.frag_cat(&pb_parts);
        let d = b.wmma(WmmaKind::Fp8_16x16x64, fa, fb, acc);
        acc = Some(Opnd::Val(d));
        // every wave has read the slot before the next barrier lets a load overwrite it
        b.wait_lds(0);
    }

    // C: lane -> column n0 + 16 wn + m, rows 16 wm + 8 g + r (r = 0..8)
    let acc = acc.expect("kt >= 2");
    let crow = b.let_(Op::ShlI32(Opnd::Val(wm), Opnd::I32(4)));
    let crow = b.add_i32(crow, row0);
    let crow = b.add_i32(crow, g8);
    let ccol = b.let_(Op::ShlI32(Opnd::Val(wn), Opnd::I32(4)));
    let ccol = b.add_i32(ccol, col0);
    let ccol = b.add_i32(ccol, m);
    let lin = b.mul_i32(crow, nn);
    let lin = b.add_i32(lin, ccol);
    let off0 = b.let_(Op::ShlI32(Opnd::Val(lin), Opnd::I32(2)));
    let row_bytes = b.let_(Op::ShlI32(Opnd::Val(nn), Opnd::I32(2)));
    for r in 0..8 {
        let off = if r == 0 { off0 } else { let rb = b.mul_i32(row_bytes, Opnd::I32(r)); b.add_i32(off0, rb) };
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(pc), byte_off: Opnd::Val(off) });
        let x = b.frag_slice(acc, r as u8, 1);
        b.store_global(q, 0, x, MemTy::B32, false);
    }
    let mut f = b.finish();
    // the tile origins and descriptor addresses are wave-uniform integers: SALU, and the descriptor operands are SGPR tuples
    // `dce`: the unused group 1 of the descriptors built per load is dead code (its constants must not occupy SGPRs)
    f.lower_opts.extend(["dce"].iter().map(|s| s.to_string()).chain(DEFAULT_SCALAR.iter().map(|s| s.to_string())));
    f
}

/// Group 0 of a tensor descriptor (valid, LDS address, the 57-bit global address of `base + byte_off`, type 2): the same words `FnBuilder::tensor_desc_2d` builds, without its group 1.
fn group0(b: &mut FnBuilder, base: ValueId, byte_off: ValueId, lds: i32) -> ValueId {
    let p = b.let_(Op::PtrAdd { base: Opnd::Val(base), byte_off: Opnd::Val(byte_off) });
    let lo = b.let_(Op::PtrLo(Opnd::Val(p)));
    let hi = b.let_(Op::PtrHi(Opnd::Val(p)));
    let hm = b.let_(Op::AndI32(Opnd::Val(hi), Opnd::I32(0x01ff_ffff)));
    let w3 = b.let_(Op::AddI32(Opnd::Val(hm), Opnd::I32(0x8000_0000u32 as i32)));
    b.let_(Op::RsrcCat { words: [Opnd::I32(1), Opnd::I32(lds), Opnd::Val(lo), Opnd::Val(w3)] })
}

/// The kernel name of the stage-template form.
pub const STAGE_NAME: &str = "gemm_g1250_tdm_stage_fp8";

/// The same GEMM written on the stage template (`amdgpu_stage::gemm::tdm::TdmGemmBody` driven by `Pipeline` with `Cadence::OneWindow`): the loop, the 3-slot ring arithmetic, the ring markers
/// and the barrier come from the pipeline driver; the producer part runs at iteration offset 2. Kernargs: a @0, b @8, c @16, k @24 (K bytes), n @28. Any K that is a multiple of 64.
pub fn gemm_tdm_stage_fp8() -> Function {
    use amdgpu_dsl::{ev, I32};
    use amdgpu_stage::gemm::tdm::TdmGemmBody;
    use amdgpu_stage::{Cadence, Pipeline};
    use amdgpu_tile::lane::Role;
    use amdgpu_tile::shared::{Layout, LdsPlan};
    let mut b = FnBuilder::new(STAGE_NAME, 128);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("c", DType::Ptr, 16, 8);
    b.arg("k", DType::I32, 24, 4);
    b.arg("n", DType::I32, 28, 4);
    // the host contract (tensor bounds, docs/gfx1250-hang-rules.md): the grid covers C exactly (M = grid y x BM, N = grid x x BN); A / B hold M x K / N x K fp8 bytes
    b.ka_is(28, Ext::grid_x() * Ext::c(BN as i64));
    let (m_rows, n_cols) = (Ext::grid_y() * Ext::c(BM as i64), Ext::grid_x() * Ext::c(BN as i64));
    // K = 64 x K64 (at least two K tiles)
    b.sym_range("K64", 2, 1 << 14);
    b.ka_is(24, Ext::sym("K64") * Ext::c(KB as i64));
    b.arg_tensor("a", m_rows.clone() * Ext::sym("K64") * Ext::c(KB as i64));
    b.arg_tensor("b", n_cols.clone() * Ext::sym("K64") * Ext::c(KB as i64));
    b.arg_tensor("c", m_rows * n_cols * Ext::c(4));
    let mut lds = LdsPlan::new();
    let [ra, rb] = lds.ring_group(SLOTS as u32, [("A", (BM * KB) as i64, Layout::Linear), ("B", (BN * KB) as i64, Layout::Linear)]);
    lds.commit(&mut b);
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    let pc = b.let_(Op::KernargPtr { offset: 16 });
    let kk = b.let_(Op::KernargI32 { offset: 24 });
    let nn = b.let_(Op::KernargI32 { offset: 28 });
    let tid = b.let_(Op::WorkitemId);
    let wgx = b.let_(Op::WorkgroupIdX);
    let wgy = b.let_(Op::WorkgroupIdY);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let w = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let wm = b.let_(Op::LshrI32(Opnd::Val(w), Opnd::I32(1)));
    let wn = b.let_(Op::AndI32(Opnd::Val(w), Opnd::I32(1)));
    let m = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(15)));
    let g = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::I32(4)));
    let row0 = b.mul_i32(wgy, Opnd::I32(BM));
    let col0 = b.mul_i32(wgx, Opnd::I32(BN));
    let a_org = b.mul_i32(row0, kk);
    let b_org = b.mul_i32(col0, kk);
    let g8 = b.let_(Op::ShlI32(Opnd::Val(g), Opnd::I32(3)));
    let rowa = b.let_(Op::ShlI32(Opnd::Val(wm), Opnd::I32(4)));
    let rowa = b.add_i32(rowa, m);
    let abase = b.let_(Op::ShlI32(Opnd::Val(rowa), Opnd::I32(6)));
    let abase = b.add_i32(abase, g8);
    let rowb = b.let_(Op::ShlI32(Opnd::Val(wn), Opnd::I32(4)));
    let rowb = b.add_i32(rowb, m);
    let bbase = b.let_(Op::ShlI32(Opnd::Val(rowb), Opnd::I32(6)));
    let bbase = b.add_i32(bbase, g8);
    let tile = TensorTile { data_size: 1, tensor_dim: [Opnd::I32(KB), Opnd::I32(BM)], stride0: Opnd::Val(kk), tile: [KB as u16, BM as u16], pad: None, wg_mask: 0 };
    let (_, g1) = {
        let z = b.add_i32(a_org, Opnd::I32(0));
        b.tensor_desc_2d(&tile, Opnd::Val(pa), Opnd::Val(z), Opnd::I32(0))
    };
    let nkv = b.bind_i32(ev(kk).lshr(6));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(24).ceil_div(64), "nkv = K / 64 >= 1 (a K that is a multiple of 64)");
    let pipe = Pipeline::new(Cadence::OneWindow, nkv, kk, KB, Ext::ka(24).ceil_div(64));
    let mut body = TdmGemmBody { a: ra, b: rb, pa, pb, a_org, b_org, g1, abase, bbase, acc: None };
    pipe.run(&mut b, Role::All, &mut body);
    let _ = I32::c(0);

    let accv = body.acc.expect("the pipeline bound the accumulator");
    let crow = b.let_(Op::ShlI32(Opnd::Val(wm), Opnd::I32(4)));
    let crow = b.add_i32(crow, row0);
    let crow = b.add_i32(crow, g8);
    let ccol = b.let_(Op::ShlI32(Opnd::Val(wn), Opnd::I32(4)));
    let ccol = b.add_i32(ccol, col0);
    let ccol = b.add_i32(ccol, m);
    let lin = b.mul_i32(crow, nn);
    let lin = b.add_i32(lin, ccol);
    let off0 = b.let_(Op::ShlI32(Opnd::Val(lin), Opnd::I32(2)));
    let row_bytes = b.let_(Op::ShlI32(Opnd::Val(nn), Opnd::I32(2)));
    for r in 0..8 {
        let off = if r == 0 { off0 } else { let rbv = b.mul_i32(row_bytes, Opnd::I32(r)); b.add_i32(off0, rbv) };
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(pc), byte_off: Opnd::Val(off) });
        let x = b.frag_slice(Opnd::Var(accv), r as u8, 1);
        b.store_global(q, 0, x, MemTy::B32, false);
    }
    let mut f = b.finish();
    f.lower_opts.extend(["dce"].iter().map(|s| s.to_string()).chain(DEFAULT_SCALAR.iter().map(|s| s.to_string())));
    f
}

/// The MXFP4 kernel name.
pub const F4_NAME: &str = "gemm_g1250_tdm_stage_f4";

/// MXFP4 GEMM on the TDM stage template (`amdgpu_stage::gemm::tdm_f4::TdmF4Body`): C[M][N] f32 = A[M][K] x B[N][K]^T, fp4 e2m1 operands (two per byte, low nibble first) with E8M0 scales
/// per 32 K (`sa` [M][K / 32] bytes, `sb` [N][K / 32] bytes), `v_wmma_scale_f32_32x16x128_f4`. A 128 x 64 C tile per workgroup (4 waves 2 x 2, each wave 2 x 2 WMMAs of 32 x 16), K tiles of
/// 128, the 3-slot ring of the OneWindow cadence. Kernargs: a @0, b @8, sa @16, sb @24, c @32 (pointers), k @40 (K elements, a multiple of 128), n @44. Grid (N / 64, M / 128).
pub fn gemm_tdm_stage_f4() -> Function {
    gemm_tdm_stage_f4_with(F4_NAME, F4Shape { wm: 2, wn: 2, tm: 2, tn: 2, ks: 1, relax: false, cx: 1, cy: 1 }, true, false)
}

/// 128 x 128 tile (4 waves, each 2 x 4 WMMAs).
pub const F4_128X128: &str = "gemm_g1250_tdm_stage_f4_128x128";
/// 256 x 64 tile (4 waves, each 4 x 2 WMMAs).
pub const F4_256X64: &str = "gemm_g1250_tdm_stage_f4_256x64";
/// 256 x 128 tile (4 waves, each 4 x 4 WMMAs): the back end currently rejects it (a scaled WMMA's scale operand must live in v0..v255 while the accumulators go above). NOT registered (a failing
/// kernel cannot be in the compiled / reachable sets); `tests/g1250_repros.rs` (`--ignored`) compiles it and shows the error.
pub const F4_256X128: &str = "gemm_g1250_tdm_stage_f4_256x128";

pub fn gemm_tdm_stage_f4_128x128() -> Function {
    gemm_tdm_stage_f4_with(F4_128X128, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 1, relax: false, cx: 1, cy: 1 }, true, false)
}

pub fn gemm_tdm_stage_f4_256x64() -> Function {
    gemm_tdm_stage_f4_with(F4_256X64, F4Shape { wm: 2, wn: 2, tm: 4, tn: 2, ks: 1, relax: false, cx: 1, cy: 1 }, true, false)
}

pub fn gemm_tdm_stage_f4_256x128() -> Function {
    gemm_tdm_stage_f4_with(F4_256X128, F4Shape { wm: 2, wn: 2, tm: 4, tn: 4, ks: 1, relax: false, cx: 1, cy: 1 }, true, false)
}

/// K sub-steps per ring slot (a slot holds `ks x 128` K elements: more WMMAs per barrier / tensor wait; the fragment loads of the next sub-step are issued before the WMMAs of the current one).
pub const F4_128X128_KS2: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2";
pub const F4_128X128_KS4: &str = "gemm_g1250_tdm_stage_f4_128x128_ks4";
pub const F4_256X128_KS2: &str = "gemm_g1250_tdm_stage_f4_256x128_ks2";

pub fn gemm_tdm_stage_f4_128x128_ks2() -> Function {
    gemm_tdm_stage_f4_with(F4_128X128_KS2, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 2, relax: false, cx: 1, cy: 1 }, true, false)
}

/// The vendor-shaped tile: 256 x 256 per workgroup, 4 waves (2 x 2) of 128 x 128 (4 x 8 WMMAs of 32 x 16 each per K step), ks = 2.
pub const F4_256X256_KS2: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2";
pub fn gemm_tdm_stage_f4_256x256_ks2() -> Function {
    gemm_tdm_stage_f4_with(F4_256X256_KS2, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1 }, true, false)
}

pub fn gemm_tdm_stage_f4_128x128_ks4() -> Function {
    gemm_tdm_stage_f4_with(F4_128X128_KS4, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 4, relax: false, cx: 1, cy: 1 }, true, false)
}

pub fn gemm_tdm_stage_f4_256x128_ks2() -> Function {
    gemm_tdm_stage_f4_with(F4_256X128_KS2, F4Shape { wm: 2, wn: 2, tm: 4, tn: 4, ks: 2, relax: false, cx: 1, cy: 1 }, true, false)
}

/// Relaxed tensor waits (`F4Shape::relax`): the 128 x 128 tile with 2 sub-steps per slot.
pub const F4_128X128_KS2_RELAX: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_relax";

pub fn gemm_tdm_stage_f4_128x128_ks2_relax() -> Function {
    gemm_tdm_stage_f4_with(F4_128X128_KS2_RELAX, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 2, relax: true, cx: 1, cy: 1 }, true, false)
}

/// Cluster multicast (docs/mxfp4-gemm-gfx1250-plan.md step 8): the 128 x 128 tile, 2 K sub-steps per slot, in a cluster of 2 x 2 / 2 x 1 / 1 x 2 workgroups (launch with the cluster attribute and
/// a grid divisible by it).
pub const F4_128X128_KS2_C2X2: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_c2x2";
pub const F4_128X128_KS2_C2X1: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_c2x1";
pub const F4_128X128_KS2_C1X2: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_c1x2";

pub fn gemm_tdm_stage_f4_128x128_ks2_cluster(name: &str, cx: i32, cy: i32) -> Function {
    gemm_tdm_stage_f4_with(name, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 2, relax: false, cx, cy }, true, false)
}

/// The 8-wave 256 x 64 tile (4 x 2 waves, each 2 x 2 WMMAs, cooperative loads, the original interleaved ring layout): the configuration that hung a device on 2026-11-17 (declared LDS 86,784 B).
/// Regression kernel of that incident: its cause was NOT established when it was registered; run it on one work-group first, watchdogged (docs/mxfp4-gemm-gfx1250-plan.md).
pub const F4_8W: &str = "gemm_g1250_tdm_stage_f4_8w_256x64";

pub fn gemm_tdm_stage_f4_8w_256x64() -> Function {
    gemm_tdm_stage_f4_with(F4_8W, F4Shape { wm: 4, wn: 2, tm: 2, tn: 2, ks: 1, relax: false, cx: 1, cy: 1 }, true, true)
}

/// The ks = 2 kernels (the K tile of a slot = 2 x 128 elements): `_ks2_grouped` is the layout of the original wrong-result build (the four parts interleaved per slot by `ring_group`, declared 103,936
/// bytes), `_ks2` the same kernel with four separate rings (78,336 bytes). They returned wrong data until the wait pass used the gfx1250 DScnt range; now regression kernels of that bug.
pub const F4_KS2_GROUPED: &str = "gemm_g1250_tdm_stage_f4_ks2_grouped";
pub const F4_KS2: &str = "gemm_g1250_tdm_stage_f4_ks2";

pub fn gemm_tdm_stage_f4_ks2_grouped() -> Function {
    gemm_tdm_stage_f4_with(F4_KS2_GROUPED, F4Shape { wm: 2, wn: 2, tm: 2, tn: 2, ks: 2, relax: false, cx: 1, cy: 1 }, true, true)
}

pub fn gemm_tdm_stage_f4_ks2() -> Function {
    gemm_tdm_stage_f4_with(F4_KS2, F4Shape { wm: 2, wn: 2, tm: 2, tn: 2, ks: 2, relax: false, cx: 1, cy: 1 }, true, false)
}

/// The MXFP4 stage-template GEMM for a workgroup shape (`sh`) and the loading scheme (`coop`: every wave loads a slice of the tile; otherwise every wave loads all of it). The four rings (A, B and the
/// two scale tiles) are separate rings (`grouped`: interleaved per slot by `ring_group`, the original layout, which declares `base of the last part + slots x stride` bytes), so the declared LDS is
/// exactly `slots x (A + B + scales)` bytes.
pub fn gemm_tdm_stage_f4_with(name: &str, sh: F4Shape, coop: bool, grouped: bool) -> Function {
    use amdgpu_dsl::ev;
    use amdgpu_stage::gemm::tdm_f4::{F4Shape, TdmF4Body};
    use amdgpu_stage::{Cadence, Pipeline};
    use amdgpu_tile::lane::Role;
    use amdgpu_tile::shared::{Layout, LdsPlan};
    let (bm, bn) = (sh.bm(), sh.bn());
    let mut b = FnBuilder::new(name, (sh.waves() * 32) as u32);
    for (i, n) in ["a", "b", "sa", "sb", "c"].iter().enumerate() {
        b.arg(n, DType::Ptr, 8 * i as u32, 8);
    }
    b.arg("k", DType::I32, 40, 4);
    b.arg("n", DType::I32, 44, 4);
    // the host contract (docs/gfx1250-hang-rules.md, tensor bounds): the grid covers the output exactly (N = grid x * BN, M = grid y * BM), K a multiple of the slot's K; the buffers hold A / B
    // (fp4: K / 2 bytes per row), their scales (K / 32 bytes per row) and C (f32, M x N). Every tensor tile and every store is proven inside these extents.
    // K = 128 * K128 (K128 a multiple of ks, at least two K slots so the prologue's second stage is a real tile): the extents below are polynomials, no divisions
    b.sym_range("K128", 2 * sh.ks as i64, 1 << 13);
    b.ka_is(40, Ext::sym("K128") * Ext::c(128));
    b.ka_is(44, Ext::grid_x() * Ext::c(bn as i64));
    let (m_rows, n_cols) = (Ext::grid_y() * Ext::c(bm as i64), Ext::grid_x() * Ext::c(bn as i64));
    b.arg_tensor("a", m_rows.clone() * Ext::sym("K128") * Ext::c(64));
    b.arg_tensor("b", n_cols.clone() * Ext::sym("K128") * Ext::c(64));
    b.arg_tensor("sa", m_rows.clone() * Ext::sym("K128") * Ext::c(4));
    b.arg_tensor("sb", n_cols.clone() * Ext::sym("K128") * Ext::c(4));
    b.arg_tensor("c", m_rows * n_cols * Ext::c(4));
    let mut lds = LdsPlan::new();
    let (ra, rb, rsa, rsb) = if grouped {
        let [ra, rb, rsa, rsb] = lds.ring_group(SLOTS as u32, [("A", (bm * 64 * sh.ks) as i64, Layout::Linear), ("B", (bn * 64 * sh.ks) as i64, Layout::Linear), ("SA", (bm * 4 * sh.ks) as i64, Layout::Linear), ("SB", (bn * 4 * sh.ks) as i64, Layout::Linear)]);
        (ra, rb, rsa, rsb)
    } else {
        (
            lds.ring("A", (bm * 64 * sh.ks) as i64, SLOTS as u32, Layout::Linear),
            lds.ring("B", (bn * 64 * sh.ks) as i64, SLOTS as u32, Layout::Linear),
            lds.ring("SA", (bm * 4 * sh.ks) as i64, SLOTS as u32, Layout::Linear),
            lds.ring("SB", (bn * 4 * sh.ks) as i64, SLOTS as u32, Layout::Linear),
        )
    };
    lds.commit(&mut b);
    let ptr = |b: &mut FnBuilder, off: u32| b.let_(Op::KernargPtr { offset: off });
    let (pa, pb, psa, psb, pc) = (ptr(&mut b, 0), ptr(&mut b, 8), ptr(&mut b, 16), ptr(&mut b, 24), ptr(&mut b, 32));
    let kk = b.let_(Op::KernargI32 { offset: 40 });
    let nn = b.let_(Op::KernargI32 { offset: 44 });
    let tid = b.let_(Op::WorkitemId);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let w = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let (cx, cy) = (sh.cx, sh.cy);
    let clustered = cx * cy > 1;
    // the workgroup's tile coordinates and, in a cluster, its position inside the cluster (TTMP6[3:0] / [7:4]); in a cluster launch TTMP9 / TTMP7[15:0] are the CLUSTER ids
    let (wgx, wgy, wg_x, wg_y) = if clustered {
        let bfe = |b: &mut FnBuilder, v: ValueId, off: i32, wd: i32| {
            let sh = if off == 0 { v } else { b.let_(Op::LshrI32(Opnd::Val(v), Opnd::I32(off))) };
            b.let_(Op::AndI32(Opnd::Val(sh), Opnd::I32((1 << wd) - 1)))
        };
        let t6 = b.let_(Op::ReadTtmp { idx: 6 });
        let t7 = b.let_(Op::ReadTtmp { idx: 7 });
        let t9 = b.let_(Op::ReadTtmp { idx: 9 });
        let (wx, wy) = (bfe(&mut b, t6, 0, 4), bfe(&mut b, t6, 4, 4));
        let cyid = b.let_(Op::AndI32(Opnd::Val(t7), Opnd::I32(0xffff)));
        let gx0 = b.mul_i32(t9, Opnd::I32(cx));
        let gx = b.add_i32(gx0, wx);
        let gy0 = b.mul_i32(cyid, Opnd::I32(cy));
        let gy = b.add_i32(gy0, wy);
        // the launch contract, counted as trusted facts: a workgroup's position lies inside the declared cluster, and cluster id x cluster dims + position lies inside the grid (the host
        // launches a grid divisible by the cluster; `amdc` hosts check it)
        b.assume_trusted(wx, Ext::c(0), Ext::c(cx as i64 - 1), "cluster position x < cluster dim x (TTMP6[3:0])");
        b.assume_trusted(wy, Ext::c(0), Ext::c(cy as i64 - 1), "cluster position y < cluster dim y (TTMP6[7:4])");
        b.assume_trusted(gx, Ext::c(0), Ext::grid_x() - Ext::c(1), "tile x = cluster id x * cx + position < grid x (grid divisible by the cluster)");
        b.assume_trusted(gy, Ext::c(0), Ext::grid_y() - Ext::c(1), "tile y = cluster id y * cy + position < grid y (grid divisible by the cluster)");
        (gx, gy, wx, wy)
    } else {
        let gx = b.let_(Op::WorkgroupIdX);
        let gy = b.let_(Op::WorkgroupIdY);
        (gx, gy, gx, gy)
    };
    assert!((sh.wn as u32).is_power_of_two(), "wn a power of two");
    let wi = b.let_(Op::LshrI32(Opnd::Val(w), Opnd::I32(sh.wn.trailing_zeros() as i32)));
    let wj = b.let_(Op::AndI32(Opnd::Val(w), Opnd::I32(sh.wn - 1)));
    let m = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(15)));
    let g = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::I32(4)));
    let row0 = b.mul_i32(wgy, Opnd::I32(bm));
    let col0 = b.mul_i32(wgx, Opnd::I32(bn));
    // bytes per row: fp4 K / 2, scales K / 32
    let kb = b.let_(Op::LshrI32(Opnd::Val(kk), Opnd::I32(1)));
    let ks = b.let_(Op::LshrI32(Opnd::Val(kk), Opnd::I32(5)));
    // cooperative loads: wave w loads rows [w * R / waves, ..) of each tile (R / waves rows)
    let nw = sh.waves();
    assert!(coop || !clustered, "a cluster needs cooperative loads");
    let (diva, divb) = if coop { (nw * cx, nw * cy) } else { (1, 1) };
    assert!(bm % diva == 0 && bn % divb == 0, "the tile splits evenly over the waves of the cluster row / column");
    let (rpa, rpb) = (bm / diva, bn / divb);
    // the wave id is wave-uniform: read it into an SGPR so the slice offsets and descriptors stay scalar
    let wid = b.let_(Op::ReadFirstLane(Opnd::Val(w)));
    let wsl = if coop { wid } else { b.c_i32(0) };
    let leader = b.bind_cond(ev(wid).eq(0));
    // the slice of the A tile (shared by the cluster row) this wave loads: slice index = (workgroup x in the cluster) x waves + wave; of the B tile (shared by the cluster column): (workgroup y) x waves + wave
    let (sa_i, sb_i) = if clustered {
        let xa = b.mul_i32(wg_x, Opnd::I32(nw));
        let ya = b.mul_i32(wg_y, Opnd::I32(nw));
        (b.add_i32(xa, wsl), b.add_i32(ya, wsl))
    } else {
        (wsl, wsl)
    };
    let rowa_w = b.mul_i32(sa_i, Opnd::I32(rpa));
    let rowb_w = b.mul_i32(sb_i, Opnd::I32(rpb));
    let row0w = b.add_i32(row0, rowa_w);
    let col0w = b.add_i32(col0, rowb_w);
    let a_org = b.mul_i32(row0w, kb);
    let b_org = b.mul_i32(col0w, kb);
    let sa_org = b.mul_i32(row0w, ks);
    let sb_org = b.mul_i32(col0w, ks);
    let wl = [
        b.mul_i32(rowa_w, Opnd::I32(64 * sh.ks)),
        b.mul_i32(rowb_w, Opnd::I32(64 * sh.ks)),
        b.mul_i32(rowa_w, Opnd::I32(4 * sh.ks)),
        b.mul_i32(rowb_w, Opnd::I32(4 * sh.ks)),
    ];
    // per-lane LDS offsets inside a slot: the wave's first row of A / B + the lane's row m, 16 h bytes into the 64-byte row; scale rows
    let h16 = b.let_(Op::ShlI32(Opnd::Val(g), Opnd::I32(4)));
    let a_row = b.let_(Op::MulI32(Opnd::Val(wi), Opnd::I32(sh.tm * 32)));
    let a_row_m = b.add_i32(a_row, m);
    let abase = b.mul_i32(a_row_m, Opnd::I32(64 * sh.ks));
    let abase = b.add_i32(abase, h16);
    let b_row = b.let_(Op::MulI32(Opnd::Val(wj), Opnd::I32(sh.tn * 16)));
    let b_row_m = b.add_i32(b_row, m);
    let bbase = b.mul_i32(b_row_m, Opnd::I32(64 * sh.ks));
    let bbase = b.add_i32(bbase, h16);
    let a_row_l = b.add_i32(a_row, lane);
    let sabase = b.mul_i32(a_row_l, Opnd::I32(4 * sh.ks));
    let sbbase = b.mul_i32(b_row_m, Opnd::I32(4 * sh.ks));
    // descriptor group 1 of the four tensors: [rows][K / 2] bytes (tile 64 bytes x rows), scales [rows][K / 128] dwords (tile 1 dword x rows)
    let g1 = |b: &mut FnBuilder, base: ValueId, org: ValueId, esz: u8, dim0: Opnd, rows: i32, stride: Opnd, t0: u16, mask: Option<Opnd>| -> ValueId {
        let tile = TensorTile { data_size: esz, tensor_dim: [dim0, Opnd::I32(rows)], stride0: stride, tile: [t0, rows as u16], pad: None, wg_mask: 0 };
        let z = b.add_i32(org, Opnd::I32(0));
        b.tensor_desc_2d_m(&tile, Opnd::Val(base), Opnd::Val(z), Opnd::I32(0), mask).1
    };
    // multicast masks (bit = the flat cluster-local workgroup index, x fastest): the A tile goes to the workgroups of this cluster row, the B tile to those of this cluster column
    let (mask_row, mask_col) = if clustered {
        let ysh = b.mul_i32(wg_y, Opnd::I32(cx));
        let rowbits = b.let_(Op::ShlI32(Opnd::I32((1 << cx) - 1), Opnd::Val(ysh)));
        let colpat: i32 = (0..cy).map(|j| 1 << (j * cx)).sum();
        let colbits = b.let_(Op::ShlI32(Opnd::I32(colpat), Opnd::Val(wg_x)));
        let ra_ = b.let_(Op::AndI32(Opnd::Val(rowbits), Opnd::I32(0xffff)));
        let rb_ = b.let_(Op::AndI32(Opnd::Val(colbits), Opnd::I32(0xffff)));
        (Some(Opnd::Val(ra_)), Some(Opnd::Val(rb_)))
    } else {
        (None, None)
    };
    let k128 = b.let_(Op::LshrI32(Opnd::Val(kk), Opnd::I32(7)));
    let g1a = g1(&mut b, pa, a_org, 1, Opnd::I32(64 * sh.ks), rpa, Opnd::Val(kb), (64 * sh.ks) as u16, mask_row);
    let g1b = g1(&mut b, pb, b_org, 1, Opnd::I32(64 * sh.ks), rpb, Opnd::Val(kb), (64 * sh.ks) as u16, mask_col);
    let g1sa = g1(&mut b, psa, sa_org, 4, Opnd::I32(sh.ks), rpa, Opnd::Val(k128), sh.ks as u16, mask_row);
    let g1sb = g1(&mut b, psb, sb_org, 4, Opnd::I32(sh.ks), rpb, Opnd::Val(k128), sh.ks as u16, mask_col);
    let lgk = 7 + sh.ks.trailing_zeros() as i32;   // K elements per slot: 128 ks
    assert!(sh.ks.count_ones() == 1, "ks a power of two");
    let nkv = b.bind_i32(ev(kk).lshr(lgk));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(40).ceil_div(128 * sh.ks as i64), "nkv = K / (128 ks) >= 1 (a K that is a multiple of the slot's K)");
    let pipe = Pipeline::new(Cadence::OneWindow, nkv, kk, 128 * sh.ks, Ext::ka(40).ceil_div(128 * sh.ks as i64));
    let mut body = TdmF4Body { sh, a: ra, b: rb, sa: rsa, sb: rsb, pa, pb, psa, psb, a_org, b_org, sa_org, sb_org, g1a, g1b, g1sa, g1sb, abase, bbase, sabase, sbbase, wl, acc: Vec::new(), steady: false, leader };
    if clustered {
        b.cluster_barrier();   // every workgroup of the cluster is resident before anyone multicasts into its LDS
    }
    pipe.run(&mut b, Role::All, &mut body);

    // C: wave (wi, wj), block (i, j): rows R + v % 8 + 8 h + 16 (v / 8), column m
    let cols_l = b.let_(Op::ShlI32(Opnd::Val(nn), Opnd::I32(2)));
    let g8 = b.let_(Op::ShlI32(Opnd::Val(g), Opnd::I32(3)));
    for i in 0..sh.tm {
        for j in 0..sh.tn {
            let acc = body.acc[(i * sh.tn + j) as usize];
            // first row of the block (the lane's row group) and the lane's column
            let r0 = b.let_(Op::MulI32(Opnd::Val(wi), Opnd::I32(sh.tm * 32)));
            let r0 = b.add_i32(r0, row0);
            let r0 = b.add_i32(r0, Opnd::I32(i * 32));
            let r0 = b.add_i32(r0, g8);
            let c0 = b.let_(Op::MulI32(Opnd::Val(wj), Opnd::I32(sh.tn * 16)));
            let c0 = b.add_i32(c0, col0);
            let c0 = b.add_i32(c0, Opnd::I32(j * 16));
            let c0 = b.add_i32(c0, m);
            let lin = b.mul_i32(r0, nn);
            let lin = b.add_i32(lin, c0);
            let off0 = b.let_(Op::ShlI32(Opnd::Val(lin), Opnd::I32(2)));
            for v in 0..16i32 {
                let rowv = (v % 8) + 16 * (v / 8);
                let off = if rowv == 0 { off0 } else { let rb_ = b.mul_i32(cols_l, Opnd::I32(rowv)); b.add_i32(off0, rb_) };
                let q = b.let_(Op::PtrAdd { base: Opnd::Val(pc), byte_off: Opnd::Val(off) });
                let x = b.frag_slice(Opnd::Var(acc), v as u8, 1);
                b.store_global(q, 0, x, MemTy::B32, false);
            }
        }
    }
    let mut f = b.finish();
    f.lower_opts.extend(["dce"].iter().map(|s| s.to_string()).chain(DEFAULT_SCALAR.iter().map(|s| s.to_string())));
    if clustered {
        f.kd.push(("cluster_dims".into(), (cx * cy) as i64));   // the launch must declare the cluster (and divide the grid by it)
    }
    f
}
