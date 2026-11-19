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
    use amdgpu_tile::lane::{Role, Roles};
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
    gemm_tdm_stage_f4_with(F4_NAME, F4Shape { wm: 2, wn: 2, tm: 2, tn: 2, ks: 1, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

/// 128 x 128 tile (4 waves, each 2 x 4 WMMAs).
pub const F4_128X128: &str = "gemm_g1250_tdm_stage_f4_128x128";
/// 256 x 64 tile (4 waves, each 4 x 2 WMMAs).
pub const F4_256X64: &str = "gemm_g1250_tdm_stage_f4_256x64";
/// 256 x 128 tile (4 waves, each 4 x 4 WMMAs): the back end currently rejects it (a scaled WMMA's scale operand must live in v0..v255 while the accumulators go above). NOT registered (a failing
/// kernel cannot be in the compiled / reachable sets); `tests/g1250_repros.rs` (`--ignored`) compiles it and shows the error.
pub const F4_256X128: &str = "gemm_g1250_tdm_stage_f4_256x128";

pub fn gemm_tdm_stage_f4_128x128() -> Function {
    gemm_tdm_stage_f4_with(F4_128X128, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 1, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

pub fn gemm_tdm_stage_f4_256x64() -> Function {
    gemm_tdm_stage_f4_with(F4_256X64, F4Shape { wm: 2, wn: 2, tm: 4, tn: 2, ks: 1, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

pub fn gemm_tdm_stage_f4_256x128() -> Function {
    gemm_tdm_stage_f4_with(F4_256X128, F4Shape { wm: 2, wn: 2, tm: 4, tn: 4, ks: 1, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

/// K sub-steps per ring slot (a slot holds `ks x 128` K elements: more WMMAs per barrier / tensor wait; the fragment loads of the next sub-step are issued before the WMMAs of the current one).
pub const F4_128X128_KS2: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2";
pub const F4_128X128_KS4: &str = "gemm_g1250_tdm_stage_f4_128x128_ks4";
pub const F4_256X128_KS2: &str = "gemm_g1250_tdm_stage_f4_256x128_ks2";

pub fn gemm_tdm_stage_f4_128x128_ks2() -> Function {
    gemm_tdm_stage_f4_with(F4_128X128_KS2, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

/// The vendor-shaped tile: 256 x 256 per workgroup, 4 waves (2 x 2) of 128 x 128 (4 x 8 WMMAs of 32 x 16 each per K step), ks = 2.
pub const F4_256X256_KS2: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2";
pub const F4_256X256_KS2_HB: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_hb";
/// `gemm_tdm_stage_f4_256x256_ks2` with the tile origins hoisted out of the pipeline (`gemm_tdm_stage_f4_opts`).
pub fn gemm_tdm_stage_f4_256x256_ks2_hb() -> Function {
    gemm_tdm_stage_f4_opts(F4_256X256_KS2_HB, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, true, false)
}

pub const F4_256X256_KS2_PAD: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_pad";
/// `gemm_tdm_stage_f4_256x256_ks2` with the planner's TDM row padding (conflict-free A / B fragment reads; `f4_row_pad`).
pub fn gemm_tdm_stage_f4_256x256_ks2_pad() -> Function {
    gemm_tdm_stage_f4_opts(F4_256X256_KS2_PAD, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true)
}

pub const F4_256X256_KS2_PAD_RELAX: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_pad_relax";
/// `_256x256_ks2_pad` with the relaxed tensor waits (each window completes the previous stage's tile; `F4Shape::relax`).
pub fn gemm_tdm_stage_f4_256x256_ks2_pad_relax() -> Function {
    gemm_tdm_stage_f4_opts(F4_256X256_KS2_PAD_RELAX, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: true, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true)
}

/// The padded 256 x 256 kernel with the relaxed tensor wait over a deeper ring: `slots - 2` stages stay in flight at the end of a window (4 slots: two tiles of lookahead past the next one).
pub const F4_256X256_KS2_PAD_RELAX_S4: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_pad_relax_s4";
pub fn gemm_tdm_stage_f4_256x256_ks2_pad_ring(name: &str, ring: RingOpts) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: true, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, ring)
}

/// The padded 256 x 256 kernel with wave roles: waves 0, 1 issue the A and SA loads, waves 2, 3 the B and SB loads (2 tensor loads per wave per stage), each role its own copy of the loop;
/// `relax` = the relaxed tensor wait.
pub const F4_256X256_KS2_PAD_ROLES: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_pad_roles";
pub const F4_256X256_KS2_PAD_ROLES_RELAX: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_pad_roles_relax";
pub fn gemm_tdm_stage_f4_256x256_ks2_pad_roles(name: &str, relax: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { slots: SLOTS as u8, roles: true, ..Default::default() })
}

pub const F4_128X128_KS2_PAD: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_pad";
pub fn gemm_tdm_stage_f4_128x128_ks2_pad() -> Function {
    gemm_tdm_stage_f4_opts(F4_128X128_KS2_PAD, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true)
}

/// The cluster variants with the planner's TDM row padding.
pub fn gemm_tdm_stage_f4_256x256_ks2_cluster_pad(name: &str, cx: i32, cy: i32) -> Function {
    gemm_tdm_stage_f4_opts(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx, cy, bf16: false, lock: false }, true, false, false, true)
}

pub fn gemm_tdm_stage_f4_256x256_ks2() -> Function {
    gemm_tdm_stage_f4_with(F4_256X256_KS2, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

/// The 256 x 256 tile in a cluster of `cx` x `cy` workgroups (the vendor launches 4 x 4): A (and its scales) multicast along the cluster row, B along the column.
pub fn gemm_tdm_stage_f4_256x256_ks2_cluster(name: &str, cx: i32, cy: i32) -> Function {
    gemm_tdm_stage_f4_with(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx, cy, bf16: false, lock: false }, true, false)
}

/// The same with the cluster kept in lockstep (one cluster barrier per K tile).
pub fn gemm_tdm_stage_f4_256x256_ks2_cluster_lock(name: &str, cx: i32, cy: i32) -> Function {
    gemm_tdm_stage_f4_with(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx, cy, bf16: false, lock: true }, true, false)
}

/// The 256 x 256 tile with a bf16 output (like the vendor kernel): half the C bytes.
pub const F4_256X256_KS2_BF16: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_bf16";
pub fn gemm_tdm_stage_f4_256x256_ks2_bf16() -> Function {
    gemm_tdm_stage_f4_with(F4_256X256_KS2_BF16, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1, bf16: true, lock: false }, true, false)
}

pub fn gemm_tdm_stage_f4_128x128_ks4() -> Function {
    gemm_tdm_stage_f4_with(F4_128X128_KS4, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 4, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

pub fn gemm_tdm_stage_f4_256x128_ks2() -> Function {
    gemm_tdm_stage_f4_with(F4_256X128_KS2, F4Shape { wm: 2, wn: 2, tm: 4, tn: 4, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

/// Relaxed tensor waits (`F4Shape::relax`): the 128 x 128 tile with 2 sub-steps per slot.
pub const F4_128X128_KS2_RELAX: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_relax";

pub fn gemm_tdm_stage_f4_128x128_ks2_relax() -> Function {
    gemm_tdm_stage_f4_with(F4_128X128_KS2_RELAX, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 2, relax: true, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

/// Cluster multicast (docs/mxfp4-gemm-gfx1250-plan.md step 8): the 128 x 128 tile, 2 K sub-steps per slot, in a cluster of 2 x 2 / 2 x 1 / 1 x 2 workgroups (launch with the cluster attribute and
/// a grid divisible by it).
pub const F4_128X128_KS2_C2X2: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_c2x2";
pub const F4_128X128_KS2_C2X1: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_c2x1";
pub const F4_128X128_KS2_C1X2: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_c1x2";

pub fn gemm_tdm_stage_f4_128x128_ks2_cluster(name: &str, cx: i32, cy: i32) -> Function {
    gemm_tdm_stage_f4_with(name, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 2, relax: false, cx, cy, bf16: false, lock: false }, true, false)
}

/// The 8-wave 256 x 64 tile (4 x 2 waves, each 2 x 2 WMMAs, cooperative loads, the original interleaved ring layout): the configuration that hung a device on 2026-11-17 (declared LDS 86,784 B).
/// Regression kernel of that incident: its cause was NOT established when it was registered; run it on one work-group first, watchdogged (docs/mxfp4-gemm-gfx1250-plan.md).
pub const F4_8W: &str = "gemm_g1250_tdm_stage_f4_8w_256x64";

pub fn gemm_tdm_stage_f4_8w_256x64() -> Function {
    gemm_tdm_stage_f4_with(F4_8W, F4Shape { wm: 4, wn: 2, tm: 2, tn: 2, ks: 1, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, true)
}

/// The ks = 2 kernels (the K tile of a slot = 2 x 128 elements): `_ks2_grouped` is the layout of the original wrong-result build (the four parts interleaved per slot by `ring_group`, declared 103,936
/// bytes), `_ks2` the same kernel with four separate rings (78,336 bytes). They returned wrong data until the wait pass used the gfx1250 DScnt range; now regression kernels of that bug.
pub const F4_KS2_GROUPED: &str = "gemm_g1250_tdm_stage_f4_ks2_grouped";
pub const F4_KS2: &str = "gemm_g1250_tdm_stage_f4_ks2";

pub fn gemm_tdm_stage_f4_ks2_grouped() -> Function {
    gemm_tdm_stage_f4_with(F4_KS2_GROUPED, F4Shape { wm: 2, wn: 2, tm: 2, tn: 2, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, true)
}

pub fn gemm_tdm_stage_f4_ks2() -> Function {
    gemm_tdm_stage_f4_with(F4_KS2, F4Shape { wm: 2, wn: 2, tm: 2, tn: 2, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false)
}

/// The MXFP4 stage-template GEMM for a workgroup shape (`sh`) and the loading scheme (`coop`: every wave loads a slice of the tile; otherwise every wave loads all of it). The four rings (A, B and the
/// two scale tiles) are separate rings (`grouped`: interleaved per slot by `ring_group`, the original layout, which declares `base of the last part + slots x stride` bytes), so the declared LDS is
/// exactly `slots x (A + B + scales)` bytes.
pub fn gemm_tdm_stage_f4_with(name: &str, sh: F4Shape, coop: bool, grouped: bool) -> Function {
    // the planner's TDM row padding by default for ks >= 2 (conflict-free fragment reads; a08-1 at 4096^3: 256x256 ks2 2.79 -> 5.17 PF/s,
    // 128x128 ks2 1.57 -> 3.75, 128x64 ks2 1.52 -> 2.26; the ks = 1 kernels measured flat (0.97 / 1.62 vs 0.99 / 1.64) and stay unpadded);
    // `gemm_tdm_stage_f4_opts(.., pad = false)` keeps the unpadded layout (`_nopad` kernel) for A/B
    // and the C epilogue through LDS + one tensor store per wave for the f32, non-cluster kernels (256x256 ks2: +22 % at 4096^3, +27 % at
    // 8192^3, a08-1); `_256x256_ks2_epi(.., 0)` / RingOpts::epilogue_lds = 0 keep the direct stores
    let epi = if !sh.bf16 && sh.cx * sh.cy == 1 { 2 } else { 0 };
    gemm_tdm_stage_f4_ring(name, sh, coop, grouped, false, sh.ks >= 2, RingOpts { epilogue_lds: epi, ..RingOpts::default() })
}

pub const F4_256X256_KS2_PERSIST: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_persist";
/// The padded 256 x 256 ks2 kernel on a persistent grid (`RingOpts::persistent`; host PERSIST=G).
pub fn gemm_tdm_stage_f4_256x256_ks2_persist() -> Function {
    gemm_tdm_stage_f4_ring(F4_256X256_KS2_PERSIST, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { persistent: true, ..RingOpts::default() })
}

pub const F4_128X128_KS2_PERSIST: &str = "gemm_g1250_tdm_stage_f4_128x128_ks2_persist";
pub fn gemm_tdm_stage_f4_128x128_ks2_persist() -> Function {
    gemm_tdm_stage_f4_ring(F4_128X128_KS2_PERSIST, F4Shape { wm: 2, wn: 2, tm: 2, tn: 4, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { persistent: true, ..RingOpts::default() })
}

pub const F4_256X128_KS2_PERSIST: &str = "gemm_g1250_tdm_stage_f4_256x128_ks2_persist";
pub fn gemm_tdm_stage_f4_256x128_ks2_persist() -> Function {
    gemm_tdm_stage_f4_ring(F4_256X128_KS2_PERSIST, F4Shape { wm: 2, wn: 2, tm: 4, tn: 4, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { persistent: true, ..RingOpts::default() })
}

pub const F4_256X256_KS2_SS: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_ss";
/// The padded 256 x 256 ks2 kernel with the preshuffled scale layout (host SCALE_SHUF=1).
pub fn gemm_tdm_stage_f4_256x256_ks2_ss() -> Function {
    gemm_tdm_stage_f4_ring(F4_256X256_KS2_SS, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { scale_shuf: true, ..RingOpts::default() })
}

/// The padded 256 x 256 kernel with the preshuffled scales composed with the relaxed tensor wait and / or the wave roles (host SCALE_SHUF=1).
pub const F4_256X256_KS2_PAD_SS_RELAX: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_pad_ss_relax";
pub const F4_256X256_KS2_PAD_SS_ROLES: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_pad_ss_roles";
pub const F4_256X256_KS2_PAD_SS_ROLES_RELAX: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_pad_ss_roles_relax";
pub fn gemm_tdm_stage_f4_256x256_ks2_pad_ss_opts(name: &str, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { scale_shuf: true, roles, ..RingOpts::default() })
}

pub const F4_256X256_KS2_EPIB: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_epib";
pub const F4_256X256_KS2_EPIT: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_epit";
/// The padded 256 x 256 ks2 kernel with the C epilogue staged through LDS: b128 global stores (`_epib`) or one tensor store per wave (`_epit`).
pub fn gemm_tdm_stage_f4_256x256_ks2_epi(name: &str, epi: u8) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: epi, ..RingOpts::default() })
}

/// The padded 256 x 256 kernel with the LDS-staged epilogue (`epi`: 1 = b128 global stores, 2 = a tensor store per wave) composed with the relaxed tensor wait and / or the wave roles.
pub const F4_256X256_KS2_EPIT_RELAX: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_epit_relax";
pub const F4_256X256_KS2_EPIT_ROLES: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles";
pub const F4_256X256_KS2_EPIT_ROLES_RELAX: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax";
pub fn gemm_tdm_stage_f4_256x256_ks2_epi_opts(name: &str, epi: u8, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: epi, roles, ..RingOpts::default() })
}

/// The 256 x 128 tile (4 waves of 128 x 64) with the TDM-store epilogue: twice the workgroups of the 256 x 256 tile (512 at 4096^3), for the shapes that fill the machine only once.
pub fn gemm_tdm_stage_f4_256x128_ks2_epit_opts(name: &str, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 4, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: 2, roles, ..RingOpts::default() })
}

/// The best 256 x 256 kernel with the tile-order swizzle (host SWIZZLE=G, G = `swizzle` tile rows per group).
pub fn gemm_tdm_stage_f4_256x256_ks2_epit_swz(name: &str, relax: bool, roles: bool, swizzle: u8) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: 2, roles, swizzle, ..RingOpts::default() })
}

/// The best 256 x 256 kernel with the lazy prologue: the loop starts after tile 0 only (the next tile is still in flight).
pub fn gemm_tdm_stage_f4_256x256_ks2_epit_lazy(name: &str, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: 2, roles, lazy: true, ..RingOpts::default() })
}

/// The best 256 x 256 kernel with an instruction prefetch of `n` x 4 KiB at the start.
pub fn gemm_tdm_stage_f4_256x256_ks2_epit_ipf(name: &str, relax: bool, roles: bool, n: u8) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: 2, roles, iprefetch: n, ..RingOpts::default() })
}

/// The 256 x 256 tile with K slots of 512 (ks 4) in a 2-slot ring: one barrier and one stage (2 tensor loads per wave) per 512 K instead of 256, the next slot loading during the whole window.
pub fn gemm_tdm_stage_f4_256x256_ks4_s2(name: &str, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 4, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { slots: 2, epilogue_lds: 2, roles, stamps: name.ends_with("_stamps"), stagger: if name.contains("_stag") { name.rsplit("_stag").next().and_then(|s| s.parse().ok()).unwrap_or(0) } else { 0 }, wave_order: if name.contains("_gray") { 1 } else if name.contains("_wt") { 2 } else { 0 }, ..RingOpts::default() })
}

/// The ks 4 two-slot tile in a lockstep multicast cluster of `cx` x `cy` workgroups (one cluster barrier per 512 K).
pub fn gemm_tdm_stage_f4_256x256_ks4_s2_cluster(name: &str, cx: i32, cy: i32) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 4, relax: false, cx, cy, bf16: false, lock: true }, true, false, false, true, RingOpts { slots: 2, epilogue_lds: 2, roles: name.contains("_roles_"), ..RingOpts::default() })
}

/// The 256 x 256 kernel with the operand swap and the bf16 direct epilogue (host OUT_BF16=1): no LDS staging, no tensor store.
pub fn gemm_tdm_stage_f4_256x256_ks2_swapbf16(name: &str, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: true, lock: false }, true, false, false, true, RingOpts { roles, swap: true, stamps: name.ends_with("_stamps"), persistent: name.ends_with("_persist"), ..RingOpts::default() })
}

/// The 2-slot ks 4 tile with the swap + bf16 LDS-staged epilogue (host OUT_BF16=1): 16-byte conflict-free LDS stores of packed bf16 and four tensor stores of 64-byte-wide chunks per wave.
pub fn gemm_tdm_stage_f4_256x256_ks4_s2_bf16(name: &str, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 4, relax: false, cx: 1, cy: 1, bf16: true, lock: false }, true, false, false, true, RingOpts { slots: 2, epilogue_lds: 3, swap: true, roles, stamps: name.ends_with("_stamps"), ..RingOpts::default() })
}

/// The best 256 x 256 kernel with the K-loop phase stamps (host STAMPS=1).
pub fn gemm_tdm_stage_f4_256x256_ks2_epit_stamps(name: &str, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: 2, roles, stamps: true, ..RingOpts::default() })
}

/// The best 256 x 256 kernel on a persistent grid (host PERSIST=G): no cross-tile overlap (the LDS-staged epilogue uses the whole LDS), only the tile loop.
pub fn gemm_tdm_stage_f4_256x256_ks2_epit_persist(name: &str, relax: bool, roles: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, true, RingOpts { epilogue_lds: 2, roles, persistent: true, ..RingOpts::default() })
}

/// The padded 256 x 256 kernel with the TDM-store epilogue in a cluster of `cx` x `cy` workgroups, lockstep (one cluster barrier per K tile).
pub fn gemm_tdm_stage_f4_256x256_ks2_epit_cluster(name: &str, cx: i32, cy: i32) -> Function {
    gemm_tdm_stage_f4_ring(name, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx, cy, bf16: false, lock: true }, true, false, false, true, RingOpts { epilogue_lds: 2, ..RingOpts::default() })
}

pub const F4_256X256_KS2_NOPAD: &str = "gemm_g1250_tdm_stage_f4_256x256_ks2_nopad";
/// The unpadded LDS layout (8-way bank conflicts on the fragment reads), for A/B against the default.
pub fn gemm_tdm_stage_f4_256x256_ks2_nopad() -> Function {
    gemm_tdm_stage_f4_opts(F4_256X256_KS2_NOPAD, F4Shape { wm: 2, wn: 2, tm: 4, tn: 8, ks: 2, relax: false, cx: 1, cy: 1, bf16: false, lock: false }, true, false, false, false)
}

/// The A / B row padding (bytes) that makes the fragment reads conflict-free: the TDM pads every `64 ks`-byte row (interval = the row's
/// dwords) by the fewest dwords whose bank-conflict cost (`amdgpu_dsl::tdm_pad`) is the least, for the window's read pattern (16 lanes of a
/// b128 phase on rows m = lane & 15, 16 bytes at h = lane >> 4 of the 32-byte piece).
/// The same for the swap mode's permuted A operand: it reads rows (m & 7) + 16 (m >> 3) (the other operand keeps rows m): the pad minimising the cost of both patterns together.
pub fn f4_row_pad_perm(ks: i32) -> i32 {
    use amdgpu_dsl::tdm_pad::{cost, LaneRead};
    let row = (64 * ks) as u32;
    let mk = |rowmap: &dyn Fn(u32) -> u32| -> Vec<LaneRead> {
        (0..2u32).map(|piece| LaneRead { bytes: 16, lanes: (0..32u32).map(|l| Some(rowmap(l & 15) * row + 16 * (l >> 4) + 32 * piece)).collect() }).collect()
    };
    let pa = mk(&|m| (m & 7) + 16 * (m >> 3));
    let pb = mk(&|m| m);
    let iv = (row / 4) as u16;
    let total = |pad: Option<(u16, u16)>| cost(&pa, pad) + cost(&pb, pad);
    let mut best = (total(None), 0u16);
    for am in 1..=16u16 {
        let c = total(Some((iv, am)));
        if c < best.0 {
            best = (c, am);
        }
    }
    4 * best.1 as i32
}

pub fn f4_row_pad(ks: i32) -> i32 {
    use amdgpu_dsl::tdm_pad::{cost, LaneRead};
    let row = (64 * ks) as u32;
    let reads: Vec<LaneRead> = (0..2u32)
        .map(|piece| LaneRead { bytes: 16, lanes: (0..32u32).map(|l| Some((l & 15) * row + 16 * (l >> 4) + 32 * piece)).collect() })
        .collect();
    let iv = (row / 4) as u16;
    let mut best = (cost(&reads, None), 0u16);
    for am in 1..=16u16 {
        let c = cost(&reads, Some((iv, am)));
        if c < best.0 {
            best = (c, am);
        }
    }
    4 * best.1 as i32
}

/// `gemm_tdm_stage_f4_with`; `hoist_base`: the per-wave global tile origins (`pointer + row origin`, 64-bit) are formed once before the
/// pipeline, so a stage adds only its K byte offset (the descriptor address advances by the K offset alone; no per-stage origin add).
/// The ring options of the MXFP4 stage kernels (`Pipeline::slots`).
#[derive(Clone, Copy, Debug)]
pub struct RingOpts {
    /// ring slots (>= 3): the producer runs `slots - 1` tiles ahead
    pub slots: u8,
    /// wave roles: waves 0, 1 issue the A and SA loads (and run their own copy of the loop), waves 2, 3 the B and SB loads (`TdmF4Body::roles`)
    pub roles: bool,
    /// persistent grid (no cluster): a tile loop inside the kernel; workgroup w takes the tile column w % TX and the rows w / TX, + ystep,
    /// .. (TX = N / BN a power of two; kernargs: tile rows @48, log2 TX @52, the row step @56); each tile is one complete pipeline run
    pub persistent: bool,
    /// the scales in the vendor's preshuffled layout (host SCALE_SHUF=1; `TdmF4Body::scale_shuf`)
    pub scale_shuf: bool,
    /// the C epilogue through LDS (f32, non-cluster): 0 = the direct per-element stores; 1 = each wave writes its 128-column block row-major
    /// into the dead ring (`ds_store_b32`, rows padded by 16 bytes), then reads rows back as b128 and stores them with `global_store_b128`
    /// (512 coalesced bytes per instruction); 2 = the same staging, then one `tensor_store_from_lds` per wave
    pub epilogue_lds: u8,
    /// tile order: the workgroup's linear tile id (row-major, the hardware dispatch order) is remapped so that the workgroups running together cover a block of `swizzle` tile rows x all tile
    /// columns in column-major order inside the block (0 = off): the A rows and B columns of concurrently running workgroups overlap, so the L2 serves them instead of DRAM. Host contract: tile
    /// columns N / BN a power of two (kernarg `lgx` @52 = its log2), tile rows a multiple of `swizzle` (a power of two)
    pub swizzle: u8,
    /// `Pipeline::lazy_prologue`: wait only for tile 0 before the loop
    pub lazy: bool,
    /// instruction prefetch at the kernel start: this many `s_prefetch_inst` of 32 lines each from the kernel's first instruction (0 = none): the first iteration of the loop then finds its
    /// code in the instruction cache instead of fetching it line by line from L2
    pub iprefetch: u8,
    /// the K-loop phase stamps (`TdmF4Body::stamps`): a debug buffer argument `dbg` @64, per wave 8 dwords: the summed cycles of the stage issue, the window, the wait, the barrier,
    /// the steady iteration count
    pub stamps: bool,
    /// operand swap: the WMMA's A operand is the B matrix (tile rows = N) and its B operand the A matrix, so a lane's accumulator holds 8 CONSECUTIVE columns of C for a fixed row (instead of 8
    /// consecutive rows of one column): with `bf16` output the epilogue packs them and stores 16 bytes per lane straight from the registers (no LDS staging, no tensor store). Needs bm == bn.
    pub swap: bool,
    /// stagger: the workgroups of odd tile parity (tile column + tile row) sleep `stagger` x 64 clocks at the start, so the epilogue write bursts of the tiles in flight do not all align (0 = off)
    pub stagger: u16,
    /// wave grid order: 0 = (wi, wj) = (w >> 1, w & 1); 1 = Gray order (w0 (0,0), w1 (0,1), w2 (1,1), w3 (1,0)); 2 = transposed (wi, wj) = (w & 1, w >> 1)
    pub wave_order: u8,
}

impl RingOpts {
    /// the swap mode's row permutation and LDS-staged bf16 epilogue (`epilogue_lds == 3`)
    fn perm(&self) -> bool {
        self.swap && self.epilogue_lds == 3
    }
}

impl Default for RingOpts {
    fn default() -> Self {
        RingOpts { slots: SLOTS as u8, roles: false, persistent: false, scale_shuf: false, epilogue_lds: 0, swizzle: 0, lazy: false, iprefetch: 0, stamps: false, swap: false, stagger: 0, wave_order: 0 }
    }
}

pub fn gemm_tdm_stage_f4_opts(name: &str, sh: F4Shape, coop: bool, grouped: bool, hoist_base: bool, pad: bool) -> Function {
    gemm_tdm_stage_f4_ring(name, sh, coop, grouped, hoist_base, pad, RingOpts::default())
}

/// The MXFP4 stage kernel with an explicit ring depth (see `RingOpts`).
pub fn gemm_tdm_stage_f4_ring(name: &str, sh: F4Shape, coop: bool, grouped: bool, hoist_base: bool, pad: bool, ring: RingOpts) -> Function {
    use amdgpu_dsl::ev;
    use amdgpu_stage::gemm::tdm_f4::{F4Shape, TdmF4Body};
    use amdgpu_stage::{Cadence, Pipeline};
    use amdgpu_tile::lane::{Role, Roles};
    use amdgpu_tile::shared::{Layout, LdsPlan};
    let (bm, bn) = (sh.bm(), sh.bn());
    // A / B rows in LDS: 64 ks bytes + the TDM padding (`pad`: planner-chosen, conflict-free fragment reads)
    let row_pad = if pad { if ring.perm() { f4_row_pad_perm(sh.ks) } else { f4_row_pad(sh.ks) } } else { 0 };
    let row_l = 64 * sh.ks + row_pad;
    let mut b = FnBuilder::new(name, (sh.waves() * 32) as u32);
    for (i, n) in ["a", "b", "sa", "sb", "c"].iter().enumerate() {
        b.arg(n, DType::Ptr, 8 * i as u32, 8);
    }
    b.arg("k", DType::I32, 40, 4);
    b.arg("n", DType::I32, 44, 4);
    if ring.stamps {
        b.arg("dbg", DType::Ptr, 64, 8);
        b.arg_tensor("dbg", Ext::grid_y() * Ext::c(64 * 4 * 64));
    }
    let persistent = ring.persistent;
    if ring.roles && sh.cx * sh.cy > 1 {
        // the two role copies of the loop signal / wait the cluster barrier in different loops; both run nkv - 1 iterations by construction (a trusted, listed fact)
        b.barrier_wait_trips_match(-3);
    }
    assert!(!(persistent && sh.cx * sh.cy > 1), "persistent: no cluster yet");
    if persistent {
        // the outermost loop is the persistent tile loop (G5 verifies one tile body ending quiescent)
        b.persistent_tile_loop();
        b.arg("tiles_y", DType::I32, 48, 4);
        b.arg("lg_tiles_x", DType::I32, 52, 4);
        b.arg("ystep", DType::I32, 56, 4);
    }
    // the host contract (docs/gfx1250-hang-rules.md, tensor bounds): the grid covers the output exactly (N = grid x * BN, M = grid y * BM), K a multiple of the slot's K; the buffers hold A / B
    // (fp4: K / 2 bytes per row), their scales (K / 32 bytes per row) and C (f32, M x N). Every tensor tile and every store is proven inside these extents.
    // K = 128 * K128 (K128 a multiple of ks, at least two K slots so the prologue's second stage is a real tile): the extents below are polynomials, no divisions
    b.sym_range("K128", (ring.slots as i64 - 1) * sh.ks as i64, 1 << 13);   // at least `slots - 1` K slots: the prologue stages tiles 0 .. slots - 2 as real tiles
    b.ka_is(40, Ext::sym("K128") * Ext::c(128));
    // (persistent: the tile counts are symbols TX / TY, N = TX * BN, kernarg tiles_y = TY; the grid is any number of workgroups)
    let (m_rows, n_cols) = if persistent {
        b.sym_range("TX", 1, 1 << 12);
        b.sym_range("TY", 1, 1 << 12);
        b.ka_is(44, Ext::sym("TX") * Ext::c(bn as i64));
        b.ka_is(48, Ext::sym("TY"));
        (Ext::sym("TY") * Ext::c(bm as i64), Ext::sym("TX") * Ext::c(bn as i64))
    } else {
        b.ka_is(44, Ext::grid_x() * Ext::c(bn as i64));
        (Ext::grid_y() * Ext::c(bm as i64), Ext::grid_x() * Ext::c(bn as i64))
    };
    b.arg_tensor("a", m_rows.clone() * Ext::sym("K128") * Ext::c(64));
    b.arg_tensor("b", n_cols.clone() * Ext::sym("K128") * Ext::c(64));
    b.arg_tensor("sa", m_rows.clone() * Ext::sym("K128") * Ext::c(4));
    b.arg_tensor("sb", n_cols.clone() * Ext::sym("K128") * Ext::c(4));
    b.arg_tensor("c", m_rows * n_cols * Ext::c(if sh.bf16 { 2 } else { 4 }));
    let mut lds = LdsPlan::new();
    let (ra, rb, rsa, rsb) = if grouped {
        let [ra, rb, rsa, rsb] = lds.ring_group(ring.slots as u32, [("A", (bm * row_l) as i64, Layout::Linear), ("B", (bn * row_l) as i64, Layout::Linear), ("SA", (bm * 4 * sh.ks) as i64, Layout::Linear), ("SB", (bn * 4 * sh.ks) as i64, Layout::Linear)]);
        (ra, rb, rsa, rsb)
    } else {
        (
            lds.ring("A", (bm * row_l) as i64, ring.slots as u32, Layout::Linear),
            lds.ring("B", (bn * row_l) as i64, ring.slots as u32, Layout::Linear),
            lds.ring("SA", (bm * 4 * sh.ks) as i64, ring.slots as u32, Layout::Linear),
            lds.ring("SB", (bn * 4 * sh.ks) as i64, ring.slots as u32, Layout::Linear),
        )
    };
    // the LDS-staged epilogue aliases the (dead) ring from offset 0: the plan must cover waves x (wave rows) x (wave columns x 4 + 16) bytes
    let epi = ring.epilogue_lds;
    assert!(epi == 0 || epi == 3 || !sh.bf16, "the LDS epilogue: f32 output (bf16: epilogue_lds 3 with the swap)");
    assert!(epi != 3 || (ring.swap && sh.bf16), "epilogue_lds 3: bf16 output with the operand swap");
    let (wrows, wcols) = (sh.tm * 32, sh.tn * 16);
    // (the tensor store reads its LDS rows contiguously - the TDM's padding did not apply to the store on MI450: rows 1.. came out shifted -
    // so its staging rows are unpadded)
    let epi_rs = if epi == 3 { wcols * 2 } else if epi == 2 { wcols * 4 } else { wcols * 4 + 16 };
    if epi > 0 {
        let ring_bytes = ring.slots as i64 * ((bm * row_l + bn * row_l + bm * 4 * sh.ks + bn * 4 * sh.ks) as i64);
        let need = (sh.waves() * wrows * epi_rs) as i64;
        if need > ring_bytes {
            let _ = lds.ring("EPI", need - ring_bytes, 1, Layout::Linear);
        }
    }
    lds.commit(&mut b);
    let ptr = |b: &mut FnBuilder, off: u32| b.let_(Op::KernargPtr { offset: off });
    let (pa, pb, psa, psb, pc) = (ptr(&mut b, 0), ptr(&mut b, 8), ptr(&mut b, 16), ptr(&mut b, 24), ptr(&mut b, 32));
    // swap: the A operand reads the B matrix and its scales (and vice versa)
    assert!(!ring.swap || (bm == bn && sh.cx * sh.cy == 1), "operand swap: a square tile, no cluster");
    let (pa, pb, psa, psb) = if ring.swap { (pb, pa, psb, psa) } else { (pa, pb, psa, psb) };
    let pdbg = if ring.stamps { ptr(&mut b, 64) } else { pc };
    let t_start = if ring.stamps {
        let c = b.shader_cycles();
        b.let_(Op::PtrLo(Opnd::Val(c)))
    } else {
        pc
    };
    let kk = b.let_(Op::KernargI32 { offset: 40 });
    let nn = b.let_(Op::KernargI32 { offset: 44 });
    let tid = b.let_(Op::WorkitemId);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let w = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let (cx, cy) = (sh.cx, sh.cy);
    let clustered = cx * cy > 1;
    let mut persist_var: Option<amdgpu_dsl::VarId> = None;
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
    } else if persistent {
        // tile column w & (TX - 1), first row w >> log2 TX; the row loop opens here and closes after the epilogue
        let wid = b.let_(Op::WorkgroupIdX);
        let lgx = b.let_(Op::KernargI32 { offset: 52 });
        let one = b.c_i32(1);
        let txn = b.let_(Op::ShlI32(Opnd::Val(one), Opnd::Val(lgx)));
        let txm = b.add_i32(txn, Opnd::I32(-1));
        let gx = b.let_(Op::AndI32(Opnd::Val(wid), Opnd::Val(txm)));
        let gy0 = b.let_(Op::LshrI32(Opnd::Val(wid), Opnd::Val(lgx)));
        b.assume_trusted(gx, Ext::c(0), Ext::sym("TX") - Ext::c(1), "persistent launch: the tile column w mod 2^lg_tiles_x < TX (N / BN = 2^lg_tiles_x)");
        let vy = b.begin_loop(&[(DType::I32, Opnd::Val(gy0))])[0];
        persist_var = Some(vy);
        let gy = b.let_(Op::AddI32(Opnd::Var(vy), Opnd::I32(0)));
        b.assume_trusted(gy, Ext::c(0), Ext::sym("TY") - Ext::c(1), "persistent launch: the loop runs while the tile row < TY (the first row w >> lg_tiles_x < TY: at most TX * TY workgroups)");
        (gx, gy, gx, gy)
    } else if ring.swizzle > 0 {
        assert!(ring.swizzle.is_power_of_two(), "the swizzle group (tile rows) a power of two");
        let lgg = ring.swizzle.trailing_zeros() as i32;
        b.arg("lgx", DType::I32, 52, 4);
        let gx0 = b.let_(Op::WorkgroupIdX);
        let gy0 = b.let_(Op::WorkgroupIdY);
        let lgx = b.let_(Op::KernargI32 { offset: 52 });
        let one = b.c_i32(1);
        let tx = b.let_(Op::ShlI32(Opnd::Val(one), Opnd::Val(lgx)));
        let lin0 = b.mul_i32(gy0, tx);
        let lin = b.add_i32(lin0, gx0);
        let span_lg = b.add_i32(lgx, Opnd::I32(lgg));
        let span = b.let_(Op::ShlI32(Opnd::Val(one), Opnd::Val(span_lg)));
        let spm = b.add_i32(span, Opnd::I32(-1));
        let grp = b.let_(Op::LshrI32(Opnd::Val(lin), Opnd::Val(span_lg)));
        let r = b.let_(Op::AndI32(Opnd::Val(lin), Opnd::Val(spm)));
        let rg = b.let_(Op::AndI32(Opnd::Val(r), Opnd::I32(ring.swizzle as i32 - 1)));
        let rowbase = b.let_(Op::ShlI32(Opnd::Val(grp), Opnd::I32(lgg)));
        let gy = b.add_i32(rowbase, rg);
        let gx = b.let_(Op::LshrI32(Opnd::Val(r), Opnd::I32(lgg)));
        // a permutation of the grid's tiles (host contract: the grid is N / BN = 2^lgx tile columns by a multiple of `swizzle` tile rows): the remapped tile lies inside the grid
        b.assume_trusted(gx, Ext::c(0), Ext::grid_x() - Ext::c(1), "tile swizzle: a permutation of the grid's tiles (tile columns = 2^lgx, tile rows a multiple of the group)");
        b.assume_trusted(gy, Ext::c(0), Ext::grid_y() - Ext::c(1), "tile swizzle: a permutation of the grid's tiles (tile columns = 2^lgx, tile rows a multiple of the group)");
        (gx, gy, gx, gy)
    } else {
        let gx = b.let_(Op::WorkgroupIdX);
        let gy = b.let_(Op::WorkgroupIdY);
        (gx, gy, gx, gy)
    };
    assert!((sh.wn as u32).is_power_of_two(), "wn a power of two");
    let (wi, wj) = match ring.wave_order {
        1 => {
            assert!(sh.wm == 2 && sh.wn == 2, "wave order: a 2 x 2 wave grid");
            let hi = b.let_(Op::LshrI32(Opnd::Val(w), Opnd::I32(1)));
            let x = b.let_(Op::XorI32(Opnd::Val(w), Opnd::Val(hi)));
            (hi, b.let_(Op::AndI32(Opnd::Val(x), Opnd::I32(1))))
        }
        2 => {
            assert!(sh.wm == 2 && sh.wn == 2, "wave order: a 2 x 2 wave grid");
            (b.let_(Op::AndI32(Opnd::Val(w), Opnd::I32(1))), b.let_(Op::LshrI32(Opnd::Val(w), Opnd::I32(1))))
        }
        _ => (b.let_(Op::LshrI32(Opnd::Val(w), Opnd::I32(sh.wn.trailing_zeros() as i32))), b.let_(Op::AndI32(Opnd::Val(w), Opnd::I32(sh.wn - 1)))),
    };
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
    // a cluster multicast is a rendezvous of IDENTICAL requests (the ISA: one wave per workgroup issues the same request with the same mask; the data goes to every requesting workgroup once all have
    // asked): so the workgroups of a cluster row issue the very same slices of the A tile, those of a cluster column the same slices of the B tile, and the slices are per wave only, as without a cluster
    assert!(!ring.roles || (coop && nw == 4), "wave roles: 4 cooperative waves");
    let div = if ring.roles { 2 } else if coop { nw } else { 1 };   // roles: the two waves of a role split each of its tiles
    assert!(bm % div == 0 && bn % div == 0, "the tile splits evenly over the waves");
    let (rpa, rpb) = (bm / div, bn / div);
    // the wave id is wave-uniform: read it into an SGPR so the slice offsets and descriptors stay scalar
    let wid = b.let_(Op::ReadFirstLane(Opnd::Val(w)));
    let wsl = if ring.roles { b.let_(Op::AndI32(Opnd::Val(wid), Opnd::I32(1))) } else if coop { wid } else { b.c_i32(0) };   // the wave's index inside its role
    let leader = b.bind_cond(ev(wid).eq(0));
    let rowa_w = b.mul_i32(wsl, Opnd::I32(rpa));
    let rowb_w = b.mul_i32(wsl, Opnd::I32(rpb));
    let (org_a, org_b) = if ring.swap { (col0, row0) } else { (row0, col0) };   // the global row origin of the A / B operand tile
    let row0w = b.add_i32(org_a, rowa_w);
    let col0w = b.add_i32(org_b, rowb_w);
    let a_org = b.mul_i32(row0w, kb);
    let b_org = b.mul_i32(col0w, kb);
    let sa_org = b.mul_i32(row0w, ks);
    let sb_org = b.mul_i32(col0w, ks);
    let wl = [
        b.mul_i32(rowa_w, Opnd::I32(row_l)),
        b.mul_i32(rowb_w, Opnd::I32(row_l)),
        b.mul_i32(rowa_w, Opnd::I32(4 * sh.ks)),
        b.mul_i32(rowb_w, Opnd::I32(4 * sh.ks)),
    ];
    // per-lane LDS offsets inside a slot: the wave's first row of A / B + the lane's row m, 16 h bytes into the 64-byte row; scale rows
    let h16 = b.let_(Op::ShlI32(Opnd::Val(g), Opnd::I32(4)));
    let a_row = b.let_(Op::MulI32(Opnd::Val(wi), Opnd::I32(sh.tm * 32)));
    let a_row_m = b.add_i32(a_row, m);
    let a_row_m = if ring.perm() {
        // the permuted operand row of (lane m, half 0): m & 7 + 16 (m >> 3)
        let lo = b.let_(Op::AndI32(Opnd::Val(m), Opnd::I32(7)));
        let hi = b.let_(Op::LshrI32(Opnd::Val(m), Opnd::I32(3)));
        let hi16 = b.let_(Op::ShlI32(Opnd::Val(hi), Opnd::I32(4)));
        let pm = b.add_i32(lo, hi16);
        b.add_i32(a_row, pm)
    } else {
        a_row_m
    };
    let abase = b.mul_i32(a_row_m, Opnd::I32(row_l));
    let abase = b.add_i32(abase, h16);
    let b_row = b.let_(Op::MulI32(Opnd::Val(wj), Opnd::I32(sh.tn * 16)));
    let b_row_m = b.add_i32(b_row, m);
    let bbase = b.mul_i32(b_row_m, Opnd::I32(row_l));
    let bbase = b.add_i32(bbase, h16);
    let a_row_l = if ring.perm() {
        // the scale row of operand row r = lane: the B-matrix row with bits 3 and 4 of r exchanged
        let lo = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(7)));
        let b3 = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(8)));
        let b4 = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(16)));
        let b3s = b.let_(Op::ShlI32(Opnd::Val(b3), Opnd::I32(1)));
        let b4s = b.let_(Op::LshrI32(Opnd::Val(b4), Opnd::I32(1)));
        let p1 = b.add_i32(lo, b3s);
        let p2 = b.add_i32(p1, b4s);
        b.add_i32(a_row, p2)
    } else {
        b.add_i32(a_row, lane)
    };
    // scale row offsets: row * 4 ks bytes, or (preshuffled; the wave's first row a multiple of 32) the row block's 128 ks bytes + 4 per row
    let (sabase, sbbase) = if ring.scale_shuf {
        let ab = b.mul_i32(a_row, Opnd::I32(4 * sh.ks));
        let l4 = b.let_(Op::ShlI32(Opnd::Val(lane), Opnd::I32(2)));
        let sa_ = b.add_i32(ab, l4);
        let bb = b.mul_i32(b_row, Opnd::I32(4 * sh.ks));
        let m4 = b.let_(Op::ShlI32(Opnd::Val(m), Opnd::I32(2)));
        let sb_ = b.add_i32(bb, m4);
        (sa_, sb_)
    } else {
        (b.mul_i32(a_row_l, Opnd::I32(4 * sh.ks)), b.mul_i32(b_row_m, Opnd::I32(4 * sh.ks)))
    };
    // descriptor group 1 of the four tensors: [rows][K / 2] bytes (tile 64 bytes x rows), scales [rows][K / 128] dwords (tile 1 dword x rows)
    let g1 = |b: &mut FnBuilder, base: ValueId, org: ValueId, esz: u8, dim0: Opnd, rows: i32, stride: Opnd, t0: u16, mask: Option<Opnd>, pad: Option<(u16, u16)>| -> ValueId {
        let tile = TensorTile { data_size: esz, tensor_dim: [dim0, Opnd::I32(rows)], stride0: stride, tile: [t0, rows as u16], pad, wg_mask: 0 };
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
    let rpad = (row_pad > 0).then(|| ((16 * sh.ks) as u16, (row_pad / 4) as u16));
    let g1a = g1(&mut b, pa, a_org, 1, Opnd::I32(64 * sh.ks), rpa, Opnd::Val(kb), (64 * sh.ks) as u16, mask_row, rpad);
    let g1b = g1(&mut b, pb, b_org, 1, Opnd::I32(64 * sh.ks), rpb, Opnd::Val(kb), (64 * sh.ks) as u16, mask_col, rpad);
    let (g1sa, g1sb) = if ring.scale_shuf {
        // preshuffled scales: rows of a 32-row block are K / 4 dwords (K / 128 tiles of 32 dwords); the slice is rows / 32 of them, 32 ks
        // dwords each
        assert!(rpa % 32 == 0 && rpb % 32 == 0, "the scale slices are whole 32-row blocks");
        let k4 = b.let_(Op::LshrI32(Opnd::Val(kk), Opnd::I32(2)));
        (
            g1(&mut b, psa, sa_org, 4, Opnd::I32(32 * sh.ks), rpa / 32, Opnd::Val(k4), (32 * sh.ks) as u16, mask_row, None),
            g1(&mut b, psb, sb_org, 4, Opnd::I32(32 * sh.ks), rpb / 32, Opnd::Val(k4), (32 * sh.ks) as u16, mask_col, None),
        )
    } else {
        (
            g1(&mut b, psa, sa_org, 4, Opnd::I32(sh.ks), rpa, Opnd::Val(k128), sh.ks as u16, mask_row, None),
            g1(&mut b, psb, sb_org, 4, Opnd::I32(sh.ks), rpb, Opnd::Val(k128), sh.ks as u16, mask_col, None),
        )
    };
    let lgk = 7 + sh.ks.trailing_zeros() as i32;   // K elements per slot: 128 ks
    assert!(sh.ks.count_ones() == 1, "ks a power of two");
    let nkv = b.bind_i32(ev(kk).lshr(lgk));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(40).ceil_div(128 * sh.ks as i64), "nkv = K / (128 ks) >= 1 (a K that is a multiple of the slot's K)");
    let mut pipe = Pipeline::new(Cadence::OneWindow, nkv, kk, 128 * sh.ks, Ext::ka(40).ceil_div(128 * sh.ks as i64));
    pipe.slots = ring.slots;
    pipe.lazy_prologue = ring.lazy;
    let (pa, pb, psa, psb, a_org, b_org, sa_org, sb_org) = if hoist_base {
        let base = |b: &mut FnBuilder, p: ValueId, o: ValueId| b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(o) });
        let z = b.let_(Op::ConstI32(0));
        (base(&mut b, pa, a_org), base(&mut b, pb, b_org), base(&mut b, psa, sa_org), base(&mut b, psb, sb_org), z, z, z, z)
    } else {
        (pa, pb, psa, psb, a_org, b_org, sa_org, sb_org)
    };
    if ring.stagger > 0 {
        let par = b.let_(Op::XorI32(Opnd::Val(wgx), Opnd::Val(wgy)));
        let odd = b.let_(Op::AndI32(Opnd::Val(par), Opnd::I32(1)));
        let c = b.bind_cond(ev(odd).eq(1));
        b.begin_if(c);
        b.sleep(ring.stagger as u32);
        b.end_if();
    }
    if ring.iprefetch > 0 {
        let pc = b.get_pc();
        let zero = b.c_i32(0);
        for k in 0..ring.iprefetch as u32 {
            b.prefetch_inst_len(pc, k * 4096, zero, 31);
        }
    }
    if clustered {
        b.cluster_barrier();   // every workgroup of the cluster is resident before anyone multicasts into its LDS
    }

    let mut emit = |b: &mut FnBuilder, role: Role| {
        let mut body = TdmF4Body { sh, a: ra.clone(), b: rb.clone(), sa: rsa.clone(), sb: rsb.clone(), pa, pb, psa, psb, a_org, b_org, sa_org, sb_org, g1a, g1b, g1sa, g1sb, abase, bbase, sabase, sbbase, wl, acc: Vec::new(), steady: false, leader, depth: 3, roles: ring.roles, stamps: ring.stamps, perm: ring.perm(), stamp_vars: Vec::new(), st_t1: None, st_t2: None, row_pad, scale_shuf: ring.scale_shuf };
        pipe.run(b, role, &mut body);
        let t_post = if ring.stamps {
            let c = b.shader_cycles();
            Some(b.let_(Op::PtrLo(Opnd::Val(c))))
        } else {
            None
        };
        if ring.stamps {
            // the per-wave record: [tile row][tile column & 63][wave] x 32 bytes of the debug buffer (the host keeps N / BN <= 64)
            let gxm = b.let_(Op::AndI32(Opnd::Val(wgx), Opnd::I32(63)));
            let wgl = b.bind_i32(ev(wgy) * 64 + ev(gxm));
            let rec = b.bind_i32(((ev(wgl) << 2) + ev(w)) * 64);
            for (k, var) in body.stamp_vars.clone().iter().take(4).enumerate() {
                let v = b.let_(Op::AddI32(Opnd::Var(*var), Opnd::I32(0)));
                b.stamp_store(Opnd::Val(pdbg), Opnd::Val(rec), k as u32, v);
            }
            let it = b.let_(Op::AddI32(Opnd::Val(nkv), Opnd::I32(-1)));
            b.stamp_store(Opnd::Val(pdbg), Opnd::Val(rec), 4, it);
        }
        if persistent {
            // the next tile's first stage overwrites slot 0: every wave has finished reading the last window first
            b.wait_lds(0);
            b.barrier();
        }
        if epi == 3 {
            // the staged bf16 epilogue: the lane's 16 contiguous columns of a row go to LDS as two 16-byte stores into 64-byte-wide column chunks (one per 32 columns), which the tensor store
            // then writes as 128-row tiles of 64 bytes; every wave owns 32 KiB of the (dead) ring
            if !persistent {
                b.wait_lds(0);
                b.barrier();
            }
            let wbase = b.mul_i32(wid, Opnd::I32(32768));
            let g32 = b.let_(Op::ShlI32(Opnd::Val(g), Opnd::I32(5)));
            let m64 = b.let_(Op::ShlI32(Opnd::Val(m), Opnd::I32(6)));
            let l0 = b.add_i32(wbase, m64);
            let l0 = b.add_i32(l0, g32);
            for i in 0..sh.tm {
                for j in 0..sh.tn {
                    let acc = body.acc[(i * sh.tn + j) as usize];
                    for h in 0..2i32 {
                        let mut parts: Vec<Opnd> = Vec::new();
                        for q in 0..4i32 {
                            let x0 = b.frag_slice(Opnd::Var(acc), (h * 8 + 2 * q) as u8, 1);
                            let x1 = b.frag_slice(Opnd::Var(acc), (h * 8 + 2 * q + 1) as u8, 1);
                            let f0 = b.bitcast_f32(x0);
                            let f1 = b.bitcast_f32(x1);
                            parts.push(Opnd::Val(b.let_(Op::CvtPkBf16F32(Opnd::Val(f0), Opnd::Val(f1)))));
                        }
                        let v4 = b.frag_cat(&parts);
                        let addr = b.bind_i32(ev(l0) + (i * 8192 + j * 1024 + h * 16));
                        b.store_lds_b128(addr, v4);
                    }
                }
            }
            b.wait_lds(0);
            let r0 = b.mul_i32(wj, Opnd::I32(sh.tn * 16));
            let r0 = b.add_i32(r0, row0);
            let tile = TensorTile { data_size: 2, tensor_dim: [Opnd::I32(32), Opnd::I32(sh.tn * 16)], stride0: Opnd::Val(nn), tile: [32, (sh.tn * 16) as u16], pad: None, wg_mask: 0 };
            for i in 0..sh.tm {
                let c0 = b.mul_i32(wi, Opnd::I32(sh.tm * 32));
                let c0 = b.add_i32(c0, col0);
                let c0 = b.add_i32(c0, Opnd::I32(i * 32));
                let lin = b.mul_i32(r0, nn);
                let lin = b.add_i32(lin, c0);
                let org = b.let_(Op::ShlI32(Opnd::Val(lin), Opnd::I32(1)));
                let org_s = b.let_(Op::ReadFirstLane(Opnd::Val(org)));
                let lw = b.bind_i32(ev(wbase) + i * 8192);
                let lw_s = b.let_(Op::ReadFirstLane(Opnd::Val(lw)));
                let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(pc), Opnd::Val(org_s), Opnd::Val(lw_s));
                b.tensor_store_from_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default());
            }
            b.wait_tensor(0);
            if persistent {
                b.wait_lds(0);
                b.barrier();
            }
        } else if ring.swap && sh.bf16 {
            // D[n][m]: the lane holds, for its column m (a row of C), 8 consecutive n at v = 0..7 (+ 16 more at v = 8..15): bf16 pairs packed into 4 dwords, one 16-byte store each
            let g8s = b.let_(Op::ShlI32(Opnd::Val(g), Opnd::I32(3)));
            let row_bytes = b.let_(Op::ShlI32(Opnd::Val(nn), Opnd::I32(1)));
            for i in 0..sh.tm {
                for j in 0..sh.tn {
                    let acc = body.acc[(i * sh.tn + j) as usize];
                    let mr = b.mul_i32(wj, Opnd::I32(sh.tn * 16));
                    let mr = b.add_i32(mr, row0);
                    let mr = b.add_i32(mr, Opnd::I32(j * 16));
                    let mr = b.add_i32(mr, m);
                    let n0 = b.mul_i32(wi, Opnd::I32(sh.tm * 32));
                    let n0 = b.add_i32(n0, col0);
                    let n0 = b.add_i32(n0, Opnd::I32(i * 32));
                    let n0 = b.add_i32(n0, g8s);
                    let rb = b.mul_i32(mr, row_bytes);
                    let nb = b.let_(Op::ShlI32(Opnd::Val(n0), Opnd::I32(1)));
                    let off = b.add_i32(rb, nb);
                    for h in 0..2i32 {
                        let mut parts: Vec<Opnd> = Vec::new();
                        for q in 0..4i32 {
                            let x0 = b.frag_slice(Opnd::Var(acc), (h * 8 + 2 * q) as u8, 1);
                            let x1 = b.frag_slice(Opnd::Var(acc), (h * 8 + 2 * q + 1) as u8, 1);
                            let f0 = b.bitcast_f32(x0);
                            let f1 = b.bitcast_f32(x1);
                            parts.push(Opnd::Val(b.let_(Op::CvtPkBf16F32(Opnd::Val(f0), Opnd::Val(f1)))));
                        }
                        let v4 = b.frag_cat(&parts);
                        let p = b.let_(Op::PtrAdd { base: Opnd::Val(pc), byte_off: Opnd::Val(off) });
                        b.store_f32x4(p, (h * 32) as u32, v4);
                    }
                }
            }
        } else if epi > 0 {
            // every wave has finished reading the ring (its last window) before any wave writes C into it
            if !persistent {
                b.wait_lds(0);
                b.barrier();
            }
            // wave block (wrows x wcols f32) row-major at wave * wrows * epi_rs, row stride epi_rs (+16 bytes: the two lane groups' rows 8
            // apart fall in different banks)
            let wl0 = b.mul_i32(wid, Opnd::I32(wrows * epi_rs));
            let g8 = b.let_(Op::ShlI32(Opnd::Val(g), Opnd::I32(3)));
            let lrow = b.mul_i32(g8, Opnd::I32(epi_rs));
            let m4 = b.let_(Op::ShlI32(Opnd::Val(m), Opnd::I32(2)));
            let lb = b.add_i32(wl0, lrow);
            let lb = b.add_i32(lb, m4);
            for i in 0..sh.tm {
                let bi = b.add_i32(lb, Opnd::I32(i * 32 * epi_rs));
                for j in 0..sh.tn {
                    let acc = body.acc[(i * sh.tn + j) as usize];
                    for v in 0..16i32 {
                        let rowv = (v % 8) + 16 * (v / 8);
                        let addr = b.bind_i32(ev(bi) + (rowv * epi_rs + j * 64));
                        let x = b.frag_slice(Opnd::Var(acc), v as u8, 1);
                        b.store_lds_b32(addr, x);
                    }
                }
            }
            b.wait_lds(0);
            let t_e1 = if ring.stamps { let c = b.shader_cycles(); Some(b.let_(Op::PtrLo(Opnd::Val(c)))) } else { None };
            // the wave's C block origin (f32 elements): rows row0 + wi * wrows, columns col0 + wj * wcols
            let r0 = b.mul_i32(wi, Opnd::I32(wrows));
            let r0 = b.add_i32(r0, row0);
            let c0 = b.mul_i32(wj, Opnd::I32(wcols));
            let c0 = b.add_i32(c0, col0);
            let lin = b.mul_i32(r0, nn);
            let lin = b.add_i32(lin, c0);
            let org = b.let_(Op::ShlI32(Opnd::Val(lin), Opnd::I32(2)));
            if epi == 1 {
                // rows back as b128: lane l holds columns 4 l .. 4 l + 3 of a row (wcols / 4 lanes per row)
                assert!(wcols % 4 == 0 && 32 % (wcols / 4) == 0 || (wcols / 4) % 32 == 0, "rows split over the lanes");
                let lanes_per_row = wcols / 4;
                assert!(lanes_per_row % 32 == 0 || 32 % lanes_per_row == 0);
                let (rows_per_inst, chunks) = if lanes_per_row >= 32 { (1, lanes_per_row / 32) } else { (32 / lanes_per_row, 1) };
                assert!(rows_per_inst == 1, "a wave block of at least 128 columns");
                let l16 = b.let_(Op::ShlI32(Opnd::Val(lane), Opnd::I32(4)));
                let lr = b.add_i32(wl0, l16);
                let cols_b = b.let_(Op::ShlI32(Opnd::Val(nn), Opnd::I32(2)));
                let gofs = b.add_i32(org, l16);
                for q in 0..wrows {
                    for ch in 0..chunks {
                        let la = b.bind_i32(ev(lr) + (q * epi_rs + ch * 512));
                        let v4 = b.load_lds_b128(la);
                        let rq = b.mul_i32(cols_b, Opnd::I32(q));
                        let off = b.add_i32(gofs, rq);
                        let off = if ch == 0 { off } else { b.add_i32(off, Opnd::I32(ch * 512)) };
                        let p = b.let_(Op::PtrAdd { base: Opnd::Val(pc), byte_off: Opnd::Val(off) });
                        b.store_f32x4(p, 0, v4);
                    }
                }
            } else {
                // one tensor store per wave: the block (wcols x wrows f32) from LDS (rows padded by 4 dwords) to C (row stride N)
                let tile = TensorTile { data_size: 4, tensor_dim: [Opnd::I32(wcols), Opnd::I32(wrows)], stride0: Opnd::Val(nn), tile: [wcols as u16, wrows as u16], pad: None, wg_mask: 0 };
                // (the origin and the LDS base are wave-uniform: scalar descriptor words)
                let org_s = b.let_(Op::ReadFirstLane(Opnd::Val(org)));
                let wl0_s = b.let_(Op::ReadFirstLane(Opnd::Val(wl0)));
                let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(pc), Opnd::Val(org_s), Opnd::Val(wl0_s));
                b.tensor_store_from_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default());
                let t_e2 = if ring.stamps { let c = b.shader_cycles(); Some(b.let_(Op::PtrLo(Opnd::Val(c)))) } else { None };
                if let (Some(e1), Some(e2)) = (t_e1, t_e2) {
                    let gxm = b.let_(Op::AndI32(Opnd::Val(wgx), Opnd::I32(63)));
                    let wgl = b.bind_i32(ev(wgy) * 64 + ev(gxm));
                    let rec = b.bind_i32(((ev(wgl) << 2) + ev(w)) * 64);
                    b.stamp_store(Opnd::Val(pdbg), Opnd::Val(rec), 9, e1);
                    b.stamp_store(Opnd::Val(pdbg), Opnd::Val(rec), 10, e2);
                }
                b.wait_tensor(0);
            }
            if persistent {
                // the next tile's first stage writes the ring: every wave's staging reads are done
                b.wait_lds(0);
                b.barrier();
            }
        } else {
        // C: wave (wi, wj), block (i, j): rows R + v % 8 + 8 h + 16 (v / 8), column m
        let esh = if sh.bf16 { 1 } else { 2 };   // log2 of the output element size
        let cols_l = b.let_(Op::ShlI32(Opnd::Val(nn), Opnd::I32(esh)));
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
                let off0 = b.let_(Op::ShlI32(Opnd::Val(lin), Opnd::I32(esh)));
                for v in 0..16i32 {
                    let rowv = (v % 8) + 16 * (v / 8);
                    let off = if rowv == 0 { off0 } else { let rb_ = b.mul_i32(cols_l, Opnd::I32(rowv)); b.add_i32(off0, rb_) };
                    let q = b.let_(Op::PtrAdd { base: Opnd::Val(pc), byte_off: Opnd::Val(off) });
                    let x = b.frag_slice(Opnd::Var(acc), v as u8, 1);
                    if sh.bf16 {
                        // the accumulator rounded to bf16 (RNE) in the low half of a dword, stored as 16 bits
                        let xf = b.bitcast_f32(x);
                        let p = b.let_(Op::CvtPkBf16F32(Opnd::Val(xf), Opnd::Val(xf)));
                        b.store_u16(q, 0, p);
                    } else {
                        b.store_global(q, 0, x, MemTy::B32, false);
                    }
                }
            }
        }
        }
        if let Some(tp) = t_post {
            let c = b.shader_cycles();
            let t_end = b.let_(Op::PtrLo(Opnd::Val(c)));
            let gxm = b.let_(Op::AndI32(Opnd::Val(wgx), Opnd::I32(63)));
            let wgl = b.bind_i32(ev(wgy) * 64 + ev(gxm));
            let rec = b.bind_i32(((ev(wgl) << 2) + ev(w)) * 64);
            let l0 = b.let_(Op::AddI32(Opnd::Var(body.stamp_vars[5]), Opnd::I32(0)));
            b.stamp_store(Opnd::Val(pdbg), Opnd::Val(rec), 5, l0);
            b.stamp_store(Opnd::Val(pdbg), Opnd::Val(rec), 6, tp);
            b.stamp_store(Opnd::Val(pdbg), Opnd::Val(rec), 7, t_end);
            b.stamp_store(Opnd::Val(pdbg), Opnd::Val(rec), 8, t_start);
        }
    };
    if ring.roles {
        // two code copies by wave: waves 0, 1 (role A) and waves 2, 3 (role B), the same barrier structure in both
        let is_a = b.bind_cond(ev(wid).lt(2));
        let roles = Roles { is_a, wave_in_group: wsl, group: 2 };
        roles.each(&mut b, |b, role| emit(b, role));
    } else {
        emit(&mut b, Role::All);
    }
    if let Some(vy) = persist_var {
        let ystep = b.let_(Op::KernargI32 { offset: 56 });
        let tyk = b.let_(Op::KernargI32 { offset: 48 });
        let ny = b.bind_i32(amdgpu_dsl::varx(vy) + ev(ystep));
        b.set_var(vy, Opnd::Val(ny));
        let done = b.bind_cond(ev(ny).ge(ev(tyk)));
        b.break_if(done);
        b.end_loop();
    }
    let mut f = b.finish();
    f.lower_opts.extend(["dce"].iter().map(|s| s.to_string()).chain(DEFAULT_SCALAR.iter().map(|s| s.to_string())));
    if persistent {
        f.lower_opts.push("zeroinit".to_string());
    }
    if clustered {
        f.kd.push(("cluster_dims".into(), (cx * cy) as i64));   // the launch must declare the cluster (and divide the grid by it)
    }
    f
}
