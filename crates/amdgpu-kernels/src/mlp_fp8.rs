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

use amdgpu_dsl::tile::{ev, Lane};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};
use amdgpu_isa::MmaAtom;
use amdgpu_tile::contract_stage::{check_boundary, LiveFrag, StageBoundary, Storage};
use amdgpu_tile::orient::{consumer_pairs, POperand};
use amdgpu_layout::Movement;

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
}

pub fn by_name(name: &str) -> Option<Function> {
    let mode = match name {
        "mlp_fp8_fused" => Mode::Fused,
        "mlp_fp8_g1" => Mode::G1,
        "mlp_fp8_g2" => Mode::G2,
        _ => return None,
    };
    Some(build(name, mode))
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

fn build(name: &str, mode: Mode) -> Function {
    // the boundary contract: zero-cost hand-off for every k block of GEMM 2
    if mode == Mode::Fused {
        for jb in 0..N1 / 64 {
            let (p, c) = p_boundary(jb);
            let rep = check_boundary(&p, &c).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(rep.frags.iter().all(|f| f.1 <= Movement::Register && f.2 == 0), "{name}: the P hand-off is not free: {:?}", rep.frags);
        }
    }
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
    let b = &mut fb;
    // ABI: in (X or H) @0, W1 @8, W2 @16, out (Y or H) @24, M @32
    for (n, o) in [("ptr_in", 0u32), ("ptr_w1", 8), ("ptr_w2", 16), ("ptr_out", 24)] {
        b.arg(n, DType::Ptr, o, 8);
    }
    b.arg("m", DType::I32, 32, 4);
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 1024);
    let m = mt.clone() * Ext::c(128);
    b.ka_is(32, m.clone());
    b.grid_is(0, mt);
    b.arg_tensor("ptr_in", m.clone() * Ext::c(K1));
    b.arg_tensor("ptr_w1", Ext::c(N1 * K1));
    b.arg_tensor("ptr_w2", Ext::c(N2 * N1));
    b.arg_tensor("ptr_out", m.clone() * Ext::c(if mode == Mode::G1 { N1 } else { 2 * N2 }));
    let p_in = b.let_(Op::KernargPtr { offset: 0 });
    let p_w1 = b.let_(Op::KernargPtr { offset: 8 });
    let p_w2 = b.let_(Op::KernargPtr { offset: 16 });
    let p_out = b.let_(Op::KernargPtr { offset: 24 });
    let m_v = b.let_(Op::KernargI32 { offset: 32 });
    let tid = b.let_(Op::WorkitemId);
    let lane = Lane::from_tid(b, tid);
    let bx = b.let_(Op::WorkgroupIdX);
    let r_in = { let n = b.bind_i32(ev(m_v) * (K1 as i32)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_in, n) };
    let r_w1 = b.buffer_desc(p_w1, (N1 * K1) as u32);
    let r_w2 = b.buffer_desc(p_w2, (N2 * N1) as u32);
    let r_out = { let n = b.bind_i32(ev(m_v) * (if mode == Mode::G1 { N1 } else { 2 * N2 }) as i32); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_out, n) };
    // this lane's row m (B operand row / accumulator column) and lane half
    let row = b.bind_i32((ev(bx) << 7) + (lane.wave() << 5) + lane.j());
    let h16 = b.bind_i32(lane.half() << 4);
    let c127 = b.c_i32(127);
    // operand fragment: 16 bytes at `base + imm` and `+ 32` (the atom's k halves 16 h + 0..15 and 32 + 16 h + 0..15)
    let frag = |b: &mut FnBuilder, r: ValueId, voff: ValueId, soff: i32, imm: u32| -> ValueId {
        let lo = b.buffer_load_b128(r, voff, soff, imm);
        let hi = b.buffer_load_b128(r, voff, soff, imm + 32);
        b.cat_frag8(lo, hi)
    };
    // weight rows: lane j -> row j of a 32-row tile, 256 bytes per row
    let wrow = b.bind_i32((lane.j() << 8) + ev(h16));
    let p_words: Vec<Vec<ValueId>> = if mode == Mode::G2 {
        // GEMM 2 operand from H (natural n1 order): 4 blocks of 64
        let ho = b.bind_i32((ev(row) << 8) + ev(h16));
        (0..N1 / 64).map(|jb| vec![frag(b, r_in, ho, 0, (64 * jb) as u32)]).collect()
    } else {
        // ---- GEMM 1: H^T tiles t (32 n1 x 32 m), A = W1 rows, B = X rows ----
        let xo = b.bind_i32((ev(row) << 8) + ev(h16));
        let zero = b.frag_zero(DType::Frag16);
        let mut acc: Vec<ValueId> = vec![zero; (N1 / 32) as usize];
        for kb in 0..K1 / 64 {
            let xf = frag(b, r_in, xo, 0, (64 * kb) as u32);
            for (t, a) in acc.iter_mut().enumerate() {
                let wf = frag(b, r_w1, wrow, (t as i32) * 32 * 256, (64 * kb) as u32);
                *a = b.mfma_scale_fp8(wf, xf, *a, c127, c127);
            }
        }
        // ---- elementwise: relu(alpha x) -> fp8, 4 values per word in accumulator order ----
        let alpha = b.c_f32(ALPHA);
        let zf = b.c_f32(0.0);
        let words: Vec<Vec<ValueId>> = acc.iter().map(|a| {
            (0..4u8).map(|q| {
                let xs: Vec<ValueId> = (0..4u8).map(|i| { let u = b.frag16_lane(*a, 4 * q + i); let f = b.bitcast_f32(u); let s = b.mul_f32(f, alpha); b.max_f32(s, zf) }).collect();
                b.cvt_pk_fp8x4_f32(xs[0], xs[1], xs[2], xs[3])
            }).collect()
        }).collect();
        if mode == Mode::G1 {
            // H[m][n1]: word (t, q) = n1 32 t + 8 q + 4 h .. + 3 of row m
            let h4 = b.bind_i32(lane.half() << 2);
            let vo = b.bind_i32((ev(row) << 8) + ev(h4));
            for (t, ws) in words.iter().enumerate() {
                for (q, w) in ws.iter().enumerate() {
                    let f = b.bitcast_f32(*w);
                    b.buffer_store_f32(r_out, vo, 0i32, (32 * t + 8 * q) as u32, f);
                }
            }
            return fb.finish();
        }
        // the hand-off: block jb = words of tiles 2 jb, 2 jb + 1 (the StageBoundary checked above: a register renaming)
        (0..N1 / 64).map(|jb| { let w: Vec<ValueId> = words[2 * jb as usize].iter().chain(words[2 * jb as usize + 1].iter()).copied().collect(); vec![b.frag8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])] }).collect()
    };
    // ---- GEMM 2: Y^T tiles u (32 n2 x 32 m), A = W2 rows (n1 contiguous; key-permuted copy in the fused kernel), B = P ----
    let zero = b.frag_zero(DType::Frag16);
    let mut acc2: Vec<ValueId> = vec![zero; (N2 / 32) as usize];
    for (jb, pf) in p_words.iter().enumerate() {
        for (u, a) in acc2.iter_mut().enumerate() {
            let wf = frag(b, r_w2, wrow, (u as i32) * 32 * 256, (64 * jb) as u32);
            *a = b.mfma_scale_fp8(wf, pf[0], *a, c127, c127);
        }
    }
    // ---- epilogue: Y[m][n2] bf16, 4 consecutive n2 per (u, q) ----
    let h8 = b.bind_i32(lane.half() << 3);
    let vo = b.bind_i32((ev(row) << 9) + ev(h8));
    for (u, a) in acc2.iter().enumerate() {
        for q in 0..4u8 {
            let x: Vec<ValueId> = (0..4u8).map(|i| { let v = b.frag16_lane(*a, 4 * q + i); b.bitcast_f32(v) }).collect();
            let w0 = b.cvt_pk_bf16_f32(x[0], x[1]);
            let w1 = b.cvt_pk_bf16_f32(x[2], x[3]);
            let f2 = b.frag2(w0, w1);
            b.buffer_store_b64(r_out, vo, 0i32, (64 * u + 16 * q as usize) as u32, f2);
        }
    }
    let _ = Opnd::I32(0);
    fb.finish()
}
