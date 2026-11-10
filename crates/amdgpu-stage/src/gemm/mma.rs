//! GEMM MMA helpers: the scaled / unscaled MFMA blocks on loop-carried operands and the deferred block-scale updates.

use amdgpu_dsl::{FnBuilder, Op, Opnd, ValueId, VarId};
use amdgpu_tile::ev;
use amdgpu_tile::mma::TiledMma;
use super::formats::Elem;

/// `gemm` on loop variables (the operands of the register-staged tile are loop-carried): same operand / scale exchange as `amdgpu_tile::gemm`.
#[allow(clippy::too_many_arguments)]
pub fn mma_var(b: &mut FnBuilder, mma: &TiledMma, elem: Elem, unit: Option<ValueId>, acc: VarId, a: VarId, bf: VarId, sa: VarId, sb: VarId, byte_a: u8) {
    let p = mma_blocks(b, mma, elem, unit, &[(acc, a, bf, sa, sb, byte_a)], 0, None);
    assert!(p.is_empty(), "mma_var: block-scaled MMAs go through mma_blocks");
}

/// The MMAs of several 16 x 16 blocks of the K tile (acc, A frag, B frag, A scale, B scale, A scale byte). fp8: one scaled MFMA per block.
/// bf16: the 32-byte fragments are two 16-byte bf16 operands (8 elements each, the same K permutation for A and B); the first halves of every
/// block, then the second halves, so the two MFMAs into one accumulator are not back to back.
/// `a_hi_off` (A8W4): the hi A fragment of the A variable v is VarId(v + a_hi_off) (the loop state lays the hi fragments right after the lo ones).
pub fn mma_blocks(b: &mut FnBuilder, mma: &TiledMma, elem: Elem, unit: Option<ValueId>, blocks: &[(VarId, VarId, VarId, VarId, VarId, u8)], a_hi_off: u32, w4: Option<ValueId>) -> Vec<(VarId, ValueId, VarId, VarId)> {
    assert_eq!(mma.atom().name, "MfmaScale16x16x128F8");
    if elem == Elem::Bf16W4 {
        // A16W4: per (A image k, half h) step: every B fragment of the unit dequantized once (dword 2 k + h = 8 fp4 -> 8 bf16 times the
        // group scale 2^(e - 127), e = byte 2 (k % 2) of B scale word k / 2), then the bf16 MFMAs of all blocks (distance = the unit's blocks)
        let _ = w4;
        let kind = amdgpu_dsl::MmaKind::new(amdgpu_dsl::MmaShape::M16N16K32, amdgpu_dsl::MmaTy::Bf16);
        let mut bsets: Vec<(VarId, VarId)> = Vec::new();
        for &(_, _, bf, _, sb, _) in blocks {
            if !bsets.iter().any(|x| x.0 == bf) {
                bsets.push((bf, sb));
            }
        }
        // every (k, h) operand of the unit's B fragments first, then the MFMAs (the dequant results are well ahead of their MFMA reads)
        let mut all: Vec<Vec<(VarId, ValueId)>> = Vec::new();
        for k in 0..4u32 {
            for h in 0..2u8 {
                let ops: Vec<(VarId, ValueId)> = bsets
                    .iter()
                    .map(|&(bf, sb)| {
                        // dword d of the fragment holds K group d of the tile (measured with the host's W4DBG=6 probe): scale byte d % 4 of
                        // word d / 4 (the block's two scale words)
                        let d0 = 2 * k + h as u32;
                        let sw = b.bitcast_i32(Opnd::Var(VarId(sb.0 + d0 / 4)));
                        let e = b.bind_i32((ev(sw).lshr(8 * (d0 % 4) as i32)) & 0xff);
                        let sbits = b.bind_i32(ev(e) << 23);
                        let sf = b.bitcast_f32(sbits);
                        let d = 2 * k as u8 + h;
                        // dword d of the B fragment (a one-dword view of the variable)
                        let dw = b.frag_slice(Opnd::Var(bf), d, 1);
                        let dwi = b.bitcast_i32(dw);
                        let w: Vec<ValueId> = (0..4i32)
                            .map(|byte| {
                                let x = if byte == 0 { dwi } else { b.bind_i32(ev(dwi).lshr(8 * byte)) };
                                // (fp4 pair -> two f32 -> bf16 pair: exact for these values)
                                let f2 = b.cvt(amdgpu_dsl::CvtOp::ScalePkF32Fp4, &[Opnd::Val(x), Opnd::Val(sf)], 0);
                                let (lo, hi) = (b.frag2_lane(f2, 0), b.frag2_lane(f2, 1));
                                let (lo, hi) = (b.bitcast_f32(lo), b.bitcast_f32(hi));
                                b.cvt_pk_bf16_f32(lo, hi)
                            })
                            .collect();
                        (bf, b.pack4(w[0], w[1], w[2], w[3]))
                    })
                    .collect();
                all.push(ops);
            }
        }
        for k in 0..4u32 {
            for h in 0..2u8 {
                let ops = &all[(2 * k + h as u32) as usize];
                for &(acc, a, bf, _, _, _) in blocks {
                    let bop = ops.iter().find(|x| x.0 == bf).unwrap().1;
                    let ak = b.var_slice4(VarId(a.0 + k * a_hi_off), h);
                    if mma.swap_ab {
                        b.var_mfma(acc, kind, bop, ak, false);
                    } else {
                        b.var_mfma(acc, kind, ak, bop, false);
                    }
                }
            }
        }
        return Vec::new();
    }
    if elem == Elem::Fp8W4 {
        // per block, half h: A fragment (lo / hi, 32 bytes = 128 fp8), B 16-byte half h (128 fp4), scale byte h of both scale words
        for h in 0..2u8 {
            for &(acc, a, bf, sa, sb, byte_a) in blocks {
                let ah = if h == 0 { a } else { VarId(a.0 + a_hi_off) };
                let ys = b.var_slice4(bf, h);
                let _ = byte_a;
                if mma.swap_ab {
                    let k = amdgpu_dsl::MmaKind::scaled_f8f6f4(amdgpu_dsl::MmaShape::M16N16K128, amdgpu_dsl::MmaTy::Fp4, amdgpu_dsl::MmaTy::Fp8);
                    b.var_mfma_scaled(acc, k, ys, Opnd::Var(ah), amdgpu_dsl::MmaScale { sa: Opnd::Var(sb), sb: Opnd::Var(sa), opsel: h | (h << 4) }, false);
                } else {
                    let k = amdgpu_dsl::MmaKind::scaled_f8f6f4(amdgpu_dsl::MmaShape::M16N16K128, amdgpu_dsl::MmaTy::Fp8, amdgpu_dsl::MmaTy::Fp4);
                    b.var_mfma_scaled(acc, k, Opnd::Var(ah), ys, amdgpu_dsl::MmaScale { sa: Opnd::Var(sa), sb: Opnd::Var(sb), opsel: h | (h << 4) }, false);
                }
            }
        }
        return Vec::new();
    }
    let ops = |&(acc, a, bf, sa, sb, byte_a): &(VarId, VarId, VarId, VarId, VarId, u8)| {
        if mma.swap_ab { (acc, bf, a, sb, sa, 0, byte_a) } else { (acc, a, bf, sa, sb, byte_a, 0) }
    };
    match elem {
        Elem::Fp8W4 | Elem::Bf16W4 => unreachable!("mixed precision handled above"),
        Elem::Fp8 => {
            for blk in blocks {
                let (acc, x, y, sx, sy, bx, by) = ops(blk);
                b.var_mfma_scale16(acc, Opnd::Var(x), Opnd::Var(y), Opnd::Var(sx), Opnd::Var(sy), bx, by);
            }
        }
        Elem::Fp8Bs => {
            // fp32 block scales (1 x 128 A, 128 x 128 B): per block the K tile's product into a fresh accumulator (unit E8M0 scales, zero C);
            // the acc += t (sa sb) updates are returned and emitted by `bs_fma` at the end of the tile (MFMA latency off the critical path)
            let k = amdgpu_dsl::MmaKind::scaled_f8f6f4(amdgpu_dsl::MmaShape::M16N16K128, amdgpu_dsl::MmaTy::Fp8, amdgpu_dsl::MmaTy::Fp8);
            let u = Opnd::Val(unit.expect("unit scales"));
            return blocks
                .iter()
                .map(|blk| {
                    let (_, x, y, ..) = ops(blk);
                    let t = b.mfma_scaled(k, Opnd::Var(x), Opnd::Var(y), None, amdgpu_dsl::MmaScale { sa: u, sb: u, opsel: 0 });
                    (blk.0, t, blk.3, blk.4)
                })
                .collect();
        }
        Elem::Fp4 => {
            // two fp4 x fp4 scaled MFMAs per block: 16-byte half h of each fragment, the scale byte h of the (shifted) scale words
            let k = amdgpu_dsl::MmaKind::scaled_f8f6f4(amdgpu_dsl::MmaShape::M16N16K128, amdgpu_dsl::MmaTy::Fp4, amdgpu_dsl::MmaTy::Fp4);
            for h in 0..2u8 {
                for blk in blocks {
                    let (acc, x, y, sx, sy, ..) = ops(blk);
                    let xs = b.var_slice4(x, h);
                    let ys = b.var_slice4(y, h);
                    // (shuffled scales: `by` = 2 on the odd tile of a pair - its bytes are 2, 3 of the pair's dword)
                    let sel = h + blk.5;
                    b.var_mfma_scaled(acc, k, xs, ys, amdgpu_dsl::MmaScale { sa: Opnd::Var(sx), sb: Opnd::Var(sy), opsel: sel | (sel << 4) }, false);
                }
            }
        }
        Elem::I8 => {
            let k = amdgpu_dsl::MmaKind::new(amdgpu_dsl::MmaShape::M16N16K64, amdgpu_dsl::MmaTy::I8);
            for h in 0..2u8 {
                for blk in blocks {
                    let (acc, x, y, ..) = ops(blk);
                    let xs = b.var_slice4(x, h);
                    let ys = b.var_slice4(y, h);
                    b.var_mfma(acc, k, xs, ys, false);
                }
            }
        }
        Elem::Bf16 => {
            let k = amdgpu_dsl::MmaKind::new(amdgpu_dsl::MmaShape::M16N16K32, amdgpu_dsl::MmaTy::Bf16);
            for h in 0..2u8 {
                for blk in blocks {
                    let (acc, x, y, ..) = ops(blk);
                    let xs = b.var_slice4(x, h);
                    let ys = b.var_slice4(y, h);
                    b.var_mfma(acc, k, xs, ys, false);
                }
            }
        }
    }
    Vec::new()
}

/// The deferred block-scale updates: acc += t (sa sb) lane by lane.
pub fn bs_fma(b: &mut FnBuilder, pending: &[(VarId, ValueId, VarId, VarId)]) {
    for &(acc, t, sa, sb) in pending {
        let (fa, fb) = (b.bitcast_f32(Opnd::Var(sa)), b.bitcast_f32(Opnd::Var(sb)));
        let s = b.mul_f32(fa, fb);
        for r in 0..4u8 {
            let tl = b.frag4_lane(t, r);
            let tl = b.bitcast_f32(tl);
            let al = b.frag4_lane(Opnd::Var(acc), r);
            let al = b.bitcast_f32(al);
            let f = b.let_(Op::FmaF32(Opnd::Val(tl), Opnd::Val(s), Opnd::Val(al)));
            b.var_set_lane(acc, r, f);
        }
    }
}
