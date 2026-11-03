//! MXFP8 blockscale GEMM (P4 target: aiter PR #6696 4-wave pin-AGPR kernel, docs/mxfp8-gemm-4wave-analysis.md), same 96-byte ABI:
//!   C[m, n] (bf16) = sum_k A[m, k] 2^(sa[k/128][m] - 127) * B[n, k] 2^(sb[n/128][k/128] - 127)
//!   A [M, K] fp8 e4m3 row-major; B [N, K] fp8 preshuffled (aiter shuffle_weight 16x16); sa E8M0 [K/128][M]; sb E8M0 [N/128][K/128].
//!
//! v0 (correctness first): VGPR-only, no LDS. CTA 128x128 (4 waves 2x2, 64x64 per wave = 2x2 tiles of v_mfma_scale_f32_32x32x64_f8f6f4), K step 64,
//! operand fragments loaded straight from global memory. EVERY address is derived from `amdgpu-layout` layouts (tensor layouts x MFMA thread-value
//! layouts, issue plans with immediate folding); the only hand code is the lowering of the symbolic index expressions to DSL integer ops.
//! The MFMA runs "swapped" (hardware A = the B fragment, hardware B = the A fragment, scales swapped), so a lane holds 4 consecutive N of one row M
//! of C (one 8-byte bf16 store per 4 values), exactly like the vendor kernel.

use amdgpu_layout::{layout_to_offsets, make_layout, tup, Dyns, ImmRange, IssuePlan, Ix, Layout, Swizzle};

use crate::{DType, FnBuilder, Function, Op, Opnd, ValueId, I32};

/// Lowers a symbolic layout index (`amdgpu_layout::Ix`) to a DSL integer expression; `sym` maps the dynamic symbols to bound values.
fn lower(ix: &Ix, sym: &dyn Fn(u32) -> I32) -> I32 {
    match ix {
        Ix::C(k) => I32::c(*k as i32),
        Ix::Sym(s) => sym(*s),
        Ix::Sum(terms, k) => {
            let mut acc: Option<I32> = None;
            for (t, c) in terms {
                let v = lower(t, sym);
                let tv = if *c == 1 { v } else { v * (*c as i32) };
                acc = Some(match acc {
                    None => tv,
                    Some(a) => a + tv,
                });
            }
            let a = acc.expect("sum with terms");
            if *k == 0 {
                a
            } else {
                a + (*k as i32)
            }
        }
        Ix::Mul(a, b) => lower(a, sym) * lower(b, sym),
        Ix::Div(a, k) => {
            assert!(k.count_ones() == 1, "layout lowering: division by a non power of two {k}");
            lower(a, sym).lshr(k.trailing_zeros() as i32)
        }
        Ix::Mod(a, k) => {
            assert!(k.count_ones() == 1, "layout lowering: remainder by a non power of two {k}");
            lower(a, sym) & (*k as i32 - 1)
        }
        Ix::Xor(a, b) => lower(a, sym) ^ lower(b, sym),
        Ix::And(a, m) => lower(a, sym) & (*m as i32),
        Ix::Shl(a, s) => lower(a, sym) << (*s as i32),
        Ix::Shr(a, s) => lower(a, sym).lshr(*s as i32),
    }
}

/// MFMA 32x32x64 f8 operand thread-value layout (element index row + 32 k, bytes): lane l holds row l % 32 and k = 16 (l / 32) + {0..15} and
/// 32 + 16 (l / 32) + {0..15}.
fn mfma_op_32x32x64() -> Layout {
    make_layout(tup![tup![32, 2], tup![16, 2]], tup![tup![1, 512], tup![32, 1024]])
}
/// MFMA 32x32 C thread-value layout in the swapped orientation (element index n + 32 m of the C tile): lane l value r -> hardware row (= n)
/// 8 (r / 4) + 4 (l / 32) + r % 4, hardware column (= m) l % 32.
fn mfma_c_32x32_swapped() -> Layout {
    make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![32, 4], tup![1, 8]])
}

pub fn gemm_mxfp8_bs_v0() -> Function {
    let mut b = FnBuilder::new("gemm_mxfp8_bs_v0", 256);
    // ---- 96-byte ABI (opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ----
    for (name, off) in [("ptr_a", 0u32), ("ptr_b", 8), ("ptr_c", 16), ("ptr_sfa", 64), ("ptr_sfb", 72)] {
        b.arg(name, DType::Ptr, off, 8);
    }
    for (name, off) in [("m", 24u32), ("n", 28), ("k", 32), ("batch", 36), ("stride_a", 40), ("stride_b", 44), ("stride_c", 48), ("stride_a_batch", 52), ("stride_b_batch", 56), ("stride_c_batch", 60), ("stride_sfa", 80), ("stride_sfb", 84), ("stride_sfa_batch", 88), ("stride_sfb_batch", 92)] {
        b.arg(name, DType::I32, off, 4);
    }
    let ptr_a = b.let_(Op::KernargPtr { offset: 0 });
    let ptr_b = b.let_(Op::KernargPtr { offset: 8 });
    let ptr_c = b.let_(Op::KernargPtr { offset: 16 });
    let ptr_sfa = b.let_(Op::KernargPtr { offset: 64 });
    let ptr_sfb = b.let_(Op::KernargPtr { offset: 72 });
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let n_v = b.let_(Op::KernargI32 { offset: 28 });
    let k_v = b.let_(Op::KernargI32 { offset: 32 });
    let kb_v = b.let_(Op::KernargI32 { offset: 84 }); // stride_sfb = K / 128

    // ---- symbols: the GPU values the layouts are written in ----
    let mut d = Dyns::new();
    let s_lane = d.sym("lane", 0, 63);
    let s_k = d.sym_aligned("K", 128, 1 << 20, 128);
    let s_m = d.sym_aligned("M", 128, 1 << 20, 128);
    let s_n = d.sym_aligned("N", 128, 1 << 20, 128);
    let s_kb = d.sym("KB", 1, 1 << 13);
    let tid = b.let_(Op::WorkitemId);
    let lane = b.bind_i32(crate::tile::ev(tid) & 63);
    let wave = b.bind_i32(crate::tile::ev(tid).lshr(6));
    let syms = [lane, k_v, m_v, n_v, kb_v];
    let symf = |s: u32| -> I32 { crate::tile::ev(syms[s as usize]) };
    let bx = b.let_(Op::WorkgroupIdX); // N tile
    let by = b.let_(Op::WorkgroupIdY); // M tile
    // per-wave origin of its 64x64 C block
    let m_w = b.bind_i32((crate::tile::ev(by) << 7) + ((crate::tile::ev(wave) & 1) << 6));
    let n_w = b.bind_i32((crate::tile::ev(bx) << 7) + (crate::tile::ev(wave).lshr(1) << 6));

    // ---- tensor layouts (byte offsets) ----
    // A [M, K] row-major
    let a_tile = make_layout(tup![32, 64], tup![s_k.clone(), 1]);
    // B preshuffled (shuffle_weight 16x16): (n, k) -> (n / 16) 16 K + (k / 16) 256 + (n % 16) 16 + k % 16
    let b_tile = make_layout(tup![tup![16, 2], tup![16, 4]], tup![tup![16, s_k.mul_c(16)], tup![1, 256]]);
    // C bf16 [M, N] row-major, tile index (n, m) of the swapped C layout
    let c_tile = make_layout(tup![32, 32], tup![2, s_n.mul_c(2)]);
    let op_plan = IssuePlan::new(mfma_op_32x32x64(), 16); // two 16-byte loads per 32-byte fragment
    let c_plan = IssuePlan::new(mfma_c_32x32_swapped(), 4); // 4 consecutive n per store (8 bytes)

    let _ = (&s_m, &s_kb);
    // symbolic per-wave origins
    let mut d2 = d.clone();
    let s_mw = d2.sym_aligned("m_w", 0, 1 << 20, 64);
    let s_nw = d2.sym_aligned("n_w", 0, 1 << 20, 64);
    let syms2 = [lane, k_v, m_v, n_v, kb_v, m_w, n_w];
    let symf2 = |s: u32| -> I32 { crate::tile::ev(syms2[s as usize]) };
    // A fragment address of sub-tile i (rows m_w + 32 i), per issue: ptr_a + offset
    let mut a_addr: Vec<Vec<ValueId>> = Vec::new();
    let mut b_addr: Vec<Vec<ValueId>> = Vec::new();
    for i in 0..2i64 {
        let rows0 = s_mw.add_c(32 * i);
        let exprs = op_plan.offsets(&a_tile, &s_lane, &rows0.mul(&s_k), &d2);
        a_addr.push(exprs.iter().map(|e| {
            let off = b.bind_i32(lower(e, &symf2));
            b.ptr_add(ptr_a, off)
        }).collect());
        // B: rows n_w + 32 i are whole 16-row groups: origin offset (n0 / 16) 16 K = n0 K
        let n0 = s_nw.add_c(32 * i);
        let exprs = op_plan.offsets(&b_tile, &s_lane, &n0.mul(&s_k), &d2);
        b_addr.push(exprs.iter().map(|e| {
            let off = b.bind_i32(lower(e, &symf2));
            b.ptr_add(ptr_b, off)
        }).collect());
    }
    // scale addresses: sa [KB][M]: byte (kb M + m), m = m_w + 32 i + lane % 32 (the swapped MFMA's B-operand rows); loaded as the dword holding it
    let sa_lay = make_layout(tup![2, 32], tup![32, 1]); // (i, lane%32) -> row offset inside the wave block
    let mut sa_byte: Vec<ValueId> = Vec::new();
    for i in 0..2i64 {
        let r = sa_lay.crd2idx(&tup![i, s_lane.mod_c(32, &d2)], &d2).add(&s_mw);
        sa_byte.push(b.bind_i32(lower(&r, &symf2)));
    }
    // sb [N/128][KB]: byte (n_w / 128) KB + kb
    let sb_row = b.bind_i32(lower(&s_nw.div_c(128, &d2).mul(&s_kb), &symf2));

    // C store addresses (8 bytes per issue): ptr_c + 2 (m N + n)
    let mut c_addr: Vec<Vec<(ValueId, u32)>> = Vec::new();
    for i in 0..2i64 {
        for j in 0..2i64 {
            let origin = s_mw.add_c(32 * i).mul(&s_n).mul_c(2).add(&s_nw.add_c(32 * j).mul_c(2));
            let exprs = c_plan.offsets(&c_tile, &s_lane, &origin, &d2);
            let plan = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::GLOBAL, &d2);
            let bases: Vec<ValueId> = plan.bases.iter().map(|e| {
                let off = b.bind_i32(lower(e, &symf2));
                b.ptr_add(ptr_c, off)
            }).collect();
            c_addr.push(plan.issues.iter().map(|&(bi, imm)| (bases[bi], imm as u32)).collect());
        }
    }

    // ---- K loop: per 128 keys, two MFMA k-steps of 64; accumulators C[i][j] (i = m sub-tile, j = n sub-tile) ----
    let zero16 = b.frag_zero(DType::Frag16);
    let c0 = b.c_i32(0);
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::Val(c0))];
    inits.extend((0..4).map(|_| (DType::Frag16, Opnd::Val(zero16))));
    let vars = b.begin_loop(&inits);
    let kb = vars[0];
    let kbx = crate::tile::varx(kb);
    // scales of this K block
    let sa_v: Vec<ValueId> = (0..2).map(|i| {
        let byte = b.bind_i32(kbx.clone() * crate::tile::ev(m_v) + crate::tile::ev(sa_byte[i]));
        let dw = b.bind_i32(byte_dword(&crate::tile::ev(byte)));
        let p = b.ptr_add(ptr_sfa, dw);
        let w = b.load_f32(p, 0);
        let wi = b.bitcast_i32(w);
        b.bind_i32(crate::tile::ev(wi).lshr((crate::tile::ev(byte) & 3) << 3))
    }).collect();
    let sb_v = {
        let byte = b.bind_i32(crate::tile::ev(sb_row) + kbx.clone());
        let dw = b.bind_i32(byte_dword(&crate::tile::ev(byte)));
        let p = b.ptr_add(ptr_sfb, dw);
        let w = b.load_f32(p, 0);
        let wi = b.bitcast_i32(w);
        b.bind_i32(crate::tile::ev(wi).lshr((crate::tile::ev(byte) & 3) << 3))
    };
    for h in 0..2i32 {
        // K byte offset of this step: 128 kb + 64 h (A: + k; B preshuffled: + (k / 16) 256 = 16 k)
        let ka = b.bind_i32((kbx.clone() << 7) + 64 * h);
        let kbo = b.bind_i32(crate::tile::ev(ka) << 4);
        let load_frag = |b: &mut FnBuilder, addrs: &[ValueId], koff: ValueId| -> ValueId {
            let p0 = b.ptr_add(addrs[0], koff);
            let p1 = b.ptr_add(addrs[1], koff);
            let lo = b.load_b128(p0, 0);
            let hi = b.load_b128(p1, 0);
            b.cat_frag8(lo, hi)
        };
        let af: Vec<ValueId> = (0..2).map(|i| load_frag(&mut b, &a_addr[i], ka)).collect();
        let bf: Vec<ValueId> = (0..2).map(|j| load_frag(&mut b, &b_addr[j], kbo)).collect();
        for i in 0..2usize {
            for j in 0..2usize {
                // swapped: hardware A = B fragment (rows n), hardware B = A fragment (rows m); scale_a = sb, scale_b = sa
                b.var_mfma_scale(vars[1 + 2 * i + j], bf[j], af[i], sb_v, sa_v[i]);
            }
        }
    }
    let kb_next = b.bind_i32(kbx + 1);
    b.set_var(kb, kb_next);
    let stop = b.bind_cond(!crate::tile::ev(kb_next).lt(crate::tile::ev(kb_v)));
    b.break_if(stop);
    b.end_loop();
    // ---- epilogue: bf16, 8-byte stores ----
    for i in 0..2usize {
        for j in 0..2usize {
            let acc = vars[1 + 2 * i + j];
            for g in 0..4usize {
                let v: Vec<ValueId> = (0..4).map(|q| {
                    let l = b.frag16_lane(Opnd::Var(acc), (4 * g + q) as u8);
                    b.bitcast_f32(l)
                }).collect();
                let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
                let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
                let f2 = b.frag2(w0, w1);
                let (p, imm) = c_addr[2 * i + j][g];
                b.store_f32x2(p, imm, f2);
            }
        }
    }
    let _ = n_v;
    b.finish()
}

/// Byte address of the dword containing byte `x` (scale tensors are read a dword at a time; the MFMA uses scale byte 0 after the shift).
fn byte_dword(x: &I32) -> I32 {
    x.clone() & -4
}
