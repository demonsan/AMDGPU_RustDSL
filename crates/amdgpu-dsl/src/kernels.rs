//! Kernel definitions written against the typed DSL.
//!
//! `attention_fwd` is a real (if deliberately simple) dense attention kernel:
//! one wave per block, one lane per query row, online softmax over the key
//! dimension, f32 accumulation and f16 output. It is the first correctness
//! milestone; tiled MFMA/PV variants are separate kernels added later.

use crate::*;

/// Head dimension of `attention_fwd`.
pub const HEAD_DIM: usize = 64;
/// Query rows handled per block (one per lane of a wave64).
pub const Q_ROWS: usize = 64;
/// Kernarg layout for `attention_fwd`.
pub const KERNARG_SIZE: u32 = 48;

/// Dense attention: `out[b,h,q,:] = softmax(scale * q[b,h,q,:] @ k[b,h,:,:]^T) @ v[b,h,:,:]`.
///
/// Layout assumption (checked by the host harness): `[B, H, S, D]` contiguous,
/// `D = HEAD_DIM`, and `S` a multiple of `Q_ROWS`. Grid is
/// `(S / Q_ROWS, H, B)`, block is `(Q_ROWS, 1, 1)`.
pub fn attention_impl(debug: bool) -> Function {
    let name = if debug { "attn_debug" } else { "attention_fwd" };
    let mut b = FnBuilder::new(name, Q_ROWS as u32);
    b.arg("q", DType::Ptr, 0, 8);
    b.arg("k", DType::Ptr, 8, 8);
    b.arg("v", DType::Ptr, 16, 8);
    b.arg("o", DType::Ptr, 24, 8);
    b.arg("seqlen_k", DType::I32, 32, 4);
    b.arg("scale", DType::F32, 36, 4);
    b.arg("nhead", DType::I32, 40, 4);
    b.arg("seqlen_q", DType::I32, 44, 4);

    // --- identifiers and arguments -------------------------------------
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let by = b.let_(Op::WorkgroupIdY);
    let bz = b.let_(Op::WorkgroupIdZ);

    let q_ptr = b.let_(Op::KernargPtr { offset: 0 });
    let k_ptr = b.let_(Op::KernargPtr { offset: 8 });
    let v_ptr = b.let_(Op::KernargPtr { offset: 16 });
    let o_ptr = b.let_(Op::KernargPtr { offset: 24 });
    let seqlen_k = b.let_(Op::KernargI32 { offset: 32 });
    let scale = b.let_(Op::KernargF32 { offset: 36 });
    let nhead = b.let_(Op::KernargI32 { offset: 40 });
    let seqlen_q = b.let_(Op::KernargI32 { offset: 44 });

    let c1 = b.c_i32(1);
    let c7 = b.c_i32(7); // log2(128 bytes per row)
    let c_qrows = b.c_i32(Q_ROWS as i32);
    let c_row_bytes = b.c_i32((HEAD_DIM * 2) as i32);
    let c_zero_f = b.c_f32(0.0);
    let c_neg_inf = b.c_f32(f32::NEG_INFINITY);

    // --- addresses ------------------------------------------------------
    let block_row0 = b.mul_i32(bx, c_qrows);
    let row = b.add_i32(block_row0, tid);
    let head_base = b.mul_i32(bz, nhead);
    let head_index = b.add_i32(head_base, by);

    let head_row = b.mul_i32(head_index, seqlen_q);
    let q_row_index = b.add_i32(head_row, row);
    let q_off = b.mul_i32(q_row_index, c_row_bytes);
    let q_row_ptr = b.ptr_add(q_ptr, q_off);
    let o_row_ptr = b.ptr_add(o_ptr, q_off);

    // Per-key rows share one invariant base offset.
    let kv_head = b.mul_i32(head_index, seqlen_k);
    let kv_base = b.mul_i32(kv_head, c_row_bytes);

    // --- load the query row, widening to f32 ----------------------------
    let mut q = Vec::with_capacity(HEAD_DIM);
    let mut d = 0usize;
    while d < HEAD_DIM {
        let mut bits = [0u32; 4];
        for (i, slot) in bits.iter_mut().enumerate() {
            *slot = b.load_u16(q_row_ptr, (2 * (d + i)) as u32);
        }
        for i in 0..4 {
            q.push(b.cvt_f16_to_f32(bits[i]));
        }
        d += 4;
    }

    // --- online softmax loop over keys ----------------------------------
    // Loop-carried: key index, running max, running normaliser, output row.
    let mut inits: Vec<(DType, Opnd)> = Vec::with_capacity(3 + HEAD_DIM);
    let k_init = b.c_i32(0);
    inits.push((DType::I32, Opnd::Val(k_init)));
    inits.push((DType::F32, Opnd::Val(c_neg_inf)));
    inits.push((DType::F32, Opnd::Val(c_zero_f)));
    for _ in 0..HEAD_DIM {
        inits.push((DType::F32, Opnd::Val(c_zero_f)));
    }
    let vars = b.begin_loop(&inits);
    let k_var = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let acc: Vec<VarId> = vars[3..].to_vec();

    let k_shifted = b.shl_i32(k_var, c7);
    let row_off = b.add_i32(kv_base, k_shifted);
    let k_row_ptr = b.ptr_add(k_ptr, row_off);
    let v_row_ptr = b.ptr_add(v_ptr, row_off);

    // qk = q . k  (f32 accumulate; grouped loads keep waitcnts cheap)
    let mut s = c_zero_f;
    let mut d = 0usize;
    while d < HEAD_DIM {
        let mut bits = [0u32; 4];
        for (i, slot) in bits.iter_mut().enumerate() {
            *slot = b.load_u16(k_row_ptr, (2 * (d + i)) as u32);
        }
        for i in 0..4 {
            let kf = b.cvt_f16_to_f32(bits[i]);
            s = b.fma_f32(q[d + i], kf, s);
        }
        d += 4;
    }

    let scaled = b.mul_f32(s, scale);
    let m_new = b.max_f32(m_var, scaled);
    let m_delta = b.sub_f32(m_var, m_new);
    let corr = b.exp_f32(m_delta);
    let p_delta = b.sub_f32(scaled, m_new);
    let p = b.exp_f32(p_delta);
    let l_new = b.fma_f32(l_var, corr, p);

    // out = out * corr + p * v
    let mut next_acc = Vec::with_capacity(HEAD_DIM);
    let mut d = 0usize;
    while d < HEAD_DIM {
        let mut bits = [0u32; 4];
        for (i, slot) in bits.iter_mut().enumerate() {
            *slot = b.load_u16(v_row_ptr, (2 * (d + i)) as u32);
        }
        for i in 0..4 {
            let vf = b.cvt_f16_to_f32(bits[i]);
            let decayed = b.mul_f32(acc[d + i], corr);
            next_acc.push(b.fma_f32(p, vf, decayed));
        }
        d += 4;
    }

    if debug {
        // Last-iteration values of the softmax intermediates, stored after the
        // real outputs so the harness can tell "math wrong" from "loop-carried
        // variable lost".
        let corr16 = b.cvt_f32_to_f16(corr);
        let p16 = b.cvt_f32_to_f16(p);
        let lnew16 = b.cvt_f32_to_f16(l_new);
        let s16 = b.cvt_f32_to_f16(scaled);
        b.store_u16(o_row_ptr, 8, Opnd::Val(corr16));
        b.store_u16(o_row_ptr, 10, Opnd::Val(p16));
        b.store_u16(o_row_ptr, 12, Opnd::Val(lnew16));
        b.store_u16(o_row_ptr, 14, Opnd::Val(s16));
    }

    let k_next = b.add_i32(k_var, c1);
    b.set_var(k_var, k_next);
    b.set_var(m_var, m_new);
    b.set_var(l_var, l_new);
    for i in 0..HEAD_DIM {
        b.set_var(acc[i], next_acc[i]);
    }

    let more = b.cmp_lt_i32(k_next, seqlen_k);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // --- normalise and store -------------------------------------------
    if debug {
        // Diagnostic kernel: write the loop-carried state so the harness can
        // compare it against the expected values directly.
        let l_f16 = b.cvt_f32_to_f16(l_var);
        let m_f16 = b.cvt_f32_to_f16(m_var);
        let acc0_f16 = b.cvt_f32_to_f16(acc[0]);
        let q0_f16 = b.cvt_f32_to_f16(q[0]);
        b.store_u16(o_row_ptr, 0, Opnd::Val(l_f16));
        b.store_u16(o_row_ptr, 2, Opnd::Val(m_f16));
        b.store_u16(o_row_ptr, 4, Opnd::Val(acc0_f16));
        b.store_u16(o_row_ptr, 6, Opnd::Val(q0_f16));
    } else {
        let inv_l = b.rcp_f32(l_var);
        let mut d = 0usize;
        while d < HEAD_DIM {
            let mut out = [0u32; 4];
            for (i, slot) in out.iter_mut().enumerate() {
                let normed = b.mul_f32(acc[d + i], inv_l);
                *slot = b.cvt_f32_to_f16(normed);
            }
            for i in 0..4 {
                b.store_u16(o_row_ptr, (2 * (d + i)) as u32, Opnd::Val(out[i]));
            }
            d += 4;
        }
    }

    b.finish()
}

/// Dense attention kernel.
pub fn attention_fwd() -> Function {
    attention_impl(false)
}

/// Same attention kernel, but stores `l`, `m`, `acc[0]` and `q[0]` at the start
/// of each output row instead of the normalised result. Used to debug the
/// loop-carried softmax state on hardware.
pub fn attn_debug() -> Function {
    attention_impl(true)
}

/// Minimal probe kernel: `o[tid] = q[tid]`, plus `n` stored at byte 128.
/// Used to validate the kernarg/ABI plumbing independently of attention math.
pub fn probe_store() -> Function {
    let mut b = FnBuilder::new("probe_store", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("q", DType::Ptr, 8, 8);
    b.arg("count", DType::I32, 16, 4);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let q = b.let_(Op::KernargPtr { offset: 8 });
    let count = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let c2 = b.c_i32(2);
    let off = b.mul_i32(tid, c2);
    let src = b.ptr_add(q, off);
    let dst = b.ptr_add(o, off);
    let val = b.load_u16(src, 0);
    b.store_u16(dst, 0, val);
    let c128 = b.c_i32(128);
    let dst2 = b.ptr_add(o, c128);
    b.store_u16(dst2, 0, count);
    b.finish()
}

/// Probe kernel: writes `exp(1)`, `exp(2)` and `rcp(3)` so the harness can
/// confirm the base and accuracy of the hardware transcendental functions.
pub fn probe_math() -> Function {
    let mut b = FnBuilder::new("probe_math", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let one = b.c_f32(1.0);
    let two = b.c_f32(2.0);
    let three = b.c_f32(3.0);
    let e1 = b.exp_f32(one);
    let e2 = b.exp_f32(two);
    let r3 = b.rcp_f32(three);
    b.store_f32(o, 0, e1);
    b.store_f32(o, 4, e2);
    b.store_f32(o, 8, r3);
    b.finish()
}