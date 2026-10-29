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

/// Minimal probe kernel: `o[tid] = q[tid]`, plus the constant 7 stored at byte 128.
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

/// MFMA micro-kernel: one `v_mfma_f32_16x16x16_f16` per lane with all-ones
/// operands, so every accumulator element must be exactly 16.0 regardless of
/// the fragment layout.
pub fn probe_mfma() -> Function {
    let mut b = FnBuilder::new("probe_mfma", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    // Two f16 1.0 values in one dword.
    let ones = b.c_i32(0x3C003C00_u32 as i32);
    let zero = b.c_i32(0);
    let a = b.frag2(ones, ones);
    let bfrag = b.frag2(ones, ones);
    let c = b.frag4(zero, zero, zero, zero);
    let d = b.mfma_f16(a, bfrag, c);

    let c16 = b.c_i32(16);
    let lane_off = b.mul_i32(tid, c16);
    let row = b.ptr_add(o, lane_off);
    for i in 0..4u8 {
        let bits = b.frag4_lane(d, i);
        let value = b.bitcast_f32(bits);
        b.store_f32(row, 4 * i as u32, value);
    }
    b.finish()
}
/// LDS probe: every lane writes its id to LDS, then reads the value written by
/// the next lane after a workgroup barrier.
pub fn probe_lds() -> Function {
    let mut b = FnBuilder::new("probe_lds", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c63 = b.c_i32(63);
    let next = b.add_i32(tid, c1);
    let wrapped = b.let_(Op::AndI32(Opnd::Val(next), Opnd::Val(c63)));
    let self_off = b.mul_i32(tid, c4);
    let next_off = b.mul_i32(wrapped, c4);
    b.store_lds_b32(Opnd::Val(self_off), Opnd::Val(tid));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let raw = b.load_lds_b32(Opnd::Val(next_off));
    let value = b.bitcast_f32(Opnd::Val(raw));
    let dst = b.ptr_add(o, self_off);
    b.store_f32(dst, 0, Opnd::Val(value));
    b.finish()
}
/// Wide MFMA probe: `v_mfma_f32_32x32x8_f16` with all-ones operands and a
/// zeroed 16-dword accumulator, so every element must be exactly K = 8.0.
pub fn probe_mfma_wide() -> Function {
    let mut b = FnBuilder::new("probe_mfma_wide", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let ones = b.c_i32(0x3C003C00_u32 as i32);
    let a = b.frag2(ones, ones);
    let bfrag = b.frag2(ones, ones);
    let c = b.frag_zero(DType::Frag16);
    let d = b.mfma_f16_wide(a, bfrag, c);

    let c64 = b.c_i32(64);
    let lane_off = b.mul_i32(tid, c64);
    let row = b.ptr_add(o, lane_off);
    for i in 0..16u8 {
        let bits = b.frag16_lane(d, i);
        let value = b.bitcast_f32(bits);
        b.store_f32(row, 4 * i as u32, value);
    }
    b.finish()
}
/// FP8 MFMA probe: all-ones E4M3 operands, so `16x16x32` yields K = 32.0 and
/// `32x32x16` yields K = 16.0 in every accumulator element.
pub fn probe_mfma_fp8() -> Function {
    let mut b = FnBuilder::new("probe_mfma_fp8", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    // E4M3 1.0 is 0x38; four fp8 values per dword.
    let ones4 = b.c_i32(0x38383838_u32 as i32);

    let a_small = b.frag2(ones4, ones4);
    let b_small = b.frag2(ones4, ones4);
    let c_small = b.frag_zero(DType::Frag4);
    let d_small = b.mfma_fp8_16x16x32(a_small, b_small, c_small);

    let a_wide = b.frag2(ones4, ones4);
    let b_wide = b.frag2(ones4, ones4);
    let c_wide = b.frag_zero(DType::Frag16);
    let d_wide = b.mfma_fp8_32x32x16(a_wide, b_wide, c_wide);

    let c80 = b.c_i32(80);
    let lane_off = b.mul_i32(tid, c80);
    let row = b.ptr_add(o, lane_off);
    for i in 0..4u8 {
        let bits = b.frag4_lane(d_small, i);
        let value = b.bitcast_f32(bits);
        b.store_f32(row, 4 * i as u32, value);
    }
    for i in 0..16u8 {
        let bits = b.frag16_lane(d_wide, i);
        let value = b.bitcast_f32(bits);
        b.store_f32(row, 16 + 4 * i as u32, value);
    }
    b.finish()
}

/// Block-scaled fp8 MFMA probe. All-ones E4M3 A/B with K = 64, so the result
/// depends only on how the scale operands are interpreted:
///   (127, 127)          -> 64   if scales are E8M0 exponents with 2^(x-127)
///   (127, 0x807f)       -> 96   if the dword packs two 32-block scales
///                       -> 128  if only the low byte is used
///   (0x807f, 127)       -> 96 or 128
///   (0x807f, 0x807f)    -> 160  or 256
pub fn probe_mfma_scale() -> Function {
    let mut b = FnBuilder::new("probe_mfma_scale", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let ones = b.c_i32(0x38383838_u32 as i32);

    let a = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let bfrag = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let zero = b.frag_zero(DType::Frag16);

    // 127 -> 2^0 = 1x, 128 -> 2^1 = 2x.
    let d1 = b.mfma_scale_fp8(a, bfrag, zero, 127, 127);
    let d2 = b.mfma_scale_fp8(a, bfrag, zero, 127, 128);
    // A zero low byte makes the scale ~0 if only the low byte is used; if the
    // dword packs two 32-block scales these two differ (64 vs 32).
    let d3 = b.mfma_scale_fp8(a, bfrag, zero, 127, 0x8000);
    let d4 = b.mfma_scale_fp8(a, bfrag, zero, 127, 0x7f00);

    let c256 = b.c_i32(256);
    let lane_off = b.mul_i32(tid, c256);
    let row = b.ptr_add(o, lane_off);
    for (v, frag) in [d1, d2, d3, d4].iter().enumerate() {
        for i in 0..16u8 {
            let bits = b.frag16_lane(*frag, i);
            let value = b.bitcast_f32(bits);
            b.store_f32(row, (v * 64 + 4 * i as usize) as u32, value);
        }
    }
    b.finish()
}

/// Minimal multi-wave LDS exchange: every lane writes its id into LDS, the whole
/// workgroup synchronises, and each lane reads the id written by the partner
/// role (waves 0-3 <-> 4-7).
pub fn probe_roles() -> Function {
    let mut b = FnBuilder::new("probe_roles", 512);
    b.lds(2048);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c6 = b.c_i32(6);
    // Wave id from the vector workitem id: uniform inside a wave, so it is safe
    // to use it in the per-lane address computation.
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let c4 = b.c_i32(4);
    let c7 = b.c_i32(7);
    let c63 = b.c_i32(63);
    let c64 = b.c_i32(64);

    let off = b.mul_i32(tid, c4);
    b.store_lds_b32(Opnd::Val(off), Opnd::Val(tid));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let wave_plus = b.add_i32(wave, c4);
    let partner = b.let_(Op::AndI32(Opnd::Val(wave_plus), Opnd::Val(c7)));
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let partner_row = b.mul_i32(partner, c64);
    let partner_index = b.add_i32(partner_row, lane);
    let partner_off = b.mul_i32(partner_index, c4);
    let got = b.load_lds_b32(Opnd::Val(partner_off));
    let value = b.bitcast_f32(got);

    let out_off = b.mul_i32(tid, c4);
    let out_ptr = b.ptr_add(o, out_off);
    b.store_f32(out_ptr, 0, value);

    b.finish()
}

/// Two-wave LDS probe (block 128): every lane writes its id, the workgroup
/// synchronises, then the lane reads its own slot and the partner wave's slot.
pub fn probe_waves2() -> Function {
    let mut b = FnBuilder::new("probe_waves2", 128);
    b.lds(512);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c6 = b.c_i32(6);
    let c63 = b.c_i32(63);
    let c64 = b.c_i32(64);
    let c128 = b.c_i32(128);
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let off = b.mul_i32(tid, c4);

    b.store_lds_b32(Opnd::Val(off), Opnd::Val(tid));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let own = b.load_lds_b32(Opnd::Val(off));
    let own_value = b.bitcast_f32(Opnd::Val(own));
    let own_ptr = b.ptr_add(o, off);
    b.store_f32(own_ptr, 0, Opnd::Val(own_value));

    let partner = b.xor_i32(wave, c1);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let partner_row = b.mul_i32(partner, c64);
    let partner_index = b.add_i32(partner_row, lane);
    let partner_off = b.mul_i32(partner_index, c4);
    let got = b.load_lds_b32(Opnd::Val(partner_off));
    let value = b.bitcast_f32(Opnd::Val(got));
    let out2_index = b.add_i32(tid, c128);
    let out2_off = b.mul_i32(out2_index, c4);
    let out2_ptr = b.ptr_add(o, out2_off);
    b.store_f32(out2_ptr, 0, Opnd::Val(value));

    b.finish()
}
/// Byte-for-byte clone of `probe_lds` with exactly one change: the lane reads
/// its own slot instead of the neighbour's. Everything else (LDS size, barrier
/// sequence, single read, single store, workgroup size) is identical.
pub fn probe_lds_self2() -> Function {
    let mut b = FnBuilder::new("probe_lds_self2", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let self_off = b.mul_i32(tid, c4);
    b.store_lds_b32(Opnd::Val(self_off), Opnd::Val(tid));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let raw = b.load_lds_b32(Opnd::Val(self_off));
    let value = b.bitcast_f32(Opnd::Val(raw));
    let dst = b.ptr_add(o, self_off);
    b.store_f32(dst, 0, Opnd::Val(value));
    b.finish()
}
/// FMHA-shaped structural skeleton: 8 waves per block, waves 0-3 take role A and
/// waves 4-7 role B. Four iterations, each with a role-dependent LDS write, two
/// paired workgroup barriers, and a cross-role LDS read. This is the control
/// structure the FMHA reference core loop needs (roles + loop + paired
/// barriers), with data values chosen so a wrong partner or a stale read shows
/// up immediately.
pub fn probe_roles_loop() -> Function {
    let mut b = FnBuilder::new("probe_roles_loop", 512);
    b.lds(2048);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c6 = b.c_i32(6);
    let c7 = b.c_i32(7);
    let c63 = b.c_i32(63);
    let c64 = b.c_i32(64);
    let c512 = b.c_i32(512);
    let c1000 = b.c_i32(1000);
    let c100000 = b.c_i32(100000);

    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let is_a = b.cmp_lt_i32(wave, c4);
    let lds_off = b.mul_i32(tid, c4);
    let wave_plus = b.add_i32(wave, c4);
    let partner = b.let_(Op::AndI32(Opnd::Val(wave_plus), Opnd::Val(c7)));
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let partner_row = b.mul_i32(partner, c64);
    let partner_index = b.add_i32(partner_row, lane);
    let partner_off = b.mul_i32(partner_index, c4);

    let iter_init = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(iter_init))]);
    let iter = vars[0];

    let scaled = b.mul_i32(iter, c1000);
    let base = b.add_i32(scaled, tid);
    b.begin_if(Opnd::Val(is_a));
    b.store_lds_b32(Opnd::Val(lds_off), Opnd::Val(base));
    b.else_();
    let marked = b.add_i32(base, c100000);
    b.store_lds_b32(Opnd::Val(lds_off), Opnd::Val(marked));
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let got = b.load_lds_b32(Opnd::Val(partner_off));
    let out_index_base = b.mul_i32(iter, c512);
    let out_index = b.add_i32(out_index_base, tid);
    let out_off = b.mul_i32(out_index, c4);
    let out_ptr = b.ptr_add(o, out_off);
    let out_value = b.bitcast_f32(Opnd::Val(got));
    b.store_f32(out_ptr, 0, Opnd::Val(out_value));

    // Second barrier: no wave may overwrite LDS before everyone has read it.
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let next = b.add_i32(iter, c1);
    b.set_var(iter, next);
    let more = b.cmp_lt_i32(next, c4);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    b.finish()
}
/// FMHA compute shape: two distinct operands staged in LDS.
///
/// Each lane stages an A tile (16 B at `tid*16`) and a B tile (16 B at
/// `8192 + tid*16`), the workgroup synchronises, both tiles are read back with
/// `ds_read_b128`, doubled into Frag8 operands, and multiplied with the
/// block-scaled fp8 MFMA. All-ones E4M3 with K = 64 means every output element
/// must be exactly 64.0.
pub fn probe_fmha_datapath() -> Function {
    let mut b = FnBuilder::new("probe_fmha_datapath", 512);
    b.lds(16384);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let ones = b.c_i32(0x38383838_u32 as i32);
    let c8k = b.c_i32(8192);
    let c16 = b.c_i32(16);
    let frag = b.frag4(ones, ones, ones, ones);

    let off = b.mul_i32(tid, c16);
    b.store_lds_b128(Opnd::Val(off), Opnd::Val(frag));
    let boff = b.add_i32(c8k, off);
    b.store_lds_b128(Opnd::Val(boff), Opnd::Val(frag));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let a_lo = b.load_lds_b128(Opnd::Val(off));
    let b_lo = b.load_lds_b128(Opnd::Val(boff));

    let a8 = b.cat_frag8(a_lo, a_lo);
    let b8 = b.cat_frag8(b_lo, b_lo);
    let zero = b.frag_zero(DType::Frag16);
    let d = b.mfma_scale_fp8(a8, b8, zero, 127, 127);

    let row_off = b.mul_i32(tid, c16);
    let row = b.ptr_add(o, row_off);
    for i in 0..4u8 {
        let bits = b.frag16_lane(d, i);
        let value = b.bitcast_f32(bits);
        b.store_f32(row, 4 * i as u32, value);
    }
    b.finish()
}
/// A/B paired-loop skeleton with paired barriers.
///
/// 8 waves per block: waves 0-3 are role A, waves 4-7 role B. The outer loop
/// runs two iterations, each with a sequential A phase and B phase. Every inner
/// iteration writes a phase-dependent value into the exchange area of LDS,
/// synchronises, reads the partner lane's value, adds it into this lane's
/// accumulator slot in LDS (read-modify-write) and synchronises again. Keeping
/// the accumulator in LDS avoids the two current IR restrictions (no mutation of
/// an outer loop variable from an inner loop, no use of a loop variable after
/// the loop) and still exercises nested loops, role-dependent values and paired
/// barriers, with a closed-form expected result.
pub fn probe_ab_loops() -> Function {
    let mut b = FnBuilder::new("probe_ab_loops", 512);
    b.lds(4096);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let c6 = b.c_i32(6);
    let c10 = b.c_i32(10);
    let c20 = b.c_i32(20);
    let c256 = b.c_i32(256);
    let c1000 = b.c_i32(1000);
    let c2000 = b.c_i32(2000);
    let c2048 = b.c_i32(2048);

    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let is_a = b.cmp_lt_i32(wave, c4);
    let off = b.mul_i32(tid, c4);
    let partner = b.let_(Op::XorI32(Opnd::Val(tid), Opnd::Val(c256)));
    let partner_off = b.mul_i32(partner, c4);
    let acc_off = b.add_i32(off, c2048);

    let zero = b.c_i32(0);
    b.store_lds_b32(Opnd::Val(acc_off), Opnd::Val(zero));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let outer_init = b.c_i32(0);
    let outer_vars = b.begin_loop(&[(DType::I32, Opnd::Val(outer_init))]);
    let outer = outer_vars[0];

    // ---- A phase -----------------------------------------------------------
    let ia_init = b.c_i32(0);
    let inner_a = b.begin_loop(&[(DType::I32, Opnd::Val(ia_init))]);
    let ia = inner_a[0];

    let t10 = b.mul_i32(outer, c10);
    let base_a = b.add_i32(t10, ia);
    let base_a_tid = b.add_i32(base_a, tid);
    b.begin_if(Opnd::Val(is_a));
    b.store_lds_b32(Opnd::Val(off), Opnd::Val(base_a_tid));
    b.else_();
    let marked_a = b.add_i32(base_a_tid, c1000);
    b.store_lds_b32(Opnd::Val(off), Opnd::Val(marked_a));
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let got_a = b.load_lds_b32(Opnd::Val(partner_off));
    let acc_a = b.load_lds_b32(Opnd::Val(acc_off));
    let acc_a_next = b.add_i32(acc_a, got_a);
    b.store_lds_b32(Opnd::Val(acc_off), Opnd::Val(acc_a_next));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let ia_next = b.add_i32(ia, c1);
    b.set_var(ia, ia_next);
    let more_a = b.cmp_lt_i32(ia_next, c2);
    let stop_a = b.let_(Op::NotI1(Opnd::Val(more_a)));
    b.break_if(Opnd::Val(stop_a));
    b.end_loop();

    // ---- B phase -----------------------------------------------------------
    let ib_init = b.c_i32(0);
    let inner_b = b.begin_loop(&[(DType::I32, Opnd::Val(ib_init))]);
    let ib = inner_b[0];

    let t20 = b.mul_i32(outer, c20);
    let base_b = b.add_i32(t20, ib);
    let base_b_tid = b.add_i32(base_b, tid);
    b.begin_if(Opnd::Val(is_a));
    b.store_lds_b32(Opnd::Val(off), Opnd::Val(base_b_tid));
    b.else_();
    let marked_b = b.add_i32(base_b_tid, c2000);
    b.store_lds_b32(Opnd::Val(off), Opnd::Val(marked_b));
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let got_b = b.load_lds_b32(Opnd::Val(partner_off));
    let acc_b = b.load_lds_b32(Opnd::Val(acc_off));
    let acc_b_next = b.add_i32(acc_b, got_b);
    b.store_lds_b32(Opnd::Val(acc_off), Opnd::Val(acc_b_next));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let ib_next = b.add_i32(ib, c1);
    b.set_var(ib, ib_next);
    let more_b = b.cmp_lt_i32(ib_next, c2);
    let stop_b = b.let_(Op::NotI1(Opnd::Val(more_b)));
    b.break_if(Opnd::Val(stop_b));
    b.end_loop();

    let outer_next = b.add_i32(outer, c1);
    b.set_var(outer, outer_next);
    let more_outer = b.cmp_lt_i32(outer_next, c2);
    let stop_outer = b.let_(Op::NotI1(Opnd::Val(more_outer)));
    b.break_if(Opnd::Val(stop_outer));
    b.end_loop();

    let total = b.load_lds_b32(Opnd::Val(acc_off));
    let out_off = b.mul_i32(tid, c4);
    let out_ptr = b.ptr_add(o, out_off);
    let value = b.bitcast_f32(total);
    b.store_f32(out_ptr, 0, Opnd::Val(value));
    b.finish()
}
/// Nested loops with a register accumulator: the inner loop updates the outer
/// loop's accumulator directly (previously rejected by the verifier), and the
/// accumulator is read after both loops (previously out of scope).
///
/// Expected value per lane: `3 + 2*tid` (inner indices 0+1 per outer iteration,
/// plus the outer index, plus tid).
pub fn probe_nested_acc() -> Function {
    let mut b = FnBuilder::new("probe_nested_acc", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let zero = b.c_i32(0);

    let outer_vars = b.begin_loop(&[(DType::I32, Opnd::Val(zero)), (DType::I32, Opnd::Val(zero))]);
    let acc = outer_vars[0];
    let outer = outer_vars[1];

    let inner_vars = b.begin_loop(&[(DType::I32, Opnd::Val(zero))]);
    let k = inner_vars[0];
    let acc_k = b.add_i32(acc, k);
    b.set_var(acc, acc_k);
    let k_next = b.add_i32(k, c1);
    b.set_var(k, k_next);
    let more_k = b.cmp_lt_i32(k_next, c2);
    let stop_k = b.let_(Op::NotI1(Opnd::Val(more_k)));
    b.break_if(Opnd::Val(stop_k));
    b.end_loop();

    let acc_o = b.add_i32(acc, outer);
    let acc_t = b.add_i32(acc_o, tid);
    b.set_var(acc, acc_t);
    let outer_next = b.add_i32(outer, c1);
    b.set_var(outer, outer_next);
    let more_outer = b.cmp_lt_i32(outer_next, c2);
    let stop_outer = b.let_(Op::NotI1(Opnd::Val(more_outer)));
    b.break_if(Opnd::Val(stop_outer));
    b.end_loop();

    let out_off = b.mul_i32(tid, c4);
    let out_ptr = b.ptr_add(o, out_off);
    let value = b.bitcast_f32(acc);
    b.store_f32(out_ptr, 0, Opnd::Val(value));
    b.finish()
}
/// FMHA accumulation loop: a Frag16 register accumulator is carried across loop
/// iterations and updated by `v_mfma_scale_f32_32x32x64_f8f6f4` each iteration
/// (the accumulator is a loop variable of fragment type). Two iterations with
/// all-ones E4M3 data at K = 64 must produce 128.0 in every element.
pub fn probe_loop_mfma() -> Function {
    let mut b = FnBuilder::new("probe_loop_mfma", 512);
    b.lds(16384);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let ones = b.c_i32(0x38383838_u32 as i32);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let c8 = b.c_i32(8);
    let c8k = b.c_i32(8192);
    let c16 = b.c_i32(16);
    let c0 = b.c_i32(0);
    let frag = b.frag4(ones, ones, ones, ones);
    let zero = b.frag_zero(DType::Frag16);

    let off = b.mul_i32(tid, c16);
    let boff = b.add_i32(c8k, off);

    let vars = b.begin_loop(&[(DType::Frag16, Opnd::Val(zero)), (DType::I32, Opnd::Val(c0))]);
    let acc = vars[0];
    let it = vars[1];

    b.store_lds_b128(Opnd::Val(off), Opnd::Val(frag));
    b.store_lds_b128(Opnd::Val(boff), Opnd::Val(frag));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let a_lo = b.load_lds_b128(Opnd::Val(off));
    let b_lo = b.load_lds_b128(Opnd::Val(boff));
    let a8 = b.cat_frag8(a_lo, a_lo);
    let b8 = b.cat_frag8(b_lo, b_lo);
    let d = b.mfma_scale_fp8(a8, b8, acc, 127, 127);
    // Diagnostic: store D[0] of each iteration at o[4096 + it*4].
    let diag = b.frag16_lane(d, 0);
    let diag_f = b.bitcast_f32(diag);
    let c4096 = b.c_i32(4096);
    let diag_off = b.mul_i32(it, c4);
    let diag_off_full = b.add_i32(diag_off, c4096);
    let diag_ptr = b.ptr_add(o, diag_off_full);
    b.store_f32(diag_ptr, 0, diag_f);
    b.set_var(acc, d);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // Diagnostic: final accumulator lane 0 at o[5120].
    let c5120 = b.c_i32(5120);
    let final_ptr = b.ptr_add(o, c5120);
    let acc0 = b.frag16_lane(acc, 0);
    let acc0_f = b.bitcast_f32(acc0);
    b.store_f32(final_ptr, 0, acc0_f);

    // Per-lane final accumulator: two dwords each, 8 bytes per lane.
    let row_off = b.mul_i32(tid, c8);
    let row = b.ptr_add(o, row_off);
    for i in 0..2u8 {
        let bits = b.frag16_lane(acc, i);
        let value = b.bitcast_f32(bits);
        b.store_f32(row, 4 * i as u32, value);
    }
    b.finish()
}
/// FMHA core loop shape: A and B tiles are staged in separate phases with
/// paired barriers, read back with `ds_read_b128`, concatenated into `Frag8`
/// operands and accumulated into a `Frag16` register accumulator by the
/// block-scaled fp8 MFMA. Two iterations of K = 64 with all-ones data must give
/// 128.0 in every element.
pub fn probe_fmha_core() -> Function {
    let mut b = FnBuilder::new("probe_fmha_core", 512);
    b.lds(16384);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let ones = b.c_i32(0x38383838_u32 as i32);
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c8 = b.c_i32(8);
    let c16 = b.c_i32(16);
    let c8k = b.c_i32(8192);
    let frag = b.frag4(ones, ones, ones, ones);
    let zero = b.frag_zero(DType::Frag16);

    let off = b.mul_i32(tid, c16);
    let boff = b.add_i32(c8k, off);

    let vars = b.begin_loop(&[(DType::Frag16, Opnd::Val(zero)), (DType::I32, Opnd::Val(c0))]);
    let acc = vars[0];
    let it = vars[1];

    // A phase.
    b.store_lds_b128(Opnd::Val(off), Opnd::Val(frag));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    // B phase.
    b.store_lds_b128(Opnd::Val(boff), Opnd::Val(frag));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    // Compute.
    b.wait_lds(0);
    let a_lo = b.load_lds_b128(Opnd::Val(off));
    let b_lo = b.load_lds_b128(Opnd::Val(boff));
    let a8 = b.cat_frag8(a_lo, a_lo);
    let b8 = b.cat_frag8(b_lo, b_lo);
    let d = b.mfma_scale_fp8(a8, b8, acc, 127, 127);
    b.set_var(acc, d);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    let row_off = b.mul_i32(tid, c8);
    let row = b.ptr_add(o, row_off);
    for i in 0..2u8 {
        let bits = b.frag16_lane(acc, i);
        let value = b.bitcast_f32(bits);
        b.store_f32(row, 4 * i as u32, value);
    }
    b.finish()
}
/// Global -> LDS DMA probe (64 threads): `global_load_lds_dwordx4` of 1 KB and `global_load_lds_dword` of 256 B,
/// read back through ordinary LDS reads. Lane i must land at byte 16 i (4 i) of its wave's M0 base.
///
/// * `o[0..256]`    (dwords): the 1 KB copied with the 16-byte form, read back with `ds_read_b128`
/// * `o[256..320]`  (dwords): the 256 B copied with the 4-byte form, read back with `ds_read_b32`
pub fn probe_dma() -> Function {
    let mut b = FnBuilder::new("probe_dma", 64);
    b.lds(70000);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("i", DType::Ptr, 8, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let inp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let c16 = b.c_i32(16);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(68096); // above 64 KB
    let c0 = b.c_i32(0);
    let c36864 = b.c_i32(36864);

    let off16 = b.mul_i32(tid, c16);
    let p16 = b.ptr_add(inp, off16);
    b.load_global_lds(p16, c36864, 0, 4);
    let off4 = b.mul_i32(tid, c4);
    let src4 = b.add_i32(off4, c1024);
    let p4 = b.ptr_add(inp, src4);
    b.load_global_lds(p4, c2048, 0, 1);
    b.wait_vm(0);
    b.wait_lds(0);
    b.barrier();

    let ra = b.add_i32(off16, c36864);
    let v = b.load_lds_b128(Opnd::Val(ra));
    for k in 0..4u8 {
        let w = b.frag4_lane(v, k);
        let f = b.bitcast_f32(w);
        let op = b.ptr_add(o, off16);
        b.store_f32(op, 4 * k as u32, f);
    }
    let a4 = b.add_i32(off4, c2048);
    let w4 = b.load_lds_b32(Opnd::Val(a4));
    let f4 = b.bitcast_f32(w4);
    let ob = b.add_i32(off4, c1024);
    let op4 = b.ptr_add(o, ob);
    b.store_f32(op4, 0, f4);
    b.finish()
}
/// Multi-wave DMA tile probe (256 threads): four waves each issue four 1 KB `global_load_lds_dwordx4` (run
/// r = wave + 4 q at LDS byte 1024 r, rows of 128 bytes), then every thread reads its 4 x 16 bytes back.
/// `o` must equal `i` (16 KB).
pub fn probe_dma_tile() -> Function {
    let mut b = FnBuilder::new("probe_dma_tile", 256);
    b.lds(std::env::var("PD_LDS").ok().and_then(|v| v.parse().ok()).unwrap_or(16384));
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("i", DType::Ptr, 8, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let inp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c3 = b.c_i32(3);
    let c6 = b.c_i32(6);
    let c7 = b.c_i32(7);
    let c10 = b.c_i32(10);
    let c63 = b.c_i32(63);
    let c16 = b.c_i32(16);
    let c128 = b.c_i32(128);
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let lane_hi = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::Val(c3)));
    let lane_lo = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::Val(c7)));
    let w8 = b.shl_i32(wave, c3);
    let rowb = b.add_i32(lane_hi, w8);
    let cbytes = b.mul_i32(lane_lo, c16);
    let w1024 = b.shl_i32(wave, c10);
    let order: Vec<i32> = if std::env::var("PD_REV").is_ok() { vec![3, 2, 1, 0] } else { vec![0, 1, 2, 3] };
    for q in order {
        let cq = b.c_i32(32 * q);
        let row = b.add_i32(rowb, cq);
        let r128 = b.mul_i32(row, c128);
        let off = b.add_i32(r128, cbytes);
        let sp = b.ptr_add(inp, off);
        let pd_base: i32 = std::env::var("PD_BASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let pd_qs: i32 = std::env::var("PD_QS").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
        let cl = b.c_i32(pd_base + pd_qs * q);
        let l = b.add_i32(w1024, cl);
        b.load_global_lds(sp, l, 0, 4);
        if let Some(w) = std::env::var("PD_WAIT").ok().and_then(|v| v.parse::<u8>().ok()) {
            b.wait_vm(w);
        }
    }
    if std::env::var("PD_DUMMY").is_ok() {
        // a harmless trailing DMA into spare LDS (needs PD_LDS >= 18432)
        let spare = b.c_i32(17408);
        let r0 = b.mul_i32(lane, c16);
        let sp = b.ptr_add(inp, r0);
        b.load_global_lds(sp, spare, 0, 4);
    }
    b.wait_vm(0);
    b.wait_lds(0);
    b.barrier();
    let off16 = b.mul_i32(tid, c16);
    for k in 0..4i32 {
        let pd_base: i32 = std::env::var("PD_BASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let pd_qs: i32 = std::env::var("PD_QS").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
        let ck = b.c_i32(pd_base + pd_qs * k);
        let a = b.add_i32(off16, ck);
        let v = b.load_lds_b128(Opnd::Val(a));
        let op = b.ptr_add(o, a);
        for w in 0..4u8 {
            let x = b.frag4_lane(v, w);
            let f = b.bitcast_f32(x);
            b.store_f32(op, 4 * w as u32, f);
        }
    }
    b.finish()
}
/// Single-wave DMA sequence probe (64 threads): sixteen `global_load_lds_dwordx4`, DMA q landing at LDS byte 1024 q,
/// then read back linearly. `o` must equal `i` (16 KB).
pub fn probe_dma_seq() -> Function {
    let stride: i32 = std::env::var("PD_STRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let mut b = FnBuilder::new("probe_dma_seq", 64);
    b.lds(20000 + 16 * 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("i", DType::Ptr, 8, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let inp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c16 = b.c_i32(16);
    let off16 = b.mul_i32(tid, c16);
    for q in 0..16i32 {
        let cq = b.c_i32(1024 * q);
        let cl = b.c_i32(stride * q);
        let src = b.add_i32(off16, cq);
        let sp = b.ptr_add(inp, src);
        b.load_global_lds(sp, cl, 0, 4);
    }
    b.wait_vm(0);
    b.wait_lds(0);
    b.barrier();
    for q in 0..16i32 {
        let cl = b.c_i32(stride * q);
        let cq = b.c_i32(1024 * q);
        let a = b.add_i32(off16, cl);
        let v = b.load_lds_b128(Opnd::Val(a));
        let od = b.add_i32(off16, cq);
        let op = b.ptr_add(o, od);
        for w in 0..4u8 {
            let x = b.frag4_lane(v, w);
            let f = b.bitcast_f32(x);
            b.store_f32(op, 4 * w as u32, f);
        }
    }
    b.finish()
}
/// DMA + K-fragment read probe (256 threads): the tile of `probe_dma_tile` is staged with the (optionally
/// XOR-swizzled, PD_SWZ=1) DMA layout of `fmha_fwd_hd128_mxfp8_sv`, then thread t (j = t & 31, g = (t >> 5) & 1)
/// reads the four 16-byte K fragment chunks `g ^ 2 k` of row `j` (+ 32 rows per wave group w = t >> 6) the way the
/// kernel does and stores them. `o[(t * 4 + k) * 16 ..]` must equal `i[(j + 32 (t >> 6)) * 128 + 16 (g ^ 2 k) ..]`.
pub fn probe_dma_kread() -> Function {
    // PD_SWZ: 0/empty = row-major, 1 = 3-bit XOR swizzle, 2 = 2-bit XOR swizzle
    let swz_mode: u32 = std::env::var("PD_SWZ").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let swz = swz_mode != 0 && swz_mode != 4;
    let fmask: i32 = if swz_mode == 2 { 3 } else { 7 };
    let mut b = FnBuilder::new("probe_dma_kread", 256);
    b.lds(20000);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("i", DType::Ptr, 8, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let inp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c1 = b.c_i32(1);
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c6 = b.c_i32(6);
    let c7 = b.c_i32(7);
    let cfm = b.c_i32(fmask);
    let c10 = b.c_i32(10);
    let c31 = b.c_i32(31);
    let c63 = b.c_i32(63);
    let c128 = b.c_i32(128);
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let lane_hi = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::Val(c3)));
    let lane_lo = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::Val(c7)));
    let w_odd = b.let_(Op::AndI32(Opnd::Val(wave), Opnd::Val(c1)));
    let c2 = b.c_i32(2);
    let w4 = b.shl_i32(w_odd, c2);
    let lane_h4 = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::Val(c4)));
    let fs0 = b.add_i32(w4, lane_h4);
    let fsw_raw = b.let_(Op::AndI32(Opnd::Val(fs0), Opnd::Val(cfm)));
    let zero = b.c_i32(0);
    let fsw = if swz { fsw_raw } else { zero };
    let cch = b.xor_i32(lane_lo, fsw);
    let cbytes = b.shl_i32(cch, c4);
    let w8 = b.shl_i32(wave, c3);
    let rowb = b.add_i32(lane_hi, w8);
    let w1024 = b.shl_i32(wave, c10);
    if std::env::var("PD_NODMA").is_ok() {
        // reference staging: plain loads + ds_write, row-major (no swizzle)
        let c64s = b.c_i32(64);
        let t64s = b.mul_i32(tid, c64s);
        let sp = b.ptr_add(inp, t64s);
        for w in 0..4u32 {
            let d = b.load_b128(sp, 16 * w);
            let cw = b.c_i32(16 * w as i32);
            let la = b.add_i32(t64s, cw);
            b.store_lds_b128(Opnd::Val(la), Opnd::Val(d));
        }
    } else {
    if swz_mode == 4 {
        // vendor-style runs: run rho = wave + 4 q holds rows (rho & 7) + 8 s + 64 (rho >> 3) at LDS 1040 rho
        let cb4 = b.shl_i32(lane_lo, c4);
        let s8 = b.shl_i32(lane_hi, c3);
        let rb4 = b.add_i32(s8, wave);
        let c1040 = b.c_i32(1040);
        let w1040 = b.mul_i32(wave, c1040);
        for q in 0..4i32 {
            let cq = b.c_i32(4 * (q & 1) + 64 * (q >> 1));
            let row = b.add_i32(rb4, cq);
            let r128 = b.mul_i32(row, c128);
            let off = b.add_i32(r128, cb4);
            let sp = b.ptr_add(inp, off);
            let cl = b.c_i32(4160 * q);
            let l = b.add_i32(w1040, cl);
            b.load_global_lds(sp, l, 0, 4);
        }
    } else {
    for q in 0..4i32 {
        let cq = b.c_i32(32 * q);
        let row = b.add_i32(rowb, cq);
        let r128 = b.mul_i32(row, c128);
        let off = b.add_i32(r128, cbytes);
        let sp = b.ptr_add(inp, off);
        let cl = b.c_i32(4096 * q);
        let l = b.add_i32(w1024, cl);
        b.load_global_lds(sp, l, 0, 4);
    }
    }
    // trailing DMA (see emit_fmha_sv_role)
    {
        let spare = b.c_i32(17408);
        let c16 = b.c_i32(16);
        let r0 = b.mul_i32(lane, c16);
        let sp2 = b.ptr_add(inp, r0);
        b.load_global_lds(sp2, spare, 0, 4);
    }
    }
    b.wait_vm(0);
    b.wait_lds(0);
    if let Some(n) = std::env::var("PD_NOPS").ok().and_then(|v| v.parse::<u32>().ok()) {
        for _ in 0..n {
            b.set_prio(0);
        }
    }
    b.barrier();
    if std::env::var("PD_B2").is_ok() {
        b.wait_vm(0);
        b.barrier();
    }
    // K-fragment style reads
    let t5 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let g = b.let_(Op::AndI32(Opnd::Val(t5), Opnd::Val(c1)));
    let jj = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let wsel = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let w32 = b.shl_i32(wsel, c5);
    let rowk = b.add_i32(jj, w32);
    let jrow = b.mul_i32(rowk, c128);
    let js1 = b.let_(Op::LshrI32(Opnd::Val(rowk), Opnd::Val(c1)));
    let f0 = b.let_(Op::AndI32(Opnd::Val(js1), Opnd::Val(cfm)));
    // PD_SWZ=3: DMA with the 3-bit swizzle but reads WITHOUT it (tests whether the hardware ignores the lane permutation)
    let fk = if swz && swz_mode != 3 { f0 } else { zero };
    let x = b.xor_i32(g, fk);
    let t64 = b.shl_i32(tid, c6);
    // mode 4 base: ((R & 7) + 8 (R >> 6)) 1040 + ((R >> 3) & 7) 128 with R = j + 32 wave
    let r_u = b.let_(Op::AndI32(Opnd::Val(rowk), Opnd::Val(c7)));
    let r_h = b.let_(Op::LshrI32(Opnd::Val(rowk), Opnd::Val(c6)));
    let r_h8 = b.shl_i32(r_h, c3);
    let r_uh = b.add_i32(r_u, r_h8);
    let c1040r = b.c_i32(1040);
    let r_a = b.mul_i32(r_uh, c1040r);
    let r_s0 = b.let_(Op::LshrI32(Opnd::Val(rowk), Opnd::Val(c3)));
    let r_s = b.let_(Op::AndI32(Opnd::Val(r_s0), Opnd::Val(c7)));
    let r_s128 = b.mul_i32(r_s, c128);
    let base4 = b.add_i32(r_a, r_s128);
    for k in 0..4i32 {
        let c2k = b.c_i32(2 * k);
        let xk = b.xor_i32(x, c2k);
        let xs = b.shl_i32(xk, c4);
        let a = if swz_mode == 4 { b.add_i32(base4, xs) } else { b.add_i32(jrow, xs) };
        let v = b.load_lds_b128(Opnd::Val(a));
        let ob = b.c_i32(16 * k);
        let o1 = b.add_i32(t64, ob);
        let op = b.ptr_add(o, o1);
        for w in 0..4u8 {
            let xw = b.frag4_lane(v, w);
            let f = b.bitcast_f32(xw);
            b.store_f32(op, 4 * w as u32, f);
        }
    }
    b.finish()
}
/// DMA address-pattern probe (64 threads, one `global_load_lds_dwordx4` to M0 = 0): lane i fetches the 16 bytes at a
/// pattern-dependent source offset (PD_MAP: 0 identity, 1 reversed, 2 stride 1024, 3 stride 128, 4 rows 1024 apart x
/// 8 chunks) and the 1 KB of LDS is read back linearly: `o` slot u holds the data some lane fetched.
pub fn probe_dma_map() -> Function {
    let mode: u32 = std::env::var("PD_MAP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut b = FnBuilder::new("probe_dma_map", 64);
    b.lds(4096);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("i", DType::Ptr, 8, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let inp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c16 = b.c_i32(16);
    let c3 = b.c_i32(3);
    let c7 = b.c_i32(7);
    let c0 = b.c_i32(0);
    let off = match mode {
        0 => b.mul_i32(tid, c16),
        1 => {
            let c63 = b.c_i32(63);
            let r = b.xor_i32(tid, c63);
            b.mul_i32(r, c16)
        }
        2 => {
            let c1024 = b.c_i32(1024);
            b.mul_i32(tid, c1024)
        }
        3 => {
            let c128 = b.c_i32(128);
            b.mul_i32(tid, c128)
        }
        _ => {
            let hi = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c3)));
            let lo = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c7)));
            let c1024 = b.c_i32(1024);
            let a = b.mul_i32(hi, c1024);
            let bb = b.mul_i32(lo, c16);
            b.add_i32(a, bb)
        }
    };
    let sp = b.ptr_add(inp, off);
    b.load_global_lds(sp, c0, 0, 4);
    b.wait_vm(0);
    b.wait_lds(0);
    b.barrier();
    let o16 = b.mul_i32(tid, c16);
    let v = b.load_lds_b128(Opnd::Val(o16));
    let op = b.ptr_add(o, o16);
    for w in 0..4u8 {
        let x = b.frag4_lane(v, w);
        let f = b.bitcast_f32(x);
        b.store_f32(op, 4 * w as u32, f);
    }
    b.finish()
}
/// Pack instruction probe: checks the semantics of `v_cvt_pknorm_u16_f32` and
/// of the fp8 packing path.
///
/// * `o[tid]`      = pknorm(1.0, 0.5)  -> low 0xFFFF, high 0x8000
/// * `o[512+tid]`  = fp8 pack, low half of 0xAAAAAAAA replaced
/// * `o[1024+tid]` = fp8 pack, high half of 0xAAAAAAAA replaced
/// * `o[1536+tid]` = the 0xAAAAAAAA constant itself (control)
///
/// The bf16 inputs are two 1.0 values (0x3F80) and the scale is 1.0, so the
/// packed fp8 bytes are both E4M3 1.0 (0x38).
pub fn probe_pack() -> Function {
    let mut b = FnBuilder::new("probe_pack", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c4 = b.c_i32(4);
    let c512 = b.c_i32(512);
    let c1024 = b.c_i32(1024);
    let c1536 = b.c_i32(1536);

    let old = b.c_i32(0xAAAA_AAAA_u32 as i32);
    let src = b.c_i32(0x3F80_3F80_u32 as i32);
    let scale = b.c_f32(1.0);
    let one = b.c_f32(1.0);
    let half = b.c_f32(0.5);

    let pn = b.cvt_pknorm_u16_f32(one, half);
    let lo = b.cvt_scalef32_pk_fp8_bf16(old, src, scale, false);
    let hi = b.cvt_scalef32_pk_fp8_bf16(old, src, scale, true);

    let off = b.mul_i32(tid, c4);
    let p0 = b.ptr_add(o, off);
    let v0 = b.bitcast_f32(pn);
    b.store_f32(p0, 0, v0);

    let idx1 = b.add_i32(tid, c512);
    let off1 = b.mul_i32(idx1, c4);
    let p1 = b.ptr_add(o, off1);
    let v1 = b.bitcast_f32(lo);
    b.store_f32(p1, 0, v1);

    let idx2 = b.add_i32(tid, c1024);
    let off2 = b.mul_i32(idx2, c4);
    let p2 = b.ptr_add(o, off2);
    let v2 = b.bitcast_f32(hi);
    b.store_f32(p2, 0, v2);

    let idx3 = b.add_i32(tid, c1536);
    let off3 = b.mul_i32(idx3, c4);
    let p3 = b.ptr_add(o, off3);
    let v3 = b.bitcast_f32(old);
    b.store_f32(p3, 0, v3);

    b.finish()
}
/// Real-data block-scaled fp8 matmul probe.
///
/// The input buffer `o` holds, at fixed byte offsets:
///
/// * `0`    : matrix A, 32 x 64 fp8, row major (2048 bytes)
/// * `2048` : matrix B transposed, 32 x 64 fp8 (B^T[j][k] = B[k][j])
/// * `4096` : A scale bytes, one dword per lane (only the low byte is read)
/// * `4352` : B scale bytes, one dword per lane
/// * `4608` : output, 32 x 32 f32, row major (4096 bytes)
///
/// The MFMA fragment layout comes from the CDNA4 ISA (7.1.3):
///
/// * `A[i][k]`  lives in lane `i + 32 * (k / 32)` at byte `k % 32` of the lane's
///   8-dword fragment,
/// * `B[k][j]`  lives in lane `j + 32 * (k / 32)` at byte `k % 32`,
/// * the scale byte of lane `l` for matrix A applies to row `l % 32` and K block
///   `l / 32` (same for B with row replaced by the B row index),
/// * `D[i][j]`  lives in lane `j + 32 * ((i / 4) % 2)` at register
///   `4 * (i / 8) + (i % 4)`.
pub fn probe_fp8_matmul() -> Function {
    let mut b = FnBuilder::new("probe_fp8_matmul", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c31 = b.c_i32(31);
    let c64 = b.c_i32(64);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c4352 = b.c_i32(4352);
    let c4608 = b.c_i32(4608);

    // lane decomposition: row index j (lane % 32) and K block (lane / 32)
    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let p = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));

    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(p, c5);
    let frag_off = b.add_i32(row, koff);
    let a_base = b.ptr_add(o, frag_off);
    let bt_off = b.add_i32(frag_off, c2048);
    let bt_base = b.ptr_add(o, bt_off);

    let a0 = b.load_f32(a_base, 0);
    let a0 = b.bitcast_i32(a0);
    let a1 = b.load_f32(a_base, 4);
    let a1 = b.bitcast_i32(a1);
    let a2 = b.load_f32(a_base, 8);
    let a2 = b.bitcast_i32(a2);
    let a3 = b.load_f32(a_base, 12);
    let a3 = b.bitcast_i32(a3);
    let a4 = b.load_f32(a_base, 16);
    let a4 = b.bitcast_i32(a4);
    let a5 = b.load_f32(a_base, 20);
    let a5 = b.bitcast_i32(a5);
    let a6 = b.load_f32(a_base, 24);
    let a6 = b.bitcast_i32(a6);
    let a7 = b.load_f32(a_base, 28);
    let a7 = b.bitcast_i32(a7);
    let a8 = b.frag8(a0, a1, a2, a3, a4, a5, a6, a7);

    let b0 = b.load_f32(bt_base, 0);
    let b0 = b.bitcast_i32(b0);
    let b1 = b.load_f32(bt_base, 4);
    let b1 = b.bitcast_i32(b1);
    let b2 = b.load_f32(bt_base, 8);
    let b2 = b.bitcast_i32(b2);
    let b3 = b.load_f32(bt_base, 12);
    let b3 = b.bitcast_i32(b3);
    let b4 = b.load_f32(bt_base, 16);
    let b4 = b.bitcast_i32(b4);
    let b5 = b.load_f32(bt_base, 20);
    let b5 = b.bitcast_i32(b5);
    let b6 = b.load_f32(bt_base, 24);
    let b6 = b.bitcast_i32(b6);
    let b7 = b.load_f32(bt_base, 28);
    let b7 = b.bitcast_i32(b7);
    let b8v = b.frag8(b0, b1, b2, b3, b4, b5, b6, b7);

    let tid4 = b.mul_i32(tid, c4);
    let sa_off = b.add_i32(tid4, c4096);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(tid4, c4352);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);

    let zero = b.frag_zero(DType::Frag16);
    let d = b.mfma_scale_fp8(a8, b8v, zero, sa, sb);

    // Diagnostics: raw loaded words of lane `tid`.
    let c4b = b.c_i32(4);
    let c8704 = b.c_i32(8704);
    let c8960 = b.c_i32(8960);
    let c9216 = b.c_i32(9216);
    let diag = b.mul_i32(tid, c4b);
    let d8704 = b.add_i32(diag, c8704);
    let pa = b.ptr_add(o, d8704);
    let va = b.bitcast_f32(a0);
    b.store_f32(pa, 0, va);
    let d8960 = b.add_i32(diag, c8960);
    let pb = b.ptr_add(o, d8960);
    let vb = b.bitcast_f32(b0);
    b.store_f32(pb, 0, vb);
    let d9216 = b.add_i32(diag, c9216);
    let ps = b.ptr_add(o, d9216);
    let vs = b.bitcast_f32(sa);
    b.store_f32(ps, 0, vs);

    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(p, c9);
    let lane_off = b.add_i32(j4, p512);
    let dyn_off = b.add_i32(lane_off, c4608);
    let dptr = b.ptr_add(o, dyn_off);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as u32;
        let lane = b.frag16_lane(d, r);
        let v = b.bitcast_f32(lane);
        b.store_f32(dptr, i_const * 128, v);
    }
    b.finish()
}
/// Load-path bisect for the fp8 matmul probe: reads the same A / B^T / scale
/// words but stores them raw, so the host can check the address arithmetic and
/// the fragment mapping independently of the MFMA.
///
/// `o[1152 + tid*8 + d]`  = A word `d` of lane `tid`
/// `o[1664 + tid*8 + d]`  = B^T word `d` of lane `tid`
/// `o[2176 + tid]`        = the A scale word of lane `tid`
pub fn probe_fp8_load() -> Function {
    let mut b = FnBuilder::new("probe_fp8_load", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c31 = b.c_i32(31);
    let c64 = b.c_i32(64);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c4608 = b.c_i32(4608);
    let c6656 = b.c_i32(6656);
    let c8704 = b.c_i32(8704);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let p = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(p, c5);
    let frag_off = b.add_i32(row, koff);
    let a_base = b.ptr_add(o, frag_off);
    let bt_off = b.add_i32(frag_off, c2048);
    let bt_base = b.ptr_add(o, bt_off);

    let tid32 = b.shl_i32(tid, c5);
    let a_out = b.add_i32(tid32, c4608);
    let a_ptr = b.ptr_add(o, a_out);
    for d in 0..8u32 {
        let v = b.load_f32(a_base, 4 * d);
        b.store_f32(a_ptr, 4 * d, v);
    }
    let b_out = b.add_i32(tid32, c6656);
    let b_ptr = b.ptr_add(o, b_out);
    for d in 0..8u32 {
        let v = b.load_f32(bt_base, 4 * d);
        b.store_f32(b_ptr, 4 * d, v);
    }

    let tid4 = b.mul_i32(tid, c4);
    let sa_off = b.add_i32(tid4, c4096);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa = b.load_f32(sa_ptr, 0);
    let sa_out = b.add_i32(tid4, c8704);
    let sa_o = b.ptr_add(o, sa_out);
    b.store_f32(sa_o, 0, sa);

    b.finish()
}
/// Minimal block-scaled fp8 MFMA bisect, stage 1: constant `Frag8` operands
/// (every fp8 byte is E4M3 1.0), scale 127, so every `D` element must be 64.0.
///
/// `o[tid * 16 + r]` = D register `r` of lane `tid`.
pub fn probe_fp8_min() -> Function {
    let mut b = FnBuilder::new("probe_fp8_min", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let ones = b.c_i32(0x38383838_u32 as i32);
    let sc = b.c_i32(127);
    let a8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let b8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let zero = b.frag_zero(DType::Frag16);
    let d = b.mfma_scale_fp8(a8, b8, zero, sc, sc);

    let c64 = b.c_i32(64);
    let base = b.mul_i32(tid, c64);
    let p = b.ptr_add(o, base);
    for r in 0..16u8 {
        let lane = b.frag16_lane(d, r);
        let v = b.bitcast_f32(lane);
        b.store_f32(p, 4 * r as u32, v);
    }
    b.finish()
}

/// Minimal block-scaled fp8 MFMA bisect, stage 2: the same MFMA but with A / B^T
/// and the scale bytes loaded from global memory, stored back through the same
/// simple per-lane addressing as stage 1 (no layout math on the store side).
pub fn probe_fp8_minload() -> Function {
    let mut b = FnBuilder::new("probe_fp8_minload", 64);
    b.lds(256);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c31 = b.c_i32(31);
    let c64 = b.c_i32(64);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c4352 = b.c_i32(4352);
    let c4608 = b.c_i32(4608);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let a_base = b.ptr_add(o, frag_off);
    let bt_off = b.add_i32(frag_off, c2048);
    let bt_base = b.ptr_add(o, bt_off);

    let a0 = b.load_f32(a_base, 0);
    let a0 = b.bitcast_i32(a0);
    let a1 = b.load_f32(a_base, 4);
    let a1 = b.bitcast_i32(a1);
    let a2 = b.load_f32(a_base, 8);
    let a2 = b.bitcast_i32(a2);
    let a3 = b.load_f32(a_base, 12);
    let a3 = b.bitcast_i32(a3);
    let a4 = b.load_f32(a_base, 16);
    let a4 = b.bitcast_i32(a4);
    let a5 = b.load_f32(a_base, 20);
    let a5 = b.bitcast_i32(a5);
    let a6 = b.load_f32(a_base, 24);
    let a6 = b.bitcast_i32(a6);
    let a7 = b.load_f32(a_base, 28);
    let a7 = b.bitcast_i32(a7);
    let a8 = b.frag8(a0, a1, a2, a3, a4, a5, a6, a7);

    let b0 = b.load_f32(bt_base, 0);
    let b0 = b.bitcast_i32(b0);
    let b1 = b.load_f32(bt_base, 4);
    let b1 = b.bitcast_i32(b1);
    let b2 = b.load_f32(bt_base, 8);
    let b2 = b.bitcast_i32(b2);
    let b3 = b.load_f32(bt_base, 12);
    let b3 = b.bitcast_i32(b3);
    let b4 = b.load_f32(bt_base, 16);
    let b4 = b.bitcast_i32(b4);
    let b5 = b.load_f32(bt_base, 20);
    let b5 = b.bitcast_i32(b5);
    let b6 = b.load_f32(bt_base, 24);
    let b6 = b.bitcast_i32(b6);
    let b7 = b.load_f32(bt_base, 28);
    let b7 = b.bitcast_i32(b7);
    let b8 = b.frag8(b0, b1, b2, b3, b4, b5, b6, b7);

    let tid4 = b.mul_i32(tid, c4);
    let sa_off = b.add_i32(tid4, c4096);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(tid4, c4352);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);

    let zero = b.frag_zero(DType::Frag16);
    let d = b.mfma_scale_fp8(a8, b8, zero, sa, sb);

    let out_base = b.mul_i32(tid, c64);
    let out_off = b.add_i32(out_base, c4608);
    let dp = b.ptr_add(o, out_off);
    for r in 0..16u8 {
        let lane = b.frag16_lane(d, r);
        let v = b.bitcast_f32(lane);
        b.store_f32(dp, 4 * r as u32, v);
    }
    b.finish()
}
/// Real-data MXFP8 QK^T followed by a row-wise softmax.
///
/// The kernel runs the verified block-scaled fp8 MFMA, writes the 32x32 score
/// tile into LDS through the D fragment layout, then every lane reloads one full
/// row (lane `l` handles row `l & 31`), computes `exp(x - max) / sum(exp)` with
/// plain VALU ops and stores the normalised row to global memory.
///
/// `o[0..2048]`  : Q (32x64 fp8), `o[2048..4096]`: K^T (32x64 fp8),
/// `o[4096..4352]` / `o[4352..4608]`: per-lane E8M0 scales,
/// `o[4608..8704]` : softmax output, row major 32x32 f32.
pub fn probe_fp8_softmax() -> Function {
    let mut b = FnBuilder::new("probe_fp8_softmax", 64);
    b.lds(4096);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c31 = b.c_i32(31);
    let c64 = b.c_i32(64);
    let c128 = b.c_i32(128);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c4352 = b.c_i32(4352);
    let c4608 = b.c_i32(4608);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));

    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let a_base = b.ptr_add(o, frag_off);
    let bt_off = b.add_i32(frag_off, c2048);
    let bt_base = b.ptr_add(o, bt_off);

    let a0 = b.load_f32(a_base, 0);
    let a0 = b.bitcast_i32(a0);
    let a1 = b.load_f32(a_base, 4);
    let a1 = b.bitcast_i32(a1);
    let a2 = b.load_f32(a_base, 8);
    let a2 = b.bitcast_i32(a2);
    let a3 = b.load_f32(a_base, 12);
    let a3 = b.bitcast_i32(a3);
    let a4 = b.load_f32(a_base, 16);
    let a4 = b.bitcast_i32(a4);
    let a5 = b.load_f32(a_base, 20);
    let a5 = b.bitcast_i32(a5);
    let a6 = b.load_f32(a_base, 24);
    let a6 = b.bitcast_i32(a6);
    let a7 = b.load_f32(a_base, 28);
    let a7 = b.bitcast_i32(a7);
    let a8 = b.frag8(a0, a1, a2, a3, a4, a5, a6, a7);

    let b0 = b.load_f32(bt_base, 0);
    let b0 = b.bitcast_i32(b0);
    let b1 = b.load_f32(bt_base, 4);
    let b1 = b.bitcast_i32(b1);
    let b2 = b.load_f32(bt_base, 8);
    let b2 = b.bitcast_i32(b2);
    let b3 = b.load_f32(bt_base, 12);
    let b3 = b.bitcast_i32(b3);
    let b4 = b.load_f32(bt_base, 16);
    let b4 = b.bitcast_i32(b4);
    let b5 = b.load_f32(bt_base, 20);
    let b5 = b.bitcast_i32(b5);
    let b6 = b.load_f32(bt_base, 24);
    let b6 = b.bitcast_i32(b6);
    let b7 = b.load_f32(bt_base, 28);
    let b7 = b.bitcast_i32(b7);
    let b8v = b.frag8(b0, b1, b2, b3, b4, b5, b6, b7);

    let tid4 = b.mul_i32(tid, c4);
    let sa_off = b.add_i32(tid4, c4096);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(tid4, c4352);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);

    let zero = b.frag_zero(DType::Frag16);
    let d = b.mfma_scale_fp8(a8, b8v, zero, sa, sb);

    // Store the score tile into LDS with the D fragment layout.
    let j4 = b.mul_i32(j, c4);
    let c9 = b.c_i32(9);
    let p512 = b.shl_i32(pblk, c9);
    let lds_base = b.add_i32(j4, p512);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(d, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_base, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // Row softmax: lane l handles row (tid & 31); lanes 32..63 duplicate it.
    let rbase = b.mul_i32(j, c128);
    let mut vals: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32u32 {
        let coff = b.c_i32(4 * k as i32);
        let addr = b.add_i32(rbase, coff);
        let raw = b.load_lds_b32(Opnd::Val(addr));
        vals.push(b.bitcast_f32(raw));
    }
    b.wait_lds(0);

    let mut vmax = vals[0];
    for k in 1..32usize {
        vmax = b.max_f32(vmax, vals[k]);
    }
    let mut exps: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let shifted = b.sub_f32(vals[k], vmax);
        exps.push(b.exp_f32(shifted));
    }
    let mut total = exps[0];
    for k in 1..32usize {
        total = b.add_f32(total, exps[k]);
    }
    let inv = b.rcp_f32(total);

    let out_base = b.add_i32(rbase, c4608);
    let out_ptr = b.ptr_add(o, out_base);
    for k in 0..32u32 {
        let scaled = b.mul_f32(exps[k as usize], inv);
        b.store_f32(out_ptr, 4 * k, scaled);
    }
    b.finish()
}
/// Single-tile MXFP8 attention: QK^T -> row softmax -> fp8 quantisation -> PV.
///
/// Input image in `o`:
/// * `0`    : Q (32x64 fp8), `2048`: K^T (32x64 fp8)
/// * `4096` / `4352`: per-lane E8M0 scales for Q and K
/// * `4608` : V^T padded to 32x64 (upper 32 bytes zero)
/// * `6656` : output O (32x32 f32, row major)
/// * `10752`: the quantised P bytes, one row per lane (32 bytes)
pub fn probe_fp8_fmha_tile() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_tile", 64);
    b.lds(4096);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c128 = b.c_i32(128);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c4352 = b.c_i32(4352);
    let c4608 = b.c_i32(4608);
    let c6656 = b.c_i32(6656);
    let c10752 = b.c_i32(10752);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));

    // ---- phase 1: QK^T ----
    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let a_base = b.ptr_add(o, frag_off);
    let kt_off = b.add_i32(frag_off, c2048);
    let kt_base = b.ptr_add(o, kt_off);
    let v_off = b.add_i32(frag_off, c4608);
    let v_base = b.ptr_add(o, v_off);

    let a0 = b.load_f32(a_base, 0);
    let a0 = b.bitcast_i32(a0);
    let a1 = b.load_f32(a_base, 4);
    let a1 = b.bitcast_i32(a1);
    let a2 = b.load_f32(a_base, 8);
    let a2 = b.bitcast_i32(a2);
    let a3 = b.load_f32(a_base, 12);
    let a3 = b.bitcast_i32(a3);
    let a4 = b.load_f32(a_base, 16);
    let a4 = b.bitcast_i32(a4);
    let a5 = b.load_f32(a_base, 20);
    let a5 = b.bitcast_i32(a5);
    let a6 = b.load_f32(a_base, 24);
    let a6 = b.bitcast_i32(a6);
    let a7 = b.load_f32(a_base, 28);
    let a7 = b.bitcast_i32(a7);
    let a8 = b.frag8(a0, a1, a2, a3, a4, a5, a6, a7);

    let k0 = b.load_f32(kt_base, 0);
    let k0 = b.bitcast_i32(k0);
    let k1 = b.load_f32(kt_base, 4);
    let k1 = b.bitcast_i32(k1);
    let k2 = b.load_f32(kt_base, 8);
    let k2 = b.bitcast_i32(k2);
    let k3 = b.load_f32(kt_base, 12);
    let k3 = b.bitcast_i32(k3);
    let k4 = b.load_f32(kt_base, 16);
    let k4 = b.bitcast_i32(k4);
    let k5 = b.load_f32(kt_base, 20);
    let k5 = b.bitcast_i32(k5);
    let k6 = b.load_f32(kt_base, 24);
    let k6 = b.bitcast_i32(k6);
    let k7 = b.load_f32(kt_base, 28);
    let k7 = b.bitcast_i32(k7);
    let k8 = b.frag8(k0, k1, k2, k3, k4, k5, k6, k7);

    let tid4 = b.mul_i32(tid, c4);
    let sa_off = b.add_i32(tid4, c4096);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(tid4, c4352);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);

    let zero = b.frag_zero(DType::Frag16);
    let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

    // ---- phase 2: store S to LDS, row softmax ----
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_base = b.add_i32(j4, p512);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(s_frag, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_base, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let rbase = b.mul_i32(j, c128);
    let mut vals: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32u32 {
        let coff = b.c_i32(4 * k as i32);
        let addr = b.add_i32(rbase, coff);
        let raw = b.load_lds_b32(Opnd::Val(addr));
        vals.push(b.bitcast_f32(raw));
    }
    b.wait_lds(0);

    let mut vmax = vals[0];
    for k in 1..32usize {
        vmax = b.max_f32(vmax, vals[k]);
    }
    let mut exps: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let shifted = b.sub_f32(vals[k], vmax);
        exps.push(b.exp_f32(shifted));
    }
    let mut total = exps[0];
    for k in 1..32usize {
        total = b.add_f32(total, exps[k]);
    }
    let inv = b.rcp_f32(total);

    // ---- phase 3: quantise the probabilities to fp8 (values scaled by 16) ----
    let c16f = b.c_f32(16.0);
    let mut quant: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let p = b.mul_f32(exps[k], inv);
        quant.push(b.mul_f32(p, c16f));
    }
    let mut pairs: Vec<ValueId> = Vec::with_capacity(16);
    for m in 0..16usize {
        pairs.push(b.cvt_pk_fp8_f32(quant[2 * m], quant[2 * m + 1]));
    }
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let hi = b.shl_i32(pairs[2 * m + 1], c16i);
        words.push(b.add_i32(pairs[2 * m], hi));
    }

    // Store the quantised bytes for the host reference.
    let prow = b.mul_i32(j, c32);
    let p_off = b.add_i32(prow, c10752);
    let p_ptr = b.ptr_add(o, p_off);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }

    // ---- phase 4: PV via the same block-scaled MFMA ----
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    let v0 = b.load_f32(v_base, 0);
    let v0 = b.bitcast_i32(v0);
    let v1 = b.load_f32(v_base, 4);
    let v1 = b.bitcast_i32(v1);
    let v2 = b.load_f32(v_base, 8);
    let v2 = b.bitcast_i32(v2);
    let v3 = b.load_f32(v_base, 12);
    let v3 = b.bitcast_i32(v3);
    let v4 = b.load_f32(v_base, 16);
    let v4 = b.bitcast_i32(v4);
    let v5 = b.load_f32(v_base, 20);
    let v5 = b.bitcast_i32(v5);
    let v6 = b.load_f32(v_base, 24);
    let v6 = b.bitcast_i32(v6);
    let v7 = b.load_f32(v_base, 28);
    let v7 = b.bitcast_i32(v7);
    let v8 = b.frag8(v0, v1, v2, v3, v4, v5, v6, v7);

    // P was scaled by 16 while quantising, so the B scale of 2^-4 folds the
    // normalisation into the MFMA itself.
    let sc = b.c_i32(127);
    let sc_quarter = b.c_i32(123);
    let zero2 = b.frag_zero(DType::Frag16);
    let o_frag = b.mfma_scale_fp8(p8, v8, zero2, sc, sc_quarter);

    let lane_off = b.add_i32(j4, p512);
    let out_dyn = b.add_i32(lane_off, c6656);
    let out_ptr = b.ptr_add(o, out_dyn);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as u32;
        let lane = b.frag16_lane(o_frag, r);
        let v = b.bitcast_f32(lane);
        b.store_f32(out_ptr, i_const * 128, v);
    }
    b.finish()
}

/// Two-tile MXFP8 attention: a loop over two key tiles, a 64-wide row softmax and
/// one PV over all 64 keys.
///
/// Input image in `o`:
/// * `0`: Q (32x64 fp8); `2048` / `4096`: K^T for tiles 0 and 1;
/// * `6144`: combined V^T (`VT[j*64 + t*32 + key] = V_t[key][j]`);
/// * `8192` / `8448`: Q scales per tile; `8704` / `8960`: K scales per tile;
/// * `9216` / `10240`: quantised P bytes per tile; `11264`: output O (32x32 f32).
pub fn probe_fp8_fmha_2tile() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_2tile", 64);
    b.lds(8448);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c6144 = b.c_i32(6144);
    let c8192 = b.c_i32(8192);
    let c8704 = b.c_i32(8704);
    let c9216 = b.c_i32(9216);
    let c11264 = b.c_i32(11264);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_dyn = b.add_i32(j4, p512);

    // Q fragment, loaded once.
    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let q_base = b.ptr_add(o, frag_off);
    let q0 = b.load_f32(q_base, 0);
    let q0 = b.bitcast_i32(q0);
    let q1 = b.load_f32(q_base, 4);
    let q1 = b.bitcast_i32(q1);
    let q2 = b.load_f32(q_base, 8);
    let q2 = b.bitcast_i32(q2);
    let q3 = b.load_f32(q_base, 12);
    let q3 = b.bitcast_i32(q3);
    let q4 = b.load_f32(q_base, 16);
    let q4 = b.bitcast_i32(q4);
    let q5 = b.load_f32(q_base, 20);
    let q5 = b.bitcast_i32(q5);
    let q6 = b.load_f32(q_base, 24);
    let q6 = b.bitcast_i32(q6);
    let q7 = b.load_f32(q_base, 28);
    let q7 = b.bitcast_i32(q7);
    let a8 = b.frag8(q0, q1, q2, q3, q4, q5, q6, q7);

    let tid4 = b.mul_i32(tid, c4);
    let zero = b.frag_zero(DType::Frag16);

    // Loop over the two key tiles, accumulating the score tiles in LDS.
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0))]);
    let tile = vars[0];
    let tile_off = b.mul_i32(tile, c2048);
    let kt_dyn0 = b.add_i32(frag_off, tile_off);
    let kt_dyn = b.add_i32(kt_dyn0, c2048);
    let kt_base = b.ptr_add(o, kt_dyn);
    let k0 = b.load_f32(kt_base, 0);
    let k0 = b.bitcast_i32(k0);
    let k1 = b.load_f32(kt_base, 4);
    let k1 = b.bitcast_i32(k1);
    let k2 = b.load_f32(kt_base, 8);
    let k2 = b.bitcast_i32(k2);
    let k3 = b.load_f32(kt_base, 12);
    let k3 = b.bitcast_i32(k3);
    let k4 = b.load_f32(kt_base, 16);
    let k4 = b.bitcast_i32(k4);
    let k5 = b.load_f32(kt_base, 20);
    let k5 = b.bitcast_i32(k5);
    let k6 = b.load_f32(kt_base, 24);
    let k6 = b.bitcast_i32(k6);
    let k7 = b.load_f32(kt_base, 28);
    let k7 = b.bitcast_i32(k7);
    let k8 = b.frag8(k0, k1, k2, k3, k4, k5, k6, k7);

    let sat0 = b.mul_i32(tile, c256);
    let sat1 = b.add_i32(sat0, tid4);
    let sa_off = b.add_i32(sat1, c8192);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sbt0 = b.mul_i32(tile, c256);
    let sbt1 = b.add_i32(sbt0, tid4);
    let sb_off = b.add_i32(sbt1, c8704);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);

    let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

    let tls = b.mul_i32(tile, c4096);
    let lds_base = b.add_i32(lds_dyn, tls);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(s_frag, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_base, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }

    let tile_next = b.add_i32(tile, c1);
    b.set_var(tile, tile_next);
    let more = b.cmp_lt_i32(tile_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // 64-wide row softmax.
    let rbase = b.mul_i32(j, c128);
    let mut vals: Vec<ValueId> = Vec::with_capacity(64);
    for tile_c in 0..2u32 {
        for k in 0..32u32 {
            let coff = b.c_i32((tile_c * 4096 + 4 * k) as i32);
            let addr = b.add_i32(rbase, coff);
            let raw = b.load_lds_b32(Opnd::Val(addr));
            vals.push(b.bitcast_f32(raw));
        }
    }
    b.wait_lds(0);

    let mut vmax = vals[0];
    for k in 1..64usize {
        vmax = b.max_f32(vmax, vals[k]);
    }
    let mut exps: Vec<ValueId> = Vec::with_capacity(64);
    for k in 0..64usize {
        let shifted = b.sub_f32(vals[k], vmax);
        exps.push(b.exp_f32(shifted));
    }
    let mut total = exps[0];
    for k in 1..64usize {
        total = b.add_f32(total, exps[k]);
    }
    let inv_l = b.rcp_f32(total);

    // Publish 1/l for the row so every lane can normalise its own elements.
    let inv_j = b.mul_i32(j, c4);
    let inv_addr = b.add_i32(inv_j, c8192);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(inv_addr), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // Quantise the lane group's 32 probabilities to fp8, scaled by 16 and 1/l.
    let c16f = b.c_f32(16.0);
    let scale16 = b.mul_f32(inv_l, c16f);
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        // low half: probs of tile 0, high half: tile 1
        let lo_a = b.mul_f32(exps[4 * m], scale16);
        let lo_b = b.mul_f32(exps[4 * m + 1], scale16);
        let pair_lo0 = b.cvt_pk_fp8_f32(lo_a, lo_b);
        let lo_c = b.mul_f32(exps[4 * m + 2], scale16);
        let lo_d = b.mul_f32(exps[4 * m + 3], scale16);
        let pair_lo1 = b.cvt_pk_fp8_f32(lo_c, lo_d);
        let hi_a = b.mul_f32(exps[32 + 4 * m], scale16);
        let hi_b = b.mul_f32(exps[32 + 4 * m + 1], scale16);
        let pair_hi0 = b.cvt_pk_fp8_f32(hi_a, hi_b);
        let hi_c = b.mul_f32(exps[32 + 4 * m + 2], scale16);
        let hi_d = b.mul_f32(exps[32 + 4 * m + 3], scale16);
        let pair_hi1 = b.cvt_pk_fp8_f32(hi_c, hi_d);
        let s_lo = b.shl_i32(pair_lo1, c16i);
        let w_lo = b.add_i32(pair_lo0, s_lo);
        let s_hi = b.shl_i32(pair_hi1, c16i);
        let w_hi = b.add_i32(pair_hi0, s_hi);
        let diff = b.xor_i32(w_lo, w_hi);
        let pick = b.mul_i32(diff, pblk);
        words.push(b.xor_i32(w_lo, pick));
    }

    // Store the quantised bytes for the host reference.
    let pb_off = b.mul_i32(pblk, c1024);
    let j32 = b.mul_i32(j, c32);
    let pb_off2 = b.add_i32(pb_off, j32);
    let pb_off3 = b.add_i32(pb_off2, c9216);
    let p_ptr = b.ptr_add(o, pb_off3);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }

    // PV over all 64 keys.
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    let vt_dyn = b.add_i32(frag_off, c6144);
    let vt_base = b.ptr_add(o, vt_dyn);
    let v0 = b.load_f32(vt_base, 0);
    let v0 = b.bitcast_i32(v0);
    let v1 = b.load_f32(vt_base, 4);
    let v1 = b.bitcast_i32(v1);
    let v2 = b.load_f32(vt_base, 8);
    let v2 = b.bitcast_i32(v2);
    let v3 = b.load_f32(vt_base, 12);
    let v3 = b.bitcast_i32(v3);
    let v4 = b.load_f32(vt_base, 16);
    let v4 = b.bitcast_i32(v4);
    let v5 = b.load_f32(vt_base, 20);
    let v5 = b.bitcast_i32(v5);
    let v6 = b.load_f32(vt_base, 24);
    let v6 = b.bitcast_i32(v6);
    let v7 = b.load_f32(vt_base, 28);
    let v7 = b.bitcast_i32(v7);
    let v8 = b.frag8(v0, v1, v2, v3, v4, v5, v6, v7);

    let zero2 = b.frag_zero(DType::Frag16);
    let o_frag = b.mfma_scale_fp8(p8, v8, zero2, c127, c123);

    // The probabilities already carry 1/l, so only the 1/16 quantisation factor
    // remains; it is folded into the B scale above.
    let o_out_dyn = b.add_i32(j4, p512);
    let o_out_off = b.add_i32(o_out_dyn, c11264);
    let o_ptr = b.ptr_add(o, o_out_off);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_frag, r);
        let lanev = b.bitcast_f32(lane);
        b.store_f32(o_ptr, (i_const * 128) as u32, lanev);
    }
    b.finish()
}
/// Online-softmax MXFP8 attention over `NT = 4` key tiles (128 keys).
///
/// Flash-style loop: running row max `m` and row sum `l` are loop-carried, the
/// O accumulator lives in LDS and is rescaled each iteration with the per-row
/// correction `exp(m_old - m_new)`, which every lane reads back from LDS for the
/// rows its fragment elements belong to.
///
/// Input image in `o` (S = 256 buffer): `0` Q; `2048 + 2048 t` K^T tile t;
/// `10240 + 2048 t` V^T tile t (32x64, upper 32 bytes zero); `18432 + 256 t` Q
/// scales; `19456 + 256 t` K scales; output O at `20480`; quantised P bytes of
/// tile t at `24576 + 1024 t`.
pub fn probe_fp8_fmha_online() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_online", 64);
    b.lds(8448);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c4t = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c10240 = b.c_i32(10240);
    let c18432 = b.c_i32(18432);
    let c19456 = b.c_i32(19456);
    let c20480 = b.c_i32(20480);
    let c24576 = b.c_i32(24576);
    let lds_corr = b.c_i32(4096);
    let lds_o = b.c_i32(4224);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_dyn = b.add_i32(j4, p512);
    let pblk16 = b.shl_i32(pblk, c4t);
    let corr_row = b.add_i32(j4, lds_corr);
    let o_lds_dyn = b.add_i32(lds_dyn, lds_o);

    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let tid4 = b.mul_i32(tid, c4);
    let zero = b.frag_zero(DType::Frag16);

    // Q fragment, loaded once.
    let q_base = b.ptr_add(o, frag_off);
    let q0 = b.load_f32(q_base, 0);
    let q0 = b.bitcast_i32(q0);
    let q1 = b.load_f32(q_base, 4);
    let q1 = b.bitcast_i32(q1);
    let q2 = b.load_f32(q_base, 8);
    let q2 = b.bitcast_i32(q2);
    let q3 = b.load_f32(q_base, 12);
    let q3 = b.bitcast_i32(q3);
    let q4 = b.load_f32(q_base, 16);
    let q4 = b.bitcast_i32(q4);
    let q5 = b.load_f32(q_base, 20);
    let q5 = b.bitcast_i32(q5);
    let q6 = b.load_f32(q_base, 24);
    let q6 = b.bitcast_i32(q6);
    let q7 = b.load_f32(q_base, 28);
    let q7 = b.bitcast_i32(q7);
    let a8 = b.frag8(q0, q1, q2, q3, q4, q5, q6, q7);

    // Zero the LDS O accumulator (each lane owns 16 distinct elements).
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(o_lds_dyn, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(c0));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
    ]);
    let tile = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];

    // ---- S_t = Q K_t^T ----
    let tile_off = b.mul_i32(tile, c2048);
    let kt0 = b.add_i32(frag_off, tile_off);
    let kt1 = b.add_i32(kt0, c2048);
    let kt_base = b.ptr_add(o, kt1);
    let k0 = b.load_f32(kt_base, 0);
    let k0 = b.bitcast_i32(k0);
    let k1 = b.load_f32(kt_base, 4);
    let k1 = b.bitcast_i32(k1);
    let k2 = b.load_f32(kt_base, 8);
    let k2 = b.bitcast_i32(k2);
    let k3 = b.load_f32(kt_base, 12);
    let k3 = b.bitcast_i32(k3);
    let k4 = b.load_f32(kt_base, 16);
    let k4 = b.bitcast_i32(k4);
    let k5 = b.load_f32(kt_base, 20);
    let k5 = b.bitcast_i32(k5);
    let k6 = b.load_f32(kt_base, 24);
    let k6 = b.bitcast_i32(k6);
    let k7 = b.load_f32(kt_base, 28);
    let k7 = b.bitcast_i32(k7);
    let k8 = b.frag8(k0, k1, k2, k3, k4, k5, k6, k7);

    let sc0 = b.mul_i32(tile, c256);
    let sc1 = b.add_i32(sc0, tid4);
    let sa_off = b.add_i32(sc1, c18432);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(sc1, c19456);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);
    let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(s_frag, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_dyn, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- online softmax update for this lane's row ----
    let rbase = b.mul_i32(j, c128);
    let mut vals: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32u32 {
        let coff = b.c_i32(4 * k as i32);
        let addr = b.add_i32(rbase, coff);
        let raw = b.load_lds_b32(Opnd::Val(addr));
        vals.push(b.bitcast_f32(raw));
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..32usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..32usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    // Publish the row correction for the lanes that own this row's O elements.
    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16) and PV for tile t ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let a = b.mul_f32(exps[4 * m], c16f);
        let bb = b.mul_f32(exps[4 * m + 1], c16f);
        let p0 = b.cvt_pk_fp8_f32(a, bb);
        let c = b.mul_f32(exps[4 * m + 2], c16f);
        let d = b.mul_f32(exps[4 * m + 3], c16f);
        let p1 = b.cvt_pk_fp8_f32(c, d);
        let sh = b.shl_i32(p1, c16i);
        words.push(b.add_i32(p0, sh));
    }
    let pt0 = b.mul_i32(tile, c1024);
    let pt1 = b.mul_i32(j, c32);
    let pt2 = b.add_i32(pt0, pt1);
    let pt3 = b.add_i32(pt2, c24576);
    let p_ptr = b.ptr_add(o, pt3);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    let vt0 = b.add_i32(frag_off, tile_off);
    let vt1 = b.add_i32(vt0, c10240);
    let vt_base = b.ptr_add(o, vt1);
    let v0 = b.load_f32(vt_base, 0);
    let v0 = b.bitcast_i32(v0);
    let v1 = b.load_f32(vt_base, 4);
    let v1 = b.bitcast_i32(v1);
    let v2 = b.load_f32(vt_base, 8);
    let v2 = b.bitcast_i32(v2);
    let v3 = b.load_f32(vt_base, 12);
    let v3 = b.bitcast_i32(v3);
    let v4 = b.load_f32(vt_base, 16);
    let v4 = b.bitcast_i32(v4);
    let v5 = b.load_f32(vt_base, 20);
    let v5 = b.bitcast_i32(v5);
    let v6 = b.load_f32(vt_base, 24);
    let v6 = b.bitcast_i32(v6);
    let v7 = b.load_f32(vt_base, 28);
    let v7 = b.bitcast_i32(v7);
    let v8 = b.frag8(v0, v1, v2, v3, v4, v5, v6, v7);
    let o_frag = b.mfma_scale_fp8(p8, v8, zero, c127, c123);

    // ---- O = O * corr[row] + PV, kept in LDS ----
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_frag, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let ooff = b.c_i32(i_const * 128);
        let oaddr = b.add_i32(o_lds_dyn, ooff);
        let oraw = b.load_lds_b32(Opnd::Val(oaddr));
        let ov = b.bitcast_f32(oraw);
        let scaled = b.mul_f32(ov, cf);
        let sum = b.add_f32(scaled, lanev);
        let sum_bits = b.bitcast_i32(sum);
        b.store_lds_b32(Opnd::Val(oaddr), Opnd::Val(sum_bits));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let tile_next = b.add_i32(tile, c1);
    b.set_var(tile, tile_next);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    let c4n = b.c_i32(4);
    let more = b.cmp_lt_i32(tile_next, c4n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out0 = b.add_i32(lds_dyn, c20480);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let ooff = b.c_i32(i_const * 128);
        let oaddr = b.add_i32(o_lds_dyn, ooff);
        let oraw = b.load_lds_b32(Opnd::Val(oaddr));
        let ov = b.bitcast_f32(oraw);
        let res = b.mul_f32(ov, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    b.finish()
}

/// Same algorithm as probe_fp8_fmha_online, but the O accumulator is a loop-carried
/// Frag16 held in registers: each iteration rescales it per row (lane values times the
/// LDS-published correction) and feeds it back as the MFMA accumulator input.
pub fn probe_fp8_fmha_oreg() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_oreg", 64);
    b.lds(8448);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c4t = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c10240 = b.c_i32(10240);
    let c18432 = b.c_i32(18432);
    let c19456 = b.c_i32(19456);
    let c20480 = b.c_i32(20480);
    let c24576 = b.c_i32(24576);
    let lds_corr = b.c_i32(4096);
    let lds_o = b.c_i32(4224);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_dyn = b.add_i32(j4, p512);
    let pblk16 = b.shl_i32(pblk, c4t);
    let corr_row = b.add_i32(j4, lds_corr);
    let o_lds_dyn = b.add_i32(lds_dyn, lds_o);

    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let tid4 = b.mul_i32(tid, c4);
    let zero = b.frag_zero(DType::Frag16);

    // Q fragment, loaded once.
    let q_base = b.ptr_add(o, frag_off);
    let q0 = b.load_f32(q_base, 0);
    let q0 = b.bitcast_i32(q0);
    let q1 = b.load_f32(q_base, 4);
    let q1 = b.bitcast_i32(q1);
    let q2 = b.load_f32(q_base, 8);
    let q2 = b.bitcast_i32(q2);
    let q3 = b.load_f32(q_base, 12);
    let q3 = b.bitcast_i32(q3);
    let q4 = b.load_f32(q_base, 16);
    let q4 = b.bitcast_i32(q4);
    let q5 = b.load_f32(q_base, 20);
    let q5 = b.bitcast_i32(q5);
    let q6 = b.load_f32(q_base, 24);
    let q6 = b.bitcast_i32(q6);
    let q7 = b.load_f32(q_base, 28);
    let q7 = b.bitcast_i32(q7);
    let a8 = b.frag8(q0, q1, q2, q3, q4, q5, q6, q7);

    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let tile = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_var = vars[3];

    // ---- S_t = Q K_t^T ----
    let tile_off = b.mul_i32(tile, c2048);
    let kt0 = b.add_i32(frag_off, tile_off);
    let kt1 = b.add_i32(kt0, c2048);
    let kt_base = b.ptr_add(o, kt1);
    let k0 = b.load_f32(kt_base, 0);
    let k0 = b.bitcast_i32(k0);
    let k1 = b.load_f32(kt_base, 4);
    let k1 = b.bitcast_i32(k1);
    let k2 = b.load_f32(kt_base, 8);
    let k2 = b.bitcast_i32(k2);
    let k3 = b.load_f32(kt_base, 12);
    let k3 = b.bitcast_i32(k3);
    let k4 = b.load_f32(kt_base, 16);
    let k4 = b.bitcast_i32(k4);
    let k5 = b.load_f32(kt_base, 20);
    let k5 = b.bitcast_i32(k5);
    let k6 = b.load_f32(kt_base, 24);
    let k6 = b.bitcast_i32(k6);
    let k7 = b.load_f32(kt_base, 28);
    let k7 = b.bitcast_i32(k7);
    let k8 = b.frag8(k0, k1, k2, k3, k4, k5, k6, k7);

    let sc0 = b.mul_i32(tile, c256);
    let sc1 = b.add_i32(sc0, tid4);
    let sa_off = b.add_i32(sc1, c18432);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(sc1, c19456);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);
    let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(s_frag, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_dyn, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- online softmax update for this lane's row ----
    let rbase = b.mul_i32(j, c128);
    let mut vals: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32u32 {
        let coff = b.c_i32(4 * k as i32);
        let addr = b.add_i32(rbase, coff);
        let raw = b.load_lds_b32(Opnd::Val(addr));
        vals.push(b.bitcast_f32(raw));
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..32usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..32usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    // Publish the row correction for the lanes that own this row's O elements.
    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16) and PV for tile t ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let a = b.mul_f32(exps[4 * m], c16f);
        let bb = b.mul_f32(exps[4 * m + 1], c16f);
        let p0 = b.cvt_pk_fp8_f32(a, bb);
        let c = b.mul_f32(exps[4 * m + 2], c16f);
        let d = b.mul_f32(exps[4 * m + 3], c16f);
        let p1 = b.cvt_pk_fp8_f32(c, d);
        let sh = b.shl_i32(p1, c16i);
        words.push(b.add_i32(p0, sh));
    }
    let pt0 = b.mul_i32(tile, c1024);
    let pt1 = b.mul_i32(j, c32);
    let pt2 = b.add_i32(pt0, pt1);
    let pt3 = b.add_i32(pt2, c24576);
    let p_ptr = b.ptr_add(o, pt3);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    let vt0 = b.add_i32(frag_off, tile_off);
    let vt1 = b.add_i32(vt0, c10240);
    let vt_base = b.ptr_add(o, vt1);
    let v0 = b.load_f32(vt_base, 0);
    let v0 = b.bitcast_i32(v0);
    let v1 = b.load_f32(vt_base, 4);
    let v1 = b.bitcast_i32(v1);
    let v2 = b.load_f32(vt_base, 8);
    let v2 = b.bitcast_i32(v2);
    let v3 = b.load_f32(vt_base, 12);
    let v3 = b.bitcast_i32(v3);
    let v4 = b.load_f32(vt_base, 16);
    let v4 = b.bitcast_i32(v4);
    let v5 = b.load_f32(vt_base, 20);
    let v5 = b.bitcast_i32(v5);
    let v6 = b.load_f32(vt_base, 24);
    let v6 = b.bitcast_i32(v6);
    let v7 = b.load_f32(vt_base, 28);
    let v7 = b.bitcast_i32(v7);
    let v8 = b.frag8(v0, v1, v2, v3, v4, v5, v6, v7);
    // ---- O = MFMA(P, V, C = O * corr[row]); O stays in registers ----
    let mut scaled_lanes: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let sc = b.mul_f32(lanev, cf);
        scaled_lanes.push(b.bitcast_i32(sc));
    }
    let c_scaled = b.frag16_build(&scaled_lanes);
    let o_frag = b.mfma_scale_fp8(p8, v8, c_scaled, c127, c123);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let tile_next = b.add_i32(tile, c1);
    b.set_var(tile, tile_next);
    b.set_var(o_var, o_frag);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    let c4n = b.c_i32(4);
    let more = b.cmp_lt_i32(tile_next, c4n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out0 = b.add_i32(lds_dyn, c20480);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let res = b.mul_f32(lanev, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    b.finish()
}

/// Eight-wave (512 thread) version of the online-softmax kernel: every wave owns one
/// 32-row query tile and its own LDS region, all waves share the K/V tiles and run the
/// same barrier sequence, as in the reference kernel's block of 512.
pub fn probe_fp8_fmha_multiwave() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_multiwave", 512);
    b.lds(67584);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid_full = b.let_(Op::WorkitemId);
    let c63w = b.c_i32(63);
    let c6w = b.c_i32(6);
    let tid = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c63w)));
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid_full), Opnd::Val(c6w)));

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c4t = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c10240 = b.c_i32(24576); // V^T tiles
    let c18432 = b.c_i32(32768); // Q scales
    let c19456 = b.c_i32(33792); // K scales
    let c20480 = b.c_i32(34816); // O out (per wave +4096)
    let c24576 = b.c_i32(67584); // P bytes (per wave +4096)
    let c8448w = b.c_i32(8448);
    let c2048w = b.c_i32(2048);
    let c4096w = b.c_i32(4096);
    let c16384 = b.c_i32(16384);
    let wbase = b.mul_i32(wave, c8448w);
    let wq = b.mul_i32(wave, c2048w);
    let wout = b.mul_i32(wave, c4096w);
    let lds_corr0 = b.c_i32(4096);
    let lds_corr = b.add_i32(lds_corr0, wbase);
    let lds_o = b.c_i32(4224);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_loc = b.add_i32(j4, p512);
    let lds_dyn = b.add_i32(lds_loc, wbase);
    let pblk16 = b.shl_i32(pblk, c4t);
    let corr_row = b.add_i32(j4, lds_corr);
    let o_lds_dyn = b.add_i32(lds_dyn, lds_o);

    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let tid4 = b.mul_i32(tid, c4);
    let zero = b.frag_zero(DType::Frag16);

    // Q fragment, loaded once.
    let q_off = b.add_i32(frag_off, wq);
    let q_base = b.ptr_add(o, q_off);
    let q0 = b.load_f32(q_base, 0);
    let q0 = b.bitcast_i32(q0);
    let q1 = b.load_f32(q_base, 4);
    let q1 = b.bitcast_i32(q1);
    let q2 = b.load_f32(q_base, 8);
    let q2 = b.bitcast_i32(q2);
    let q3 = b.load_f32(q_base, 12);
    let q3 = b.bitcast_i32(q3);
    let q4 = b.load_f32(q_base, 16);
    let q4 = b.bitcast_i32(q4);
    let q5 = b.load_f32(q_base, 20);
    let q5 = b.bitcast_i32(q5);
    let q6 = b.load_f32(q_base, 24);
    let q6 = b.bitcast_i32(q6);
    let q7 = b.load_f32(q_base, 28);
    let q7 = b.bitcast_i32(q7);
    let a8 = b.frag8(q0, q1, q2, q3, q4, q5, q6, q7);

    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let tile = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_var = vars[3];

    // ---- S_t = Q K_t^T ----
    let tile_off = b.mul_i32(tile, c2048);
    let kt0 = b.add_i32(frag_off, tile_off);
    let kt1 = b.add_i32(kt0, c16384);
    let kt_base = b.ptr_add(o, kt1);
    let k0 = b.load_f32(kt_base, 0);
    let k0 = b.bitcast_i32(k0);
    let k1 = b.load_f32(kt_base, 4);
    let k1 = b.bitcast_i32(k1);
    let k2 = b.load_f32(kt_base, 8);
    let k2 = b.bitcast_i32(k2);
    let k3 = b.load_f32(kt_base, 12);
    let k3 = b.bitcast_i32(k3);
    let k4 = b.load_f32(kt_base, 16);
    let k4 = b.bitcast_i32(k4);
    let k5 = b.load_f32(kt_base, 20);
    let k5 = b.bitcast_i32(k5);
    let k6 = b.load_f32(kt_base, 24);
    let k6 = b.bitcast_i32(k6);
    let k7 = b.load_f32(kt_base, 28);
    let k7 = b.bitcast_i32(k7);
    let k8 = b.frag8(k0, k1, k2, k3, k4, k5, k6, k7);

    let sc0 = b.mul_i32(tile, c256);
    let sc1 = b.add_i32(sc0, tid4);
    let sa_off = b.add_i32(sc1, c18432);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(sc1, c19456);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);
    let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(s_frag, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_dyn, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- online softmax update for this lane's row ----
    let rbase0 = b.mul_i32(j, c128);
    let rbase = b.add_i32(rbase0, wbase);
    let mut vals: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32u32 {
        let coff = b.c_i32(4 * k as i32);
        let addr = b.add_i32(rbase, coff);
        let raw = b.load_lds_b32(Opnd::Val(addr));
        vals.push(b.bitcast_f32(raw));
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..32usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..32usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    // Publish the row correction for the lanes that own this row's O elements.
    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16) and PV for tile t ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let a = b.mul_f32(exps[4 * m], c16f);
        let bb = b.mul_f32(exps[4 * m + 1], c16f);
        let p0 = b.cvt_pk_fp8_f32(a, bb);
        let c = b.mul_f32(exps[4 * m + 2], c16f);
        let d = b.mul_f32(exps[4 * m + 3], c16f);
        let p1 = b.cvt_pk_fp8_f32(c, d);
        let sh = b.shl_i32(p1, c16i);
        words.push(b.add_i32(p0, sh));
    }
    let pt0 = b.mul_i32(tile, c1024);
    let pt1 = b.mul_i32(j, c32);
    let pt2 = b.add_i32(pt0, pt1);
    let pt2b = b.add_i32(pt2, wout);
    let pt3 = b.add_i32(pt2b, c24576);
    let p_ptr = b.ptr_add(o, pt3);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    let vt0 = b.add_i32(frag_off, tile_off);
    let vt1 = b.add_i32(vt0, c10240);
    let vt_base = b.ptr_add(o, vt1);
    let v0 = b.load_f32(vt_base, 0);
    let v0 = b.bitcast_i32(v0);
    let v1 = b.load_f32(vt_base, 4);
    let v1 = b.bitcast_i32(v1);
    let v2 = b.load_f32(vt_base, 8);
    let v2 = b.bitcast_i32(v2);
    let v3 = b.load_f32(vt_base, 12);
    let v3 = b.bitcast_i32(v3);
    let v4 = b.load_f32(vt_base, 16);
    let v4 = b.bitcast_i32(v4);
    let v5 = b.load_f32(vt_base, 20);
    let v5 = b.bitcast_i32(v5);
    let v6 = b.load_f32(vt_base, 24);
    let v6 = b.bitcast_i32(v6);
    let v7 = b.load_f32(vt_base, 28);
    let v7 = b.bitcast_i32(v7);
    let v8 = b.frag8(v0, v1, v2, v3, v4, v5, v6, v7);
    // ---- O = MFMA(P, V, C = O * corr[row]); O stays in registers ----
    let mut scaled_lanes: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let sc = b.mul_f32(lanev, cf);
        scaled_lanes.push(b.bitcast_i32(sc));
    }
    let c_scaled = b.frag16_build(&scaled_lanes);
    let o_frag = b.mfma_scale_fp8(p8, v8, c_scaled, c127, c123);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let tile_next = b.add_i32(tile, c1);
    b.set_var(tile, tile_next);
    b.set_var(o_var, o_frag);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    let c4n = b.c_i32(4);
    let more = b.cmp_lt_i32(tile_next, c4n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out00 = b.add_i32(lds_loc, wout);
    let o_out0 = b.add_i32(o_out00, c20480);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let res = b.mul_f32(lanev, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    b.finish()
}

/// Role-split version of the eight-wave kernel: K and V are staged through a double-buffered
/// LDS region by a producer/consumer split (waves 0-3 copy K, waves 4-7 copy V), the two role
/// branches end in the same workgroup barrier, and every wave reads its fragments with
/// ds_read_b128.
pub fn probe_fp8_fmha_pingpong() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_pingpong", 512);
    b.lds(75776);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid_full = b.let_(Op::WorkitemId);
    let c63w = b.c_i32(63);
    let c6w = b.c_i32(6);
    let tid = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c63w)));
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid_full), Opnd::Val(c6w)));

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c4t = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c10240 = b.c_i32(24576); // V^T tiles
    let c18432 = b.c_i32(32768); // Q scales
    let c19456 = b.c_i32(33792); // K scales
    let c20480 = b.c_i32(34816); // O out (per wave +4096)
    let c24576 = b.c_i32(67584); // P bytes (per wave +4096)
    let c8448w = b.c_i32(8448);
    let c2048w = b.c_i32(2048);
    let c4096w = b.c_i32(4096);
    let c16384 = b.c_i32(16384);
    let wbase = b.mul_i32(wave, c8448w);
    let wq = b.mul_i32(wave, c2048w);
    let wout = b.mul_i32(wave, c4096w);
    let kvb = b.c_i32(67584);
    let c255 = b.c_i32(255);
    let c3k = b.c_i32(3);
    let c1k = b.c_i32(1);
    let c2048k = b.c_i32(2048);
    let c4096k = b.c_i32(4096);
    let c16384k = b.c_i32(16384);
    let c24576k = b.c_i32(24576);
    let is_a = b.cmp_lt_i32(wave, c4);
    let rt = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c255)));
    let rt8 = b.shl_i32(rt, c3k);
    let lds_corr0 = b.c_i32(4096);
    let lds_corr = b.add_i32(lds_corr0, wbase);
    let lds_o = b.c_i32(4224);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_loc = b.add_i32(j4, p512);
    let lds_dyn = b.add_i32(lds_loc, wbase);
    let pblk16 = b.shl_i32(pblk, c4t);
    let corr_row = b.add_i32(j4, lds_corr);
    let o_lds_dyn = b.add_i32(lds_dyn, lds_o);

    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let tid4 = b.mul_i32(tid, c4);
    let zero = b.frag_zero(DType::Frag16);

    // Q fragment, loaded once.
    let q_off = b.add_i32(frag_off, wq);
    let q_base = b.ptr_add(o, q_off);
    let q0 = b.load_f32(q_base, 0);
    let q0 = b.bitcast_i32(q0);
    let q1 = b.load_f32(q_base, 4);
    let q1 = b.bitcast_i32(q1);
    let q2 = b.load_f32(q_base, 8);
    let q2 = b.bitcast_i32(q2);
    let q3 = b.load_f32(q_base, 12);
    let q3 = b.bitcast_i32(q3);
    let q4 = b.load_f32(q_base, 16);
    let q4 = b.bitcast_i32(q4);
    let q5 = b.load_f32(q_base, 20);
    let q5 = b.bitcast_i32(q5);
    let q6 = b.load_f32(q_base, 24);
    let q6 = b.bitcast_i32(q6);
    let q7 = b.load_f32(q_base, 28);
    let q7 = b.bitcast_i32(q7);
    let a8 = b.frag8(q0, q1, q2, q3, q4, q5, q6, q7);

    // Prologue: stage tile 0 into LDS slot 0 with the same role split.
    {
        let gk0 = b.add_i32(c16384k, rt8);
        let gv0 = b.add_i32(c24576k, rt8);
        let lk = b.add_i32(kvb, rt8);
        let lv0 = b.add_i32(kvb, c2048k);
        let lv = b.add_i32(lv0, rt8);
        b.begin_if(is_a);
        {
            let gp = b.ptr_add(o, gk0);
            let w0 = b.load_f32(gp, 0);
            let w1 = b.load_f32(gp, 4);
            let w0 = b.bitcast_i32(w0);
            let w1 = b.bitcast_i32(w1);
            let lk4 = b.add_i32(lk, c4);
            b.store_lds_b32(Opnd::Val(lk), Opnd::Val(w0));
            b.store_lds_b32(Opnd::Val(lk4), Opnd::Val(w1));
            b.wait_lds(0);
            b.barrier();
            b.wait_lds(0);
        }
        b.else_();
        {
            let gp = b.ptr_add(o, gv0);
            let w0 = b.load_f32(gp, 0);
            let w1 = b.load_f32(gp, 4);
            let w0 = b.bitcast_i32(w0);
            let w1 = b.bitcast_i32(w1);
            let lv4 = b.add_i32(lv, c4);
            b.store_lds_b32(Opnd::Val(lv), Opnd::Val(w0));
            b.store_lds_b32(Opnd::Val(lv4), Opnd::Val(w1));
            b.wait_lds(0);
            b.barrier();
            b.wait_lds(0);
        }
        b.end_if();
    }    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let tile = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_var = vars[3];

    // ---- S_t = Q K_t^T ----
    let slot = b.let_(Op::AndI32(tile.into(), Opnd::Val(c1k)));
    let slot_off = b.mul_i32(slot, c4096k);
    let kslot0 = b.add_i32(slot_off, kvb);
    let kaddr = b.add_i32(kslot0, frag_off);
    let kaddr_hi = b.add_i32(kaddr, c16i);
    let k_lo = b.load_lds_b128(Opnd::Val(kaddr));
    let k_hi = b.load_lds_b128(Opnd::Val(kaddr_hi));
    let k8 = b.cat_frag8(k_lo, k_hi);

    let sc0 = b.mul_i32(tile, c256);
    let sc1 = b.add_i32(sc0, tid4);
    let sa_off = b.add_i32(sc1, c18432);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(sc1, c19456);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);
    let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(s_frag, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_dyn, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- online softmax update for this lane's row ----
    let rbase0 = b.mul_i32(j, c128);
    let rbase = b.add_i32(rbase0, wbase);
    let mut vals: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32u32 {
        let coff = b.c_i32(4 * k as i32);
        let addr = b.add_i32(rbase, coff);
        let raw = b.load_lds_b32(Opnd::Val(addr));
        vals.push(b.bitcast_f32(raw));
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..32usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..32usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    // Publish the row correction for the lanes that own this row's O elements.
    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16) and PV for tile t ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let a = b.mul_f32(exps[4 * m], c16f);
        let bb = b.mul_f32(exps[4 * m + 1], c16f);
        let p0 = b.cvt_pk_fp8_f32(a, bb);
        let c = b.mul_f32(exps[4 * m + 2], c16f);
        let d = b.mul_f32(exps[4 * m + 3], c16f);
        let p1 = b.cvt_pk_fp8_f32(c, d);
        let sh = b.shl_i32(p1, c16i);
        words.push(b.add_i32(p0, sh));
    }
    let pt0 = b.mul_i32(tile, c1024);
    let pt1 = b.mul_i32(j, c32);
    let pt2 = b.add_i32(pt0, pt1);
    let pt2b = b.add_i32(pt2, wout);
    let pt3 = b.add_i32(pt2b, c24576);
    let p_ptr = b.ptr_add(o, pt3);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    let vslot0 = b.add_i32(kslot0, c2048k);
    let vaddr = b.add_i32(vslot0, frag_off);
    let vaddr_hi = b.add_i32(vaddr, c16i);
    let v_lo = b.load_lds_b128(Opnd::Val(vaddr));
    let v_hi = b.load_lds_b128(Opnd::Val(vaddr_hi));
    let v8 = b.cat_frag8(v_lo, v_hi);
    // ---- O = MFMA(P, V, C = O * corr[row]); O stays in registers ----
    let mut scaled_lanes: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let sc = b.mul_f32(lanev, cf);
        scaled_lanes.push(b.bitcast_i32(sc));
    }
    let c_scaled = b.frag16_build(&scaled_lanes);
    let o_frag = b.mfma_scale_fp8(p8, v8, c_scaled, c127, c123);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    // Producer/consumer stage of the next tile: role A (waves 0-3) copies K, role B
    // (waves 4-7) copies V into the other LDS slot; both branches end in the same
    // workgroup barrier.
    let tn = b.add_i32(tile, c1k);
    let tn_off = b.mul_i32(tn, c2048k);
    let ns = b.let_(Op::AndI32(Opnd::Val(tn), Opnd::Val(c1k)));
    let ns_off = b.mul_i32(ns, c4096k);
    let nslot = b.add_i32(ns_off, kvb);
    let gk0 = b.add_i32(tn_off, c16384k);
    let gk = b.add_i32(gk0, rt8);
    let lk = b.add_i32(nslot, rt8);
    let gv0 = b.add_i32(tn_off, c24576k);
    let gv = b.add_i32(gv0, rt8);
    let lv0 = b.add_i32(nslot, c2048k);
    let lv = b.add_i32(lv0, rt8);
    b.begin_if(is_a);
    {
        let gp = b.ptr_add(o, gk);
        let w0 = b.load_f32(gp, 0);
        let w1 = b.load_f32(gp, 4);
        let w0 = b.bitcast_i32(w0);
        let w1 = b.bitcast_i32(w1);
        let lk4 = b.add_i32(lk, c4);
        b.store_lds_b32(Opnd::Val(lk), Opnd::Val(w0));
        b.store_lds_b32(Opnd::Val(lk4), Opnd::Val(w1));
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }
    b.else_();
    {
        let gp = b.ptr_add(o, gv);
        let w0 = b.load_f32(gp, 0);
        let w1 = b.load_f32(gp, 4);
        let w0 = b.bitcast_i32(w0);
        let w1 = b.bitcast_i32(w1);
        let lv4 = b.add_i32(lv, c4);
        b.store_lds_b32(Opnd::Val(lv), Opnd::Val(w0));
        b.store_lds_b32(Opnd::Val(lv4), Opnd::Val(w1));
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }
    b.end_if();
    let tile_next = b.add_i32(tile, c1);
    b.set_var(tile, tile_next);
    b.set_var(o_var, o_frag);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    let c4n = b.c_i32(4);
    let more = b.cmp_lt_i32(tile_next, c4n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out00 = b.add_i32(lds_loc, wout);
    let o_out0 = b.add_i32(o_out00, c20480);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let res = b.mul_f32(lanev, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    b.finish()
}

/// Separate steady loops per role: waves 0-3 run an A loop that copies K, waves 4-7 run a
/// B loop that copies V; each loop has its own backedge and the same barrier schedule.
pub fn probe_fp8_fmha_abloops() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_abloops", 512);
    b.lds(75776);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid_full = b.let_(Op::WorkitemId);
    let c63w = b.c_i32(63);
    let c6w = b.c_i32(6);
    let tid = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c63w)));
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid_full), Opnd::Val(c6w)));

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c4t = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c10240 = b.c_i32(24576); // V^T tiles
    let c18432 = b.c_i32(32768); // Q scales
    let c19456 = b.c_i32(33792); // K scales
    let c20480 = b.c_i32(34816); // O out (per wave +4096)
    let c24576 = b.c_i32(67584); // P bytes (per wave +4096)
    let c8448w = b.c_i32(8448);
    let c2048w = b.c_i32(2048);
    let c4096w = b.c_i32(4096);
    let c16384 = b.c_i32(16384);
    let wbase = b.mul_i32(wave, c8448w);
    let wq = b.mul_i32(wave, c2048w);
    let wout = b.mul_i32(wave, c4096w);
    let kvb = b.c_i32(67584);
    let c255 = b.c_i32(255);
    let c3k = b.c_i32(3);
    let c1k = b.c_i32(1);
    let c2048k = b.c_i32(2048);
    let c4096k = b.c_i32(4096);
    let c16384k = b.c_i32(16384);
    let c24576k = b.c_i32(24576);
    let is_a = b.cmp_lt_i32(wave, c4);
    let rt = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c255)));
    let rt8 = b.shl_i32(rt, c3k);
    let lds_corr0 = b.c_i32(4096);
    let lds_corr = b.add_i32(lds_corr0, wbase);
    let lds_o = b.c_i32(4224);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_loc = b.add_i32(j4, p512);
    let lds_dyn = b.add_i32(lds_loc, wbase);
    let pblk16 = b.shl_i32(pblk, c4t);
    let corr_row = b.add_i32(j4, lds_corr);
    let o_lds_dyn = b.add_i32(lds_dyn, lds_o);

    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let tid4 = b.mul_i32(tid, c4);
    let zero = b.frag_zero(DType::Frag16);

    // Q fragment, loaded once.
    let q_off = b.add_i32(frag_off, wq);
    let q_base = b.ptr_add(o, q_off);
    let q0 = b.load_f32(q_base, 0);
    let q0 = b.bitcast_i32(q0);
    let q1 = b.load_f32(q_base, 4);
    let q1 = b.bitcast_i32(q1);
    let q2 = b.load_f32(q_base, 8);
    let q2 = b.bitcast_i32(q2);
    let q3 = b.load_f32(q_base, 12);
    let q3 = b.bitcast_i32(q3);
    let q4 = b.load_f32(q_base, 16);
    let q4 = b.bitcast_i32(q4);
    let q5 = b.load_f32(q_base, 20);
    let q5 = b.bitcast_i32(q5);
    let q6 = b.load_f32(q_base, 24);
    let q6 = b.bitcast_i32(q6);
    let q7 = b.load_f32(q_base, 28);
    let q7 = b.bitcast_i32(q7);
    let a8 = b.frag8(q0, q1, q2, q3, q4, q5, q6, q7);

    // Prologue: stage tile 0 into LDS slot 0 with the same role split.
    {
        let gk0 = b.add_i32(c16384k, rt8);
        let gv0 = b.add_i32(c24576k, rt8);
        let lk = b.add_i32(kvb, rt8);
        let lv0 = b.add_i32(kvb, c2048k);
        let lv = b.add_i32(lv0, rt8);
        b.begin_if(is_a);
        {
            let gp = b.ptr_add(o, gk0);
            let w0 = b.load_f32(gp, 0);
            let w1 = b.load_f32(gp, 4);
            let w0 = b.bitcast_i32(w0);
            let w1 = b.bitcast_i32(w1);
            let lk4 = b.add_i32(lk, c4);
            b.store_lds_b32(Opnd::Val(lk), Opnd::Val(w0));
            b.store_lds_b32(Opnd::Val(lk4), Opnd::Val(w1));
            b.wait_lds(0);
            b.barrier();
            b.wait_lds(0);
        }
        b.else_();
        {
            let gp = b.ptr_add(o, gv0);
            let w0 = b.load_f32(gp, 0);
            let w1 = b.load_f32(gp, 4);
            let w0 = b.bitcast_i32(w0);
            let w1 = b.bitcast_i32(w1);
            let lv4 = b.add_i32(lv, c4);
            b.store_lds_b32(Opnd::Val(lv), Opnd::Val(w0));
            b.store_lds_b32(Opnd::Val(lv4), Opnd::Val(w1));
            b.wait_lds(0);
            b.barrier();
            b.wait_lds(0);
        }
        b.end_if();
    }    // Role A and role B each run their own steady loop with its own backedge; both
    // execute the same four collective barriers per iteration plus one in the epilogue.
    b.begin_if(is_a);
    {
    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let tile = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_var = vars[3];

    // ---- S_t = Q K_t^T ----
    let slot = b.let_(Op::AndI32(tile.into(), Opnd::Val(c1k)));
    let slot_off = b.mul_i32(slot, c4096k);
    let kslot0 = b.add_i32(slot_off, kvb);
    let kaddr = b.add_i32(kslot0, frag_off);
    let kaddr_hi = b.add_i32(kaddr, c16i);
    let k_lo = b.load_lds_b128(Opnd::Val(kaddr));
    let k_hi = b.load_lds_b128(Opnd::Val(kaddr_hi));
    let k8 = b.cat_frag8(k_lo, k_hi);

    let sc0 = b.mul_i32(tile, c256);
    let sc1 = b.add_i32(sc0, tid4);
    let sa_off = b.add_i32(sc1, c18432);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(sc1, c19456);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);
    let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(s_frag, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_dyn, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- online softmax update for this lane's row ----
    let rbase0 = b.mul_i32(j, c128);
    let rbase = b.add_i32(rbase0, wbase);
    let mut vals: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32u32 {
        let coff = b.c_i32(4 * k as i32);
        let addr = b.add_i32(rbase, coff);
        let raw = b.load_lds_b32(Opnd::Val(addr));
        vals.push(b.bitcast_f32(raw));
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..32usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..32usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    // Publish the row correction for the lanes that own this row's O elements.
    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16) and PV for tile t ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let a = b.mul_f32(exps[4 * m], c16f);
        let bb = b.mul_f32(exps[4 * m + 1], c16f);
        let p0 = b.cvt_pk_fp8_f32(a, bb);
        let c = b.mul_f32(exps[4 * m + 2], c16f);
        let d = b.mul_f32(exps[4 * m + 3], c16f);
        let p1 = b.cvt_pk_fp8_f32(c, d);
        let sh = b.shl_i32(p1, c16i);
        words.push(b.add_i32(p0, sh));
    }
    let pt0 = b.mul_i32(tile, c1024);
    let pt1 = b.mul_i32(j, c32);
    let pt2 = b.add_i32(pt0, pt1);
    let pt2b = b.add_i32(pt2, wout);
    let pt3 = b.add_i32(pt2b, c24576);
    let p_ptr = b.ptr_add(o, pt3);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    let vslot0 = b.add_i32(kslot0, c2048k);
    let vaddr = b.add_i32(vslot0, frag_off);
    let vaddr_hi = b.add_i32(vaddr, c16i);
    let v_lo = b.load_lds_b128(Opnd::Val(vaddr));
    let v_hi = b.load_lds_b128(Opnd::Val(vaddr_hi));
    let v8 = b.cat_frag8(v_lo, v_hi);
    // ---- O = MFMA(P, V, C = O * corr[row]); O stays in registers ----
    let mut scaled_lanes: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let sc = b.mul_f32(lanev, cf);
        scaled_lanes.push(b.bitcast_i32(sc));
    }
    let c_scaled = b.frag16_build(&scaled_lanes);
    let o_frag = b.mfma_scale_fp8(p8, v8, c_scaled, c127, c123);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    // Producer/consumer stage of the next tile: role A (waves 0-3) copies K, role B
    // (waves 4-7) copies V into the other LDS slot; both branches end in the same
    // workgroup barrier.
    let tn = b.add_i32(tile, c1k);
    let tn_off = b.mul_i32(tn, c2048k);
    let ns = b.let_(Op::AndI32(Opnd::Val(tn), Opnd::Val(c1k)));
    let ns_off = b.mul_i32(ns, c4096k);
    let nslot = b.add_i32(ns_off, kvb);
    let gk0 = b.add_i32(tn_off, c16384k);
    let gk = b.add_i32(gk0, rt8);
    let lk = b.add_i32(nslot, rt8);
    let gv0 = b.add_i32(tn_off, c24576k);
    let gv = b.add_i32(gv0, rt8);
    let lv0 = b.add_i32(nslot, c2048k);
    let lv = b.add_i32(lv0, rt8);

    {
        let gp = b.ptr_add(o, gk);
        let w0 = b.load_f32(gp, 0);
        let w1 = b.load_f32(gp, 4);
        let w0 = b.bitcast_i32(w0);
        let w1 = b.bitcast_i32(w1);
        let lk4 = b.add_i32(lk, c4);
        b.store_lds_b32(Opnd::Val(lk), Opnd::Val(w0));
        b.store_lds_b32(Opnd::Val(lk4), Opnd::Val(w1));
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }

    let tile_next = b.add_i32(tile, c1);
    b.set_var(tile, tile_next);
    b.set_var(o_var, o_frag);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    let c4n = b.c_i32(4);
    let more = b.cmp_lt_i32(tile_next, c4n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out00 = b.add_i32(lds_loc, wout);
    let o_out0 = b.add_i32(o_out00, c20480);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let res = b.mul_f32(lanev, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    }
    b.else_();
    {
    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let tile = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_var = vars[3];

    // ---- S_t = Q K_t^T ----
    let slot = b.let_(Op::AndI32(tile.into(), Opnd::Val(c1k)));
    let slot_off = b.mul_i32(slot, c4096k);
    let kslot0 = b.add_i32(slot_off, kvb);
    let kaddr = b.add_i32(kslot0, frag_off);
    let kaddr_hi = b.add_i32(kaddr, c16i);
    let k_lo = b.load_lds_b128(Opnd::Val(kaddr));
    let k_hi = b.load_lds_b128(Opnd::Val(kaddr_hi));
    let k8 = b.cat_frag8(k_lo, k_hi);

    let sc0 = b.mul_i32(tile, c256);
    let sc1 = b.add_i32(sc0, tid4);
    let sa_off = b.add_i32(sc1, c18432);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(sc1, c19456);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);
    let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(s_frag, r);
        let coff = b.c_i32(i_const * 128);
        let addr = b.add_i32(lds_dyn, coff);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- online softmax update for this lane's row ----
    let rbase0 = b.mul_i32(j, c128);
    let rbase = b.add_i32(rbase0, wbase);
    let mut vals: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32u32 {
        let coff = b.c_i32(4 * k as i32);
        let addr = b.add_i32(rbase, coff);
        let raw = b.load_lds_b32(Opnd::Val(addr));
        vals.push(b.bitcast_f32(raw));
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..32usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(32);
    for k in 0..32usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..32usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    // Publish the row correction for the lanes that own this row's O elements.
    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16) and PV for tile t ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let a = b.mul_f32(exps[4 * m], c16f);
        let bb = b.mul_f32(exps[4 * m + 1], c16f);
        let p0 = b.cvt_pk_fp8_f32(a, bb);
        let c = b.mul_f32(exps[4 * m + 2], c16f);
        let d = b.mul_f32(exps[4 * m + 3], c16f);
        let p1 = b.cvt_pk_fp8_f32(c, d);
        let sh = b.shl_i32(p1, c16i);
        words.push(b.add_i32(p0, sh));
    }
    let pt0 = b.mul_i32(tile, c1024);
    let pt1 = b.mul_i32(j, c32);
    let pt2 = b.add_i32(pt0, pt1);
    let pt2b = b.add_i32(pt2, wout);
    let pt3 = b.add_i32(pt2b, c24576);
    let p_ptr = b.ptr_add(o, pt3);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    let vslot0 = b.add_i32(kslot0, c2048k);
    let vaddr = b.add_i32(vslot0, frag_off);
    let vaddr_hi = b.add_i32(vaddr, c16i);
    let v_lo = b.load_lds_b128(Opnd::Val(vaddr));
    let v_hi = b.load_lds_b128(Opnd::Val(vaddr_hi));
    let v8 = b.cat_frag8(v_lo, v_hi);
    // ---- O = MFMA(P, V, C = O * corr[row]); O stays in registers ----
    let mut scaled_lanes: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let sc = b.mul_f32(lanev, cf);
        scaled_lanes.push(b.bitcast_i32(sc));
    }
    let c_scaled = b.frag16_build(&scaled_lanes);
    let o_frag = b.mfma_scale_fp8(p8, v8, c_scaled, c127, c123);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    // Producer/consumer stage of the next tile: role A (waves 0-3) copies K, role B
    // (waves 4-7) copies V into the other LDS slot; both branches end in the same
    // workgroup barrier.
    let tn = b.add_i32(tile, c1k);
    let tn_off = b.mul_i32(tn, c2048k);
    let ns = b.let_(Op::AndI32(Opnd::Val(tn), Opnd::Val(c1k)));
    let ns_off = b.mul_i32(ns, c4096k);
    let nslot = b.add_i32(ns_off, kvb);
    let gk0 = b.add_i32(tn_off, c16384k);
    let gk = b.add_i32(gk0, rt8);
    let lk = b.add_i32(nslot, rt8);
    let gv0 = b.add_i32(tn_off, c24576k);
    let gv = b.add_i32(gv0, rt8);
    let lv0 = b.add_i32(nslot, c2048k);
    let lv = b.add_i32(lv0, rt8);

    {
        let gp = b.ptr_add(o, gv);
        let w0 = b.load_f32(gp, 0);
        let w1 = b.load_f32(gp, 4);
        let w0 = b.bitcast_i32(w0);
        let w1 = b.bitcast_i32(w1);
        let lv4 = b.add_i32(lv, c4);
        b.store_lds_b32(Opnd::Val(lv), Opnd::Val(w0));
        b.store_lds_b32(Opnd::Val(lv4), Opnd::Val(w1));
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }

    let tile_next = b.add_i32(tile, c1);
    b.set_var(tile, tile_next);
    b.set_var(o_var, o_frag);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    let c4n = b.c_i32(4);
    let more = b.cmp_lt_i32(tile_next, c4n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out00 = b.add_i32(lds_loc, wout);
    let o_out0 = b.add_i32(o_out00, c20480);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let res = b.mul_f32(lanev, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    }
    b.end_if();
    b.finish()
}
/// Measurement probe for `ds_read_b64_tr_b8`. LDS byte `a` holds `a & 0xFF`; lane `l` of the
/// first instruction supplies byte address `8 * l`, the second one `8 * l + 512`. The two
/// result dwords of both instructions are stored to `o[tid*4 .. tid*4+3]` so the host can
/// decode the transpose pattern.
pub fn probe_lds_tr_b8() -> Function {
    let mut b = FnBuilder::new("probe_lds_tr_b8", 64);
    b.lds(2048);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c2 = b.c_i32(2);
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c16 = b.c_i32(16);
    let c256 = b.c_i32(256);
    let c512 = b.c_i32(512);
    let rep = b.c_i32(0x0101_0101);
    let base_pat = b.c_i32(0x0302_0100);

    // word value for byte address 4*tid: bytes (4tid .. 4tid+3)
    let off4 = b.shl_i32(tid, c2);
    let scaled = b.mul_i32(off4, rep);
    let value = b.add_i32(scaled, base_pat);
    for k in 0..4i32 {
        let ck = b.c_i32(256 * k);
        let addr = b.add_i32(off4, ck);
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(value));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let addr1 = b.shl_i32(tid, c3);
    let addr2 = b.add_i32(addr1, c512);
    let r1 = b.load_lds_tr_b8(Opnd::Val(addr1));
    let r2 = b.load_lds_tr_b8(Opnd::Val(addr2));
    b.wait_lds(0);

    let out_off = b.shl_i32(tid, c4);
    let out_ptr = b.ptr_add(o, out_off);
    for i in 0..2u8 {
        let lane = b.frag16_lane(r1, i);
        let v = b.bitcast_f32(lane);
        b.store_f32(out_ptr, 4 * i as u32, v);
    }
    for i in 0..2u8 {
        let lane = b.frag16_lane(r2, i);
        let v = b.bitcast_f32(lane);
        b.store_f32(out_ptr, 8 + 4 * i as u32, v);
    }
    let _ = (c16, c256);
    b.finish()
}

/// Block-scaled fp8 matmul whose B operand is row-major in LDS and is turned into the MFMA
/// fragment by four ds_read_b64_tr_b8 reads (address formula measured with
/// probe_lds_tr_b8).
pub fn probe_fp8_matmul_tr() -> Function {
    let mut b = FnBuilder::new("probe_fp8_matmul_tr", 64);
    b.lds(2048);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c9 = b.c_i32(9);
    let c31 = b.c_i32(31);
    let c64 = b.c_i32(64);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c4352 = b.c_i32(4352);
    let c4608 = b.c_i32(4608);

    // lane decomposition: row index j (lane % 32) and K block (lane / 32)
    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let p = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));

    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(p, c5);
    let frag_off = b.add_i32(row, koff);
    let a_base = b.ptr_add(o, frag_off);
    let bt_off = b.add_i32(frag_off, c2048);
    let bt_base = b.ptr_add(o, bt_off);

    let a0 = b.load_f32(a_base, 0);
    let a0 = b.bitcast_i32(a0);
    let a1 = b.load_f32(a_base, 4);
    let a1 = b.bitcast_i32(a1);
    let a2 = b.load_f32(a_base, 8);
    let a2 = b.bitcast_i32(a2);
    let a3 = b.load_f32(a_base, 12);
    let a3 = b.bitcast_i32(a3);
    let a4 = b.load_f32(a_base, 16);
    let a4 = b.bitcast_i32(a4);
    let a5 = b.load_f32(a_base, 20);
    let a5 = b.bitcast_i32(a5);
    let a6 = b.load_f32(a_base, 24);
    let a6 = b.bitcast_i32(a6);
    let a7 = b.load_f32(a_base, 28);
    let a7 = b.bitcast_i32(a7);
    let a8 = b.frag8(a0, a1, a2, a3, a4, a5, a6, a7);

    // B is row-major (B[k][j] at 2048 + 32k + j). Stage it into LDS (lane copies 32 bytes),
    // then build the B fragment with four ds_read_b64_tr_b8 reads.
    let c7 = b.c_i32(7);
    let c1 = b.c_i32(1);
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c32 = b.c_i32(32);
    let c16 = b.c_i32(16);
    let c8 = b.c_i32(8);
    let c1k = b.c_i32(1);
    let tid32 = b.mul_i32(tid, c32);
    let g_off = b.add_i32(tid32, c2048);
    let g_ptr = b.ptr_add(o, g_off);
    for w in 0..8u32 {
        let v = b.load_f32(g_ptr, 4 * w);
        let vi = b.bitcast_i32(v);
        let wc = b.c_i32(4 * w as i32);
        let la = b.add_i32(tid32, wc);
        b.store_lds_b32(Opnd::Val(la), Opnd::Val(vi));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    // lane M: group g = M / 16, i = (M % 16) / 2, h = M % 2
    //   address(r) = p * 1024 + (8r + i) * 32 + 16 * (g % 2) + 8 * h
    let tid_s1 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c1k)));
    let i_idx = b.let_(Op::AndI32(Opnd::Val(tid_s1), Opnd::Val(c7)));
    let tid_s4 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c4)));
    let g2 = b.let_(Op::AndI32(Opnd::Val(tid_s4), Opnd::Val(c1)));
    let h_idx = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1)));
    let p1024 = b.mul_i32(p, c1024);
    let i32b = b.mul_i32(i_idx, c32);
    let g216 = b.mul_i32(g2, c16);
    let h8 = b.mul_i32(h_idx, c8);
    let t0 = b.add_i32(p1024, i32b);
    let t1 = b.add_i32(t0, g216);
    let base0 = b.add_i32(t1, h8);
    let mut bwords: Vec<ValueId> = Vec::with_capacity(8);
    for r in 0..4i32 {
        let rc = b.c_i32(256 * r);
        let addr = b.add_i32(base0, rc);
        let tr = b.load_lds_tr_b8(Opnd::Val(addr));
        bwords.push(b.frag16_lane(tr, 0));
        bwords.push(b.frag16_lane(tr, 1));
    }
    let b8v = b.frag8(bwords[0], bwords[1], bwords[2], bwords[3], bwords[4], bwords[5], bwords[6], bwords[7]);

    let tid4 = b.mul_i32(tid, c4);
    let sa_off = b.add_i32(tid4, c4096);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_f = b.load_f32(sa_ptr, 0);
    let sa = b.bitcast_i32(sa_f);
    let sb_off = b.add_i32(tid4, c4352);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_f = b.load_f32(sb_ptr, 0);
    let sb = b.bitcast_i32(sb_f);

    let zero = b.frag_zero(DType::Frag16);
    let d = b.mfma_scale_fp8(a8, b8v, zero, sa, sb);

    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(p, c9);
    let lane_off = b.add_i32(j4, p512);
    let dyn_off = b.add_i32(lane_off, c4608);
    let dptr = b.ptr_add(o, dyn_off);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as u32;
        let lane = b.frag16_lane(d, r);
        let v = b.bitcast_f32(lane);
        b.store_f32(dptr, i_const * 128, v);
    }
    b.finish()
}
/// Online-softmax MXFP8 attention with 64 keys per iteration and V read through
/// `ds_read_b64_tr_b8`.
///
/// Two 32-key score tiles per iteration, a 64-wide online softmax update, quantised P as the
/// full K = 64 A fragment (lane group g holds keys 32g..32g+31) and the V tile (64 keys x 32
/// dims, row-major) staged in LDS and turned into the B fragment by four transposing reads.
///
/// Input image in `o` (S = 256 buffer): `0` Q; `2048 + 2048 t` K^T tile t (t < 4);
/// `10240 + 2048 it` row-major V of iteration it; `14336 + 256 t` Q scales, `15360 + 256 t` K
/// scales; output O at `16384`; quantised P bytes at `20480 + 2048 it + 64 row + key`.
pub fn probe_fp8_fmha_tr() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_tr", 64);
    b.lds(10496);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c7 = b.c_i32(7);
    let c8 = b.c_i32(8);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c512 = b.c_i32(512);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c10240 = b.c_i32(10240);
    let c14336 = b.c_i32(14336);
    let c15360 = b.c_i32(15360);
    let c16384 = b.c_i32(16384);
    let c20480 = b.c_i32(20480);
    let lds_corr = b.c_i32(8192);
    let lds_v = b.c_i32(8320);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_dyn = b.add_i32(j4, p512);
    let pblk16 = b.shl_i32(pblk, c4);
    let corr_row = b.add_i32(j4, lds_corr);
    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let tid4 = b.mul_i32(tid, c4);
    let tid32 = b.mul_i32(tid, c32);
    let zero = b.frag_zero(DType::Frag16);

    // transposing-read base address inside the staged V tile (see probe_fp8_matmul_tr)
    let tid_s1 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c1)));
    let i_idx = b.let_(Op::AndI32(Opnd::Val(tid_s1), Opnd::Val(c7)));
    let tid_s4 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c4)));
    let g2 = b.let_(Op::AndI32(Opnd::Val(tid_s4), Opnd::Val(c1)));
    let h_idx = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1)));
    let p1024 = b.mul_i32(pblk, c1024);
    let i32b = b.mul_i32(i_idx, c32);
    let g216 = b.mul_i32(g2, c16i);
    let h8 = b.mul_i32(h_idx, c8);
    let tr0 = b.add_i32(p1024, i32b);
    let tr1 = b.add_i32(tr0, g216);
    let tr2 = b.add_i32(tr1, h8);
    let tr_base = b.add_i32(tr2, lds_v);

    // Q fragment, loaded once.
    let q_base = b.ptr_add(o, frag_off);
    let q0 = b.load_f32(q_base, 0);
    let q0 = b.bitcast_i32(q0);
    let q1 = b.load_f32(q_base, 4);
    let q1 = b.bitcast_i32(q1);
    let q2 = b.load_f32(q_base, 8);
    let q2 = b.bitcast_i32(q2);
    let q3 = b.load_f32(q_base, 12);
    let q3 = b.bitcast_i32(q3);
    let q4 = b.load_f32(q_base, 16);
    let q4 = b.bitcast_i32(q4);
    let q5 = b.load_f32(q_base, 20);
    let q5 = b.bitcast_i32(q5);
    let q6 = b.load_f32(q_base, 24);
    let q6 = b.bitcast_i32(q6);
    let q7 = b.load_f32(q_base, 28);
    let q7 = b.bitcast_i32(q7);
    let a8 = b.frag8(q0, q1, q2, q3, q4, q5, q6, q7);

    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_var = vars[3];

    // ---- two score tiles for this iteration ----
    let it4096 = b.mul_i32(it, c4096);
    let it512 = b.mul_i32(it, c512);
    for s in 0..2i32 {
        let ks = b.c_i32(2048 + 2048 * s);
        let kt0 = b.add_i32(frag_off, it4096);
        let kt1 = b.add_i32(kt0, ks);
        let kt_base = b.ptr_add(o, kt1);
        let k0 = b.load_f32(kt_base, 0);
        let k0 = b.bitcast_i32(k0);
        let k1 = b.load_f32(kt_base, 4);
        let k1 = b.bitcast_i32(k1);
        let k2 = b.load_f32(kt_base, 8);
        let k2 = b.bitcast_i32(k2);
        let k3 = b.load_f32(kt_base, 12);
        let k3 = b.bitcast_i32(k3);
        let k4 = b.load_f32(kt_base, 16);
        let k4 = b.bitcast_i32(k4);
        let k5 = b.load_f32(kt_base, 20);
        let k5 = b.bitcast_i32(k5);
        let k6 = b.load_f32(kt_base, 24);
        let k6 = b.bitcast_i32(k6);
        let k7 = b.load_f32(kt_base, 28);
        let k7 = b.bitcast_i32(k7);
        let k8 = b.frag8(k0, k1, k2, k3, k4, k5, k6, k7);

        let ss = b.c_i32(256 * s);
        let sc0 = b.add_i32(it512, ss);
        let sc1 = b.add_i32(sc0, tid4);
        let sa_off = b.add_i32(sc1, c14336);
        let sa_ptr = b.ptr_add(o, sa_off);
        let sa_f = b.load_f32(sa_ptr, 0);
        let sa = b.bitcast_i32(sa_f);
        let sb_off = b.add_i32(sc1, c15360);
        let sb_ptr = b.ptr_add(o, sb_off);
        let sb_f = b.load_f32(sb_ptr, 0);
        let sb = b.bitcast_i32(sb_f);
        let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

        let reg = b.c_i32(4096 * s);
        let lbase = b.add_i32(lds_dyn, reg);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as i32;
            let lane = b.frag16_lane(s_frag, r);
            let coff = b.c_i32(i_const * 128);
            let addr = b.add_i32(lbase, coff);
            b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
        }
    }

    // ---- stage the row-major V tile of this iteration into LDS ----
    let it2048 = b.mul_i32(it, c2048);
    let v_g0 = b.add_i32(tid32, it2048);
    let v_g1 = b.add_i32(v_g0, c10240);
    let v_g = b.ptr_add(o, v_g1);
    let v_l = b.add_i32(tid32, lds_v);
    for w in 0..8u32 {
        let v = b.load_f32(v_g, 4 * w);
        let vi = b.bitcast_i32(v);
        let wc = b.c_i32(4 * w as i32);
        let la = b.add_i32(v_l, wc);
        b.store_lds_b32(Opnd::Val(la), Opnd::Val(vi));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- 64-wide online softmax update for this lane's row ----
    let rbase = b.mul_i32(j, c128);
    let mut vals: Vec<ValueId> = Vec::with_capacity(64);
    for s in 0..2u32 {
        for k in 0..32u32 {
            let coff = b.c_i32((s * 4096 + 4 * k) as i32);
            let addr = b.add_i32(rbase, coff);
            let raw = b.load_lds_b32(Opnd::Val(addr));
            vals.push(b.bitcast_f32(raw));
        }
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..64usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(64);
    for k in 0..64usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..64usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16), lane group g keeps keys 32g..32g+31 ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let mut halves: Vec<ValueId> = Vec::with_capacity(2);
        for half in 0..2usize {
            let base = 32 * half + 4 * m;
            let a = b.mul_f32(exps[base], c16f);
            let bb = b.mul_f32(exps[base + 1], c16f);
            let p0 = b.cvt_pk_fp8_f32(a, bb);
            let c = b.mul_f32(exps[base + 2], c16f);
            let d = b.mul_f32(exps[base + 3], c16f);
            let p1 = b.cvt_pk_fp8_f32(c, d);
            let sh = b.shl_i32(p1, c16i);
            halves.push(b.add_i32(p0, sh));
        }
        let diff = b.xor_i32(halves[0], halves[1]);
        let pick = b.mul_i32(diff, pblk);
        words.push(b.xor_i32(halves[0], pick));
    }
    let pb0 = b.mul_i32(it, c2048);
    let pb1 = b.mul_i32(j, c64);
    let pb2 = b.mul_i32(pblk, c32);
    let pb3 = b.add_i32(pb0, pb1);
    let pb4 = b.add_i32(pb3, pb2);
    let pb5 = b.add_i32(pb4, c20480);
    let p_ptr = b.ptr_add(o, pb5);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);

    // ---- V fragment through four transposing reads ----
    let mut vw: Vec<ValueId> = Vec::with_capacity(8);
    for r in 0..4i32 {
        let rc = b.c_i32(256 * r);
        let addr = b.add_i32(tr_base, rc);
        let tr = b.load_lds_tr_b8(Opnd::Val(addr));
        vw.push(b.frag16_lane(tr, 0));
        vw.push(b.frag16_lane(tr, 1));
    }
    let v8 = b.frag8(vw[0], vw[1], vw[2], vw[3], vw[4], vw[5], vw[6], vw[7]);

    // ---- O = MFMA(P, V, C = O * corr[row]) ----
    let mut scaled_lanes: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let sc = b.mul_f32(lanev, cf);
        scaled_lanes.push(b.bitcast_i32(sc));
    }
    let c_scaled = b.frag16_build(&scaled_lanes);
    let o_frag = b.mfma_scale_fp8(p8, v8, c_scaled, c127, c123);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    b.set_var(o_var, o_frag);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out0 = b.add_i32(lds_dyn, c16384);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let res = b.mul_f32(lanev, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    let _ = c3;
    b.finish()
}

/// 512-thread version of the 64-key tr kernel: eight query tiles, shared double-buffered V in LDS
/// copied by role A (first half) and role B (second half), separate A/B steady loops with the
/// same barrier schedule, V consumed with ds_read_b64_tr_b8.
pub fn probe_fp8_fmha_trab() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_trab", 512);
    b.lds(70656);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid_full = b.let_(Op::WorkitemId);
    let c63w = b.c_i32(63);
    let c6w = b.c_i32(6);
    let tid = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c63w)));
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid_full), Opnd::Val(c6w)));

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c7 = b.c_i32(7);
    let c8 = b.c_i32(8);
    let c9 = b.c_i32(9);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c512 = b.c_i32(512);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c24576 = b.c_i32(24576);
    let c14336 = b.c_i32(28672);
    let c15360 = b.c_i32(29696);
    let c16384 = b.c_i32(30720);
    let c20480 = b.c_i32(63488);
    let c8320 = b.c_i32(8320);
    let c255 = b.c_i32(255);
    let c4096w = b.c_i32(4096);
    let c2048w = b.c_i32(2048);
    let kvb = b.c_i32(66560);
    let wbase = b.mul_i32(wave, c8320);
    let wq = b.mul_i32(wave, c2048w);
    let wout = b.mul_i32(wave, c4096w);
    let lds_corr0 = b.c_i32(8192);
    let lds_corr = b.add_i32(lds_corr0, wbase);
    let is_a = b.cmp_lt_i32(wave, c4);
    let rt = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c255)));
    let rt4 = b.shl_i32(rt, c2);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_loc = b.add_i32(j4, p512);
    let lds_dyn = b.add_i32(lds_loc, wbase);
    let pblk16 = b.shl_i32(pblk, c4);
    let corr_row = b.add_i32(j4, lds_corr);
    let row = b.mul_i32(j, c64);
    let koff = b.shl_i32(pblk, c5);
    let frag_off = b.add_i32(row, koff);
    let tid4 = b.mul_i32(tid, c4);
    let tid32 = b.mul_i32(tid, c32);
    let zero = b.frag_zero(DType::Frag16);

    // transposing-read base address inside the staged V tile (see probe_fp8_matmul_tr)
    let tid_s1 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c1)));
    let i_idx = b.let_(Op::AndI32(Opnd::Val(tid_s1), Opnd::Val(c7)));
    let tid_s4 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c4)));
    let g2 = b.let_(Op::AndI32(Opnd::Val(tid_s4), Opnd::Val(c1)));
    let h_idx = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1)));
    let p1024 = b.mul_i32(pblk, c1024);
    let i32b = b.mul_i32(i_idx, c32);
    let g216 = b.mul_i32(g2, c16i);
    let h8 = b.mul_i32(h_idx, c8);
    let tr0 = b.add_i32(p1024, i32b);
    let tr1 = b.add_i32(tr0, g216);
    let tr2 = b.add_i32(tr1, h8);
    let tr_base0 = b.add_i32(tr2, kvb);

    // Q fragment, loaded once.
    let q_off = b.add_i32(frag_off, wq);
    let q_base = b.ptr_add(o, q_off);
    let q0 = b.load_f32(q_base, 0);
    let q0 = b.bitcast_i32(q0);
    let q1 = b.load_f32(q_base, 4);
    let q1 = b.bitcast_i32(q1);
    let q2 = b.load_f32(q_base, 8);
    let q2 = b.bitcast_i32(q2);
    let q3 = b.load_f32(q_base, 12);
    let q3 = b.bitcast_i32(q3);
    let q4 = b.load_f32(q_base, 16);
    let q4 = b.bitcast_i32(q4);
    let q5 = b.load_f32(q_base, 20);
    let q5 = b.bitcast_i32(q5);
    let q6 = b.load_f32(q_base, 24);
    let q6 = b.bitcast_i32(q6);
    let q7 = b.load_f32(q_base, 28);
    let q7 = b.bitcast_i32(q7);
    let a8 = b.frag8(q0, q1, q2, q3, q4, q5, q6, q7);

    // Prologue: stage V_0 into slot 0 with the same role split.
    {
        let pg = b.add_i32(c24576, rt4);
        let pl = b.add_i32(kvb, rt4);
        b.begin_if(is_a);
        {
            let gp = b.ptr_add(o, pg);
            let w0 = b.load_f32(gp, 0);
            let w0 = b.bitcast_i32(w0);
            b.store_lds_b32(Opnd::Val(pl), Opnd::Val(w0));
            b.wait_lds(0);
            b.barrier();
            b.wait_lds(0);
        }
        b.else_();
        {
            let gp = b.ptr_add(o, pg);
            let w0 = b.load_f32(gp, 1024);
            let w0 = b.bitcast_i32(w0);
            let pl1k = b.add_i32(pl, c1024);
            b.store_lds_b32(Opnd::Val(pl1k), Opnd::Val(w0));
            b.wait_lds(0);
            b.barrier();
            b.wait_lds(0);
        }
        b.end_if();
    }    b.begin_if(is_a);
    {
    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_var = vars[3];

    // ---- two score tiles for this iteration ----
    let it4096 = b.mul_i32(it, c4096);
    let it512 = b.mul_i32(it, c512);
    for s in 0..2i32 {
        let ks = b.c_i32(16384 + 2048 * s);
        let kt0 = b.add_i32(frag_off, it4096);
        let kt1 = b.add_i32(kt0, ks);
        let kt_base = b.ptr_add(o, kt1);
        let k0 = b.load_f32(kt_base, 0);
        let k0 = b.bitcast_i32(k0);
        let k1 = b.load_f32(kt_base, 4);
        let k1 = b.bitcast_i32(k1);
        let k2 = b.load_f32(kt_base, 8);
        let k2 = b.bitcast_i32(k2);
        let k3 = b.load_f32(kt_base, 12);
        let k3 = b.bitcast_i32(k3);
        let k4 = b.load_f32(kt_base, 16);
        let k4 = b.bitcast_i32(k4);
        let k5 = b.load_f32(kt_base, 20);
        let k5 = b.bitcast_i32(k5);
        let k6 = b.load_f32(kt_base, 24);
        let k6 = b.bitcast_i32(k6);
        let k7 = b.load_f32(kt_base, 28);
        let k7 = b.bitcast_i32(k7);
        let k8 = b.frag8(k0, k1, k2, k3, k4, k5, k6, k7);

        let ss = b.c_i32(256 * s);
        let sc0 = b.add_i32(it512, ss);
        let sc1 = b.add_i32(sc0, tid4);
        let sa_off = b.add_i32(sc1, c14336);
        let sa_ptr = b.ptr_add(o, sa_off);
        let sa_f = b.load_f32(sa_ptr, 0);
        let sa = b.bitcast_i32(sa_f);
        let sb_off = b.add_i32(sc1, c15360);
        let sb_ptr = b.ptr_add(o, sb_off);
        let sb_f = b.load_f32(sb_ptr, 0);
        let sb = b.bitcast_i32(sb_f);
        let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

        let reg = b.c_i32(4096 * s);
        let lbase = b.add_i32(lds_dyn, reg);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as i32;
            let lane = b.frag16_lane(s_frag, r);
            let coff = b.c_i32(i_const * 128);
            let addr = b.add_i32(lbase, coff);
            b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
        }
    }

    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- 64-wide online softmax update for this lane's row ----
    let rbase0 = b.mul_i32(j, c128);
    let rbase = b.add_i32(rbase0, wbase);
    let mut vals: Vec<ValueId> = Vec::with_capacity(64);
    for s in 0..2u32 {
        for k in 0..32u32 {
            let coff = b.c_i32((s * 4096 + 4 * k) as i32);
            let addr = b.add_i32(rbase, coff);
            let raw = b.load_lds_b32(Opnd::Val(addr));
            vals.push(b.bitcast_f32(raw));
        }
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..64usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(64);
    for k in 0..64usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..64usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16), lane group g keeps keys 32g..32g+31 ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let mut halves: Vec<ValueId> = Vec::with_capacity(2);
        for half in 0..2usize {
            let base = 32 * half + 4 * m;
            let a = b.mul_f32(exps[base], c16f);
            let bb = b.mul_f32(exps[base + 1], c16f);
            let p0 = b.cvt_pk_fp8_f32(a, bb);
            let c = b.mul_f32(exps[base + 2], c16f);
            let d = b.mul_f32(exps[base + 3], c16f);
            let p1 = b.cvt_pk_fp8_f32(c, d);
            let sh = b.shl_i32(p1, c16i);
            halves.push(b.add_i32(p0, sh));
        }
        let diff = b.xor_i32(halves[0], halves[1]);
        let pick = b.mul_i32(diff, pblk);
        words.push(b.xor_i32(halves[0], pick));
    }
    let pb0 = b.mul_i32(it, c2048);
    let pb1 = b.mul_i32(j, c64);
    let pb2 = b.mul_i32(pblk, c32);
    let pb3 = b.add_i32(pb0, pb1);
    let pb4 = b.add_i32(pb3, pb2);
    let pb4b = b.add_i32(pb4, wout);
    let pb5 = b.add_i32(pb4b, c20480);
    let p_ptr = b.ptr_add(o, pb5);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);

    // ---- V fragment through four transposing reads from the current slot ----
    let slot_bit = b.let_(Op::AndI32(it.into(), Opnd::Val(c1)));
    let slot_off = b.mul_i32(slot_bit, c2048);
    let tr_base = b.add_i32(tr_base0, slot_off);
    let mut vw: Vec<ValueId> = Vec::with_capacity(8);
    for r in 0..4i32 {
        let rc = b.c_i32(256 * r);
        let addr = b.add_i32(tr_base, rc);
        let tr = b.load_lds_tr_b8(Opnd::Val(addr));
        vw.push(b.frag16_lane(tr, 0));
        vw.push(b.frag16_lane(tr, 1));
    }
    let v8 = b.frag8(vw[0], vw[1], vw[2], vw[3], vw[4], vw[5], vw[6], vw[7]);

    // ---- O = MFMA(P, V, C = O * corr[row]) ----
    let mut scaled_lanes: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let sc = b.mul_f32(lanev, cf);
        scaled_lanes.push(b.bitcast_i32(sc));
    }
    let c_scaled = b.frag16_build(&scaled_lanes);
    let o_frag = b.mfma_scale_fp8(p8, v8, c_scaled, c127, c123);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // Producer stage of V_{it+1}: role A copies the first 1024 bytes, role B the second.
    let itn = b.add_i32(it, c1);
    let itn_g = b.mul_i32(itn, c2048);
    let itn_slot = b.let_(Op::AndI32(Opnd::Val(itn), Opnd::Val(c1)));
    let itn_l = b.mul_i32(itn_slot, c2048);
    let sg0 = b.add_i32(itn_g, c24576);
    let sg = b.add_i32(sg0, rt4);
    let sl0 = b.add_i32(itn_l, kvb);
    let sl = b.add_i32(sl0, rt4);

    {
        let gp = b.ptr_add(o, sg);
        let w0 = b.load_f32(gp, 0);
        let w0 = b.bitcast_i32(w0);
        b.store_lds_b32(Opnd::Val(sl), Opnd::Val(w0));
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    b.set_var(o_var, o_frag);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out00 = b.add_i32(lds_loc, wout);
    let o_out0 = b.add_i32(o_out00, c16384);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let res = b.mul_f32(lanev, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    let _ = (c3, c32, c256, c4096, c8);
    }
    b.else_();
    {
    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_var = vars[3];

    // ---- two score tiles for this iteration ----
    let it4096 = b.mul_i32(it, c4096);
    let it512 = b.mul_i32(it, c512);
    for s in 0..2i32 {
        let ks = b.c_i32(16384 + 2048 * s);
        let kt0 = b.add_i32(frag_off, it4096);
        let kt1 = b.add_i32(kt0, ks);
        let kt_base = b.ptr_add(o, kt1);
        let k0 = b.load_f32(kt_base, 0);
        let k0 = b.bitcast_i32(k0);
        let k1 = b.load_f32(kt_base, 4);
        let k1 = b.bitcast_i32(k1);
        let k2 = b.load_f32(kt_base, 8);
        let k2 = b.bitcast_i32(k2);
        let k3 = b.load_f32(kt_base, 12);
        let k3 = b.bitcast_i32(k3);
        let k4 = b.load_f32(kt_base, 16);
        let k4 = b.bitcast_i32(k4);
        let k5 = b.load_f32(kt_base, 20);
        let k5 = b.bitcast_i32(k5);
        let k6 = b.load_f32(kt_base, 24);
        let k6 = b.bitcast_i32(k6);
        let k7 = b.load_f32(kt_base, 28);
        let k7 = b.bitcast_i32(k7);
        let k8 = b.frag8(k0, k1, k2, k3, k4, k5, k6, k7);

        let ss = b.c_i32(256 * s);
        let sc0 = b.add_i32(it512, ss);
        let sc1 = b.add_i32(sc0, tid4);
        let sa_off = b.add_i32(sc1, c14336);
        let sa_ptr = b.ptr_add(o, sa_off);
        let sa_f = b.load_f32(sa_ptr, 0);
        let sa = b.bitcast_i32(sa_f);
        let sb_off = b.add_i32(sc1, c15360);
        let sb_ptr = b.ptr_add(o, sb_off);
        let sb_f = b.load_f32(sb_ptr, 0);
        let sb = b.bitcast_i32(sb_f);
        let s_frag = b.mfma_scale_fp8(a8, k8, zero, sa, sb);

        let reg = b.c_i32(4096 * s);
        let lbase = b.add_i32(lds_dyn, reg);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as i32;
            let lane = b.frag16_lane(s_frag, r);
            let coff = b.c_i32(i_const * 128);
            let addr = b.add_i32(lbase, coff);
            b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
        }
    }

    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- 64-wide online softmax update for this lane's row ----
    let rbase0 = b.mul_i32(j, c128);
    let rbase = b.add_i32(rbase0, wbase);
    let mut vals: Vec<ValueId> = Vec::with_capacity(64);
    for s in 0..2u32 {
        for k in 0..32u32 {
            let coff = b.c_i32((s * 4096 + 4 * k) as i32);
            let addr = b.add_i32(rbase, coff);
            let raw = b.load_lds_b32(Opnd::Val(addr));
            vals.push(b.bitcast_f32(raw));
        }
    }
    b.wait_lds(0);
    let mut tmax = vals[0];
    for k in 1..64usize {
        tmax = b.max_f32(tmax, vals[k]);
    }
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let mut exps: Vec<ValueId> = Vec::with_capacity(64);
    for k in 0..64usize {
        let sh = b.sub_f32(vals[k], m_new);
        exps.push(b.exp_f32(sh));
    }
    let mut rsum = exps[0];
    for k in 1..64usize {
        rsum = b.add_f32(rsum, exps[k]);
    }
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);

    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- quantise P (x16), lane group g keeps keys 32g..32g+31 ----
    let mut words: Vec<ValueId> = Vec::with_capacity(8);
    for m in 0..8usize {
        let mut halves: Vec<ValueId> = Vec::with_capacity(2);
        for half in 0..2usize {
            let base = 32 * half + 4 * m;
            let a = b.mul_f32(exps[base], c16f);
            let bb = b.mul_f32(exps[base + 1], c16f);
            let p0 = b.cvt_pk_fp8_f32(a, bb);
            let c = b.mul_f32(exps[base + 2], c16f);
            let d = b.mul_f32(exps[base + 3], c16f);
            let p1 = b.cvt_pk_fp8_f32(c, d);
            let sh = b.shl_i32(p1, c16i);
            halves.push(b.add_i32(p0, sh));
        }
        let diff = b.xor_i32(halves[0], halves[1]);
        let pick = b.mul_i32(diff, pblk);
        words.push(b.xor_i32(halves[0], pick));
    }
    let pb0 = b.mul_i32(it, c2048);
    let pb1 = b.mul_i32(j, c64);
    let pb2 = b.mul_i32(pblk, c32);
    let pb3 = b.add_i32(pb0, pb1);
    let pb4 = b.add_i32(pb3, pb2);
    let pb4b = b.add_i32(pb4, wout);
    let pb5 = b.add_i32(pb4b, c20480);
    let p_ptr = b.ptr_add(o, pb5);
    for m in 0..8usize {
        let v = b.bitcast_f32(words[m]);
        b.store_f32(p_ptr, 4 * m as u32, v);
    }
    let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);

    // ---- V fragment through four transposing reads from the current slot ----
    let slot_bit = b.let_(Op::AndI32(it.into(), Opnd::Val(c1)));
    let slot_off = b.mul_i32(slot_bit, c2048);
    let tr_base = b.add_i32(tr_base0, slot_off);
    let mut vw: Vec<ValueId> = Vec::with_capacity(8);
    for r in 0..4i32 {
        let rc = b.c_i32(256 * r);
        let addr = b.add_i32(tr_base, rc);
        let tr = b.load_lds_tr_b8(Opnd::Val(addr));
        vw.push(b.frag16_lane(tr, 0));
        vw.push(b.frag16_lane(tr, 1));
    }
    let v8 = b.frag8(vw[0], vw[1], vw[2], vw[3], vw[4], vw[5], vw[6], vw[7]);

    // ---- O = MFMA(P, V, C = O * corr[row]) ----
    let mut scaled_lanes: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let sc = b.mul_f32(lanev, cf);
        scaled_lanes.push(b.bitcast_i32(sc));
    }
    let c_scaled = b.frag16_build(&scaled_lanes);
    let o_frag = b.mfma_scale_fp8(p8, v8, c_scaled, c127, c123);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // Producer stage of V_{it+1}: role A copies the first 1024 bytes, role B the second.
    let itn = b.add_i32(it, c1);
    let itn_g = b.mul_i32(itn, c2048);
    let itn_slot = b.let_(Op::AndI32(Opnd::Val(itn), Opnd::Val(c1)));
    let itn_l = b.mul_i32(itn_slot, c2048);
    let sg0 = b.add_i32(itn_g, c24576);
    let sg = b.add_i32(sg0, rt4);
    let sl0 = b.add_i32(itn_l, kvb);
    let sl = b.add_i32(sl0, rt4);

    {
        let gp = b.ptr_add(o, sg);
        let w0 = b.load_f32(gp, 1024);
        let w0 = b.bitcast_i32(w0);
        let sl1k = b.add_i32(sl, c1024);
        b.store_lds_b32(Opnd::Val(sl1k), Opnd::Val(w0));
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    b.set_var(o_var, o_frag);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write O ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let o_out00 = b.add_i32(lds_loc, wout);
    let o_out0 = b.add_i32(o_out00, c16384);
    let o_ptr = b.ptr_add(o, o_out0);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        let cf = b.bitcast_f32(craw);
        let lane = b.frag16_lane(o_var, r);
        let lanev = b.bitcast_f32(lane);
        let res = b.mul_f32(lanev, cf);
        b.store_f32(o_ptr, (i_const * 128) as u32, res);
    }
    let _ = (c3, c32, c256, c4096, c8);
    }
    b.end_if();
    b.finish()
}
/// Load eight consecutive dwords from `ptr` and build an fp8 `Frag8`.
fn frag8_from_global(b: &mut FnBuilder, ptr: ValueId) -> ValueId {
    let mut w: Vec<ValueId> = Vec::with_capacity(8);
    for i in 0..8u32 {
        let v = b.load_f32(ptr, 4 * i);
        w.push(b.bitcast_i32(v));
    }
    b.frag8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
}

/// Head dimension 128, KV tile 128: online-softmax MXFP8 attention for one 32-row query tile.
///
/// Per iteration: four 32-key score tiles, each the sum of two K = 64 block-scaled MFMAs (the
/// two halves of the 128-wide head, chained through the accumulator); a two-pass 128-wide
/// softmax per row (a max pass, then exp/sum/quantise 32 keys at a time so only the 32
/// quantised words stay live); eight PV MFMAs (two 64-key halves x four 32-wide output tiles)
/// whose B fragments come from the row-major V tile through `ds_read_b64_tr_b8` (pitch 128).
/// The four 32x32 O accumulators are loop-carried `Frag16`s.
///
/// Input image in `o` (S = 1024 buffer): `0` Q (32 x 128), `4096 + 4096 t` K^T tile t
/// (32 keys x 128, t < 8), `36864 + 16384 it` row-major V (128 keys x 128), `69632 + 256 c` Q
/// scales of head chunk c, `70144 + 256 (2 t + c)` K scales; output O (32 x 128 f32) at
/// `74240`; quantised P bytes at `90624 + 4096 it + 128 row + 32 chunk + key`.
pub fn probe_fp8_fmha_hd128() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_hd128", 64);
    b.lds(33024);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c7 = b.c_i32(7);
    let c8 = b.c_i32(8);
    let c9 = b.c_i32(9);
    let c11 = b.c_i32(11);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c256 = b.c_i32(256);
    let c1024 = b.c_i32(1024);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c16384 = b.c_i32(16384);
    let c36864 = b.c_i32(36864);
    let c69632 = b.c_i32(69632);
    let c70144 = b.c_i32(70144);
    let c74240 = b.c_i32(74240);
    let c90624 = b.c_i32(90624);
    let lds_corr = b.c_i32(16384);
    let lds_v = b.c_i32(16512);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_dyn = b.add_i32(j4, p512);
    let pblk16 = b.shl_i32(pblk, c4);
    let corr_row = b.add_i32(j4, lds_corr);
    let tid4 = b.mul_i32(tid, c4);
    let tid256 = b.shl_i32(tid, c8);
    let zero = b.frag_zero(DType::Frag16);

    // fragment byte offset in a row-major (pitch 128) operand: row j, K block p
    let j128 = b.mul_i32(j, c128);
    let p32 = b.shl_i32(pblk, c5);
    let frag_row = b.add_i32(j128, p32);

    // transposing-read lane address (pitch 128): p*32P + i*P + 16*(g%2) + 8*h
    let tid_s1 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c1)));
    let i_idx = b.let_(Op::AndI32(Opnd::Val(tid_s1), Opnd::Val(c7)));
    let tid_s4 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c4)));
    let g2 = b.let_(Op::AndI32(Opnd::Val(tid_s4), Opnd::Val(c1)));
    let h_idx = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1)));
    let c12 = b.c_i32(12);
    let pP = b.shl_i32(pblk, c12);
    let iP = b.mul_i32(i_idx, c128);
    let g216 = b.mul_i32(g2, c16i);
    let h8 = b.mul_i32(h_idx, c8);
    let tr0 = b.add_i32(pP, iP);
    let tr1 = b.add_i32(tr0, g216);
    let tr2 = b.add_i32(tr1, h8);
    let tr_base = b.add_i32(tr2, lds_v);

    // Q fragments for the two head chunks and their scales, loaded once.
    let q_ptr0 = b.ptr_add(o, frag_row);
    let a8_0 = frag8_from_global(&mut b, q_ptr0);
    let c64 = b.c_i32(64);
    let q_off1 = b.add_i32(frag_row, c64);
    let q_ptr1 = b.ptr_add(o, q_off1);
    let a8_1 = frag8_from_global(&mut b, q_ptr1);
    let sa0_off = b.add_i32(tid4, c69632);
    let sa0_ptr = b.ptr_add(o, sa0_off);
    let sa0_f = b.load_f32(sa0_ptr, 0);
    let sa0 = b.bitcast_i32(sa0_f);
    let sa1_ptr = b.ptr_add(o, sa0_off);
    let sa1_f = b.load_f32(sa1_ptr, 256);
    let sa1 = b.bitcast_i32(sa1_f);

    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_vars = [vars[3], vars[4], vars[5], vars[6]];

    // ---- four 32-key score tiles, two chained MFMAs each ----
    let it16384 = b.mul_i32(it, c16384);
    let it8 = b.mul_i32(it, c2048);
    for s in 0..4i32 {
        let kt_s = b.c_i32(4096 + 4096 * s);
        let kt0 = b.add_i32(frag_row, it16384);
        let kt1 = b.add_i32(kt0, kt_s);
        let kp0 = b.ptr_add(o, kt1);
        let k8_0 = frag8_from_global(&mut b, kp0);
        let kt2 = b.add_i32(kt1, c64);
        let kp1 = b.ptr_add(o, kt2);
        let k8_1 = frag8_from_global(&mut b, kp1);
        // K scale index (2t + c) * 256 with t = 4 it + s -> 2048 it + 512 s + 256 c
        let sb_s = b.c_i32(512 * s);
        let sb0 = b.add_i32(it8, sb_s);
        let sb1 = b.add_i32(sb0, tid4);
        let sb2 = b.add_i32(sb1, c70144);
        let sbp = b.ptr_add(o, sb2);
        let sbf0 = b.load_f32(sbp, 0);
        let sb_c0 = b.bitcast_i32(sbf0);
        let sbf1 = b.load_f32(sbp, 256);
        let sb_c1 = b.bitcast_i32(sbf1);
        let s0 = b.mfma_scale_fp8(a8_0, k8_0, zero, sa0, sb_c0);
        let s_frag = b.mfma_scale_fp8(a8_1, k8_1, s0, sa1, sb_c1);
        let reg = b.c_i32(4096 * s);
        let lbase = b.add_i32(lds_dyn, reg);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as i32;
            let lane = b.frag16_lane(s_frag, r);
            let coff = b.c_i32(i_const * 128);
            let addr = b.add_i32(lbase, coff);
            b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
        }
    }

    // ---- stage this iteration's row-major V (128 x 128) into LDS: 256 bytes per lane ----
    let vit = b.mul_i32(it, c16384);
    let vg0 = b.add_i32(tid256, vit);
    let vg1 = b.add_i32(vg0, c36864);
    let vg = b.ptr_add(o, vg1);
    let vl = b.add_i32(tid256, lds_v);
    let mut la = vl;
    for w in 0..64u32 {
        let v = b.load_f32(vg, 4 * w);
        let vi = b.bitcast_i32(v);
        b.store_lds_b32(Opnd::Val(la), Opnd::Val(vi));
        la = b.add_i32(la, c4);
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- pass 1: row max over the 128 keys ----
    let rbase = b.mul_i32(j, c128);
    let mut tmax: Option<ValueId> = None;
    for s in 0..4u32 {
        let sc = b.c_i32((s * 4096) as i32);
        let mut addr = b.add_i32(rbase, sc);
        for _k in 0..32u32 {
            let raw = b.load_lds_b32(Opnd::Val(addr));
            addr = b.add_i32(addr, c4);
            let v = b.bitcast_f32(raw);
            tmax = Some(match tmax {
                None => v,
                Some(m) => b.max_f32(m, v),
            });
        }
    }
    let tmax = tmax.unwrap();
    let m_old = b.bitcast_f32(m_var);
    let m_new = b.max_f32(m_old, tmax);
    let dm = b.sub_f32(m_old, m_new);
    let corr = b.exp_f32(dm);
    let corr_bits = b.bitcast_i32(corr);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));

    // ---- pass 2: exp, row sum and fp8 quantisation, 32 keys at a time ----
    let mut chunk_words: Vec<Vec<ValueId>> = Vec::with_capacity(4);
    let mut rsum: Option<ValueId> = None;
    for s in 0..4u32 {
        let mut exps: Vec<ValueId> = Vec::with_capacity(32);
        let sc2 = b.c_i32((s * 4096) as i32);
        let mut addr2 = b.add_i32(rbase, sc2);
        for _k in 0..32u32 {
            let raw = b.load_lds_b32(Opnd::Val(addr2));
            addr2 = b.add_i32(addr2, c4);
            let v = b.bitcast_f32(raw);
            let sh = b.sub_f32(v, m_new);
            exps.push(b.exp_f32(sh));
        }
        for e in &exps {
            rsum = Some(match rsum {
                None => *e,
                Some(acc) => b.add_f32(acc, *e),
            });
        }
        let mut words: Vec<ValueId> = Vec::with_capacity(8);
        for m in 0..8usize {
            let a = b.mul_f32(exps[4 * m], c16f);
            let bb = b.mul_f32(exps[4 * m + 1], c16f);
            let p0 = b.cvt_pk_fp8_f32(a, bb);
            let c = b.mul_f32(exps[4 * m + 2], c16f);
            let d = b.mul_f32(exps[4 * m + 3], c16f);
            let p1 = b.cvt_pk_fp8_f32(c, d);
            let sh = b.shl_i32(p1, c16i);
            words.push(b.add_i32(p0, sh));
        }
        chunk_words.push(words);
    }
    let rsum = rsum.unwrap();
    let l_old = b.bitcast_f32(l_var);
    let l_scaled = b.mul_f32(l_old, corr);
    let l_new = b.add_f32(l_scaled, rsum);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // Publish the quantised bytes (all four chunks) for the host reference.
    let pb0 = b.mul_i32(it, c4096);
    let pb1 = b.mul_i32(j, c128);
    let pb2 = b.add_i32(pb0, pb1);
    let pb3 = b.add_i32(pb2, c90624);
    let p_ptr = b.ptr_add(o, pb3);
    for s in 0..4usize {
        for m in 0..8usize {
            let v = b.bitcast_f32(chunk_words[s][m]);
            b.store_f32(p_ptr, (32 * s + 4 * m) as u32, v);
        }
    }

    // ---- row corrections for this lane's O elements ----
    let mut corrs: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        corrs.push(b.bitcast_f32(craw));
    }

    // ---- PV: two 64-key halves x four 32-wide output tiles ----
    let mut p8: Vec<ValueId> = Vec::with_capacity(2);
    for hh in 0..2usize {
        let mut sel: Vec<ValueId> = Vec::with_capacity(8);
        for m in 0..8usize {
            let lo = chunk_words[2 * hh][m];
            let hi = chunk_words[2 * hh + 1][m];
            let diff = b.xor_i32(lo, hi);
            let pick = b.mul_i32(diff, pblk);
            sel.push(b.xor_i32(lo, pick));
        }
        p8.push(b.frag8(sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7]));
    }
    for e in 0..4usize {
        let mut scaled: Vec<ValueId> = Vec::with_capacity(16);
        for r in 0..16u8 {
            let lane = b.frag16_lane(o_vars[e], r);
            let lanev = b.bitcast_f32(lane);
            let sc = b.mul_f32(lanev, corrs[r as usize]);
            scaled.push(b.bitcast_i32(sc));
        }
        let mut acc = b.frag16_build(&scaled);
        for hh in 0..2usize {
            let mut vw: Vec<ValueId> = Vec::with_capacity(8);
            for r in 0..4i32 {
                let off = b.c_i32((hh as i32) * 8192 + (e as i32) * 32 + 1024 * r);
                let addr = b.add_i32(tr_base, off);
                let tr = b.load_lds_tr_b8(Opnd::Val(addr));
                vw.push(b.frag16_lane(tr, 0));
                vw.push(b.frag16_lane(tr, 1));
            }
            let v8 = b.frag8(vw[0], vw[1], vw[2], vw[3], vw[4], vw[5], vw[6], vw[7]);
            acc = b.mfma_scale_fp8(p8[hh], v8, acc, c127, c123);
        }
        b.set_var(o_vars[e], acc);
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let m_bits = b.bitcast_i32(m_new);
    let l_bits = b.bitcast_i32(l_new);
    b.set_var(m_var, m_bits);
    b.set_var(l_var, l_bits);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write the 32 x 128 output ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let mut invs: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        invs.push(b.bitcast_f32(craw));
    }
    let p2048 = b.shl_i32(pblk, c11);
    let od0 = b.add_i32(j4, p2048);
    let od1 = b.add_i32(od0, c74240);
    for e in 0..4usize {
        let eo = b.c_i32(128 * e as i32);
        let od2 = b.add_i32(od1, eo);
        let o_ptr = b.ptr_add(o, od2);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as u32;
            let lane = b.frag16_lane(o_vars[e], r);
            let lanev = b.bitcast_f32(lane);
            let res = b.mul_f32(lanev, invs[r as usize]);
            // the global offset field is only 13 bits wide: fold the row offset into the pointer
            let row_off = b.c_i32((i_const * 512) as i32);
            let row_ptr = b.ptr_add(o_ptr, row_off);
            b.store_f32(row_ptr, 0, res);
        }
    }
    let _ = (c32, c256, c1024, c2048);
    b.finish()
}

fn c64_const(b: &mut FnBuilder) -> ValueId {
    b.c_i32(64)
}
/// Head dim 128, KV tile 128 with LDS-staged K and V and 64-key online-softmax substeps.
///
/// Compared with `probe_fp8_fmha_hd128` the score tiles no longer all live in LDS at once: each
/// 128-key iteration copies its K and V tiles into LDS once (K fragments are then read with
/// `ds_read_b128`, V fragments with `ds_read_b64_tr_b8`) and runs two 64-key substeps, each a
/// full online-softmax update (two score tiles, 64-wide softmax, four PV MFMAs). This is the
/// shape that fits eight waves into the 160 KiB LDS budget.
///
/// Input image in `o` (S = 1024 buffer): as `probe_fp8_fmha_hd128`; the quantised P bytes of
/// substep `sub` (0..3) are at `90624 + 2048 sub + 64 row + key`.
pub fn probe_fp8_fmha_hd128b() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_hd128b", 64);
    b.lds(41216);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c7 = b.c_i32(7);
    let c8 = b.c_i32(8);
    let c9 = b.c_i32(9);
    let c11 = b.c_i32(11);
    let c12 = b.c_i32(12);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c16384 = b.c_i32(16384);
    let c36864 = b.c_i32(36864);
    let c69632 = b.c_i32(69632);
    let c70144 = b.c_i32(70144);
    let c74240 = b.c_i32(74240);
    let c90624 = b.c_i32(90624);
    let lds_corr = b.c_i32(8192);
    let lds_k = b.c_i32(8320);
    let lds_v = b.c_i32(24704);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_dyn = b.add_i32(j4, p512);
    let pblk16 = b.shl_i32(pblk, c4);
    let corr_row = b.add_i32(j4, lds_corr);
    let tid4 = b.mul_i32(tid, c4);
    let tid256 = b.shl_i32(tid, c8);
    let zero = b.frag_zero(DType::Frag16);

    let j128 = b.mul_i32(j, c128);
    let p32 = b.shl_i32(pblk, c5);
    let frag_row = b.add_i32(j128, p32);
    let kf_base = b.add_i32(frag_row, lds_k);

    let tid_s1 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c1)));
    let i_idx = b.let_(Op::AndI32(Opnd::Val(tid_s1), Opnd::Val(c7)));
    let tid_s4 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c4)));
    let g2 = b.let_(Op::AndI32(Opnd::Val(tid_s4), Opnd::Val(c1)));
    let h_idx = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1)));
    let pP = b.shl_i32(pblk, c12);
    let iP = b.mul_i32(i_idx, c128);
    let g216 = b.mul_i32(g2, c16i);
    let h8 = b.mul_i32(h_idx, c8);
    let tr0 = b.add_i32(pP, iP);
    let tr1 = b.add_i32(tr0, g216);
    let tr2 = b.add_i32(tr1, h8);
    let tr_base = b.add_i32(tr2, lds_v);

    // Q fragments (two head chunks) and the Q scales, loaded once.
    let q_ptr0 = b.ptr_add(o, frag_row);
    let a8_0 = frag8_from_global(&mut b, q_ptr0);
    let q_off1 = b.add_i32(frag_row, c64);
    let q_ptr1 = b.ptr_add(o, q_off1);
    let a8_1 = frag8_from_global(&mut b, q_ptr1);
    let sa0_off = b.add_i32(tid4, c69632);
    let sa_ptr = b.ptr_add(o, sa0_off);
    let sa0_f = b.load_f32(sa_ptr, 0);
    let sa0 = b.bitcast_i32(sa0_f);
    let sa1_f = b.load_f32(sa_ptr, 256);
    let sa1 = b.bitcast_i32(sa1_f);

    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_vars = [vars[3], vars[4], vars[5], vars[6]];

    // ---- stage K (4 tiles) and V (128 keys) of this iteration into LDS ----
    let it16384 = b.mul_i32(it, c16384);
    let kg0 = b.add_i32(tid256, it16384);
    let kg1 = b.add_i32(kg0, c4096);
    let kg = b.ptr_add(o, kg1);
    let vg1 = b.add_i32(kg0, c36864);
    let vg = b.ptr_add(o, vg1);
    let mut kla = b.add_i32(tid256, lds_k);
    let mut vla = b.add_i32(tid256, lds_v);
    for w in 0..64u32 {
        let kv = b.load_f32(kg, 4 * w);
        let ki = b.bitcast_i32(kv);
        b.store_lds_b32(Opnd::Val(kla), Opnd::Val(ki));
        kla = b.add_i32(kla, c4);
        let vv = b.load_f32(vg, 4 * w);
        let vi = b.bitcast_i32(vv);
        b.store_lds_b32(Opnd::Val(vla), Opnd::Val(vi));
        vla = b.add_i32(vla, c4);
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let it8 = b.mul_i32(it, c2048);
    let it4096 = b.mul_i32(it, c4096);
    let rbase = b.mul_i32(j, c128);
    for hh in 0..2usize {
        // ---- two score tiles of this 64-key substep ----
        for sl in 0..2usize {
            let s = 2 * hh + sl;
            let ko = b.c_i32((4096 * s) as i32);
            let kaddr0 = b.add_i32(kf_base, ko);
            let k_lo0 = b.load_lds_b128(Opnd::Val(kaddr0));
            let kaddr0h = b.add_i32(kaddr0, c16i);
            let k_hi0 = b.load_lds_b128(Opnd::Val(kaddr0h));
            let k8_0 = b.cat_frag8(k_lo0, k_hi0);
            let kaddr1 = b.add_i32(kaddr0, c64);
            let k_lo1 = b.load_lds_b128(Opnd::Val(kaddr1));
            let kaddr1h = b.add_i32(kaddr1, c16i);
            let k_hi1 = b.load_lds_b128(Opnd::Val(kaddr1h));
            let k8_1 = b.cat_frag8(k_lo1, k_hi1);
            let sb_s = b.c_i32((512 * s) as i32);
            let sb0 = b.add_i32(it8, sb_s);
            let sb1 = b.add_i32(sb0, tid4);
            let sb2 = b.add_i32(sb1, c70144);
            let sbp = b.ptr_add(o, sb2);
            let sbf0 = b.load_f32(sbp, 0);
            let sb_c0 = b.bitcast_i32(sbf0);
            let sbf1 = b.load_f32(sbp, 256);
            let sb_c1 = b.bitcast_i32(sbf1);
            let s0 = b.mfma_scale_fp8(a8_0, k8_0, zero, sa0, sb_c0);
            let s_frag = b.mfma_scale_fp8(a8_1, k8_1, s0, sa1, sb_c1);
            let reg = b.c_i32((4096 * sl) as i32);
            let lbase = b.add_i32(lds_dyn, reg);
            for r in 0..16u8 {
                let i_const = ((r / 4) * 8 + (r % 4)) as i32;
                let lane = b.frag16_lane(s_frag, r);
                let coff = b.c_i32(i_const * 128);
                let addr = b.add_i32(lbase, coff);
                b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
            }
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);

        // ---- 64-wide online softmax update (max pass, then exp/sum/quantise per 32 keys) ----
        let mut tmax: Option<ValueId> = None;
        for sl in 0..2u32 {
            let sc = b.c_i32((sl * 4096) as i32);
            let mut addr = b.add_i32(rbase, sc);
            for _k in 0..32u32 {
                let raw = b.load_lds_b32(Opnd::Val(addr));
                addr = b.add_i32(addr, c4);
                let v = b.bitcast_f32(raw);
                tmax = Some(match tmax {
                    None => v,
                    Some(m) => b.max_f32(m, v),
                });
            }
        }
        let tmax = tmax.unwrap();
        let m_old = b.bitcast_f32(m_var);
        let m_new = b.max_f32(m_old, tmax);
        let dm = b.sub_f32(m_old, m_new);
        let corr = b.exp_f32(dm);
        let corr_bits = b.bitcast_i32(corr);
        b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));

        let mut chunk_words: Vec<Vec<ValueId>> = Vec::with_capacity(2);
        let mut rsum: Option<ValueId> = None;
        for sl in 0..2u32 {
            let sc2 = b.c_i32((sl * 4096) as i32);
            let mut addr2 = b.add_i32(rbase, sc2);
            let mut exps: Vec<ValueId> = Vec::with_capacity(32);
            for _k in 0..32u32 {
                let raw = b.load_lds_b32(Opnd::Val(addr2));
                addr2 = b.add_i32(addr2, c4);
                let v = b.bitcast_f32(raw);
                let sh = b.sub_f32(v, m_new);
                exps.push(b.exp_f32(sh));
            }
            for e in &exps {
                rsum = Some(match rsum {
                    None => *e,
                    Some(acc) => b.add_f32(acc, *e),
                });
            }
            let mut words: Vec<ValueId> = Vec::with_capacity(8);
            for m in 0..8usize {
                let a = b.mul_f32(exps[4 * m], c16f);
                let bb = b.mul_f32(exps[4 * m + 1], c16f);
                let p0 = b.cvt_pk_fp8_f32(a, bb);
                let c = b.mul_f32(exps[4 * m + 2], c16f);
                let d = b.mul_f32(exps[4 * m + 3], c16f);
                let p1 = b.cvt_pk_fp8_f32(c, d);
                let sh = b.shl_i32(p1, c16i);
                words.push(b.add_i32(p0, sh));
            }
            chunk_words.push(words);
        }
        let rsum = rsum.unwrap();
        let l_old = b.bitcast_f32(l_var);
        let l_scaled = b.mul_f32(l_old, corr);
        let l_new = b.add_f32(l_scaled, rsum);
        let m_bits = b.bitcast_i32(m_new);
        let l_bits = b.bitcast_i32(l_new);
        b.set_var(m_var, m_bits);
        b.set_var(l_var, l_bits);
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);

        // publish the quantised bytes of substep (2 it + hh) for the host reference
        let pbh = b.c_i32((2048 * hh) as i32);
        let pb0 = b.mul_i32(it, c4096);
        let pb1 = b.add_i32(pb0, pbh);
        let pb2 = b.mul_i32(j, c64);
        let pb3 = b.add_i32(pb1, pb2);
        let pb4 = b.add_i32(pb3, c90624);
        let p_ptr = b.ptr_add(o, pb4);
        for q in 0..2usize {
            for m in 0..8usize {
                let v = b.bitcast_f32(chunk_words[q][m]);
                b.store_f32(p_ptr, (32 * q + 4 * m) as u32, v);
            }
        }
        let _ = it4096;

        // ---- row corrections and the PV step ----
        let mut corrs: Vec<ValueId> = Vec::with_capacity(16);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as i32;
            let ccoff = b.c_i32(i_const * 4);
            let caddr0 = b.add_i32(pblk16, lds_corr);
            let caddr = b.add_i32(caddr0, ccoff);
            let craw = b.load_lds_b32(Opnd::Val(caddr));
            corrs.push(b.bitcast_f32(craw));
        }
        let mut sel: Vec<ValueId> = Vec::with_capacity(8);
        for m in 0..8usize {
            let lo = chunk_words[0][m];
            let hi = chunk_words[1][m];
            let diff = b.xor_i32(lo, hi);
            let pick = b.mul_i32(diff, pblk);
            sel.push(b.xor_i32(lo, pick));
        }
        let p8 = b.frag8(sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7]);
        for e in 0..4usize {
            let mut scaled: Vec<ValueId> = Vec::with_capacity(16);
            for r in 0..16u8 {
                let lane = b.frag16_lane(o_vars[e], r);
                let lanev = b.bitcast_f32(lane);
                let sc = b.mul_f32(lanev, corrs[r as usize]);
                scaled.push(b.bitcast_i32(sc));
            }
            let acc = b.frag16_build(&scaled);
            let mut vw: Vec<ValueId> = Vec::with_capacity(8);
            for r in 0..4i32 {
                let off = b.c_i32((hh as i32) * 8192 + (e as i32) * 32 + 1024 * r);
                let addr = b.add_i32(tr_base, off);
                let tr = b.load_lds_tr_b8(Opnd::Val(addr));
                vw.push(b.frag16_lane(tr, 0));
                vw.push(b.frag16_lane(tr, 1));
            }
            let v8 = b.frag8(vw[0], vw[1], vw[2], vw[3], vw[4], vw[5], vw[6], vw[7]);
            let o_new = b.mfma_scale_fp8(p8, v8, acc, c127, c123);
            b.set_var(o_vars[e], o_new);
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write the 32 x 128 output ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let mut invs: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        invs.push(b.bitcast_f32(craw));
    }
    let p2048 = b.shl_i32(pblk, c11);
    let od0 = b.add_i32(j4, p2048);
    let od1 = b.add_i32(od0, c74240);
    for e in 0..4usize {
        let eo = b.c_i32(128 * e as i32);
        let od2 = b.add_i32(od1, eo);
        let o_ptr = b.ptr_add(o, od2);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as u32;
            let lane = b.frag16_lane(o_vars[e], r);
            let lanev = b.bitcast_f32(lane);
            let res = b.mul_f32(lanev, invs[r as usize]);
            let row_off = b.c_i32((i_const * 512) as i32);
            let row_ptr = b.ptr_add(o_ptr, row_off);
            b.store_f32(row_ptr, 0, res);
        }
    }
    b.finish()
}

/// 512-thread block of the LDS-staged hd128 kernel: eight query tiles (256 rows), shared K and V in
/// LDS copied by role A (K, waves 0-3) and role B (V, waves 4-7), each role running its own steady
/// loop with the same barrier schedule.
pub fn probe_fp8_fmha_hd128c() -> Function {
    let mut b = FnBuilder::new("probe_fp8_fmha_hd128c", 512);
    b.lds(99456);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid_full = b.let_(Op::WorkitemId);
    let c63w = b.c_i32(63);
    let c6w = b.c_i32(6);
    let tid = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c63w)));
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid_full), Opnd::Val(c6w)));

    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c7 = b.c_i32(7);
    let c8 = b.c_i32(8);
    let c9 = b.c_i32(9);
    let c11 = b.c_i32(11);
    let c12 = b.c_i32(12);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c16384 = b.c_i32(16384);
    let c65536 = b.c_i32(65536);
    let c32768 = b.c_i32(32768);
    let c69632 = b.c_i32(98304);
    let c70144 = b.c_i32(98816);
    let c74240 = b.c_i32(102912);
    let c90624 = b.c_i32(233984);
    let c8320 = b.c_i32(8320);
    let c255 = b.c_i32(255);
    let c6w2 = b.c_i32(6);
    let c14w = b.c_i32(14);
    let c13w = b.c_i32(13);
    let c12w = b.c_i32(12);
    let wbase = b.mul_i32(wave, c8320);
    let wq = b.shl_i32(wave, c12w);
    let wout = b.shl_i32(wave, c14w);
    let wpb = b.shl_i32(wave, c13w);
    let lds_corr0 = b.c_i32(8192);
    let lds_corr = b.add_i32(lds_corr0, wbase);
    let lds_k = b.c_i32(66560);
    let lds_v = b.c_i32(82944);
    let is_a = b.cmp_lt_i32(wave, c4);
    let rt = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c255)));
    let rt64 = b.shl_i32(rt, c6w2);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let lds_loc = b.add_i32(j4, p512);
    let lds_dyn = b.add_i32(lds_loc, wbase);
    let pblk16 = b.shl_i32(pblk, c4);
    let corr_row = b.add_i32(j4, lds_corr);
    let tid4 = b.mul_i32(tid, c4);
    let tid256 = b.shl_i32(tid, c8);

    let j128 = b.mul_i32(j, c128);
    let p32 = b.shl_i32(pblk, c5);
    let frag_row = b.add_i32(j128, p32);
    let kf_base = b.add_i32(frag_row, lds_k);

    let tid_s1 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c1)));
    let i_idx = b.let_(Op::AndI32(Opnd::Val(tid_s1), Opnd::Val(c7)));
    let tid_s4 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c4)));
    let g2 = b.let_(Op::AndI32(Opnd::Val(tid_s4), Opnd::Val(c1)));
    let h_idx = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1)));
    let pP = b.shl_i32(pblk, c12);
    let iP = b.mul_i32(i_idx, c128);
    let g216 = b.mul_i32(g2, c16i);
    let h8 = b.mul_i32(h_idx, c8);
    let tr0 = b.add_i32(pP, iP);
    let tr1 = b.add_i32(tr0, g216);
    let tr2 = b.add_i32(tr1, h8);
    let tr_base = b.add_i32(tr2, lds_v);

    // Q fragments (two head chunks) and the Q scales, loaded once.
    let q_off0 = b.add_i32(frag_row, wq);
    let q_ptr0 = b.ptr_add(o, q_off0);
    let a8_0 = frag8_from_global(&mut b, q_ptr0);
    let q_off1 = b.add_i32(q_off0, c64);
    let q_ptr1 = b.ptr_add(o, q_off1);
    let a8_1 = frag8_from_global(&mut b, q_ptr1);
    let sa0_off = b.add_i32(tid4, c69632);
    let sa_ptr = b.ptr_add(o, sa0_off);
    let sa0_f = b.load_f32(sa_ptr, 0);
    let sa0 = b.bitcast_i32(sa0_f);
    let sa1_f = b.load_f32(sa_ptr, 256);
    let sa1 = b.bitcast_i32(sa1_f);

    b.begin_if(is_a);
    {
    let zero = b.frag_zero(DType::Frag16);
    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_vars = [vars[3], vars[4], vars[5], vars[6]];

    let it16384 = b.mul_i32(it, c16384);    // role A (waves 0-3) copies this iteration's K tiles: 64 bytes per thread
    {
        let kg0 = b.add_i32(rt64, it16384);
        let kg1 = b.add_i32(kg0, c32768);
        let kg = b.ptr_add(o, kg1);
        let mut kla = b.add_i32(rt64, lds_k);
        for w in 0..16u32 {
            let kv = b.load_f32(kg, 4 * w);
            let ki = b.bitcast_i32(kv);
            b.store_lds_b32(Opnd::Val(kla), Opnd::Val(ki));
            kla = b.add_i32(kla, c4);
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }
    let it8 = b.mul_i32(it, c2048);
    let it4096 = b.mul_i32(it, c4096);
    let rbase0 = b.mul_i32(j, c128);
    let rbase = b.add_i32(rbase0, wbase);
    for hh in 0..2usize {
        // ---- two score tiles of this 64-key substep ----
        for sl in 0..2usize {
            let s = 2 * hh + sl;
            let ko = b.c_i32((4096 * s) as i32);
            let kaddr0 = b.add_i32(kf_base, ko);
            let k_lo0 = b.load_lds_b128(Opnd::Val(kaddr0));
            let kaddr0h = b.add_i32(kaddr0, c16i);
            let k_hi0 = b.load_lds_b128(Opnd::Val(kaddr0h));
            let k8_0 = b.cat_frag8(k_lo0, k_hi0);
            let kaddr1 = b.add_i32(kaddr0, c64);
            let k_lo1 = b.load_lds_b128(Opnd::Val(kaddr1));
            let kaddr1h = b.add_i32(kaddr1, c16i);
            let k_hi1 = b.load_lds_b128(Opnd::Val(kaddr1h));
            let k8_1 = b.cat_frag8(k_lo1, k_hi1);
            let sb_s = b.c_i32((512 * s) as i32);
            let sb0 = b.add_i32(it8, sb_s);
            let sb1 = b.add_i32(sb0, tid4);
            let sb2 = b.add_i32(sb1, c70144);
            let sbp = b.ptr_add(o, sb2);
            let sbf0 = b.load_f32(sbp, 0);
            let sb_c0 = b.bitcast_i32(sbf0);
            let sbf1 = b.load_f32(sbp, 256);
            let sb_c1 = b.bitcast_i32(sbf1);
            let zero_s = b.frag_zero(DType::Frag16);
            let s0 = b.mfma_scale_fp8(a8_0, k8_0, zero_s, sa0, sb_c0);
            let s_frag = b.mfma_scale_fp8(a8_1, k8_1, s0, sa1, sb_c1);
            let reg = b.c_i32((4096 * sl) as i32);
            let lbase = b.add_i32(lds_dyn, reg);
            for r in 0..16u8 {
                let i_const = ((r / 4) * 8 + (r % 4)) as i32;
                let lane = b.frag16_lane(s_frag, r);
                let coff = b.c_i32(i_const * 128);
                let addr = b.add_i32(lbase, coff);
                b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
            }
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);

        // ---- 64-wide online softmax update (max pass, then exp/sum/quantise per 32 keys) ----
        let mut tmax: Option<ValueId> = None;
        for sl in 0..2u32 {
            let sc = b.c_i32((sl * 4096) as i32);
            let mut addr = b.add_i32(rbase, sc);
            for _k in 0..32u32 {
                let raw = b.load_lds_b32(Opnd::Val(addr));
                addr = b.add_i32(addr, c4);
                let v = b.bitcast_f32(raw);
                tmax = Some(match tmax {
                    None => v,
                    Some(m) => b.max_f32(m, v),
                });
            }
        }
        let tmax = tmax.unwrap();
        let m_old = b.bitcast_f32(m_var);
        let m_new = b.max_f32(m_old, tmax);
        let dm = b.sub_f32(m_old, m_new);
        let corr = b.exp_f32(dm);
        let corr_bits = b.bitcast_i32(corr);
        b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));

        let pbh = b.c_i32((2048 * hh) as i32);
        let pb0 = b.mul_i32(it, c4096);
        let pb1 = b.add_i32(pb0, pbh);
        let pb2 = b.mul_i32(j, c64);
        let pb3 = b.add_i32(pb1, pb2);
        let pb3b = b.add_i32(pb3, wpb);
        let pb4 = b.add_i32(pb3b, c90624);
        let p_ptr_pub = b.ptr_add(o, pb4);
        let mut chunk_words: Vec<Vec<ValueId>> = Vec::with_capacity(2);
        let mut rsum: Option<ValueId> = None;
        for sl in 0..2u32 {
            let sc2 = b.c_i32((sl * 4096) as i32);
            let mut addr2 = b.add_i32(rbase, sc2);
            let mut exps: Vec<ValueId> = Vec::with_capacity(32);
            for _k in 0..32u32 {
                let raw = b.load_lds_b32(Opnd::Val(addr2));
                addr2 = b.add_i32(addr2, c4);
                let v = b.bitcast_f32(raw);
                let sh = b.sub_f32(v, m_new);
                exps.push(b.exp_f32(sh));
            }
            for e in &exps {
                rsum = Some(match rsum {
                    None => *e,
                    Some(acc) => b.add_f32(acc, *e),
                });
            }
            let mut words: Vec<ValueId> = Vec::with_capacity(8);
            for m in 0..8usize {
                let a = b.mul_f32(exps[4 * m], c16f);
                let bb = b.mul_f32(exps[4 * m + 1], c16f);
                let p0 = b.cvt_pk_fp8_f32(a, bb);
                let c = b.mul_f32(exps[4 * m + 2], c16f);
                let d = b.mul_f32(exps[4 * m + 3], c16f);
                let p1 = b.cvt_pk_fp8_f32(c, d);
                let sh = b.shl_i32(p1, c16i);
                words.push(b.add_i32(p0, sh));
            }
            // publish this chunk's quantised bytes right away and keep only what PV needs
            for m in 0..8usize {
                let v = b.bitcast_f32(words[m]);
                b.store_f32(p_ptr_pub, (32 * sl as usize + 4 * m) as u32, v);
            }
            if sl == 0 {
                chunk_words.push(words);
            } else {
                let mut selw: Vec<ValueId> = Vec::with_capacity(8);
                for m in 0..8usize {
                    let lo = chunk_words[0][m];
                    let hi = words[m];
                    let diff = b.xor_i32(lo, hi);
                    let pick = b.mul_i32(diff, pblk);
                    selw.push(b.xor_i32(lo, pick));
                }
                chunk_words.clear();
                chunk_words.push(selw);
            }
        }
        let rsum = rsum.unwrap();
        let l_old = b.bitcast_f32(l_var);
        let l_scaled = b.mul_f32(l_old, corr);
        let l_new = b.add_f32(l_scaled, rsum);
        let m_bits = b.bitcast_i32(m_new);
        let l_bits = b.bitcast_i32(l_new);
        b.set_var(m_var, m_bits);
        b.set_var(l_var, l_bits);
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);

        // ---- row corrections and the PV step ----
        let sel = chunk_words[0].clone();
        let p8 = b.frag8(sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7]);
        for e in 0..4usize {
            let mut scaled: Vec<ValueId> = Vec::with_capacity(16);
            for r in 0..16u8 {
                let lane = b.frag16_lane(o_vars[e], r);
                let lanev = b.bitcast_f32(lane);
                let i_const = ((r / 4) * 8 + (r % 4)) as i32;
                let ccoff = b.c_i32(i_const * 4);
                let caddr0 = b.add_i32(pblk16, lds_corr);
                let caddr = b.add_i32(caddr0, ccoff);
                let craw = b.load_lds_b32(Opnd::Val(caddr));
                let cf = b.bitcast_f32(craw);
                let sc = b.mul_f32(lanev, cf);
                scaled.push(b.bitcast_i32(sc));
            }
            let acc = b.frag16_build(&scaled);
            let mut vw: Vec<ValueId> = Vec::with_capacity(8);
            for r in 0..4i32 {
                let off = b.c_i32((hh as i32) * 8192 + (e as i32) * 32 + 1024 * r);
                let addr = b.add_i32(tr_base, off);
                let tr = b.load_lds_tr_b8(Opnd::Val(addr));
                vw.push(b.frag16_lane(tr, 0));
                vw.push(b.frag16_lane(tr, 1));
            }
            let v8 = b.frag8(vw[0], vw[1], vw[2], vw[3], vw[4], vw[5], vw[6], vw[7]);
            let o_new = b.mfma_scale_fp8(p8, v8, acc, c127, c123);
            b.set_var(o_vars[e], o_new);
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write the 32 x 128 output ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let mut invs: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        invs.push(b.bitcast_f32(craw));
    }
    let p2048 = b.shl_i32(pblk, c11);
    let od0 = b.add_i32(j4, p2048);
    let od00 = b.add_i32(od0, wout);
    let od1 = b.add_i32(od00, c74240);
    for e in 0..4usize {
        let eo = b.c_i32(128 * e as i32);
        let od2 = b.add_i32(od1, eo);
        let o_ptr = b.ptr_add(o, od2);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as u32;
            let lane = b.frag16_lane(o_vars[e], r);
            let lanev = b.bitcast_f32(lane);
            let res = b.mul_f32(lanev, invs[r as usize]);
            let row_off = b.c_i32((i_const * 512) as i32);
            let row_ptr = b.ptr_add(o_ptr, row_off);
            b.store_f32(row_ptr, 0, res);
        }
    }
    }
    b.else_();
    {
    let zero = b.frag_zero(DType::Frag16);
    let m_init = b.bitcast_i32(neg_big);
    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_vars = [vars[3], vars[4], vars[5], vars[6]];

    let it16384 = b.mul_i32(it, c16384);    // role B (waves 4-7) copies this iteration's V tile: 64 bytes per thread
    {
        let vg0 = b.add_i32(rt64, it16384);
        let vg1 = b.add_i32(vg0, c65536);
        let vg = b.ptr_add(o, vg1);
        let mut vla = b.add_i32(rt64, lds_v);
        for w in 0..16u32 {
            let vv = b.load_f32(vg, 4 * w);
            let vi = b.bitcast_i32(vv);
            b.store_lds_b32(Opnd::Val(vla), Opnd::Val(vi));
            vla = b.add_i32(vla, c4);
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }
    let it8 = b.mul_i32(it, c2048);
    let it4096 = b.mul_i32(it, c4096);
    let rbase0 = b.mul_i32(j, c128);
    let rbase = b.add_i32(rbase0, wbase);
    for hh in 0..2usize {
        // ---- two score tiles of this 64-key substep ----
        for sl in 0..2usize {
            let s = 2 * hh + sl;
            let ko = b.c_i32((4096 * s) as i32);
            let kaddr0 = b.add_i32(kf_base, ko);
            let k_lo0 = b.load_lds_b128(Opnd::Val(kaddr0));
            let kaddr0h = b.add_i32(kaddr0, c16i);
            let k_hi0 = b.load_lds_b128(Opnd::Val(kaddr0h));
            let k8_0 = b.cat_frag8(k_lo0, k_hi0);
            let kaddr1 = b.add_i32(kaddr0, c64);
            let k_lo1 = b.load_lds_b128(Opnd::Val(kaddr1));
            let kaddr1h = b.add_i32(kaddr1, c16i);
            let k_hi1 = b.load_lds_b128(Opnd::Val(kaddr1h));
            let k8_1 = b.cat_frag8(k_lo1, k_hi1);
            let sb_s = b.c_i32((512 * s) as i32);
            let sb0 = b.add_i32(it8, sb_s);
            let sb1 = b.add_i32(sb0, tid4);
            let sb2 = b.add_i32(sb1, c70144);
            let sbp = b.ptr_add(o, sb2);
            let sbf0 = b.load_f32(sbp, 0);
            let sb_c0 = b.bitcast_i32(sbf0);
            let sbf1 = b.load_f32(sbp, 256);
            let sb_c1 = b.bitcast_i32(sbf1);
            let zero_s = b.frag_zero(DType::Frag16);
            let s0 = b.mfma_scale_fp8(a8_0, k8_0, zero_s, sa0, sb_c0);
            let s_frag = b.mfma_scale_fp8(a8_1, k8_1, s0, sa1, sb_c1);
            let reg = b.c_i32((4096 * sl) as i32);
            let lbase = b.add_i32(lds_dyn, reg);
            for r in 0..16u8 {
                let i_const = ((r / 4) * 8 + (r % 4)) as i32;
                let lane = b.frag16_lane(s_frag, r);
                let coff = b.c_i32(i_const * 128);
                let addr = b.add_i32(lbase, coff);
                b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
            }
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);

        // ---- 64-wide online softmax update (max pass, then exp/sum/quantise per 32 keys) ----
        let mut tmax: Option<ValueId> = None;
        for sl in 0..2u32 {
            let sc = b.c_i32((sl * 4096) as i32);
            let mut addr = b.add_i32(rbase, sc);
            for _k in 0..32u32 {
                let raw = b.load_lds_b32(Opnd::Val(addr));
                addr = b.add_i32(addr, c4);
                let v = b.bitcast_f32(raw);
                tmax = Some(match tmax {
                    None => v,
                    Some(m) => b.max_f32(m, v),
                });
            }
        }
        let tmax = tmax.unwrap();
        let m_old = b.bitcast_f32(m_var);
        let m_new = b.max_f32(m_old, tmax);
        let dm = b.sub_f32(m_old, m_new);
        let corr = b.exp_f32(dm);
        let corr_bits = b.bitcast_i32(corr);
        b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(corr_bits));

        let pbh = b.c_i32((2048 * hh) as i32);
        let pb0 = b.mul_i32(it, c4096);
        let pb1 = b.add_i32(pb0, pbh);
        let pb2 = b.mul_i32(j, c64);
        let pb3 = b.add_i32(pb1, pb2);
        let pb3b = b.add_i32(pb3, wpb);
        let pb4 = b.add_i32(pb3b, c90624);
        let p_ptr_pub = b.ptr_add(o, pb4);
        let mut chunk_words: Vec<Vec<ValueId>> = Vec::with_capacity(2);
        let mut rsum: Option<ValueId> = None;
        for sl in 0..2u32 {
            let sc2 = b.c_i32((sl * 4096) as i32);
            let mut addr2 = b.add_i32(rbase, sc2);
            let mut exps: Vec<ValueId> = Vec::with_capacity(32);
            for _k in 0..32u32 {
                let raw = b.load_lds_b32(Opnd::Val(addr2));
                addr2 = b.add_i32(addr2, c4);
                let v = b.bitcast_f32(raw);
                let sh = b.sub_f32(v, m_new);
                exps.push(b.exp_f32(sh));
            }
            for e in &exps {
                rsum = Some(match rsum {
                    None => *e,
                    Some(acc) => b.add_f32(acc, *e),
                });
            }
            let mut words: Vec<ValueId> = Vec::with_capacity(8);
            for m in 0..8usize {
                let a = b.mul_f32(exps[4 * m], c16f);
                let bb = b.mul_f32(exps[4 * m + 1], c16f);
                let p0 = b.cvt_pk_fp8_f32(a, bb);
                let c = b.mul_f32(exps[4 * m + 2], c16f);
                let d = b.mul_f32(exps[4 * m + 3], c16f);
                let p1 = b.cvt_pk_fp8_f32(c, d);
                let sh = b.shl_i32(p1, c16i);
                words.push(b.add_i32(p0, sh));
            }
            // publish this chunk's quantised bytes right away and keep only what PV needs
            for m in 0..8usize {
                let v = b.bitcast_f32(words[m]);
                b.store_f32(p_ptr_pub, (32 * sl as usize + 4 * m) as u32, v);
            }
            if sl == 0 {
                chunk_words.push(words);
            } else {
                let mut selw: Vec<ValueId> = Vec::with_capacity(8);
                for m in 0..8usize {
                    let lo = chunk_words[0][m];
                    let hi = words[m];
                    let diff = b.xor_i32(lo, hi);
                    let pick = b.mul_i32(diff, pblk);
                    selw.push(b.xor_i32(lo, pick));
                }
                chunk_words.clear();
                chunk_words.push(selw);
            }
        }
        let rsum = rsum.unwrap();
        let l_old = b.bitcast_f32(l_var);
        let l_scaled = b.mul_f32(l_old, corr);
        let l_new = b.add_f32(l_scaled, rsum);
        let m_bits = b.bitcast_i32(m_new);
        let l_bits = b.bitcast_i32(l_new);
        b.set_var(m_var, m_bits);
        b.set_var(l_var, l_bits);
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);

        // ---- row corrections and the PV step ----
        let sel = chunk_words[0].clone();
        let p8 = b.frag8(sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7]);
        for e in 0..4usize {
            let mut scaled: Vec<ValueId> = Vec::with_capacity(16);
            for r in 0..16u8 {
                let lane = b.frag16_lane(o_vars[e], r);
                let lanev = b.bitcast_f32(lane);
                let i_const = ((r / 4) * 8 + (r % 4)) as i32;
                let ccoff = b.c_i32(i_const * 4);
                let caddr0 = b.add_i32(pblk16, lds_corr);
                let caddr = b.add_i32(caddr0, ccoff);
                let craw = b.load_lds_b32(Opnd::Val(caddr));
                let cf = b.bitcast_f32(craw);
                let sc = b.mul_f32(lanev, cf);
                scaled.push(b.bitcast_i32(sc));
            }
            let acc = b.frag16_build(&scaled);
            let mut vw: Vec<ValueId> = Vec::with_capacity(8);
            for r in 0..4i32 {
                let off = b.c_i32((hh as i32) * 8192 + (e as i32) * 32 + 1024 * r);
                let addr = b.add_i32(tr_base, off);
                let tr = b.load_lds_tr_b8(Opnd::Val(addr));
                vw.push(b.frag16_lane(tr, 0));
                vw.push(b.frag16_lane(tr, 1));
            }
            let v8 = b.frag8(vw[0], vw[1], vw[2], vw[3], vw[4], vw[5], vw[6], vw[7]);
            let o_new = b.mfma_scale_fp8(p8, v8, acc, c127, c123);
            b.set_var(o_vars[e], o_new);
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, c2);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: divide by l and write the 32 x 128 output ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let inv_bits = b.bitcast_i32(inv_l);
    b.store_lds_b32(Opnd::Val(corr_row), Opnd::Val(inv_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let mut invs: Vec<ValueId> = Vec::with_capacity(16);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as i32;
        let ccoff = b.c_i32(i_const * 4);
        let caddr0 = b.add_i32(pblk16, lds_corr);
        let caddr = b.add_i32(caddr0, ccoff);
        let craw = b.load_lds_b32(Opnd::Val(caddr));
        invs.push(b.bitcast_f32(craw));
    }
    let p2048 = b.shl_i32(pblk, c11);
    let od0 = b.add_i32(j4, p2048);
    let od00 = b.add_i32(od0, wout);
    let od1 = b.add_i32(od00, c74240);
    for e in 0..4usize {
        let eo = b.c_i32(128 * e as i32);
        let od2 = b.add_i32(od1, eo);
        let o_ptr = b.ptr_add(o, od2);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as u32;
            let lane = b.frag16_lane(o_vars[e], r);
            let lanev = b.bitcast_f32(lane);
            let res = b.mul_f32(lanev, invs[r as usize]);
            let row_off = b.c_i32((i_const * 512) as i32);
            let row_ptr = b.ptr_add(o_ptr, row_off);
            b.store_f32(row_ptr, 0, res);
        }
    }
    }
    b.end_if();
    b.finish()
}
/// Block-scaled fp8 matmul with real 1x32 E8M0 scales (the reference kernel's scale recipe).
///
/// The scaled MFMA takes one scale byte per lane and uses lane `row + 32 * (byte / 16)` for the
/// data byte `byte` of any lane group, so one scale block of 32 logical K values occupies bytes
/// 0..15 of *both* lane groups (block 0) or bytes 16..31 of both (block 1). The fragment for lane
/// group `g` is therefore: bytes 0..15 = K `16 g .. 16 g + 15`, bytes 16..31 = K `32 + 16 g ..`;
/// lane group `g` supplies the scale of block `g` (`dword >> 8 g`, low byte, op_sel 0).
///
/// Image in `o`: A (32 x 64, pitch 64) at 0, B^T (32 x 64, pitch 64) at 2048, one E8M0 dword per
/// A row at 4096 (bytes = scales of K blocks 0, 1, ...) and per B row at 4352; D (32 x 32 f32) at
/// 4608 in the usual MFMA output layout.
pub fn probe_fp8_matmul_blk() -> Function {
    let mut b = FnBuilder::new("probe_fp8_matmul_blk", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c6 = b.c_i32(6);
    let c9 = b.c_i32(9);
    let c31 = b.c_i32(31);
    let c2048 = b.c_i32(2048);
    let c4096 = b.c_i32(4096);
    let c4352 = b.c_i32(4352);
    let c4608 = b.c_i32(4608);

    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let g = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let row64 = b.shl_i32(j, c6);
    let g16 = b.shl_i32(g, c4);
    let frag_off = b.add_i32(row64, g16);
    let a_base = b.ptr_add(o, frag_off);
    let bt_off = b.add_i32(frag_off, c2048);
    let bt_base = b.ptr_add(o, bt_off);

    let mut aw: Vec<ValueId> = Vec::with_capacity(8);
    let mut bw: Vec<ValueId> = Vec::with_capacity(8);
    for w in 0..8u32 {
        let off = if w < 4 { 4 * w } else { 32 + 4 * (w - 4) };
        let av = b.load_f32(a_base, off);
        aw.push(b.bitcast_i32(av));
        let bv = b.load_f32(bt_base, off);
        bw.push(b.bitcast_i32(bv));
    }
    let a8 = b.frag8(aw[0], aw[1], aw[2], aw[3], aw[4], aw[5], aw[6], aw[7]);
    let b8 = b.frag8(bw[0], bw[1], bw[2], bw[3], bw[4], bw[5], bw[6], bw[7]);

    // scale dwords of the lane's row, shifted so that byte 0 is block g
    let j4 = b.mul_i32(j, c4);
    let sa_off = b.add_i32(j4, c4096);
    let sa_ptr = b.ptr_add(o, sa_off);
    let sa_raw = b.load_f32(sa_ptr, 0);
    let sa_i = b.bitcast_i32(sa_raw);
    let sb_off = b.add_i32(j4, c4352);
    let sb_ptr = b.ptr_add(o, sb_off);
    let sb_raw = b.load_f32(sb_ptr, 0);
    let sb_i = b.bitcast_i32(sb_raw);
    let g8 = b.shl_i32(g, c3);
    let sa = b.let_(Op::LshrI32(Opnd::Val(sa_i), Opnd::Val(g8)));
    let sb = b.let_(Op::LshrI32(Opnd::Val(sb_i), Opnd::Val(g8)));

    let zero = b.frag_zero(DType::Frag16);
    let d = b.mfma_scale_fp8(a8, b8, zero, sa, sb);

    let p512 = b.shl_i32(g, c9);
    let lane_off = b.add_i32(j4, p512);
    let dyn_off = b.add_i32(lane_off, c4608);
    let dptr = b.ptr_add(o, dyn_off);
    for r in 0..16u8 {
        let i_const = ((r / 4) * 8 + (r % 4)) as u32;
        let lane = b.frag16_lane(d, r);
        let v = b.bitcast_f32(lane);
        b.store_f32(dptr, i_const * 128, v);
    }
    b.finish()
}
/// Values shared by the two role loops of `fmha_fwd_hd128_mxfp8`.
struct FmhaCtx {
    ptr_o: ValueId,
    ptr_k: ValueId,
    ptr_v: ValueId,
    ptr_kd: ValueId,
    k_seqs: ValueId,
    v_seqs: ValueId,
    q_heads: ValueId,
    o_seqs: ValueId,
    o_hs: ValueId,
    o_bs: ValueId,
    bidx: ValueId,
    hidx: ValueId,
    row0: ValueId,
    wave: ValueId,
    j: ValueId,
    pblk: ValueId,
    pblk16: ValueId,
    j4: ValueId,
    g8: ValueId,
    lds_dyn: ValueId,
    lds_corr: ValueId,
    corr_row: ValueId,
    lds_k: ValueId,
    lds_v: ValueId,
    kbase_off: ValueId,
    vbase_off: ValueId,
    kd_base: ValueId,
    rt_row: ValueId,
    rt_half64: ValueId,
    rt64: ValueId,
    tr_base: ValueId,
    kf_base: ValueId,
    a8_0: ValueId,
    a8_1: ValueId,
    sa0: ValueId,
    sa1: ValueId,
    vd_raw: ValueId,
    nkv: ValueId,
}

/// One role's steady loop plus the epilogue of `fmha_fwd_hd128_mxfp8`. Role A copies the K tile
/// of each iteration into LDS, role B the V tile; everything else (and the barrier schedule) is
/// identical.
fn emit_fmha_role(b: &mut FnBuilder, cx: &FmhaCtx, role_a: bool) {
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c7 = b.c_i32(7);
    let c11 = b.c_i32(11);
    let c16i = b.c_i32(16);
    let c32v = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let c128 = b.c_i32(128);
    let c4096 = b.c_i32(4096);
    let c32v = b.c_i32(32);
    let neg_big = b.c_f32(-1.0e30);
    let c16f = b.c_f32(16.0);
    let zero = b.frag_zero(DType::Frag16);
    let m_init = b.bitcast_i32(neg_big);

    let vars = b.begin_loop(&[
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
    ]);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_vars = [vars[3], vars[4], vars[5], vars[6]];

    // ---- stage this iteration's K (role A) or V (role B) tile into LDS ----
    let key0 = b.shl_i32(it, c7);
    let (seqs, base_off, ptr, lds_base) = if role_a {
        (cx.k_seqs, cx.kbase_off, cx.ptr_k, cx.lds_k)
    } else {
        (cx.v_seqs, cx.vbase_off, cx.ptr_v, cx.lds_v)
    };
    let srow = b.add_i32(key0, cx.rt_row);
    let soff0 = b.mul_i32(srow, seqs);
    let soff1 = b.add_i32(soff0, base_off);
    let soff = b.add_i32(soff1, cx.rt_half64);
    let sp = b.ptr_add(ptr, soff);
    let mut sla = b.add_i32(lds_base, cx.rt64);
    for w in 0..16u32 {
        let v = b.load_f32(sp, 4 * w);
        let vi = b.bitcast_i32(v);
        b.store_lds_b32(Opnd::Val(sla), Opnd::Val(vi));
        sla = b.add_i32(sla, c4);
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    let hq4 = b.mul_i32(cx.q_heads, c4);
    let rbase0 = b.mul_i32(cx.j, c128);
    let wbase_v = b_wbase(b, cx);
    let rbase = b.add_i32(rbase0, wbase_v);
    for hh in 0..2usize {
        // ---- two 32-key score tiles of this 64-key substep ----
        for sl in 0..2usize {
            let s = 2 * hh + sl;
            let ko = b.c_i32((4096 * s) as i32);
            let kaddr0 = b.add_i32(cx.kf_base, ko);
            let k_lo0 = b.load_lds_b128(Opnd::Val(kaddr0));
            let kaddr0h = b.add_i32(kaddr0, c32v);
            let k_hi0 = b.load_lds_b128(Opnd::Val(kaddr0h));
            let k8_0 = b.cat_frag8(k_lo0, k_hi0);
            let kaddr1 = b.add_i32(kaddr0, c64);
            let k_lo1 = b.load_lds_b128(Opnd::Val(kaddr1));
            let kaddr1h = b.add_i32(kaddr1, c32v);
            let k_hi1 = b.load_lds_b128(Opnd::Val(kaddr1h));
            let k8_1 = b.cat_frag8(k_lo1, k_hi1);
            // K scale dword of key (128 it + 32 s + j)
            let kkey0 = b.add_i32(key0, cx.j);
            let kcs = b.c_i32(32 * s as i32);
            let kkey = b.add_i32(kkey0, kcs);
            let kd_row = b.mul_i32(kkey, hq4);
            let kd_off = b.add_i32(kd_row, cx.kd_base);
            let kd_ptr = b.ptr_add(cx.ptr_kd, kd_off);
            let kd_raw = b.load_f32(kd_ptr, 0);
            let kd = b.bitcast_i32(kd_raw);
            let sb0 = b.let_(Op::LshrI32(Opnd::Val(kd), Opnd::Val(cx.g8)));
            let sh16 = b.add_i32(cx.g8, c16i);
            let sb1 = b.let_(Op::LshrI32(Opnd::Val(kd), Opnd::Val(sh16)));
            let zero_s = b.frag_zero(DType::Frag16);
            let s0 = b.mfma_scale_fp8(cx.a8_0, k8_0, zero_s, cx.sa0, sb0);
            let s_frag = b.mfma_scale_fp8(cx.a8_1, k8_1, s0, cx.sa1, sb1);
            let reg = b.c_i32((4096 * sl) as i32);
            let lbase = b.add_i32(cx.lds_dyn, reg);
            for r in 0..16u8 {
                let i_const = ((r / 4) * 8 + (r % 4)) as i32;
                let lane = b.frag16_lane(s_frag, r);
                let coff = b.c_i32(i_const * 128);
                let addr = b.add_i32(lbase, coff);
                b.store_lds_b32(Opnd::Val(addr), Opnd::Val(lane));
            }
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);

        // ---- 64-wide online softmax (base 2): max pass, then exp2/sum/quantise per 32 keys ----
        let mut tmax: Option<ValueId> = None;
        for sl in 0..2u32 {
            let sc = b.c_i32((sl * 4096) as i32);
            let mut addr = b.add_i32(rbase, sc);
            for _k in 0..32u32 {
                let raw = b.load_lds_b32(Opnd::Val(addr));
                addr = b.add_i32(addr, c4);
                let v = b.bitcast_f32(raw);
                tmax = Some(match tmax {
                    None => v,
                    Some(m) => b.max_f32(m, v),
                });
            }
        }
        let tmax = tmax.unwrap();
        let m_old = b.bitcast_f32(m_var);
        let m_new = b.max_f32(m_old, tmax);
        let dm = b.sub_f32(m_old, m_new);
        let corr = b.exp2_f32(dm);
        let corr_bits = b.bitcast_i32(corr);
        b.store_lds_b32(Opnd::Val(cx.corr_row), Opnd::Val(corr_bits));

        let mut chunk_words: Vec<Vec<ValueId>> = Vec::with_capacity(2);
        let mut rsum: Option<ValueId> = None;
        for sl in 0..2u32 {
            let sc2 = b.c_i32((sl * 4096) as i32);
            let mut addr2 = b.add_i32(rbase, sc2);
            let mut exps: Vec<ValueId> = Vec::with_capacity(32);
            for _k in 0..32u32 {
                let raw = b.load_lds_b32(Opnd::Val(addr2));
                addr2 = b.add_i32(addr2, c4);
                let v = b.bitcast_f32(raw);
                let sh = b.sub_f32(v, m_new);
                exps.push(b.exp2_f32(sh));
            }
            for e in &exps {
                rsum = Some(match rsum {
                    None => *e,
                    Some(acc) => b.add_f32(acc, *e),
                });
            }
            let mut words: Vec<ValueId> = Vec::with_capacity(8);
            for m in 0..8usize {
                let a = b.mul_f32(exps[4 * m], c16f);
                let bb = b.mul_f32(exps[4 * m + 1], c16f);
                let p0 = b.cvt_pk_fp8_f32(a, bb);
                let c = b.mul_f32(exps[4 * m + 2], c16f);
                let d = b.mul_f32(exps[4 * m + 3], c16f);
                let p1 = b.cvt_pk_fp8_f32(c, d);
                let sh = b.shl_i32(p1, c16i);
                words.push(b.add_i32(p0, sh));
            }
            if sl == 0 {
                chunk_words.push(words);
            } else {
                let mut selw: Vec<ValueId> = Vec::with_capacity(8);
                for m in 0..8usize {
                    let lo = chunk_words[0][m];
                    let hi = words[m];
                    let diff = b.xor_i32(lo, hi);
                    let pick = b.mul_i32(diff, cx.pblk);
                    selw.push(b.xor_i32(lo, pick));
                }
                chunk_words.clear();
                chunk_words.push(selw);
            }
        }
        let rsum = rsum.unwrap();
        let l_old = b.bitcast_f32(l_var);
        let l_scaled = b.mul_f32(l_old, corr);
        let l_new = b.add_f32(l_scaled, rsum);
        let m_bits = b.bitcast_i32(m_new);
        let l_bits = b.bitcast_i32(l_new);
        b.set_var(m_var, m_bits);
        b.set_var(l_var, l_bits);
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);

        // ---- PV: four 32-wide output tiles for this 64-key half ----
        let sel = chunk_words[0].clone();
        let p8 = b.frag8(sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7]);
        for e in 0..4usize {
            let mut scaled: Vec<ValueId> = Vec::with_capacity(16);
            for r in 0..16u8 {
                let lane = b.frag16_lane(o_vars[e], r);
                let lanev = b.bitcast_f32(lane);
                let i_const = ((r / 4) * 8 + (r % 4)) as i32;
                let ccoff = b.c_i32(i_const * 4);
                let caddr0 = b.add_i32(cx.pblk16, cx.lds_corr);
                let caddr = b.add_i32(caddr0, ccoff);
                let craw = b.load_lds_b32(Opnd::Val(caddr));
                let cf = b.bitcast_f32(craw);
                let sc = b.mul_f32(lanev, cf);
                scaled.push(b.bitcast_i32(sc));
            }
            let acc = b.frag16_build(&scaled);
            let mut vw: Vec<ValueId> = Vec::with_capacity(8);
            for r in 0..4i32 {
                let off = b.c_i32((hh as i32) * 8192 + (e as i32) * 32 + 1024 * r);
                let addr = b.add_i32(cx.tr_base, off);
                let tr = b.load_lds_tr_b8(Opnd::Val(addr));
                vw.push(b.frag16_lane(tr, 0));
                vw.push(b.frag16_lane(tr, 1));
            }
            let v8 = b.frag8(vw[0], vw[1], vw[2], vw[3], vw[4], vw[5], vw[6], vw[7]);
            let o_new = b.mfma_scale_fp8(p8, v8, acc, c127, c123);
            b.set_var(o_vars[e], o_new);
        }
        b.wait_lds(0);
        b.barrier();
        b.wait_lds(0);
    }

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, cx.nkv);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: O * (V descale / l) -> BF16 ----
    let l_final = b.bitcast_f32(l_var);
    let inv_l = b.rcp_f32(l_final);
    let fin = b.mul_f32(inv_l, cx.vd_raw);
    let fin_bits = b.bitcast_i32(fin);
    b.store_lds_b32(Opnd::Val(cx.corr_row), Opnd::Val(fin_bits));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    // output byte offset of (batch, head, row 0 of this wave + 4p, column j)
    let ob0 = b.mul_i32(cx.bidx, cx.o_bs);
    let ob1 = b.mul_i32(cx.hidx, cx.o_hs);
    let c2v = b.c_i32(2);
    let p4 = b.shl_i32(cx.pblk, c2v);
    let orow0 = b.shl_i32(cx.wave, c5);
    let orow1 = b.add_i32(cx.row0, p4);
    let _ = orow0;
    let orow_off = b.mul_i32(orow1, cx.o_seqs);
    let j2 = b.shl_i32(cx.j, c1);
    let ob2 = b.add_i32(ob0, ob1);
    let ob3 = b.add_i32(ob2, orow_off);
    let ob4 = b.add_i32(ob3, j2);
    let _ = (c4096, c11);
    for e in 0..4usize {
        let eo = b.c_i32(64 * e as i32);
        let ob5 = b.add_i32(ob4, eo);
        let o_ptr = b.ptr_add(cx.ptr_o, ob5);
        for r in 0..16u8 {
            let i_const = ((r / 4) * 8 + (r % 4)) as i32;
            let ccoff = b.c_i32(i_const * 4);
            let caddr0 = b.add_i32(cx.pblk16, cx.lds_corr);
            let caddr = b.add_i32(caddr0, ccoff);
            let craw = b.load_lds_b32(Opnd::Val(caddr));
            let cf = b.bitcast_f32(craw);
            let lane = b.frag16_lane(o_vars[e], r);
            let lanev = b.bitcast_f32(lane);
            let res = b.mul_f32(lanev, cf);
            let packed = b.cvt_pk_bf16_f32(res, res);
            let ic = b.c_i32(i_const);
            let roff = b.mul_i32(ic, cx.o_seqs);
            let rptr = b.ptr_add(o_ptr, roff);
            b.store_u16(rptr, 0, packed);
        }
    }
}

fn b_wbase(b: &mut FnBuilder, cx: &FmhaCtx) -> ValueId {
    let c8320 = b.c_i32(8320);
    b.mul_i32(cx.wave, c8320)
}

/// ABI-compatible FMHA forward kernel for the vendor `fwd_hd128_mxfp8` interface.
///
/// Same 656-byte kernarg layout, BSHD fp8 (e4m3) Q/K/V with head dim 128, uint8 E8M0 1x32 Q/K
/// descales `[B, S, H, 4]`, scalar f32 V descale, BF16 output `[B, S, H, 128]`, grid
/// `(S / 256, H, B)`, block 512. Like the vendor kernel it computes `P = softmax_2(S)`, i.e.
/// `exp2(S)` with `S = sum Qdeq * Kdeq`, and ignores the softmax-scale slot.
///
/// Requirements of this version: `S` a multiple of 256, `kv_seq_len` a multiple of 128,
/// `gqa = 1`. Structure: eight query tiles of 32 rows per block, K (role A, waves 0-3) and V
/// (role B, waves 4-7) staged through LDS, two 64-key online-softmax substeps per 128-key tile,
/// four register-resident O accumulators, separate A/B steady loops with the same barrier
/// schedule.
pub fn fmha_fwd_hd128_mxfp8() -> Function {
    let mut b = FnBuilder::new("fmha_fwd_hd128_mxfp8", 512);
    b.lds(99456);
    b.arg("ptr_o", DType::Ptr, 0x000, 8);
    b.arg("ptr_q", DType::Ptr, 0x010, 8);
    b.arg("ptr_k", DType::Ptr, 0x020, 8);
    b.arg("ptr_v", DType::Ptr, 0x030, 8);
    b.arg("s_Seqs", DType::I32, 0x070, 4);
    b.arg("s_Hs", DType::I32, 0x090, 4);
    b.arg("s_Bs", DType::I32, 0x0A0, 4);
    b.arg("s_k_Seqs", DType::I32, 0x0C0, 4);
    b.arg("s_k_Hs", DType::I32, 0x0D0, 4);
    b.arg("s_k_Bs", DType::I32, 0x0E0, 4);
    b.arg("s_kv_seq_len", DType::I32, 0x110, 4);
    b.arg("s_q_head_num", DType::I32, 0x140, 4);
    b.arg("s_v_Seqs", DType::I32, 0x150, 4);
    b.arg("s_v_Hs", DType::I32, 0x160, 4);
    b.arg("s_v_Bs", DType::I32, 0x170, 4);
    b.arg("s_o_Seqs", DType::I32, 0x180, 4);
    b.arg("s_o_Hs", DType::I32, 0x190, 4);
    b.arg("s_o_Bs", DType::I32, 0x1A0, 4);
    b.arg("ptr_q_descale", DType::Ptr, 0x200, 8);
    b.arg("ptr_k_descale", DType::Ptr, 0x210, 8);
    b.arg("ptr_v_descale", DType::Ptr, 0x220, 8);
    b.arg("s_descale_q_Bs", DType::I32, 0x230, 4);
    b.arg("s_descale_q_Hs", DType::I32, 0x240, 4);
    b.arg("s_descale_k_Bs", DType::I32, 0x250, 4);
    b.arg("s_descale_k_Hs", DType::I32, 0x260, 4);
    b.arg("abi_tail", DType::I32, 0x280, 16);

    let ptr_o = b.let_(Op::KernargPtr { offset: 0x000 });
    let ptr_q = b.let_(Op::KernargPtr { offset: 0x010 });
    let ptr_k = b.let_(Op::KernargPtr { offset: 0x020 });
    let ptr_v = b.let_(Op::KernargPtr { offset: 0x030 });
    let q_seqs = b.let_(Op::KernargI32 { offset: 0x070 });
    let q_hs = b.let_(Op::KernargI32 { offset: 0x090 });
    let q_bs = b.let_(Op::KernargI32 { offset: 0x0A0 });
    let k_seqs = b.let_(Op::KernargI32 { offset: 0x0C0 });
    let k_hs = b.let_(Op::KernargI32 { offset: 0x0D0 });
    let k_bs = b.let_(Op::KernargI32 { offset: 0x0E0 });
    let kv_len = b.let_(Op::KernargI32 { offset: 0x110 });
    let q_heads = b.let_(Op::KernargI32 { offset: 0x140 });
    let v_seqs = b.let_(Op::KernargI32 { offset: 0x150 });
    let v_hs = b.let_(Op::KernargI32 { offset: 0x160 });
    let v_bs = b.let_(Op::KernargI32 { offset: 0x170 });
    let o_seqs = b.let_(Op::KernargI32 { offset: 0x180 });
    let o_hs = b.let_(Op::KernargI32 { offset: 0x190 });
    let o_bs = b.let_(Op::KernargI32 { offset: 0x1A0 });
    let ptr_qd = b.let_(Op::KernargPtr { offset: 0x200 });
    let ptr_kd = b.let_(Op::KernargPtr { offset: 0x210 });
    let ptr_vd = b.let_(Op::KernargPtr { offset: 0x220 });
    let dq_bs = b.let_(Op::KernargI32 { offset: 0x230 });
    let dq_hs = b.let_(Op::KernargI32 { offset: 0x240 });
    let dk_bs = b.let_(Op::KernargI32 { offset: 0x250 });
    let dk_hs = b.let_(Op::KernargI32 { offset: 0x260 });

    let tid_full = b.let_(Op::WorkitemId);
    let qtile = b.let_(Op::WorkgroupIdX);
    let hidx = b.let_(Op::WorkgroupIdY);
    let bidx = b.let_(Op::WorkgroupIdZ);

    let c1 = b.c_i32(1);
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c6 = b.c_i32(6);
    let c7 = b.c_i32(7);
    let c8 = b.c_i32(8);
    let c9 = b.c_i32(9);
    let c12 = b.c_i32(12);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c63 = b.c_i32(63);
    let c128 = b.c_i32(128);
    let c255 = b.c_i32(255);
    let c8320 = b.c_i32(8320);

    let tid = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c63)));
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid_full), Opnd::Val(c6)));
    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let j4 = b.mul_i32(j, c4);
    let p512 = b.shl_i32(pblk, c9);
    let wbase = b.mul_i32(wave, c8320);
    let lds_loc = b.add_i32(j4, p512);
    let lds_dyn = b.add_i32(lds_loc, wbase);
    let pblk16 = b.shl_i32(pblk, c4);
    let lds_corr0 = b.c_i32(8192);
    let lds_corr = b.add_i32(lds_corr0, wbase);
    let corr_row = b.add_i32(j4, lds_corr);
    let lds_k = b.c_i32(66560);
    let lds_v = b.c_i32(82944);
    let g8 = b.shl_i32(pblk, c3);
    let g16 = b.shl_i32(pblk, c4);

    let wave32 = b.shl_i32(wave, c5);
    let qt256 = b.shl_i32(qtile, c8);
    let row0 = b.add_i32(qt256, wave32);
    let row = b.add_i32(row0, j);
    let q_row_off = b.mul_i32(row, q_seqs);
    let q_b_off = b.mul_i32(bidx, q_bs);
    let q_h_off = b.mul_i32(hidx, q_hs);
    let q_off0 = b.add_i32(q_row_off, q_b_off);
    let q_off1 = b.add_i32(q_off0, q_h_off);
    let q_off = b.add_i32(q_off1, g16);

    // Q fragments: chunk c, bytes 0..15 = K 64c+16g.., bytes 16..31 = K 64c+32+16g..
    let mut a8: Vec<ValueId> = Vec::with_capacity(2);
    for c in 0..2u32 {
        let qp = b.ptr_add(ptr_q, q_off);
        let mut w: Vec<ValueId> = Vec::with_capacity(8);
        for i in 0..8u32 {
            let off = 64 * c + if i < 4 { 4 * i } else { 32 + 4 * (i - 4) };
            let v = b.load_f32(qp, off);
            w.push(b.bitcast_i32(v));
        }
        a8.push(b.frag8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]));
    }
    let hq4 = b.mul_i32(q_heads, c4);
    let qd_row = b.mul_i32(row, hq4);
    let qd_b = b.mul_i32(bidx, dq_bs);
    let qd_h = b.mul_i32(hidx, dq_hs);
    let qd0 = b.add_i32(qd_row, qd_b);
    let qd1 = b.add_i32(qd0, qd_h);
    let qd_ptr = b.ptr_add(ptr_qd, qd1);
    let qd_raw = b.load_f32(qd_ptr, 0);
    let qd = b.bitcast_i32(qd_raw);
    let sa0 = b.let_(Op::LshrI32(Opnd::Val(qd), Opnd::Val(g8)));
    let sh16 = b.add_i32(g8, c16i);
    let sa1 = b.let_(Op::LshrI32(Opnd::Val(qd), Opnd::Val(sh16)));
    let vd_raw = b.load_f32(ptr_vd, 0);

    let kb0 = b.mul_i32(bidx, k_bs);
    let kb1 = b.mul_i32(hidx, k_hs);
    let kbase_off = b.add_i32(kb0, kb1);
    let vb0 = b.mul_i32(bidx, v_bs);
    let vb1 = b.mul_i32(hidx, v_hs);
    let vbase_off = b.add_i32(vb0, vb1);
    let kd_b = b.mul_i32(bidx, dk_bs);
    let kd_h = b.mul_i32(hidx, dk_hs);
    let kd_base = b.add_i32(kd_b, kd_h);

    let rt = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c255)));
    let rt_row = b.let_(Op::LshrI32(Opnd::Val(rt), Opnd::Val(c1)));
    let rt_half = b.let_(Op::AndI32(Opnd::Val(rt), Opnd::Val(c1)));
    let rt_half64 = b.shl_i32(rt_half, c6);
    let rt64 = b.shl_i32(rt, c6);
    let is_a = b.cmp_lt_i32(wave, c4);

    let tid_s1 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c1)));
    let i_idx = b.let_(Op::AndI32(Opnd::Val(tid_s1), Opnd::Val(c7)));
    let tid_s4 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c4)));
    let g2 = b.let_(Op::AndI32(Opnd::Val(tid_s4), Opnd::Val(c1)));
    let h_idx = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1)));
    let pP = b.shl_i32(pblk, c12);
    let iP = b.mul_i32(i_idx, c128);
    let g216 = b.mul_i32(g2, c16i);
    let h8 = b.mul_i32(h_idx, c8);
    let tr0 = b.add_i32(pP, iP);
    let tr1 = b.add_i32(tr0, g216);
    let tr2 = b.add_i32(tr1, h8);
    let tr_base = b.add_i32(tr2, lds_v);
    let j128 = b.mul_i32(j, c128);
    let kf0 = b.add_i32(j128, g16);
    let kf_base = b.add_i32(kf0, lds_k);
    let nkv = b.let_(Op::LshrI32(Opnd::Val(kv_len), Opnd::Val(c7)));

    let cx = FmhaCtx {
        ptr_o,
        ptr_k,
        ptr_v,
        ptr_kd,
        k_seqs,
        v_seqs,
        q_heads,
        o_seqs,
        o_hs,
        o_bs,
        bidx,
        hidx,
        row0,
        wave,
        j,
        pblk,
        pblk16,
        j4,
        g8,
        lds_dyn,
        lds_corr,
        corr_row,
        lds_k,
        lds_v,
        kbase_off,
        vbase_off,
        kd_base,
        rt_row,
        rt_half64,
        rt64,
        tr_base,
        kf_base,
        a8_0: a8[0],
        a8_1: a8[1],
        sa0,
        sa1,
        vd_raw,
        nkv,
    };
    b.begin_if(is_a);
    emit_fmha_role(&mut b, &cx, true);
    b.else_();
    emit_fmha_role(&mut b, &cx, false);
    b.end_if();
    b.finish()
}
/// Values shared by the two role loops of the transposed FMHA kernel.
struct FmhaTCtx {
    ptr_o: ValueId,
    ptr_k: ValueId,
    ptr_v: ValueId,
    ptr_kd: ValueId,
    k_seqs: ValueId,
    v_seqs: ValueId,
    kd_stride: ValueId,
    kv_len: ValueId,
    kv_last: ValueId,
    g4: ValueId,
    o_seqs: ValueId,
    o_hs: ValueId,
    o_bs: ValueId,
    bidx: ValueId,
    hidx: ValueId,
    row: ValueId,
    row0: ValueId,
    s_len: ValueId,
    g8: ValueId,
    g_bytes8: ValueId,
    j: ValueId,
    kbase_off: ValueId,
    vbase_off: ValueId,
    kd_base: ValueId,
    rt_row: ValueId,
    rt_half64: ValueId,
    rt64: ValueId,
    tr_base: ValueId,
    kf_base: ValueId,
    ex_own: ValueId,
    ex_partner: ValueId,
    b0: ValueId,
    b1: ValueId,
    sq0: ValueId,
    sq1: ValueId,
    vd_raw: ValueId,
    nkv: ValueId,
    lds_k: ValueId,
    lds_v: ValueId,
    /// wave index within its role (0..3)
    wave4: ValueId,
    /// DMA layout only: per-lane K / V read addresses for the four 32-byte column groups (XOR swizzle)
    kfx: Vec<ValueId>,
    trx: Vec<ValueId>,
}

/// One role's steady loop and epilogue of the transposed FMHA kernel.
///
/// The kernel computes `S^T = K Q^T`, so every lane owns one query (`j`) and sixteen keys of each
/// 32-key score tile: row max and row sum are register reductions plus one exchange with the
/// partner lane `l ^ 32`, the online-softmax correction is lane-local, and `P^T` drops straight
/// into the B operand of `O^T = V^T P^T` (the V fragment, a column read of the row-major V tile,
/// comes from `ds_read_b64_tr_b8`). No score tile ever goes through LDS.
fn emit_fmha_t_role(b: &mut FnBuilder, cx: &FmhaTCtx, role_a: bool) {
    // Debug ablations for performance attribution (never set in normal builds).
    let ablate = std::env::var("FMHA_ABLATE").unwrap_or_default();
    let no_pv = ablate.contains("nopv");
    let no_exp = ablate.contains("noexp");
    let no_stage = ablate.contains("nostage");
    let no_bar = ablate.contains("nobar");
    let no_s = ablate.contains("nos");
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let c7 = b.c_i32(7);
    let c16i = b.c_i32(16);
    let c32v = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let neg_big = b.c_f32(-1.0e30);
    let c128v = b.c_i32(128);
    let pitch = fmha_t_pitch();
    let c4f = b.c_f32(4.0);
    let c_sixteenth = b.c_f32(0.0625);
    let zero = b.frag_zero(DType::Frag16);
    let m_init = b.bitcast_i32(neg_big);
    let zero_f = b.c_f32(0.0);
    let l_init = b.bitcast_i32(zero_f);

    // Global -> LDS staging of one 128-key K (role A) or V (role B) tile: every thread of the
    // role copies 64 bytes (four 16-byte loads), so all loads of a tile are in flight together.
    let (seqs, base_off, ptr, lds_base) = if role_a {
        (cx.k_seqs, cx.kbase_off, cx.ptr_k, cx.lds_k)
    } else {
        (cx.v_seqs, cx.vbase_off, cx.ptr_v, cx.lds_v)
    };
    let sdst = b.add_i32(lds_base, cx.rt64);
    let kdlds = fmha_t_opt("kdlds");
    let nbuf_l = if fmha_t_opt("dbuf") { 2 } else { 1 };
    let lds_kd = b.c_i32(nbuf_l * 256 * pitch + 2048);
    let rt_row4 = b.shl_i32(cx.rt_row, c2);
    let kdb0 = b.add_i32(lds_kd, rt_row4);
    let stage_loads = |b: &mut FnBuilder, key0: ValueId| -> Vec<ValueId> {
        let srow_raw = b.add_i32(key0, cx.rt_row);
        let srow_ok = b.cmp_lt_i32(srow_raw, cx.kv_len);
        // rows past the end of the keys (partial last tile) are clamped to the last valid row
        let srow = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(srow_ok), a: Opnd::Val(srow_raw), b: Opnd::Val(cx.kv_last) });
        let soff0 = b.mul_i32(srow, seqs);
        let soff1 = b.add_i32(soff0, base_off);
        let soff = b.add_i32(soff1, cx.rt_half64);
        let sp = b.ptr_add(ptr, soff);
        let mut v: Vec<ValueId> = (0..4u32).map(|w| b.load_b128(sp, 16 * w)).collect();
        if role_a && kdlds {
            // K scale dword of this thread's key row, staged with the K tile
            let kd_row = b.mul_i32(srow, cx.kd_stride);
            let kd_off = b.add_i32(kd_row, cx.kd_base);
            let kd_ptr = b.ptr_add(cx.ptr_kd, kd_off);
            let kd_raw = b.load_f32(kd_ptr, 0);
            v.push(b.bitcast_i32(kd_raw));
        }
        v
    };
    let stage_stores = |b: &mut FnBuilder, dstb: ValueId, kdb: ValueId, data: &[ValueId]| {
        if data.len() == 5 {
            b.store_lds_b32(Opnd::Val(kdb), Opnd::Val(data[4]));
        }
        for (w, d) in data.iter().take(4).enumerate() {
            let c = b.c_i32(16 * w as i32);
            let a = b.add_i32(dstb, c);
            b.store_lds_b128(Opnd::Val(a), Opnd::Val(*d));
        }
    };
    let key_first = b.c_i32(0);
    let first = stage_loads(b, key_first);
    stage_stores(b, sdst, kdb0, &first);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let _ = (no_pv, no_exp, no_stage, no_bar, no_s);

    if (role_a && fmha_t_opt("prioA")) || (!role_a && fmha_t_opt("prioB")) {
        b.set_prio(1);
    }
    let lazy = fmha_t_opt("lazy");
    let msum = fmha_t_opt("msum");
    // row sums by MFMA: A = all-ones (E4M3 1.0), B = P^T, so every lane accumulates its query's sum
    let ones_w = b.c_i32(0x38383838);
    let ones8 = b.frag8(ones_w, ones_w, ones_w, ones_w, ones_w, ones_w, ones_w, ones_w);
    let mut init_vars = vec![
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(l_init)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::I32, Opnd::Val(c0)),
    ];
    if msum {
        init_vars.push((DType::Frag16, Opnd::Val(zero)));
    }
    let vars = b.begin_loop(&init_vars);
    let it = vars[0];
    let par = vars[7];
    let lacc = if msum { Some(vars[8]) } else { None };
    let m_var = vars[1];
    let l_var = vars[2];
    let o_vars = [vars[3], vars[4], vars[5], vars[6]];
    let dbuf = fmha_t_opt("dbuf");
    // double buffering: compute reads buffer `par`, the next tile is staged into the other one
    let j4 = b.shl_i32(cx.j, c2);
    let (kf_b, tr_b, sdst_n, kdb_n, kd_rd) = if dbuf {
        let cbuf = b.c_i32(256 * fmha_t_pitch());
        let co = b.mul_i32(par, cbuf);
        let npar = b.xor_i32(par, c1);
        let no = b.mul_i32(npar, cbuf);
        let c512 = b.c_i32(512);
        let kco = b.mul_i32(par, c512);
        let kno = b.mul_i32(npar, c512);
        let kd0 = b.add_i32(lds_kd, j4);
        (b.add_i32(cx.kf_base, co), b.add_i32(cx.tr_base, co), b.add_i32(sdst, no), b.add_i32(kdb0, kno), b.add_i32(kd0, kco))
    } else {
        (cx.kf_base, cx.tr_base, sdst, kdb0, b.add_i32(lds_kd, j4))
    };

    // ---- prefetch the next tile (or this one again after the last iteration) ----
    let key0 = b.shl_i32(it, c7);
    let it_p1 = b.add_i32(it, c1);
    let has_next = b.cmp_lt_i32(it_p1, cx.nkv);
    let nxt = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(has_next), a: Opnd::Val(it_p1), b: it.into() });
    let key_next = b.shl_i32(nxt, c7);
    let hq4 = cx.kd_stride;
    // K scale dwords of the four 32-key tiles, loaded before the prefetch so that waiting for
    // them (global loads complete in order) does not also wait for the prefetched tile
    let kd_hoisted: Vec<ValueId> = if fmha_t_opt("kdhoist") {
        (0..4i32)
            .map(|s| {
                let kkey0 = b.add_i32(key0, cx.j);
                let kcs = b.c_i32(32 * s);
                let kkey_raw = b.add_i32(kkey0, kcs);
                let kk_ok = b.cmp_lt_i32(kkey_raw, cx.kv_len);
                let kkey = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(kk_ok), a: Opnd::Val(kkey_raw), b: Opnd::Val(cx.kv_last) });
                let kd_row = b.mul_i32(kkey, hq4);
                let kd_off = b.add_i32(kd_row, cx.kd_base);
                let kd_ptr = b.ptr_add(cx.ptr_kd, kd_off);
                let kd_raw = b.load_f32(kd_ptr, 0);
                b.bitcast_i32(kd_raw)
            })
            .collect()
    } else {
        Vec::new()
    };
    let pre = if no_stage { first.clone() } else { stage_loads(b, key_next) };
    let compute = |b: &mut FnBuilder, masked: bool| {        for hh in 0..2usize {
            // ---- two 32-key score tiles (S^T = K Q^T) of this 64-key substep ----
            let mut sv: Vec<Vec<ValueId>> = Vec::with_capacity(2);
            let mut sfr: Vec<ValueId> = Vec::with_capacity(2);
            for sl in 0..2usize {
                let s = 2 * hh + sl;
                let ko = b.c_i32(32 * pitch * (s as i32));
                let kaddr0 = b.add_i32(kf_b, ko);
                let kaddr0h = b.add_i32(kaddr0, c32v);
                let k8_0 = b.load_lds_b128x2(Opnd::Val(kaddr0), Opnd::Val(kaddr0h));
                let kaddr1 = b.add_i32(kaddr0, c64);
                let kaddr1h = b.add_i32(kaddr1, c32v);
                let k8_1 = b.load_lds_b128x2(Opnd::Val(kaddr1), Opnd::Val(kaddr1h));
                // K scale dword of key (128 it + 32 s + j)
                let kkey0 = b.add_i32(key0, cx.j);
                let kcs = b.c_i32(32 * s as i32);
                let kkey_raw = b.add_i32(kkey0, kcs);
            let kk_ok = b.cmp_lt_i32(kkey_raw, cx.kv_len);
            let kkey = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(kk_ok), a: Opnd::Val(kkey_raw), b: Opnd::Val(cx.kv_last) });
                let kd_row = b.mul_i32(kkey, hq4);
                let kd_off = b.add_i32(kd_row, cx.kd_base);
                let kd_ptr = b.ptr_add(cx.ptr_kd, kd_off);
                let kd = if kdlds {
                    let koff = b.c_i32(128 * s as i32);
                    let ka = b.add_i32(kd_rd, koff);
                    b.load_lds_b32(Opnd::Val(ka))
                } else if fmha_t_opt("kdconst") {
                    b.c_i32(0x7f7f7f7f)
                } else if !kd_hoisted.is_empty() {
                    kd_hoisted[s]
                } else {
                    let kd_raw = b.load_f32(kd_ptr, 0);
                    b.bitcast_i32(kd_raw)
                };
                let sk0 = b.let_(Op::LshrI32(Opnd::Val(kd), Opnd::Val(cx.g8)));
                let sh16 = b.add_i32(cx.g8, c16i);
                let sk1 = b.let_(Op::LshrI32(Opnd::Val(kd), Opnd::Val(sh16)));
                let zero_s = b.frag_zero(DType::Frag16);
                let s_frag = if fmha_t_opt("abl_qk") {
                    zero_s
                } else {
                    let s0 = b.mfma_scale_fp8(k8_0, cx.b0, zero_s, sk0, cx.sq0);
                    b.mfma_scale_fp8(k8_1, cx.b1, s0, sk1, cx.sq1)
                };
                let mut vals: Vec<ValueId> = Vec::with_capacity(16);
                let kb_s0 = b.add_i32(key0, cx.g4);
                let kb_cs = b.c_i32(32 * s as i32);
                let kb_s = b.add_i32(kb_s0, kb_cs);
                for r in 0..16u8 {
                    let lane = b.frag16_lane(s_frag, r);
                    let v = b.bitcast_f32(lane);
                    if masked {
                        // key of this value: 128 it + 32 s + 8 (r / 4) + 4 g + r % 4
                        let kc = b.c_i32((8 * (r / 4) + (r % 4)) as i32);
                        let kidx = b.add_i32(kb_s, kc);
                        let ok = b.cmp_lt_i32(kidx, cx.kv_len);
                        vals.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(neg_big) }));
                    } else {
                        vals.push(v);
                    }
                }
                sv.push(vals);
                sfr.push(s_frag);
            }

            // ---- register softmax: lane-local max, exchange with the partner lane ----
            let mut flat: Vec<ValueId> = Vec::with_capacity(32);
            for sl in 0..2usize {
                for r in 0..16usize {
                    flat.push(sv[sl][r]);
                }
            }
            // v_max3_f32 folds two new values per instruction
            let mut mloc = b.max3_f32(flat[0], flat[1], flat[2]);
            let mut k = 3usize;
            while k + 1 < flat.len() && !fmha_t_opt("abl_max") {
                mloc = b.max3_f32(mloc, flat[k], flat[k + 1]);
                k += 2;
            }
            if k < flat.len() {
                mloc = b.max_f32(mloc, flat[k]);
            }
            let mt = if fmha_t_opt("permlane") {
                b.partner_max_f32(mloc)
            } else {
                let mbits = b.bitcast_i32(mloc);
                b.store_lds_b32(Opnd::Val(cx.ex_own), Opnd::Val(mbits));
                b.wait_lds(0);
                let mp_raw = b.load_lds_b32(Opnd::Val(cx.ex_partner));
                b.wait_lds(0);
                let mpart = b.bitcast_f32(mp_raw);
                b.max_f32(mloc, mpart)
            };
            let m_old = b.bitcast_f32(m_var);
            let corr_eager: Option<ValueId>;
            let m_use: ValueId;
            if lazy {
                // lazy rescaling: the running max only moves when some lane of the wave grows by
                // more than 4 (P x 16 then stays below 16 * 2^4 = 256 < 448, the E4M3 maximum)
                let tau = if fmha_t_opt("lazyinf") { b.c_f32(1.0e9) } else { c4f };
                let thr = b.add_f32(m_old, tau);
                let grow = b.cmp_lt_f32(thr, mt);
                b.begin_if(Opnd::Val(grow));
                // overshoot: move the running max "over" above the tile max so the next triggers are rarer
                let over = [("over8", 8.0f32), ("over6", 6.0), ("over4", 4.0), ("over3", 3.0), ("over2", 2.0)].iter().find(|(n, _)| fmha_t_opt(n)).map(|x| x.1).unwrap_or(0.0);
                let mt_o = if over > 0.0 {
                    let co = b.c_f32(over);
                    b.add_f32(mt, co)
                } else {
                    mt
                };
                let m_new = b.max_f32(m_old, mt_o);
                let dm = b.sub_f32(m_old, m_new);
                let corr = b.exp2_f32(dm);
                let m_bits = b.bitcast_i32(m_new);
                b.set_var(m_var, m_bits);
                if fmha_t_opt("lazypk") {
                    let cd = b.frag2_dup(corr);
                    for e in 0..4usize {
                        b.var_scale_pk(o_vars[e], cd);
                    }
                } else {
                    for e in 0..4usize {
                        if !fmha_t_opt("abl_rescale") {
                            b.var_scale_f32(o_vars[e], corr);
                        }
                    }
                }
                match lacc {
                    Some(la) if fmha_t_opt("lacc1") => b.var_scale_lane(la, 0, corr),
                    Some(la) => b.var_scale_f32(la, corr),
                    None => {
                        let l_old = b.bitcast_f32(l_var);
                        let l_scaled = b.mul_f32(l_old, corr);
                        let l_bits = b.bitcast_i32(l_scaled);
                        b.set_var(l_var, l_bits);
                    }
                }
                b.end_if();
                corr_eager = None;
                m_use = b.bitcast_f32(m_var);
            } else {
                let m_new = b.max_f32(m_old, mt);
                let dm = b.sub_f32(m_old, m_new);
                let corr = b.exp2_f32(dm);
                let m_bits = b.bitcast_i32(m_new);
                b.set_var(m_var, m_bits);
                corr_eager = Some(corr);
                m_use = m_new;
            }

            // exp2(s - (m - 4)) = 16 exp2(s - m): the P x 16 quantisation scale comes for free
            let m4 = b.sub_f32(m_use, c4f);
            let pk_sub = !masked && fmha_t_opt("pksub");
            let negm4_dup = if pk_sub {
                let negm4 = b.sub_f32(c4f, m_use);
                Some(b.frag2_dup(negm4))
            } else {
                None
            };
            let mut rsum: Option<ValueId> = None;
            let mut words: Vec<ValueId> = Vec::with_capacity(8);
            for sl in 0..2usize {
                let mut exps: Vec<ValueId> = Vec::with_capacity(16);
                if let Some(nd) = negm4_dup {
                    for pr in 0..8usize {
                        let sp = b.frag16_pair(sfr[sl], (2 * pr) as u8);
                        let t = b.pk_add_f32(sp, nd);
                        for h2 in 0..2u8 {
                            let l = b.frag16_lane(t, h2);
                            let sh = b.bitcast_f32(l);
                            exps.push(if no_exp { sh } else { b.exp2_f32(sh) });
                        }
                    }
                } else {
                    for r in 0..16usize {
                        let sh = if fmha_t_opt("abl_sub") { sv[sl][r] } else { b.sub_f32(sv[sl][r], m4) };
                        exps.push(if no_exp { sh } else { b.exp2_f32(sh) });
                    }
                }
                for e in exps.iter().take(if fmha_t_opt("abl_sum") { 1 } else { 16 }) {
                    rsum = Some(match rsum {
                        None => *e,
                        Some(acc) => b.add_f32(acc, *e),
                    });
                }
                for m in 0..4usize {
                    if fmha_t_opt("abl_cvt") {
                        words.push(b.bitcast_i32(exps[4 * m]));
                        continue;
                    }
                    if fmha_t_opt("cvt4") {
                        words.push(b.cvt_pk_fp8x4_f32(exps[4 * m], exps[4 * m + 1], exps[4 * m + 2], exps[4 * m + 3]));
                        continue;
                    }
                    let p0 = b.cvt_pk_fp8_f32(exps[4 * m], exps[4 * m + 1]);
                    let p1 = b.cvt_pk_fp8_f32(exps[4 * m + 2], exps[4 * m + 3]);
                    let sh = b.shl_i32(p1, c16i);
                    words.push(b.add_i32(p0, sh));
                }
            }
            if !msum {
                let rsum16 = rsum.unwrap();
                let rsum = b.mul_f32(rsum16, c_sixteenth);
                let l_old = b.bitcast_f32(l_var);
                let l_base = match corr_eager {
                    Some(corr) => b.mul_f32(l_old, corr),
                    None => l_old,
                };
                let l_new = b.add_f32(l_base, rsum);
                let l_bits = b.bitcast_i32(l_new);
                b.set_var(l_var, l_bits);
            }
            let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);

            // ---- O^T += V^T P^T, four 32-wide dim tiles; the correction is lane-local ----
            let corr_dup = match corr_eager {
                Some(corr) if fmha_t_opt("pkscale") => Some(b.frag2_dup(corr)),
                _ => None,
            };
            if let (Some(la), Some(corr)) = (lacc, corr_eager) {
                b.var_scale_f32(la, corr);
            }
            if let Some(la) = lacc {
                b.var_mfma_scale(la, ones8, p8, c127, c123);
            }
            for e in 0..4usize {
                // in place: O *= corr, then O += V^T P^T (no copies)
                match (corr_dup, corr_eager) {
                    _ if fmha_t_opt("abl_rescale") => {}
                    (_, None) => {}
                    (Some(cd), _) => b.var_scale_pk(o_vars[e], cd),
                    (None, Some(corr)) => b.var_scale_f32(o_vars[e], corr),
                }
                let mut addrs: Vec<ValueId> = Vec::with_capacity(4);
                for r in 0..4i32 {
                    let off = b.c_i32((hh as i32) * 64 * pitch + 16 * pitch * r + (e as i32) * 32);
                    addrs.push(b.add_i32(tr_b, off));
                }
                if !no_pv {
                    let v8 = b.load_lds_tr_b8x4(addrs[0], addrs[1], addrs[2], addrs[3]);
                    b.var_mfma_scale(o_vars[e], v8, p8, c127, c123);
                }
            }
        }
    };
    // the last iteration may hold fewer than 128 keys: only that one pays for the key mask
    let key_end = b.add_i32(key0, c128v);
    let is_tail = b.cmp_lt_i32(cx.kv_len, key_end);
    b.begin_if(Opnd::Val(is_tail));
    compute(b, true);
    b.else_();
    compute(b, false);
    b.end_if();
    // all waves are done with this iteration's K/V; then publish the prefetched tile
    if !dbuf {
        b.wait_lds(0);
        if !no_bar {
            b.barrier();
        }
    }
    if !no_stage {
        stage_stores(b, sdst_n, kdb_n, &pre);
    }
    b.wait_lds(0);
    if !no_bar {
        b.barrier();
    }
    b.wait_lds(0);
    if dbuf {
        let npar2 = b.xor_i32(par, c1);
        b.set_var(par, npar2);
    }

    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, cx.nkv);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: O * (V descale / l_total) -> BF16, four dims per 8-byte store ----
    let l_tot = if let Some(la) = lacc {
        // the MFMA sum already covers the keys of both lane halves
        let l0 = b.frag16_lane(la, 0);
        b.bitcast_f32(l0)
    } else {
        let l_final = b.bitcast_f32(l_var);
        let lbits = b.bitcast_i32(l_final);
        b.store_lds_b32(Opnd::Val(cx.ex_own), Opnd::Val(lbits));
        b.wait_lds(0);
        let lp_raw = b.load_lds_b32(Opnd::Val(cx.ex_partner));
        b.wait_lds(0);
        let lpart = b.bitcast_f32(lp_raw);
        b.add_f32(l_final, lpart)
    };
    let inv_l = b.rcp_f32(l_tot);
    let fin = b.mul_f32(inv_l, cx.vd_raw);

    let ob0 = b.mul_i32(cx.bidx, cx.o_bs);
    let ob1 = b.mul_i32(cx.hidx, cx.o_hs);
    let orow = b.mul_i32(cx.row, cx.o_seqs);
    let ob2 = b.add_i32(ob0, ob1);
    let ob3 = b.add_i32(ob2, orow);
    let ob4 = b.add_i32(ob3, cx.g_bytes8);
    let o_ptr0 = b.ptr_add(cx.ptr_o, ob4);
    // Waves whose rows all lie past the end of a partial query tile only duplicate the last row:
    // they must not store (their lazy-rescale decisions differ from the wave holding the real row,
    // so their rounding would differ and the same-address stores would race). The condition is
    // wave-uniform; the loop above still ran for the barriers.
    let wave_live = b.cmp_lt_i32(cx.row0, cx.s_len);
    b.begin_if(Opnd::Val(wave_live));
    for e in 0..4usize {
        for q in 0..4usize {
            let mut v4: Vec<ValueId> = Vec::with_capacity(4);
            for k in 0..4usize {
                let lane = b.frag16_lane(o_vars[e], (4 * q + k) as u8);
                let lanev = b.bitcast_f32(lane);
                v4.push(b.mul_f32(lanev, fin));
            }
            let w0 = b.cvt_pk_bf16_f32(v4[0], v4[1]);
            let w1 = b.cvt_pk_bf16_f32(v4[2], v4[3]);
            // dims 32 e + 8 q + 4 g ..+3, two bytes each
            let off = (64 * e + 16 * q) as u32;
            let f0 = b.bitcast_f32(w0);
            let f1 = b.bitcast_f32(w1);
            b.store_f32(o_ptr0, off, f0);
            b.store_f32(o_ptr0, off + 4, f1);
        }
    }
    b.end_if();
}

/// Phase-shifted ping-pong variant of `emit_fmha_t_role` (kernel `fmha_fwd_hd128_mxfp8_pp`).
///
/// Every wave runs the same software-pipelined program, one 64-key substep `u` being split into
/// a VALU phase `V(u)` (softmax of S_u -> P_u) and an MFMA phase `M(u+1)` (PV of P_u with the V
/// tile plus QK of substep u+1), separated by workgroup barriers. Role A waves start with the
/// MFMA phase, role B waves one phase later, so the two waves that share a SIMD are in
/// complementary phases: one wave's MFMA/LDS work overlaps the other's VALU work. K/V tiles live
/// in a two-deep LDS ring (K(t+1) is written by role A in the third phase of iteration t, V(t+1)
/// by role B in the fourth), so the loop needs four barriers per 128-key tile and no others.
fn emit_fmha_pp_role(b: &mut FnBuilder, cx: &FmhaTCtx, role_a: bool) {
    let ablate = std::env::var("FMHA_ABLATE").unwrap_or_default();
    let no_exp = ablate.contains("noexp");
    let no_pv = ablate.contains("nopv");
    let no_bar = ablate.contains("nobar");
    let no_stage = ablate.contains("nostage");
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c7 = b.c_i32(7);
    let c16i = b.c_i32(16);
    let c32v = b.c_i32(32);
    let c64 = b.c_i32(64);
    let c123 = b.c_i32(123);
    let c127 = b.c_i32(127);
    let neg_big = b.c_f32(-1.0e30);
    let c128v = b.c_i32(128);
    let pitch = fmha_t_pitch();
    let c4f = b.c_f32(4.0);
    let c_sixteenth = b.c_f32(0.0625);
    let zero = b.frag_zero(DType::Frag16);
    let m_init = b.bitcast_i32(neg_big);
    let zero_f = b.c_f32(0.0);
    let l_init = b.bitcast_i32(zero_f);
    let cbuf = b.c_i32(128 * pitch);
    let hq4 = cx.kd_stride;
    let lazy = fmha_t_opt("lazy");
    let msum = fmha_t_opt("msum");
    let use_prio = !fmha_t_opt("pp_noprio");

    // ---- staging of one 128-key K (role A) or V (role B) tile, as in the plain kernel ----
    let (seqs, base_off, ptr, lds_base) = if role_a {
        (cx.k_seqs, cx.kbase_off, cx.ptr_k, cx.lds_k)
    } else {
        (cx.v_seqs, cx.vbase_off, cx.ptr_v, cx.lds_v)
    };
    let sdst = b.add_i32(lds_base, cx.rt64);
    let stage_loads = |b: &mut FnBuilder, key0: ValueId| -> Vec<ValueId> {
        let srow_raw = b.add_i32(key0, cx.rt_row);
        let srow_ok = b.cmp_lt_i32(srow_raw, cx.kv_len);
        let srow = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(srow_ok), a: Opnd::Val(srow_raw), b: Opnd::Val(cx.kv_last) });
        let soff0 = b.mul_i32(srow, seqs);
        let soff1 = b.add_i32(soff0, base_off);
        let soff = b.add_i32(soff1, cx.rt_half64);
        let sp = b.ptr_add(ptr, soff);
        (0..4u32).map(|w| b.load_b128(sp, 16 * w)).collect()
    };
    let stage_stores = |b: &mut FnBuilder, dstb: ValueId, data: &[ValueId]| {
        for (w, d) in data.iter().enumerate() {
            let c = b.c_i32(16 * w as i32);
            let a = b.add_i32(dstb, c);
            b.store_lds_b128(Opnd::Val(a), Opnd::Val(*d));
        }
    };
    // K scale dword of 32-key tile `s` of the tile starting at `key0`
    let kd_load = |b: &mut FnBuilder, key0: ValueId, s: i32| -> ValueId {
        let kkey0 = b.add_i32(key0, cx.j);
        let kcs = b.c_i32(32 * s);
        let kkey_raw = b.add_i32(kkey0, kcs);
        let kk_ok = b.cmp_lt_i32(kkey_raw, cx.kv_len);
        let kkey = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(kk_ok), a: Opnd::Val(kkey_raw), b: Opnd::Val(cx.kv_last) });
        let kd_row = b.mul_i32(kkey, hq4);
        let kd_off = b.add_i32(kd_row, cx.kd_base);
        let kd_ptr = b.ptr_add(cx.ptr_kd, kd_off);
        let kd_raw = b.load_f32(kd_ptr, 0);
        b.bitcast_i32(kd_raw)
    };
    // K fragments of 32-key tile `s` (0..3 within the 128-key tile) from the K buffer at `kbuf`
    let k_frags = |b: &mut FnBuilder, kbuf: ValueId, s: usize| -> (ValueId, ValueId) {
        let ko = b.c_i32(32 * pitch * (s as i32));
        let kaddr0 = b.add_i32(kbuf, ko);
        let kaddr0h = b.add_i32(kaddr0, c32v);
        let k8_0 = b.load_lds_b128x2(Opnd::Val(kaddr0), Opnd::Val(kaddr0h));
        let kaddr1 = b.add_i32(kaddr0, c64);
        let kaddr1h = b.add_i32(kaddr1, c32v);
        let k8_1 = b.load_lds_b128x2(Opnd::Val(kaddr1), Opnd::Val(kaddr1h));
        (k8_0, k8_1)
    };

    // ---- prologue: stage tile 0 (K by role A, V by role B) and compute S_0 = QK(0) ----
    let key_first = b.c_i32(0);
    let first = stage_loads(b, key_first);
    stage_stores(b, sdst, &first);
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    if !role_a {
        // role B runs one phase behind role A
        b.barrier();
    }
    let mut s_init: Vec<ValueId> = Vec::with_capacity(2);
    for sl in 0..2usize {
        let (k8_0, k8_1) = k_frags(b, cx.kf_base, sl);
        let kd = kd_load(b, key_first, sl as i32);
        let sk0 = b.let_(Op::LshrI32(Opnd::Val(kd), Opnd::Val(cx.g8)));
        let sh16 = b.add_i32(cx.g8, c16i);
        let sk1 = b.let_(Op::LshrI32(Opnd::Val(kd), Opnd::Val(sh16)));
        let zero_s = b.frag_zero(DType::Frag16);
        let s0 = b.mfma_scale_fp8(k8_0, cx.b0, zero_s, sk0, cx.sq0);
        s_init.push(b.mfma_scale_fp8(k8_1, cx.b1, s0, sk1, cx.sq1));
    }
    b.barrier(); // end of the phase M(0)

    let ones_w = b.c_i32(0x38383838);
    let ones8 = b.frag8(ones_w, ones_w, ones_w, ones_w, ones_w, ones_w, ones_w, ones_w);
    let mut init_vars = vec![
        (DType::I32, Opnd::Val(c0)),
        (DType::I32, Opnd::Val(m_init)),
        (DType::I32, Opnd::Val(l_init)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag16, Opnd::Val(s_init[0])),
        (DType::Frag16, Opnd::Val(s_init[1])),
    ];
    if msum {
        init_vars.push((DType::Frag16, Opnd::Val(zero)));
    }
    let vars = b.begin_loop(&init_vars);
    let it = vars[0];
    let m_var = vars[1];
    let l_var = vars[2];
    let o_vars = [vars[3], vars[4], vars[5], vars[6]];
    let par = vars[7];
    let s_vars = [vars[8], vars[9]];
    let lacc = if msum { Some(vars[10]) } else { None };

    // per-iteration LDS buffer selection: tile t in buffer `par`, tile t+1 in the other one
    let cur_off = b.mul_i32(par, cbuf);
    let npar = b.xor_i32(par, c1);
    let nxt_off = b.mul_i32(npar, cbuf);
    let kf_cur = b.add_i32(cx.kf_base, cur_off);
    let kf_nxt = b.add_i32(cx.kf_base, nxt_off);
    let tr_cur = b.add_i32(cx.tr_base, cur_off);
    let sdst_n = b.add_i32(sdst, nxt_off);

    let key0 = b.shl_i32(it, c7);
    let it_p1 = b.add_i32(it, c1);
    let has_next = b.cmp_lt_i32(it_p1, cx.nkv);
    let nxt = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(has_next), a: Opnd::Val(it_p1), b: it.into() });
    let key_next = b.shl_i32(nxt, c7);
    let key_t1 = b.shl_i32(it_p1, c7); // tile t+1 for QK (scale rows are clamped; data unused past the end)
    // K scales for QK(2t+1) (tile t, 32-key tiles 2,3) and QK(2t+2) (tile t+1, tiles 0,1), then the prefetch
    let kd_a2 = kd_load(b, key0, 2);
    let kd_a3 = kd_load(b, key0, 3);
    let kd_b0 = kd_load(b, key_t1, 0);
    let kd_b1 = kd_load(b, key_t1, 1);
    let pre = if no_stage { first.clone() } else { stage_loads(b, key_next) };

    // ---- VALU phase: softmax of the scores in `s_vars` (64 keys, hh selects the key offset) ----
    let softmax = |b: &mut FnBuilder, masked: bool, hh: usize| -> ValueId {
        let mut sv: Vec<Vec<ValueId>> = Vec::with_capacity(2);
        for sl in 0..2usize {
            let s = 2 * hh + sl;
            let kb_s0 = b.add_i32(key0, cx.g4);
            let kb_cs = b.c_i32(32 * s as i32);
            let kb_s = b.add_i32(kb_s0, kb_cs);
            let mut vals: Vec<ValueId> = Vec::with_capacity(16);
            for r in 0..16u8 {
                let lane = b.frag16_lane(s_vars[sl], r);
                let v = b.bitcast_f32(lane);
                if masked {
                    let kc = b.c_i32((8 * (r / 4) + (r % 4)) as i32);
                    let kidx = b.add_i32(kb_s, kc);
                    let ok = b.cmp_lt_i32(kidx, cx.kv_len);
                    vals.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(neg_big) }));
                } else {
                    vals.push(v);
                }
            }
            sv.push(vals);
        }
        let mut flat: Vec<ValueId> = Vec::with_capacity(32);
        for sl in 0..2usize {
            for r in 0..16usize {
                flat.push(sv[sl][r]);
            }
        }
        let mut mloc = b.max3_f32(flat[0], flat[1], flat[2]);
        let mut k = 3usize;
        while k + 1 < flat.len() {
            mloc = b.max3_f32(mloc, flat[k], flat[k + 1]);
            k += 2;
        }
        if k < flat.len() {
            mloc = b.max_f32(mloc, flat[k]);
        }
        let mt = if fmha_t_opt("permlane") {
            b.partner_max_f32(mloc)
        } else {
            let mbits = b.bitcast_i32(mloc);
            b.store_lds_b32(Opnd::Val(cx.ex_own), Opnd::Val(mbits));
            b.wait_lds(0);
            let mp_raw = b.load_lds_b32(Opnd::Val(cx.ex_partner));
            b.wait_lds(0);
            let mpart = b.bitcast_f32(mp_raw);
            b.max_f32(mloc, mpart)
        };
        let m_old = b.bitcast_f32(m_var);
        let rescale = |b: &mut FnBuilder, corr: ValueId| {
            for e in 0..4usize {
                b.var_scale_f32(o_vars[e], corr);
            }
            match lacc {
                Some(la) => b.var_scale_f32(la, corr),
                None => {
                    let l_old = b.bitcast_f32(l_var);
                    let l_scaled = b.mul_f32(l_old, corr);
                    let l_bits = b.bitcast_i32(l_scaled);
                    b.set_var(l_var, l_bits);
                }
            }
        };
        let m_use: ValueId;
        if lazy {
            let thr = b.add_f32(m_old, c4f);
            let grow = b.cmp_lt_f32(thr, mt);
            b.begin_if(Opnd::Val(grow));
            let m_new = b.max_f32(m_old, mt);
            let dm = b.sub_f32(m_old, m_new);
            let corr = b.exp2_f32(dm);
            let m_bits = b.bitcast_i32(m_new);
            b.set_var(m_var, m_bits);
            rescale(b, corr);
            b.end_if();
            m_use = b.bitcast_f32(m_var);
        } else {
            let m_new = b.max_f32(m_old, mt);
            let dm = b.sub_f32(m_old, m_new);
            let corr = b.exp2_f32(dm);
            let m_bits = b.bitcast_i32(m_new);
            b.set_var(m_var, m_bits);
            rescale(b, corr);
            m_use = m_new;
        }
        let m4 = b.sub_f32(m_use, c4f);
        let pk_sub = !masked && fmha_t_opt("pksub");
        let negm4_dup = if pk_sub {
            let negm4 = b.sub_f32(c4f, m_use);
            Some(b.frag2_dup(negm4))
        } else {
            None
        };
        let mut rsum: Option<ValueId> = None;
        let mut words: Vec<ValueId> = Vec::with_capacity(8);
        for sl in 0..2usize {
            let mut exps: Vec<ValueId> = Vec::with_capacity(16);
            if let Some(nd) = negm4_dup {
                for pr in 0..8usize {
                    let sp = b.frag16_pair(s_vars[sl], (2 * pr) as u8);
                    let t = b.pk_add_f32(sp, nd);
                    for h2 in 0..2u8 {
                        let l = b.frag16_lane(t, h2);
                        let sh = b.bitcast_f32(l);
                        exps.push(if no_exp { sh } else { b.exp2_f32(sh) });
                    }
                }
            } else {
                for r in 0..16usize {
                    let sh = b.sub_f32(sv[sl][r], m4);
                    exps.push(if no_exp { sh } else { b.exp2_f32(sh) });
                }
            }
            if !msum {
                for e in &exps {
                    rsum = Some(match rsum {
                        None => *e,
                        Some(acc) => b.add_f32(acc, *e),
                    });
                }
            }
            for m in 0..4usize {
                if fmha_t_opt("cvt4") {
                    words.push(b.cvt_pk_fp8x4_f32(exps[4 * m], exps[4 * m + 1], exps[4 * m + 2], exps[4 * m + 3]));
                    continue;
                }
                let p0 = b.cvt_pk_fp8_f32(exps[4 * m], exps[4 * m + 1]);
                let p1 = b.cvt_pk_fp8_f32(exps[4 * m + 2], exps[4 * m + 3]);
                let sh = b.shl_i32(p1, c16i);
                words.push(b.add_i32(p0, sh));
            }
        }
        if !msum {
            let rsum16 = rsum.unwrap();
            let rsum = b.mul_f32(rsum16, c_sixteenth);
            let l_old = b.bitcast_f32(l_var);
            let l_new = b.add_f32(l_old, rsum);
            let l_bits = b.bitcast_i32(l_new);
            b.set_var(l_var, l_bits);
        }
        b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7])
    };

    // ---- MFMA phase: O += V^T P^T for the 64 keys `hh` of the V buffer at `tr_cur`, and the
    // scores of the next substep (K from `kbuf`, 32-key tiles s0, s0 + 1) into `s_vars` ----
    let mfma_phase = |b: &mut FnBuilder, p8: ValueId, hh: usize, kbuf: ValueId, s0: usize, kds: [ValueId; 2]| {
        if let Some(la) = lacc {
            b.var_mfma_scale(la, ones8, p8, c127, c123);
        }
        for e in 0..4usize {
            let mut addrs: Vec<ValueId> = Vec::with_capacity(4);
            for r in 0..4i32 {
                let off = b.c_i32((hh as i32) * 64 * pitch + 16 * pitch * r + (e as i32) * 32);
                addrs.push(b.add_i32(tr_cur, off));
            }
            if no_pv {
                continue;
            }
            let v8 = b.load_lds_tr_b8x4(addrs[0], addrs[1], addrs[2], addrs[3]);
            b.var_mfma_scale(o_vars[e], v8, p8, c127, c123);
        }
        for sl in 0..2usize {
            let (k8_0, k8_1) = k_frags(b, kbuf, s0 + sl);
            let sk0 = b.let_(Op::LshrI32(Opnd::Val(kds[sl]), Opnd::Val(cx.g8)));
            let sh16 = b.add_i32(cx.g8, c16i);
            let sk1 = b.let_(Op::LshrI32(Opnd::Val(kds[sl]), Opnd::Val(sh16)));
            b.var_mfma_scale_z(s_vars[sl], k8_0, cx.b0, sk0, cx.sq0);
            b.var_mfma_scale(s_vars[sl], k8_1, cx.b1, sk1, cx.sq1);
        }
    };

    // ---- the four phases of one 128-key tile ----
    let key_end = b.add_i32(key0, c128v);
    let is_tail = b.cmp_lt_i32(cx.kv_len, key_end);
    let phase_end = |b: &mut FnBuilder, next_is_mfma: bool| {
        if use_prio {
            let hi = if fmha_t_opt("pp_prioinv") { !next_is_mfma } else { next_is_mfma };
            b.set_prio(if hi { 1 } else { 0 });
        }
        b.wait_lds(0);
        if !no_bar {
            b.barrier();
        }
    };
    let four_phases = |b: &mut FnBuilder, masked: bool| {
        // P1: V(2t)
        let pa = softmax(b, masked, 0);
        phase_end(b, true);
        // P2: M(2t+1): PV(P_a, V(t) rows 0..63) + QK(2t+1) (tile t, 32-key tiles 2,3)
        mfma_phase(b, pa, 0, kf_cur, 2, [kd_a2, kd_a3]);
        phase_end(b, false);
        // P3: V(2t+1), then role A publishes K(t+1)
        let pb = softmax(b, masked, 1);
        if role_a && !no_stage {
            stage_stores(b, sdst_n, &pre);
        }
        phase_end(b, true);
        // P4: M(2t+2): PV(P_b, V(t) rows 64..127) + QK(2t+2) (tile t+1, tiles 0,1); role B publishes V(t+1)
        mfma_phase(b, pb, 1, kf_nxt, 0, [kd_b0, kd_b1]);
        if !role_a && !no_stage {
            stage_stores(b, sdst_n, &pre);
        }
        phase_end(b, false);
    };
    b.begin_if(Opnd::Val(is_tail));
    four_phases(b, true);
    b.else_();
    four_phases(b, false);
    b.end_if();
    let npar2 = b.xor_i32(par, c1);
    b.set_var(par, npar2);
    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, cx.nkv);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    if role_a {
        b.barrier(); // role A's trailing idle phase (role B started one phase late)
    }
    // ---- epilogue: O * (V descale / l_total) -> BF16, four dims per 8-byte store ----
    let l_tot = if let Some(la) = lacc {
        // the MFMA sum already covers the keys of both lane halves
        let l0 = b.frag16_lane(la, 0);
        b.bitcast_f32(l0)
    } else {
        let l_final = b.bitcast_f32(l_var);
        let lbits = b.bitcast_i32(l_final);
        b.store_lds_b32(Opnd::Val(cx.ex_own), Opnd::Val(lbits));
        b.wait_lds(0);
        let lp_raw = b.load_lds_b32(Opnd::Val(cx.ex_partner));
        b.wait_lds(0);
        let lpart = b.bitcast_f32(lp_raw);
        b.add_f32(l_final, lpart)
    };
    let inv_l = b.rcp_f32(l_tot);
    let fin = b.mul_f32(inv_l, cx.vd_raw);

    let ob0 = b.mul_i32(cx.bidx, cx.o_bs);
    let ob1 = b.mul_i32(cx.hidx, cx.o_hs);
    let orow = b.mul_i32(cx.row, cx.o_seqs);
    let ob2 = b.add_i32(ob0, ob1);
    let ob3 = b.add_i32(ob2, orow);
    let ob4 = b.add_i32(ob3, cx.g_bytes8);
    let o_ptr0 = b.ptr_add(cx.ptr_o, ob4);
    // Waves whose rows all lie past the end of a partial query tile only duplicate the last row:
    // they must not store (their lazy-rescale decisions differ from the wave holding the real row,
    // so their rounding would differ and the same-address stores would race). The condition is
    // wave-uniform; the loop above still ran for the barriers.
    let wave_live = b.cmp_lt_i32(cx.row0, cx.s_len);
    b.begin_if(Opnd::Val(wave_live));
    for e in 0..4usize {
        for q in 0..4usize {
            let mut v4: Vec<ValueId> = Vec::with_capacity(4);
            for k in 0..4usize {
                let lane = b.frag16_lane(o_vars[e], (4 * q + k) as u8);
                let lanev = b.bitcast_f32(lane);
                v4.push(b.mul_f32(lanev, fin));
            }
            let w0 = b.cvt_pk_bf16_f32(v4[0], v4[1]);
            let w1 = b.cvt_pk_bf16_f32(v4[2], v4[3]);
            // dims 32 e + 8 q + 4 g ..+3, two bytes each
            let off = (64 * e + 16 * q) as u32;
            let f0 = b.bitcast_f32(w0);
            let f1 = b.bitcast_f32(w1);
            b.store_f32(o_ptr0, off, f0);
            b.store_f32(o_ptr0, off + 4, f1);
        }
    }
    b.end_if();
}

/// Vendor-style softmax variant of `emit_fmha_t_role` (kernel `fmha_fwd_hd128_mxfp8_sv`).
///
/// No `v_exp`/`v_sub` on the hot path: the scores come out of the MFMA already biased,
/// `S' = (s - r + 131) / 512`, because the accumulator input of the QK MFMA is a broadcast bias
/// produced by a tiny f16 MFMA (A = ones, B = bias/8 in all eight k slots), and the Q scale is lowered by 2^-9.
/// `v_cvt_pknorm_u16_f32` turns S' into a u16 = 65535 S' whose bit pattern, read as bf16, is
/// 2^(512 S' - 127) (Schraudolph exp2: piecewise linear mantissa, <= 6% error), and
/// `v_cvt_scalef32_pk_fp8_bf16` packs two of them into E4M3. P is thus 16 2^(s - r). The reference `r`
/// only moves ("rollback") when some lane's S' exceeds the E4M3 range (s - r > 4); the rollback
/// shifts the S registers by the (per-lane) change, rescales O and the MFMA row sum and stores the new
/// bias. Row sums come from the same quantised P (ones-MFMA), so the normalisation is consistent.
fn emit_fmha_sv_role(b: &mut FnBuilder, cx: &FmhaTCtx, role_a: bool) {
    let ablate = std::env::var("FMHA_ABLATE").unwrap_or_default();
    let no_stage = ablate.contains("nostage");
    let no_bar = ablate.contains("nobar");
    let no_pv = ablate.contains("nopv");
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c6 = b.c_i32(6);
    let c7m = b.c_i32(7);
    let c8 = b.c_i32(8);
    let c10 = b.c_i32(10);
    let c7 = b.c_i32(7);
    let c16i = b.c_i32(16);
    let c32v = b.c_i32(32);
    let c64 = b.c_i32(64);
    // P scale exponent: P = 2^(4 - shd) 2^(s - r) in the fp8 (default shd = 0: x16); a larger shd leaves more headroom (fewer rollbacks)
    let shd: i32 = if fmha_t_opt("sh4") { 4 } else if fmha_t_opt("sh2") { 2 } else { 0 };
    let c123 = b.c_i32(123 + shd);
    let c127 = b.c_i32(127);
    let neg_big = b.c_f32(-1.0e30);
    let c128v = b.c_i32(128);
    let pitch = fmha_t_pitch();
    let zero = b.frag_zero(DType::Frag16);
    let hq4 = cx.kd_stride;
    let c_thr = b.c_f32(if fmha_t_opt("svinf") { 1.0e9 } else { 135.0 / 512.0 });
    let c_b131 = b.c_f32((131 - shd) as f32 / 512.0);
    let c8f = b.c_f32(8.0);
    let c0125 = b.c_f32(0.125);
    let c512f = b.c_f32(512.0);
    let one_f = b.c_f32(1.0);
    let zero_f = b.c_f32(0.0);
    let cffff = b.c_i32(0xffff);
    let cm9 = b.c_i32(-9);

    // ---- staging as in the plain kernel ----
    let (seqs, base_off, ptr, lds_base) = if role_a {
        (cx.k_seqs, cx.kbase_off, cx.ptr_k, cx.lds_k)
    } else {
        (cx.v_seqs, cx.vbase_off, cx.ptr_v, cx.lds_v)
    };
    let sdst = b.add_i32(lds_base, cx.rt64);
    let kdlds = fmha_t_opt("kdlds");
    let dma_l_pre = fmha_t_opt("dma");
    let nbuf_l = if fmha_t_opt("dbuf") { 2 } else { 1 };
    let cbuf_i: i32 = if dma_l_pre { 33280 } else { 256 * pitch };
    let lds_kd = b.c_i32(nbuf_l * cbuf_i + 2048);
    let rt_row4 = b.shl_i32(cx.rt_row, c2);
    let kdb0 = b.add_i32(lds_kd, rt_row4);
    let dma_l = fmha_t_opt("dma");
    let dma = dma_l && !(role_a && fmha_t_opt("nodmak")) && !(!role_a && fmha_t_opt("nodmav"));
    // DMA staging (global -> LDS, no VGPRs, no ds_write). One wave instruction writes 1 KB to M0 + 16 lane in the order of the
    // lane addresses (the hardware lands the data in ADDRESS order, it ignores a lane permutation: an XOR swizzle is impossible),
    // so a "run" is eight rows of 128 bytes. Runs are 1040 bytes apart (like the vendor kernel's 0x410) and run rho holds
    // the rows {(rho & 7) + 8 s + 64 (rho >> 3)}: the 16 lanes of a K fragment read then fall in 16 different bank groups.
    let lane_v = {
        let t8 = b.shl_i32(cx.g4, c3);
        b.add_i32(cx.j, t8)
    };
    let lane_hi = b.let_(Op::LshrI32(Opnd::Val(lane_v), Opnd::Val(c3)));
    let lane_lo = b.let_(Op::AndI32(Opnd::Val(lane_v), Opnd::Val(c7m)));
    let w4_odd = b.let_(Op::AndI32(Opnd::Val(cx.wave4), Opnd::Val(c1)));
    let cbytes = b.shl_i32(lane_lo, c4);
    let w8s = b.shl_i32(lane_hi, c3);
    let rowb = b.add_i32(w8s, cx.wave4);
    let c1040 = b.c_i32(1040);
    let w1040 = b.mul_i32(cx.wave4, c1040);
    let lds_tile = b.add_i32(lds_base, w1040);
    let kd_w = b.shl_i32(w4_odd, c6);
    let kdl_w = b.shl_i32(w4_odd, c8);
    let dummy_lds = b.c_i32(nbuf_l * cbuf_i + 2048 + 1024);
    let dma_issue = |b: &mut FnBuilder, key0: ValueId, buf_off: ValueId, kd_off: ValueId| {
        let rb = b.add_i32(key0, rowb);
        let mut last_sp = None;
        for q in 0..4i32 {
            // run rho = wave4 + 4 q: rows (wave4 + 4 (q & 1)) + 64 (q >> 1) + 8 s
            let cq = b.c_i32(4 * (q & 1) + 64 * (q >> 1));
            let row_raw = b.add_i32(rb, cq);
            let ok = b.cmp_lt_i32(row_raw, cx.kv_len);
            let row = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(row_raw), b: Opnd::Val(cx.kv_last) });
            let o0 = b.mul_i32(row, seqs);
            let o1 = b.add_i32(o0, base_off);
            let o2 = b.add_i32(o1, cbytes);
            let sp = b.ptr_add(ptr, o2);
            last_sp = Some(sp);
            let cl = b.c_i32(4 * 1040 * q);
            let l0 = b.add_i32(lds_tile, buf_off);
            let lq = b.add_i32(l0, cl);
            b.load_global_lds(sp, lq, 0, 4);
        }
        if role_a && kdlds {
            // K scale dwords of the tile's 128 keys: waves 0/1 (and 2/3 again, same data) fetch 64 keys each
            let k0 = b.add_i32(key0, kd_w);
            let kk = b.add_i32(k0, lane_v);
            let kok = b.cmp_lt_i32(kk, cx.kv_len);
            let kkey = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(kok), a: Opnd::Val(kk), b: Opnd::Val(cx.kv_last) });
            let kd_row = b.mul_i32(kkey, hq4);
            let kd_o = b.add_i32(kd_row, cx.kd_base);
            let kd_ptr = b.ptr_add(cx.ptr_kd, kd_o);
            let l0 = b.add_i32(lds_kd, kdl_w);
            let lq = b.add_i32(l0, kd_off);
            b.load_global_lds(kd_ptr, lq, 0, 1);
        }
        // trailing DMA into a spare 1 KB region (a precaution: early experiments lost the last DMA of a wave's group)
        if let Some(sp) = last_sp {
            b.load_global_lds(sp, dummy_lds, 0, 4);
        }
    };    let stage_loads = |b: &mut FnBuilder, key0: ValueId| -> Vec<ValueId> {
        let srow_raw = b.add_i32(key0, cx.rt_row);
        let srow_ok = b.cmp_lt_i32(srow_raw, cx.kv_len);
        let srow = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(srow_ok), a: Opnd::Val(srow_raw), b: Opnd::Val(cx.kv_last) });
        let soff0 = b.mul_i32(srow, seqs);
        let soff1 = b.add_i32(soff0, base_off);
        let soff = b.add_i32(soff1, cx.rt_half64);
        let sp = b.ptr_add(ptr, soff);
        let mut v: Vec<ValueId> = (0..4u32).map(|w| b.load_b128(sp, 16 * w)).collect();
        if role_a && kdlds {
            // K scale dword of this thread's key row, staged with the K tile
            let kd_row = b.mul_i32(srow, hq4);
            let kd_off = b.add_i32(kd_row, cx.kd_base);
            let kd_ptr = b.ptr_add(cx.ptr_kd, kd_off);
            let kd_raw = b.load_f32(kd_ptr, 0);
            v.push(b.bitcast_i32(kd_raw));
        }
        v
    };
    let stage_stores = |b: &mut FnBuilder, dstb: ValueId, kdb: ValueId, data: &[ValueId]| {
        if data.len() == 5 {
            b.store_lds_b32(Opnd::Val(kdb), Opnd::Val(data[4]));
        }
        for (w, d) in data.iter().take(4).enumerate() {
            let c = b.c_i32(16 * w as i32);
            let a = b.add_i32(dstb, c);
            b.store_lds_b128(Opnd::Val(a), Opnd::Val(*d));
        }
    };
    let kdpf = fmha_t_opt("kdpf");
    let kd_load = |b: &mut FnBuilder, key0: ValueId, s: i32| -> ValueId {
        let kkey0 = b.add_i32(key0, cx.j);
        let kcs = b.c_i32(32 * s);
        let kkey_raw = b.add_i32(kkey0, kcs);
        let kk_ok = b.cmp_lt_i32(kkey_raw, cx.kv_len);
        let kkey = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(kk_ok), a: Opnd::Val(kkey_raw), b: Opnd::Val(cx.kv_last) });
        let kd_row = b.mul_i32(kkey, hq4);
        let kd_off = b.add_i32(kd_row, cx.kd_base);
        let kd_ptr = b.ptr_add(cx.ptr_kd, kd_off);
        let kd_raw = b.load_f32(kd_ptr, 0);
        b.bitcast_i32(kd_raw)
    };
    let key_first = b.c_i32(0);
    let kd_first: Vec<ValueId> = if kdpf { (0..4i32).map(|s| kd_load(b, key_first, s)).collect() } else { Vec::new() };
    let first = if dma { Vec::new() } else { stage_loads(b, key_first) };
    if dma {
        dma_issue(b, key_first, c0, c0);
        b.wait_vm(0);
        if fmha_t_opt("dmasync") {
            b.barrier();
            b.wait_vm(0);
        }
    } else {
        stage_stores(b, sdst, kdb0, &first);
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    // ---- loop state: bias word (f16 of B/8 in both halves, B = (131 - r)/512), O, scores, row-sum accumulator ----
    let ones_w = b.c_i32(0x38383838);
    let ones8 = b.frag8(ones_w, ones_w, ones_w, ones_w, ones_w, ones_w, ones_w, ones_w);
    let ones_h = b.c_i32(0x3C003C00);
    let ones_f16 = b.frag2(ones_h, ones_h);
    let w_init = b.c_i32(0x48004800u32 as i32); // B = 64: the first substep always rolls back
    let bias_init = b.frag2(w_init, w_init);
    let sq0m = b.add_i32(cx.sq0, cm9); // Q scales 2^-9: S' = S / 512
    let sq1m = b.add_i32(cx.sq1, cm9);
    if fmha_t_opt("dbgk") {
        // debug: dump the K and V fragments of tile 0 as read from LDS into the output rows, then stop
        let (k8_0, k8_1) = if dma {
            let a0 = cx.kfx[0];
            let a1 = cx.kfx[1];
            let a2 = cx.kfx[2];
            let a3 = cx.kfx[3];
            (b.load_lds_b128x2(Opnd::Val(a0), Opnd::Val(a1)), b.load_lds_b128x2(Opnd::Val(a2), Opnd::Val(a3)))
        } else {
            let kaddr0h = b.add_i32(cx.kf_base, c32v);
            let kaddr1 = b.add_i32(cx.kf_base, c64);
            let kaddr1h = b.add_i32(kaddr1, c32v);
            (b.load_lds_b128x2(Opnd::Val(cx.kf_base), Opnd::Val(kaddr0h)), b.load_lds_b128x2(Opnd::Val(kaddr1), Opnd::Val(kaddr1h)))
        };
        let mut addrs: Vec<ValueId> = Vec::new();
        for r in 0..4i32 {
            let off = b.c_i32(16 * pitch * r);
            let base = if dma { cx.trx[0] } else { cx.tr_base };
            addrs.push(b.add_i32(base, off));
        }
        let v8 = b.load_lds_tr_b8x4(addrs[0], addrs[1], addrs[2], addrs[3]);
        let ob0 = b.mul_i32(cx.bidx, cx.o_bs);
        let ob1 = b.mul_i32(cx.hidx, cx.o_hs);
        let orow = b.mul_i32(cx.row, cx.o_seqs);
        let ob2 = b.add_i32(ob0, ob1);
        let ob3 = b.add_i32(ob2, orow);
        let ob4 = b.add_i32(ob3, cx.g_bytes8);
        let o_ptr0 = b.ptr_add(cx.ptr_o, ob4);
        for (n, fr) in [k8_0, k8_1, v8].iter().enumerate() {
            let cnt = 8u8;
            for q in 0..cnt {
                let w = b.frag16_lane(*fr, q);
                let f = b.bitcast_f32(w);
                b.store_f32(o_ptr0, (n as u32 * 32 + q as u32 * 4) as u32, f);
            }
        }
        return;
    }    let mut init_vars = vec![
        (DType::I32, Opnd::Val(c0)),
        (DType::Frag2, Opnd::Val(bias_init)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
        (DType::Frag16, Opnd::Val(zero)),
    ];
    if kdpf {
        for s in 0..4usize {
            init_vars.push((DType::I32, Opnd::Val(kd_first[s])));
        }
    }
    let dbuf = fmha_t_opt("dbuf");
    let stmid = fmha_t_opt("stmid");
    let par_idx = init_vars.len();
    init_vars.push((DType::I32, Opnd::Val(c0)));
    let vars = b.begin_loop(&init_vars);
    let par = vars[par_idx];
    // double buffering: compute reads buffer par, the next tile is staged into the other one
    let j4 = b.shl_i32(cx.j, c2);
    let (kf_b, tr_b, sdst_n, kdb_n, kd_rd, buf_nxt, kd_nxt) = if dbuf {
        let cbuf = b.c_i32(cbuf_i);
        let co = b.mul_i32(par, cbuf);
        let npar = b.xor_i32(par, c1);
        let no = b.mul_i32(npar, cbuf);
        let c512 = b.c_i32(512);
        let kco = b.mul_i32(par, c512);
        let kno = b.mul_i32(npar, c512);
        let kd0 = b.add_i32(lds_kd, j4);
        (b.add_i32(cx.kf_base, co), b.add_i32(cx.tr_base, co), b.add_i32(sdst, no), b.add_i32(kdb0, kno), b.add_i32(kd0, kco), no, kno)
    } else {
        (cx.kf_base, cx.tr_base, sdst, kdb0, b.add_i32(lds_kd, j4), c0, c0)
    };

    let it = vars[0];
    let bias2 = vars[1];
    let o_vars = [vars[2], vars[3], vars[4], vars[5]];
    let s_vars = [vars[6], vars[7]];
    let la = vars[8];

    let key0 = b.shl_i32(it, c7);
    let it_p1 = b.add_i32(it, c1);
    let has_next = b.cmp_lt_i32(it_p1, cx.nkv);
    let nxt = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(has_next), a: Opnd::Val(it_p1), b: it.into() });
    let key_next = b.shl_i32(nxt, c7);
    let pre = if dma {
        Vec::new()
    } else if no_stage {
        first.clone()
    } else {
        stage_loads(b, key_next)
    };
    if dma {
        dma_issue(b, key_next, buf_nxt, kd_nxt);
    }
    // K scales of the next tile, requested now and consumed one iteration later
    let kd_next: Vec<ValueId> = if kdpf { (0..4i32).map(|s| kd_load(b, key_next, s)).collect() } else { Vec::new() };

    let compute = |b: &mut FnBuilder, masked: bool| {
        for hh in 0..2usize {
            // ---- scores S' of this 64-key substep: bias MFMA, then the two scaled K Q^T products ----
            for sl in 0..2usize {
                let s = 2 * hh + sl;
                b.var_mfma_f16_z(s_vars[sl], ones_f16, bias2);
                let ko = if dma_l { b.c_i32(512 * ((s as i32) & 1) + 8320 * ((s as i32) >> 1)) } else { b.c_i32(32 * pitch * (s as i32)) };
                let kaddr0 = b.add_i32(kf_b, ko);
                let kaddr0h = b.add_i32(kaddr0, c32v);
                let k8_0 = b.load_lds_b128x2(Opnd::Val(kaddr0), Opnd::Val(kaddr0h));
                let kaddr1 = b.add_i32(kaddr0, c64);
                let kaddr1h = b.add_i32(kaddr1, c32v);
                let k8_1 = b.load_lds_b128x2(Opnd::Val(kaddr1), Opnd::Val(kaddr1h));
                let kd = if fmha_t_opt("kdconst") {
                    Opnd::Val(b.c_i32(0x7f7f7f7f))
                } else if kdlds {
                    let koff = b.c_i32(128 * s as i32);
                    let ka = b.add_i32(kd_rd, koff);
                    Opnd::Val(b.load_lds_b32(Opnd::Val(ka)))
                } else if kdpf {
                    Opnd::Var(vars[9 + s])
                } else {
                    Opnd::Val(kd_load(b, key0, s as i32))
                };
                let sk0 = b.let_(Op::LshrI32(kd, Opnd::Val(cx.g8)));
                let sh16 = b.add_i32(cx.g8, c16i);
                let sk1 = b.let_(Op::LshrI32(kd, Opnd::Val(sh16)));
                b.var_mfma_scale(s_vars[sl], k8_0, cx.b0, sk0, sq0m);
                b.var_mfma_scale(s_vars[sl], k8_1, cx.b1, sk1, sq1m);
            }
            // key validity masks (tail tile only): invalid keys never raise the max and give P = 0
            // (recomputed at every use: keeping 32 masks live would not fit the register file)
            let mut kb: Vec<ValueId> = Vec::new();
            if masked {
                for sl in 0..2usize {
                    let s = 2 * hh + sl;
                    let kb_s0 = b.add_i32(key0, cx.g4);
                    let kb_cs = b.c_i32(32 * s as i32);
                    kb.push(b.add_i32(kb_s0, kb_cs));
                }
            }
            let valid = |b: &mut FnBuilder, sl: usize, r: u8| -> ValueId {
                let kc = b.c_i32((8 * (r / 4) + (r % 4)) as i32);
                let kidx = b.add_i32(kb[sl], kc);
                b.cmp_lt_i32(kidx, cx.kv_len)
            };
            // ---- max over the 32 scores, partner lane, overflow test ----
            let mut flat: Vec<ValueId> = Vec::with_capacity(32);
            for sl in 0..2usize {
                for r in 0..16u8 {
                    let lane = b.frag16_lane(s_vars[sl], r);
                    let v = b.bitcast_f32(lane);
                    if masked {
                        let ok = valid(b, sl, r);
                        flat.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(neg_big) }));
                    } else {
                        flat.push(v);
                    }
                }
            }
            let mut mloc = b.max3_f32(flat[0], flat[1], flat[2]);
            let mut k = 3usize;
            while k + 1 < flat.len() {
                mloc = b.max3_f32(mloc, flat[k], flat[k + 1]);
                k += 2;
            }
            if k < flat.len() {
                mloc = b.max_f32(mloc, flat[k]);
            }
            let mt = b.partner_max_f32(mloc);
            let grow = b.cmp_lt_f32(c_thr, mt);
            b.begin_if(Opnd::Val(grow));
            {
                // rollback: new reference = this lane's max (only for lanes that overflowed)
                let w_old_u = b.frag2_lane(bias2, 0);
                let w_old = b.bitcast_i32(w_old_u);
                let lo16 = b.let_(Op::AndI32(Opnd::Val(w_old), Opnd::Val(cffff)));
                let b8 = b.let_(Op::CvtF16ToF32(Opnd::Val(lo16)));
                let b_old = b.mul_f32(b8, c8f);
                let tt = b.sub_f32(c_b131, mt);
                let b_new = b.add_f32(b_old, tt);
                let q8 = b.mul_f32(b_new, c0125);
                let h_raw = b.let_(Op::CvtF32ToF16(Opnd::Val(q8)));
                let h = b.let_(Op::AndI32(Opnd::Val(h_raw), Opnd::Val(cffff)));
                let hq = b.let_(Op::CvtF16ToF32(Opnd::Val(h)));
                let b_q = b.mul_f32(hq, c8f);
                let d_full = b.sub_f32(b_q, b_old);
                let delta = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(grow), a: Opnd::Val(d_full), b: Opnd::Val(zero_f) });
                let h_sh = b.shl_i32(h, c16i);
                let w_q = b.let_(Op::AddI32(Opnd::Val(h), Opnd::Val(h_sh)));
                let w_new = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(grow), a: Opnd::Val(w_q), b: Opnd::Val(w_old) });
                let dupd = b.frag2_dup(delta);
                b.var_add_pk(s_vars[0], dupd);
                b.var_add_pk(s_vars[1], dupd);
                let ex = b.mul_f32(delta, c512f);
                let corr = b.exp2_f32(ex);
                if fmha_t_opt("svpk") {
                    let cd = b.frag2_dup(corr);
                    for e in 0..4usize {
                        b.var_scale_pk(o_vars[e], cd);
                    }
                } else {
                    for e in 0..4usize {
                        b.var_scale_f32(o_vars[e], corr);
                    }
                }
                b.var_scale_lane(la, 0, corr);
                let nb = b.frag2(w_new, w_new);
                b.set_var(bias2, nb);
            }
            b.end_if();
            // ---- P = fp8(2^(s - r) x 16) via pknorm + bf16 reinterpretation ----
            let mut words: Vec<ValueId> = Vec::with_capacity(8);
            for sl in 0..2usize {
                let mut xs: Vec<ValueId> = Vec::with_capacity(16);
                for r in 0..16u8 {
                    let lane = b.frag16_lane(s_vars[sl], r);
                    let v = b.bitcast_f32(lane);
                    if masked {
                        let ok = valid(b, sl, r);
                        xs.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(zero_f) }));
                    } else {
                        xs.push(v);
                    }
                }
                for m in 0..4usize {
                    words.push(b.cvt_pknorm_p8x4(xs[4 * m], xs[4 * m + 1], xs[4 * m + 2], xs[4 * m + 3], one_f));
                }
            }
            let p8 = b.frag8(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
            // ---- O^T += V^T P^T and the row sums (ones x P^T) ----
            if hh == 0 && stmid && !no_stage && dbuf {
                // publish the next tile early: its global loads had a whole substep to land
                stage_stores(b, sdst_n, kdb_n, &pre);
            }
            b.var_mfma_scale(la, ones8, p8, c127, c123);
            for e in 0..4usize {
                let mut addrs: Vec<ValueId> = Vec::with_capacity(4);
                for r in 0..4i32 {
                    let off = if dma_l {
                        b.c_i32((hh as i32) * 8320 + 256 * r + (e as i32) * 32)
                    } else {
                        b.c_i32((hh as i32) * 64 * pitch + 16 * pitch * r + (e as i32) * 32)
                    };
                    addrs.push(b.add_i32(tr_b, off));
                }
                if no_pv {
                    continue;
                }
                let v8 = b.load_lds_tr_b8x4(addrs[0], addrs[1], addrs[2], addrs[3]);
                b.var_mfma_scale(o_vars[e], v8, p8, c127, c123);
            }
        }
    };
    let key_end = b.add_i32(key0, c128v);
    let is_tail = b.cmp_lt_i32(cx.kv_len, key_end);
    b.begin_if(Opnd::Val(is_tail));
    compute(b, true);
    b.else_();
    compute(b, false);
    b.end_if();
    if dma {
        b.wait_vm(0);
        if fmha_t_opt("dmasync") {
            b.barrier();
            b.wait_vm(0);
        }
    } else {
        if !dbuf {
            b.wait_lds(0);
            if !no_bar {
                b.barrier();
            }
        }
        if !no_stage && !(stmid && dbuf) {
            stage_stores(b, sdst_n, kdb_n, &pre);
        }
    }
    b.wait_lds(0);
    if !no_bar {
        b.barrier();
    }
    b.wait_lds(0);
    if dbuf {
        let npar2 = b.xor_i32(par, c1);
        b.set_var(par, npar2);
    }
    if kdpf {
        for s in 0..4usize {
            b.set_var(vars[9 + s], kd_next[s]);
        }
    }
    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, cx.nkv);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();

    // ---- epilogue: O * (V descale / l) -> BF16 (l = MFMA row sum of the quantised P) ----
    let l0 = b.frag16_lane(la, 0);
    let l_tot = b.bitcast_f32(l0);
    let inv_l = b.rcp_f32(l_tot);
    let fin = b.mul_f32(inv_l, cx.vd_raw);
    let ob0 = b.mul_i32(cx.bidx, cx.o_bs);
    let ob1 = b.mul_i32(cx.hidx, cx.o_hs);
    let orow = b.mul_i32(cx.row, cx.o_seqs);
    let ob2 = b.add_i32(ob0, ob1);
    let ob3 = b.add_i32(ob2, orow);
    let ob4 = b.add_i32(ob3, cx.g_bytes8);
    let o_ptr0 = b.ptr_add(cx.ptr_o, ob4);
    let wave_live = b.cmp_lt_i32(cx.row0, cx.s_len);
    b.begin_if(Opnd::Val(wave_live));
    for e in 0..4usize {
        for q in 0..4usize {
            let mut v4: Vec<ValueId> = Vec::with_capacity(4);
            for k in 0..4usize {
                let lane = b.frag16_lane(o_vars[e], (4 * q + k) as u8);
                let lanev = b.bitcast_f32(lane);
                v4.push(b.mul_f32(lanev, fin));
            }
            let w0 = b.cvt_pk_bf16_f32(v4[0], v4[1]);
            let w1 = b.cvt_pk_bf16_f32(v4[2], v4[3]);
            let off = (64 * e + 16 * q) as u32;
            let f0 = b.bitcast_f32(w0);
            let f1 = b.bitcast_f32(w1);
            b.store_f32(o_ptr0, off, f0);
            b.store_f32(o_ptr0, off + 4, f1);
        }
    }
    b.end_if();
}
/// Transposed (`S^T = K Q^T`) variant of `fmha_fwd_hd128_mxfp8`: same ABI, same requirements, but
/// all score data lives in registers (lane = query), the online-softmax correction is lane-local
/// and LDS holds only the K and V tiles.
/// LDS row pitch (bytes) of the K/V tiles of the transposed FMHA kernel (env FMHA_PITCH, read at
/// build time, for bank-conflict experiments; default 144: 16-byte aligned and conflict-free enough, 128 = unpadded).
fn fmha_t_opt(name: &str) -> bool {
    let v = std::env::var("FMHA_OPT").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "cvt4,pksub,lazy,msum,permlane".to_string());
    let has = |n: &str| v.split(',').any(|x| x == n);
    // DMA staging needs the K scales in LDS and the two-buffer ring
    if name == "kdlds" && has("nokd") {
        return false;
    }
    has(name) || (has("dma") && (name == "dbuf" || name == "kdlds"))
}

fn fmha_t_pitch() -> i32 {
    // the DMA layout is a dense 128-byte row (conflicts are avoided by an XOR swizzle instead of padding)
    let dflt = if fmha_t_opt("dma") { 128 } else { 144 };
    std::env::var("FMHA_PITCH").ok().and_then(|v| v.parse().ok()).unwrap_or(dflt)
}
pub fn fmha_fwd_hd128_mxfp8_t() -> Function {
    fmha_t_build("fmha_fwd_hd128_mxfp8_t", false, false)
}

/// Phase-shifted ping-pong variant (see `emit_fmha_pp_role`): same ABI and numerics.
pub fn fmha_fwd_hd128_mxfp8_pp() -> Function {
    fmha_t_build("fmha_fwd_hd128_mxfp8_pp", true, false)
}

/// Vendor-style softmax variant (Schraudolph exp2 through `v_cvt_pknorm_u16_f32`, see `emit_fmha_sv_role`).
pub fn fmha_fwd_hd128_mxfp8_sv() -> Function {
    fmha_t_build("fmha_fwd_hd128_mxfp8_sv", false, true)
}

fn fmha_t_build(name: &str, pp: bool, sv: bool) -> Function {
    let mut b = FnBuilder::new(name, 512);
    let nbuf = if fmha_t_opt("dbuf") { 2 } else { 1 };
    if pp {
        b.lds((512 * fmha_t_pitch() + 2048) as u32);
    } else {
        let pad: i32 = std::env::var("FMHA_LDSPAD").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let cb = if fmha_t_opt("dma") { 33280 } else { 256 * fmha_t_pitch() };
        b.lds((nbuf * cb + 2048 + 1024 + 1024 + pad) as u32);
    }
    b.arg("ptr_o", DType::Ptr, 0x000, 8);
    b.arg("ptr_q", DType::Ptr, 0x010, 8);
    b.arg("ptr_k", DType::Ptr, 0x020, 8);
    b.arg("ptr_v", DType::Ptr, 0x030, 8);
    b.arg("s_seq_len", DType::I32, 0x060, 4);
    b.arg("s_Seqs", DType::I32, 0x070, 4);
    b.arg("s_Hs", DType::I32, 0x090, 4);
    b.arg("s_Bs", DType::I32, 0x0A0, 4);
    b.arg("s_gqa", DType::I32, 0x0B0, 4);
    b.arg("s_k_Seqs", DType::I32, 0x0C0, 4);
    b.arg("s_k_Hs", DType::I32, 0x0D0, 4);
    b.arg("s_k_Bs", DType::I32, 0x0E0, 4);
    b.arg("s_kv_seq_len", DType::I32, 0x110, 4);
    b.arg("s_q_head_num", DType::I32, 0x140, 4);
    b.arg("s_v_Seqs", DType::I32, 0x150, 4);
    b.arg("s_v_Hs", DType::I32, 0x160, 4);
    b.arg("s_v_Bs", DType::I32, 0x170, 4);
    b.arg("s_o_Seqs", DType::I32, 0x180, 4);
    b.arg("s_o_Hs", DType::I32, 0x190, 4);
    b.arg("s_o_Bs", DType::I32, 0x1A0, 4);
    b.arg("ptr_q_descale", DType::Ptr, 0x200, 8);
    b.arg("ptr_k_descale", DType::Ptr, 0x210, 8);
    b.arg("ptr_v_descale", DType::Ptr, 0x220, 8);
    b.arg("s_descale_q_Bs", DType::I32, 0x230, 4);
    b.arg("s_descale_q_Hs", DType::I32, 0x240, 4);
    b.arg("s_descale_k_Bs", DType::I32, 0x250, 4);
    b.arg("s_descale_k_Hs", DType::I32, 0x260, 4);
    b.arg("abi_tail", DType::I32, 0x280, 16);

    let ptr_o = b.let_(Op::KernargPtr { offset: 0x000 });
    let ptr_q = b.let_(Op::KernargPtr { offset: 0x010 });
    let ptr_k = b.let_(Op::KernargPtr { offset: 0x020 });
    let ptr_v = b.let_(Op::KernargPtr { offset: 0x030 });
    let s_len = b.let_(Op::KernargI32 { offset: 0x060 });
    let gqa = b.let_(Op::KernargI32 { offset: 0x0B0 });
    let q_seqs = b.let_(Op::KernargI32 { offset: 0x070 });
    let q_hs = b.let_(Op::KernargI32 { offset: 0x090 });
    let q_bs = b.let_(Op::KernargI32 { offset: 0x0A0 });
    let k_seqs = b.let_(Op::KernargI32 { offset: 0x0C0 });
    let k_hs = b.let_(Op::KernargI32 { offset: 0x0D0 });
    let k_bs = b.let_(Op::KernargI32 { offset: 0x0E0 });
    let kv_len = b.let_(Op::KernargI32 { offset: 0x110 });
    let v_seqs = b.let_(Op::KernargI32 { offset: 0x150 });
    let v_hs = b.let_(Op::KernargI32 { offset: 0x160 });
    let v_bs = b.let_(Op::KernargI32 { offset: 0x170 });
    let o_seqs = b.let_(Op::KernargI32 { offset: 0x180 });
    let o_hs = b.let_(Op::KernargI32 { offset: 0x190 });
    let o_bs = b.let_(Op::KernargI32 { offset: 0x1A0 });
    let ptr_qd = b.let_(Op::KernargPtr { offset: 0x200 });
    let ptr_kd = b.let_(Op::KernargPtr { offset: 0x210 });
    let ptr_vd = b.let_(Op::KernargPtr { offset: 0x220 });
    let dq_bs = b.let_(Op::KernargI32 { offset: 0x230 });
    let dq_hs = b.let_(Op::KernargI32 { offset: 0x240 });
    let dk_bs = b.let_(Op::KernargI32 { offset: 0x250 });
    let dk_hs = b.let_(Op::KernargI32 { offset: 0x260 });

    let tid_full = b.let_(Op::WorkitemId);
    let qtile = b.let_(Op::WorkgroupIdX);
    let hidx = b.let_(Op::WorkgroupIdY);
    let bidx = b.let_(Op::WorkgroupIdZ);

    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c5 = b.c_i32(5);
    let c6 = b.c_i32(6);
    let c7 = b.c_i32(7);
    let c8 = b.c_i32(8);
    let c15 = b.c_i32(15);
    let c16i = b.c_i32(16);
    let c31 = b.c_i32(31);
    let c32 = b.c_i32(32);
    let c63 = b.c_i32(63);
    let c128 = b.c_i32(128);
    let c255 = b.c_i32(255);
    let c256 = b.c_i32(256);

    let tid = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c63)));
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid_full), Opnd::Val(c6)));
    let j = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c31)));
    let pblk = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c5)));
    let g8 = b.shl_i32(pblk, c3);
    let g16 = b.shl_i32(pblk, c4);
    let g_bytes8 = b.shl_i32(pblk, c3);
    let lds_k = b.c_i32(0);
    let dma_l = fmha_t_opt("dma");
    let lds_v = if dma_l { b.c_i32(16640) } else { b.c_i32(if pp { 256 } else { 128 } * fmha_t_pitch()) };
    let lds_ex = if dma_l { b.c_i32(nbuf * 33280) } else { b.c_i32(if pp { 512 } else { nbuf * 256 } * fmha_t_pitch()) };

    // per-wave exchange slots (own lane, partner lane l ^ 32)
    let wave256 = b.mul_i32(wave, c256);
    let tid4 = b.mul_i32(tid, c4);
    let ex0 = b.add_i32(lds_ex, wave256);
    let ex_own = b.add_i32(ex0, tid4);
    let tid_x = b.xor_i32(tid, c32);
    let tid_x4 = b.mul_i32(tid_x, c4);
    let ex_partner = b.add_i32(ex0, tid_x4);

    let wave32 = b.shl_i32(wave, c5);
    let qt256 = b.shl_i32(qtile, c8);
    let row0 = b.add_i32(qt256, wave32);
    let row_raw = b.add_i32(row0, j);
    // rows past the end of a partial query tile reuse the last valid row: they compute and store
    // exactly its (identical) result instead of touching memory out of bounds
    let cm1 = b.c_i32(-1);
    let s_last = b.add_i32(s_len, cm1);
    let row_ok = b.cmp_lt_i32(row_raw, s_len);
    let ablate_pre = std::env::var("FMHA_ABLATE").unwrap_or_default();
    let row = if ablate_pre.contains("noclamp") {
        row_raw
    } else {
        b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_ok), a: Opnd::Val(row_raw), b: Opnd::Val(s_last) })
    };
    let q_row_off = b.mul_i32(row, q_seqs);
    let q_b_off = b.mul_i32(bidx, q_bs);
    let q_h_off = b.mul_i32(hidx, q_hs);
    let q_off0 = b.add_i32(q_row_off, q_b_off);
    let q_off1 = b.add_i32(q_off0, q_h_off);
    let q_off = b.add_i32(q_off1, g16);

    // Q fragments (B operand of S^T = K Q^T): same permuted K order as the K fragments
    let mut bq: Vec<ValueId> = Vec::with_capacity(2);
    for c in 0..2u32 {
        let qp = b.ptr_add(ptr_q, q_off);
        let mut w: Vec<ValueId> = Vec::with_capacity(8);
        for i in 0..8u32 {
            let off = 64 * c + if i < 4 { 4 * i } else { 32 + 4 * (i - 4) };
            let v = b.load_f32(qp, off);
            w.push(b.bitcast_i32(v));
        }
        bq.push(b.frag8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]));
    }
    let c5b = b.c_i32(5);
    let qd_stride = b.let_(Op::LshrI32(Opnd::Val(q_seqs), Opnd::Val(c5b)));
    let kd_stride = b.let_(Op::LshrI32(Opnd::Val(k_seqs), Opnd::Val(c5b)));
    let qd_row = b.mul_i32(row, qd_stride);
    let qd_b = b.mul_i32(bidx, dq_bs);
    let qd_h = b.mul_i32(hidx, dq_hs);
    let qd0 = b.add_i32(qd_row, qd_b);
    let qd1 = b.add_i32(qd0, qd_h);
    let qd_ptr = b.ptr_add(ptr_qd, qd1);
    let qd_raw = b.load_f32(qd_ptr, 0);
    let qd = b.bitcast_i32(qd_raw);
    let sq0 = b.let_(Op::LshrI32(Opnd::Val(qd), Opnd::Val(g8)));
    let sh16 = b.add_i32(g8, c16i);
    let sq1 = b.let_(Op::LshrI32(Opnd::Val(qd), Opnd::Val(sh16)));
    let vd_raw = b.load_f32(ptr_vd, 0);

    // KV head of this query head (GQA ratio is a power of two <= 16)
    let gq2 = b.cmp_lt_i32(c1, gqa);
    let gq4 = b.cmp_lt_i32(c2, gqa);
    let gq8 = b.cmp_lt_i32(c4, gqa);
    let gq16 = b.cmp_lt_i32(c8, gqa);
    let c0k = b.c_i32(0);
    let s1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gq2), a: Opnd::Val(c1), b: Opnd::Val(c0k) });
    let s2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gq4), a: Opnd::Val(c2), b: Opnd::Val(s1) });
    let s3 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gq8), a: Opnd::Val(c3), b: Opnd::Val(s2) });
    let gshift = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gq16), a: Opnd::Val(c4), b: Opnd::Val(s3) });
    let kvh = if ablate_pre.contains("nogqa") { hidx } else { b.let_(Op::LshrI32(Opnd::Val(hidx), Opnd::Val(gshift))) };
    let kb0 = b.mul_i32(bidx, k_bs);
    let kb1 = b.mul_i32(kvh, k_hs);
    let kbase_off = b.add_i32(kb0, kb1);
    let vb0 = b.mul_i32(bidx, v_bs);
    let vb1 = b.mul_i32(kvh, v_hs);
    let vbase_off = b.add_i32(vb0, vb1);
    let kd_b = b.mul_i32(bidx, dk_bs);
    let kd_h = b.mul_i32(kvh, dk_hs);
    let kd_base = b.add_i32(kd_b, kd_h);

    let rt = b.let_(Op::AndI32(Opnd::Val(tid_full), Opnd::Val(c255)));
    let rt_row = b.let_(Op::LshrI32(Opnd::Val(rt), Opnd::Val(c1)));
    let rt_half = b.let_(Op::AndI32(Opnd::Val(rt), Opnd::Val(c1)));
    let rt_half64 = b.shl_i32(rt_half, c6);
    let pitch_c = b.c_i32(fmha_t_pitch());
    let rt_rowp = b.mul_i32(rt_row, pitch_c);
    let rt64 = b.add_i32(rt_rowp, rt_half64);
    let is_a = b.cmp_lt_i32(wave, c4);

    // transposing-read lane address for A' = V^T (key of byte b: 16 r + 8 (i/4) + 4 g + i % 4):
    //   key_part = 8 * ((M & 15) >> 3) + 4 g + ((M >> 1) & 3); col_part = 16 * ((M >> 4) & 1) + 8 * (M & 1)
    let m15 = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c15)));
    let m_h3 = b.let_(Op::LshrI32(Opnd::Val(m15), Opnd::Val(c3)));
    let key_a = b.shl_i32(m_h3, c3);
    let key_b = b.shl_i32(pblk, c2);
    let m_s1 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c1)));
    let key_c = b.let_(Op::AndI32(Opnd::Val(m_s1), Opnd::Val(c3)));
    let key0p = b.add_i32(key_a, key_b);
    let key_part = b.add_i32(key0p, key_c);
    let key_off = if fmha_t_opt("dma") {
        let c1040 = b.c_i32(1040);
        let ku = b.let_(Op::AndI32(Opnd::Val(key_part), Opnd::Val(c7)));
        let kv = b.mul_i32(ku, c1040);
        let kh = b.let_(Op::LshrI32(Opnd::Val(key_part), Opnd::Val(c3)));
        let kh128 = b.mul_i32(kh, c128);
        b.add_i32(kv, kh128)
    } else {
        b.mul_i32(key_part, pitch_c)
    };
    let m_s4 = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c4)));
    let m_gl = b.let_(Op::AndI32(Opnd::Val(m_s4), Opnd::Val(c1)));
    let m_h = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1)));
    let col_a = b.shl_i32(m_gl, c4);
    let col_b = b.shl_i32(m_h, c3);
    let col_part = b.add_i32(col_a, col_b);
    let tr0 = b.add_i32(key_off, col_part);
    let tr_base = b.add_i32(tr0, lds_v);
    // DMA layout (vendor style): 1040-byte runs of eight rows {u + 8 s}: row R at ((R & 7) + 8 (R >> 6)) 1040 + ((R >> 3) & 7) 128
    let j128 = if fmha_t_opt("dma") {
        let c1040 = b.c_i32(1040);
        let ju = b.let_(Op::AndI32(Opnd::Val(j), Opnd::Val(c7)));
        let jv = b.mul_i32(ju, c1040);
        let jh = b.let_(Op::LshrI32(Opnd::Val(j), Opnd::Val(c3)));
        let jh128 = b.mul_i32(jh, c128);
        b.add_i32(jv, jh128)
    } else {
        b.mul_i32(j, pitch_c)
    };
    let kf0 = b.add_i32(j128, g16);
    let kf_base = b.add_i32(kf0, lds_k);
    let c127p = b.c_i32(127);
    let kv_len_r = b.add_i32(kv_len, c127p);
    let nkv = b.let_(Op::LshrI32(Opnd::Val(kv_len_r), Opnd::Val(c7)));
    let cm1k = b.c_i32(-1);
    let kv_last = b.add_i32(kv_len, cm1k);
    let g4 = b.shl_i32(pblk, c2);
    let wave4 = b.let_(Op::AndI32(Opnd::Val(wave), Opnd::Val(c3)));
    let mut kfx: Vec<ValueId> = Vec::new();
    let mut trx: Vec<ValueId> = Vec::new();
    if false {
        let j_s1 = b.let_(Op::LshrI32(Opnd::Val(j), Opnd::Val(c1)));
        let f_k0 = b.let_(Op::AndI32(Opnd::Val(j_s1), Opnd::Val(c7)));
        let zero_sw = b.c_i32(0);
        let f_k = if fmha_t_opt("noswz") { zero_sw } else { f_k0 };
        let gbit = pblk;
        let x = b.xor_i32(gbit, f_k);
        let jrow = b.mul_i32(j, c128);
        for k in 0..4i32 {
            let c2k = b.c_i32(2 * k);
            let xk = b.xor_i32(x, c2k);
            let xs = b.shl_i32(xk, c4);
            let a0 = b.add_i32(jrow, xs);
            kfx.push(b.add_i32(a0, lds_k));
        }
        let kp_s1 = b.let_(Op::LshrI32(Opnd::Val(key_part), Opnd::Val(c1)));
        let f_t0 = b.let_(Op::AndI32(Opnd::Val(kp_s1), Opnd::Val(c7)));
        let f_t = if fmha_t_opt("noswz") { zero_sw } else { f_t0 };
        let y = b.xor_i32(m_gl, f_t);
        let trow = b.mul_i32(key_part, c128);
        let mh8 = b.shl_i32(m_h, c3);
        for e in 0..4i32 {
            let c2e = b.c_i32(2 * e);
            let ye = b.xor_i32(y, c2e);
            let ys = b.shl_i32(ye, c4);
            let a0 = b.add_i32(trow, ys);
            let a1 = b.add_i32(a0, mh8);
            trx.push(b.add_i32(a1, lds_v));
        }
    }

    let cx = FmhaTCtx {
        ptr_o,
        ptr_k,
        ptr_v,
        ptr_kd,
        k_seqs,
        v_seqs,
        kd_stride,
        kv_len,
        kv_last,
        g4,
        o_seqs,
        o_hs,
        o_bs,
        bidx,
        hidx,
        row,
        row0,
        s_len,
        g8,
        g_bytes8,
        j,
        kbase_off,
        vbase_off,
        kd_base,
        rt_row,
        rt_half64,
        rt64,
        tr_base,
        kf_base,
        ex_own,
        ex_partner,
        b0: bq[0],
        b1: bq[1],
        sq0,
        sq1,
        vd_raw,
        nkv,
        lds_k,
        lds_v,
        wave4,
        kfx,
        trx,
    };
    b.begin_if(is_a);
    if pp {
        emit_fmha_pp_role(&mut b, &cx, true);
    } else if sv {
        emit_fmha_sv_role(&mut b, &cx, true);
    } else {
        emit_fmha_t_role(&mut b, &cx, true);
    }
    b.else_();
    if pp {
        emit_fmha_pp_role(&mut b, &cx, false);
    } else if sv {
        emit_fmha_sv_role(&mut b, &cx, false);
    } else {
        emit_fmha_t_role(&mut b, &cx, false);
    }
    b.end_if();
    b.finish()
}

// ---------------------------------------------------------------------------------------------
// sched_* microkernels: small loops with a runtime trip count `n` (kernarg offset 8) used to compare
// the scheduler's machine model with hardware and to exercise cross-block / cross-iteration
// scheduling (docs/scheduling.md). Output: per-thread results at o[tid * 64 ..].
// ---------------------------------------------------------------------------------------------

fn sched_begin(name: &str) -> (FnBuilder, ValueId, ValueId, ValueId) {
    let mut b = FnBuilder::new(name, 512);
    b.lds(16384);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("count", DType::I32, 8, 4);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let n = b.let_(Op::KernargI32 { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    (b, o, n, tid)
}

/// Close a counted loop: it += 1; break unless it < n.
fn sched_end_loop(b: &mut FnBuilder, it: VarId, n: ValueId) {
    let c1 = b.c_i32(1);
    let it_next = b.add_i32(it, c1);
    b.set_var(it, it_next);
    let more = b.cmp_lt_i32(it_next, n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();
}

/// 16 independent chains of one VALU op per iteration. kind: 0 = v_mul_f32, 1 = v_max3_f32, 2 = v_exp_f32.
fn sched_valu_kernel(name: &str, kind: u32) -> Function {
    let (mut b, o, n, tid) = sched_begin(name);
    let c0 = b.c_i32(0);
    let c64 = b.c_i32(64);
    let step = b.c_f32(1.0001);
    let step_i = b.bitcast_i32(step);
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::Val(c0))];
    for k in 0..16 {
        let f = b.c_f32(0.5 + k as f32);
        let fi = b.bitcast_i32(f);
        inits.push((DType::I32, Opnd::Val(fi)));
    }
    let vars = b.begin_loop(&inits);
    let it = vars[0];
    let stepf = b.bitcast_f32(step_i);
    for k in 0..16 {
        let x = b.bitcast_f32(vars[1 + k]);
        let y = match kind {
            0 => b.mul_f32(x, stepf),
            1 => b.max3_f32(x, stepf, stepf),
            _ => b.exp2_f32(x),
        };
        let yi = b.bitcast_i32(y);
        b.set_var(vars[1 + k], yi);
    }
    sched_end_loop(&mut b, it, n);
    let off = b.mul_i32(tid, c64);
    let row = b.ptr_add(o, off);
    for k in 0..16 {
        let x = b.bitcast_f32(vars[1 + k]);
        b.store_f32(row, 4 * k as u32, x);
    }
    b.finish()
}

pub fn sched_valu_mul() -> Function {
    sched_valu_kernel("sched_valu_mul", 0)
}
pub fn sched_valu_max3() -> Function {
    sched_valu_kernel("sched_valu_max3", 1)
}
pub fn sched_valu_exp() -> Function {
    sched_valu_kernel("sched_valu_exp", 2)
}

/// In-place scaled MFMA chain, no VALU.
pub fn sched_mfma_chain() -> Function {
    let (mut b, o, n, tid) = sched_begin("sched_mfma_chain");
    let c0 = b.c_i32(0);
    let c64 = b.c_i32(64);
    let ones = b.c_i32(0x38383838_u32 as i32);
    let f4 = b.frag4(ones, ones, ones, ones);
    let a8 = b.cat_frag8(f4, f4);
    let twos = b.c_i32(0x40404040_u32 as i32);
    let f4b = b.frag4(twos, twos, twos, twos);
    let b8 = b.cat_frag8(f4b, f4b);
    let zero = b.frag_zero(DType::Frag16);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::Frag16, Opnd::Val(zero))]);
    b.var_mfma_scale(vars[1], a8, b8, 127, 127);
    sched_end_loop(&mut b, vars[0], n);
    let off = b.mul_i32(tid, c64);
    let row = b.ptr_add(o, off);
    for i in 0..16u8 {
        let l = b.frag16_lane(vars[1], i);
        let x = b.bitcast_f32(l);
        b.store_f32(row, 4 * i as u32, x);
    }
    b.finish()
}

/// Each iteration: S = A x B (zero accumulator), then a VALU post-processing chain over `lanes` of S
/// (sub, exp2, add into a loop-carried sum). Iterations are independent apart from the sum: the
/// MFMA of iteration i+1 can overlap the VALU work of iteration i (software pipelining target).
fn sched_mfma_valu_kernel(name: &str, lanes: u8) -> Function {
    let (mut b, o, n, tid) = sched_begin(name);
    let c0 = b.c_i32(0);
    let c64 = b.c_i32(64);
    let ones = b.c_i32(0x38383838_u32 as i32);
    let f4 = b.frag4(ones, ones, ones, ones);
    let a8 = b.cat_frag8(f4, f4);
    let twos = b.c_i32(0x40404040_u32 as i32);
    let f4b = b.frag4(twos, twos, twos, twos);
    let b8 = b.cat_frag8(f4b, f4b);
    let zf = b.c_f32(0.0);
    let zi = b.bitcast_i32(zf);
    let m = b.c_f32(1.5);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(zi))]);
    let zero = b.frag_zero(DType::Frag16);
    let s = b.mfma_scale_fp8(a8, b8, zero, 127, 127);
    let mut sum = b.bitcast_f32(vars[1]);
    for r in 0..lanes {
        let l = b.frag16_lane(s, r);
        let v = b.bitcast_f32(l);
        let d = b.sub_f32(v, m);
        let e = b.exp2_f32(d);
        sum = b.add_f32(sum, e);
    }
    let si = b.bitcast_i32(sum);
    b.set_var(vars[1], si);
    sched_end_loop(&mut b, vars[0], n);
    let off = b.mul_i32(tid, c64);
    let row = b.ptr_add(o, off);
    let fin = b.bitcast_f32(vars[1]);
    b.store_f32(row, 0, fin);
    b.finish()
}

pub fn sched_mfma_valu16() -> Function {
    sched_mfma_valu_kernel("sched_mfma_valu16", 16)
}
pub fn sched_mfma_valu4() -> Function {
    sched_mfma_valu_kernel("sched_mfma_valu4", 4)
}

/// Loop with an if/else diamond: each arm computes an MFMA and its own consumer chain; the join then does
/// independent VALU work (a second chain on loop-carried state) that can fill the arm's MFMA latency
/// when the join prefix is duplicated into the arms (stage 2).
pub fn sched_diamond() -> Function {
    let (mut b, o, n, tid) = sched_begin("sched_diamond");
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c64 = b.c_i32(64);
    let ones = b.c_i32(0x38383838_u32 as i32);
    let f4 = b.frag4(ones, ones, ones, ones);
    let a8 = b.cat_frag8(f4, f4);
    let twos = b.c_i32(0x40404040_u32 as i32);
    let f4b = b.frag4(twos, twos, twos, twos);
    let b8 = b.cat_frag8(f4b, f4b);
    let zf = b.c_f32(0.0);
    let zi = b.bitcast_i32(zf);
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(zi))];
    for k in 0..8 {
        let f = b.c_f32(1.0 + k as f32);
        let fi = b.bitcast_i32(f);
        inits.push((DType::I32, Opnd::Val(fi)));
    }
    let vars = b.begin_loop(&inits);
    let it = vars[0];
    let m = b.c_f32(1.5);
    // arm result goes into loop var 1 (sum)
    let odd = b.let_(Op::AndI32(Opnd::Var(it), Opnd::Val(c1)));
    let zero_i = b.c_i32(0);
    let cond = b.cmp_lt_i32(zero_i, odd);
    b.begin_if(Opnd::Val(cond));
    {
        let z = b.frag_zero(DType::Frag16);
        let s = b.mfma_scale_fp8(a8, b8, z, 127, 127);
        let mut sum = b.bitcast_f32(vars[1]);
        for r in 0..4u8 {
            let l = b.frag16_lane(s, r);
            let v = b.bitcast_f32(l);
            let d = b.sub_f32(v, m);
            let e = b.exp2_f32(d);
            sum = b.add_f32(sum, e);
        }
        let si = b.bitcast_i32(sum);
        b.set_var(vars[1], si);
    }
    b.else_();
    {
        let z = b.frag_zero(DType::Frag16);
        let s = b.mfma_scale_fp8(b8, a8, z, 127, 127);
        let mut sum = b.bitcast_f32(vars[1]);
        for r in 0..4u8 {
            let l = b.frag16_lane(s, r);
            let v = b.bitcast_f32(l);
            let d = b.add_f32(v, m);
            let e = b.exp2_f32(d);
            sum = b.add_f32(sum, e);
        }
        let si = b.bitcast_i32(sum);
        b.set_var(vars[1], si);
    }
    b.end_if();
    // join: independent chains on loop-carried state
    for k in 0..8 {
        let x = b.bitcast_f32(vars[2 + k]);
        let y = b.mul_f32(x, m);
        let y2 = b.add_f32(y, m);
        let yi = b.bitcast_i32(y2);
        b.set_var(vars[2 + k], yi);
    }
    sched_end_loop(&mut b, it, n);
    let off = b.mul_i32(tid, c64);
    let row = b.ptr_add(o, off);
    let fin = b.bitcast_f32(vars[1]);
    b.store_f32(row, 0, fin);
    for k in 0..8 {
        let x = b.bitcast_f32(vars[2 + k]);
        b.store_f32(row, 4 + 4 * k as u32, x);
    }
    b.finish()
}