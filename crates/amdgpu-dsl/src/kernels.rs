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
/// runs two iterations; inside it there are two sequential inner loops (an A
/// phase and a B phase), each of which writes a phase-dependent value into LDS,
/// synchronises, reads the partner lane's slot, accumulates it, and synchronises
/// again. Nested loops, loop-carried accumulators, role-dependent values and
/// paired barriers are all exercised, with a closed-form expected result.
pub fn probe_ab_loops() -> Function {
    let mut b = FnBuilder::new("probe_ab_loops", 512);
    b.lds(2048);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);

    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    let c4 = b.c_i32(4);
    let c6 = b.c_i32(6);
    let c10 = b.c_i32(10);
    let c20 = b.c_i32(20);
    let c1000 = b.c_i32(1000);
    let c2000 = b.c_i32(2000);
    let c256 = b.c_i32(256);
    let c512 = b.c_i32(512);
    let c1024 = b.c_i32(1024);

    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let is_a = b.cmp_lt_i32(wave, c4);
    let off = b.mul_i32(tid, c4);
    let partner = b.let_(Op::XorI32(Opnd::Val(tid), Opnd::Val(c256)));
    let partner_off = b.mul_i32(partner, c4);

    // Outer loop carries the accumulator and the outer index.
    let zero = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(zero)), (DType::I32, Opnd::Val(zero))]);
    let acc = vars[0];
    let outer = vars[1];

    // ---- A phase -----------------------------------------------------------
    let ia_init = b.c_i32(0);
    let inner = b.begin_loop(&[(DType::I32, Opnd::Val(ia_init))]);
    let ia = inner[0];

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
    let acc_a = b.add_i32(acc, got_a);
    b.set_var(acc, acc_a);
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
    let acc_b = b.add_i32(acc, got_b);
    b.set_var(acc, acc_b);
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

    // Store the final accumulator: o[tid*4] (raw bits).
    let out_off = b.mul_i32(tid, c4);
    let out_ptr = b.ptr_add(o, out_off);
    let value = b.bitcast_f32(acc);
    b.store_f32(out_ptr, 0, Opnd::Val(value));
    let _ = (c512, c1024);
    b.finish()
}