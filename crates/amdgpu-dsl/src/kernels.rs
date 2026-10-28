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