//! Kernel definitions written against the typed DSL.
//!
//! `attention_fwd` is a real (if deliberately simple) dense attention kernel:
//! one wave per block, one lane per query row, online softmax over the key
//! dimension, f32 accumulation and f16 output. It is the first correctness
//! milestone; tiled MFMA/PV variants are separate kernels added later.

use amdgpu_dsl::*;

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
    // GPU-semantics: the output holds 4 iterations x 512 lanes x 4 bytes; one 2 KB exchange tile (v1.1): both wave groups write their own half (same version =
    // cooperative fill) and read the partner's data one barrier later
    b.arg_tensor("o", Ext::c(4 * 512 * 4));
    b.lds_tile("xchg", 0, 2048);
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
    b.assume_trusted(iter, Ext::c(0), Ext::c(3), "counter bounded by the loop exit test (next < 4)");

    let scaled = b.mul_i32(iter, c1000);
    let base = b.add_i32(scaled, tid);
    b.begin_if(Opnd::Val(is_a));
    b.tile_produce("xchg", Opnd::I32(0), Opnd::Var(iter));
    b.store_lds_b32(Opnd::Val(lds_off), Opnd::Val(base));
    b.tile_end();
    b.else_();
    let marked = b.add_i32(base, c100000);
    b.tile_produce("xchg", Opnd::I32(0), Opnd::Var(iter));
    b.store_lds_b32(Opnd::Val(lds_off), Opnd::Val(marked));
    b.tile_end();
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);

    b.tile_consume("xchg", Opnd::I32(0), Opnd::Var(iter));
    let got = b.load_lds_b32(Opnd::Val(partner_off));
    b.tile_end();
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
    // PD_NEW=1: one M0 per wave (run 1040 (4 wave + q) via signed instruction offsets, compensated source address),
    // readback at LDS 1040 rho into o[1024 rho ..]
    let new_mode = std::env::var("PD_NEW").map(|v| !v.is_empty()).unwrap_or(false);
    let mut b = FnBuilder::new("probe_dma_tile", 256);
    b.lds(std::env::var("PD_LDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20000));
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("i", DType::Ptr, 8, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let inp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let c6 = b.c_i32(6);
    let c63 = b.c_i32(63);
    let c16 = b.c_i32(16);
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let l16 = b.mul_i32(lane, c16);
    if new_mode {
        let c1024 = b.c_i32(1024);
        let c4160 = b.c_i32(4160);
        let c3120 = b.c_i32(3120);
        let w4 = b.mul_i32(wave, c4);
        let w4k = b.mul_i32(w4, c1024);
        let wsrc = b.add_i32(w4k, l16);
        let w4160 = b.mul_i32(wave, c4160);
        let top = b.add_i32(w4160, c3120);
        for q in 0..4i32 {
            let cq = b.c_i32(1024 * q + 1040 * (3 - q));
            let off = b.add_i32(wsrc, cq);
            let sp = b.ptr_add(inp, off);
            b.load_global_lds(sp, top, -(1040 * (3 - q)), 4);
        }
        b.wait_vm(0);
        b.wait_lds(0);
        b.barrier();
        // thread t reads 4 x 16 B of run rho = 4 wave + q at LDS 1040 rho + 16 lane
        let c1040 = b.c_i32(1040);
        for q in 0..4i32 {
            let cq = b.c_i32(q);
            let rho = b.add_i32(w4, cq);
            let la = b.mul_i32(rho, c1040);
            let a = b.add_i32(la, l16);
            let v = b.load_lds_b128(Opnd::Val(a));
            let r1024 = b.mul_i32(rho, c1024);
            let od = b.add_i32(r1024, l16);
            let op = b.ptr_add(o, od);
            for w in 0..4u8 {
                let x = b.frag4_lane(v, w);
                let f = b.bitcast_f32(x);
                b.store_f32(op, 4 * w as u32, f);
            }
        }
        return b.finish();
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
    let swz = swz_mode != 0 && swz_mode != 4 && swz_mode != 5;
    let row_stride: i32 = std::env::var("PD_ROWSTRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let head_off: i32 = std::env::var("PD_HEADOFF").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
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
    if std::env::var("PD_NODMA").map(|v| !v.is_empty()).unwrap_or(false) {
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
    if swz_mode == 5 {
        // kernel scheme: wave w owns runs rho = 4 w + q (rows (4 (w & 1) + q) + 8 s + 64 (w >> 1)); one M0 per wave, run offsets via
        // the signed instruction offset with a compensated source address
        let cb4 = b.shl_i32(lane_lo, c4);
        let s8 = b.shl_i32(lane_hi, c3);
        let w_hi = b.let_(Op::LshrI32(Opnd::Val(wave), Opnd::Val(c1)));
        let w_hi64 = b.shl_i32(w_hi, c6);
        let w_lo4 = w4;
        let rb0 = b.add_i32(s8, w_hi64);
        let rb4 = b.add_i32(rb0, w_lo4);
        let c4160 = b.c_i32(4160);
        let w4160 = b.mul_i32(wave, c4160);
        let c3120 = b.c_i32(3120);
        let top = b.add_i32(w4160, c3120);
        let crs = b.c_i32(row_stride);
        let cho = b.c_i32(head_off);
        for q in 0..4i32 {
            let cq = b.c_i32(q);
            let row = b.add_i32(rb4, cq);
            let r0 = b.mul_i32(row, crs);
            let r1 = b.add_i32(r0, cho);
            let off = b.add_i32(r1, cb4);
            let comp = b.c_i32(1040 * (3 - q));
            let off2 = b.add_i32(off, comp);
            let sp = b.ptr_add(inp, off2);
            b.load_global_lds(sp, top, -(1040 * (3 - q)), 4);
        }
    } else if swz_mode == 4 {
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
            b.load_global_lds(sp, w1040, 0, 4);
        }
    } else {
    for q in 0..4i32 {
        let cq = b.c_i32(32 * q);
        let row = b.add_i32(rowb, cq);
        let r128 = b.mul_i32(row, c128);
        let off = b.add_i32(r128, cbytes);
        let sp = b.ptr_add(inp, off);
        b.load_global_lds(sp, w1024, 0, 4);
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
    if std::env::var("PD_B2").map(|v| !v.is_empty()).unwrap_or(false) {
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
        let a = if swz_mode == 4 || swz_mode == 5 { b.add_i32(base4, xs) } else { b.add_i32(jrow, xs) };
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
            let rs: i32 = std::env::var("PD_MAPSTRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
            let c1024 = b.c_i32(rs);
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

/// Transposed (`S^T = K Q^T`) variant of `fmha_fwd_hd128_mxfp8`: same ABI, same requirements, but
/// all score data lives in registers (lane = query), the online-softmax correction is lane-local
/// and LDS holds only the K and V tiles.
/// LDS row pitch (bytes) of the K/V tiles of the transposed FMHA kernel (env FMHA_PITCH, read at
/// build time, for bank-conflict experiments; default 144: 16-byte aligned and conflict-free enough, 128 = unpadded).
/// End-to-end probe for the safe-asm path: loads a float4 per lane, MFMA + VALU, stores one float per lane (all written in safe Rust, see
/// `amdgpu_safeasm::examples::global_probe_block`). out[tid] = in[4*tid..4*tid+4].sum() + 64.0.
/// Memory-order probe for the fused-combine (last-arriver) pattern. `mode`: 0 = full fences (`fence_release` + `fence_acquire`: wait, `buffer_wbl2 sc1`,
/// wait / wait, `buffer_inv sc1`), 1 = lite (waits only, no L2 write-back / invalidate), 2 = none (no fence at all: expected to be able to fail; it proves
/// the probe can detect a violation). Modes 3 and 4 use PLAIN stores/loads (no sc1): 3 = no fence (must be able to fail across XCDs), 4 = full fences
/// (`buffer_wbl2 sc1` / `buffer_inv sc1`) only, the LLVM agent-scope release/acquire sequence. Every lane of every 64-thread block writes `(bid+1)*1000+tid` to `part[bid*64+tid]` with a write-through store,
/// releases, does `old = atomic_add(counter, 1)` (per lane: the lane that gets `old == nblocks*64-1` is the last arriver in the whole grid), and that
/// wave acquires and sums all partials with coherent loads into `result[0]`.
fn probe_last_arriver_mode(name: &str, mode: u8) -> Function {
    let mut b = FnBuilder::new(name, 64);
    b.arg("part", DType::Ptr, 0, 8);
    b.arg("counter", DType::Ptr, 8, 8);
    b.arg("result", DType::Ptr, 16, 8);
    b.arg("nblocks", DType::I32, 24, 4);
    b.arg("seed", DType::I32, 28, 4);
    // GPU-semantics (v4 protocol): `part` is a global region written by every block (generation = split id = block id) and read in full by the LAST ARRIVER only
    b.ka_range(24, 1, i64::MAX);
    b.grid_is(0, Ext::ka(24));
    b.arg_tensor("part", Ext::ka(24) * Ext::c(256));
    b.arg_tensor("counter", Ext::c(4));
    b.arg_tensor("result", Ext::c(4));
    b.buf_region("part", "part", Ext::c(0), Ext::ka(24) * Ext::c(256));
    b.buf_last_arriver("part", "counter", Ext::ka(24));
    let part = b.let_(Op::KernargPtr { offset: 0 });
    let counter = b.let_(Op::KernargPtr { offset: 8 });
    let result = b.let_(Op::KernargPtr { offset: 16 });
    let nb = b.let_(Op::KernargI32 { offset: 24 });
    let seed = b.let_(Op::KernargI32 { offset: 28 });
    let tid = b.let_(Op::WorkitemId);
    let bid = b.let_(Op::WorkgroupIdX);
    let c64 = b.c_i32(64);
    let c1000 = b.c_i32(1000);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let cm1 = b.c_i32(-1);
    let row = b.mul_i32(bid, c64);
    let idx = b.add_i32(row, tid);
    let bp1 = b.add_i32(bid, c1);
    let val = b.mul_i32(bp1, c1000);
    let val = b.add_i32(val, tid);
    let val = b.add_i32(val, seed); // different data every run: stale cache lines from an earlier run cannot match by accident
    let valf = b.bitcast_f32(Opnd::Val(val));
    let off = b.mul_i32(idx, c4);
    let pp = b.ptr_add(part, off);
    b.buf_produce("part", bid);
    if mode >= 3 {
        b.store_f32(pp, 0, Opnd::Val(valf));
    } else {
        b.store_f32_wt(pp, 0, Opnd::Val(valf));
    }
    b.tile_end();
    match mode {
        0 | 4 => b.fence_release(),
        1 => b.fence_release_lite(),
        _ => {}
    }
    let old = b.atomic_add_u32(counter, 0, Opnd::Val(c1));
    let total = b.mul_i32(nb, c64);
    let last_ticket = b.add_i32(total, cm1);
    let is_last = b.let_(Op::CmpGeI32(Opnd::Val(old), Opnd::Val(last_ticket)));
    b.begin_if(Opnd::Val(is_last));
    match mode {
        0 | 4 => b.fence_acquire(),
        1 => b.fence_acquire_lite(),
        _ => {}
    }
    let zero = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(zero)), (DType::I32, Opnd::Val(zero))]);
    let (i, s) = (vars[0], vars[1]);
    b.assume_trusted(i, Ext::c(0), Ext::ka(24) * Ext::c(64) - Ext::c(1), "the loop exits once i + 1 >= nblocks * 64, so the body only runs with i <= nblocks * 64 - 1");
    let io = b.mul_i32(i, c4);
    let q = b.ptr_add(part, io);
    let slice = b.let_(Op::LshrI32(Opnd::Var(i), Opnd::I32(6)));
    b.buf_consume("part", slice);
    let x = if mode >= 3 { b.load_f32(q, 0) } else { b.load_f32_coherent(q, 0) };
    b.tile_end();
    let xi = b.bitcast_i32(Opnd::Val(x));
    let s2 = b.add_i32(s, xi);
    b.set_var(s, Opnd::Val(s2));
    let i2 = b.add_i32(i, c1);
    b.set_var(i, Opnd::Val(i2));
    let more = b.cmp_lt_i32(i2, total);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    let sf = b.bitcast_f32(Opnd::Var(s));
    b.store_f32_wt(result, 0, Opnd::Val(sf));
    b.end_if();
    b.finish()
}

pub fn probe_last_arriver() -> Function {
    probe_last_arriver_mode("probe_last_arriver", 0)
}

pub fn probe_last_arriver_lite() -> Function {
    probe_last_arriver_mode("probe_last_arriver_lite", 1)
}

pub fn probe_last_arriver_nofence() -> Function {
    probe_last_arriver_mode("probe_last_arriver_nofence", 2)
}

pub fn probe_last_arriver_plain() -> Function {
    probe_last_arriver_mode("probe_last_arriver_plain", 3)
}

pub fn probe_last_arriver_plain_fenced() -> Function {
    probe_last_arriver_mode("probe_last_arriver_plain_fenced", 4)
}

/// 8-wave (512-thread) last-arriver probe: the shape of the real fused KV-split combine. `s_barrier` does NOT drain VM_CNT, so each wave drains its own stores
/// before the block barrier; only then does thread 0 take the ticket (value 1, other threads add 0), the ticket is broadcast through LDS, and in the
/// last block EVERY wave does the acquire and copies a slice of the partials with coherent loads into `out` (exact per-element verification by
/// `probe_la_verify`). `mode`: 0 = full fences, 1 = lite (sc1 accesses, per-wave drain only), 2 = plain accesses + full fences, 3 = plain accesses + no
/// drain/fence at all (must be able to fail), 4 = sc1 accesses but NO per-wave drain before the barrier (shows whether the drain matters).
/// Partial value = hash(global index + seed); `stagger` (kernarg) adds a block-dependent busy loop before the stores to skew arrival order.
fn probe_la8_mode(name: &str, mode: u8) -> Function {
    let mut b = FnBuilder::new(name, 512);
    b.lds(2048);
    b.arg("part", DType::Ptr, 0, 8);
    b.arg("counter", DType::Ptr, 8, 8);
    b.arg("out", DType::Ptr, 16, 8);
    b.arg("nblocks", DType::I32, 24, 4);
    b.arg("seed", DType::I32, 28, 4);
    b.arg("stagger", DType::I32, 32, 4);
    let part = b.let_(Op::KernargPtr { offset: 0 });
    let counter = b.let_(Op::KernargPtr { offset: 8 });
    let out = b.let_(Op::KernargPtr { offset: 16 });
    let nb = b.let_(Op::KernargI32 { offset: 24 });
    let seed = b.let_(Op::KernargI32 { offset: 28 });
    let stagger = b.let_(Op::KernargI32 { offset: 32 });
    let tid = b.let_(Op::WorkitemId);
    let bid = b.let_(Op::WorkgroupIdX);
    let c512 = b.c_i32(512);
    let c4 = b.c_i32(4);
    let c1 = b.c_i32(1);
    let c0 = b.c_i32(0);
    let c37 = b.c_i32(37);
    let c63 = b.c_i32(63);
    let chash = b.c_i32(-1640531535); // 0x9E3779B1
    // block-dependent delay: ((bid*37 + seed) & 63) * stagger loop iterations
    let d0 = b.mul_i32(bid, c37);
    let d1 = b.add_i32(d0, seed);
    let d2 = b.let_(Op::AndI32(Opnd::Val(d1), Opnd::Val(c63)));
    let iters = b.mul_i32(d2, stagger);
    let dv = b.begin_loop(&[(DType::I32, Opnd::Val(c0))]);
    let dn = b.add_i32(dv[0], c1);
    b.set_var(dv[0], Opnd::Val(dn));
    let dmore = b.cmp_lt_i32(dn, iters);
    let dstop = b.let_(Op::NotI1(Opnd::Val(dmore)));
    b.break_if(Opnd::Val(dstop));
    b.end_loop();
    // partial store: element idx = bid*512 + tid
    let row = b.mul_i32(bid, c512);
    let idx = b.add_i32(row, tid);
    let hv = b.add_i32(idx, seed);
    let val = b.mul_i32(hv, chash);
    let valf = b.bitcast_f32(Opnd::Val(val));
    let off = b.mul_i32(idx, c4);
    let pp = b.ptr_add(part, off);
    if mode == 2 || mode == 3 {
        b.store_f32(pp, 0, Opnd::Val(valf));
    } else {
        b.store_f32_wt(pp, 0, Opnd::Val(valf));
    }
    match mode {
        0 | 2 => b.fence_release(),     // per-wave: drain, wbl2, drain
        1 => b.fence_release_lite(),    // per-wave drain
        _ => {}                         // 3, 4: no drain before the barrier
    }
    b.barrier();
    // ticket: thread 0 adds 1, everybody else adds 0
    let is0 = b.cmp_lt_i32(tid, c1);
    let one = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::Val(c1), b: Opnd::Val(c0) });
    let old = b.atomic_add_u32(counter, 0, Opnd::Val(one));
    let lo = b.mul_i32(tid, c4);
    b.store_lds_b32(Opnd::Val(lo), Opnd::Val(old));
    b.wait_lds(0);
    b.barrier();
    let ticket = b.load_lds_b32(Opnd::Val(c0)); // thread 0's old value: broadcast
    let cm1 = b.c_i32(-1);
    let nbm1 = b.add_i32(nb, cm1);
    let is_last = b.let_(Op::CmpGeI32(Opnd::Val(ticket), Opnd::Val(nbm1)));
    b.begin_if(Opnd::Val(is_last));
    match mode {
        0 | 2 => b.fence_acquire(),
        1 | 4 => b.fence_acquire_lite(),
        _ => {}
    }
    let total = b.mul_i32(nb, c512);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(tid))]);
    let i = vars[0];
    let io = b.mul_i32(i, c4);
    let q = b.ptr_add(part, io);
    let x = if mode == 2 || mode == 3 { b.load_f32(q, 0) } else { b.load_f32_coherent(q, 0) };
    let qo = b.ptr_add(out, io);
    if mode == 2 || mode == 3 {
        b.store_f32(qo, 0, Opnd::Val(x));
    } else {
        b.store_f32_wt(qo, 0, Opnd::Val(x));
    }
    let i2 = b.add_i32(i, c512);
    b.set_var(i, Opnd::Val(i2));
    let more = b.cmp_lt_i32(i2, total);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    b.end_if();
    b.finish()
}

pub fn probe_la8() -> Function {
    probe_la8_mode("probe_la8", 0)
}
pub fn probe_la8_lite() -> Function {
    probe_la8_mode("probe_la8_lite", 1)
}
pub fn probe_la8_plain_fenced() -> Function {
    probe_la8_mode("probe_la8_plain_fenced", 2)
}
pub fn probe_la8_plain() -> Function {
    probe_la8_mode("probe_la8_plain", 3)
}
pub fn probe_la8_nodrain() -> Function {
    probe_la8_mode("probe_la8_nodrain", 4)
}

/// Exact per-element verifier for the probes above: counts elements of `out` that differ from hash(i + seed) with a device atomic.
pub fn probe_la_verify() -> Function {
    let mut b = FnBuilder::new("probe_la_verify", 256);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg("bad", DType::Ptr, 8, 8);
    b.arg("total", DType::I32, 16, 4);
    b.arg("seed", DType::I32, 20, 4);
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let bad = b.let_(Op::KernargPtr { offset: 8 });
    let total = b.let_(Op::KernargI32 { offset: 16 });
    let seed = b.let_(Op::KernargI32 { offset: 20 });
    let tid = b.let_(Op::WorkitemId);
    let bid = b.let_(Op::WorkgroupIdX);
    let c256 = b.c_i32(256);
    let c4 = b.c_i32(4);
    let c1 = b.c_i32(1);
    let c0 = b.c_i32(0);
    let chash = b.c_i32(-1640531535);
    let row = b.mul_i32(bid, c256);
    let i = b.add_i32(row, tid);
    let inb = b.cmp_lt_i32(i, total);
    b.begin_if(Opnd::Val(inb));
    let off = b.mul_i32(i, c4);
    let p = b.ptr_add(out, off);
    let x = b.load_f32(p, 0);
    let xi = b.bitcast_i32(Opnd::Val(x));
    let hv = b.add_i32(i, seed);
    let want = b.mul_i32(hv, chash);
    let d = b.let_(Op::XorI32(Opnd::Val(xi), Opnd::Val(want)));
    let neg = b.cmp_lt_i32(d, c0);
    let pos = b.cmp_lt_i32(c0, d);
    let inner = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(pos), a: Opnd::Val(c1), b: Opnd::Val(c0) });
    let nz = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(neg), a: Opnd::Val(c1), b: Opnd::Val(inner) });
    let _old = b.atomic_add_u32(bad, 0, Opnd::Val(nz));
    b.end_if();
    b.finish()
}

/// Conversion-chain probe, DSL path vs safe-asm path (`probe_cvtchain_asm` embeds `examples::cvt_chain_block`): each lane loads 8 scores, runs
/// pk_add / pk_mul / pknorm / svcorr (lshr, and, perm, pk_sub) / scalef32 fp8 packing, stages the two P words in LDS and reads them back with the
/// transposed `ds_read_b64_tr_b8`; the two results are stored. Both kernels must produce bit-identical outputs.
fn probe_cvtchain(name: &str, use_asm: bool) -> Function {
    let mut b = FnBuilder::new(name, 64);
    b.lds(512);
    b.arg("inp", DType::Ptr, 0, 8);
    b.arg("outp", DType::Ptr, 8, 8);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let outp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c32 = b.c_i32(32);
    let c8 = b.c_i32(8);
    let off_in = b.mul_i32(tid, c32);
    let pin = b.ptr_add(inp, off_in);
    let v0 = b.load_b128(pin, 0);
    let v1 = b.load_b128(pin, 16);
    let dl = b.c_f32(0.25);
    let sl = b.c_f32(0.5);
    let delta2 = b.frag2_dup(dl);
    let scale2 = b.frag2_dup(sl);
    let addr = b.mul_i32(tid, c8);
    let t = if use_asm {
        b.asm(amdgpu_safeasm::examples::cvt_chain_block(), &[Opnd::Val(v0), Opnd::Val(v1), Opnd::Val(delta2), Opnd::Val(scale2), Opnd::Val(addr)])
    } else {
        let cmask = b.c_i32(0x0007_0007);
        let csel = b.c_i32(0x0c0c_0400);
        let cscale = b.c_i32(127);
        let c4 = b.c_i32(4);
        let mut words = Vec::new();
        for v in [v0, v1] {
            let mut ys = Vec::new();
            for j in 0..2u8 {
                let a = b.frag4_lane(v, 2 * j);
                let c = b.frag4_lane(v, 2 * j + 1);
                let p = b.frag2(a, c);
                let q = b.pk_add_f32(p, delta2);
                let r = b.pk_mul_f32(q, scale2);
                let lo = b.frag2_lane(r, 0);
                let hi = b.frag2_lane(r, 1);
                let lof = b.bitcast_f32(lo);
                let hif = b.bitcast_f32(hi);
                let u = b.cvt_pknorm_u16_f32(lof, hif);
                let sh = b.let_(Op::LshrI32(Opnd::Val(u), Opnd::Val(c4)));
                let m = b.let_(Op::AndI32(Opnd::Val(sh), Opnd::Val(cmask)));
                let x = b.perm_b32(m, m, csel);
                let y = b.pk_sub_u16(u, x);
                ys.push(y);
            }
            words.push(b.cvt_p8x4_from_u16(ys[0], ys[1], cscale));
        }
        let c4b = b.c_i32(4);
        let addr1 = b.add_i32(addr, c4b);
        b.store_lds_b32(addr, words[0]);
        b.store_lds_b32(addr1, words[1]);
        b.wait_lds(0);
        b.load_lds_tr_b8(addr)
    };
    let l0 = b.frag2_lane(t, 0);
    let l1 = b.frag2_lane(t, 1);
    let f0 = b.bitcast_f32(l0);
    let f1 = b.bitcast_f32(l1);
    let off_out = b.mul_i32(tid, c8);
    let po = b.ptr_add(outp, off_out);
    b.store_f32(po, 0, f0);
    b.store_f32(po, 4, f1);
    b.finish()
}

pub fn probe_cvtchain_dsl() -> Function {
    probe_cvtchain("probe_cvtchain_dsl", false)
}
pub fn probe_cvtchain_asm() -> Function {
    probe_cvtchain("probe_cvtchain_asm", true)
}

/// out[2*tid] = tid + 1000, out[2*tid+1] = tid + 2000 through one `global_store_dwordx2` per lane (covers the x2 store on both backends).
pub fn probe_store_x2() -> Function {
    let mut b = FnBuilder::new("probe_store_x2", 64);
    b.arg("outp", DType::Ptr, 0, 8);
    let outp = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c8 = b.c_i32(8);
    let c1000 = b.c_i32(1000);
    let c2000 = b.c_i32(2000);
    let a = b.add_i32(tid, c1000);
    let c = b.add_i32(tid, c2000);
    let f2 = b.frag2(a, c);
    let off = b.mul_i32(tid, c8);
    let po = b.ptr_add(outp, off);
    b.store_f32x2(po, 8, f2);
    b.store_f32x2(po, 0, f2); // offset 0 and a non-zero offset form
    b.finish()
}

/// Scaled MFMA with an explicit C operand (persistent bias accumulator form): bias = ones x ones (every element 64.0), S = ones x ones + bias (a loop
/// variable written through `var_mfma_scale_c`, C != dst) = 128.0 in every element; out[tid] = S[0].
pub fn probe_mfma_cform() -> Function {
    let mut b = FnBuilder::new("probe_mfma_cform", 64);
    b.arg("outp", DType::Ptr, 0, 8);
    let outp = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let c127 = b.c_i32(127);
    let ones = b.c_i32(0x3838_3838);
    let a8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let b8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let z = b.frag_zero(DType::Frag16);
    let bias = b.mfma_scale_fp8(a8, b8, z, c127, c127); // 64.0 everywhere
    let z2 = b.frag_zero(DType::Frag16);
    let vars = b.begin_loop(&[(DType::Frag16, Opnd::Val(z2))]);
    b.var_mfma_scale_c(vars[0], a8, b8, bias, c127, c127);
    let zero = b.c_i32(0);
    let stop = b.cmp_lt_i32(zero, c127); // always true: one iteration
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    let l = b.frag16_lane(Opnd::Var(vars[0]), 0);
    let lf = b.bitcast_f32(l);
    let off = b.mul_i32(tid, c4);
    let po = b.ptr_add(outp, off);
    b.store_f32(po, 0, lf);
    b.finish()
}

/// Unscaled fp8 MFMA (`v_mfma_f32_32x32x64_f8f6f4`) vs the scaled form with both scales 127: S1 = ones x ones (zero-C form) then accumulate in place (loop var),
/// plus a separate-C form; every result must equal the scaled-form result (exactly 64, 128 and 192 per element). out[tid] = S[0] (unscaled path),
/// out[64+tid] = scaled path.
pub fn probe_mfma_unscaled() -> Function {
    let mut b = FnBuilder::new("probe_mfma_unscaled", 64);
    b.arg("outp", DType::Ptr, 0, 8);
    let outp = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let c127 = b.c_i32(127);
    let zero = b.c_i32(0);
    let ones = b.c_i32(0x3838_3838);
    let a8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let b8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let z = b.frag_zero(DType::Frag16);
    // unscaled: zero-C then +64 in place (loop var), then a separate-C form (+64): 64 * 3 = 192
    let s0 = b.mfma_fp8_32x32x64(a8, b8, z);
    let z2 = b.frag_zero(DType::Frag16);
    let vars = b.begin_loop(&[(DType::Frag16, Opnd::Val(z2))]);
    b.var_mfma_fp8_32x32x64_z(vars[0], a8, b8);
    b.var_mfma_fp8_32x32x64(vars[0], a8, b8);
    let stop = b.cmp_lt_i32(zero, c127);
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    let z3 = b.frag_zero(DType::Frag16);
    let vars2 = b.begin_loop(&[(DType::Frag16, Opnd::Val(z3))]);
    b.var_mfma_fp8_32x32x64_c(vars2[0], a8, b8, s0); // = 64 (s0) + 64
    let stop2 = b.cmp_lt_i32(zero, c127);
    b.break_if(Opnd::Val(stop2));
    b.end_loop();
    // scaled reference: 64, then 128 by accumulation, then 192 by a third
    let r0 = b.mfma_scale_fp8(a8, b8, z, c127, c127);
    let r1 = b.mfma_scale_fp8(a8, b8, r0, c127, c127);
    let off = b.mul_i32(tid, c4);
    let po = b.ptr_add(outp, off);
    let u1 = b.frag16_lane(Opnd::Var(vars[0]), 0); // 128
    let u2 = b.frag16_lane(Opnd::Var(vars2[0]), 5); // 128
    let rr = b.frag16_lane(r1, 0); // 128
    let s_unscaled = b.frag16_lane(s0, 3); // 64
    let f1 = b.bitcast_f32(u1);
    let f2 = b.bitcast_f32(u2);
    let fr = b.bitcast_f32(rr);
    let fs = b.bitcast_f32(s_unscaled);
    b.store_f32(po, 0, f1);
    b.store_f32(po, 256, f2);
    b.store_f32(po, 512, fr);
    b.store_f32(po, 768, fs);
    b.finish()
}

/// MUBUF read/write probe (see tools/buffer_probe.sh): copies through a buffer descriptor (x4 / x2 / x1 loads and stores, 12-bit immediates, SGPR soffset) into
/// out[0..], the same copies through the FLAT path into out[4096..], then the out-of-range behaviour of a sized descriptor into out[8192..].
pub fn probe_buffer_rw() -> Function {
    let mut b = FnBuilder::new("probe_buffer_rw", 64);
    b.arg("inp", DType::Ptr, 0, 8);
    b.arg("outp", DType::Ptr, 8, 8);
    b.arg("soff", DType::I32, 16, 4);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let outp = b.let_(Op::KernargPtr { offset: 8 });
    let s_off = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let c8 = b.c_i32(8);
    let c16 = b.c_i32(16);
    let c0 = b.c_i32(0);
    let o4 = b.mul_i32(tid, c4);
    let o8 = b.mul_i32(tid, c8);
    let o16 = b.mul_i32(tid, c16);
    let rin = b.buffer_rsrc(inp);
    let rout = b.buffer_rsrc(outp);
    // --- MUBUF path
    let v4 = b.buffer_load_b128(rin, o16, c0, 0);
    b.buffer_store_b128(rout, o16, c0, 0, v4); // out[0..1024)
    let v2 = b.buffer_load_b64(rin, o8, c0, 1024);
    b.buffer_store_b64(rout, o8, c0, 1024, v2); // out[1024..1536)  <- in[1024..1536)
    let v1 = b.buffer_load_f32(rin, o4, c0, 2048);
    b.buffer_store_f32(rout, o4, c0, 1536, v1); // out[1536..1792)  <- in[2048..2304)
    let v1s = b.buffer_load_f32(rin, o4, s_off, 0); // soffset (SGPR) = 3072
    b.buffer_store_f32(rout, o4, c0, 2048, v1s); // out[2048..2304) <- in[3072..3328)
    // --- FLAT path, same data, 4096 bytes further
    let c4096f = b.c_i32(4096);
    let outf = b.ptr_add(outp, c4096f);
    let pin16 = b.ptr_add(inp, o16);
    let pout16 = b.ptr_add(outf, o16);
    let f4 = b.load_b128(pin16, 0);
    b.store_f32x4(pout16, 0, f4);
    let pin8 = b.ptr_add(inp, o8);
    let pout8 = b.ptr_add(outf, o8);
    let g2 = b.load_f32(pin8, 1024);
    let g2b = b.load_f32(pin8, 1028);
    b.store_f32(pout8, 1024, g2);
    b.store_f32(pout8, 1028, g2b);
    let pin4 = b.ptr_add(inp, o4);
    let pout4 = b.ptr_add(outf, o4);
    let g1 = b.load_f32(pin4, 2048);
    b.store_f32(pout4, 1536, g1);
    let g1s = b.add_i32(o4, s_off);
    let pins = b.ptr_add(inp, g1s);
    let g1v = b.load_f32(pins, 0);
    b.store_f32(pout4, 2048, g1v);
    // --- out-of-range: a descriptor of 512 bytes (lanes >= 32 read offset >= num_records: zero; their stores are dropped)
    let rin_s = b.buffer_desc(inp, 512);
    let z4 = b.buffer_load_b128(rin_s, o16, c0, 0);
    let c8192 = b.c_i32(8192);
    let o16c = b.add_i32(o16, c8192);
    b.buffer_store_b128(rout, o16c, c0, 0, z4); // out[8192..9216): lanes 32..63 must be ZERO
    let rout_s = b.buffer_desc(outp, 9216 + 512);
    let c9216 = b.c_i32(9216);
    let o16b = b.add_i32(o16, c9216);
    b.buffer_store_b128(rout_s, o16b, c0, 0, v4); // out[9216..10240): lanes 32..63 (offset >= 9728) DROPPED, sentinel must survive
    b.finish()
}

/// MUBUF -> LDS DMA probe: `buffer_load_dwordx4 ... lds` / `buffer_load_dword ... lds` (M0 slot base + imm + lane*size, SGPR soffset) vs the FLAT `global_load_lds`
/// equivalents; the staged LDS contents are copied out and compared by the host.
pub fn probe_buffer_lds() -> Function {
    let mut b = FnBuilder::new("probe_buffer_lds", 64);
    b.lds(8192);
    b.arg("inp", DType::Ptr, 0, 8);
    b.arg("outp", DType::Ptr, 8, 8);
    b.arg("soff", DType::I32, 16, 4);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let outp = b.let_(Op::KernargPtr { offset: 8 });
    let s_off = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let c0 = b.c_i32(0);
    let c4 = b.c_i32(4);
    let c16 = b.c_i32(16);
    let o4 = b.mul_i32(tid, c4);
    let o16 = b.mul_i32(tid, c16);
    let rin = b.buffer_rsrc(inp);
    // MUBUF: slot 0 <- in[0..1024), slot 1024 <- in[soff..] (soffset 2048), dword form slot 2048 <- in[0..256), imm form: slot (0 + imm 3072)
    b.load_buffer_lds(rin, o16, c0, c0, 0, 4);
    let c1024 = b.c_i32(1024);
    b.load_buffer_lds(rin, o16, s_off, c1024, 0, 4);
    let c2048 = b.c_i32(2048);
    b.load_buffer_lds(rin, o4, c0, c2048, 0, 1);
    b.load_buffer_lds(rin, o16, c0, c0, 3072, 4); // M0 = 0, imm 3072: LDS dest 3072 + 16*lane, memory offset 3072 + voff
    // FLAT global_load_lds equivalents into the upper half
    let pin16 = b.ptr_add(inp, o16);
    let c4096 = b.c_i32(4096);
    b.load_global_lds(pin16, c4096, 0, 4);
    let c5120 = b.c_i32(5120);
    let g_s = b.add_i32(o16, s_off);
    let pin_s = b.ptr_add(inp, g_s);
    b.load_global_lds(pin_s, c5120, 0, 4);
    let pin4 = b.ptr_add(inp, o4);
    let c6144 = b.c_i32(6144);
    b.load_global_lds(pin4, c6144, 0, 1);
    // global_load_lds adds the instruction offset to BOTH the memory address and the LDS destination (like the MUBUF form): 4096 + 3072 = 7168
    b.load_global_lds(pin16, c4096, 3072, 4);
    b.wait_vm(0);
    b.barrier();
    // copy the staged slots out: out[0..4096) = LDS[0..4096) ordered 16 B/lane, flat copies after it
    let pout16 = b.ptr_add(outp, o16);
    let c4096o = b.c_i32(4096);
    let outf = b.ptr_add(outp, c4096o);
    let pout16f = b.ptr_add(outf, o16);
    for (k, base) in [0i32, 1024, 3072, 4096, 5120, 7168].into_iter().enumerate() {
        let ca = b.c_i32(base);
        let a = b.add_i32(o16, ca);
        let v = b.load_lds_b128(a);
        if k < 4 {
            b.store_f32x4(pout16, (k * 1024) as u32, v);
        } else {
            b.store_f32x4(pout16f, ((k - 4) * 1024) as u32, v);
        }
    }
    let pout4 = b.ptr_add(outf, o4);
    let ca = b.c_i32(2048);
    let a = b.add_i32(o4, ca);
    let v = b.load_lds_b32(a);
    let vf = b.bitcast_f32(v);
    b.store_f32(pout4, 2048, vf);
    let cb = b.c_i32(6144);
    let a2 = b.add_i32(o4, cb);
    let v2 = b.load_lds_b32(a2);
    let vf2 = b.bitcast_f32(v2);
    b.store_f32(pout4, 2304, vf2);
    b.finish()
}

/// Runtime `num_records` descriptor (a kernarg value = 512 bytes): plain load, `buffer_load_dwordx4 ... lds` and `buffer_load_dword ... lds` through it; the LDS is
/// pre-filled with 0xCDCDCDCD so the host can tell zero-fill from dropped out-of-range lanes (tools/buffer_probe.sh dyn).
pub fn probe_buffer_dyn() -> Function {
    let mut b = FnBuilder::new("probe_buffer_dyn", 64);
    b.lds(4096);
    b.arg("inp", DType::Ptr, 0, 8);
    b.arg("outp", DType::Ptr, 8, 8);
    b.arg("num", DType::I32, 16, 4);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let outp = b.let_(Op::KernargPtr { offset: 8 });
    let s_num = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let c0 = b.c_i32(0);
    let c4 = b.c_i32(4);
    let c16 = b.c_i32(16);
    let o4 = b.mul_i32(tid, c4);
    let o16 = b.mul_i32(tid, c16);
    let d = b.buffer_desc_dyn(inp, s_num);
    let v = b.buffer_load_b128(d, o16, c0, 0);
    let pout16 = b.ptr_add(outp, o16);
    b.store_f32x4(pout16, 0, v); // out[0..1024): plain load, lanes with offset >= num read zero
    // LDS sentinel
    let cd = b.c_i32(0xCDCD_CDCDu32 as i32);
    let sent = b.frag4(cd, cd, cd, cd);
    b.store_lds_b128(o16, sent);
    let c1024 = b.c_i32(1024);
    let a1 = b.add_i32(o4, c1024);
    b.store_lds_b32(a1, cd);
    b.wait_lds(0);
    b.barrier();
    b.load_buffer_lds(d, o16, c0, c0, 0, 4); // LDS[0..1024)
    b.load_buffer_lds(d, o16, c0, c1024, 0, 1); // LDS[1024..1280): dword form, same voffset (lanes >= 32 out of range)
    b.wait_vm(0);
    b.barrier();
    let lv = b.load_lds_b128(o16);
    b.store_f32x4(pout16, 1024, lv); // out[1024..2048)
    let lw = b.load_lds_b32(a1);
    let lwf = b.bitcast_f32(lw);
    let pout4 = b.ptr_add(outp, o4);
    b.store_f32(pout4, 2048, lwf); // out[2048..2304)
    b.finish()
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------
// probe_llvm_* : function / ABI level feature kernels for the LLVM-test coverage audit (tools/llvm_feature_probe.sh runs them on the GPU).
// ---------------------------------------------------------------------------------------------------------------------------------------------------

/// Kernarg layout: three pointers, an i32, an f32 at a 4-byte (not 8-byte) aligned offset, another i32. c[i] = a[i] + b[i] + s (f32); c[64] = n, c[65] = m (raw bits).
pub fn probe_llvm_kernarg() -> Function {
    let mut b = FnBuilder::new("probe_llvm_kernarg", 64);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("n", DType::I32, 16, 4);
    b.arg("s", DType::F32, 20, 4);
    b.arg("c", DType::Ptr, 24, 8);
    b.arg("m", DType::I32, 32, 4);
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    let n = b.let_(Op::KernargI32 { offset: 16 });
    let s = b.let_(Op::KernargF32 { offset: 20 });
    let pc = b.let_(Op::KernargPtr { offset: 24 });
    let m = b.let_(Op::KernargI32 { offset: 32 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(tid, c4);
    let qa = b.ptr_add(pa, off);
    let qb = b.ptr_add(pb, off);
    let qc = b.ptr_add(pc, off);
    let x = b.load_f32(qa, 0);
    let y = b.load_f32(qb, 0);
    let xy = b.add_f32(x, y);
    let r = b.add_f32(xy, s);
    b.store_f32(qc, 0, r);
    let nf = b.bitcast_f32(n);
    let mf = b.bitcast_f32(m);
    b.store_f32(pc, 256, nf);
    b.store_f32(pc, 260, mf);
    b.finish()
}

/// Workgroup / workitem ids on a (3, 2, 2) grid of 64-thread blocks: out[((z * 2 + y) * 3 + x) * 64 + tid] = (z * 100 + y * 10 + x) * 1000 + tid.
pub fn probe_llvm_ids() -> Function {
    let mut b = FnBuilder::new("probe_llvm_ids", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let x = b.let_(Op::WorkgroupIdX);
    let y = b.let_(Op::WorkgroupIdY);
    let z = b.let_(Op::WorkgroupIdZ);
    let c2 = b.c_i32(2);
    let c3 = b.c_i32(3);
    let c4 = b.c_i32(4);
    let c10 = b.c_i32(10);
    let c64 = b.c_i32(64);
    let c100 = b.c_i32(100);
    let c1000 = b.c_i32(1000);
    let z2 = b.mul_i32(z, c2);
    let zy = b.add_i32(z2, y);
    let zy3 = b.mul_i32(zy, c3);
    let lin = b.add_i32(zy3, x);
    let blk = b.mul_i32(lin, c64);
    let idx = b.add_i32(blk, tid);
    let z100 = b.mul_i32(z, c100);
    let y10 = b.mul_i32(y, c10);
    let t1 = b.add_i32(z100, y10);
    let t2 = b.add_i32(t1, x);
    let t3 = b.mul_i32(t2, c1000);
    let val = b.add_i32(t3, tid);
    let off = b.mul_i32(idx, c4);
    let q = b.ptr_add(o, off);
    let vf = b.bitcast_f32(val);
    b.store_f32(q, 0, vf);
    b.finish()
}

/// `probe_llvm_ids` written with the operator expression layer (`expr.rs`): the same arithmetic, ONE expression per quantity instead of a constant + mul/add per step.
/// The emitted instructions are identical to the manual kernel (tests/expr_layer.rs); the contract lines add no code.
pub fn probe_expr_ids() -> Function {
    let mut b = FnBuilder::new("probe_expr_ids", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.grid_is(0, Ext::c(3));
    b.grid_is(1, Ext::c(2));
    b.grid_is(2, Ext::c(2));
    b.arg_tensor("o", Ext::c(3 * 2 * 2 * 64 * 4));
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let (tid, x, y, z) = (b.let_(Op::WorkitemId), b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    let (tid, x, y, z) = (b.i32v(tid), b.i32v(x), b.i32v(y), b.i32v(z));
    b.hoist_i32(&[2, 3, 4, 10, 64, 100, 1000]); // the manual kernel declares its constants up front: same order => same register allocation
    let idx = b.let_i32(((&z * 2 + &y) * 3 + &x) * 64 + &tid);
    let val = b.bind_i32(((&z * 100 + &y * 10 + &x) * 1000) + &tid);
    let q = b.bind_ptr(b.ptrv(o) + idx * 4);
    let vf = b.bind_f32(b.i32v(val).bitcast_f32());
    b.store_f32(q, 0, vf);
    b.finish()
}

/// `probe_llvm_kernarg` with operators: address arithmetic `tid * 4`, the pointer offsets, the f32 sum and the bit-casts are expressions.
pub fn probe_expr_kernarg() -> Function {
    let mut b = FnBuilder::new("probe_expr_kernarg", 64);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("n", DType::I32, 16, 4);
    b.arg("s", DType::F32, 20, 4);
    b.arg("c", DType::Ptr, 24, 8);
    b.arg("m", DType::I32, 32, 4);
    b.arg_tensor("a", Ext::c(256));
    b.arg_tensor("b", Ext::c(256));
    b.arg_tensor("c", Ext::c(264));
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    let n = b.let_(Op::KernargI32 { offset: 16 });
    let s = b.let_(Op::KernargF32 { offset: 20 });
    let pc = b.let_(Op::KernargPtr { offset: 24 });
    let m = b.let_(Op::KernargI32 { offset: 32 });
    let tid = b.let_(Op::WorkitemId);
    let (pa, pb, pc, n, s, m, tid) = (b.ptrv(pa), b.ptrv(pb), b.ptrv(pc), b.i32v(n), b.f32v(s), b.i32v(m), b.i32v(tid));
    let off = b.let_i32(&tid * 4);
    let (qa, qb, qc) = (b.bind_ptr(&pa + &off), b.bind_ptr(&pb + &off), b.bind_ptr(&pc + &off));
    let x = b.load_f32(qa, 0);
    let y = b.load_f32(qb, 0);
    let r = b.bind_f32(b.f32v(x) + b.f32v(y) + &s);
    b.store_f32(qc, 0, r);
    let (nf, mf) = (b.bind_f32(n.bitcast_f32()), b.bind_f32(m.bitcast_f32()));
    let pc = b.bind_ptr(pc);
    b.store_f32(pc, 256, nf);
    b.store_f32(pc, 260, mf);
    b.finish()
}
/// Wave roles (A = waves 0-3, B = waves 4-7) with an if / else on a wave-uniform condition, static LDS, barrier and waitcnt: A writes 1000 + tid, B writes 2000 + tid,
/// then every lane reads the partner role's slot (tid +- 256): out[tid] = (tid < 256) ? 2000 + tid + 256 : 1000 + tid - 256.
pub fn probe_llvm_roles() -> Function {
    let mut b = FnBuilder::new("probe_llvm_roles", 512);
    b.lds(2048);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let c256 = b.c_i32(256);
    let c512 = b.c_i32(512);
    let c1000 = b.c_i32(1000);
    let c2000 = b.c_i32(2000);
    let off = b.mul_i32(tid, c4);
    let role_a = b.cmp_lt_i32(tid, c256);
    b.begin_if(Opnd::Val(role_a));
    let va = b.add_i32(tid, c1000);
    b.store_lds_b32(Opnd::Val(off), Opnd::Val(va));
    b.else_();
    let vb = b.add_i32(tid, c2000);
    b.store_lds_b32(Opnd::Val(off), Opnd::Val(vb));
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    // partner slot: (tid + 256) mod 512 = tid + 256 for A, tid - 256 for B
    let plus = b.add_i32(tid, c256);
    let minus = b.add_i32(tid, c256);
    let wrap = b.let_(Op::CmpLtI32(Opnd::Val(plus), Opnd::Val(c512)));
    let cm2 = b.c_i32(-2);
    let neg = b.mul_i32(c256, cm2);
    let wrapped = b.add_i32(plus, neg);
    let _ = minus;
    let pidx = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wrap), a: Opnd::Val(plus), b: Opnd::Val(wrapped) });
    let poff = b.mul_i32(pidx, c4);
    let got = b.load_lds_b32(Opnd::Val(poff));
    let q = b.ptr_add(o, off);
    let gf = b.bitcast_f32(got);
    b.store_f32(q, 0, gf);
    b.finish()
}

/// Static LDS (8 KiB) with b32 and b128 accesses, barrier and partial waitcnt: lane t writes t at [4 t] and (t, t + 1000, t + 2000, t + 3000) at [4096 + 16 t];
/// after the barrier it reads the mirrored lane (63 - t): out[t] = 63 - t, out[64 + t] = word 2 of the mirrored b128 = (63 - t) + 2000.
pub fn probe_llvm_lds() -> Function {
    let mut b = FnBuilder::new("probe_llvm_lds", 64);
    b.lds(8192);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let c16 = b.c_i32(16);
    let c63 = b.c_i32(63);
    let c1000 = b.c_i32(1000);
    let c2000 = b.c_i32(2000);
    let c3000 = b.c_i32(3000);
    let c4096 = b.c_i32(4096);
    let off4 = b.mul_i32(tid, c4);
    let off16a = b.mul_i32(tid, c16);
    let off16 = b.add_i32(off16a, c4096);
    b.store_lds_b32(Opnd::Val(off4), Opnd::Val(tid));
    let w1 = b.add_i32(tid, c1000);
    let w2 = b.add_i32(tid, c2000);
    let w3 = b.add_i32(tid, c3000);
    let f4 = b.frag4(tid, w1, w2, w3);
    b.store_lds_b128(Opnd::Val(off16), Opnd::Val(f4));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let mirror = b.let_(Op::AddI32(Opnd::Val(c63), Opnd::Val(tid)));
    let _ = mirror;
    let cm1 = b.c_i32(-1);
    let neg = b.mul_i32(tid, cm1);
    let m = b.add_i32(c63, neg);
    let mo4 = b.mul_i32(m, c4);
    let mo16a = b.mul_i32(m, c16);
    let mo16 = b.add_i32(mo16a, c4096);
    let v = b.load_lds_b32(Opnd::Val(mo4));
    let vv = b.load_lds_b128(Opnd::Val(mo16));
    let w2l = b.frag4_lane(vv, 2);
    let q = b.ptr_add(o, off4);
    let vf = b.bitcast_f32(v);
    let wf = b.bitcast_f32(w2l);
    b.store_f32(q, 0, vf);
    b.store_f32(q, 256, wf);
    b.finish()
}

/// Wave-level (uniform) branching with nesting on the workgroup id: out[wg] = wg is even ? (wg < 4 ? 11 : 12) : (wg < 4 ? 21 : 22), 8 workgroups, lane 0 stores.
pub fn probe_llvm_branch() -> Function {
    let mut b = FnBuilder::new("probe_llvm_branch", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let wg = b.let_(Op::WorkgroupIdX);
    let tid = b.let_(Op::WorkitemId);
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c11 = b.c_i32(11);
    let c12 = b.c_i32(12);
    let c21 = b.c_i32(21);
    let c22 = b.c_i32(22);
    let odd = b.let_(Op::AndI32(Opnd::Val(wg), Opnd::Val(c1)));
    let is_even = b.cmp_lt_i32(odd, c1);
    let low = b.cmp_lt_i32(wg, c4);
    let lane0 = b.cmp_lt_i32(tid, c1);
    let _ = c0;
    let off = b.mul_i32(wg, c4);
    let q = b.ptr_add(o, off);
    b.begin_if(Opnd::Val(is_even));
    b.begin_if(Opnd::Val(low));
    let v11 = b.bitcast_f32(c11);
    b.begin_if(Opnd::Val(lane0));
    b.store_f32(q, 0, v11);
    b.end_if();
    b.else_();
    let v12 = b.bitcast_f32(c12);
    b.begin_if(Opnd::Val(lane0));
    b.store_f32(q, 0, v12);
    b.end_if();
    b.end_if();
    b.else_();
    b.begin_if(Opnd::Val(low));
    let v21 = b.bitcast_f32(c21);
    b.begin_if(Opnd::Val(lane0));
    b.store_f32(q, 0, v21);
    b.end_if();
    b.else_();
    let v22 = b.bitcast_f32(c22);
    b.begin_if(Opnd::Val(lane0));
    b.store_f32(q, 0, v22);
    b.end_if();
    b.end_if();
    b.end_if();
    b.finish()
}

/// Loop variables: for i in 0..n (n from the kernarg): acc += i (i32), facc += 0.5 (f32). out[tid] = acc (raw bits), out[64 + tid] = facc.
pub fn probe_llvm_loop() -> Function {
    let mut b = FnBuilder::new("probe_llvm_loop", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("n", DType::I32, 8, 4);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let n = b.let_(Op::KernargI32 { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let zf = b.c_f32(0.0);
    let half = b.c_f32(0.5);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(c0)), (DType::F32, Opnd::Val(zf))]);
    let it = vars[0];
    let acc = vars[1];
    let facc = vars[2];
    let more = b.cmp_lt_i32(Opnd::Var(it), n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    let acc_n = b.add_i32(Opnd::Var(acc), Opnd::Var(it));
    let facc_n = b.add_f32(Opnd::Var(facc), half);
    let it_n = b.add_i32(Opnd::Var(it), c1);
    b.set_var(acc, acc_n);
    b.set_var(facc, facc_n);
    b.set_var(it, it_n);
    b.end_loop();
    let off = b.mul_i32(tid, c4);
    let q = b.ptr_add(o, off);
    let accf = b.bitcast_f32(Opnd::Var(acc));
    b.store_f32(q, 0, accf);
    b.store_f32(q, 256, Opnd::Var(facc));
    b.finish()
}

/// Global atomics: every lane of every workgroup adds 1 to a counter and stores the returned old value: out[wg * 64 + tid] = old; counter ends at 64 * grid.
pub fn probe_llvm_atomic() -> Function {
    let mut b = FnBuilder::new("probe_llvm_atomic", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("ctr", DType::Ptr, 8, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let ctr = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let wg = b.let_(Op::WorkgroupIdX);
    let c1 = b.c_i32(1);
    let c4 = b.c_i32(4);
    let c64 = b.c_i32(64);
    let old = b.atomic_add_u32(ctr, 0, c1);
    let blk = b.mul_i32(wg, c64);
    let idx = b.add_i32(blk, tid);
    let off = b.mul_i32(idx, c4);
    let q = b.ptr_add(o, off);
    let of = b.bitcast_f32(old);
    b.store_f32(q, 0, of);
    b.finish()
}

/// Compare / select forms: out[t] = (t >= thr) ? 1.0 : 2.0 with an i32 greater-or-equal compare feeding a select (the `maskreg` lowering turns it into a mask compare), plus a
/// float compare `x < y` select.
pub fn probe_llvm_maskge() -> Function {
    let mut b = FnBuilder::new("probe_llvm_maskge", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("thr", DType::I32, 8, 4);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let thr = b.let_(Op::KernargI32 { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let one = b.c_f32(1.0);
    let two = b.c_f32(2.0);
    let ge = b.let_(Op::CmpGeI32(Opnd::Val(tid), Opnd::Val(thr)));
    let r = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ge), a: Opnd::Val(one), b: Opnd::Val(two) });
    let tf = b.bitcast_f32(tid);
    let lt = b.cmp_lt_f32(tf, one);
    let five = b.c_f32(5.0);
    let six = b.c_f32(6.0);
    let r2 = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(lt), a: Opnd::Val(five), b: Opnd::Val(six) });
    let off = b.mul_i32(tid, c4);
    let q = b.ptr_add(o, off);
    b.store_f32(q, 0, r);
    b.store_f32(q, 256, r2);
    b.finish()
}

/// Two independent kernels (each its own code object / kernel descriptor): k1 writes out[t] = t, k2 reads it and writes out[t] = 2 * out[t] + 1.
pub fn probe_llvm_k1() -> Function {
    let mut b = FnBuilder::new("probe_llvm_k1", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(tid, c4);
    let q = b.ptr_add(o, off);
    let tf = b.bitcast_f32(tid);
    b.store_f32(q, 0, tf);
    b.finish()
}

pub fn probe_llvm_k2() -> Function {
    let mut b = FnBuilder::new("probe_llvm_k2", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(tid, c4);
    let q = b.ptr_add(o, off);
    let x = b.load_f32(q, 0);
    let xi = b.bitcast_i32(x);
    let two = b.c_i32(2);
    let one = b.c_i32(1);
    let y = b.mul_i32(xi, two);
    let z = b.add_i32(y, one);
    let zf = b.bitcast_f32(z);
    b.store_f32(q, 0, zf);
    b.finish()
}
pub fn probe_log_store4() -> Function {
    // out[4*tid + j] = log2(in[tid]) for j in 0..4: covers v_log_f32 and global_store_dwordx4 on both backends
    let mut b = FnBuilder::new("probe_log_store4", 64);
    b.arg("inp", DType::Ptr, 0, 8);
    b.arg("outp", DType::Ptr, 8, 8);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let outp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let c16 = b.c_i32(16);
    let off_in = b.mul_i32(tid, c4);
    let off_out = b.mul_i32(tid, c16);
    let pin = b.ptr_add(inp, off_in);
    let pout = b.ptr_add(outp, off_out);
    let x = b.load_f32(pin, 0);
    let l = b.log2_f32(x);
    let li = b.bitcast_i32(l);
    let f4 = b.frag4(li, li, li, li);
    b.store_f32x4(pout, 0, f4);
    b.finish()
}

pub fn probe_safeasm_e2e() -> Function {
    let mut b = FnBuilder::new("probe_safeasm_e2e", 64);
    b.arg("inp", DType::Ptr, 0, 8);
    b.arg("outp", DType::Ptr, 8, 8);
    let inp = b.let_(Op::KernargPtr { offset: 0 });
    let outp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c16 = b.c_i32(16);
    let c4 = b.c_i32(4);
    let off_in = b.mul_i32(tid, c16);
    let off_out = b.mul_i32(tid, c4);
    let pin = b.ptr_add(inp, off_in);
    let pout = b.ptr_add(outp, off_out);
    let _r = b.asm(amdgpu_safeasm::examples::global_probe_block(), &[Opnd::Val(pin), Opnd::Val(pout)]);
    b.finish()
}

fn sched_begin(name: &str) -> (FnBuilder, ValueId, ValueId, ValueId) {
    // Build-time workgroup-size override for occupancy experiments (SCHED_WG=1024 launches a
    // 16-wave block = 4 waves/SIMD on one CU). The descriptor's `.max_flat_workgroup_size` must
    // cover the launched block size or the dispatch fails, so it has to be set at build time.
    let wg = std::env::var("SCHED_WG").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(512);
    let mut b = FnBuilder::new(name, wg);
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

// ---------------------------------------------------------------------------
// Sparse-fp8 MFMA (`v_smfmac_f32_16x16x128_fp8_fp8`) layout probe.
//
// One 64-lane wave. Each experiment keeps its 4-dword D in a distinct register and the dump happens
// after every smfmac has been issued (a fresh V4 per result forces the register allocator to give
// each experiment its own registers, so a WAW race with the previous smfmac's pending writeback
// cannot corrupt a later experiment's zero-initialised accumulator).
//
// Constants are chosen so the anonymous questions have unambiguous answers:
//   * all-ones A and B: D = (number of K positions contracted) -> 64 vs 128 tells us whether one
//     smfmac covers half or all of the 128 keys;
//   * accumulating the two candidate index patterns answers "how many instructions cover all K";
//   * A zeroed on lanes 32..63 answers whether the compressed A data is consumed per lane;
//   * an index selected by a lane mask (tid < 32) answers whether SRC2 is per lane or per wave;
//   * marker dwords in A (1,2,4,8) tell which compressed-A dword feeds which row.
fn probe_smfmac_dump(b: &mut FnBuilder, base: ValueId, slot: usize, d: ValueId) {
    // global_store_dword only carries a 13-bit signed immediate offset, so each 1 KB slot gets its
    // own base pointer (base = o + tid*16).
    let off = b.c_i32((slot * 1024) as i32);
    let row = b.ptr_add(base, off);
    for i in 0..4u8 {
        let bits = b.frag4_lane(d, i);
        let v = b.bitcast_f32(bits);
        b.store_f32(row, 4 * i as u32, v);
    }
}

/// Same for the first four dwords of a sixteen-dword fragment (the msum reference path).
fn probe_smfmac_dump16(b: &mut FnBuilder, base: ValueId, slot: usize, d: ValueId) {
    let off = b.c_i32((slot * 1024) as i32);
    let row = b.ptr_add(base, off);
    for i in 0..4u8 {
        let bits = b.frag16_lane(d, i);
        let v = b.bitcast_f32(bits);
        b.store_f32(row, 4 * i as u32, v);
    }
}

pub fn probe_smfmac_ones() -> Function {
    let mut b = FnBuilder::new("probe_smfmac_ones", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c16 = b.c_i32(16);
    let lane_off = b.mul_i32(tid, c16);
    let base = b.ptr_add(o, lane_off);

    let ones = b.c_i32(0x38383838u32 as i32); // E4M3 1.0 in all four bytes
    let zero = b.c_i32(0);
    let m1 = b.c_i32(0x38383838u32 as i32); // 1.0
    let m2 = b.c_i32(0x40404040u32 as i32); // 2.0
    let m4 = b.c_i32(0x48484848u32 as i32); // 4.0
    let m8 = b.c_i32(0x50505050u32 as i32); // 8.0

    let a_ones = b.frag4(ones, ones, ones, ones);
    let a_zero = b.frag4(zero, zero, zero, zero);
    let a_mark = b.frag4(m1, m2, m4, m8);
    let b_ones = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let b_mark = b.frag8(m1, m2, m4, m8, m1, m2, m4, m8);
    let c0 = b.frag_zero(DType::Frag4);

    // 2-bit-pair fields, LSB first: 0x44.. -> pairs (0,1), 0xee.. -> pairs (2,3), 0x55.. -> (1,1).
    let i01 = b.c_i32(0x44444444u32 as i32);
    let i23 = b.c_i32(0xeeeeeeeeu32 as i32);
    let i11 = b.c_i32(0x55555555u32 as i32);
    let c32 = b.c_i32(32);
    let lt32 = b.cmp_lt_i32(tid, c32);
    let sel_idx = b.let_(Op::Select { ty: DType::I32, cond: lt32.into(), a: i01.into(), b: i23.into() });
    let sel_a = b.let_(Op::Select { ty: DType::I32, cond: lt32.into(), a: ones.into(), b: zero.into() });
    let a_lane = b.frag4(sel_a, sel_a, sel_a, sel_a);

    let mut results: Vec<ValueId> = Vec::new();
    // 0/1: one smfmac with each index pattern, all-ones A and B -> contracted K positions.
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, c0));
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_ones, i23, c0));
    // 2: both patterns accumulated -> full 128-key row sum if complementary.
    let d = b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, c0);
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_ones, i23, d));
    // 3: reversed order.
    let d = b.smfmac_fp8_16x16x128(a_ones, b_ones, i23, c0);
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, d));
    // 4: three instructions (01, 23, 01) -> 192 if additive.
    let d = b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, c0);
    let d = b.smfmac_fp8_16x16x128(a_ones, b_ones, i23, d);
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, d));
    // 5: duplicate of slot 0 (detects any remaining cross-experiment register reuse).
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, c0));
    // 6: degenerate pattern (1,1).
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_ones, i11, c0));
    // 7: zero A -> D must be zero.
    results.push(b.smfmac_fp8_16x16x128(a_zero, b_ones, i01, c0));
    // 8: per-lane SRC2 (lanes 0..31 use 0x44, lanes 32..63 use 0xee).
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_ones, sel_idx, c0));
    // 9: A zeroed on lanes 32..63 -> which D elements depend on those lanes' A data.
    results.push(b.smfmac_fp8_16x16x128(a_lane, b_ones, i01, c0));
    // 10/11: marker A dwords (1,2,4,8) with all-ones B: D = sum of selected A dword values, so the
    //        per-D-dword totals reveal which compressed-A dword feeds which rows.
    results.push(b.smfmac_fp8_16x16x128(a_mark, b_ones, i01, c0));
    results.push(b.smfmac_fp8_16x16x128(a_mark, b_ones, i23, c0));
    // 12/13: marker B dwords with all-ones A: which dense-B dword is selected by each index pattern.
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_mark, i01, c0));
    results.push(b.smfmac_fp8_16x16x128(a_ones, b_mark, i23, c0));

    // 14: current row-sum reference: two accumulated scaled MFMAs (msum), A = all-ones 8-dword
    //     fragment, B = the two P halves; the first four D dwords are dumped.
    let ones8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let z16 = b.frag_zero(DType::Frag16);
    let c127 = b.c_i32(127);
    let p1 = b.frag8(m1, m2, m4, m8, m1, m2, m4, m8);
    let p2 = b.frag8(m8, m4, m2, m1, m8, m4, m2, m1);
    let m0 = b.mfma_scale_fp8(ones8, p1, z16, c127, c127);
    let msum = b.mfma_scale_fp8(ones8, p2, m0, c127, c127);
    // 15-18: smfmac variants on the same P halves (accumulating chains; see the dword-3 hazard note).
    let s15a = b.smfmac_fp8_16x16x128(a_ones, p1, i01, c0);
    let s15 = b.smfmac_fp8_16x16x128(a_ones, p2, i23, s15a);
    let s16a = b.smfmac_fp8_16x16x128(a_ones, p1, i01, c0);
    let s16 = b.smfmac_fp8_16x16x128(a_ones, p2, i01, s16a);
    let s17a = b.smfmac_fp8_16x16x128(a_ones, p1, i01, c0);
    let s17 = b.smfmac_fp8_16x16x128(a_ones, p1, i23, s17a);
    let s18a = b.smfmac_fp8_16x16x128(a_ones, p1, i01, c0);
    let s18b = b.smfmac_fp8_16x16x128(a_ones, p1, i23, s18a);
    let s18c = b.smfmac_fp8_16x16x128(a_ones, p2, i01, s18b);
    let s18 = b.smfmac_fp8_16x16x128(a_ones, p2, i23, s18c);

    // 19/20: is the 4th D dword written at all? Pre-initialise the accumulator to a known value
    //        (1000.0f) and check that a single smfmac adds exactly 64 (ones A/B) to every dword.
    let k1000 = b.c_i32(0x447A0000u32 as i32);
    let acc19 = b.frag4(k1000, k1000, k1000, k1000);
    let s19 = b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, acc19);
    let acc20 = b.frag4(k1000, k1000, k1000, k1000);
    let s20a = b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, acc20);
    let s20 = b.smfmac_fp8_16x16x128(a_ones, b_ones, i23, s20a);
    // 23-26: row-sum mapping with a *lane-dependent* P (even lanes 1.0, odd lanes 2.0), so the row
    //        sums differ and a wrong P-half/index pairing shows up. 23 = msum reference.
    let c1i = b.c_i32(1);
    let oddbit = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c1i)));
    let lane_is_even = b.cmp_lt_i32(oddbit, c1i);
    let lane_val = b.let_(Op::Select { ty: DType::I32, cond: lane_is_even.into(), a: m1.into(), b: m2.into() });
    let pv = b.frag8(lane_val, lane_val, lane_val, lane_val, lane_val, lane_val, lane_val, lane_val);
    let m23 = b.mfma_scale_fp8(ones8, pv, z16, c127, c127);
    let m23b = b.mfma_scale_fp8(ones8, pv, m23, c127, c127);
    let s24a = b.smfmac_fp8_16x16x128(a_ones, pv, i01, c0);
    let s24 = b.smfmac_fp8_16x16x128(a_ones, pv, i23, s24a);
    let s25a = b.smfmac_fp8_16x16x128(a_ones, pv, i01, c0);
    let s25 = b.smfmac_fp8_16x16x128(a_ones, pv, i01, s25a);
    let s26a = b.smfmac_fp8_16x16x128(a_ones, pv, i01, c0);
    let s26 = b.smfmac_fp8_16x16x128(a_ones, pv, i01, s26a);

    // 27: preset accumulator dumped WITHOUT any smfmac - control for the dump/read path itself.
    let acc27 = b.frag4(k1000, k1000, k1000, k1000);
    // 28/29: preset + single smfmac, read back through VALU (forcing a VALU consumer) and dumped as a
    //        sum (28) and as a rebuilt fragment (29).
    let acc28 = b.frag4(k1000, k1000, k1000, k1000);
    let s28 = b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, acc28);
    let b28_0 = b.frag4_lane(s28, 0);
    let b28_1 = b.frag4_lane(s28, 1);
    let b28_2 = b.frag4_lane(s28, 2);
    let b28_3 = b.frag4_lane(s28, 3);
    let v28_0 = b.bitcast_f32(b28_0);
    let v28_1 = b.bitcast_f32(b28_1);
    let v28_2 = b.bitcast_f32(b28_2);
    let v28_3 = b.bitcast_f32(b28_3);
    let sum28a = b.add_f32(v28_0, v28_1);
    let sum28b = b.add_f32(v28_2, v28_3);
    let sum28 = b.add_f32(sum28a, sum28b);
    let acc29 = b.frag4(k1000, k1000, k1000, k1000);
    let s29 = b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, acc29);
    let l29_0 = b.frag4_lane(s29, 0);
    let l29_1 = b.frag4_lane(s29, 1);
    let l29_2 = b.frag4_lane(s29, 2);
    let l29_3 = b.frag4_lane(s29, 3);
    let i29_0 = b.bitcast_i32(l29_0);
    let i29_1 = b.bitcast_i32(l29_1);
    let i29_2 = b.bitcast_i32(l29_2);
    let i29_3 = b.bitcast_i32(l29_3);
    let r29 = b.frag4(i29_0, i29_1, i29_2, i29_3);

    // 30-33: how much VALU work between the accumulator pre-set and the smfmac makes the C read
    //        visible? The filler chain is *stored* (slots 60+) so it cannot be dead-code-eliminated.
    let c1f = b.c_f32(1.0);
    let mut prev = b.c_f32(0.0);
    let mut s35: Vec<ValueId> = Vec::new();
    for &(slot, extra) in &[(30usize, 0usize), (31, 1), (32, 2), (33, 4)] {
        let acc = b.frag4(k1000, k1000, k1000, k1000);
        let mut filler = prev;
        for _ in 0..extra {
            filler = b.add_f32(filler, c1f);
        }
        prev = filler;
        let s = b.smfmac_fp8_16x16x128(a_ones, b_ones, i01, acc);
        let l0 = b.frag4_lane(s, 0);
        let l1 = b.frag4_lane(s, 1);
        let l2 = b.frag4_lane(s, 2);
        let l3 = b.frag4_lane(s, 3);
        let i0 = b.bitcast_i32(l0);
        let i1 = b.bitcast_i32(l1);
        let i2 = b.bitcast_i32(l2);
        let i3 = b.bitcast_i32(l3);
        s35.push(b.frag4(i0, i1, i2, i3));
        // keep the filler chain alive: one store per filler variant into its own slot
        let off_f = b.c_i32(((60 + extra) * 1024) as i32);
        let row_f = b.ptr_add(base, off_f);
        b.store_f32(row_f, 0, filler);
    }
    // 40-44: which P Frag8 goes into which smfmac / index pattern? P_a depends on the lane, P_b is a
    //        constant 4.0, so the three pairings give different results. 44 = msum reference.
    let p_a = b.frag8(lane_val, lane_val, lane_val, lane_val, lane_val, lane_val, lane_val, lane_val);
    let p_b = b.frag8(m4, m4, m4, m4, m4, m4, m4, m4);
    let m44 = b.mfma_scale_fp8(ones8, p_a, z16, c127, c127);
    let m44b = b.mfma_scale_fp8(ones8, p_b, m44, c127, c127);
    let s40a = b.smfmac_fp8_16x16x128(a_ones, p_a, i01, c0);
    let s40 = b.smfmac_fp8_16x16x128(a_ones, p_b, i23, s40a);
    let s41a = b.smfmac_fp8_16x16x128(a_ones, p_a, i01, c0);
    let s41 = b.smfmac_fp8_16x16x128(a_ones, p_b, i01, s41a);
    let s42a = b.smfmac_fp8_16x16x128(a_ones, p_a, i01, c0);
    let s42 = b.smfmac_fp8_16x16x128(a_ones, p_a, i23, s42a);
    let s43a = b.smfmac_fp8_16x16x128(a_ones, p_b, i01, c0);
    let s43 = b.smfmac_fp8_16x16x128(a_ones, p_a, i23, s43a);

    for (e, d) in results.iter().enumerate() {
        probe_smfmac_dump(&mut b, base, e, *d);
    }
    for (k, d) in s35.iter().enumerate() {
        probe_smfmac_dump(&mut b, base, 30 + k, *d);
    }
    probe_smfmac_dump16(&mut b, base, 44, m44b);
    probe_smfmac_dump(&mut b, base, 40, s40);
    probe_smfmac_dump(&mut b, base, 41, s41);
    probe_smfmac_dump(&mut b, base, 42, s42);
    probe_smfmac_dump(&mut b, base, 43, s43);
    probe_smfmac_dump(&mut b, base, 27, acc27);
    let off28 = b.c_i32(28 * 1024);
    let row28 = b.ptr_add(base, off28);
    b.store_f32(row28, 0, sum28);
    probe_smfmac_dump(&mut b, base, 29, r29);
    probe_smfmac_dump(&mut b, base, 19, s19);
    probe_smfmac_dump(&mut b, base, 20, s20);
    probe_smfmac_dump16(&mut b, base, 23, m23b);
    probe_smfmac_dump(&mut b, base, 24, s24);
    probe_smfmac_dump(&mut b, base, 25, s25);
    probe_smfmac_dump(&mut b, base, 26, s26);
    probe_smfmac_dump16(&mut b, base, 14, msum);
    probe_smfmac_dump(&mut b, base, 15, s15);
    probe_smfmac_dump(&mut b, base, 16, s16);
    probe_smfmac_dump(&mut b, base, 17, s17);
    probe_smfmac_dump(&mut b, base, 18, s18);
    b.finish()
}

// ---------------------------------------------------------------------------
// gpu-semantics audit attack kernels (not shipped; used by gpu-semantics-occ)
// ---------------------------------------------------------------------------



// ---------------------------------------------------------------------------------------------
// probe_agpr_*: AGPR support probes (docs/agpr.md). The three variants compute the same thing: a scaled-MFMA accumulation loop whose A/B
// fragments come from LDS. `ref` keeps everything in arch VGPRs; `acc` pins the accumulator in AGPRs (acc_cd form MFMA); `full` also reads the
// fragments with ds_read_b128 straight into AGPRs. Output (identical for all three): per-thread 16 accumulator lanes at o[tid * 64 ..].
// ---------------------------------------------------------------------------------------------
fn probe_agpr_kernel(name: &str, pin_acc: bool, pin_frags: bool, acc_base: Option<u32>) -> Function {
    let (mut b, o, n, tid) = sched_begin(name);
    let c0 = b.c_i32(0);
    let c16 = b.c_i32(16);
    let c64 = b.c_i32(64);
    let c8192 = b.c_i32(8192);
    let mul = b.c_i32(0x01010101);
    let add = b.c_i32(0x04030201);
    let mask = b.c_i32(0x3f3f3f3f);
    let t1 = b.mul_i32(tid, mul);
    let t2 = b.add_i32(t1, add);
    let w = b.let_(Op::AndI32(Opnd::Val(t2), Opnd::Val(mask)));
    let data = b.frag4(w, w, w, w);
    let off = b.mul_i32(tid, c16);
    let off2 = b.add_i32(off, c8192);
    b.store_lds_b128(Opnd::Val(off), Opnd::Val(data));
    b.store_lds_b128(Opnd::Val(off2), Opnd::Val(data));
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let mut a8 = b.load_lds_b128x2(Opnd::Val(off), Opnd::Val(off2));
    let mut b8 = b.load_lds_b128x2(Opnd::Val(off2), Opnd::Val(off));
    if pin_frags {
        a8 = b.agpr(a8);
        b8 = b.agpr(b8);
    }
    let zero = b.frag_zero(DType::Frag16);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::Frag16, Opnd::Val(zero))]);
    if pin_acc {
        match acc_base {
            Some(base) => b.pin_agpr_at(vars[1], base),
            None => b.pin_agpr(vars[1]),
        }
    }
    b.var_mfma_scale(vars[1], a8, b8, 127, 127);
    sched_end_loop(&mut b, vars[0], n);
    let row_off = b.mul_i32(tid, c64);
    let row = b.ptr_add(o, row_off);
    for i in 0..16u8 {
        let l = b.frag16_lane(vars[1], i);
        let x = b.bitcast_f32(l);
        b.store_f32(row, 4 * i as u32, x);
    }
    b.finish()
}

pub fn probe_agpr_ref() -> Function {
    probe_agpr_kernel("probe_agpr_ref", false, false, None)
}
pub fn probe_agpr_acc() -> Function {
    probe_agpr_kernel("probe_agpr_acc", true, false, None)
}
pub fn probe_agpr_full() -> Function {
    probe_agpr_kernel("probe_agpr_full", true, true, None)
}
/// Accumulator pinned at a[64:79] (exact placement), fragments in AGPRs.
pub fn probe_agpr_pinned() -> Function {
    probe_agpr_kernel("probe_agpr_pinned", true, true, Some(64))
}

/// Regression probe for the silent miscompile found in the GEMM (docs/agpr.md "LDS spill + DMA operand wait"): the loop-invariant buffer voffset of an LDS-DMA load is spilled to LDS by the
/// register-pressure ladder (`AMDC_RA_LDS_SPILL=N AMDC_RA_LDS_SPILL_TARGET=0` forces it) and reloaded by a `ds_read` right before the DMA; the DMA must wait for that read.
/// Without the fix (`AMDC_WAITS_SKIP_FENCE_OPERANDS=1`) the wait is missing and the DMA reads a stale voffset. Output: per thread, n * (first dword of the 16 bytes it DMA-ed), identical
/// with and without the spill.
pub fn probe_dma_voff_spill() -> Function {
    let (mut b, o, n, tid) = sched_begin("probe_dma_voff_spill");
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c16 = b.c_i32(16);
    let c63 = b.c_i32(63);
    let c1024 = b.c_i32(1024);
    let mul = b.c_i32(0x01010101);
    let add = b.c_i32(0x04030201);
    let o16 = b.mul_i32(tid, c16);
    let t1 = b.mul_i32(tid, mul);
    let t2 = b.add_i32(t1, add);
    let mask = b.c_i32(0x3f3f3f3f);
    let w = b.let_(Op::AndI32(Opnd::Val(t2), Opnd::Val(mask)));
    let data = b.frag4(w, w, w, w);
    // stage the input in the output buffer: out[tid*16 ..] = pattern(tid)
    let pin = b.ptr_add(o, o16);
    b.store_f32x4(pin, 0, data);
    b.wait_vm(0);
    let rin = b.buffer_rsrc(o);
    let c6 = b.c_i32(6);
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let ldsb = b.mul_i32(wave, c1024);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let lane16 = b.mul_i32(lane, c16);
    let rd = b.add_i32(ldsb, lane16);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(c0))]);
    let (it, acc) = (vars[0], vars[1]);
    b.buffer_load_lds_b128(rin, o16, c0, ldsb, 0); // descriptor base = o: the input staged in out[0 .. 16*tid) (overwritten by the result after the loop)
    b.wait_vm(0);
    let x = b.load_lds_b32(rd);
    let a2 = b.add_i32(acc, x);
    b.set_var(acc, a2);
    let _ = c1;
    sched_end_loop(&mut b, it, n);
    let c4 = b.c_i32(4);
    let off4 = b.mul_i32(tid, c4);
    let row = b.ptr_add(o, off4);
    let r = b.bitcast_f32(acc);
    b.store_f32(row, 0, r);
    b.finish()
}

/// Stamp probe (`FnBuilder::memtime_lo`): per wave record of 8 u32 in `out`: [t0, t1, t2, t3, x, 0, 0, 0] where t0..t3 are `s_memtime` low words taken around a dependent
/// VALU chain (t0 -> t1), back to back (t1 -> t2, the stamp cost itself) and around a scalar chain (t2 -> t3); x keeps the chains alive. Grid (n, 1, 1), block 256.
pub fn probe_stamp() -> Function {
    let mut b = FnBuilder::new("probe_stamp", 256);
    b.arg("out", DType::Ptr, 0, 8);
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let bx = b.let_(Op::WorkgroupIdX);
    let tid = b.let_(Op::WorkitemId);
    let rec = b.stamp_rec_off(bx, 8, 4);
    let c3 = b.c_i32(3);
    let t0 = b.memtime_lo();
    let mut x = tid;
    for _ in 0..128 {
        x = b.add_i32(x, c3);
        x = b.let_(Op::XorI32(Opnd::Val(x), Opnd::Val(tid)));
    }
    let t1 = b.memtime_lo();
    let t2 = b.memtime_lo();
    let mut y = bx;
    for _ in 0..64 {
        y = b.mul_i32(y, c3);
    }
    let t3 = b.memtime_lo();
    b.stamp_store(out, rec, 0, t0);
    b.stamp_store(out, rec, 1, t1);
    b.stamp_store(out, rec, 2, t2);
    b.stamp_store(out, rec, 3, t3);
    b.stamp_store(out, rec, 4, x);
    b.stamp_store(out, rec, 5, y);
    b.finish()
}

/// Generic one-hot probe of `v_smfmac_f32_16x16x128_fp8_fp8` (docs/isa-probes-smfmac-tr.md): probe `bx` (one 64-lane wave per block) loads lane `l`'s A (4 dwords at
/// `a_in[(64 bx + l) * 16]`), B (8 dwords at `b_in[(64 bx + l) * 32]`) and SRC2 index dword (`i_in[(64 bx + l) * 4]`) straight from global memory, issues ONE
/// smfmac with a zero accumulator and stores D (4 dwords per lane) to `d_out[(64 bx + l) * 16]`. The host chooses every register bit.
pub fn probe_smfmac_gen() -> Function {
    let mut b = FnBuilder::new("probe_smfmac_gen", 64);
    b.arg("a_in", DType::Ptr, 0, 8);
    b.arg("b_in", DType::Ptr, 8, 8);
    b.arg("i_in", DType::Ptr, 16, 8);
    b.arg("d_out", DType::Ptr, 24, 8);
    let a_in = b.let_(Op::KernargPtr { offset: 0 });
    let b_in = b.let_(Op::KernargPtr { offset: 8 });
    let i_in = b.let_(Op::KernargPtr { offset: 16 });
    let d_out = b.let_(Op::KernargPtr { offset: 24 });
    let bx = b.let_(Op::WorkgroupIdX);
    let tid = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let base = b.mul_i32(bx, c64);
    let g = b.add_i32(base, tid);
    let c4 = b.c_i32(4);
    let c16 = b.c_i32(16);
    let c32 = b.c_i32(32);
    let ao = b.mul_i32(g, c16);
    let bo = b.mul_i32(g, c32);
    let io = b.mul_i32(g, c4);
    let ap = b.ptr_add(a_in, ao);
    let bp = b.ptr_add(b_in, bo);
    let ip = b.ptr_add(i_in, io);
    let dp = b.ptr_add(d_out, ao);
    let mut aw = Vec::new();
    for i in 0..4u32 {
        let f = b.load_f32(ap, 4 * i);
        aw.push(b.bitcast_i32(f));
    }
    let mut bw = Vec::new();
    for i in 0..8u32 {
        let f = b.load_f32(bp, 4 * i);
        bw.push(b.bitcast_i32(f));
    }
    let fi = b.load_f32(ip, 0);
    let idx = b.bitcast_i32(fi);
    let a = b.frag4(aw[0], aw[1], aw[2], aw[3]);
    let bb = b.frag8(bw[0], bw[1], bw[2], bw[3], bw[4], bw[5], bw[6], bw[7]);
    let c0 = b.frag_zero(DType::Frag4);
    let d = b.smfmac_fp8_16x16x128(a, bb, idx, c0);
    b.store_f32x4(dp, 0, d);
    b.finish()
}

/// Generic probe of `ds_read_b64_tr_b8`: lane `l` of probe `bx` reads LDS byte address `addr_in[64 bx + l]` (the LDS holds the 4096 bytes of `src`, copied in by the
/// kernel) and stores the two result dwords to `d_out[(64 bx + l) * 8]`.
pub fn probe_lds_tr_b8_gen() -> Function {
    let mut b = FnBuilder::new("probe_lds_tr_b8_gen", 64);
    b.lds(4096);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("addr_in", DType::Ptr, 8, 8);
    b.arg("d_out", DType::Ptr, 16, 8);
    let src = b.let_(Op::KernargPtr { offset: 0 });
    let addr_in = b.let_(Op::KernargPtr { offset: 8 });
    let d_out = b.let_(Op::KernargPtr { offset: 16 });
    let bx = b.let_(Op::WorkgroupIdX);
    let tid = b.let_(Op::WorkitemId);
    let c2 = b.c_i32(2);
    let c3 = b.c_i32(3);
    let c64 = b.c_i32(64);
    // copy the 4096 source bytes into LDS: lane l writes dwords l + 64 k, k = 0..15
    for k in 0..16i32 {
        let ck = b.c_i32(64 * k);
        let dw = b.add_i32(tid, ck);
        let off = b.shl_i32(dw, c2);
        let p = b.ptr_add(src, off);
        let f = b.load_f32(p, 0);
        let w = b.bitcast_i32(f);
        b.store_lds_b32(Opnd::Val(off), Opnd::Val(w));
    }
    b.wait_lds(0);
    b.barrier();
    b.wait_lds(0);
    let base = b.mul_i32(bx, c64);
    let g = b.add_i32(base, tid);
    let ao = b.shl_i32(g, c2);
    let ap = b.ptr_add(addr_in, ao);
    let fa = b.load_f32(ap, 0);
    let addr = b.bitcast_i32(fa);
    let r = b.load_lds_tr_b8(Opnd::Val(addr));
    b.wait_lds(0);
    let oo = b.shl_i32(g, c3);
    let dp = b.ptr_add(d_out, oo);
    for i in 0..2u8 {
        let lane = b.frag16_lane(r, i);
        let v = b.bitcast_f32(lane);
        b.store_f32(dp, 4 * i as u32, v);
    }
    b.finish()
}

/// Scale byte select of `v_mfma_scale_f32_32x32x64_f8f6f4` (op_sel / op_sel_hi, see tools/mfma_opsel_probe.sh): A = B = ones (fp8 1.0), scale words with four
/// different E8M0 bytes, one in-place MFMA per (byte_a, byte_b): out[64 k + tid] = 64 * 2^(ea - 127) * 2^(eb - 127) for the selected bytes.
pub fn probe_mfma_scale_opsel() -> Function {
    let mut b = FnBuilder::new("probe_mfma_scale_opsel", 64);
    b.arg("outp", DType::Ptr, 0, 8);
    let outp = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let zero = b.c_i32(0);
    let one = b.c_i32(1);
    let ones = b.c_i32(0x3838_3838);
    let a8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let b8 = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    // scale A bytes [127, 128, 129, 126], scale B bytes [127, 126, 128, 129]
    let sa = b.c_i32(0x7E81_807Fu32 as i32);
    let sb = b.c_i32(0x8180_7E7Fu32 as i32);
    let off = b.mul_i32(tid, c4);
    for (k, (ba, bb)) in [(0u8, 0u8), (1, 2), (3, 1), (2, 3)].into_iter().enumerate() {
        let z = b.frag_zero(DType::Frag16);
        let vars = b.begin_loop(&[(DType::Frag16, Opnd::Val(z))]);
        b.var_mfma_scale_sel(vars[0], a8, b8, sa, sb, ba, bb);
        let stop = b.cmp_lt_i32(zero, one);
        b.break_if(Opnd::Val(stop));
        b.end_loop();
        let r = b.frag16_lane(Opnd::Var(vars[0]), 0);
        let fr = b.bitcast_f32(r);
        let po = b.ptr_add(outp, off);
        b.store_f32(po, 256 * k as u32, fr);
    }
    b.finish()
}

pub mod handoff;

/// Single-wave MFMA / VALU overlap probe (fmha-4w design): `waves` waves per block (one block per CU, 64 * waves threads), each runs `iters`
/// iterations of `nm` independent in-place scaled 32x32x64 MFMAs (4 accumulators) and `nv` VALU ops on 16 loop-carried f32 chains that never touch
/// the MFMA registers; lane 0 of each wave stores `s_memtime` cycles of the loop to `o[wave]` (u32) and the reductions to `o[64 + ..]` (keeps them live).
pub fn probe_ovl(name: &str, waves: u32, nm: u32, nv: u32, iters: i32) -> Function {
    let mut b = FnBuilder::new(name, 64 * waves);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("i", DType::Ptr, 8, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let inp = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let c63 = b.c_i32(63);
    let c6 = b.c_i32(6);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
    let c32 = b.c_i32(32);
    let lo = b.mul_i32(lane, c32);
    let pa = b.ptr_add(inp, lo);
    let a0 = b.load_b128(pa, 0);
    let a1 = b.load_b128(pa, 16);
    let mk = |b: &mut FnBuilder, fs: [ValueId; 2]| {
        let mut w = Vec::new();
        for f in fs {
            for k in 0..4u8 {
                let x = b.frag4_lane(f, k);
                w.push(b.bitcast_i32(x));
            }
        }
        amdgpu_tile::frag::Frag::<amdgpu_tile::frag::B32x32x64Fp8>::from_words(b, &w).v
    };
    let fa = mk(&mut b, [a0, a1]);
    let fb = mk(&mut b, [a1, a0]);
    let sc = b.c_i32(127);
    let fz = b.frag_zero(DType::Frag16);
    let x0 = b.load_f32(pa, 0);
    let one = b.c_f32(1.0001);
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0))];
    for _ in 0..4 {
        inits.push((DType::Frag16, Opnd::Val(fz)));
    }
    for _ in 0..16 {
        inits.push((DType::F32, Opnd::Val(x0)));
    }
    let z = b.c_i32(0);
    inits[0].1 = Opnd::Val(z);
    b.wait_vm(0);
    let t0 = b.memtime_lo();
    let c4a = b.c_i32(4);
    let woa = b.mul_i32(wave, c4a);
    let poa = b.ptr_add(o, woa);
    let t0v = b.bind_i32(amdgpu_dsl::ev(t0) + 0);
    let t0f = b.bitcast_f32(t0v);
    b.store_f32(poa, 512, t0f);
    let vars = b.begin_loop(&inits);
    let it = vars[0];
    let accs = &vars[1..5];
    let fv = &vars[5..21];
    // interleave: one MFMA, then nv / nm VALU ops
    let per = if nm > 0 { nv / nm } else { nv };
    let mut vals: Vec<Opnd> = fv.iter().map(|&v| Opnd::Var(v)).collect();
    let mut k = 0usize;
    let mut emit_valu = |b: &mut FnBuilder, n: u32, vals: &mut Vec<Opnd>, k: &mut usize| {
        for _ in 0..n {
            let i = *k % 16;
            let j = (*k + 5) % 16;
            let r = match *k % 3 {
                0 => b.mul_f32(vals[i], one),
                1 => b.max3_f32(vals[i], vals[j], one),
                _ => b.add_f32(vals[i], one),
            };
            vals[i] = Opnd::Val(r);
            *k += 1;
        }
    };
    for m in 0..nm.max(1) {
        if nm > 0 {
            b.var_mfma_scale(accs[(m % 4) as usize], fa, fb, sc, sc);
        }
        emit_valu(&mut b, if nm > 0 { per } else { nv }, &mut vals, &mut k);
    }
    for (i, &v) in fv.iter().enumerate() {
        if let Opnd::Val(x) = vals[i] {
            b.set_var(v, x);
        }
    }
    let it1 = b.bind_i32(amdgpu_dsl::varx(it) + 1);
    b.set_var(it, it1);
    let stop = b.bind_cond(!amdgpu_dsl::ev(it1).lt(iters));
    b.break_if(stop);
    b.end_loop();
    let t1 = b.memtime_lo();
    let dt = b.bind_i32(amdgpu_dsl::ev(t1) + 0);
    let c4 = b.c_i32(4);
    let wo = b.mul_i32(wave, c4);
    let po = b.ptr_add(o, wo);
    let f = b.bitcast_f32(dt);
    b.store_f32(po, 0, f);
    let mut s = b.frag16_lane(Opnd::Var(accs[0]), 0);
    for a in &accs[1..] {
        let x = b.frag16_lane(Opnd::Var(*a), 0);
        let xf = b.bitcast_f32(x);
        let sf = b.bitcast_f32(s);
        let y = b.add_f32(sf, xf);
        s = b.bitcast_i32(y);
    }
    let mut acc = b.bitcast_f32(s);
    for &v in fv {
        acc = b.add_f32(acc, Opnd::Var(v));
    }
    b.store_f32(po, 256, acc);
    b.finish()
}