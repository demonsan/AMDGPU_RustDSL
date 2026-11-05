//! L6b: epilogues of the transposed attention (`O^T` accumulators: lane (j, h) holds dims `32 e + 8 q + 4 h + k` of query j) and the KV-split combine.

use crate::frag::{AccTile, C32x32F32};
use crate::shared::PrivRef;
use amdgpu_dsl::{ev};
use crate::{Lane};
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd, ValueId, I32};

/// `O * fin -> bf16` straight to global memory: per (e, q) the lane's 4 consecutive dims as one 8-byte store (`global_store_dwordx2`, 64 per lane), all
/// under `if live` (the wave has at least one valid row). `row_ptr` = the lane's output row INCLUDING the lane-half term `8 (lane >> 5)` bytes.
pub fn store_bf16(b: &mut FnBuilder, o: &AccTile<C32x32F32, 4>, fin: ValueId, row_ptr: ValueId, live: ValueId) {
    b.begin_if(Opnd::Val(live));
    for e in 0..4usize {
        for q in 0..4usize {
            let mut v4: Vec<ValueId> = Vec::with_capacity(4);
            for k in 0..4usize {
                let l = b.frag16_lane(Opnd::Var(o.accs[e].var), (4 * q + k) as u8);
                let lv = b.bitcast_f32(l);
                v4.push(b.mul_f32(lv, fin));
            }
            let w0 = b.cvt_pk_bf16_f32(v4[0], v4[1]);
            let w1 = b.cvt_pk_bf16_f32(v4[2], v4[3]);
            let f2 = b.frag2(w0, w1);
            b.store_f32x2(row_ptr, (64 * e + 16 * q) as u32, f2);
        }
    }
    b.end_if();
}

/// The KV-split partial epilogue: `lse` of the lane's row, and `O * fin` as fp16 through the wave's private 4 KB LDS transpose scratch, stored as
/// coalesced 16-byte segments of 128-byte partial rows (8 lanes per row). Inputs:
/// * `lse_ptr`: the lane's lse element; `live`: the wave has a valid row;
/// * `row_ptr(i)`: global pointer of the partial row read back by this lane in round i (rows `row0 + 8 i + (lane >> 3)`, clamped by the caller) INCLUDING the
///   lane's 16-byte chunk `16 (lane & 7)`;
/// * `epi`: the wave's private slice (`LdsPlan::private`), whose storage reuses dead buffers: the caller must have synchronised the workgroup before.
#[allow(clippy::too_many_arguments)]
pub fn kvsplit_partial_f16(b: &mut FnBuilder, o: &AccTile<C32x32F32, 4>, fin: ValueId, lane: &Lane, epi: &PrivRef, lse_ptr: ValueId, lse: ValueId, live: ValueId, row_ptr: &mut dyn FnMut(&mut FnBuilder, usize) -> ValueId) {
    b.begin_if(Opnd::Val(live));
    b.store_f32(lse_ptr, 0, lse);
    let lbase = b.let_i32(epi.base());
    let jm = b.let_i32(lane.j() & 7);
    let jbase = b.let_i32(&lbase + (lane.j() << 7));
    let half8 = b.let_i32(lane.half() << 3);
    // write address of chunk (4 ec + q) of this lane's row: ((chunk ^ (j & 7)) << 4) + 8 half
    let mut waddr: Vec<ValueId> = Vec::with_capacity(8);
    for ec in 0..2i32 {
        for q in 0..4i32 {
            waddr.push(b.bind_i32((&jbase + ((I32::c(4 * ec + q) ^ &jm) << 4)) + &half8));
        }
    }
    let lrow = b.let_i32(lane.lane().lshr(3));
    let lch = b.let_i32(lane.lane() & 7);
    let pch16 = b.let_i32((&lch ^ &lrow) << 4);
    let raddr: Vec<ValueId> = (0..4i32).map(|i| b.bind_i32((&lbase + ((&lrow + 8 * i) << 7)) + &pch16)).collect();
    let gptr: Vec<ValueId> = (0..4usize).map(|i| row_ptr(b, i)).collect();
    let c16 = b.c_i32(16);
    let c4 = b.c_i32(4);
    for pr in 0..2usize {
        for ec in 0..2usize {
            let e = 2 * pr + ec;
            for q in 0..4usize {
                let mut hv: Vec<ValueId> = Vec::with_capacity(4);
                for k in 0..4usize {
                    let l = b.frag16_lane(Opnd::Var(o.accs[e].var), (4 * q + k) as u8);
                    let lv = b.bitcast_f32(l);
                    let v0 = b.mul_f32(lv, fin);
                    hv.push(b.let_(Op::CvtF32ToF16(Opnd::Val(v0))));
                }
                let s0 = b.shl_i32(hv[1], c16);
                let s1 = b.shl_i32(hv[3], c16);
                let w0 = b.let_(Op::AddI32(Opnd::Val(hv[0]), Opnd::Val(s0)));
                let w1 = b.let_(Op::AddI32(Opnd::Val(hv[2]), Opnd::Val(s1)));
                let ad = waddr[4 * ec + q];
                epi.write(b, 0, |b| {
                    b.store_lds_b32(Opnd::Val(ad), Opnd::Val(w0));
                    let ad4 = b.add_i32(ad, c4);
                    b.store_lds_b32(Opnd::Val(ad4), Opnd::Val(w1));
                });
            }
        }
        b.wait_lds(0);
        for i in 0..4usize {
            let v = epi.read_back(b, 0, |b| b.load_lds_b128(Opnd::Val(raddr[i])));
            b.store_f32x4(gptr[i], (128 * pr) as u32, v);
        }
        b.wait_lds(0);
    }
    b.end_if();
}

/// The KV-split combine over `splits` fp16 partials, `per_thread` (8 or 16) consecutive output elements per thread: online max / rescale over the
/// splits (`m`, `den = sum 2^(lse - m)`, `acc = sum 2^(lse - m) Ohat`), bf16 output. `t` = the (clamped) thread element-group index, `row` = its row.
pub fn combine_splits(b: &mut FnBuilder, ohat: ValueId, lse: ValueId, out: ValueId, rows: ValueId, splits: ValueId, t: ValueId, row: ValueId, per_thread: usize, split_bound: amdgpu_dsl::Ext) {
    assert!(per_thread == 8 || per_thread == 16);
    let n = per_thread;
    let groups_lg = if n == 8 { 4 } else { 3 }; // 128 / n groups per row
    let bytes_lg = if n == 8 { 4 } else { 5 };
    let c0 = b.c_i32(0);
    let neg_big = b.c_f32(-1.0e30);
    let neg_big_i = b.bitcast_i32(neg_big);
    let zero_f = b.c_f32(0.0);
    let zero_i = b.bitcast_i32(zero_f);
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(neg_big_i))];
    for _ in 0..(n + 1) {
        inits.push((DType::I32, Opnd::Val(zero_i)));
    }
    let vars = b.begin_loop(&inits);
    let sv = vars[0];
    b.assume_trusted(sv, amdgpu_dsl::Ext::c(0), split_bound, "split counter (tile::combine_splits): starts at 0 and the loop exits when s + 1 >= splits, so the body only runs with s <= splits - 1");
    let rs = b.bind_i32(amdgpu_dsl::varx(sv) * ev(rows));
    let lp = {
        let o = b.bind_i32((&ev(rs) + ev(row)) << 2);
        b.ptr_add(lse, o)
    };
    let l = b.load_f32(lp, 0);
    let op = {
        let o = b.bind_i32(((ev(rs) << groups_lg) + ev(t)) << bytes_lg);
        b.ptr_add(ohat, o)
    };
    let ov: Vec<ValueId> = (0..n / 8).map(|h| b.load_b128(op, (16 * h) as u32)).collect();
    let m_old = b.bitcast_f32(vars[1]);
    let den = b.bitcast_f32(vars[2]);
    let m_new = b.max_f32(m_old, l);
    let neg1 = b.c_f32(-1.0);
    let dm = b.mul_f32(m_new, neg1);
    let d_old = b.add_f32(m_old, dm);
    let d_new = b.add_f32(l, dm);
    let a = b.exp2_f32(d_old);
    let w = b.exp2_f32(d_new);
    let den_s = b.mul_f32(den, a);
    let den_n = b.add_f32(den_s, w);
    let m_i = b.bitcast_i32(m_new);
    let den_i = b.bitcast_i32(den_n);
    b.set_var(vars[1], m_i);
    b.set_var(vars[2], den_i);
    let c16 = b.c_i32(16);
    for wi in 0..n / 2 {
        let wd = b.frag4_lane(ov[wi / 4], (wi % 4) as u8);
        let hs = b.let_(Op::LshrI32(Opnd::Val(wd), Opnd::Val(c16)));
        let lo = b.let_(Op::CvtF16ToF32(Opnd::Val(wd)));
        let hi = b.let_(Op::CvtF16ToF32(Opnd::Val(hs)));
        for (k, vv) in [(2 * wi, lo), (2 * wi + 1, hi)] {
            let acc = b.bitcast_f32(vars[3 + k]);
            let acc_s = b.mul_f32(acc, a);
            let wv = b.mul_f32(w, vv);
            let acc_n = b.add_f32(acc_s, wv);
            let acc_i = b.bitcast_i32(acc_n);
            b.set_var(vars[3 + k], acc_i);
        }
    }
    let it_next = b.bind_i32(amdgpu_dsl::varx(sv) + 1);
    b.set_var(sv, it_next);
    let stop = b.bind_cond(!ev(it_next).lt(ev(splits)));
    b.break_if(stop);
    b.end_loop();
    let denf = b.bitcast_f32(Opnd::Var(vars[2]));
    let rd = b.rcp_f32(denf);
    let res: Vec<ValueId> = (0..n)
        .map(|k| {
            let accf = b.bitcast_f32(Opnd::Var(vars[3 + k]));
            b.mul_f32(accf, rd)
        })
        .collect();
    let ow: Vec<ValueId> = (0..n / 2).map(|k| b.cvt_pk_bf16_f32(res[2 * k], res[2 * k + 1])).collect();
    let ob = b.bind_i32(ev(t) << bytes_lg); // bf16 out: the same 2 bytes per element as the fp16 partials
    let optr = b.ptr_add(out, ob);
    for h in 0..n / 8 {
        let f4 = b.frag4(ow[4 * h], ow[4 * h + 1], ow[4 * h + 2], ow[4 * h + 3]);
        b.store_f32x4(optr, (16 * h) as u32, f4);
    }
}

