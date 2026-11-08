//! MoE combine (docs/op-coverage-memtemplates.md T3): the top-K per-route expert outputs of a token are summed into one row.
//!
//! * `moe_reduce_bf16_k<K>_d<D>`  `out[t][d] = bf16(sum_k w[t][k] * x[t][k][d])`.  ABI x @0 (bf16 [T][K][D]), w @8 (f32 [T][K]), out @16 (bf16 [T][D]), T @24 (i32)
//! * `moe_combine_bf16_k<K>_d<D>`     `out[t][d] = bf16(sum_k x[t][k][d])`.            ABI x @0, out @8, T @16
//!
//! K 1..=16, D a multiple of 8 in 8..=65536; grid x = the column tiles (`pick_threads(D / 8)` threads, one b128 per lane and route), grid y = tokens (T <= 65535). The buffer descriptors are moved to the token's row
//! (64-bit `ptr_mad`, num_records = one token), so x may exceed 4 GiB.
//! All K loads of a lane are in flight before the sum; accumulation in f32, one round to bf16 (RNE); streaming (`nt`) accesses.

use amdgpu_dsl::{CachePolicy, CompileProfile, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::prims::bf16x8_to_f32;
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, weighted) = if let Some(r) = name.strip_prefix("moe_reduce_bf16_k") {
        (r, true)
    } else if let Some(r) = name.strip_prefix("moe_combine_bf16_k") {
        (r, false)
    } else {
        return None;
    };
    let (k, d) = rest.split_once("_d")?;
    let (k, d): (i64, i64) = (k.parse().ok()?, d.parse().ok()?);
    ((1..=16).contains(&k) && d % 8 == 0 && (8..=65536).contains(&d)).then(|| moe_reduce(name, k, d, weighted))
}

fn pack8(b: &mut FnBuilder, r: &[ValueId; 8]) -> ValueId {
    let w: Vec<ValueId> = (0..4).map(|j| b.cvt_pk_bf16_f32(r[2 * j], r[2 * j + 1])).collect();
    b.frag4(w[0], w[1], w[2], w[3])
}

/// Threads per block (64, 128, 192 or 256, at most as many waves as the row needs): the one that leaves the fewest idle lanes in the last column tile (ties: the larger block).
pub fn pick_threads(d8: i64) -> u32 {
    let max_waves = ((d8 + 63) / 64).min(4);
    let mut best = (i64::MAX, 64u32);
    for w in 1..=max_waves {
        let t = 64 * w;
        let waste = ((d8 + t - 1) / t) * t - d8;
        if waste <= best.0 {
            best = (waste, t as u32);
        }
    }
    best.1
}

pub fn moe_reduce(name: &str, k: i64, d: i64, weighted: bool) -> Function {
    let d8 = d / 8;
    let threads = pick_threads(d8);
    let tiles = (d8 + threads as i64 - 1) / threads as i64;
    let mut fb = FnBuilder::new(name, threads);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    let (o_off, t_off) = if weighted {
        b.arg("w", DType::Ptr, 8, 8);
        b.arg("out", DType::Ptr, 16, 8);
        (16, 24)
    } else {
        b.arg("out", DType::Ptr, 8, 8);
        (8, 16)
    };
    b.arg("t", DType::I32, t_off, 4);
    // the descriptors are moved to the token's row with a 64-bit `ptr_mad` (num_records = one token), so the tensors may exceed 4 GiB; T is bounded by the grid (y <= 65535)
    let tmax = 65535i64;
    let tt = Ext::sym("T");
    b.sym_range("T", 1, tmax);
    b.ka_is(t_off, tt.clone());
    b.grid_is(0, Ext::c(tiles));
    b.grid_is(1, tt.clone());
    b.arg_tensor("x", tt.clone() * Ext::c(2 * k * d));
    if weighted {
        b.arg_tensor("w", tt.clone() * Ext::c(4 * k));
    }
    b.arg_tensor("out", tt * Ext::c(2 * d));
    let p_x = b.let_(Op::KernargPtr { offset: 0 });
    let p_w = if weighted { Some(b.let_(Op::KernargPtr { offset: 8 })) } else { None };
    let p_o = b.let_(Op::KernargPtr { offset: o_off });
    let t_v = b.let_(Op::KernargI32 { offset: t_off });
    let tid = b.let_(Op::WorkitemId);
    let tile = b.let_(Op::WorkgroupIdX);
    let row = b.let_(Op::WorkgroupIdY);
    let rowc = b.let_(Op::AddI32(Opnd::Val(row), Opnd::I32(0)));
    let (cx, cw, co) = (b.c_i32((2 * k * d) as i32), b.c_i32((4 * k) as i32), b.c_i32((2 * d) as i32));
    let px = b.ptr_mad(p_x, rowc, cx);
    let r_x = b.buffer_desc(px, (2 * k * d) as u32);
    let r_w = p_w.map(|p| { let pw = b.ptr_mad(p, rowc, cw); b.buffer_desc(pw, (4 * k) as u32) });
    let po = b.ptr_mad(p_o, rowc, co);
    let r_o = b.buffer_desc(po, (2 * d) as u32);
    let col8 = b.bind_i32(ev(tile) * threads as i32 + ev(tid));
    let valid = b.let_(Op::CmpLtI32(Opnd::Val(col8), Opnd::I32(d8 as i32)));
    // the K routes of this token: all loads first
    let base = b.bind_i32(ev(col8) << 4);
    let mut xs = Vec::new();
    for r in 0..k {
        let off = if r == 0 { base } else { b.bind_i32(ev(base) + (2 * d * r) as i32) };
        xs.push(b.buffer_load_c(r_x, Some(Opnd::Val(off)), 0i32, 0, MemTy::B128, CachePolicy::NT));
    }
    let ws: Vec<ValueId> = match r_w {
        Some(rw) => (0..k)
            .map(|r| {
                let wo = b.c_i32((4 * r) as i32);
                b.buffer_load_f32(rw, wo, 0i32, 0)
            })
            .collect(),
        None => Vec::new(),
    };
    let mut acc: Option<[ValueId; 8]> = None;
    for r in 0..k as usize {
        let f = bf16x8_to_f32(b, xs[r]);
        let mut term = [0; 8];
        for i in 0..8 {
            term[i] = if weighted { b.mul_f32(f[i], ws[r]) } else { f[i] };
        }
        acc = Some(match acc {
            None => term,
            Some(a) => {
                let mut n = [0; 8];
                for i in 0..8 {
                    n[i] = b.add_f32(a[i], term[i]);
                }
                n
            }
        });
    }
    let o = pack8(b, &acc.unwrap());
    let offo = b.bind_i32(ev(col8) << 4);
    let offo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(valid), a: Opnd::Val(offo), b: Opnd::I32(-16) });
    b.buffer_store_c(r_o, Some(Opnd::Val(offo)), 0i32, 0, o, MemTy::B128, false, CachePolicy::NT);
    fb.finish()
}