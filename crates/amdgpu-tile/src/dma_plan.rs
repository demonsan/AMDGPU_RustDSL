//! GEMM operand staging: the K-tile LDS image and the per-issue DMA plans (dense, padded A image, gathered MoE rows).

use amdgpu_dsl::{FnBuilder, I32, ValueId};
use amdgpu_layout::{composition, IssuePlan, Ix, make_layout, right_inverse, tup};
use crate::{Ctx, ev, Lane};
use amdgpu_layout::Layout;

/// LDS image of a [rows x 128 bytes] K-tile: 16-row x 16-byte interleave (the aiter B preshuffle pattern with K = 128):
/// (r, k) -> (r / 16) 2048 + (k / 16) 256 + (r % 16) 16 + k % 16. A 16-lane group of every operand read covers 256 consecutive bytes (all banks once).
pub fn smem_tile(rows: i64) -> Layout {
    make_layout(tup![tup![16, rows / 16], tup![16, 8]], tup![tup![16, 2048], tup![1, 256]])
}
/// The same image restricted to one 16-row MMA operand block (element index r + 16 k, 16 x 128).
pub fn smem_block() -> Layout {
    make_layout(tup![16, tup![16, 8]], tup![16, tup![1, 256]])
}

/// One DMA stage of a [rows x 128 B] tile: per issue the lane's voffset (bound once, loop invariant) and the LDS offset of its 1 KB block.
pub struct Dma {
    pub issues: Vec<(ValueId, i32)>,
    /// per-lane gathered rows (MoE): every issue its own M0, no instruction offset (the voffset is the exact source offset)
    pub gather: bool,
    /// LDS bytes per 1 KB issue block (1024, or 1056 for the padded A image); the issue LDS offsets above are in 1 KB units (`act`)
    pub pitch: i32,
}

impl Dma {
    /// (M0 offset of the issue's group of 4, instruction offset) for the virtual 1 KB offset `lds`
    pub fn act(&self, lds: i32) -> (i32, u32) {
        if self.gather {
            return (lds / 1024 * self.pitch, 0);
        }
        ((lds - lds % 4096) / 1024 * self.pitch, ((lds % 4096) / 1024 * self.pitch) as u32)
    }
    /// the M0 group of an issue
    pub fn group(&self, lds: i32) -> i32 {
        if self.gather { lds } else { lds - lds % 4096 }
    }
}

/// Turns the per-issue source offsets (minus the instruction offset `pitch (q % 4)`) into voffsets, lowered through the GF(2) bit-form normal form
/// (amdgpu_layout::BitForm: shift groups, CSE across plans by Ctx::bind).
pub fn finish_dma_p(b: &mut FnBuilder, ctx: &Ctx, gofs: Vec<Ix>, pitch: i32) -> Dma {
    let adj: Vec<Ix> = gofs.iter().enumerate().map(|(q, g)| g.add_c(-(pitch as i64) * (q as i64 % 4))).collect();
    ctx.set_bitfield(true);
    let issues = adj.iter().enumerate().map(|(q, g)| { let e = ctx.bind(b, g); (b.bind_i32(e), 1024 * q as i32) }).collect();
    ctx.set_bitfield(false);
    Dma { issues, pitch, gather: false }
}
/// LDS byte `d = 1024 (issues (tid / 64) + q) + 16 (tid % 64)` holds tile element `smem^-1(d)`, so the source offset is `glob o right_inverse(smem)`;
/// proved here for sampled threads against a direct search of the LDS layout.
pub fn dma_plan(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64, waves: i64) -> Dma {
    let issues = rows * 128 / 1024 / waves;
    let sm = smem_tile(rows);
    let tv = make_layout(tup![tup![64, waves], tup![16, issues]], tup![tup![16, 1024 * issues], tup![1, 1024]]);
    let g = composition(glob, &right_inverse(&sm));
    let gofs = IssuePlan::new(tv, 16).offsets(&g, s_tid, &Ix::C(0), &ctx.dyns);
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    for th in [0i64, 1, 15, 16, 63, 64, 100, 127, 255].into_iter().filter(|t| *t < 64 * waves) {
        let env = |s: u32| if s == tid_id { th } else { proof_k };
        for q in 0..issues {
            let lds = 1024 * (issues * (th / 64) + q) + 16 * (th % 64);
            let (r, c) = (0..rows).flat_map(|r| (0..8).map(move |c| (r, 16 * c))).find(|&(r, c)| sm.at(r + rows * c) == lds).expect("LDS byte covered");
            let want = glob.crd2idx(&tup![r, c], &ctx.dyns).eval(&env);
            assert_eq!(gofs[q as usize].eval(&env), want, "DMA rows {rows} thread {th} issue {q}");
        }
    }
    // MUBUF form: voffset (loop invariant) + soffset + the instruction offset 1024 (q % 4) = the global offset; M0 = the group's first 1 KB block
    finish_dma_p(b, ctx, gofs, 1024)
}

/// The A DMA of the padded image with GATHERED rows (MoE): the same lane -> (tile row lane / 8 + 8 block, bytes 16 (lane % 8)) map as
/// `dma_plan_rows_pad`; the source row of tile row r is `row_src(r)` (per lane, loop invariant); exact voffsets, no instruction offset.
pub fn dma_plan_gather(b: &mut FnBuilder, lane: &Lane, rows: i64, waves: i64, k_v: ValueId, row_src: &dyn Fn(&mut FnBuilder, I32) -> ValueId) -> Dma {
    let issues = rows * 128 / 1024 / waves;
    let mut out = Vec::new();
    for q in 0..issues {
        let r = (lane.wave() * issues as i32 + q as i32) * 8 + lane.lane().lshr(3);
        let src = row_src(b, r);
        let v = b.bind_i32(ev(src) * ev(k_v) + ((lane.lane() & 7) << 4));
        out.push((v, 1024 * q as i32));
    }
    // the row-index loads complete here: the pipeline's counted vmcnt waits count only its DMAs
    b.wait_vm(0);
    Dma { issues: out, pitch: 1056, gather: true }
}

/// A-tile DMA for the PADDED row-major LDS image (the vendor form): 1 KB blocks of 8 whole 128-byte rows at a pitch of 1056 bytes (32 bytes of
/// padding per block spread the banks of the fragment reads), lanes linear in the source (lane -> row lane / 8, bytes 16 (lane % 8)): fully
/// monotone 128-byte segments, no source-side permutation. LDS image: (r, k) -> 1056 (r / 8) + 128 (r % 8) + k.
pub fn dma_plan_rows_pad(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64, waves: i64) -> Dma {
    let issues = rows * 128 / 1024 / waves;
    let d = &ctx.dyns;
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    let mut out = Vec::new();
    for q in 0..issues {
        // block index (issues (tid / 64) + q), row inside the block lane / 8, bytes 16 (lane % 8)
        let blk = s_tid.div_c(64, d).mul_c(issues).add_c(q);
        let lane = s_tid.mod_c(64, d);
        let r = blk.mul_c(8).add(&lane.div_c(8, d));
        let k = lane.mod_c(8, d).mul_c(16);
        let g = glob.crd2idx(&tup![r.clone(), k.clone()], d);
        for th in [0i64, 1, 7, 8, 63, 64, 100, 127, 255].into_iter().filter(|t| *t < 64 * waves) {
            let env = |s: u32| if s == tid_id { th } else { proof_k };
            let (bb, ll) = (issues * (th / 64) + q, th % 64);
            let want = glob.crd2idx(&tup![8 * bb + ll / 8, 16 * (ll % 8)], d).eval(&env);
            assert_eq!(g.eval(&env), want, "padded A DMA thread {th} issue {q}");
        }
        out.push(g);
    }
    finish_dma_p(b, ctx, out, 1056)
}
