//! MUBUF LDS-direct DMA partitions (`buffer_load_dword[x4] ... lds`), the aiter-GEMM form: `voffset` = the lane's source offset computed ONCE, `soffset`
//! (SGPR) = the per-tile advance, `M0` = the wave's destination base, the instruction immediate shifting source AND destination of each issue.
//!
//! Derivation: for thread `t = lane + 64 wave` and issue q, the TV layout gives the tile element of the issue's first value; `src` maps it to a byte
//! offset in global memory (relative to the descriptor base + soffset), `dst` to the slot-relative LDS byte. The hardware writes lane l at
//! `M0 + imm + bytes * l`, so the destination must be lane-contiguous (proved here by enumeration); `imm_q` = the destination step of issue q (when it fits
//! the 12-bit field), `voff_q = src - imm_q` (one VGPR when the source advances like the destination - the vendor case - otherwise one per distinct
//! offset), `M0_q` = the destination of lane 0 minus `imm_q` (wave-uniform).

use crate::Ctx;
use amdgpu_dsl::tile::shared::{Shared, SlotRef};
use amdgpu_dsl::{FnBuilder, Opnd, ValueId, I32};
use amdgpu_isa::CopyAtom;
use amdgpu_layout::{composition, ComposedLayout, IntTuple, Ix, Layout};

pub struct BufDmaPart {
    sh: Shared,
    bytes: i64,
    voffs: Vec<I32>,
    m0s: Vec<I32>,
    /// per issue: (voffset index, M0 index, M0 constant, immediate)
    issues: Vec<(usize, usize, i64, i64)>,
}

fn intern(v: &mut Vec<Ix>, x: Ix) -> usize {
    match v.iter().position(|y| *y == x) {
        Some(i) => i,
        None => {
            v.push(x);
            v.len() - 1
        }
    }
}

/// `tv`: thread (`lane + 64 wave`, `threads` of them) x value (element) -> tile element; `src`: tile element (column-major index of the tile) -> global
/// byte offset (static strides); `dst`: tile element -> slot-relative LDS byte (swizzle included); `elem_bits` of one element.
#[allow(clippy::too_many_arguments)]
pub fn partition_buffer_dma(b: &mut FnBuilder, ctx: &Ctx, atom: CopyAtom, tv: &Layout, src: &Layout, dst: &ComposedLayout, elem_bits: u32, lane: &Ix, wave: &Ix, sh: &Shared) -> BufDmaPart {
    let lstride = atom.lds_lane_stride.unwrap_or_else(|| panic!("partition_buffer_dma: {} is not an LDS-direct DMA", atom.name));
    let vec = atom.bytes * 8 / elem_bits as i64;
    let (nt, nv) = (tv.get(0).size().st(), tv.get(1).size().st());
    assert!(nv % vec == 0, "partition_buffer_dma: {nv} values per thread are not whole {vec}-element instructions");
    let issues = nv / vec;
    let eb = elem_bits as i64 / 8;
    // build-time proofs on the concrete threads: per issue, the destination is M0 + imm + bytes * lane (lane-contiguous, wave-uniform base) and every
    // instruction moves `bytes` consecutive bytes on both sides
    let (lt, lv) = (tv.get(0), tv.get(1));
    let sd = |t: i64, v: i64| -> (i64, i64) {
        let e = lt.at(t) + lv.at(v);
        (src.at(e), dst.at(e))
    };
    for t in 0..nt {
        for q in 0..issues {
            let (s0, d0) = sd(t, q * vec);
            let (_, dl0) = sd(t - t % 64, q * vec);
            if d0 != dl0 + lstride * (t % 64) {
                panic!("partition_buffer_dma: thread {t} issue {q} lands at LDS byte {d0}, not M0 + {} (the DMA destination is lane-contiguous)", lstride * (t % 64));
            }
            for k in 1..vec {
                let (s, d) = sd(t, q * vec + k);
                if s != s0 + k * eb || d != d0 + k * eb {
                    panic!("partition_buffer_dma: thread {t} issue {q}: value {k} is not contiguous with the first ({s} / {d})");
                }
            }
        }
    }
    let d = &ctx.dyns;
    let thr = lane.add(&wave.mul_c(64));
    let comp_s = composition(src, tv);
    let comp_d = composition(&dst.inner, tv);
    let Ix::Sym(lane_id) = lane else { panic!("partition_buffer_dma: the lane must be a symbol") };
    let mut voffs: Vec<Ix> = Vec::new();
    let mut m0s: Vec<Ix> = Vec::new();
    let mut out = Vec::with_capacity(issues as usize);
    let (_, d00) = sd(0, 0);
    for q in 0..issues {
        let crd = IntTuple::T(vec![IntTuple::L(thr.clone()), IntTuple::L(Ix::C(q * vec))]);
        let s = comp_s.crd2idx(&crd, d);
        let dd = dst.swz.apply_ix(&comp_d.crd2idx(&crd, d), d);
        let (_, dq0) = sd(0, q * vec);
        let step = dq0 - d00;
        let imm = if atom.imm.fits(step) { step } else { 0 };
        let vo = intern(&mut voffs, s.add_c(-imm));
        let m0 = dd.subst(*lane_id, &Ix::C(0), d).add_c(-imm);
        let (m0d, m0c) = m0.split_const();
        let mi = intern(&mut m0s, m0d);
        out.push((vo, mi, m0c, imm));
    }
    let voffs = voffs.iter().map(|x| ctx.bind(b, x)).collect();
    let m0s = m0s.iter().map(|x| { let v = ctx.bind(b, x); if sh.base == 0 { v } else { b.let_i32(v + sh.base as i32) } }).collect();
    BufDmaPart { sh: sh.clone(), bytes: atom.bytes, voffs, m0s, issues: out }
}

impl BufDmaPart {
    /// The DMA of one tile into `slot`: `soff` = the tile's advance in the source (SGPR), `rsrc` = the buffer descriptor.
    pub fn issue(&self, b: &mut FnBuilder, rsrc: ValueId, soff: impl Into<Opnd> + Copy, slot: &SlotRef) {
        assert_eq!(slot.sh.name, self.sh.name, "BufDmaPart: slot of another buffer");
        let off = slot.off_v(b);
        let m0b: Vec<I32> = self.m0s.iter().map(|m| if off.as_const() == Some(0) { m.clone() } else { b.let_i32(m + &off) }).collect();
        slot.write(b, |b| {
            for &(vo, mi, mc, imm) in &self.issues {
                let lds: ValueId = if mc == 0 { b.bind_i32(&m0b[mi]) } else { b.bind_i32(&m0b[mi] + mc as i32) };
                let v = self.voffs[vo].clone();
                let vv = b.bind_i32(v);
                match self.bytes {
                    16 => b.buffer_load_lds_b128(rsrc, vv, soff, lds, imm as u32),
                    4 => b.buffer_load_lds_b32(rsrc, vv, soff, lds, imm as u32),
                    n => panic!("BufDmaPart: no {n}-byte LDS-direct buffer load"),
                }
            }
        });
    }
    /// Instructions per tile.
    pub fn len(&self) -> usize {
        self.issues.len()
    }
    pub fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }
}
