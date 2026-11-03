//! MXFP8 blockscale GEMM (P4 target: aiter PR #6696 4-wave pin-AGPR kernel, docs/mxfp8-gemm-4wave-analysis.md), same 96-byte ABI:
//!   C[m, n] (bf16) = sum_k A[m, k] 2^(sa[k/128][m] - 127) * B[n, k] 2^(sb[n/128][k/128] - 127)
//!   A [M, K] fp8 e4m3 row-major; B [N, K] fp8 preshuffled (aiter shuffle_weight 16x16); sa E8M0 [K/128][M]; sb E8M0 [N/128][K/128].
//!
//! v0 (correctness first): VGPR-only, no LDS. CTA 128x128 (4 waves 2x2, 64x64 per wave = 2x2 tiles of v_mfma_scale_f32_32x32x64_f8f6f4), K step 64,
//! operand fragments loaded straight from global memory. EVERY address is derived from `amdgpu-layout` layouts (tensor layouts x MFMA thread-value
//! layouts, issue plans with immediate folding); the only hand code is the lowering of the symbolic index expressions to DSL integer ops.
//! The MFMA runs "swapped" (hardware A = the B fragment, hardware B = the A fragment, scales swapped), so a lane holds 4 consecutive N of one row M
//! of C (one 8-byte bf16 store per 4 values), exactly like the vendor kernel.

use amdgpu_layout::{composition, layout_to_offsets, make_layout, right_inverse, tup, Dyns, ImmRange, IssuePlan, Ix, Layout, Swizzle};

use crate::{DType, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};

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
    let ptrs = [("ptr_a", 0u32), ("ptr_b", 8), ("ptr_c", 16), ("ptr_sfa", 64), ("ptr_sfb", 72)];
    let ints = [("m", 24u32), ("n", 28), ("k", 32), ("batch", 36), ("stride_a", 40), ("stride_b", 44), ("stride_c", 48), ("stride_a_batch", 52), ("stride_b_batch", 56), ("stride_c_batch", 60), ("stride_sfa", 80), ("stride_sfb", 84), ("stride_sfa_batch", 88), ("stride_sfb_batch", 92)];
    let mut all: Vec<(&str, u32, bool)> = ptrs.iter().map(|&(n, o)| (n, o, true)).chain(ints.iter().map(|&(n, o)| (n, o, false))).collect();
    all.sort_by_key(|x| x.1);
    for (name, off, is_ptr) in all {
        if is_ptr { b.arg(name, DType::Ptr, off, 8) } else { b.arg(name, DType::I32, off, 4) }
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

// ======================================================================================================================================================
// v1: LDS-staged, 3-slot ring (tile::Pipeline OneWindow), CTA 256 (M) x 128 (N), 4 waves 2x2 of 128x64, still VGPR-only (8 accumulators)
// ======================================================================================================================================================

/// LDS layout of a [rows x 128 bytes] K-tile (rows = M for A, N for B): 16-row x 16-byte interleave, the same pattern as the aiter B preshuffle
/// ((r, k) -> (r / 16) 2048 + (k / 16) 256 + (r % 16) 16 + k % 16): every ds_read_b128 of an MFMA operand (16 consecutive rows, one 16-byte chunk per
/// 16 lanes) touches 256 consecutive bytes = all 64 banks once.
fn smem_tile(rows: i64) -> Layout {
    make_layout(tup![tup![16, rows / 16], tup![16, 8]], tup![tup![16, 2048], tup![1, 256]])
}

const CTA_M: i64 = 256;
const CTA_N: i64 = 128;

struct GemmV1 {
    // ring buffers
    a_s: crate::tile::Shared,
    b_s: crate::tile::Shared,
    // DMA: per issue (global byte offset expression relative to the tile origin + k, LDS byte offset inside the slot); M0 = slot base + lds offset
    a_dma: Vec<(I32, i64)>,
    b_dma: Vec<(I32, i64)>,
    a_org: ValueId,
    b_org: ValueId,
    ptr_a: ValueId,
    ptr_b: ValueId,
    // fragment reads: [subtile][h] -> two (base, imm) LDS byte offsets inside a slot
    a_rd: Vec<Vec<[(ValueId, i64); 2]>>,
    b_rd: Vec<Vec<[(ValueId, i64); 2]>>,
    // scales: per m-subtile the sa byte offset of row (kb = 0), the sb byte offset (kb = 0)
    sa_byte: Vec<ValueId>,
    sb_byte: ValueId,
    ptr_sfa: ValueId,
    ptr_sfb: ValueId,
    m_v: ValueId,
    zero16: ValueId,
    acc: Vec<VarId>,
    a_wave: ValueId,
    b_wave: ValueId,
    // scale staging: per slot and wave 512 bytes = [256 A-scale bytes of the CTA rows | the dword holding the B-scale byte, every lane]
    sc_s: crate::tile::Shared,
    sc_wave: ValueId,
    sa_src: ValueId,
    sb_src: ValueId,
    kb_v: ValueId,
    sa_rd: Vec<ValueId>,
    sb_shift: ValueId,
}

fn load_scale(b: &mut FnBuilder, ptr: ValueId, byte: I32) -> ValueId {
    let byte = b.bind_i32(byte);
    let dw = b.bind_i32(crate::tile::ev(byte) & -4);
    let p = b.ptr_add(ptr, dw);
    let w = b.load_f32(p, 0);
    let wi = b.bitcast_i32(w);
    b.bind_i32(crate::tile::ev(wi).lshr((crate::tile::ev(byte) & 3) << 3))
}

impl crate::tile::PipeBody for GemmV1 {
    fn stage(&mut self, b: &mut FnBuilder, _role: crate::tile::Role, key0: &I32, slot: &crate::tile::SlotIdx) {
        // A: + k bytes; B preshuffled: + (k / 16) 256 = 16 k bytes
        let ka = b.bind_i32(key0.clone() + crate::tile::ev(self.a_org));
        let kb = b.bind_i32((key0.clone() << 4) + crate::tile::ev(self.b_org));
        for (sh, dma, org, ptr, wave) in [(&self.a_s, &self.a_dma, ka, self.ptr_a, self.a_wave), (&self.b_s, &self.b_dma, kb, self.ptr_b, self.b_wave)] {
            let r = sh.at(slot);
            let sbase = r.off_v(b) + sh.base as i32 + crate::tile::ev(wave);
            r.write(b, |b| {
                for (g, lds) in dma {
                    let off = b.bind_i32(crate::tile::ev(org) + g.clone());
                    let p = b.ptr_add(ptr, off);
                    let m0 = b.bind_i32(sbase.clone() + *lds as i32);
                    b.load_global_lds(p, m0, 0, 4);
                }
            });
        }
        // scales of tile kb = key0 / 128: A rows m_cta.. (256 contiguous bytes of the [K/128][M] tensor), the B-scale dword (same address on every lane)
        let kb = b.bind_i32(key0.clone().lshr(7));
        let r = self.sc_s.at(slot);
        let ro = r.off_v(b);
        let sbase = b.bind_i32(ro + self.sc_s.base as i32 + crate::tile::ev(self.sc_wave));
        let pa = {
            let o = b.bind_i32(crate::tile::ev(kb) * crate::tile::ev(self.m_v) + crate::tile::ev(self.sa_src));
            b.ptr_add(self.ptr_sfa, o)
        };
        let pb = {
            let o = b.bind_i32((crate::tile::ev(self.sb_src) + crate::tile::ev(kb)) & -4);
            b.ptr_add(self.ptr_sfb, o)
        };
        let sb_m0 = b.bind_i32(crate::tile::ev(sbase) + 256);
        r.write(b, |b| {
            b.load_global_lds(pa, sbase, 0, 1);
            b.load_global_lds(pb, sb_m0, 0, 1);
        });
    }
    fn prologue_compute(&mut self, _b: &mut FnBuilder, _role: crate::tile::Role, _t0: &crate::tile::TileIt) {}
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        (0..8).map(|_| (DType::Frag16, Opnd::Val(self.zero16))).collect()
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        self.acc = vars.to_vec();
    }
    fn window(&mut self, b: &mut FnBuilder, _role: crate::tile::Role, t: &crate::tile::TileIt) {
        let kbx = crate::tile::ox(t.it);
        let scr = self.sc_s.at(&t.cur);
        let so = scr.off_v(b);
        let scb = b.bind_i32(so + self.sc_s.base as i32 + crate::tile::ev(self.sc_wave));
        let (sa, sb) = scr.read(b, |b| {
            let sa: Vec<ValueId> = (0..4).map(|i| {
                // the dword holding row byte r = 128 wm + 32 i + lane % 32, shifted to byte 0
                let a = b.bind_i32(crate::tile::ev(scb) + (crate::tile::ev(self.sa_rd[i]) & -4));
                let w = b.load_lds_b32(a);
                b.bind_i32(crate::tile::ev(w).lshr((crate::tile::ev(self.sa_rd[i]) & 3) << 3))
            }).collect();
            let a = b.bind_i32(crate::tile::ev(scb) + 256);
            let w = b.load_lds_b32(a);
            let sb = b.bind_i32(crate::tile::ev(w).lshr((((crate::tile::ev(self.sb_src) + kbx.clone()) & 3) << 3) + crate::tile::ev(self.sb_shift)));
            (sa, sb)
        });
        let _ = (&self.sa_byte, self.sb_byte, self.kb_v);
        let ar = self.a_s.at(&t.cur);
        let br = self.b_s.at(&t.cur);
        let (aoff, boff) = (ar.off_v(b) + self.a_s.base as i32, br.off_v(b) + self.b_s.base as i32);
        for h in 0..2usize {
            let rd = |b: &mut FnBuilder, r: &crate::tile::SlotRef, off: &I32, p: &[(ValueId, i64); 2]| -> ValueId {
                let a0 = b.bind_i32(off.clone() + crate::tile::ev(p[0].0) + p[0].1 as i32);
                let a1 = b.bind_i32(off.clone() + crate::tile::ev(p[1].0) + p[1].1 as i32);
                r.read(b, |b| b.load_lds_b128x2(a0, a1))
            };
            let af: Vec<ValueId> = (0..4).map(|i| rd(b, &ar, &aoff, &self.a_rd[i][h])).collect();
            let bf: Vec<ValueId> = (0..2).map(|j| rd(b, &br, &boff, &self.b_rd[j][h])).collect();
            for i in 0..4usize {
                for j in 0..2usize {
                    // swapped MFMA: hardware A = B fragment (rows n), hardware B = A fragment (rows m); scale_a = sb, scale_b = sa
                    b.var_mfma_scale(self.acc[2 * i + j], bf[j], af[i], sb, sa[i]);
                }
            }
        }
    }
}

/// Evaluates a symbolic offset for concrete symbol values (build-time proofs).
fn ev_ix(e: &Ix, env: &[i64]) -> i64 {
    e.eval(&|s| env[s as usize])
}

pub fn gemm_mxfp8_bs_v1() -> Function {
    use crate::tile::{ev, Cadence, LdsPlan, Pipeline, Role, Roles};
    let mut fb = FnBuilder::new("gemm_mxfp8_bs_v1", 256);
    let b = &mut fb;
    declare_abi(b);
    let ptr_a = b.let_(Op::KernargPtr { offset: 0 });
    let ptr_b = b.let_(Op::KernargPtr { offset: 8 });
    let ptr_c = b.let_(Op::KernargPtr { offset: 16 });
    let ptr_sfa = b.let_(Op::KernargPtr { offset: 64 });
    let ptr_sfb = b.let_(Op::KernargPtr { offset: 72 });
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let n_v = b.let_(Op::KernargI32 { offset: 28 });
    let k_v = b.let_(Op::KernargI32 { offset: 32 });
    let kb_v = b.let_(Op::KernargI32 { offset: 84 });
    let tid = b.let_(Op::WorkitemId);
    let lane = crate::tile::Lane::from_tid(b, tid);
    let bx = b.let_(Op::WorkgroupIdX); // N tile (128)
    let by = b.let_(Op::WorkgroupIdY); // M tile (256)
    let wm = b.bind_i32(lane.wave() & 1);
    let wn = b.bind_i32(lane.wave().lshr(1));

    // ---- the GPU values the layouts are written in ----
    let mut d = Dyns::new();
    let s_lane = d.sym("lane", 0, 63); // 0
    let s_tid = d.sym("tid", 0, 255); // 1
    let s_k = d.sym_aligned("K", 128, 1 << 20, 128); // 2
    let s_n = d.sym_aligned("N", 128, 1 << 20, 128); // 3
    let s_wm = d.sym("wm", 0, 1); // 4
    let s_wn = d.sym("wn", 0, 1); // 5
    let vals = [lane.lane, tid, k_v, n_v, wm, wn];
    let symf = |s: u32| -> I32 { ev(vals[s as usize]) };
    let proof_env = |lane: i64, tid: i64, wm: i64, wn: i64| [lane, tid, 8192, 4096, wm, wn];

    // ---- tensor layouts over the CTA tile (byte offsets relative to the tile origin, K block 0) ----
    let a_glob = make_layout(tup![CTA_M, 128], tup![s_k.clone(), 1]); // A row-major
    let b_glob = make_layout(tup![tup![16, CTA_N / 16], tup![16, 8]], tup![tup![16, s_k.mul_c(16)], tup![1, 256]]); // B preshuffled
    let a_sm = smem_tile(CTA_M);
    let b_sm = smem_tile(CTA_N);

    // ---- DMA: LDS byte = 1024 (issues (tid / 64) + q) + 16 (tid % 64) -> global offset = glob o right_inverse(smem) ----
    let dma = |rows: i64, glob: &Layout, sm: &Layout| -> Vec<(I32, i64)> {
        let issues = rows * 128 / 1024 / 4;
        let tv = make_layout(tup![tup![64, 4], tup![16, issues]], tup![tup![16, 1024 * issues], tup![1, 1024]]);
        let g = composition(glob, &right_inverse(sm));
        let gofs = IssuePlan::new(tv, 16).offsets(&g, &s_tid, &Ix::C(0), &d);
        // build-time proof: the source of every LDS byte this thread writes is the element the LDS layout puts there
        for th in [0i64, 1, 15, 16, 63, 64, 100, 255] {
            for q in 0..issues {
                let lds = 1024 * (issues * (th / 64) + q) + 16 * (th % 64);
                let (r, c) = (0..rows).flat_map(|r| (0..8).map(move |c| (r, 16 * c))).find(|&(r, c)| sm.at(r + rows * c) == lds).expect("LDS byte covered");
                let want = ev_ix(&glob.crd2idx(&tup![r, c], &d), &proof_env(0, th, 0, 0));
                assert_eq!(ev_ix(&gofs[q as usize], &proof_env(0, th, 0, 0)), want, "DMA rows {rows} thread {th} issue {q}");
            }
        }
        // M0 part: the wave's 1 KB block (the lane adds 16 lane itself)
        gofs.iter().enumerate().map(|(q, g)| (lower(g, &symf), 1024 * q as i64)).collect()
    };
    let a_dma = dma(CTA_M, &a_glob, &a_sm);
    let b_dma = dma(CTA_N, &b_glob, &b_sm);
    let issues_a = CTA_M * 128 / 4096;
    let issues_b = CTA_N * 128 / 4096;
    // wave part of M0 (1024 issues wave)
    let a_wave = b.bind_i32(lane.wave() * (1024 * issues_a) as i32);
    let b_wave = b.bind_i32(lane.wave() * (1024 * issues_b) as i32);
    let a_dma: Vec<(I32, i64)> = a_dma.into_iter().map(|(g, l)| (g, l)).collect();
    let b_dma: Vec<(I32, i64)> = b_dma.into_iter().map(|(g, l)| (g, l)).collect();
    // CTA origins: A rows by*256 -> by*256 K; B rows bx*128 (16-row groups) -> bx*128 K
    let a_org = b.bind_i32((ev(by) << 8) * ev(k_v));
    let b_org = b.bind_i32((ev(bx) << 7) * ev(k_v));

    // ---- fragment reads: smem o MFMA operand TV, origin = (wave rows + 32 subtile, 64 h); bases + ds immediates ----
    let rd = |sm: &Layout, rows0: Ix, nsub: i64, b: &mut FnBuilder| -> Vec<Vec<[(ValueId, i64); 2]>> {
        let plan = IssuePlan::new(mfma_op_32x32x64(), 16);
        let mut exprs = Vec::new();
        for i in 0..nsub {
            for h in 0..2i64 {
                let org = sm.crd2idx(&tup![rows0.add_c(32 * i), 64 * h], &d);
                let sub = make_layout(tup![tup![16, 2], tup![16, 4]], tup![tup![16, 2048], tup![1, 256]]);
                exprs.extend(plan.offsets(&sub, &s_lane, &org, &d));
            }
        }
        let op = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::DS, &d);
        let bases: Vec<ValueId> = op.bases.iter().map(|e| b.bind_i32(lower(e, &symf))).collect();
        (0..nsub as usize)
            .map(|i| (0..2usize).map(|h| { let n = 4 * i + 2 * h; [(bases[op.issues[n].0], op.issues[n].1), (bases[op.issues[n + 1].0], op.issues[n + 1].1)] }).collect())
            .collect()
    };
    let a_rd = rd(&a_sm, s_wm.mul_c(128), 4, b);
    let b_rd = rd(&b_sm, s_wn.mul_c(64), 2, b);
    // scales: sa byte (kb = 0) of row m_cta + 128 wm + 32 i + lane % 32; sb byte (kb = 0) of column block bx (CTA_N = 128 = one block)
    let sa_byte: Vec<ValueId> = (0..4i64).map(|i| {
        let r = s_wm.mul_c(128).add_c(32 * i).add(&s_lane.mod_c(32, &d));
        b.bind_i32((ev(by) << 8) + lower(&r, &symf))
    }).collect();
    let sb_byte = b.bind_i32(ev(bx) * ev(kb_v));

    // ---- LDS ring (3 slots: A 32 KB + B 16 KB each) and the pipeline ----
    let mut lds = LdsPlan::new();
    let a_s = lds.ring("A", CTA_M * 128, 3, crate::tile::Layout::Linear);
    let b_s = lds.ring("B", CTA_N * 128, 3, crate::tile::Layout::Linear);
    let sc_s = lds.ring("SC", 4 * 512, 3, crate::tile::Layout::Linear);
    lds.commit(b);
    let sc_wave = b.bind_i32(lane.wave() << 9);
    // A-scale source: m_cta + 4 lane (bytes); the lane reads back row 128 wm + 32 i + lane % 32
    let sa_src = b.bind_i32((ev(by) << 8) + (lane.lane() << 2));
    let sa_rd: Vec<ValueId> = (0..4i32).map(|i| b.bind_i32((ev(wm) << 7) + 32 * i + (lane.lane() & 31))).collect();
    // B-scale byte (bx KB + kb): dword (bx KB + kb) & ~3, byte (bx KB + kb) & 3 = ((bx KB) & 3 + kb & 3) & 3 -> shift computed per tile
    let sb_src = b.bind_i32(ev(bx) * ev(kb_v));
    let sb_shift = b.c_i32(0);
    let nkv = kb_v;
    let roles = Roles::single(&lane, 4);
    let _ = roles;
    let pipe = Pipeline::new(Cadence::OneWindow, nkv, k_v, 128, crate::Ext::ka(84));
    let zero16 = b.frag_zero(DType::Frag16);
    let a_dma = a_dma.into_iter().map(|(g, l)| (g, l)).collect::<Vec<_>>();
    let mut body = GemmV1 {
        a_s, b_s,
        a_dma: a_dma.into_iter().map(|(g, l)| (g, l)).collect(),
        b_dma,
        a_org, b_org, ptr_a, ptr_b, a_rd, b_rd, sa_byte, sb_byte, ptr_sfa, ptr_sfb, m_v, zero16, acc: Vec::new(),
        a_wave, b_wave, sc_s, sc_wave, sa_src, sb_src, kb_v, sa_rd, sb_shift,
    };
    pipe.run(b, Role::All, &mut body);

    // ---- epilogue: bf16 8-byte stores (swapped C TV layout x C row-major) ----
    let c_tile = make_layout(tup![32, 32], tup![2, s_n.mul_c(2)]);
    let c_plan = IssuePlan::new(mfma_c_32x32_swapped(), 4);
    let c_org = b.bind_i32((((ev(by) << 8) * ev(n_v)) + (ev(bx) << 7)) << 1);
    let cb = b.ptr_add(ptr_c, c_org);
    for i in 0..4i64 {
        for j in 0..2i64 {
            let org = s_wm.mul_c(128).add_c(32 * i).mul(&s_n).mul_c(2).add(&s_wn.mul_c(64).add_c(32 * j).mul_c(2));
            let exprs = c_plan.offsets(&c_tile, &s_lane, &org, &d);
            let plan = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::GLOBAL, &d);
            let bases: Vec<ValueId> = plan.bases.iter().map(|e| { let o = b.bind_i32(lower(e, &symf)); b.ptr_add(cb, o) }).collect();
            let acc = body.acc[(2 * i + j) as usize];
            for g in 0..4usize {
                let v: Vec<ValueId> = (0..4).map(|q| { let l = b.frag16_lane(Opnd::Var(acc), (4 * g + q) as u8); b.bitcast_f32(l) }).collect();
                let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
                let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
                let f2 = b.frag2(w0, w1);
                let (bi, imm) = plan.issues[g];
                b.store_f32x2(bases[bi], imm as u32, f2);
            }
        }
    }
    let _ = (m_v, s_wn);
    fb.finish()
}
/// The 96-byte ABI (opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950), declared in offset order.
fn declare_abi(b: &mut FnBuilder) {
    let ptrs = [("ptr_a", 0u32), ("ptr_b", 8), ("ptr_c", 16), ("ptr_sfa", 64), ("ptr_sfb", 72)];
    let ints = [("m", 24u32), ("n", 28), ("k", 32), ("batch", 36), ("stride_a", 40), ("stride_b", 44), ("stride_c", 48), ("stride_a_batch", 52), ("stride_b_batch", 56), ("stride_c_batch", 60), ("stride_sfa", 80), ("stride_sfb", 84), ("stride_sfa_batch", 88), ("stride_sfb_batch", 92)];
    let mut all: Vec<(&str, u32, bool)> = ptrs.iter().map(|&(n, o)| (n, o, true)).chain(ints.iter().map(|&(n, o)| (n, o, false))).collect();
    all.sort_by_key(|x| x.1);
    for (name, off, is_ptr) in all {
        if is_ptr {
            b.arg(name, DType::Ptr, off, 8)
        } else {
            b.arg(name, DType::I32, off, 4)
        }
    }
}