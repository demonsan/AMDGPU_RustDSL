//! The MXFP4 (block-scaled f4) variant of the TDM-producer GEMM stage template (`gemm::tdm`): A[M][K] x B[N][K]^T with fp4 e2m1 operands (two per byte, low nibble first) and E8M0 scales per 32 K
//! (one byte per row and K block), `v_wmma_scale_f32_32x16x128_f4`.
//!
//! The ring slot of a K tile of 128 elements holds, per workgroup: A (BM rows x 64 bytes), B (BN rows x 64 bytes), the A scales (BM rows x 4 bytes) and the B scales (BN x 4): four tensor loads
//! per stage (the scale tiles are [rows][1 dword] tiles of the [rows][K / 32 bytes] scale arrays). The window reads, per wave, the A fragments (32 rows x 128: two `ds_load_b128` per 16-row half),
//! the B fragments (16 x 128: two `ds_load_b128`), the scale words (one `ds_load_b32` per row) and issues `tm x tn` scaled WMMAs into loop-carried accumulators.
//!
//! Lane layouts (docs/target-gfx1250.md section 3, checked on hardware by `probe_g1250_wmma_f4`): a lane `l` (h = l / 16, m = l % 16) holds, for matrix row m, the dwords
//! `[K 32 h .. +32)` and `[K 64 + 32 h .. +32)` of the row, i.e. bytes `16 h .. +16` and `32 + 16 h .. +16` of the 64-byte row; the A scale of lane `l` is row `l` of the 32-row block (4 bytes: one per
//! 32 K), the B scale (lanes 0..15) row `l`; C / D dword v: row `v % 8 + 8 h + 16 (v / 8)`, column `m`.

use amdgpu_dsl::{ev, DType, FnBuilder, Op, Opnd, ValueId, VarId, WmmaKind, WmmaScale};
use amdgpu_tile::lane::{Role, TileIt};
use amdgpu_tile::shared::{Shared, SlotIdx};

use super::tdm::tdm_group0;
use crate::PipeBody;

/// The shape of the workgroup: `wm x wn` waves, each wave `tm x tn` WMMAs (32 x 16 blocks) per K tile of 128.
#[derive(Clone, Copy, Debug)]
pub struct F4Shape {
    pub wm: i32,
    pub wn: i32,
    pub tm: i32,
    pub tn: i32,
    /// K sub-steps of 128 per ring slot (a slot holds `ks x 128` K elements: A / B rows of `64 ks` bytes, scale rows of `4 ks` bytes)
    pub ks: i32,
    /// relaxed tensor waits: in the steady state each window completes the tile staged in the PREVIOUS window (`wait_tensor(4)`, the four loads of the stage just issued stay in flight), so a
    /// stage has two windows of latency instead of one; the prologue still drains
    pub relax: bool,
    /// the cluster of workgroups (cx x cy, workgroup index x fastest) that share operand tiles: the A tile (and its scales) is multicast to the workgroups of the same cluster row, the B tile to those of the same
    /// column. A multicast is a rendezvous of identical requests, so every workgroup of the row issues the very same slices (per wave) with the same mask; 1 x 1 = no cluster
    pub cx: i32,
    pub cy: i32,
    /// the output is bf16 (round to nearest even, row-major M x N x 2 bytes) instead of f32: half the C bytes (the vendor writes bf16)
    pub bf16: bool,
    /// cluster only: keep the workgroups of the cluster in lockstep, one cluster barrier per K tile (the vendor loop's `s_barrier_signal -3` / wait): the multicast rendezvous then finds its peers
    /// asking at about the same time instead of waiting for the timeout
    pub lock: bool,
}

impl F4Shape {
    pub fn bm(&self) -> i32 {
        self.wm * self.tm * 32
    }
    pub fn bn(&self) -> i32 {
        self.wn * self.tn * 16
    }
    pub fn waves(&self) -> i32 {
        self.wm * self.wn
    }
}

/// The overlapped epilogue of a persistent swap kernel (bf16 output): the finished tile's accumulators are packed into `prev` (loop variables of the tile loop, 4 dwords per 16-byte store) and
/// stored from registers by the NEXT tile's K loop, 4 stores per iteration (`window`), the remainder after that loop; `mr` / `n0` are the previous tile's lane row and column base, `hp` is
/// 1 once a previous tile exists.
#[derive(Clone)]
pub struct Ovl {
    /// the previous tile's row pointers, one per block column j: `C + ((mr + 16 j) N + n0) * 2` (the lane's row `mr` and column base `n0` of that tile), so a store is one pointer plus an
    /// immediate offset and no address arithmetic runs in the K loop
    pub ptrs: Vec<VarId>,
    pub hp: VarId,
    pub prev: Vec<VarId>,
    /// the blocks (of `tm * tn`, i-major) whose packed accumulators are carried into the next tile's loop: group g stores blocks 2 g and 2 g + 1, so `groups = kept / 2`; the other
    /// blocks are stored straight at the end of the tile (the VGPR budget holds only a part of a second accumulator set)
    pub groups: i32,
    /// all kept groups in one `if` at iteration 0 (instead of one `if` per iteration)
    pub burst: bool,
}

impl Ovl {
    /// The four 16-byte stores of group `g` of the previous tile's packed accumulators (`tm` x `tn` blocks, block index i * tn + j).
    pub fn store_group(&self, b: &mut FnBuilder, pc: ValueId, tm: i32, g: i32) {
        for q in 0..2 {
            let blk = 2 * g + q;
            // j-major blocks: a group covers 128 contiguous bytes of each of 16 rows (two i, two h) per 4 stores
            let (i, j) = (blk % tm, blk / tm);
            for h in 0..2i32 {
                let p = b.let_(Op::PtrAdd { base: Opnd::Val(pc), byte_off: Opnd::Var(self.ptrs[j as usize]) });
                b.store_f32x4(p, (i * 64 + h * 32) as u32, Opnd::Var(self.prev[(blk * 2 + h) as usize]));
            }
        }
    }
}

pub struct TdmF4Body {
    pub sh: F4Shape,
    /// the rings: A / B tiles (bytes) and the scale tiles
    pub a: Shared,
    pub b: Shared,
    pub sa: Shared,
    pub sb: Shared,
    /// kernarg pointers: A, B, A scales, B scales
    pub pa: ValueId,
    pub pb: ValueId,
    pub psa: ValueId,
    pub psb: ValueId,
    /// byte offsets of this workgroup's first row of A / B (row0 * K / 2) and of the scale arrays (row0 * K / 32)
    pub a_org: ValueId,
    pub b_org: ValueId,
    pub sa_org: ValueId,
    pub sb_org: ValueId,
    /// group 1 of the four tensors
    pub g1a: ValueId,
    pub g1b: ValueId,
    pub g1sa: ValueId,
    pub g1sb: ValueId,
    /// per-lane LDS offsets inside a slot's tile: A / B fragment (row (wave's first row + m) * 64 + 16 h) per 16-row group, and the scale rows
    pub abase: ValueId,
    pub bbase: ValueId,
    pub sabase: ValueId,
    pub sbbase: ValueId,
    /// this wave's byte offsets (LDS) of its slice of the A / B / A-scale / B-scale tile when the waves load cooperatively (each wave `rows / waves` rows; the global origins already carry the slice);
    /// zero when every wave loads the whole tile
    pub wl: [ValueId; 4],
    /// the loop-carried accumulators, tm x tn (bound by `bind_state`)
    pub acc: Vec<VarId>,
    /// set once the loop is being emitted (`bind_state`): the prologue's stages complete fully, the steady-state ones as `sh.relax` says
    pub steady: bool,
    /// the condition "this is wave 0 of the workgroup" (wave-uniform): the one wave per workgroup that signals the cluster barrier
    pub leader: ValueId,
    /// the pipeline's ring depth (set by the driver): `slots - 2` stages stay in flight at the end of a relaxed window
    pub depth: u8,
    /// LDS padding bytes after every A / B row of a slot (the TDM's pad: `row_pad / 4` dwords after each `64 ks`-byte row), 0 = none; the
    /// A / B row stride in LDS is `64 ks + row_pad`
    pub row_pad: i32,
    /// wave roles: role A (waves 0, 1) issues only the A and SA loads, role B (waves 2, 3) only B and SB (2 tensor loads per wave per stage instead of 4: the per-operation cost of the tensor
    /// unit, not the bytes, is what the many tiny scale loads cost); `Role::All` issues all four
    pub roles: bool,
    /// the scales in the vendor's preshuffled layout (aiter `shuffle_scale_f4`: [rows / 32][K / 128][32 rows][4 bytes]): a stage's scale
    /// slice is rows / 32 TDM rows of 128 ks contiguous bytes (instead of rows of 4 ks bytes); in LDS [row block][k tile][32][4]
    pub scale_shuf: bool,
    /// timing stamps of the K loop (the shader-cycle counter at four points of every steady iteration, summed per phase into loop variables; the kernel stores the sums): stage issue,
    /// window (fragment reads + WMMA issue), the tensor / LDS wait, the barrier and loop overhead
    pub stamps: bool,
    /// the overlapped epilogue (see `Ovl`)
    pub ovl: Option<Ovl>,
    /// the C pointer (kernarg) for the overlapped stores
    pub pc_c: ValueId,
    /// row permutation of the A operand (the swap mode's epilogue): the 32 operand rows of a block are the B-matrix rows with bits 3 and 4 of the row index exchanged, so that a lane's
    /// accumulator holds 16 CONTIGUOUS columns of C (v = 0 .. 7: n .. n + 7, v = 8 .. 15: n + 8 .. n + 15, the lane group g adding 16 g)
    pub perm: bool,
    /// the phase sums (stage issue, window, wait, barrier) and the last wait-done stamp: loop variables, bound with the accumulators
    pub stamp_vars: Vec<VarId>,
    /// the stamps of the iteration being emitted: stage end, window end
    pub st_t1: Option<ValueId>,
    pub st_t2: Option<ValueId>,
}

fn cyc(b: &mut FnBuilder) -> ValueId {
    let c = b.shader_cycles();
    b.let_(Op::PtrLo(Opnd::Val(c)))
}

fn add_to(b: &mut FnBuilder, var: VarId, delta: ValueId) {
    let n = b.bind_i32(amdgpu_dsl::varx(var) + ev(delta));
    b.set_var(var, Opnd::Val(n));
}

impl PipeBody for TdmF4Body {
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &amdgpu_dsl::I32, slot: &SlotIdx) {
        let stamp_t0 = if self.stamps && self.steady { Some(cyc(b)) } else { None };
        if let Some(t0) = stamp_t0 {
            // the barrier and loop overhead of the previous iteration: from its wait-done stamp to this stage's start
            let d = b.bind_i32(ev(t0) - amdgpu_dsl::varx(self.stamp_vars[4]));
            add_to(b, self.stamp_vars[3], d);
        }
        let (do_a, do_b) = match (self.roles, role) {
            (true, Role::A) => (true, false),
            (true, Role::B) => (false, true),
            _ => (true, true),
        };
        // `key0`: the K element offset of the tile (tile_rows = 128: tile << 7); bytes of an fp4 row: / 2; of a scale row: / 32
        let kb = b.bind_i32(key0.clone().lshr(1));
        // the scales' K byte offset: K / 32 bytes per row, or (preshuffled) K / 128 tiles of 128 bytes = K bytes
        let ks = if self.scale_shuf { b.bind_i32(key0.clone()) } else { b.bind_i32(key0.clone().lshr(5)) };
        let oa = b.add_i32(self.a_org, kb);
        let ob = b.add_i32(self.b_org, kb);
        let osa = b.add_i32(self.sa_org, ks);
        let osb = b.add_i32(self.sb_org, ks);
        let (ra, rb, rsa, rsb) = (self.a.at(slot), self.b.at(slot), self.sa.at(slot), self.sb.at(slot));
        let lds = |b: &mut FnBuilder, r: &amdgpu_tile::shared::SlotRef<'_>| Opnd::Val(b.bind_i32(r.base()));
        let (la, lb, lsa, lsb) = (lds(b, &ra), lds(b, &rb), lds(b, &rsa), lds(b, &rsb));
        let wl = self.wl;
        let (la, lb, lsa, lsb) = (Opnd::Val(b.add_i32(la, wl[0])), Opnd::Val(b.add_i32(lb, wl[1])), Opnd::Val(b.add_i32(lsa, wl[2])), Opnd::Val(b.add_i32(lsb, wl[3])));
        let g0a = tdm_group0(b, self.pa, oa, la);
        let g0b = tdm_group0(b, self.pb, ob, lb);
        let g0sa = tdm_group0(b, self.psa, osa, lsa);
        let g0sb = tdm_group0(b, self.psb, osb, lsb);
        let cp = amdgpu_dsl::TdmCpol::default();
        let (g1a, g1b, g1sa, g1sb) = (self.g1a, self.g1b, self.g1sa, self.g1sb);
        if do_a {
            ra.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0a), Opnd::Val(g1a), None, cp));
        }
        if do_b {
            rb.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0b), Opnd::Val(g1b), None, cp));
        }
        if do_a {
            rsa.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0sa), Opnd::Val(g1sa), None, cp));
        }
        if do_b {
            rsb.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0sb), Opnd::Val(g1sb), None, cp));
        }
        if let Some(t0) = stamp_t0 {
            let t1 = cyc(b);
            let d = b.bind_i32(ev(t1) - ev(t0));
            add_to(b, self.stamp_vars[0], d);
            self.st_t1 = Some(t1);
        }
    }

    fn stage_complete_keep(&mut self, b: &mut FnBuilder, _role: Role, _slot: &SlotIdx, keep: u8) {
        // tile 0 done, the `keep` later stages (this wave's tensor loads per stage each) still in flight
        let per_stage = if self.roles { 2 } else { 4 };
        b.wait_tensor(per_stage * keep);
    }

    fn set_ring_depth(&mut self, slots: u8) {
        self.depth = slots;
    }

    fn finish(&mut self, b: &mut FnBuilder, _role: Role) {
        b.wait_tensor(0);   // the last steady iteration's clamped stage is still in flight
    }

    fn stage_complete(&mut self, b: &mut FnBuilder, _role: Role, _slot: &SlotIdx) {
        if self.sh.cx * self.sh.cy > 1 && self.sh.lock {
            // our own tensor wait, every wave of the workgroup done (reads of the previous tile included: the barrier), then one wave signals the cluster barrier and all wait
            b.wait_tensor(0);
            b.wait_lds(0);
            b.barrier();
            b.begin_if(Opnd::Val(self.leader));
            b.barrier_signal(-3);
            b.end_if();
            b.barrier_wait(-3);
            return;
        }
        // the four loads of a stage (A, B and the two scale tiles) are the only operations a stage leaves outstanding: with `relax` the loads of the stage just issued may stay in flight
        // `depth - 2` stages (the next tiles' 4 loads each) stay in flight; the wait completes the one the next window reads
        let per_stage = if self.roles { 2 } else { 4 };   // tensor loads this wave issues per stage
        b.wait_tensor(if self.sh.relax && self.steady { per_stage * (self.depth - 2) } else { 0 });
        if self.stamps && self.steady {
            if let Some(t2) = self.st_t2.take() {
                b.wait_lds(0);
                let t3 = cyc(b);
                let d = b.bind_i32(ev(t3) - ev(t2));
                add_to(b, self.stamp_vars[2], d);
                b.set_var(self.stamp_vars[4], Opnd::Val(t3));
            }
        }
    }

    fn prologue_compute(&mut self, _b: &mut FnBuilder, _role: Role, _t0: &TileIt) {}

    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let n = (self.sh.tm * self.sh.tn) as usize;
        (0..n)
            .map(|_| {
                // a zero fragment value (the lowering's `zeroinit` writes the variable's own registers: no zero tuple kept live)
                let z = b.let_(Op::FragZero { ty: DType::Frag16 });
                (WmmaKind::ScaleF4_32x16x128.acc_ty(), Opnd::Val(z))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .chain(if self.stamps {
                let z = b.c_i32(0);
                let c0 = cyc(b);
                vec![(DType::I32, Opnd::Val(z)), (DType::I32, Opnd::Val(z)), (DType::I32, Opnd::Val(z)), (DType::I32, Opnd::Val(z)), (DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(c0))]
            } else {
                Vec::new()
            })
            .collect()
    }

    fn bind_state(&mut self, vars: &[VarId]) {
        let n = (self.sh.tm * self.sh.tn) as usize;
        self.acc = vars[..n].to_vec();
        self.stamp_vars = vars[n..].to_vec();
        self.steady = true;
    }

    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let sh = self.sh;
        let (ra, rb, rsa, rsb) = (self.a.at(&t.cur), self.b.at(&t.cur), self.sa.at(&t.cur), self.sb.at(&t.cur));
        let (abase, bbase, sabase, sbbase) = (self.abase, self.bbase, self.sabase, self.sbbase);
        let (arow, brow) = (64 * sh.ks + self.row_pad, 64 * sh.ks + self.row_pad);   // bytes per A / B row of the slot (padding included)
        let (srow_bytes, _) = (4 * sh.ks, 0);          // bytes per scale row of the slot
        // the fragments of sub-step `s`: A (32 rows x 128): two ds_load_b128 per 16-row half, B (16 x 128): two; the scale words
        struct Frags {
            fa: Vec<Opnd>,
            sa: Vec<Opnd>,
            fb: Vec<Opnd>,
            sb: Vec<Opnd>,
        }
        let load = |b: &mut FnBuilder, s: i32| -> Frags {
            let mut f = Frags { fa: Vec::new(), sa: Vec::new(), fb: Vec::new(), sb: Vec::new() };
            ra.read(b, |b| {
                for i in 0..sh.tm {
                    let mut parts: Vec<Opnd> = Vec::new();
                    for half in 0..2 {
                        for piece in 0..2 {
                            let addr = b.bind_i32(ra.base() + ev(abase) + (i * 32 + half * if self.perm { 8 } else { 16 }) * arow + s * 64 + piece * 32);
                            parts.push(Opnd::Val(b.load_lds_b128(addr)));
                        }
                    }
                    f.fa.push(Opnd::Val(b.frag_cat(&parts)));
                }
            });
            rsa.read(b, |b| {
                for i in 0..sh.tm {
                    let st = if self.scale_shuf { 128 * s } else { 4 * s };
                    let addr = b.bind_i32(rsa.base() + ev(sabase) + i * 32 * srow_bytes + st);
                    f.sa.push(Opnd::Val(b.load_lds_b32(addr)));
                }
            });
            rb.read(b, |b| {
                for j in 0..sh.tn {
                    let mut parts: Vec<Opnd> = Vec::new();
                    for piece in 0..2 {
                        let addr = b.bind_i32(rb.base() + ev(bbase) + j * 16 * brow + s * 64 + piece * 32);
                        parts.push(Opnd::Val(b.load_lds_b128(addr)));
                    }
                    f.fb.push(Opnd::Val(b.frag_cat(&parts)));
                }
            });
            rsb.read(b, |b| {
                for j in 0..sh.tn {
                    // preshuffled: B block j covers rows 16 j .. of the 32-row block j / 2 (lanes m at 16 (j % 2) + m)
                    let (jt, st) = if self.scale_shuf { ((j / 2) * 32 * srow_bytes + (j % 2) * 64, 128 * s) } else { (j * 16 * srow_bytes, 4 * s) };
                    let addr = b.bind_i32(rsb.base() + ev(sbbase) + jt + st);
                    f.sb.push(Opnd::Val(b.load_lds_b32(addr)));
                }
            });
            f
        };
        // software pipeline inside the window: the loads of sub-step s + 1 are issued before the WMMAs of s
        let mut cur = load(b, 0);
        for s in 0..sh.ks {
            let next = if s + 1 < sh.ks { Some(load(b, s + 1)) } else { None };
            for i in 0..sh.tm {
                for j in 0..sh.tn {
                    let acc = self.acc[(i * sh.tn + j) as usize];
                    let d = b.wmma_scaled(WmmaKind::ScaleF4_32x16x128, cur.fa[i as usize], cur.fb[j as usize], Some(Opnd::Var(acc)), WmmaScale { sa: cur.sa[i as usize], sb: cur.sb[j as usize], b_row1: false });
                    b.set_var(acc, Opnd::Val(d));
                }
            }
            if let Some(n) = next {
                cur = n;
            }
        }
        if let (Some(ovl), true) = (self.ovl.clone(), t.staged) {
            // the previous tile's packed accumulators go out 4 stores per iteration: group g in iteration g of this K loop (only when a previous tile exists)
            let sel = b.bind_i32(amdgpu_dsl::varx(ovl.hp).eq(1).select_i32(amdgpu_dsl::ox(t.it), 1000));
            if ovl.burst {
                let c = b.bind_cond(ev(sel).eq(0));
                b.begin_if(c);
                for g in 0..ovl.groups {
                    ovl.store_group(b, self.pc_c, sh.tm, g);
                }
                b.end_if();
            } else {
                for g in 0..ovl.groups {
                    let c = b.bind_cond(ev(sel).eq(g));
                    b.begin_if(c);
                    ovl.store_group(b, self.pc_c, sh.tm, g);
                    b.end_if();
                }
            }
        }
        if self.stamps && t.staged {
            let t2 = cyc(b);
            if let Some(t1) = self.st_t1.take() {
                let d = b.bind_i32(ev(t2) - ev(t1));
                add_to(b, self.stamp_vars[1], d);
            }
            self.st_t2 = Some(t2);
        }
    }
}
