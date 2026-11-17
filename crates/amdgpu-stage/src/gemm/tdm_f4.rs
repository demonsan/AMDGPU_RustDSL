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

use amdgpu_dsl::{ev, DType, FnBuilder, Opnd, ValueId, VarId, WmmaKind, WmmaScale};
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
    /// the loop-carried accumulators, tm x tn (bound by `bind_state`)
    pub acc: Vec<VarId>,
}

impl PipeBody for TdmF4Body {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &amdgpu_dsl::I32, slot: &SlotIdx) {
        // `key0`: the K element offset of the tile (tile_rows = 128: tile << 7); bytes of an fp4 row: / 2; of a scale row: / 32
        let kb = b.bind_i32(key0.clone().lshr(1));
        let ks = b.bind_i32(key0.clone().lshr(5));
        let oa = b.add_i32(self.a_org, kb);
        let ob = b.add_i32(self.b_org, kb);
        let osa = b.add_i32(self.sa_org, ks);
        let osb = b.add_i32(self.sb_org, ks);
        let (ra, rb, rsa, rsb) = (self.a.at(slot), self.b.at(slot), self.sa.at(slot), self.sb.at(slot));
        let lds = |b: &mut FnBuilder, r: &amdgpu_tile::shared::SlotRef<'_>| Opnd::Val(b.bind_i32(r.base()));
        let (la, lb, lsa, lsb) = (lds(b, &ra), lds(b, &rb), lds(b, &rsa), lds(b, &rsb));
        let g0a = tdm_group0(b, self.pa, oa, la);
        let g0b = tdm_group0(b, self.pb, ob, lb);
        let g0sa = tdm_group0(b, self.psa, osa, lsa);
        let g0sb = tdm_group0(b, self.psb, osb, lsb);
        let cp = amdgpu_dsl::TdmCpol::default();
        let (g1a, g1b, g1sa, g1sb) = (self.g1a, self.g1b, self.g1sa, self.g1sb);
        ra.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0a), Opnd::Val(g1a), None, cp));
        rb.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0b), Opnd::Val(g1b), None, cp));
        rsa.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0sa), Opnd::Val(g1sa), None, cp));
        rsb.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0sb), Opnd::Val(g1sb), None, cp));
    }

    fn stage_complete(&mut self, b: &mut FnBuilder, _role: Role, _slot: &SlotIdx) {
        b.wait_tensor(0);
    }

    fn prologue_compute(&mut self, _b: &mut FnBuilder, _role: Role, _t0: &TileIt) {}

    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let n = (self.sh.tm * self.sh.tn) as usize;
        (0..n)
            .map(|_| {
                let z = b.frag_cat(&[Opnd::I32(0); 16]);
                (WmmaKind::ScaleF4_32x16x128.acc_ty(), Opnd::Val(z))
            })
            .collect()
    }

    fn bind_state(&mut self, vars: &[VarId]) {
        self.acc = vars.to_vec();
    }

    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let sh = self.sh;
        let (ra, rb, rsa, rsb) = (self.a.at(&t.cur), self.b.at(&t.cur), self.sa.at(&t.cur), self.sb.at(&t.cur));
        let (abase, bbase, sabase, sbbase) = (self.abase, self.bbase, self.sabase, self.sbbase);
        // A: per 32-row block i of the wave: 16-row halves (rows 32 i + 16 half + m), K pieces at +0 and +32 bytes of the 64-byte row
        let mut fa: Vec<Opnd> = Vec::new();
        let mut sa: Vec<Opnd> = Vec::new();
        ra.read(b, |b| {
            for i in 0..sh.tm {
                let mut parts: Vec<Opnd> = Vec::new();
                for half in 0..2 {
                    for piece in 0..2 {
                        let addr = b.bind_i32(ra.base() + ev(abase) + (i * 32 + half * 16) * 64 + piece * 32);
                        parts.push(Opnd::Val(b.load_lds_b128(addr)));
                    }
                }
                fa.push(Opnd::Val(b.frag_cat(&parts)));
            }
        });
        rsa.read(b, |b| {
            for i in 0..sh.tm {
                let addr = b.bind_i32(rsa.base() + ev(sabase) + i * 32 * 4);
                sa.push(Opnd::Val(b.load_lds_b32(addr)));
            }
        });
        let mut fb: Vec<Opnd> = Vec::new();
        let mut sb: Vec<Opnd> = Vec::new();
        rb.read(b, |b| {
            for j in 0..sh.tn {
                let mut parts: Vec<Opnd> = Vec::new();
                for piece in 0..2 {
                    let addr = b.bind_i32(rb.base() + ev(bbase) + j * 16 * 64 + piece * 32);
                    parts.push(Opnd::Val(b.load_lds_b128(addr)));
                }
                fb.push(Opnd::Val(b.frag_cat(&parts)));
            }
        });
        rsb.read(b, |b| {
            for j in 0..sh.tn {
                let addr = b.bind_i32(rsb.base() + ev(sbbase) + j * 16 * 4);
                sb.push(Opnd::Val(b.load_lds_b32(addr)));
            }
        });
        for i in 0..sh.tm {
            for j in 0..sh.tn {
                let acc = self.acc[(i * sh.tn + j) as usize];
                let d = b.wmma_scaled(WmmaKind::ScaleF4_32x16x128, fa[i as usize], fb[j as usize], Some(Opnd::Var(acc)), WmmaScale { sa: sa[i as usize], sb: sb[j as usize], b_row1: false });
                b.set_var(acc, Opnd::Val(d));
            }
        }
    }
}
