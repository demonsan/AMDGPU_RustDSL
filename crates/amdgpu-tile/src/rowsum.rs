//! The FMHA row sums with TWO accumulated `v_smfmac_f32_16x16x128_fp8_fp8` per 128 keys (the vendor's count), derived and proven on the probe-measured
//! smfmac layouts (`amdgpu_isa::MmaAtom::smfmac_16x16x128_fp8`, `amdgpu_isa::smfmac`; docs/isa-probes-smfmac-tr.md).
//!
//! The P fragments are the B operands of the 32x32x64 PV MMA: lane l holds 32 bytes, all of query `l % 32`. Read as the smfmac's dense B operand
//! (lane l = column `n = l % 16`, k group `h = l / 16`), column n carries query n in the k groups h = 0, 2 and query n + 16 in h = 1, 3. The sparse A
//! operand selects, per D row, which (k group, register half, byte pair) it contracts: an A lane of row m in lane group g, pair q multiplies the B dword
//! `(h, r) = (2 (g & 1) + q / 4, 4 (g >> 1) + q % 4)` at the bytes its index word names. A row whose stored values q < 4 are ones contracts exactly the
//! even k groups (= query n), q >= 4 the odd ones (= query n + 16); index words with complementary byte pairs split the dword between two rows.
//!
//! The search below enumerates (which two D dwords per lane, the complementary index pair, the per-row query parity) and keeps the candidates whose
//! simulated D satisfies: lane l's two chosen dwords hold the sum of EVERY P byte of query `l % 32` exactly once (both fragments, one smfmac each,
//! accumulated) and nothing else - so the per-lane rollback rescale of those dwords stays exact. The proof is by enumeration over all 64 lanes x 16 stored
//! values x both fragments.

use amdgpu_isa::smfmac;

/// The plan: rows `4 G + r` with `r` in `dwords` are active; row `4 G + dwords[i]` uses index word `idx[i]`; its ones sit in the stored values of the
/// query parity of lane group G (`G` even: q < 4; odd: q >= 4). Lane l's row sum = D dword `dwords[0]` + `dwords[1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSum2 {
    pub dwords: [u8; 2],
    pub idx: [u32; 2],
}

impl RowSum2 {
    /// The A operand bytes of lane `l` (16 stored values; nonzero = fp8 1.0).
    pub fn a_bytes(&self, l: i64) -> [bool; 16] {
        let m = l % 16;
        let (g_row, r) = (m / 4, (m % 4) as u8);
        let mut out = [false; 16];
        if self.dwords.contains(&r) {
            for (s, o) in out.iter_mut().enumerate() {
                let q = s as i64 / 2;
                *o = (g_row % 2 == 0) == (q < 4);
            }
        }
        out
    }
    /// The SRC2 index word of lane `l`.
    pub fn idx_of(&self, l: i64) -> u32 {
        let r = ((l % 16) % 4) as u8;
        if r == self.dwords[1] {
            self.idx[1]
        } else {
            self.idx[0]
        }
    }
}

/// Simulates one smfmac with B = a P fragment: for D lane `l`, dword `r`, the multiset of P (lane, byte) it sums.
fn contributions(plan: &RowSum2) -> Vec<Vec<Vec<(i64, i64)>>> {
    // D lane l dword r = (row m = 4 (l / 16) + r, column n = l % 16); row m is computed by the A lanes with lane % 16 == m (lane groups g = 0..3)
    let mut d = vec![vec![Vec::new(); 4]; 64];
    for (l, dl) in d.iter_mut().enumerate() {
        let l = l as i64;
        for (r, dr) in dl.iter_mut().enumerate() {
            let m = 4 * (l / 16) + r as i64;
            let n = l % 16;
            for g in 0..4i64 {
                let al = m + 16 * g;
                let ab = plan.a_bytes(al);
                let idx = plan.idx_of(al);
                for s in 0..smfmac::STORED {
                    if !ab[s as usize] {
                        continue;
                    }
                    let k = smfmac::k_of(al, s, idx);
                    // B element (k, n): lane n + 16 h, register r', byte y with k = 32 h + 4 r' + y
                    let (h, rest) = (k / 32, k % 32);
                    dr.push((n + 16 * h, rest)); // (P lane, byte in the lane's 32 bytes)
                }
            }
        }
    }
    d
}

/// Proves a plan: lane l's two dwords = every byte of query `l % 32` (P lanes `l % 32` and `l % 32 + 32`, 32 bytes each) exactly once.
pub fn prove(plan: &RowSum2) -> Result<(), String> {
    let d = contributions(plan);
    for l in 0..64i64 {
        let mut got: Vec<(i64, i64)> = plan.dwords.iter().flat_map(|&r| d[l as usize][r as usize].clone()).collect();
        got.sort();
        let q = l % 32;
        let mut want: Vec<(i64, i64)> = [q, q + 32].iter().flat_map(|&pl| (0..32).map(move |y| (pl, y))).collect();
        want.sort();
        if got != want {
            return Err(format!("lane {l}: {} contributions, {} wanted (first got {:?})", got.len(), want.len(), got.first()));
        }
    }
    Ok(())
}

/// All proven plans, in search order (dword pairs, then complementary index pairs).
pub fn search() -> Vec<RowSum2> {
    let pairs = [(0u32, 1u32, 2u32, 3u32), (0, 2, 1, 3), (0, 3, 1, 2)];
    let mut out = Vec::new();
    for d0 in 0..4u8 {
        for d1 in d0 + 1..4u8 {
            for &(a0, a1, b0, b1) in &pairs {
                for swap in [false, true] {
                    let (i0, i1) = (smfmac::uniform(a0, a1), smfmac::uniform(b0, b1));
                    let plan = RowSum2 { dwords: [d0, d1], idx: if swap { [i1, i0] } else { [i0, i1] } };
                    if prove(&plan).is_ok() {
                        out.push(plan);
                    }
                }
            }
        }
    }
    out
}

/// The plan the kernels use: the first proven one (dwords 0 and 1, the vendor's `0x44444444` first).
pub fn plan() -> RowSum2 {
    let p = RowSum2 { dwords: [0, 1], idx: [smfmac::uniform(0, 1), smfmac::uniform(2, 3)] };
    prove(&p).unwrap_or_else(|e| panic!("rowsum2 plan: {e}"));
    p
}
