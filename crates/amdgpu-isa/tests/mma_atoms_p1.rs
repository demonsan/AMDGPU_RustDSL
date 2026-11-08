//! The P1-d MFMA atoms (docs/op-coverage-tileops.md) reproduce the layout `probe_mfma_p1` measured on gfx950 (tools/tileop_probe.sh):
//! A / B lane l holds row (column) l % M and k = e (l / M) + v; C as the 16x16 / 32x32 atoms.
use amdgpu_isa::MmaAtom;

#[test]
fn p1_mfma_atoms_match_the_probed_layout() {
    for (atom, per) in [
        (MmaAtom::mfma_bf16_16x16x32(), 8),
        (MmaAtom::mfma_bf16_32x32x16(), 8),
        (MmaAtom::mfma_f16_16x16x32(), 8),
        (MmaAtom::mfma_f16_32x32x16(), 8),
        (MmaAtom::mfma_i8_16x16x64(), 16),
        (MmaAtom::mfma_i8_32x32x32(), 16),
    ] {
        let m = atom.m;
        let mut seen = vec![false; (m * atom.k) as usize];
        for lane in 0..64i64 {
            for v in 0..per as i64 {
                let want = lane % m + m * (per as i64 * (lane / m) + v);
                assert_eq!(atom.a_tv.at(lane + 64 * v), want, "{} A lane {lane} value {v}", atom.name);
                assert_eq!(atom.b_tv.at(lane + 64 * v), want, "{} B", atom.name);
                seen[want as usize] = true;
            }
            for r in 0..atom.c_dwords as i64 {
                let (row, col) = if m == 16 { (4 * (lane / 16) + r, lane % 16) } else { (8 * (r / 4) + 4 * (lane / 32) + r % 4, lane % 32) };
                assert_eq!(atom.c_tv.at(lane + 64 * r), row + m * col, "{} C", atom.name);
            }
        }
        assert!(seen.iter().all(|s| *s), "{}: A covers the tile", atom.name);
        assert_eq!(atom.a_dwords * 32 / atom.ab_bits, per as u32);
    }
}
