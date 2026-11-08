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

#[test]
fn fp4_atoms_are_contiguous_in_k_and_the_fp8_atom_is_split() {
    for (atom, fp8) in [(MmaAtom::mfma_scale_16x16x128_f4(), MmaAtom::mfma_scale_16x16x128_f8()), (MmaAtom::mfma_scale_32x32x64_f4(), MmaAtom::mfma_scale_32x32x64_f8())] {
        let m = atom.m;
        let g = 64 / m;
        for lane in 0..64i64 {
            for v in 0..32i64 {
                assert_eq!(atom.a_tv.at(lane + 64 * v), lane % m + m * (32 * (lane / m) + v), "{}", atom.name);
                // the measured 8-bit order: k = 16 (l / M) + v % 16 + 16 G (v / 16)
                assert_eq!(fp8.a_tv.at(lane + 64 * v), lane % m + m * (16 * (lane / m) + v % 16 + 16 * g * (v / 16)), "{}", fp8.name);
            }
        }
    }
}

/// The measured `ds_read_b64_tr_b16` map (probe_tr_b16_p1 printout: lane d gets element d % 4 of lanes 16 (d / 16) + (d % 16) / 4 + 4 e).
#[test]
fn tr_b16_formula_and_tv_agree() {
    let tv = amdgpu_isa::tr_b16::dest_tv();
    let mut seen = vec![false; 256];
    for d in 0..64i64 {
        for e in 0..4i64 {
            let (s, h) = amdgpu_isa::tr_b16::source(d, e);
            assert_eq!(tv.at(d + 64 * e), 4 * s + h, "lane {d} element {e}");
            seen[(4 * s + h) as usize] = true;
        }
    }
    assert!(seen.iter().all(|x| *x), "a permutation of the 256 source elements");
    // spot values straight from the probe printout (dest lane: source lanes, element)
    assert_eq!((0..4).map(|e| amdgpu_isa::tr_b16::source(5, e).0).collect::<Vec<_>>(), vec![1, 5, 9, 13]);
    assert_eq!(amdgpu_isa::tr_b16::source(5, 0).1, 1);
    assert_eq!((0..4).map(|e| amdgpu_isa::tr_b16::source(46, e).0).collect::<Vec<_>>(), vec![35, 39, 43, 47]);
}
