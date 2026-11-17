//! The gfx1250 WMMA atoms reproduce the CDNA5 ISA's VGPR tables (section 7.12.2 / 7.12.6), written out here independently, and each operand layout
//! covers its tile exactly once.
use amdgpu_isa::gfx1250::{mma_atoms, scale_32x16x128_f4_a, scale_32x16x128_f4_b};
use amdgpu_isa::MmaAtom;

fn bijective(l: &amdgpu_layout::Layout, n: i64, size: i64, what: &str) {
    let mut seen = vec![0u32; size as usize];
    for i in 0..n {
        let x = l.at(i);
        assert!((0..size).contains(&x), "{what}: index {i} -> {x} outside the tile");
        seen[x as usize] += 1;
    }
    assert!(seen.iter().all(|c| *c == 1), "{what}: not a bijection");
}

#[test]
fn every_atom_covers_its_tiles_once() {
    for a in mma_atoms() {
        let vals = |dw: u32| (dw * 32 / a.ab_bits) as i64;
        bijective(&a.a_tv, 32 * vals(a.a_dwords), a.m * a.k, &format!("{} A", a.name));
        bijective(&a.b_tv, 32 * vals(a.b_dwords), a.n * a.k, &format!("{} B", a.name));
        bijective(&a.c_tv, 32 * a.c_dwords as i64, a.m * a.n, &format!("{} C", a.name));
    }
}

#[test]
fn fp8_16x16x64_matches_the_isa_table() {
    // "8-bit A-Matrix 16x64": VGPR0 lanes 0-15 K=0..3, lanes 16-31 K=8..11; VGPR1 K=4..7 / 12..15; VGPR2 16..19 / 24..27; VGPR3 20..23 / 28..31; VGPR4-7 +32
    let a = MmaAtom::wmma_f32_16x16x64_fp8();
    for lane in 0..32i64 {
        for v in 0..8i64 {
            for b in 0..4i64 {
                let h = lane / 16;
                let k = [0, 4, 16, 20][(v % 4) as usize] + 8 * h + b + 32 * (v / 4);
                assert_eq!(a.a_tv.at(lane + 32 * (4 * v + b)), lane % 16 + 16 * k, "lane {lane} vgpr {v} byte {b}");
            }
            // 32-bit C 16x16: VGPR v holds M = v (lanes 0-15) / v + 8 (lanes 16-31), N = lane % 16
            assert_eq!(a.c_tv.at(lane + 32 * v), (v + 8 * (lane / 16)) + 16 * (lane % 16));
        }
    }
    // 16x128 = two 16x64 in consecutive VGPRs
    let b = MmaAtom::wmma_f32_16x16x128_fp8();
    for lane in 0..32i64 {
        for e in 0..32i64 {
            assert_eq!(b.a_tv.at(lane + 32 * e), a.a_tv.at(lane + 32 * e));
            assert_eq!(b.a_tv.at(lane + 32 * (e + 32)), a.a_tv.at(lane + 32 * e) + 64 * 16);
        }
    }
}

#[test]
fn f4_32x16x128_matches_the_isa_table() {
    // "4-bit Floats" (block-scale layout): V0..V3 lanes 0-15 K=0..31, lanes 16-31 K=32..63; V4..V7 K=64..95 / 96..127 (nibble j of a VGPR = the
    // next K); 32x128 A: M=16..31 in the next 8 VGPRs. C 32x16: M=0..15 in VGPRs 0..7, 16..31 in VGPRs 8..15.
    let a = MmaAtom::wmma_scale_f32_32x16x128_f4();
    for lane in 0..32i64 {
        let h = lane / 16;
        for v in 0..16i64 {
            for j in 0..8i64 {
                let k = 8 * (v % 4) + j + 32 * h + 64 * ((v / 4) % 2);
                let m = lane % 16 + 16 * (v / 8);
                assert_eq!(a.a_tv.at(lane + 32 * (8 * v + j)), m + 32 * k, "A lane {lane} vgpr {v} nibble {j}");
                if v < 8 {
                    assert_eq!(a.b_tv.at(lane + 32 * (8 * v + j)), lane % 16 + 16 * k, "B lane {lane} vgpr {v} nibble {j}");
                }
            }
            let m = (v % 8) + 8 * h + 16 * (v / 8);
            assert_eq!(a.c_tv.at(lane + 32 * v), m + 32 * (lane % 16), "C lane {lane} vgpr {v}");
        }
    }
    // scales: A one VGPR, lane = row (0..31), byte = k block of 32; B lanes 0..15 (or 16..31) = column, byte = k block
    let sa = scale_32x16x128_f4_a();
    let sb = scale_32x16x128_f4_b();
    for lane in 0..32i64 {
        for byte in 0..4i64 {
            assert_eq!(sa.at(lane + 32 * byte), lane + 32 * byte);
            if lane < 16 {
                assert_eq!(sb.at(lane + 16 * byte), lane + 16 * byte);
            }
        }
    }
}
