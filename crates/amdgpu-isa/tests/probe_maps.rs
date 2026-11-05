//! Replays the recorded one-hot hardware probe maps (tests/gpu/isa_probe_host.cpp, GPU run recorded in data/*.map) against the atoms and index semantics of
//! this crate: every measured (lane, register, byte) contribution must equal what `MmaAtom::smfmac_16x16x128_fp8`, `smfmac::k_of` and `tr_b8` say.

use amdgpu_isa::{smfmac, tr_b8, MmaAtom, Provenance};

const SMF: &str = include_str!("../data/smfmac_16x16x128_fp8.map");
const TR: &str = include_str!("../data/lds_tr_b8.map");

fn line<'a>(text: &'a str, key: &str) -> Vec<&'a str> {
    text.lines().find(|l| l.split_whitespace().next() == Some(key)).unwrap_or_else(|| panic!("{key} missing")).split_whitespace().skip(1).collect()
}
fn ints(text: &str, key: &str) -> Vec<i64> {
    line(text, key).iter().map(|t| t.parse().unwrap()).collect()
}

#[test]
fn the_recorded_probe_runs_had_no_violations() {
    assert_eq!(ints(SMF, "SMFMAC_VIOLATIONS"), vec![0]);
    assert_eq!(ints(TR, "TR_VIOLATIONS"), vec![0]);
}

#[test]
fn smfmac_operand_tvs_reproduce_the_measured_rows_columns_and_d_positions() {
    let atom = MmaAtom::smfmac_16x16x128_fp8();
    assert!(matches!(atom.provenance, Provenance::Probe(_)));
    // A bytes: (lane, byte) -> element m + 16 c; the measured row class of an A byte is its lane % 16 and c = 16 (lane / 16) + byte
    let arow = ints(SMF, "AROW");
    assert_eq!(arow.len(), 1024);
    for lane in 0..64i64 {
        for byte in 0..16i64 {
            let e = atom.a_tv.at(lane + 64 * byte);
            assert_eq!(e % 16, arow[(16 * lane + byte) as usize], "A row of lane {lane} byte {byte}");
            assert_eq!(e / 16, 16 * (lane / 16) + byte);
        }
    }
    // B bytes: (lane, register, byte) -> n + 16 k; the measured column class is lane % 16, the k label 32 (lane / 16) + 4 r + y
    let bcol = ints(SMF, "BCOL");
    let bk = ints(SMF, "BK");
    for lane in 0..64i64 {
        for v in 0..32i64 {
            let e = atom.b_tv.at(lane + 64 * v);
            let b = (32 * lane + v) as usize;
            assert_eq!(e % 16, bcol[b], "B column of lane {lane} byte {v}");
            assert_eq!(e / 16, bk[b]);
        }
    }
    // D: (lane, dword) -> m + 16 n against the measured (m, n)
    let dmap = line(SMF, "DMAP");
    assert_eq!(dmap.len(), 256);
    for lane in 0..64i64 {
        for r in 0..4i64 {
            let e = atom.c_tv.at(lane + 64 * r);
            let (m, n) = dmap[(4 * lane + r) as usize].split_once(',').map(|(a, b)| (a.parse::<i64>().unwrap(), b.parse::<i64>().unwrap())).unwrap();
            assert_eq!((e % 16, e / 16), (m, n), "D lane {lane} dword {r}");
        }
    }
}

#[test]
fn smfmac_index_semantics_reproduce_every_measured_pattern() {
    let mut pats = 0;
    for l in SMF.lines() {
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.first() != Some(&"PAT") {
            continue;
        }
        let w = u32::from_str_radix(t[2], 16).unwrap();
        let name = t[1];
        let kline = SMF.lines().find(|x| x.starts_with(&format!("KA {name} "))).unwrap();
        let k: Vec<i64> = kline.split_whitespace().skip(2).map(|x| x.parse().unwrap()).collect();
        assert_eq!(k.len(), 1024);
        for lane in 0..64i64 {
            for s in 0..16i64 {
                assert_eq!(k[(16 * lane + s) as usize], smfmac::k_of(lane, s, w), "pattern {name} lane {lane} stored byte {s}");
            }
        }
        pats += 1;
    }
    assert_eq!(pats, 14, "6 uniform nibble patterns (0x44444444 .. 0xeeeeeeee) + 8 walking nibbles");
}

#[test]
fn the_two_vendor_index_words_cover_all_128_k_exactly_once() {
    // 0x44444444 keeps positions (0, 1) of every group of 4, 0xeeeeeeee keeps (2, 3): the 2 x 64 contracted k's partition 0..127 for every A lane group
    assert_eq!(smfmac::uniform(0, 1), 0x4444_4444);
    assert_eq!(smfmac::uniform(2, 3), 0xeeee_eeee);
    for row in 0..16i64 {
        let mut seen = [0u8; 128];
        for g in 0..4 {
            for s in 0..16 {
                for w in [0x4444_4444u32, 0xeeee_eeee] {
                    seen[smfmac::k_of(16 * g + row, s, w) as usize] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "row {row}: {seen:?}");
    }
}

#[test]
fn tr_b8_formula_and_tv_reproduce_the_measured_map() {
    let s = ints(TR, "TR_S");
    let h = ints(TR, "TR_H");
    assert_eq!((s.len(), h.len()), (512, 512));
    let tv = tr_b8::dest_tv();
    for d in 0..64i64 {
        for j in 0..8i64 {
            let i = (8 * d + j) as usize;
            assert_eq!(tr_b8::source(d, j), (s[i], h[i]), "dest lane {d} byte {j}");
            let e = tv.at(d + 64 * j);
            assert_eq!((e / 8, e % 8), (s[i], h[i]), "TV of dest lane {d} byte {j}");
        }
    }
}