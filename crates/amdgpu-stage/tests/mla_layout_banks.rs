//! The MLA page slot's LDS layout (`PageLayout::plan`): the bank-conflict cost of its K and V reads under the planned padding, by the
//! `tdm_pad` lane model, against the ideal (every read at its minimum number of LDS cycles). Prints the per-class numbers
//! (`cargo test -p amdgpu-stage --test mla_layout_banks -- --nocapture`) and fails if the plan stops being conflict-free.

use amdgpu_dsl::tdm_pad::{cost, LaneRead};
use amdgpu_stage::attention::mla::{PageLayout, DV};

fn k_reads(wave_row0: u32) -> Vec<LaneRead> {
    let mut v = Vec::new();
    for c in 0..8u32 {
        for h in 0..2u32 {
            v.push(LaneRead { bytes: 16, lanes: (0..32u32).map(|l| Some((wave_row0 + (l & 15)) * DV as u32 + 64 * c + 32 * (l >> 4) + 16 * h)).collect() });
        }
    }
    v
}

fn v_reads() -> Vec<LaneRead> {
    (0..8u32)
        .map(|nt| LaneRead { bytes: 8, lanes: (0..32u32).map(|f| Some((8 * (f >> 4) + (f & 3) + 4 * ((f >> 3) & 1)) * DV as u32 + 8 * ((f >> 2) & 1) + 16 * nt)).collect() })
        .collect()
}

#[test]
fn mla_page_reads_are_conflict_free_under_the_plan() {
    let layout = PageLayout::plan();
    let ideal = |r: &[LaneRead]| amdgpu_dsl::tdm_pad::plan(amdgpu_stage::attention::mla::QUARTER as u32, r).ideal;
    let k = k_reads(0);
    let v = v_reads();
    let (kp, ku, ki) = (cost(&k, layout.pad), cost(&k, None), ideal(&k));
    let (vp, vu, vi) = (cost(&v, layout.pad), cost(&v, None), ideal(&v));
    println!("pad {:?}: K reads cost {kp} (unpadded {ku}, ideal {ki}); V tr8 reads cost {vp} (unpadded {vu}, ideal {vi})", layout.pad);
    // the other waves' K rows (16 rows further each): the same cost (the padding restarts per quarter)
    for w in 1..4u32 {
        let kw = k_reads(16 * w);
        println!("  wave {w} K reads cost {}", cost(&kw, layout.pad));
    }
    // the rope chunk: lane (m, g) reads 2 x 16 B at ROPE_OFF + m 64 + 32 g + 16 h (row stride 64 B; not in the plan's read model)
    let rope: Vec<LaneRead> = (0..2u32).map(|h| LaneRead { bytes: 16, lanes: (0..32u32).map(|l| Some(amdgpu_stage::attention::mla::ROPE_OFF as u32 + (l & 15) * 64 + 32 * (l >> 4) + 16 * h)).collect() }).collect();
    println!("  rope reads cost {} (unpadded {}, ideal {})", cost(&rope, layout.pad), cost(&rope, None), ideal(&rope));
    assert_eq!(kp, ki, "K reads conflict under the planned padding");
    assert_eq!(vp, vi, "V reads conflict under the planned padding");
}
