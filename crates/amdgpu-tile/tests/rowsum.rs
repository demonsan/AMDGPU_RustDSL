//! The two-smfmac row-sum plan (crate::rowsum) on the probe-measured smfmac maps.
use amdgpu_tile::rowsum::{plan, prove, search, RowSum2};

#[test]
fn rowsum2_plan_is_proven_and_found_by_the_search() {
    let p = plan();
    let all = search();
    assert!(all.contains(&p), "{p:?} not among {all:?}");
    println!("rowsum2 plans: {}", all.len());
}

#[test]
fn rowsum2_rejects_a_same_index_pair() {
    // the same byte pair on both rows misses half of every dword
    let bad = RowSum2 { dwords: [0, 1], idx: [0x4444_4444, 0x4444_4444] };
    assert!(prove(&bad).is_err());
}
