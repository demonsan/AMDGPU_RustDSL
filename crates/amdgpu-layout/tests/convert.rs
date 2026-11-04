//! plan_convert: classification + GF(2) bit moves on the MFMA layouts.
use amdgpu_layout::*;

fn c32() -> Layout { make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![32, 4], tup![1, 8]]) }

#[test]
fn identity_and_register_renaming() {
    let c = c32();
    let p = plan_convert(&c, &c, 1);
    assert_eq!(p.movement, Movement::Identity);
    assert_eq!(p.cost, 0);

}

#[test]
fn register_permutation_is_free() {
    // value index (r % 4, r / 4) -> (r / 4, r % 4) ... a different value order with whole registers
    let a = make_layout(tup![64, tup![4, 4]], tup![1, tup![64, 256]]);
    let b = make_layout(tup![64, tup![4, 4]], tup![1, tup![256, 64]]);
    let p = plan_convert(&a, &b, 1);
    assert_eq!(p.movement, Movement::Register);
    assert_eq!(p.cost, 0);
    assert_eq!(p.reg_map.as_ref().unwrap()[1], 4);
}

#[test]
fn quad_half_cross() {
    // lanes swapped inside each quad (lane ^ 1)
    let a = make_layout(tup![tup![2, 32], 4], tup![tup![1, 2], 64]);

    let b = make_layout(tup![tup![2, 2, 16], 4], tup![tup![1, 2, 4], 64]);
    assert_eq!(plan_convert(&a, &b, 1).movement, Movement::Identity);
    // lane bits 0 and 1 exchanged: a quad permutation
    let sw = make_layout(tup![tup![2, 2, 16], 4], tup![tup![2, 1, 4], 64]);
    let p = plan_convert(&a, &sw, 1);
    assert_eq!(p.movement, Movement::Quad);
    assert_eq!(p.bit_moves().unwrap(), vec!["lane bit 0 -> lane bit 1".to_string(), "lane bit 1 -> lane bit 0".to_string()]);
    // lane bit 5 <-> value bit 0: every lane reads itself or lane ^ 32 (v_permlane32_swap)
    let x = make_layout(tup![tup![32, 2], tup![2, 2]], tup![tup![1, 64], tup![32, 128]]);
    let p = plan_convert(&a, &x, 1);
    assert_eq!(p.movement, Movement::Half);
    assert!(p.bit_moves().unwrap().contains(&"lane bit 5 -> value bit 0".to_string()));
    // lane bit 4 <-> lane bit 0: across a row of 16, not a quad: cross-lane
    let y = make_layout(tup![tup![2, 2, 2, 2, 2, 2], 4], tup![tup![16, 2, 4, 8, 1, 32], 64]);
    assert_eq!(plan_convert(&a, &y, 1).movement, Movement::CrossLane);
}

#[test]
fn mfma_c_to_its_transpose_is_cross_lane() {
    // C of the 32x32 atom vs the same tile with rows / columns exchanged in the lane mapping (lane = row): the S -> P hazard without swap_ab
    let c = c32();
    let ct = make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![1, 128], tup![32, 256]]);
    let p = plan_convert(&c, &ct, 1);
    assert_eq!(p.movement, Movement::CrossLane);
    assert!(p.cost > 0);
}
