//! StageBoundary checks: the MLP P hand-off is a zero-cost renaming; mismatches are reported.
use amdgpu_isa::MmaAtom;
use amdgpu_layout::Movement;
use amdgpu_tile::contract_stage::*;

fn frag(name: &str, atom_tv: &amdgpu_layout::Layout) -> LiveFrag {
    LiveFrag::from_tv(name, atom_tv, 32, Storage::Vgpr)
}

#[test]
fn boundary_checks() {
    let c = MmaAtom::mfma_scale_32x32x64_f8().c_tv;
    let roles = vec!["all".to_string()];
    let p = StageBoundary { live: vec![frag("S", &c)], lds_freed: vec!["ring".into()], vm: 0, lgkm: 2, barrier_phase: 1, roles: roles.clone(), ..Default::default() };
    let ok = StageBoundary { live: vec![frag("S", &c)], lds_reused: vec!["ring".into()], vm: 0, lgkm: 4, barrier_phase: 1, roles: roles.clone(), ..Default::default() };
    let r = check_boundary(&p, &ok).unwrap();
    assert_eq!(r.frags[0].1, Movement::Identity);
    // transposed expectation: allowed, but the report shows the cross-lane cost
    let ct = amdgpu_layout::make_layout(amdgpu_layout::tup![amdgpu_layout::tup![32, 2], amdgpu_layout::tup![4, 4]], amdgpu_layout::tup![amdgpu_layout::tup![1, 128], amdgpu_layout::tup![32, 256]]);
    let tr = StageBoundary { live: vec![frag("S", &ct)], ..ok.clone() };
    assert_eq!(check_boundary(&p, &tr).unwrap().frags[0].1, Movement::CrossLane);
    // errors: missing fragment, LDS reuse without free, too many outstanding lgkm, barrier phase
    assert!(check_boundary(&p, &StageBoundary { live: vec![frag("T", &c)], ..ok.clone() }).is_err());
    assert!(check_boundary(&p, &StageBoundary { lds_reused: vec!["epi".into()], ..ok.clone() }).is_err());
    assert!(check_boundary(&p, &StageBoundary { lgkm: 1, ..ok.clone() }).is_err());
    assert!(check_boundary(&p, &StageBoundary { barrier_phase: 2, ..ok.clone() }).is_err());
}
