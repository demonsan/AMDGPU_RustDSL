//! Layout algebra: the FlyDSL / CuTe documentation examples, the algebraic identities (checked as functions), dynamic outer modes, recast, swizzles.
use amdgpu_layout::proofs::*;
use amdgpu_layout::*;

fn same_fn(a: &Layout, b: &Layout) {
    let n = a.size().st();
    assert_eq!(n, b.size().st(), "sizes of {a} and {b}");
    for i in 0..n {
        assert_eq!(a.at(i), b.at(i), "{a} and {b} differ at {i}");
    }
}

#[test]
fn flydsl_coordinate_mapping_example() {
    // layout_system_guide.md §3: ((8, 16), (1, 8)): crd2idx((3, 5)) = 43, idx2crd(43) = (3, 5)
    let d = Dyns::new();
    let l = make_layout(tup![8, 16], tup![1, 8]);
    assert_eq!(l.crd2idx(&tup![3, 5], &d), Ix::C(43));
    assert_eq!(idx2crd(&Ix::C(43), &l.shape, &d), tup![3, 5]);
    assert_eq!(l.size(), Ix::C(128));
    assert_eq!(l.cosize(), Ix::C(128));
    assert_eq!(l.rank(), 2);
    assert_eq!(tup![9, tup![4, 8]].depth(), 2);
    assert_eq!(tup![9, tup![4, 8]].product(), Ix::C(288));
}

#[test]
fn cute_doc_composition() {
    // (6,2):(8,2) o (4,3):(3,1) = ((2,2),3):((24,2),8)
    let a = make_layout(tup![6, 2], tup![8, 2]);
    let b = make_layout(tup![4, 3], tup![3, 1]);
    let c = composition(&a, &b);
    assert_eq!(c, make_layout(tup![tup![2, 2], 3], tup![tup![24, 2], 8]), "{c}");
    for i in 0..12 {
        assert_eq!(c.at(i), a.at(b.at(i)));
    }
}

#[test]
fn cute_doc_complement_divide_product() {
    assert_eq!(complement(&make_layout(4, 2), 24), make_layout(tup![2, 3], tup![1, 8]));
    let ld = logical_divide(&make_layout(tup![4, 2, 3], tup![2, 1, 8]), &make_layout(4, 2));
    assert_eq!(ld, make_layout(tup![tup![2, 2], tup![2, 3]], tup![tup![4, 1], tup![2, 8]]), "{ld}");
    let lp = logical_product(&make_layout(tup![2, 2], tup![4, 1]), &make_layout(6, 1));
    assert_eq!(lp, make_layout(tup![tup![2, 2], tup![2, 3]], tup![tup![4, 1], tup![2, 8]]), "{lp}");
}

#[test]
fn coalesce_and_filter() {
    let l = make_layout(tup![2, tup![1, 6]], tup![1, tup![6, 2]]);
    assert_eq!(coalesce(&l), make_layout(12, 1));
    same_fn(&l, &coalesce(&l));
    assert_eq!(filter(&make_layout(tup![4, 3], tup![0, 4])), make_layout(3, 4));
}

#[test]
fn inverses() {
    let l = make_layout(tup![4, 8], tup![8, 1]);
    let r = right_inverse(&l);
    assert_eq!(r, make_layout(tup![8, 4], tup![4, 1]), "{r}");
    for i in 0..32 {
        assert_eq!(l.at(r.at(i)), i);
    }
    let l2 = make_layout(tup![2, 4], tup![8, 1]);
    let li = left_inverse(&l2);
    for i in 0..8 {
        assert_eq!(li.at(l2.at(i)), i, "left inverse {li} of {l2}");
    }
}

#[test]
fn products_and_divides_are_consistent() {
    let a = make_layout(tup![2, 2], tup![1, 2]);
    let b = make_layout(tup![3, 2], tup![1, 3]);
    for p in [blocked_product(&a, &b), raked_product(&a, &b)] {
        let n = p.size().st();
        assert_eq!(n, 24);
        let mut seen = vec![false; 24];
        for i in 0..n {
            let v = p.at(i);
            assert!(!seen[v as usize], "{p} not injective");
            seen[v as usize] = true;
        }
    }
    // rank 1: blocked = contiguous blocks, raked = interleaved
    same_fn(&blocked_product(&make_layout(2, 1), &make_layout(3, 1)), &make_layout(6, 1));
    let rk = raked_product(&make_layout(2, 1), &make_layout(3, 1));
    let vals: Vec<i64> = (0..6).map(|i| rk.at(i)).collect();
    assert_eq!(vals, vec![0, 2, 4, 1, 3, 5], "{rk}");
    let z = zipped_divide_bymode(&make_layout(tup![8, 6], tup![1, 8]), &tiler(&[4, 3]), &Dyns::new());
    assert_eq!(z.get(0).size(), Ix::C(12));
    assert_eq!(z.get(1).size(), Ix::C(4));
}

#[test]
fn dynamic_outer_modes_and_local_tile() {
    // a [S rows x 128 bytes] row-major tensor with a dynamic row count and row stride: tiles of 128 x 128
    let mut d = Dyns::new();
    let s = d.sym("S", 1, 1 << 20);
    let stride = d.sym("row_stride", 128, 1 << 16);
    let t = make_layout(tup![s.clone(), 128], tup![stride.clone(), 1]);
    let tile_i = d.sym("tile", 0, 1 << 13);
    let (tl, off) = local_tile(&t, &[128, 128], &[tile_i.clone(), Ix::C(0)], &d);
    assert_eq!(tl.get(0).size(), Ix::C(128));
    let env = |x: u32| [1000i64, 256, 3][x as usize];
    assert_eq!(off.eval(&env), 3 * 128 * 256, "offset {off}");
    // a row of the tile: crd2idx is symbolic in the dynamic stride
    assert_eq!(tl.crd2idx(&tup![5, 7], &d).eval(&env), 5 * 256 + 7);
}

#[test]
fn recast_between_element_widths() {
    let l16 = make_layout(tup![4, 8], tup![1, 4]); // 32 fp16 elements, column-major
    let l8 = recast_layout(&l16, 16, 8);
    assert_eq!(l8, make_layout(64, 1), "{l8}");
    assert_eq!(recast_layout(&l8, 8, 32), make_layout(16, 1));
    let rm = make_layout(tup![4, 8], tup![8, 1]); // row-major fp8, 8 bytes per row
    let w = recast_layout(&rm, 8, 32); // dwords: 4 rows x 2 dwords
    assert_eq!(w.size(), Ix::C(8), "{w}");
}

#[test]
fn structural_ops() {
    let l = make_layout(tup![2, 3, 4], tup![1, 2, 6]);
    assert_eq!(l.select(&[2, 0]), make_layout(tup![4, 2], tup![6, 1]));
    assert_eq!(l.group(1, 3), make_layout(tup![2, tup![3, 4]], tup![1, tup![2, 6]]));
    let (s, off) = slice(&l, &[None, Some(leaf(2)), None], &Dyns::new());
    assert_eq!(off, Ix::C(4));
    assert_eq!(s, make_layout(tup![2, 4], tup![1, 6]));
    assert_eq!(make_layout(2, 1).append(&make_layout(3, 2)), make_layout(tup![2, 3], tup![1, 2]));
    let o = make_ordered_layout(tup![4, 8], &[1, 0]);
    assert_eq!(o, make_layout(tup![4, 8], tup![8, 1]), "row-major via order");
}

#[test]
fn layout_tv_covers_the_tile() {
    let (mn, tv) = make_layout_tv(&make_layout(tup![4, 8], tup![8, 1]), &make_layout(tup![2, 2], tup![2, 1]));
    assert_eq!(mn.product(), Ix::C(128));
    check_tv(&tv, 128, false).expect("TV layout is a bijection onto the tile");
    assert!(check_tv(&make_layout(tup![4, 2], tup![1, 4]), 16, false).is_err(), "a TV layout that misses half of the tile");
    assert!(check_tv(&make_layout(tup![4, 4], tup![1, 2]), 10, false).is_err(), "overlapping values are not injective");
}

#[test]
fn swizzles() {
    let s = Swizzle::cute(3, 4, 3);
    assert!(s.is_bijective());
    check_swizzle_bijective(&s, 1 << 10).unwrap();
    let f = Swizzle::fmha128();
    check_swizzle_bijective(&f, 16384).unwrap();
    // the manual swz_s of kernels.rs: p = r & 1, u = (r >> 1) & 7, s2 = u2 ^ p, s1 = u0, s0 = u1 ^ p
    for r in 0..128i64 {
        let p = r & 1;
        let (u0, u1, u2) = ((r >> 1) & 1, (r >> 2) & 1, (r >> 3) & 1);
        let sr = ((u2 ^ p) << 2) + (u0 << 1) + (u1 ^ p);
        for c in 0..8i64 {
            assert_eq!(f.apply(r * 128 + c * 16), r * 128 + ((c ^ sr) << 4), "row {r} chunk {c}");
        }
    }
    // symbolic application agrees with the concrete one
    let mut d = Dyns::new();
    let x = d.sym("x", 0, 16383);
    let e = f.apply_ix(&x, &d);
    for v in [0i64, 129, 1000, 5555, 16383] {
        assert_eq!(e.eval(&|_| v), f.apply(v));
    }
}

#[test]
fn ix_simplification_uses_ranges() {
    let mut d = Dyns::new();
    let lane = d.sym("lane", 0, 63);
    assert_eq!(lane.div_c(32, &d).mod_c(2, &d), lane.div_c(32, &d), "(lane / 32) % 2 = lane / 32 for lane < 64");
    let e = lane.mod_c(32, &d).mul_c(128).add(&lane.div_c(32, &d).mul_c(16)).add_c(4096);
    let (a, c) = e.split_const();
    assert_eq!(c, 4096);
    assert_eq!(a.bits(&d) & 4096, 0);
    assert_eq!(a.bits(&d), (31 << 7) | 16);
}
