//! Bit-field normal form: exact for every value of the symbol, and cheaper than the structural lowering for the swizzled DMA / fragment offsets.
use amdgpu_layout::*;

/// The grouped lowering, evaluated: XOR over groups of `(x & M) << d` (`>> -d`), XOR the constant.
fn grouped(bf: &BitForm, x: u32) -> u32 {
    let mut v = bf.konst;
    for g in bf.groups() {
        let t = x & g.mask;
        v ^= if g.shift >= 0 { t << g.shift } else { t >> (-g.shift) };
    }
    v
}

fn check(e: &Ix, d: &Dyns, hi: i64) -> BitForm {
    let bf = e.bitform(d).unwrap_or_else(|| panic!("no bit form for {e}"));
    for x in 0..=hi {
        let want = e.eval(&|_| x);
        assert_eq!(bf.eval(x as u32) as i64, want, "form of {e} at {x}");
        assert_eq!(grouped(&bf, x as u32) as i64, want, "groups of {e} at {x}");
    }
    bf
}

#[test]
fn swizzled_dma_row_offsets() {
    // the A-tile DMA of the GEMM: LDS byte of thread t, issue q -> swizzled (row, col) -> row * 128 + col (one K-row of 128 bytes)
    let mut d = Dyns::new();
    let tid = d.sym("tid", 0, 255);
    let swz = Swizzle::fmha128();
    for q in 0..4i64 {
        let lds = tid.mod_c(64, &d).mul_c(16).add(&tid.div_c(64, &d).mul_c(4096)).add_c(1024 * q);
        let e = swz.apply_ix(&lds, &d);
        let row = e.div_c(128, &d);
        let col = e.mod_c(128, &d);
        let off = row.mul_c(128).add(&col);
        let bf = check(&off, &d, 255);
        assert!(bf.cost() < off.op_cost(), "issue {q}: grouped {} vs structural {}", bf.cost(), off.op_cost());
        // the row alone (multiplied by a dynamic stride later) is a bit form too
        let rb = check(&row, &d, 255);
        assert!(rb.cost() <= row.op_cost());
    }
}

#[test]
fn fragment_bases_and_lane_fields() {
    let mut d = Dyns::new();
    let lane = d.sym("lane", 0, 63);
    // kfx-like: 128 (lane % 32) + 16 (((lane / 32) ^ s(lane % 32)) ^ 2)
    let j = lane.mod_c(32, &d);
    let h = lane.div_c(32, &d);
    let s = Swizzle::fmha128().term_ix_down(&j.mul_c(128), &d, 4);
    let e = j.mul_c(128).add(&h.xor(&s).xor(&Ix::C(2)).mul_c(16));
    check(&e, &d, 63);
    // merged fields: (lane & 7) << 4 + ((lane >> 3) & 7) << 7 is ((lane & 63) ...) with two shift groups at most
    let f = lane.mod_c(8, &d).mul_c(16).add(&lane.div_c(8, &d).mod_c(8, &d).mul_c(128));
    let bf = check(&f, &d, 63);
    assert!(bf.groups().len() <= 2, "{:?}", bf.groups());
}

#[test]
fn not_a_bit_form() {
    let mut d = Dyns::new();
    let t = d.sym("t", 0, 255);
    let k = d.sym("K", 128, 1 << 20);
    assert!(t.mul(&k).bitform(&d).is_none(), "a product with a dynamic stride");
    assert!(t.add(&t.div_c(2, &d)).bitform(&d).is_none(), "a sum with possible carries");
    assert!(t.mul_c(3).bitform(&d).is_none(), "a non power-of-two coefficient");
    let neg = d.sym("n", -4, 4);
    assert!(neg.bitform(&d).is_none(), "a possibly negative symbol");
}

#[test]
fn can_prove_through_the_shared_constraint_context() {
    let mut d = Dyns::new();
    let lane = d.sym("lane", 0, 63);
    let k = d.sym("K", 128, 1 << 20);
    let e = lane.mul_c(4).add_c(12);
    assert!(e.can_prove_le(&Ix::C(264), &d), "63 * 4 + 12 = 264");
    assert!(!e.can_prove_le(&Ix::C(263), &d));
    // symbolic: lane * 4 + 12 <= K + 136 for every K >= 128
    assert!(e.can_prove_le(&k.add_c(136), &d));
    assert!(!e.can_prove_le(&k.add_c(135), &d));
}
