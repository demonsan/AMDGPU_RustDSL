//! The orientation search re-derives the FMHA S -> P design: S^T from the 32x32x64 atom (A = K, B = Q), P the B operand of the 32x32x64 PV MMA,
//! zero-cost conversion, and the FMHA's own key permutation (P packing in accumulator order) is one of the zero-cost permutations.
use amdgpu_isa::MmaAtom;
use amdgpu_tile::orient::*;

#[test]
fn fmha_s_to_p_is_rederived() {
    let atoms = [MmaAtom::mfma_scale_32x32x64_f8(), MmaAtom::mfma_scale_16x16x128_f8()];
    let all = search_attention(&atoms);
    for o in &all {
        println!("{:>22} S^T={:<5} -> {:>22} P={:?}  conv {:4} stats {:3} sigma {}", o.producer, o.s_transposed, o.consumer, o.p_operand, o.conv_cost, o.stats_cost, o.sigma.is_some());
    }
    let best = &all[0];
    assert_eq!((best.producer, best.s_transposed, best.consumer, best.p_operand), ("MfmaScale32x32x64F8", true, "MfmaScale32x32x64F8", POperand::B));
    assert_eq!(best.cost(), 0);
    // the un-transposed producer (A = Q, B = K) needs a cross-lane conversion
    let st = all.iter().find(|o| o.producer == "MfmaScale32x32x64F8" && !o.s_transposed && o.consumer == "MfmaScale32x32x64F8" && o.p_operand == POperand::B).unwrap();
    assert!(st.conv_cost > 0 && st.sigma.is_none());

    // the FMHA packing: P word slot j (0..31) of lane l = S value (accumulator j / 16, r = j % 16) -> key 32 (j / 16) + 8 (r / 4) + 4 (l / 32) + r % 4;
    // B operand slot j of lane l is k = 16 (l / 32) + j % 16 + 32 (j / 16): its sigma must make every lane's (query, key) multiset the producer's
    let c = MmaAtom::mfma_scale_32x32x64_f8();
    let src = producer_pairs(&c, true, 32, 64).unwrap();
    let dst = consumer_pairs(&c, POperand::B);
    for l in 0..64usize {
        let h = (l / 32) as i64;
        let mut want: Vec<(i64, i64)> = src[l].clone();
        let mut got: Vec<(i64, i64)> = dst[l].iter().map(|&(q, k)| {
            let (j0, j1) = ((k - 16 * h) % 32, k / 32);
            assert!((0..16).contains(&j0), "lane {l} k {k}");
            (q, 32 * j1 + 8 * (j0 / 4) + 4 * h + j0 % 4)
        }).collect();
        want.sort();
        got.sort();
        assert_eq!(got, want, "lane {l}: the FMHA key permutation is a zero-cost renaming");
    }
}
