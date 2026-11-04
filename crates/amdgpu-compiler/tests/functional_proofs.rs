//! Item E2: the online-softmax loop-invariant proof and the broken-recipe counterexample.
use amdgpu_compiler::functional_proofs::{naive, online, softmax_counterexample, softmax_invariant_proof};

#[test]
fn online_softmax_rebase_invariant_is_proved() {
    softmax_invariant_proof().expect("symbolic invariant proof");
}

#[test]
fn good_recipe_matches_naive_and_missing_rescale_fails() {
    let (s, v, good, bad) = softmax_counterexample();
    let (no, nd) = naive(&s, &v);
    assert!((good.0 - no).abs() < 1e-12 && (good.1 - nd).abs() < 1e-12, "good recipe {good:?} vs naive {:?}", (no, nd));
    assert!((bad.0 - no).abs() > 1e-3, "missing rescale must produce a concrete mismatch: bad={:?} naive={no}", bad);
    // the missing rescale changes an early contribution's frame; the error is exact and reproducible
    println!("E2 counterexample: scores={s:?} vals={v:?} naive={no} broken sum={}", bad.0);
}

#[test]
fn rollback_rebase_factor_is_one_for_arbitrary_lengths() {
    // numeric sanity of the same identity with frames changing up and down (rollback)
    for n in [1usize, 2, 7, 64] {
        let s: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let v: Vec<f64> = (0..n).map(|i| ((i * 7 % 11) as f64) - 5.0).collect();
        let (go, gd) = online(&s, &v, true);
        let (no, nd) = naive(&s, &v);
        assert!((go - no).abs() < 1e-9 && (gd - nd).abs() < 1e-9, "n={n}: online {go}/{gd} vs naive {no}/{nd}");
    }
}
