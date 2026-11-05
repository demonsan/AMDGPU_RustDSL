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

/// The fast-reference recipes (`Vs128Cfg::scaleconv`, lazy threshold): the frame moves by a WHOLE number of 1/512 units through the
/// conversion scale (or only when the max passes a threshold above the target), the accumulators are rescaled by the same 2^(512 delta) and the
/// tile's P are multiplied by it. The symbolic proof covers arbitrary frames m' (the induction step holds for any m'); this checks the concrete
/// recurrence of the kernel (base-2, 1/512 units, rounded / lazy frame moves) against the naive sums.
#[test]
fn integer_delta_and_lazy_frames_match_naive() {
    let naive2 = |s: &[f64], v: &[f64]| -> (f64, f64) {
        let o: f64 = s.iter().zip(v).map(|(x, y)| (512.0 * x).exp2() * y).sum();
        let d: f64 = s.iter().map(|x| (512.0 * x).exp2()).sum();
        (o, d)
    };
    for (tiles, lazy) in [(16usize, None), (16, Some(4.0 / 512.0)), (64, None), (64, Some(11.0 / 512.0))] {
        let s: Vec<f64> = (0..tiles * 128).map(|i| ((i as f64 * 0.731).sin() * 0.03) + (i / 128) as f64 * 0.0007).collect();
        let v: Vec<f64> = (0..tiles * 128).map(|i| ((i * 13 % 17) as f64) - 8.0).collect();
        let target = 131.0 / 512.0;
        let (mut b, mut o, mut d) = (0.0f64, 0.0f64, 0.0f64); // reference b: scores are carried as s + b
        for t in 0..tiles {
            let tile = &s[t * 128..(t + 1) * 128];
            let m = tile.iter().cloned().fold(f64::MIN, f64::max) + b;
            let rebase = t == 0 || match lazy { None => true, Some(h) => m > target + h };
            let delta = if rebase { ((target - m) * 512.0).round() / 512.0 } else { 0.0 };
            let f = (512.0 * delta).exp2();
            o *= f;
            d *= f;
            b += delta;
            for (x, y) in tile.iter().zip(&v[t * 128..(t + 1) * 128]) {
                // P = 2^(512 (s + b_old)) x 2^(512 delta): the conversion scale applies the whole-unit factor
                let p = (512.0 * (x + b - delta)).exp2() * f;
                o += p * y;
                d += p;
            }
        }
        let (no, nd) = naive2(&s, &v);
        let r = (o / d - no / nd).abs() / (no / nd).abs().max(1e-12);
        assert!(r < 1e-9, "tiles {tiles} lazy {lazy:?}: online {} vs naive {} (rel {r})", o / d, no / nd);
        // the final frame relation: d x 2^(-512 b) is the naive denominator
        assert!(((d * (-512.0 * b).exp2()) - nd).abs() / nd < 1e-9);
    }
}

/// The vendor's reference recipe (vendor-spec 4.7, `Vs128Cfg::vref`): scores carried as x = s + b (base 2, 512 units), the reference b moves ONLY
/// when a row max passes the threshold s92 (lazily, seeded on the first tile), by the exact (unrounded) f32 delta that puts the max at the target
/// s91; O and l are rescaled by 2^(512 delta) and the tile's P are computed from the shifted scores. The symbolic proof covers arbitrary frames;
/// this checks the concrete recurrence (with the vendor's constants and conversion scale) against the naive sums, including tiles whose max never
/// passes the threshold (frame kept) and rising sequences (many rollbacks).
#[test]
fn vendor_lazy_reference_recipe_matches_naive() {
    let s91 = f32::from_bits(0x3e7d_f17e) as f64;
    let s92 = f32::from_bits(0x3e85_d086) as f64;
    let conv = 0.25f64; // a constant factor on every P: cancels in O / l
    for (tiles, slope) in [(16usize, 0.0), (16, 0.004), (64, 0.0007), (64, -0.002)] {
        let s: Vec<f64> = (0..tiles * 128).map(|i| ((i as f64 * 0.917).sin() * 0.02) + (i / 128) as f64 * slope).collect();
        let v: Vec<f64> = (0..tiles * 128).map(|i| ((i * 11 % 19) as f64) - 9.0).collect();
        let (mut b, mut o, mut d) = (64.0f64, 0.0f64, 0.0f64); // the arbitrary initial reference (the bias words 0x4800)
        let mut rollbacks = 0;
        for t in 0..tiles {
            let tile = &s[t * 128..(t + 1) * 128];
            let m = tile.iter().cloned().fold(f64::MIN, f64::max) + b;
            if t == 0 || m > s92 {
                let delta = s91 - m;
                let f = (512.0 * delta).exp2();
                o *= f;
                d *= f;
                b += delta;
                rollbacks += 1;
            }
            for (x, y) in tile.iter().zip(&v[t * 128..(t + 1) * 128]) {
                let p = conv * (512.0 * (x + b)).exp2();
                o += p * y;
                d += p;
            }
        }
        let o_n: f64 = s.iter().zip(&v).map(|(x, y)| (512.0 * x).exp2() * y).sum();
        let d_n: f64 = s.iter().map(|x| (512.0 * x).exp2()).sum();
        let r = (o / d - o_n / d_n).abs() / (o_n / d_n).abs().max(1e-12);
        assert!(r < 1e-9, "tiles {tiles} slope {slope}: online {} vs naive {} (rel {r}, rollbacks {rollbacks})", o / d, o_n / d_n);
        assert!(rollbacks >= 1);
    }
}
