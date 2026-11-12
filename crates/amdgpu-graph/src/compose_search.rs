//! Composition search (docs/compose.md): the candidates of a dense GEMM launch, with a coarse resource / time model used only to PRUNE, never to decide.
//!
//! A candidate is one composition of stage variants for a shape: a registered tile variant (256 x 256, 128 x 128, 64 x 128 with the 3-slot ring) and a split-K factor (+ the reduce kernel). The
//! pipeline is: enumerate ([`gemm_candidates`], shape feasibility from the kernels' constraints) -> prune the dominated ones ([`prune`], conservative: only candidates the model puts clearly behind another
//! that is also no larger) -> the survivors' compile-only feasibility (`amdc summary`: VGPR / AGPR / LDS) -> MEASURED top-k (`tools/compose_search.py`, same-run on the `perf_ab` hosts). The measured winner per
//! shape bucket is what the dispatch rule (`dispatch::{i8,bf16,blockscale}_config`) records with its evidence; `tools/compose_search.py check-gemm` re-runs the search over the evidence shapes and
//! reports where the recorded rule and the measured winner disagree. No runtime search: the rule stays a pure function of the shape.

use crate::compose::gemm_tile;

/// One enumerated composition of a GEMM launch.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub kernel: String,
    /// the split-K reduce kernel when `splits > 1`
    pub reduce: Option<String>,
    pub splits: u32,
    pub tile: (u32, u32),
    /// workgroups of the launch (all splits)
    pub ctas: u64,
    /// dispatch rounds over the CUs
    pub rounds: u64,
    /// the 3-slot ring (64-row forms) instead of the 2-slot ring
    pub ring_slots: u32,
    /// LDS bytes of the kernel's rings (from the kernel facts; 0 when unknown)
    pub lds_bytes: u32,
    /// the coarse model's time estimate (us): relative only
    pub model_us: f64,
}

struct Variant {
    suffix: &'static str,
    splitk: bool,
}

/// (kernel base name, rowcol reduce suffix, K unit, the variants that exist for the kind)
fn kind_info(kind: &str) -> Option<(&'static str, bool, u64, f64, &'static [Variant])> {
    const V3: &[Variant] = &[
        Variant { suffix: "", splitk: false },
        Variant { suffix: "_128", splitk: false },
        Variant { suffix: "_128_splitk", splitk: true },
        Variant { suffix: "_64", splitk: false },
        Variant { suffix: "_64_splitk", splitk: true },
    ];
    const VI8: &[Variant] = &[
        Variant { suffix: "", splitk: false },
        Variant { suffix: "_128", splitk: false },
        Variant { suffix: "_64", splitk: false },
        Variant { suffix: "_64_splitk", splitk: true },
    ];
    const VBS: &[Variant] = &[
        Variant { suffix: "", splitk: false },
        Variant { suffix: "_splitk", splitk: true },
        Variant { suffix: "_64", splitk: false },
        Variant { suffix: "_64_splitk", splitk: true },
    ];
    // (base, scales applied in the reduce (`_rc`), K elements per tile step, per-CU peak TFLOP/s, variants)
    Some(match kind {
        "bf16" => ("gemm_a16w16_bf16", false, 64, 9.8, V3),
        "fp8_rowcol" => ("gemm_a8w8_rowcol_fp8", true, 128, 19.5, V3),
        "i8_rowcol" => ("gemm_a8w8_i8", true, 128, 19.5, VI8),
        "fp8_blockscale" => ("gemm_a8w8_blockscale", false, 128, 19.5, VBS),
        _ => return None,
    })
}

fn tile_of(kind: &str, suffix: &str) -> (u32, u32) {
    if kind == "fp8_blockscale" && suffix.is_empty() || suffix == "_splitk" {
        return (128, 128);
    }
    gemm_tile(&format!("x{suffix}"))
}

/// The feasible candidates of a dense GEMM `C[M, N] = A B^T` of `kind` on `cus` CUs (an empty list: unknown kind). Feasibility is the kernels' shape constraints: the N tile divides N, the
/// 256-row form needs whole 256 tiles, a split needs N a multiple of 1024 (the reduce covers 1024-column blocks) and K whole tile steps per part (the 64-row 3-slot forms whole 256-element pairs).
pub fn gemm_candidates(kind: &str, m: u64, n: u64, k: u64, cus: u64) -> Vec<Candidate> {
    let Some((base, rc, kunit, peak, variants)) = kind_info(kind) else { return Vec::new() };
    let mut out = Vec::new();
    for v in variants {
        let (tm, tn) = tile_of(kind, v.suffix);
        if n % tn as u64 != 0 {
            continue;
        }
        if tm == 256 && m % 256 != 0 {
            continue;
        }
        let ring_slots = if tm == 64 { 3 } else { 2 };
        let split_options: &[u32] = if v.splitk { &[2, 4, 8] } else { &[1] };
        for &s in split_options {
            if s > 1 && n % 1024 != 0 {
                continue;
            }
            let unit = if tm == 64 { 2 * kunit } else { kunit };
            if k % (unit * s as u64) != 0 {
                continue;
            }
            let kernel = format!("{base}{}", v.suffix);
            let reduce = (s > 1).then(|| format!("gemm_splitk_reduce_s{s}{}", if rc { "_rc" } else { "" }));
            let ctas = m.div_ceil(tm as u64) * (n / tn as u64) * s as u64;
            let rounds = ctas.div_ceil(cus.max(1));
            // coarse model: the CTA's K loop at the better of the compute rate of its tile and an L2-fed operand stream, dispatched in `rounds` waves; the split-K reduce streams S fp32 partials
            let iters = (k / s as u64 / kunit) as f64;
            let compute_us = 2.0 * (tm as f64) * (tn as f64) * (kunit as f64) / (peak * 1.0e6);
            let operand_bytes = ((tm + tn) as f64) * (kunit as f64) * if kind == "bf16" { 2.0 } else { 1.0 };
            let stream_us = operand_bytes / 96.0e3; // per-CU share of the L2 operand stream (bytes / us), relative only
            let cta_us = iters * compute_us.max(stream_us) + 2.0;
            let reduce_us = if s > 1 { (s as f64) * (m as f64) * (n as f64) * 4.0 / 5.0e6 + 3.0 } else { 0.0 };
            let model_us = rounds as f64 * cta_us + reduce_us;
            let lds_bytes = crate::compose::facts(&kernel).map(|f| f.lds_bytes).unwrap_or(0);
            out.push(Candidate { kernel, reduce, splits: s, tile: (tm, tn), ctas, rounds, ring_slots, lds_bytes, model_us });
        }
    }
    out
}

/// Which candidates survive the conservative pruning: a candidate is dropped when another is at least `margin` times faster in the model AND uses no more LDS. (The model is coarse; the margin keeps the true
/// winner in the measured set.)
pub fn prune(c: &[Candidate], margin: f64) -> Vec<bool> {
    c.iter().map(|a| !c.iter().any(|b| b.model_us * margin <= a.model_us && b.lds_bytes <= a.lds_bytes)).collect()
}

impl Candidate {
    pub fn to_json(&self, kept: bool) -> String {
        let red = self.reduce.as_ref().map(|r| format!("\"{r}\"")).unwrap_or_else(|| "null".into());
        format!(
            "{{\"kernel\":\"{}\",\"reduce\":{},\"splits\":{},\"tile\":[{},{}],\"ctas\":{},\"rounds\":{},\"ring_slots\":{},\"lds_bytes\":{},\"model_us\":{:.2},\"kept\":{}}}",
            self.kernel, red, self.splits, self.tile.0, self.tile.1, self.ctas, self.rounds, self.ring_slots, self.lds_bytes, self.model_us, kept
        )
    }
}


// ---------------------------------------------------------------------------------------------------------------------------------------------------
// attention families: the split decisions (decode16 splits, prefill key splits)
// ---------------------------------------------------------------------------------------------------------------------------------------------------

fn attn_candidate(kernel: String, combine: Option<String>, splits: u32, ctas: u64, cus: u64, work_per_cta: f64, merge: f64) -> Candidate {
    let rounds = ctas.div_ceil(cus.max(1));
    let lds_bytes = crate::compose::facts(&kernel).map(|f| f.lds_bytes).unwrap_or(0);
    // coarse model: rounds over the CUs x the per-CTA key loop (+ the partial merge per split); relative only
    Candidate { kernel, reduce: combine, splits, tile: (0, 0), ctas, rounds, ring_slots: 2, lds_bytes, model_us: rounds as f64 * work_per_cta + merge * splits as f64 }
}

/// Candidates of a paged decode launch (bf16, hd128, G <= 16): the fused form (S = 1) and, for pages >= 16, the last-arriver form `_c` with S in {2, 4, 8, 16} key splits (at least 256 keys per split).
pub fn decode16_candidates(b: u64, hk: u64, ctx: u64, page: u32, cus: u64) -> Vec<Candidate> {
    let n = b * hk;
    let mut out = vec![attn_candidate(format!("pa_decode16_bf16_hd128_p{page}_f"), None, 1, n, cus, ctx as f64 * 0.01, 0.0)];
    if page >= 16 {
        for s in [2u32, 4, 8, 16] {
            if ctx / s as u64 >= 256 {
                out.push(attn_candidate(format!("pa_decode16_bf16_hd128_p{page}_c"), None, s, n * s as u64, cus, ctx as f64 / s as f64 * 0.01, 0.3));
            }
        }
    }
    out
}

/// Candidates of a bf16 hd128 prefill launch: non-causal the plain kernel and `_ks` with S in {2, 4}; causal the paired `_causal` (+ `_causal_ks` S in {2, 4}), the 128-row `_causal_q128` and the tiny-grid
/// `fmha_fwd_hd128_bf16_causal`. Splits need at least four key tiles of 128 per part; a split adds the combine kernel.
pub fn prefill_bf16_candidates(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> Vec<Candidate> {
    let u = b * h * sq.div_ceil(256);
    let kt = sk.div_ceil(128);
    let combine = Some("attn_combine_bf16_hd128_q16".to_string());
    let mut out = Vec::new();
    let paired = (u + 1) / 2;
    if !causal {
        out.push(attn_candidate("fmha_fwd_hd128_bf16_v3".into(), None, 1, u, cus, kt as f64, 0.0));
        for s in [2u32, 4] {
            if kt >= 4 * s as u64 {
                out.push(attn_candidate("fmha_fwd_hd128_bf16_v3_ks".into(), combine.clone(), s, u * s as u64, cus, kt as f64 / s as f64, 2.0));
            }
        }
    } else {
        out.push(attn_candidate("fmha_fwd_hd128_bf16_v3_causal".into(), None, 1, paired, cus, kt as f64, 0.0));
        for s in [2u32, 4] {
            if kt >= 4 * s as u64 {
                out.push(attn_candidate("fmha_fwd_hd128_bf16_v3_causal_ks".into(), combine.clone(), s, paired * s as u64, cus, kt as f64 / s as f64, 2.0));
            }
        }
        out.push(attn_candidate("fmha_fwd_hd128_bf16_v3_causal_q128".into(), None, 1, paired * 2, cus, kt as f64 / 2.0, 0.0));
        out.push(attn_candidate("fmha_fwd_hd128_bf16_causal".into(), None, 1, u * 2, cus, kt as f64 / 2.0, 0.0));
    }
    out
}

/// Candidates of a fused MoE launch (stage 1 + stage 2 + sum, `moe_host`): the routing sort's block_m (the `gemm_moe{1,2}_*` tile rows) 64 (`_bm64` kernels) or 128. `reduce` names the stage-2 kernel (the launch also runs
/// `moe_sum_bf16_k<topk>`). The model counts the padded rows per expert (ceil(rows / block_m) * block_m, rows = M topk / E): relative only.
pub fn moe_candidates(dt: &str, m: u64, topk: u64, e: u64, cus: u64) -> Vec<Candidate> {
    let rows = (m * topk) as f64 / e.max(1) as f64;
    [64u32, 128]
        .iter()
        .map(|&bm| {
            let sfx = if bm == 64 { "_bm64" } else { "" };
            let blocks = (rows / bm as f64).ceil().max(1.0) as u64 * e;
            let kernel = format!("gemm_moe1_{dt}_silu{sfx}");
            let lds_bytes = crate::compose::facts(&kernel).map(|f| f.lds_bytes).unwrap_or(0);
            let rounds = blocks.div_ceil(cus.max(1));
            Candidate { kernel, reduce: Some(format!("gemm_moe2_{dt}{sfx}")), splits: 1, tile: (bm, 0), ctas: blocks, rounds, ring_slots: 2, lds_bytes, model_us: rounds as f64 * (bm as f64 * 0.12 + 4.0) }
        })
        .collect()
}

/// Candidates of an MLA decode launch: the per-wave form (16 heads per CTA) or the head-parallel form (64 heads per CTA) with S in {1, 2, 4, 8, 16} key splits (+ the combine kernel).
pub fn mla_candidates(b: u64, h: u64, ctx: u64, page: u32, cus: u64) -> Vec<Candidate> {
    let mut out = Vec::new();
    for hp in [false, true] {
        let per = if hp { h.div_ceil(64) } else { h.div_ceil(16) };
        for s in [1u32, 2, 4, 8, 16] {
            if ctx / (s as u64) < 64 {
                continue;
            }
            let kernel = format!("mla_decode_bf16_{}p{page}", if hp { "hp_" } else { "" });
            let ctas = b * per * s as u64;
            let rounds = ctas.div_ceil(cus.max(1));
            let lds_bytes = crate::compose::facts(&kernel).map(|f| f.lds_bytes).unwrap_or(0);
            out.push(Candidate { kernel, reduce: Some("attn_combine_bf16_hd512".into()), splits: s, tile: (0, 0), ctas, rounds, ring_slots: 2, lds_bytes, model_us: rounds as f64 * (ctx as f64 / s as f64 * if hp { 0.02 } else { 0.05 }) + 0.5 * s as f64 });
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_and_mla_have_candidates() {
        let m = moe_candidates("fp8", 256, 8, 64, 256);
        assert_eq!(m.len(), 2);
        assert!(m.iter().any(|c| c.kernel.ends_with("_bm64")) && m.iter().any(|c| !c.kernel.ends_with("_bm64")));
        let l = mla_candidates(4, 128, 8192, 64, 256);
        assert!(l.iter().any(|c| c.kernel.contains("_hp_") && c.splits == 2) && l.iter().any(|c| !c.kernel.contains("_hp_") && c.splits == 16), "{l:?}");
    }

    #[test]
    fn decode16_and_prefill_have_split_candidates() {
        let d = decode16_candidates(4, 8, 8192, 16, 256);
        assert!(d.iter().any(|c| c.splits == 1) && d.iter().any(|c| c.splits == 4 && c.kernel.ends_with("_c")), "{d:?}");
        assert!(decode16_candidates(4, 8, 8192, 8, 256).iter().all(|c| c.splits == 1), "pages below 16 run the fused form only");
        let p = prefill_bf16_candidates(1, 8, 4096, 4096, false, 256);
        assert!(p.iter().any(|c| c.splits == 2 && c.reduce.is_some()) && p.iter().any(|c| c.splits == 1), "{p:?}");
        assert!(prefill_bf16_candidates(1, 8, 4096, 4096, true, 256).iter().any(|c| c.kernel.ends_with("_causal_q128")));
    }

    #[test]
    fn small_m_has_split_candidates_and_a_big_grid_does_not() {
        let small = gemm_candidates("fp8_blockscale", 64, 8192, 8192, 256);
        assert!(small.iter().any(|c| c.splits > 1 && c.reduce.is_some() && c.tile == (64, 128)), "{small:?}");
        let big = gemm_candidates("i8_rowcol", 4096, 4096, 8192, 256);
        assert!(big.iter().any(|c| c.tile == (256, 256) && c.splits == 1));
        assert!(gemm_candidates("nope", 1, 1, 1, 256).is_empty());
    }

    #[test]
    fn constraints_drop_infeasible_shapes() {
        // N not a multiple of 1024: no split; N not a multiple of 256: no 256 tile
        let c = gemm_candidates("bf16", 64, 4352, 8192, 256);
        assert!(c.iter().all(|x| x.splits == 1 && x.tile != (256, 256)), "{c:?}");
    }

    #[test]
    fn pruning_keeps_the_model_best() {
        let c = gemm_candidates("fp8_rowcol", 256, 7168, 4096, 256);
        let kept = prune(&c, 1.5);
        let best = c.iter().enumerate().min_by(|a, b| a.1.model_us.partial_cmp(&b.1.model_us).unwrap()).unwrap().0;
        assert!(kept[best]);
    }
}
