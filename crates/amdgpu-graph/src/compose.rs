//! Composition planning (docs/two-weavers-plan.md, front end; docs/compose.md): for a launch shape, WHICH composition of stage variants runs - the kernel, its key / K splits and
//! the reduce / combine kernel, the CTA tile, the CTA shape (threads, waves) and the ring depth the kernel was built with.
//!
//! The decisions are the per-shape-bucket rules of [`crate::dispatch`] (each documents the measured evidence it was fitted on: same-run times against the vendor kernel per bucket); this module
//! turns their answers into one [`Composition`] record and attaches the FACTS of the chosen kernel: the CTA shape and the LDS rings its plan declares (`compose_facts.rs`, generated from the
//! kernels' own plans by `crates/amdgpu-kernels/tests/compose_facts.rs`, so the table cannot drift from the kernels). No runtime search: a rule is a pure function of the shape.

use crate::compose_facts::FACTS;
use crate::dispatch;

/// A ring of a kernel's plan (`plan::RingDecl`): LDS slots between a producer and a consumer phase.
#[derive(Clone, Debug, PartialEq)]
pub struct RingFact {
    pub name: &'static str,
    pub slots: u32,
    pub bytes_per_slot: u32,
    /// each wave owns its slot (wave-private tiles)
    pub wave_private: bool,
}

/// What a kernel's own plan declares: the CTA shape and rings, and the size of the plan (generated into `compose_facts.rs` from the kernels).
#[derive(Clone, Debug, PartialEq)]
pub struct KernelFacts {
    pub kernel: &'static str,
    pub threads: u32,
    pub waves: u32,
    pub phases: u32,
    pub units: u32,
    pub rings: &'static [RingFact],
}

/// One composition decision for a launch shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Composition {
    /// the family (`decode16`, `prefill_bf16`, `fp8pt`, `hd192`, `mxfp8`, `gemm`, `mla`, `moe`, `bwd_causal`)
    pub family: &'static str,
    /// the kernels of the launch in launch order (the main kernel first, then its reduce / combine kernel)
    pub kernels: Vec<String>,
    /// key / K splits of the main kernel (1 = none)
    pub splits: u32,
    /// the CTA tile (rows, columns) where the family has one
    pub tile: Option<(u32, u32)>,
    /// the rule function that decided (its doc comment carries the measured evidence)
    pub rule: &'static str,
    /// the plan facts of the main kernel (None: it declares no plan, i.e. it never wove)
    pub facts: Option<&'static KernelFacts>,
}

/// The plan facts of a kernel (CTA shape, rings), if it has a plan.
pub fn facts(kernel: &str) -> Option<&'static KernelFacts> {
    FACTS.iter().find(|f| f.kernel == kernel)
}

fn make(family: &'static str, rule: &'static str, kernels: Vec<String>, splits: u32, tile: Option<(u32, u32)>) -> Composition {
    let facts = facts(&kernels[0]);
    Composition { family, kernels, splits, tile, rule, facts }
}

/// The CTA tile (rows, columns) of a GEMM kernel by its name: `_64*` 64 x 128, `_128*` and the block-scale base 128 x 128, else 256 x 256.
pub fn gemm_tile(kernel: &str) -> (u32, u32) {
    if kernel.ends_with("_64") || kernel.ends_with("_64_splitk") {
        (64, 128)
    } else if kernel.ends_with("_128") || kernel.ends_with("_128_splitk") || kernel.starts_with("gemm_a8w8_blockscale") {
        (128, 128)
    } else {
        (256, 256)
    }
}

/// Paged decode (bf16, hd128, G <= 16): fused or last-arriver form and the split count.
pub fn decode16(b: u64, hk: u64, ctx: u64, page: u32) -> Composition {
    let (k, s) = dispatch::decode16_kernel(page, dispatch::decode16_plan(b, hk, ctx));
    make("decode16", "dispatch::decode16_plan / decode16_kernel", vec![k], s, None)
}

/// bf16 hd128 prefill: the plain / paired / q128 form and the key splits (+ the combine kernel when split).
pub fn prefill_bf16(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> Composition {
    let (k, s) = dispatch::bf16_prefill_kernel(b, h, sq, sk, causal, cus);
    let mut ks = vec![k.to_string()];
    if s > 1 {
        ks.push("attn_combine_bf16_hd128_q16".into());
    }
    make("prefill_bf16", "dispatch::bf16_prefill_kernel", ks, s, None)
}

/// Per-tensor FP8 prefill.
pub fn fp8pt(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> Composition {
    let (k, s) = dispatch::fp8pt_prefill_kernel(b, h, sq, sk, causal, cus);
    let mut ks = vec![k.to_string()];
    if s > 1 {
        ks.push("fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h".into());
    }
    make("fp8pt", "dispatch::fp8pt_prefill_kernel", ks, s, None)
}

/// hd192 / hd128 (MLA prefill) bf16.
pub fn hd192(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> Composition {
    let (k, s) = dispatch::hd192_prefill_kernel(b, h, sq, sk, causal, cus);
    let mut ks = vec![k.to_string()];
    if s > 1 {
        ks.push("attn_combine_bf16_hd128_q16".into());
    }
    make("hd192", "dispatch::hd192_prefill_kernel", ks, s, None)
}

/// MXFP8 prefill: the path (`vs128` / `t128` / `kvsplit`) and its kernels.
pub fn mxfp8(b: u64, h: u64, s: u64, sk: u64) -> Composition {
    let (path, sp) = dispatch::mxfp8_prefill_path(b, h, s, sk);
    let ks: Vec<String> = dispatch::mxfp8_prefill_kernels(path).iter().map(|k| k.to_string()).collect();
    make("mxfp8", "dispatch::mxfp8_prefill_path", ks, sp.max(1), None)
}

/// Dense GEMM of an operand kind: kernel, split-K and the reduce kernel, the CTA tile. None: unknown kind.
pub fn gemm(kind: &str, m: u64, n: u64, k: u64, cus: u64) -> Option<Composition> {
    let (kernel, s, reduce) = dispatch::gemm_kernels(kind, m, n, k, cus)?;
    let tile = gemm_tile(&kernel);
    let mut ks = vec![kernel];
    ks.extend(reduce);
    Some(make("gemm", "dispatch::gemm_kernels (bf16_gemm_config / i8_gemm_config / blockscale_config)", ks, s, Some(tile)))
}

/// MLA decode: per-wave or head-parallel form + the combine kernel, with the split count.
pub fn mla(b: u64, h: u64, page: u32) -> Composition {
    let (hp, s) = dispatch::mla_config(b, h);
    let k = format!("mla_decode_bf16_{}p{page}", if hp { "hp_" } else { "" });
    make("mla", "dispatch::mla_config", vec![k, "attn_combine_bf16_hd512".into()], s, None)
}

/// MoE: the routing-sort block size picks the `_bm64` stage kernels or the default ones. `dt`: `bf16` / `fp8` / `fp4`.
pub fn moe(dt: &str, m: u64, topk: u64, e: u64) -> Composition {
    let bm = dispatch::moe_block_m(m, topk, e);
    let sfx = if bm == 64 { "_bm64" } else { "" };
    let ks = vec![format!("gemm_moe1_{dt}_silu{sfx}"), format!("gemm_moe2_{dt}{sfx}"), format!("moe_sum_bf16_k{topk}")];
    make("moe", "dispatch::moe_block_m", ks, 1, Some((bm, 0)))
}

/// Causal attention backward: the dq and dkdv kernels (longest-first grids).
pub fn bwd_causal() -> Composition {
    let (dq, dkdv) = dispatch::bwd_causal_kernels();
    make("bwd_causal", "dispatch::bwd_causal_kernels", vec![dq.to_string(), dkdv.to_string()], 1, None)
}

impl Composition {
    /// The composition as JSON (`amdc compose`).
    pub fn to_json(&self) -> String {
        let ks: Vec<String> = self.kernels.iter().map(|k| format!("\"{k}\"")).collect();
        let tile = self.tile.map(|(r, c)| format!("[{r},{c}]")).unwrap_or_else(|| "null".into());
        let facts = match self.facts {
            None => "null".to_string(),
            Some(f) => {
                let rings: Vec<String> = f
                    .rings
                    .iter()
                    .map(|r| format!("{{\"name\":\"{}\",\"slots\":{},\"bytes_per_slot\":{},\"owner\":\"{}\"}}", r.name, r.slots, r.bytes_per_slot, if r.wave_private { "wave" } else { "cta" }))
                    .collect();
                format!("{{\"threads\":{},\"waves\":{},\"phases\":{},\"units\":{},\"rings\":[{}]}}", f.threads, f.waves, f.phases, f.units, rings.join(","))
            }
        };
        format!("{{\"family\":\"{}\",\"kernels\":[{}],\"splits\":{},\"tile\":{},\"rule\":\"{}\",\"plan\":{}}}", self.family, ks.join(","), self.splits, tile, self.rule, facts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gemm_composition_names_the_kernels_the_tile_and_the_splits() {
        let c = gemm("fp8_blockscale", 64, 8192, 8192, 256).unwrap();
        assert_eq!(c.family, "gemm");
        assert_eq!(c.tile, Some((64, 128)));
        assert!(c.kernels[0].ends_with("_64_splitk") && c.kernels.len() == 2 && c.splits > 1, "{c:?}");
        assert!(gemm("nope", 1, 1, 1, 256).is_none());
    }

    #[test]
    fn the_prefill_rules_add_the_combine_kernel_when_they_split() {
        let c = prefill_bf16(1, 8, 4096, 4096, false, 256);
        assert_eq!(c.kernels.len() == 2, c.splits > 1);
        let json = c.to_json();
        assert!(json.starts_with("{\"family\":\"prefill_bf16\"") && json.contains("\"plan\":"), "{json}");
    }
}
