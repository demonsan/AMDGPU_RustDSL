//! The fusion rules of docs/l1-graph.md §5 and the single-op rules. Kernel names use the format strings of the Python wrappers
//! (`python/amdgpu_jit/ops_*.py`). The supported sets mirror the families' `by_name` parsers in `amdgpu-kernels`; the kernels crate checks the
//! mirror (`tests/graph_plan_names.rs`: every name the rules produce over the canonical shapes builds). Per-shape choices (key splits, block_m,
//! the decode form) come from `crate::dispatch`, never from here.

use super::{Launch, ModelGraph, OpKind, Rule, Step};
use crate::dispatch;
use crate::opgraph::json::Json;

type R<T> = Result<T, String>;

fn last(g: &ModelGraph, t: &str) -> R<i64> {
    g.shape(t)?.last().copied().ok_or(format!("`{t}` is a scalar"))
}
fn rows(g: &ModelGraph, t: &str) -> R<i64> {
    let s = g.shape(t)?;
    Ok(s[..s.len().saturating_sub(1)].iter().product())
}
fn need(cond: bool, why: impl FnOnce() -> String) -> R<()> {
    if cond { Ok(()) } else { Err(why()) }
}
fn dt(g: &ModelGraph, t: &str, want: &str) -> R<()> {
    need(g.dtype(t) == want, || format!("`{t}` is {} (the kernel takes {want})", g.dtype(t)))
}
fn p_int(k: &str, v: i64) -> (String, Json) {
    (k.into(), Json::Int(v))
}
fn p_str(k: &str, v: &str) -> (String, Json) {
    (k.into(), Json::Str(v.into()))
}
/// `eps` of a norm op, passed through (a JSON string, e.g. "1e-6"); absent = the wrapper's default
fn eps(g: &ModelGraph, op: usize) -> Vec<(String, Json)> {
    g.ops[op].attr("eps").map(|e| vec![("eps".to_string(), e.clone())]).unwrap_or_default()
}
fn launch(rule: &'static str, g: &ModelGraph, m: &[usize], kernels: Vec<String>, py: &'static str, bind: Vec<(&'static str, String)>, params: Vec<(String, Json)>) -> Launch {
    Launch { rule, covers: m.iter().map(|&i| g.ops[i].id.clone()).collect(), kernels, py, bind, params, workspaces: Vec::new() }
}
fn quant_to<'a>(g: &'a ModelGraph, op: usize) -> &'a str {
    g.ops[op].attr_str("to").unwrap_or("fp8")
}
/// row widths of the row-kernel families (rows/mod.rs, rows/quant.rs: N % 8 == 0, 8 ..= 8192)
fn row_n(n: i64) -> R<()> {
    need(n % 8 == 0 && (8..=8192).contains(&n), || format!("row width {n} outside the built set (multiple of 8, 8 ..= 8192)"))
}
fn style(g: &ModelGraph, op: usize) -> R<&'static str> {
    match g.ops[op].attr_str("style").unwrap_or("neox") {
        "neox" => Ok("neox"),
        "gptj" => Ok("gptj"),
        s => Err(format!("rope style `{s}` (neox | gptj)")),
    }
}
/// [T, H, D] of a head tensor with the rope family's supported set (rope/mod.rs)
fn thd(g: &ModelGraph, t: &str) -> R<(i64, i64, i64)> {
    let s = g.shape(t)?;
    need(s.len() == 3, || format!("`{t}` must be [T, H, D]"))?;
    let (h, d) = (s[1], s[2]);
    need((1..=256).contains(&h) && [64, 128, 256].contains(&d) && 2 * h * d <= 65536, || format!("heads {h} x dim {d} outside the rope family (H <= 256, D in 64 / 128 / 256, 2 H D <= 65536)"))?;
    Ok((s[0], h, d))
}

// ------------------------------------------------------------------------------------------------------------------------------ fused rules

pub static FUSED: &[Rule] = &[
    Rule {
        name: "add+rmsnorm+quant_token.fp8",
        steps: &[Step { kind: OpKind::Add, links: &[] }, Step { kind: OpKind::RmsNorm, links: &[(0, 0)] }, Step { kind: OpKind::QuantToken, links: &[(0, 0)] }],
        exposes: &[(0, 0)],
        plan: |g, m| {
            let (a, n, q) = (&g.ops[m[0]], &g.ops[m[1]], &g.ops[m[2]]);
            need(quant_to(g, m[2]) == "fp8", || format!("quant_token to {} (the fused kernel quantizes to fp8)", quant_to(g, m[2])))?;
            dt(g, &a.ins[0], "bf16")?;
            let nn = last(g, &a.ins[0])?;
            row_n(nn)?;
            let bind = vec![("x", a.ins[0].clone()), ("res", a.ins[1].clone()), ("gamma", n.ins[1].clone()), ("out", q.outs[0].clone()), ("res_out", a.outs[0].clone()), ("scale", q.outs[1].clone())];
            Ok(launch("add+rmsnorm+quant_token.fp8", g, m, vec![format!("add_rmsnorm_fp8_n{nn}")], "add_rmsnorm_quant_fp8", bind, eps(g, m[1])))
        },
    },
    Rule {
        name: "qknorm+rope+kvcache",
        steps: &[Step { kind: OpKind::QkNorm, links: &[] }, Step { kind: OpKind::Rope, links: &[(0, 0)] }, Step { kind: OpKind::KvCacheWrite, links: &[(0, 0)] }],
        exposes: &[(1, 0)],
        plan: |g, m| {
            let (n, r, c) = (&g.ops[m[0]], &g.ops[m[1]], &g.ops[m[2]]);
            need(c.attr_str("layout").unwrap_or("flash") == "flash", || "the fused kernel writes the flash cache layout".into())?;
            dt(g, &n.ins[0], "bf16")?;
            let (_, h, d) = thd(g, &n.ins[0])?;
            let st = style(g, m[1])?;
            let fp8 = g.dtype(&c.ins[3]) == "fp8";
            let i64s = g.dtype(&c.ins[2]) == "i64";
            let mut params = eps(g, m[0]);
            params.push(p_str("style", st));
            if fp8 {
                let (ks, vs) = (c.attr("k_scale").cloned(), c.attr("v_scale").cloned());
                need(ks.is_some() && vs.is_some(), || "an fp8 cache needs the static k_scale / v_scale attributes".into())?;
                params.push(("k_scale".into(), ks.unwrap()));
                params.push(("v_scale".into(), vs.unwrap()));
            }
            let name = format!("qknorm_rope_kvcache_{st}_{}_h{h}_d{d}{}", if fp8 { "fp8" } else { "bf16" }, if i64s { "_i64" } else { "" });
            let bind = vec![("k", n.ins[0].clone()), ("out", r.outs[0].clone()), ("positions", r.ins[1].clone()), ("cos_sin", r.ins[2].clone()), ("gamma", n.ins[1].clone()), ("v", c.ins[1].clone()), ("key_cache", c.ins[3].clone()), ("value_cache", c.ins[4].clone()), ("slot_mapping", c.ins[2].clone())];
            Ok(launch("qknorm+rope+kvcache", g, m, vec![name], "qknorm_rope_kvcache", bind, params))
        },
    },
    Rule {
        name: "add+rmsnorm",
        steps: &[Step { kind: OpKind::Add, links: &[] }, Step { kind: OpKind::RmsNorm, links: &[(0, 0)] }],
        exposes: &[(0, 0)],
        plan: |g, m| {
            let (a, n) = (&g.ops[m[0]], &g.ops[m[1]]);
            dt(g, &a.ins[0], "bf16")?;
            let nn = last(g, &a.ins[0])?;
            row_n(nn)?;
            let bind = vec![("x", a.ins[0].clone()), ("res", a.ins[1].clone()), ("gamma", n.ins[1].clone()), ("out", n.outs[0].clone()), ("res_out", a.outs[0].clone())];
            Ok(launch("add+rmsnorm", g, m, vec![format!("add_rmsnorm_bf16_n{nn}")], "add_rmsnorm", bind, eps(g, m[1])))
        },
    },
    Rule {
        name: "rmsnorm+quant_token.fp8",
        steps: &[Step { kind: OpKind::RmsNorm, links: &[] }, Step { kind: OpKind::QuantToken, links: &[(0, 0)] }],
        exposes: &[],
        plan: |g, m| {
            let (n, q) = (&g.ops[m[0]], &g.ops[m[1]]);
            need(quant_to(g, m[1]) == "fp8", || format!("quant_token to {} (the fused kernel quantizes to fp8)", quant_to(g, m[1])))?;
            dt(g, &n.ins[0], "bf16")?;
            let nn = last(g, &n.ins[0])?;
            need(nn % 512 == 0 && (512..=8192).contains(&nn), || format!("row width {nn} outside rmsnorm_quant_fp8 (multiple of 512, 512 ..= 8192)"))?;
            let bind = vec![("x", n.ins[0].clone()), ("gamma", n.ins[1].clone()), ("q", q.outs[0].clone()), ("scale", q.outs[1].clone())];
            Ok(launch("rmsnorm+quant_token.fp8", g, m, vec![format!("rmsnorm_quant_fp8_n{nn}")], "rmsnorm_quant_fp8", bind, eps(g, m[0])))
        },
    },
    Rule {
        name: "rmsnorm+quant_mx.fp4",
        steps: &[Step { kind: OpKind::RmsNorm, links: &[] }, Step { kind: OpKind::QuantMx, links: &[(0, 0)] }],
        exposes: &[],
        plan: |g, m| {
            let (n, q) = (&g.ops[m[0]], &g.ops[m[1]]);
            need(quant_to(g, m[1]) == "mxfp4", || format!("quant_mx to {} (the fused kernel quantizes to mxfp4)", quant_to(g, m[1])))?;
            dt(g, &n.ins[0], "bf16")?;
            let nn = last(g, &n.ins[0])?;
            need(nn % 512 == 0 && (512..=16384).contains(&nn), || format!("row width {nn} outside rmsnorm_quant_mxfp4 (multiple of 512, 512 ..= 16384)"))?;
            let mm = rows(g, &n.ins[0])?;
            need(mm % 16 == 0, || format!("{mm} rows: the shuffled scale layout needs a multiple of 16"))?;
            let bind = vec![("x", n.ins[0].clone()), ("gamma", n.ins[1].clone()), ("q", q.outs[0].clone()), ("scale", q.outs[1].clone())];
            Ok(launch("rmsnorm+quant_mx.fp4", g, m, vec![format!("rmsnorm_quant_mxfp4_bf16_k{nn}")], "rmsnorm_quant_mxfp4", bind, eps(g, m[0])))
        },
    },
    Rule {
        name: "act_mul.silu+quant_token.fp8",
        steps: &[Step { kind: OpKind::ActMul, links: &[] }, Step { kind: OpKind::QuantToken, links: &[(0, 0)] }],
        exposes: &[],
        plan: |g, m| {
            let (a, q) = (&g.ops[m[0]], &g.ops[m[1]]);
            need(a.attr_str("act").unwrap_or("silu") == "silu", || format!("act {} (the fused kernel is silu)", a.attr_str("act").unwrap_or("")))?;
            need(quant_to(g, m[1]) == "fp8", || format!("quant_token to {} (the fused kernel quantizes to fp8)", quant_to(g, m[1])))?;
            dt(g, &a.ins[0], "bf16")?;
            let d = last(g, &a.ins[0])? / 2;
            row_n(d)?;
            let bind = vec![("x", a.ins[0].clone()), ("q", q.outs[0].clone()), ("scale", q.outs[1].clone())];
            Ok(launch("act_mul.silu+quant_token.fp8", g, m, vec![format!("silu_mul_quant_fp8_d{d}")], "silu_mul_quant_fp8", bind, Vec::new()))
        },
    },
    Rule {
        name: "qknorm+rope",
        steps: &[Step { kind: OpKind::QkNorm, links: &[] }, Step { kind: OpKind::Rope, links: &[(0, 0)] }],
        exposes: &[],
        plan: |g, m| {
            let (n, r) = (&g.ops[m[0]], &g.ops[m[1]]);
            dt(g, &n.ins[0], "bf16")?;
            let (_, h, d) = thd(g, &n.ins[0])?;
            let st = style(g, m[1])?;
            let mut params = eps(g, m[0]);
            params.push(p_str("style", st));
            let bind = vec![("x", n.ins[0].clone()), ("out", r.outs[0].clone()), ("positions", r.ins[1].clone()), ("cos_sin", r.ins[2].clone()), ("gamma", n.ins[1].clone())];
            Ok(launch("qknorm+rope", g, m, vec![format!("qknorm_rope_{st}_bf16_h{h}_d{d}")], "qknorm_rope", bind, params))
        },
    },
];

// ------------------------------------------------------------------------------------------------------------------------------ single ops

/// The single-op rule of op `i` (every op of the vocabulary that has a kernel of its own).
pub fn single(g: &ModelGraph, i: usize) -> R<Launch> {
    let o = &g.ops[i];
    let m = [i];
    let one = |rule: &'static str, k: String, py: &'static str, bind: Vec<(&'static str, String)>, params: Vec<(String, Json)>| Ok(launch(rule, g, &m, vec![k], py, bind, params));
    match o.kind {
        OpKind::Add => Err("no standalone add kernel (fuses as add+rmsnorm)".into()),
        OpKind::QkNorm => Err("no standalone qknorm kernel (fuses as qknorm+rope)".into()),
        OpKind::QuantMx => Err("no standalone MXFP4 quant kernel in the Python surface (fuses as rmsnorm+quant_mx)".into()),
        OpKind::RmsNorm => {
            dt(g, &o.ins[0], "bf16")?;
            let n = last(g, &o.ins[0])?;
            row_n(n)?;
            one("rmsnorm", format!("rmsnorm_bf16_n{n}"), "rmsnorm", vec![("x", o.ins[0].clone()), ("gamma", o.ins[1].clone()), ("out", o.outs[0].clone())], eps(g, i))
        }
        OpKind::QuantToken => {
            dt(g, &o.ins[0], "bf16")?;
            let to = quant_to(g, i);
            need(to == "fp8" || to == "int8", || format!("quant_token to {to} (fp8 | int8)"))?;
            let n = last(g, &o.ins[0])?;
            row_n(n)?;
            one("quant_token", format!("quant_token_bf16_{to}_n{n}"), "quant_per_token", vec![("x", o.ins[0].clone()), ("q", o.outs[0].clone()), ("scale", o.outs[1].clone())], vec![p_str("dtype", to)])
        }
        OpKind::QuantGroup => {
            dt(g, &o.ins[0], "bf16")?;
            need(quant_to(g, i) == "fp8" && o.attr_int("group").unwrap_or(128) == 128, || "quant_group: fp8, groups of 128".into())?;
            let n = last(g, &o.ins[0])?;
            need(n % 128 == 0 && (128..=8192).contains(&n), || format!("row width {n} outside quant_group128 (multiple of 128, <= 8192)"))?;
            one("quant_group", format!("quant_group128_bf16_fp8_n{n}"), "quant_per_group128", vec![("x", o.ins[0].clone()), ("q", o.outs[0].clone()), ("scale", o.outs[1].clone())], Vec::new())
        }
        OpKind::QuantStatic => {
            dt(g, &o.ins[0], "bf16")?;
            let s = o.attr("scale").cloned().ok_or("quant_static: the `scale` attribute")?;
            one("quant_static", "quant_bf16_fp8_static".into(), "quant_static_fp8", vec![("x", o.ins[0].clone()), ("out", o.outs[0].clone())], vec![("scale".into(), s)])
        }
        OpKind::ActMul => {
            dt(g, &o.ins[0], "bf16")?;
            let act = o.attr_str("act").unwrap_or("silu");
            let k = match act {
                "silu" => "silu_mul",
                "gelu" => "gelu_mul",
                "gelu_tanh" => "gelu_tanh_mul",
                "swiglu_oai" => "swiglu_oai_mul",
                a => return Err(format!("act `{a}` (silu | gelu | gelu_tanh | swiglu_oai)")),
            };
            let d = last(g, &o.ins[0])? / 2;
            need(d % 8 == 0 && (8..=65536).contains(&d), || format!("width {d} outside act_mul (multiple of 8, <= 65536)"))?;
            one("act_mul", format!("{k}_bf16_d{d}"), "act_mul", vec![("x", o.ins[0].clone()), ("out", o.outs[0].clone())], vec![p_str("act", act)])
        }
        OpKind::Rope => {
            dt(g, &o.ins[0], "bf16")?;
            let (_, h, d) = thd(g, &o.ins[0])?;
            let st = style(g, i)?;
            let bind = vec![("x", o.ins[0].clone()), ("out", o.outs[0].clone()), ("positions", o.ins[1].clone()), ("cos_sin", o.ins[2].clone())];
            one("rope", format!("rope_{st}_bf16_h{h}_d{d}"), "rope", bind, vec![p_str("style", st)])
        }
        OpKind::KvCacheWrite => {
            dt(g, &o.ins[0], "bf16")?;
            let (_, h, d) = thd(g, &o.ins[0])?;
            let fp8 = g.dtype(&o.ins[3]) == "fp8";
            let i64s = g.dtype(&o.ins[2]) == "i64";
            let layout = o.attr_str("layout").unwrap_or("flash");
            let mut bind = vec![("k", o.ins[0].clone()), ("v", o.ins[1].clone()), ("key_cache", o.ins[3].clone()), ("value_cache", o.ins[4].clone()), ("slot_mapping", o.ins[2].clone())];
            let mut params = vec![p_str("layout", layout)];
            let dt_s = if fp8 { "fp8" } else { "bf16" };
            let name = match layout {
                "flash" => {
                    need(!i64s, || "the flash cache write takes int32 slot_mapping".into())?;
                    if fp8 {
                        for k in ["k_scale", "v_scale"] {
                            params.push((k.into(), o.attr(k).cloned().ok_or(format!("an fp8 flash cache needs the static `{k}` attribute"))?));
                        }
                    }
                    format!("reshape_cache_flash_{dt_s}_h{h}_d{d}")
                }
                "blocked" => {
                    let s = g.shape(&o.ins[3])?;
                    need(s.len() == 5, || "blocked key_cache [NB, H, D/x, BS, x]".into())?;
                    let bs = s[3];
                    need([16, 32, 64, 128].contains(&bs) || (!fp8 && bs == 8), || format!("block size {bs} outside the blocked cache write"))?;
                    if fp8 {
                        need(o.outs.len() == 2, || "an fp8 blocked cache writes per-token k_scales / v_scales (two outputs)".into())?;
                        bind.push(("k_scales", o.outs[0].clone()));
                        bind.push(("v_scales", o.outs[1].clone()));
                    }
                    format!("reshape_cache_blocked_{dt_s}_h{h}_d{d}_b{bs}{}", if i64s { "_i64" } else { "" })
                }
                l => return Err(format!("cache layout `{l}` (flash | blocked)")),
            };
            one("kvcache_write", name, "reshape_cache", bind, params)
        }
        OpKind::TopkSoftmax => {
            dt(g, &o.ins[0], "bf16")?;
            let e = last(g, &o.ins[0])?;
            let k = last(g, &o.outs[0])?;
            need(e % 8 == 0 && (32..=512).contains(&e) && (1..=16).contains(&k) && k <= e, || format!("E {e}, k {k} outside topk_softmax"))?;
            let renorm = o.attr_bool("renorm").unwrap_or(false);
            let bind = vec![("gating", o.ins[0].clone()), ("weights", o.outs[0].clone()), ("indices", o.outs[1].clone())];
            one("topk_softmax", format!("topk_softmax{}_bf16_e{e}_k{k}", if renorm { "_renorm" } else { "" }), "topk_softmax", bind, vec![("renorm".into(), Json::Bool(renorm))])
        }
        OpKind::MoeExperts => moe(g, i),
        OpKind::Attention => attention(g, i),
        OpKind::PagedAttention => paged(g, i),
        OpKind::MlaAttention => mla(g, i),
    }
}

/// The two-stage MoE flow of `fused_moe` (docs/l1-graph.md §5.3): sorting, stage 1, the fp8 re-quantization, stage 2, the top-k sum.
fn moe(g: &ModelGraph, i: usize) -> R<Launch> {
    let o = &g.ops[i];
    let xs = g.shape(&o.ins[0])?;
    need(xs.len() == 2, || "moe_experts: x [M, D]".into())?;
    let (m, d) = (xs[0], xs[1]);
    let k = last(g, &o.ins[3])?;
    let e = o.attr_int("experts").ok_or("moe_experts: the `experts` attribute (weights are preshuffled bytes)")?;
    let inter = o.attr_int("inter").ok_or("moe_experts: the `inter` attribute")?;
    let fp8 = match o.attr_str("dtype").unwrap_or("bf16") {
        "bf16" => false,
        "fp8" => true,
        x => return Err(format!("moe dtype `{x}` (bf16 | fp8)")),
    };
    let act = o.attr_str("act").unwrap_or("silu");
    need(act == "silu" || act == "gelu", || format!("moe act `{act}` (silu | gelu)"))?;
    need(e >= 8 && e <= 512 && (1..=16).contains(&k) && d % 1024 == 0 && inter % 128 == 0, || format!("moe shape E {e}, k {k}, D {d}, I {inter} outside the kernels (E 8 ..= 512, k <= 16, D % 1024, I % 128)"))?;
    need(!fp8 || (m * k) % 4 == 0, || "fp8 moe: M k must be a multiple of 4".into())?;
    let bm = dispatch::moe_block_m(m as u64, k as u64, e as u64) as i64;
    let (dt_s, bms) = (if fp8 { "fp8" } else { "bf16" }, if bm == 64 { "_bm64" } else { "" });
    let mut kernels = vec![format!("moe_sorting_e{e}_k{k}_b{bm}"), format!("gemm_moe1_{dt_s}_{act}{bms}")];
    if fp8 {
        kernels.push(format!("quant_token_bf16_fp8_n{inter}"));
    }
    kernels.push(format!("gemm_moe2_{dt_s}{bms}"));
    kernels.push(format!("moe_sum_bf16_k{k}"));
    let maxp = m * k + e * bm - k;
    let ntile = (maxp + bm - 1) / bm;
    let mut ws = vec![("moe_sid".to_string(), maxp * 4), ("moe_sw".into(), maxp * 4), ("moe_seid".into(), ntile * 4), ("moe_nv".into(), 8), ("moe_ts".into(), (e + 1) * 4), ("moe_y1".into(), m * k * inter * 2)];
    if fp8 {
        ws.push(("moe_y1q".into(), m * k * inter));
        ws.push(("moe_y1s".into(), m * k * 4));
    }
    ws.push(("moe_t".into(), m * k * d * 4));
    let bind = vec![("x", o.ins[0].clone()), ("w1", o.ins[1].clone()), ("w2", o.ins[2].clone()), ("topk_ids", o.ins[3].clone()), ("topk_weights", o.ins[4].clone()), ("out", o.outs[0].clone())];
    let params = vec![p_str("dtype", dt_s), p_str("act", act), p_int("num_experts", e), p_int("inter", inter), p_int("block_m", bm)];
    let mut l = launch("moe_experts", g, &[i], kernels, "fused_moe", bind, params);
    l.workspaces = ws;
    Ok(l)
}

/// Prefill attention (bf16, hd128, BSHD): the kernel and key splits from `dispatch::bf16_prefill_kernel`; S > 1 adds the combine.
fn attention(g: &ModelGraph, i: usize) -> R<Launch> {
    let o = &g.ops[i];
    for t in &o.ins {
        dt(g, t, "bf16")?;
    }
    let (q, k) = (g.shape(&o.ins[0])?, g.shape(&o.ins[1])?);
    need(q.len() == 4 && k.len() == 4 && q[3] == 128 && k[3] == 128, || "attention: q [B, SQ, H, 128], k / v [B, SK, HK, 128]".into())?;
    let (b, sq, h, sk, hk) = (q[0], q[1], q[2], k[1], k[2]);
    need(h % hk == 0 && (h / hk > 0 && ((h / hk) & (h / hk - 1)) == 0) && h / hk <= 16, || format!("GQA group H / HK = {h} / {hk}: a power of two <= 16"))?;
    let causal = o.attr_bool("causal").unwrap_or(false);
    let (name, s) = dispatch::bf16_prefill_kernel(b as u64, h as u64, sq as u64, sk as u64, causal, g.cus);
    let mut kernels = vec![name.to_string()];
    let mut ws = Vec::new();
    if s > 1 {
        kernels.push("attn_combine_bf16_hd128_q16".into());
        ws.push(("prefill_po".to_string(), b * h * sq * s as i64 * 128 * 4));
        ws.push(("prefill_plse".into(), b * h * sq * s as i64 * 4));
    }
    let mut bind = vec![("q", o.ins[0].clone()), ("k", o.ins[1].clone()), ("v", o.ins[2].clone()), ("out", o.outs[0].clone())];
    if let Some(l) = o.outs.get(1) {
        bind.push(("lse", l.clone()));
    }
    let mut l = launch("attention", g, &[i], kernels, "prefill_bf16", bind, vec![("causal".into(), Json::Bool(causal)), p_int("splits", s as i64), p_int("cus", g.cus as i64)]);
    l.workspaces = ws;
    Ok(l)
}

/// Paged decode (bf16, hd128): the form and splits from `dispatch::decode16_plan` / `decode16_kernel`.
fn paged(g: &ModelGraph, i: usize) -> R<Launch> {
    let o = &g.ops[i];
    let (q, kc, bt) = (g.shape(&o.ins[0])?, g.shape(&o.ins[1])?, g.shape(&o.ins[3])?);
    need(q.len() == 3 && q[2] == 128 && kc.len() == 4 && kc[3] == 128 && bt.len() == 2, || "paged_attention: q [B, H, 128], caches [NP, P, HK, 128], block_table [B, MP]".into())?;
    let (b, h, p, hk) = (q[0], q[1], kc[1], kc[2]);
    need(h % hk == 0 && h / hk <= 16, || format!("GQA group H / HK = {h} / {hk} <= 16"))?;
    need(p >= 16 && (p & (p - 1)) == 0, || format!("page size {p}: a power of two >= 16"))?;
    let ctx = o.attr_int("max_context").unwrap_or(bt[1] * p);
    let pl = dispatch::decode16_plan(b as u64, hk as u64, ctx as u64);
    let (name, s) = dispatch::decode16_kernel(p as u32, pl);
    let form = match pl {
        dispatch::Decode16Plan::Fused => "fused",
        dispatch::Decode16Plan::LastC(_) => "lastc",
    };
    let bind = vec![("q", o.ins[0].clone()), ("k_cache", o.ins[1].clone()), ("v_cache", o.ins[2].clone()), ("block_table", o.ins[3].clone()), ("context_lens", o.ins[4].clone()), ("out", o.outs[0].clone())];
    Ok(launch("paged_attention", g, &[i], vec![name], "paged_decode_bf16", bind, vec![p_str("form", form), p_int("splits", s as i64), p_int("max_context", ctx)]))
}

/// MLA decode: the form and splits from `dispatch::mla_config`, then the hd512 combine.
fn mla(g: &ModelGraph, i: usize) -> R<Launch> {
    let o = &g.ops[i];
    let (q, kv) = (g.shape(&o.ins[0])?, g.shape(&o.ins[1])?);
    need(q.len() == 3 && q[2] == 576 && kv.len() == 3 && kv[2] == 576, || "mla_attention: q [B, H, 576], kv_cache [NP, P, 576]".into())?;
    let (b, h, p) = (q[0], q[1], kv[1]);
    let (hp, s) = dispatch::mla_config(b as u64, h as u64);
    let parts = if hp { s as i64 } else { 4 * s as i64 };
    let bind = vec![("q", o.ins[0].clone()), ("kv_cache", o.ins[1].clone()), ("block_table", o.ins[2].clone()), ("context_lens", o.ins[3].clone()), ("out", o.outs[0].clone())];
    let mut l = launch("mla_attention", g, &[i], vec![format!("mla_decode_bf16_{}p{p}", if hp { "hp_" } else { "" }), "attn_combine_bf16_hd512".into()], "mla_decode", bind, vec![("hp".into(), Json::Bool(hp)), p_int("splits", s as i64)]);
    l.workspaces = vec![("mla_po".into(), b * h * parts * 512 * 4), ("mla_plse".into(), b * h * parts * 4)];
    Ok(l)
}
