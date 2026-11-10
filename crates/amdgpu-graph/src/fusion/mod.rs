//! L1 model-level op graphs, fusion rules and launch plans (docs/l1-graph.md). The input is a small JSON graph of framework ops (rmsnorm,
//! quant, rope, MoE, attention, ...). The rules (`rules.rs`) map op subgraphs onto the existing fused kernels and their dispatch rules
//! (`crate::dispatch`). The output is a deterministic launch plan (`amdc graph plan`) whose launches name the Python wrapper that executes
//! them. The layer only chooses kernel names and parameters; it never changes code generation.

pub mod rules;

use crate::opgraph::json::Json;

/// The closed op vocabulary (docs/l1-graph.md §3). Floats in attributes are JSON strings (the reader has integers only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// a + b -> c
    Add,
    /// x, gamma -> y (`eps`)
    RmsNorm,
    /// x -> q, scale: dynamic per-token, scale = max|x| / 448 (fp8 e4m3) or / 127 (int8) (`to`)
    QuantToken,
    /// x -> q, scale: dynamic per group of 128 (fp8) (`to`, `group`)
    QuantGroup,
    /// x -> q, scale: MXFP4 (e2m1 + E8M0 per 32) in the shuffled a4w4 scale layout (`to`)
    QuantMx,
    /// x -> q: static per-tensor fp8, q = x / scale (`to`, `scale`)
    QuantStatic,
    /// x [M, 2D] -> y [M, D]: act(gate) * up (`act`)
    ActMul,
    /// x, positions, cos_sin -> y (`style`)
    Rope,
    /// x, gamma -> y: per-head RMSNorm (`eps`)
    QkNorm,
    /// k, v, slot_mapping, key_cache, value_cache [, scales] -> (caches written in place; blocked fp8: k_scales, v_scales) (`layout`)
    KvCacheWrite,
    /// gating -> weights, ids (`k`, `renorm`)
    TopkSoftmax,
    /// x, w1, w2, ids, weights -> y: the two-stage MoE (`act`, `dtype`, `experts`, `inter`)
    MoeExperts,
    /// q, k, v -> o [, lse]: prefill attention, BSHD (`causal`)
    Attention,
    /// q, k_cache, v_cache, block_table, context_lens -> o: paged decode (`max_context`)
    PagedAttention,
    /// q, kv_cache, block_table, context_lens -> o: MLA decode
    MlaAttention,
}

impl OpKind {
    pub fn from_name(s: &str) -> Option<OpKind> {
        Some(match s {
            "add" => OpKind::Add,
            "rmsnorm" => OpKind::RmsNorm,
            "quant_token" => OpKind::QuantToken,
            "quant_group" => OpKind::QuantGroup,
            "quant_mx" => OpKind::QuantMx,
            "quant_static" => OpKind::QuantStatic,
            "act_mul" => OpKind::ActMul,
            "rope" => OpKind::Rope,
            "qknorm" => OpKind::QkNorm,
            "kvcache_write" => OpKind::KvCacheWrite,
            "topk_softmax" => OpKind::TopkSoftmax,
            "moe_experts" => OpKind::MoeExperts,
            "attention" => OpKind::Attention,
            "paged_attention" => OpKind::PagedAttention,
            "mla_attention" => OpKind::MlaAttention,
            _ => return None,
        })
    }
    /// (inputs min, inputs max, outputs min, outputs max)
    fn arity(self) -> (usize, usize, usize, usize) {
        match self {
            OpKind::Add => (2, 2, 1, 1),
            OpKind::RmsNorm | OpKind::QkNorm => (2, 2, 1, 1),
            OpKind::QuantToken | OpKind::QuantGroup | OpKind::QuantMx => (1, 1, 2, 2),
            OpKind::QuantStatic | OpKind::ActMul => (1, 1, 1, 1),
            OpKind::Rope => (3, 3, 1, 1),
            OpKind::KvCacheWrite => (5, 5, 0, 2),
            OpKind::TopkSoftmax => (1, 1, 2, 2),
            OpKind::MoeExperts => (5, 5, 1, 1),
            OpKind::Attention => (3, 3, 1, 2),
            OpKind::PagedAttention => (5, 5, 1, 1),
            OpKind::MlaAttention => (4, 4, 1, 1),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    N(i64),
    Sym(String),
}

#[derive(Clone, Debug)]
pub struct TensorDecl {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<Dim>,
}

#[derive(Clone, Debug)]
pub struct OpNode {
    pub id: String,
    pub kind: OpKind,
    pub ins: Vec<String>,
    pub outs: Vec<String>,
    pub attrs: Vec<(String, Json)>,
}

impl OpNode {
    pub fn attr(&self, k: &str) -> Option<&Json> {
        self.attrs.iter().find(|(n, _)| n == k).map(|(_, v)| v)
    }
    pub fn attr_str(&self, k: &str) -> Option<&str> {
        self.attr(k).and_then(|v| v.str())
    }
    pub fn attr_int(&self, k: &str) -> Option<i64> {
        self.attr(k).and_then(|v| v.int())
    }
    pub fn attr_bool(&self, k: &str) -> Option<bool> {
        self.attr(k).and_then(|v| v.bool().or_else(|| v.int().map(|i| i != 0)))
    }
}

/// A model-level op graph (docs/l1-graph.md §2). Ops are listed in a topological order (checked).
#[derive(Clone, Debug)]
pub struct ModelGraph {
    pub tensors: Vec<TensorDecl>,
    pub dims: Vec<(String, i64)>,
    pub ops: Vec<OpNode>,
    pub outputs: Vec<String>,
    pub cus: u64,
}

impl ModelGraph {
    pub fn from_json_str(s: &str) -> Result<ModelGraph, String> {
        let j = Json::parse(s)?;
        let obj = |k: &str| -> Result<&[(String, Json)], String> {
            match j.get(k) {
                Some(Json::Obj(v)) => Ok(v),
                None => Ok(&[]),
                _ => Err(format!("graph: `{k}` must be an object")),
            }
        };
        let mut tensors = Vec::new();
        for (name, t) in obj("tensors")? {
            let dtype = t.get("dtype").and_then(|d| d.str()).ok_or(format!("tensor {name}: dtype"))?.to_string();
            let shape = t
                .get("shape")
                .and_then(|s| s.arr())
                .ok_or(format!("tensor {name}: shape"))?
                .iter()
                .map(|d| match d {
                    Json::Int(i) => Ok(Dim::N(*i)),
                    Json::Str(s) => Ok(Dim::Sym(s.clone())),
                    _ => Err(format!("tensor {name}: a dim is an integer or a symbol")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            tensors.push(TensorDecl { name: name.clone(), dtype, shape });
        }
        let dims = obj("dims")?.iter().map(|(k, v)| v.int().map(|i| (k.clone(), i)).ok_or(format!("dims.{k}: integer"))).collect::<Result<Vec<_>, _>>()?;
        let strs = |v: Option<&Json>, what: &str| -> Result<Vec<String>, String> {
            match v {
                None => Ok(Vec::new()),
                Some(Json::Arr(a)) => a.iter().map(|x| x.str().map(|s| s.to_string()).ok_or(format!("{what}: strings"))).collect(),
                _ => Err(format!("{what}: an array")),
            }
        };
        let mut ops = Vec::new();
        for o in j.get("ops").and_then(|o| o.arr()).ok_or("graph: `ops` array")? {
            let id = o.get("id").and_then(|x| x.str()).ok_or("op: id")?.to_string();
            let kn = o.get("op").and_then(|x| x.str()).ok_or(format!("op {id}: op"))?;
            let kind = OpKind::from_name(kn).ok_or(format!("op {id}: unknown op `{kn}`"))?;
            let attrs = match o.get("attrs") {
                Some(Json::Obj(v)) => v.clone(),
                None => Vec::new(),
                _ => return Err(format!("op {id}: attrs must be an object")),
            };
            ops.push(OpNode { kind, ins: strs(o.get("in"), &format!("op {id}: in"))?, outs: strs(o.get("out"), &format!("op {id}: out"))?, attrs, id });
        }
        let outputs = strs(j.get("outputs"), "outputs")?;
        let cus = j.get("target").and_then(|t| t.get("cus")).and_then(|c| c.int()).unwrap_or(256) as u64;
        let g = ModelGraph { tensors, dims, ops, outputs, cus };
        g.validate()?;
        Ok(g)
    }

    fn validate(&self) -> Result<(), String> {
        let mut produced: Vec<&str> = Vec::new();
        for o in &self.ops {
            let (imin, imax, omin, omax) = o.kind.arity();
            if o.ins.len() < imin || o.ins.len() > imax || o.outs.len() < omin || o.outs.len() > omax {
                return Err(format!("op {}: {:?} takes {imin}..={imax} inputs and {omin}..={omax} outputs", o.id, o.kind));
            }
            for t in o.ins.iter().chain(o.outs.iter()) {
                if self.tensor(t).is_none() {
                    return Err(format!("op {}: undeclared tensor `{t}`", o.id));
                }
            }
            for t in &o.ins {
                if self.producer(t).is_some_and(|p| !produced.contains(&self.ops[p].id.as_str())) {
                    return Err(format!("op {}: `{t}` is produced by a later op (ops must be in topological order)", o.id));
                }
            }
            for t in &o.outs {
                if self.ops.iter().filter(|p| p.outs.contains(t)).count() > 1 {
                    return Err(format!("tensor `{t}` has more than one producer"));
                }
            }
            produced.push(&o.id);
        }
        for t in &self.outputs {
            if self.tensor(t).is_none() {
                return Err(format!("outputs: undeclared tensor `{t}`"));
            }
        }
        Ok(())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorDecl> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn dtype(&self, name: &str) -> &str {
        self.tensor(name).map(|t| t.dtype.as_str()).unwrap_or("")
    }
    /// The concrete shape (symbols resolved through `dims`).
    pub fn shape(&self, name: &str) -> Result<Vec<i64>, String> {
        let t = self.tensor(name).ok_or(format!("undeclared tensor `{name}`"))?;
        t.shape
            .iter()
            .map(|d| match d {
                Dim::N(i) => Ok(*i),
                Dim::Sym(s) => self.dims.iter().find(|(k, _)| k == s).map(|(_, v)| *v).ok_or(format!("tensor `{name}`: dim `{s}` is not bound in `dims`")),
            })
            .collect()
    }
    pub fn producer(&self, t: &str) -> Option<usize> {
        self.ops.iter().position(|o| o.outs.iter().any(|x| x == t))
    }
    /// (op index, input index) of every read of `t`
    pub fn consumers(&self, t: &str) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for (i, o) in self.ops.iter().enumerate() {
            for (k, x) in o.ins.iter().enumerate() {
                if x == t {
                    v.push((i, k));
                }
            }
        }
        v
    }
    pub fn is_output(&self, t: &str) -> bool {
        self.outputs.iter().any(|x| x == t)
    }
}

/// One planned launch group: one or more kernels, executed by one Python wrapper (`py`) with the bindings `bind` (wrapper parameter ->
/// graph tensor).
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub rule: &'static str,
    pub covers: Vec<String>,
    pub kernels: Vec<String>,
    pub py: &'static str,
    pub bind: Vec<(&'static str, String)>,
    pub params: Vec<(String, Json)>,
    pub workspaces: Vec<(String, i64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rejected {
    pub rule: &'static str,
    pub at: String,
    pub why: String,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Plan {
    pub launches: Vec<Launch>,
    pub rejected: Vec<Rejected>,
    /// ops no rule covers (no kernel exists for them alone, or the shapes are unsupported)
    pub errors: Vec<(String, String)>,
}

impl Plan {
    pub fn to_json(&self) -> Json {
        let launches = self
            .launches
            .iter()
            .map(|l| {
                Json::Obj(vec![
                    ("rule".into(), Json::Str(l.rule.into())),
                    ("covers".into(), Json::Arr(l.covers.iter().map(|c| Json::Str(c.clone())).collect())),
                    ("kernels".into(), Json::Arr(l.kernels.iter().map(|c| Json::Str(c.clone())).collect())),
                    ("py".into(), Json::Str(l.py.into())),
                    ("bind".into(), Json::Obj(l.bind.iter().map(|(k, v)| (k.to_string(), Json::Str(v.clone()))).collect())),
                    ("params".into(), Json::Obj(l.params.clone())),
                    ("workspaces".into(), Json::Arr(l.workspaces.iter().map(|(n, b)| Json::Obj(vec![("name".into(), Json::Str(n.clone())), ("bytes".into(), Json::Int(*b))])).collect())),
                ])
            })
            .collect();
        let rejected = self
            .rejected
            .iter()
            .map(|r| Json::Obj(vec![("rule".into(), Json::Str(r.rule.into())), ("at".into(), Json::Str(r.at.clone())), ("why".into(), Json::Str(r.why.clone()))]))
            .collect();
        let errors = self.errors.iter().map(|(op, why)| Json::Obj(vec![("op".into(), Json::Str(op.clone())), ("why".into(), Json::Str(why.clone()))])).collect();
        Json::Obj(vec![("launches".into(), Json::Arr(launches)), ("rejected".into(), Json::Arr(rejected)), ("errors".into(), Json::Arr(errors))])
    }
}

/// A rule step: the op kind and its links to the previous step (previous step's output index, this step's input index).
pub struct Step {
    pub kind: OpKind,
    pub links: &'static [(usize, usize)],
}

/// A fusion rule (docs/l1-graph.md §4): a chain pattern, the interior tensors the kernel still writes (`exposes`: (step, output index)), and
/// the plan function (Err = the shapes / dtypes / attributes are outside the fused kernel's supported set; the reason is reported).
pub struct Rule {
    pub name: &'static str,
    pub steps: &'static [Step],
    pub exposes: &'static [(usize, usize)],
    pub plan: fn(&ModelGraph, &[usize]) -> Result<Launch, String>,
}

/// Plans the graph: maximal munch over the ops in order (longest rules first), then the single-op rules for the rest.
pub fn plan(g: &ModelGraph) -> Plan {
    plan_with(g, true)
}

/// `plan` with the fused rules switched off (`fuse = false`): every op through its single-op rule (the unfused reference plan of the
/// numerics check, docs/l1-graph.md §7.3).
pub fn plan_with(g: &ModelGraph, fuse: bool) -> Plan {
    let mut rs: Vec<&Rule> = if fuse { rules::FUSED.iter().collect() } else { Vec::new() };
    rs.sort_by_key(|r| std::cmp::Reverse(r.steps.len()));
    let mut covered = vec![false; g.ops.len()];
    let mut out = Plan::default();
    let mut launches: Vec<(usize, Launch)> = Vec::new();
    for s in 0..g.ops.len() {
        if covered[s] {
            continue;
        }
        for r in &rs {
            match match_rule(g, r, s, &covered) {
                Ok(None) => {}
                Ok(Some(m)) => match (r.plan)(g, &m) {
                    Ok(l) => {
                        for &i in &m {
                            covered[i] = true;
                        }
                        launches.push((s, l));
                        break;
                    }
                    Err(why) => out.rejected.push(Rejected { rule: r.name, at: g.ops[s].id.clone(), why }),
                },
                Err(why) => out.rejected.push(Rejected { rule: r.name, at: g.ops[s].id.clone(), why }),
            }
        }
    }
    for (i, o) in g.ops.iter().enumerate() {
        if !covered[i] {
            match rules::single(g, i) {
                Ok(l) => launches.push((i, l)),
                Err(why) => out.errors.push((o.id.clone(), why)),
            }
        }
    }
    launches.sort_by_key(|(i, _)| *i);
    out.launches = launches.into_iter().map(|(_, l)| l).collect();
    out
}

/// Ok(None): the pattern does not start here; Ok(Some(ops)): a structural match; Err: structural match with an interior tensor that escapes.
fn match_rule(g: &ModelGraph, r: &Rule, s: usize, covered: &[bool]) -> Result<Option<Vec<usize>>, String> {
    if g.ops[s].kind != r.steps[0].kind {
        return Ok(None);
    }
    let mut m = vec![s];
    for (si, st) in r.steps.iter().enumerate().skip(1) {
        let prev = &g.ops[m[si - 1]];
        let (po, pi) = st.links[0];
        let Some(t) = prev.outs.get(po) else { return Ok(None) };
        let cand = g.consumers(t).into_iter().find(|&(c, k)| {
            k == pi && !covered[c] && g.ops[c].kind == st.kind && st.links.iter().all(|&(o2, i2)| prev.outs.get(o2).is_some_and(|x| g.ops[c].ins.get(i2) == Some(x)))
        });
        let Some((c, _)) = cand else { return Ok(None) };
        m.push(c);
    }
    // interior tensors: read only inside the match and not graph outputs, unless the rule exposes them
    for (si, st) in r.steps.iter().enumerate().skip(1) {
        for &(po, _) in st.links {
            if r.exposes.contains(&(si - 1, po)) {
                continue;
            }
            let t = &g.ops[m[si - 1]].outs[po];
            if g.is_output(t) {
                return Err(format!("interior tensor `{t}` is a graph output"));
            }
            if let Some(&(c, _)) = g.consumers(t).iter().find(|(c, _)| !m.contains(c)) {
                return Err(format!("interior tensor `{t}` is also read by op {}", g.ops[c].id));
            }
        }
    }
    Ok(Some(m))
}

/// `amdc graph plan`: the plan of a JSON graph as pretty JSON.
pub fn plan_json(text: &str) -> Result<(String, bool), String> {
    let g = ModelGraph::from_json_str(text)?;
    let p = plan(&g);
    Ok((p.to_json().to_string_pretty(), p.errors.is_empty()))
}
