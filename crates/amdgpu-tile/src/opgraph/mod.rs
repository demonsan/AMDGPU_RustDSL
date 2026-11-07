//! The lightweight OP GRAPH of the stage library (docs/fusion-stage-library-design.md section 3): plain Rust data, JSON-serialisable, NOT an IR. It is the input of the skeleton
//! generator (`skeleton.rs`) and never enters the compiler.
//!
//! * nodes = stage instances: a primitive (M / A / E / R / O / Q / S / P), the registry key of the stage (`amdgpu_tile::stage`), its parameters, a resource estimate and the wave roles;
//! * edges = tensors: dtype / shape, the producer's and the consumer's layout as element tables (when both are known the boundary cost class is COMPUTED with the layout algebra), the largest
//!   on-chip scope the tensor may stay in, its storage, and the boundary cost class C0..C4 (`Edge::class`);
//! * a CUT partitions the nodes into kernels; an edge between two kernels is C4 (HBM); the budgets of a kernel (registers, LDS) and the scope requirements are checked per kernel.

pub mod json;
pub mod skeleton;

use amdgpu_layout::{plan_convert_tables, Movement};
use json::Json;

/// The compute primitives of the library (design section 1.1; the communication primitive X is out of scope).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prim {
    /// Move: global / LDS / register tile moves
    M,
    /// MmaLoop
    A,
    /// Map: element-wise
    E,
    /// Reduce
    R,
    /// Online reduction with a rescaled consumer
    O,
    /// Quantize
    Q,
    /// Select (sort / top-k)
    S,
    /// Scan / recurrence
    P,
}

impl Prim {
    pub fn code(self) -> &'static str {
        match self {
            Prim::M => "M",
            Prim::A => "A",
            Prim::E => "E",
            Prim::R => "R",
            Prim::O => "O",
            Prim::Q => "Q",
            Prim::S => "S",
            Prim::P => "P",
        }
    }
    pub fn from_code(s: &str) -> Option<Prim> {
        Some(match s {
            "M" => Prim::M,
            "A" => Prim::A,
            "E" => Prim::E,
            "R" => Prim::R,
            "O" => Prim::O,
            "Q" => Prim::Q,
            "S" => Prim::S,
            "P" => Prim::P,
            _ => return None,
        })
    }
}

/// Boundary cost class of an edge (design section 2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// every lane already holds its elements (register rename; the consumer's K permutation may be absorbed by the other operand)
    C0,
    /// quad-local re-arrangement (DPP quad_perm + cndmask)
    C1,
    /// cross-lane within a wave (wave-private LDS round trip or permlane)
    C2,
    /// needs other waves' data: LDS write + barrier + read
    C3,
    /// not fused: through HBM
    C4,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::C0 => "C0",
            Class::C1 => "C1",
            Class::C2 => "C2",
            Class::C3 => "C3",
            Class::C4 => "C4",
        }
    }
    pub fn from_name(s: &str) -> Option<Class> {
        Some(match s {
            "C0" => Class::C0,
            "C1" => Class::C1,
            "C2" => Class::C2,
            "C3" => Class::C3,
            "C4" => Class::C4,
            _ => return None,
        })
    }
}

/// The largest on-chip scope a tensor may stay in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scope {
    /// registers of one wave
    Wave,
    /// LDS of one block
    Block,
    /// across blocks (atomics / two passes): the edge must be cut or needs grid-level cooperation
    Grid,
    /// must go through HBM
    None,
}

impl Scope {
    pub fn name(self) -> &'static str {
        match self {
            Scope::Wave => "wave",
            Scope::Block => "block",
            Scope::Grid => "grid",
            Scope::None => "none",
        }
    }
    pub fn from_name(s: &str) -> Option<Scope> {
        Some(match s {
            "wave" => Scope::Wave,
            "block" => Scope::Block,
            "grid" => Scope::Grid,
            "none" => Scope::None,
            _ => return None,
        })
    }
}

/// A parameter value of a stage instance.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Int(i64),
    Str(String),
    Bool(bool),
    List(Vec<Val>),
}

impl Val {
    fn to_json(&self) -> Json {
        match self {
            Val::Int(i) => Json::Int(*i),
            Val::Str(s) => Json::Str(s.clone()),
            Val::Bool(b) => Json::Bool(*b),
            Val::List(v) => Json::Arr(v.iter().map(|x| x.to_json()).collect()),
        }
    }
    fn from_json(j: &Json) -> Result<Val, String> {
        Ok(match j {
            Json::Int(i) => Val::Int(*i),
            Json::Str(s) => Val::Str(s.clone()),
            Json::Bool(b) => Val::Bool(*b),
            Json::Arr(v) => Val::List(v.iter().map(Val::from_json).collect::<Result<_, _>>()?),
            _ => return Err("param: expected int / string / bool / list".into()),
        })
    }
    /// Rust literal for generated code.
    pub fn rust(&self) -> String {
        match self {
            Val::Int(i) => i.to_string(),
            Val::Str(s) => format!("{s:?}"),
            Val::Bool(b) => b.to_string(),
            Val::List(v) => format!("[{}]", v.iter().map(|x| x.rust()).collect::<Vec<_>>().join(", ")),
        }
    }
}

/// Where a stage runs in the kernel's program: once before the main loop, in the loop body, or once after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prologue,
    Loop,
    Epilogue,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Prologue => "prologue",
            Phase::Loop => "loop",
            Phase::Epilogue => "epilogue",
        }
    }
    pub fn from_name(s: &str) -> Option<Phase> {
        Some(match s {
            "prologue" => Phase::Prologue,
            "loop" => Phase::Loop,
            "epilogue" => Phase::Epilogue,
            _ => return None,
        })
    }
}

/// Upper-bound resource estimate of a stage (agent planning; the register allocator and GV remain the judges).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    pub vgprs: u32,
    pub agprs: u32,
    pub lds_bytes: u32,
    pub barriers: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// unique instance name (also the stage / region label)
    pub id: String,
    pub prim: Prim,
    /// registry key of the stage (`amdgpu_tile::stage::registry()`)
    pub stage: String,
    pub params: Vec<(String, Val)>,
    pub resources: Resources,
    /// wave roles ("all 4 waves: same program", ...)
    pub roles: Vec<String>,
    pub barrier_phase: u32,
    /// where in the kernel's program the stage runs (prologue / loop / epilogue)
    pub phase: Phase,
}

/// Where the tensor lives while it crosses the edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Store {
    Vgpr,
    Agpr,
    /// a named LDS region
    Lds(String),
    Global,
}

impl Store {
    fn name(&self) -> String {
        match self {
            Store::Vgpr => "vgpr".into(),
            Store::Agpr => "agpr".into(),
            Store::Lds(n) => format!("lds:{n}"),
            Store::Global => "global".into(),
        }
    }
    fn from_name(s: &str) -> Option<Store> {
        Some(match s {
            "vgpr" => Store::Vgpr,
            "agpr" => Store::Agpr,
            "global" => Store::Global,
            _ => Store::Lds(s.strip_prefix("lds:")?.to_string()),
        })
    }
}

/// A fragment layout as `table[t + threads * v]` = tensor element held by lane `t`, value `v` (the `LiveFrag` convention of `contract_stage`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragLayout {
    pub name: String,
    pub threads: usize,
    pub values: usize,
    pub elem_bits: u32,
    pub table: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    /// tensor name (unique)
    pub name: String,
    /// producer node id (None: kernel input)
    pub from: Option<String>,
    /// consumer node ids (empty: kernel output)
    pub to: Vec<String>,
    pub dtype: String,
    pub shape: Vec<i64>,
    /// logical layout name (row-major / transposed / preshuffle / page table ...), documentation + registry lookups
    pub layout: String,
    /// what the producer leaves in the lanes / what the consumer wants (when both are present the class is computed)
    pub producer: Option<FragLayout>,
    pub consumer: Option<FragLayout>,
    pub scope: Scope,
    pub store: Store,
    /// declared class (needed when no layouts are given; checked against the computed one when both exist)
    pub class_hint: Option<Class>,
    /// kernarg ABI of an external tensor: (byte offset, bytes)
    pub abi: Option<(u32, u32)>,
    /// loop-carried: the consumer reads the value produced in the PREVIOUS iteration (ignored for the dependence order)
    pub carried: bool,
}

/// A WEAVE GROUP: stage instances whose units ONE weaver interleaves (`weaver(stageA, stageB)`): the skeleton emits one slot for the group (named after it, labelled `label` = the weave phase),
/// whose default body composes the members in declaration order (`stage::compose`). The nodes stay separate in the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub name: String,
    /// the weave phase label (`Weaver::phase`, the region name)
    pub label: String,
    /// member node ids in declaration order
    pub nodes: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    pub name: String,
    /// graph-wide attributes (e.g. `threads`: the workgroup size of the generated kernels)
    pub attrs: Vec<(String, Val)>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// weave groups (optional)
    pub groups: Vec<Group>,
}

/// The computed cost class of an edge between two fragment layouts.
pub fn class_of_layouts(p: &FragLayout, c: &FragLayout) -> Result<(Class, Movement, usize), String> {
    if (p.threads, p.values) != (c.threads, c.values) {
        return Err(format!("edge '{}': producer shape {}x{} vs consumer {}x{}", c.name, p.threads, p.values, c.threads, c.values));
    }
    if p.elem_bits != c.elem_bits {
        return Err(format!("edge '{}': {} bits vs {}", c.name, p.elem_bits, c.elem_bits));
    }
    let per_reg = ((32 / c.elem_bits.min(32)) as usize).max(1);
    let plan = plan_convert_tables(&p.table, &c.table, c.threads, c.values, per_reg);
    let class = match plan.movement {
        Movement::Identity | Movement::Register | Movement::RegisterBytes => Class::C0,
        Movement::Quad => Class::C1,
        Movement::Half | Movement::CrossLane => Class::C2,
    };
    Ok((class, plan.movement, plan.cost))
}

impl Graph {
    pub fn attr_int(&self, k: &str) -> Option<i64> {
        self.attrs.iter().find(|(n, _)| n == k).and_then(|(_, v)| if let Val::Int(i) = v { Some(*i) } else { None })
    }
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
    pub fn edge(&self, name: &str) -> Option<&Edge> {
        self.edges.iter().find(|e| e.name == name)
    }

    /// The boundary class of an edge inside one kernel: computed from the layouts when both are known (a `class_hint` must then agree), else the hint; an edge whose storage is an LDS
    /// region read by other waves is at least C3.
    pub fn class(&self, e: &Edge) -> Result<Class, String> {
        let computed = match (&e.producer, &e.consumer) {
            (Some(p), Some(c)) => Some(class_of_layouts(p, c)?.0),
            _ => None,
        };
        let base = match (computed, e.class_hint) {
            (Some(c), Some(h)) if c != h => return Err(format!("edge '{}': declared class {} but the layouts give {}", e.name, h.name(), c.name())),
            (Some(c), _) => c,
            (None, Some(h)) => h,
            (None, None) => return Err(format!("edge '{}': neither layouts nor a declared class", e.name)),
        };
        Ok(base)
    }

    /// Structural validation: unique ids, endpoints exist, one producer per tensor, acyclic.
    pub fn validate(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for n in &self.nodes {
            if !seen.insert(&n.id) {
                return Err(format!("duplicate node id '{}'", n.id));
            }
        }
        let mut en = std::collections::HashSet::new();
        for e in &self.edges {
            if !en.insert(&e.name) {
                return Err(format!("duplicate tensor '{}'", e.name));
            }
            if let Some(f) = &e.from {
                if self.node(f).is_none() {
                    return Err(format!("tensor '{}': unknown producer '{f}'", e.name));
                }
            }
            for t in &e.to {
                if self.node(t).is_none() {
                    return Err(format!("tensor '{}': unknown consumer '{t}'", e.name));
                }
            }
            if e.from.is_none() && e.to.is_empty() {
                return Err(format!("tensor '{}' has neither producer nor consumer", e.name));
            }
            if e.from.is_some() && e.to.is_empty() && e.abi.is_none() {
                // a kernel output without an ABI slot is allowed (the skeleton leaves the slot to the kernel), nothing to check
            }
        }
        let mut in_group = std::collections::HashSet::new();
        for g in &self.groups {
            if g.nodes.len() < 2 {
                return Err(format!("weave group '{}' needs at least two member stages", g.name));
            }
            let phase = self.node(&g.nodes[0]).map(|n| n.phase);
            for n in &g.nodes {
                let node = self.node(n).ok_or_else(|| format!("weave group '{}': unknown node '{n}'", g.name))?;
                if !in_group.insert(n.clone()) {
                    return Err(format!("node '{n}' is in two weave groups"));
                }
                if Some(node.phase) != phase {
                    return Err(format!("weave group '{}': members are in different phases", g.name));
                }
            }
        }
        self.topo_order().map(|_| ())
    }

    /// Nodes in a dependence order (Kahn; ties by declaration order).
    pub fn topo_order(&self) -> Result<Vec<usize>, String> {
        let idx: std::collections::HashMap<&str, usize> = self.nodes.iter().enumerate().map(|(i, n)| (n.id.as_str(), i)).collect();
        let mut indeg = vec![0usize; self.nodes.len()];
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for e in &self.edges {
            if e.carried {
                continue;
            }
            if let Some(f) = &e.from {
                for t in &e.to {
                    if let (Some(&a), Some(&b)) = (idx.get(f.as_str()), idx.get(t.as_str())) {
                        succ[a].push(b);
                        indeg[b] += 1;
                    }
                }
            }
        }
        let mut order = Vec::new();
        let mut ready: Vec<usize> = (0..self.nodes.len()).filter(|&i| indeg[i] == 0).collect();
        while !ready.is_empty() {
            ready.sort_unstable();
            let i = ready.remove(0);
            order.push(i);
            for &s in &succ[i] {
                indeg[s] -= 1;
                if indeg[s] == 0 {
                    ready.push(s);
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err("the graph has a cycle".into());
        }
        Ok(order)
    }

    // ------------------------------------------------------------------ JSON

    pub fn to_json(&self) -> Json {
        let nodes = self
            .nodes
            .iter()
            .map(|n| {
                Json::Obj(vec![
                    ("id".into(), Json::Str(n.id.clone())),
                    ("prim".into(), Json::Str(n.prim.code().into())),
                    ("stage".into(), Json::Str(n.stage.clone())),
                    ("params".into(), Json::Obj(n.params.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())),
                    (
                        "resources".into(),
                        Json::Obj(vec![
                            ("vgprs".into(), Json::Int(n.resources.vgprs as i64)),
                            ("agprs".into(), Json::Int(n.resources.agprs as i64)),
                            ("lds_bytes".into(), Json::Int(n.resources.lds_bytes as i64)),
                            ("barriers".into(), Json::Int(n.resources.barriers as i64)),
                        ]),
                    ),
                    ("roles".into(), Json::Arr(n.roles.iter().map(|r| Json::Str(r.clone())).collect())),
                    ("barrier_phase".into(), Json::Int(n.barrier_phase as i64)),
                    ("phase".into(), Json::Str(n.phase.name().into())),
                ])
            })
            .collect();
        let frag = |f: &Option<FragLayout>| match f {
            None => Json::Null,
            Some(f) => Json::Obj(vec![
                ("name".into(), Json::Str(f.name.clone())),
                ("threads".into(), Json::Int(f.threads as i64)),
                ("values".into(), Json::Int(f.values as i64)),
                ("elem_bits".into(), Json::Int(f.elem_bits as i64)),
                ("table".into(), Json::Arr(f.table.iter().map(|x| Json::Int(*x)).collect())),
            ]),
        };
        let edges = self
            .edges
            .iter()
            .map(|e| {
                Json::Obj(vec![
                    ("name".into(), Json::Str(e.name.clone())),
                    ("from".into(), e.from.clone().map_or(Json::Null, Json::Str)),
                    ("to".into(), Json::Arr(e.to.iter().map(|t| Json::Str(t.clone())).collect())),
                    ("dtype".into(), Json::Str(e.dtype.clone())),
                    ("shape".into(), Json::Arr(e.shape.iter().map(|x| Json::Int(*x)).collect())),
                    ("layout".into(), Json::Str(e.layout.clone())),
                    ("producer".into(), frag(&e.producer)),
                    ("consumer".into(), frag(&e.consumer)),
                    ("scope".into(), Json::Str(e.scope.name().into())),
                    ("store".into(), Json::Str(e.store.name())),
                    ("class".into(), e.class_hint.map_or(Json::Null, |c| Json::Str(c.name().into()))),
                    ("abi".into(), e.abi.map_or(Json::Null, |(o, n)| Json::Arr(vec![Json::Int(o as i64), Json::Int(n as i64)]))),
                    ("carried".into(), Json::Bool(e.carried)),
                ])
            })
            .collect();
        let mut j = Json::Obj(vec![
            ("name".into(), Json::Str(self.name.clone())),
            ("attrs".into(), Json::Obj(self.attrs.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())),
            ("nodes".into(), Json::Arr(nodes)),
            ("edges".into(), Json::Arr(edges)),
        ]);
        if let Json::Obj(v) = &mut j {
            if !self.groups.is_empty() {
                v.push((
                    "groups".into(),
                    Json::Arr(
                        self.groups
                            .iter()
                            .map(|g| Json::Obj(vec![("name".into(), Json::Str(g.name.clone())), ("label".into(), Json::Str(g.label.clone())), ("nodes".into(), Json::Arr(g.nodes.iter().map(|n| Json::Str(n.clone())).collect()))]))
                            .collect(),
                    ),
                ));
            }
        }
        j
    }

    pub fn from_json(j: &Json) -> Result<Graph, String> {
        let s = |j: &Json, k: &str| -> Result<String, String> { j.get(k).and_then(|x| x.str()).map(|x| x.to_string()).ok_or_else(|| format!("missing string field '{k}'")) };
        let i = |j: &Json, k: &str| -> Result<i64, String> { j.get(k).and_then(|x| x.int()).ok_or_else(|| format!("missing integer field '{k}'")) };
        let mut g = Graph { name: s(j, "name")?, ..Default::default() };
        if let Some(Json::Obj(v)) = j.get("attrs") {
            for (k, x) in v {
                g.attrs.push((k.clone(), Val::from_json(x)?));
            }
        }
        for n in j.get("nodes").and_then(|x| x.arr()).ok_or("missing 'nodes'")? {
            let params = match n.get("params") {
                Some(Json::Obj(v)) => v.iter().map(|(k, x)| Ok((k.clone(), Val::from_json(x)?))).collect::<Result<Vec<_>, String>>()?,
                _ => Vec::new(),
            };
            let r = n.get("resources").ok_or("missing 'resources'")?;
            g.nodes.push(Node {
                id: s(n, "id")?,
                prim: Prim::from_code(&s(n, "prim")?).ok_or("unknown primitive")?,
                stage: s(n, "stage")?,
                params,
                resources: Resources { vgprs: i(r, "vgprs")? as u32, agprs: i(r, "agprs")? as u32, lds_bytes: i(r, "lds_bytes")? as u32, barriers: i(r, "barriers")? as u32 },
                roles: n.get("roles").and_then(|x| x.arr()).map(|v| v.iter().filter_map(|x| x.str().map(|s| s.to_string())).collect()).unwrap_or_default(),
                barrier_phase: i(n, "barrier_phase")? as u32,
                phase: Phase::from_name(&s(n, "phase")?).ok_or("unknown phase")?,
            });
        }
        let frag = |f: Option<&Json>| -> Result<Option<FragLayout>, String> {
            match f {
                None | Some(Json::Null) => Ok(None),
                Some(f) => Ok(Some(FragLayout {
                    name: s(f, "name")?,
                    threads: i(f, "threads")? as usize,
                    values: i(f, "values")? as usize,
                    elem_bits: i(f, "elem_bits")? as u32,
                    table: f.get("table").and_then(|x| x.arr()).ok_or("missing 'table'")?.iter().filter_map(|x| x.int()).collect(),
                })),
            }
        };
        for e in j.get("edges").and_then(|x| x.arr()).ok_or("missing 'edges'")? {
            g.edges.push(Edge {
                name: s(e, "name")?,
                from: e.get("from").and_then(|x| x.str()).map(|x| x.to_string()),
                to: e.get("to").and_then(|x| x.arr()).map(|v| v.iter().filter_map(|x| x.str().map(|s| s.to_string())).collect()).unwrap_or_default(),
                dtype: s(e, "dtype")?,
                shape: e.get("shape").and_then(|x| x.arr()).map(|v| v.iter().filter_map(|x| x.int()).collect()).unwrap_or_default(),
                layout: s(e, "layout")?,
                producer: frag(e.get("producer"))?,
                consumer: frag(e.get("consumer"))?,
                scope: Scope::from_name(&s(e, "scope")?).ok_or("unknown scope")?,
                store: Store::from_name(&s(e, "store")?).ok_or("unknown store")?,
                class_hint: e.get("class").and_then(|x| x.str()).and_then(Class::from_name),
                abi: match e.get("abi") {
                    Some(Json::Arr(v)) if v.len() == 2 => Some((v[0].int().unwrap_or(0) as u32, v[1].int().unwrap_or(0) as u32)),
                    _ => None,
                },
                carried: e.get("carried").and_then(|x| x.bool()).unwrap_or(false),
            });
        }
        if let Some(Json::Arr(gs)) = j.get("groups") {
            for gj in gs {
                g.groups.push(Group { name: s(gj, "name")?, label: s(gj, "label")?, nodes: gj.get("nodes").and_then(|x| x.arr()).map(|v| v.iter().filter_map(|x| x.str().map(|s| s.to_string())).collect()).unwrap_or_default() });
            }
        }
        g.validate()?;
        Ok(g)
    }

    pub fn to_json_string(&self) -> String {
        self.to_json().to_string_pretty()
    }
    pub fn from_json_str(s: &str) -> Result<Graph, String> {
        Graph::from_json(&Json::parse(s)?)
    }
}

// ---------------------------------------------------------------------- cuts

/// One kernel of a cut: a name and the node ids it contains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCut {
    pub name: String,
    pub nodes: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cut {
    pub kernels: Vec<KernelCut>,
}

/// Budget of one kernel (registers per lane, LDS bytes).
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub vgprs: u32,
    pub agprs: u32,
    pub lds_bytes: u32,
}

impl Default for Budget {
    fn default() -> Self {
        // gfx950, one wave per SIMD: 256 arch VGPR + 256 AGPR, 160 KB of LDS
        Budget { vgprs: 256, agprs: 256, lds_bytes: 160 * 1024 }
    }
}

/// One on-chip edge of a kernel / one cut edge.
#[derive(Clone, Debug)]
pub struct EdgeReport {
    pub tensor: String,
    pub from: Option<String>,
    pub to: Vec<String>,
    pub class: Class,
    pub scope: Scope,
}

#[derive(Clone, Debug)]
pub struct KernelReport {
    pub name: String,
    /// stage instances in dependence order
    pub stages: Vec<String>,
    /// edges between two stages of this kernel
    pub internal: Vec<EdgeReport>,
    /// tensors read from / written to HBM by this kernel (graph inputs / outputs and cut edges)
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub resources: Resources,
}

#[derive(Clone, Debug, Default)]
pub struct CutReport {
    pub kernels: Vec<KernelReport>,
    /// edges cut by the partition (class C4)
    pub cut_edges: Vec<EdgeReport>,
}

impl Graph {
    /// Evaluates a cut: every node in exactly one kernel, kernels in dependence order, per-edge classes (a cut edge is C4), scope requirements (a tensor that may not stay on chip must
    /// be cut) and the resource budget of each kernel (stages run one after another: the register estimate of a kernel is the largest footprint of one stage, each estimate including its live-in tensors; the estimate is
    /// the sum over stages of `temps` the author declared, an upper bound).
    pub fn evaluate_cut(&self, cut: &Cut, budget: Budget) -> Result<CutReport, String> {
        self.validate()?;
        let mut owner: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (k, kc) in cut.kernels.iter().enumerate() {
            for n in &kc.nodes {
                if self.node(n).is_none() {
                    return Err(format!("kernel '{}': unknown node '{n}'", kc.name));
                }
                if owner.insert(n.as_str(), k).is_some() {
                    return Err(format!("node '{n}' is in two kernels"));
                }
            }
        }
        for n in &self.nodes {
            if !owner.contains_key(n.id.as_str()) {
                return Err(format!("node '{}' is in no kernel", n.id));
            }
        }
        let order = self.topo_order()?;
        let rank: std::collections::HashMap<&str, usize> = order.iter().enumerate().map(|(r, &i)| (self.nodes[i].id.as_str(), r)).collect();
        let mut rep = CutReport::default();
        for (k, kc) in cut.kernels.iter().enumerate() {
            let mut stages: Vec<&str> = kc.nodes.iter().map(|s| s.as_str()).collect();
            stages.sort_by_key(|s| rank[s]);
            let mut kr = KernelReport { name: kc.name.clone(), stages: stages.iter().map(|s| s.to_string()).collect(), internal: Vec::new(), inputs: Vec::new(), outputs: Vec::new(), resources: Resources::default() };
            for s in &stages {
                let r = self.node(s).unwrap().resources;
                kr.resources.vgprs = kr.resources.vgprs.max(r.vgprs);
                kr.resources.agprs = kr.resources.agprs.max(r.agprs);
                kr.resources.lds_bytes = kr.resources.lds_bytes.max(r.lds_bytes);
                kr.resources.barriers += r.barriers;
            }
            for e in &self.edges {
                let prod_k = e.from.as_deref().map(|f| owner[f]);
                let cons_ks: Vec<usize> = e.to.iter().map(|t| owner[t.as_str()]).collect();
                let in_this_prod = prod_k == Some(k);
                let in_this_cons = cons_ks.contains(&k);
                if !in_this_prod && !in_this_cons {
                    continue;
                }
                let crosses = cons_ks.iter().any(|&c| Some(c) != prod_k) && prod_k.is_some();
                if prod_k.is_none() {
                    kr.inputs.push(e.name.clone());
                    continue;
                }
                if e.to.is_empty() {
                    kr.outputs.push(e.name.clone());
                    continue;
                }
                if crosses {
                    // between two kernels: through HBM
                    if in_this_prod {
                        kr.outputs.push(e.name.clone());
                        if !rep.cut_edges.iter().any(|c| c.tensor == e.name) {
                            rep.cut_edges.push(EdgeReport { tensor: e.name.clone(), from: e.from.clone(), to: e.to.clone(), class: Class::C4, scope: e.scope });
                        }
                    }
                    if in_this_cons && !in_this_prod {
                        kr.inputs.push(e.name.clone());
                    }
                    continue;
                }
                // fully inside this kernel
                if in_this_prod {
                    if e.scope == Scope::None {
                        return Err(format!("tensor '{}' may not stay on chip but is not cut", e.name));
                    }
                    if e.scope == Scope::Grid {
                        return Err(format!("tensor '{}' needs grid-level cooperation: cut it or declare the cooperation", e.name));
                    }
                    let mut c = self.class(e)?;
                    if e.scope == Scope::Block && c < Class::C3 {
                        c = Class::C3;
                    }
                    kr.internal.push(EdgeReport { tensor: e.name.clone(), from: e.from.clone(), to: e.to.clone(), class: c, scope: e.scope });
                }
            }
            if kr.resources.vgprs > budget.vgprs || kr.resources.agprs > budget.agprs || kr.resources.lds_bytes > budget.lds_bytes {
                return Err(format!(
                    "kernel '{}': resource estimate {} VGPR / {} AGPR / {} B LDS exceeds the budget {} / {} / {}",
                    kc.name, kr.resources.vgprs, kr.resources.agprs, kr.resources.lds_bytes, budget.vgprs, budget.agprs, budget.lds_bytes
                ));
            }
            rep.kernels.push(kr);
        }
        Ok(rep)
    }
}

impl Graph {
    /// Evaluates the kernel made of `nodes` (every other node is "the rest": edges between the two are cut edges) and returns that kernel's report.
    pub fn kernel_report(&self, nodes: &[&str]) -> Result<KernelReport, String> {
        let rest: Vec<String> = self.nodes.iter().filter(|n| !nodes.contains(&n.id.as_str())).map(|n| n.id.clone()).collect();
        let mut cut = Cut { kernels: vec![KernelCut { name: "kernel".into(), nodes: nodes.iter().map(|s| s.to_string()).collect() }] };
        if !rest.is_empty() {
            cut.kernels.push(KernelCut { name: "rest".into(), nodes: rest });
        }
        let mut r = self.evaluate_cut(&cut, Budget::default())?;
        Ok(r.kernels.remove(0))
    }
}

impl CutReport {
    /// The text description for review: per kernel the stage list, the on-chip edges with their classes, the HBM tensors and the resource estimate.
    pub fn text(&self, g: &Graph) -> String {
        let mut s = String::new();
        s.push_str(&format!("graph {}: {} kernel(s), {} cut edge(s)\n", g.name, self.kernels.len(), self.cut_edges.len()));
        for k in &self.kernels {
            s.push_str(&format!("kernel {}\n", k.name));
            s.push_str("  stages:");
            for st in &k.stages {
                let n = g.node(st).unwrap();
                s.push_str(&format!(" {}[{}:{}]", st, n.prim.code(), n.stage));
            }
            s.push('\n');
            for e in &k.internal {
                s.push_str(&format!("  on-chip {} {} -> {} : {} ({})\n", e.tensor, e.from.as_deref().unwrap_or("-"), e.to.join(","), e.class.name(), e.scope.name()));
            }
            s.push_str(&format!("  hbm in : {}\n  hbm out: {}\n", k.inputs.join(", "), k.outputs.join(", ")));
            s.push_str(&format!("  est: {} VGPR, {} AGPR, {} B LDS, {} barrier(s)\n", k.resources.vgprs, k.resources.agprs, k.resources.lds_bytes, k.resources.barriers));
        }
        for c in &self.cut_edges {
            s.push_str(&format!("cut {} {} -> {} : C4\n", c.tensor, c.from.as_deref().unwrap_or("-"), c.to.join(",")));
        }
        s
    }
}