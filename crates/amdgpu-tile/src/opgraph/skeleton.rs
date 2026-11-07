//! The SKELETON GENERATOR (design section 3, step 3): from an op graph and a kernel of a cut it emits the Rust source of the kernel's build function with the orchestration SLOTS.
//!
//! The generated text contains
//!  * a review header (the kernel's stages, its on-chip edges with their boundary classes, the HBM tensors, the resource estimate);
//!  * a trait `<Kernel>Slots` with one method per slot - `prelude` (kernel ABI, grid, LDS plan, descriptors), one per stage instance (named after the node; the kernel constructs the stage
//!    instance (`amdgpu_tile::stage`) and emits it with ITS orchestration: weave order, `declare_region`, scheduler tokens), and `loop_open` / `loop_close` around the loop-phase stages;
//!  * `build_<kernel>`: creates the `FnBuilder`, runs the structural checks of the graph (`check_kernel`: the on-chip edges keep the classes the skeleton was generated for) and calls the slots in
//!    dependence order. Nothing in the generated function touches the IR except through the slots, so filling the slots with an existing orchestration reproduces that kernel exactly.
//!
//! The text is deterministic: a test regenerates it and compares with the committed file.

use super::*;

fn ident_ok(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') && !s.chars().next().unwrap().is_ascii_digit()
}

fn camel(s: &str) -> String {
    s.split('_').filter(|p| !p.is_empty()).map(|p| p[..1].to_ascii_uppercase() + &p[1..]).collect()
}

fn params_text(n: &Node) -> String {
    if n.params.is_empty() {
        return String::new();
    }
    format!(" ({})", n.params.iter().map(|(k, v)| format!("{k}={}", v.rust())).collect::<Vec<_>>().join(", "))
}

/// Structural check run by the generated build function: the kernel made of `nodes` evaluates (scope / budget / coverage) and its on-chip edges have exactly the expected classes.
pub fn check_kernel(graph_json: &str, nodes: &[&str], expect: &[(&str, &str)]) -> Result<(), String> {
    let g = Graph::from_json_str(graph_json)?;
    let r = g.kernel_report(nodes)?;
    let got: Vec<(String, String)> = r.internal.iter().map(|e| (e.tensor.clone(), e.class.name().to_string())).collect();
    let want: Vec<(String, String)> = expect.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    if got != want {
        return Err(format!("on-chip edges {got:?} differ from the skeleton's {want:?}"));
    }
    Ok(())
}

/// One generated slot of the skeleton.
struct Slot {
    name: String,
    phase: Phase,
    /// doc lines (without the `///` prefix)
    doc: Vec<String>,
    /// default body (None: required)
    body: Option<String>,
    /// extra trait items emitted right after the slot (member stage constructors of a weave group)
    extra: Vec<String>,
}

fn suggest(e: &Edge, load: bool) -> Option<&'static str> {
    let key = if load {
        match &e.store {
            Store::Lds(_) => "move.lds_image",
            _ => "move.row_frags",
        }
    } else if e.dtype == "bf16" || e.dtype == "f16" {
        "map.store_bf16"
    } else {
        "map.epilogue_chain"
    };
    crate::stage::registry().iter().any(|d| d.key == key).then_some(key)
}

/// Generates the skeleton source of kernel `kernel` made of `nodes` (ids). `graph_file` is the path (relative to the generated file) of the graph JSON the build function embeds.
///
/// Slots, in program order: `prelude`; the prologue stages; `loop_open`, the loop stages, `loop_close`; the epilogue stages. A stage is a slot named after its node; the members of a weave
/// group are ONE slot named after the group (default body: `stage::compose` of the member stages in declaration order, the members' constructors `<node>_stage` being required trait
/// methods); an edge cut away from this kernel adds `store_<tensor>` (right after its producer) / `load_<tensor>` (right before its first consumer) slots whose default body is a `todo!` naming the
/// registry `move.*` stage that matches the tensor.
pub fn generate(g: &Graph, kernel: &str, nodes: &[&str], graph_file: &str) -> Result<String, String> {
    if !ident_ok(kernel) {
        return Err(format!("kernel name '{kernel}' is not a lower-case identifier"));
    }
    let rep = g.kernel_report(nodes)?;
    let threads = g.attr_int("threads").ok_or("the graph has no `threads` attribute")?;
    let order = g.topo_order()?;
    let stages: Vec<&Node> = order.iter().map(|&i| &g.nodes[i]).filter(|n| nodes.contains(&n.id.as_str())).collect();
    for n in &stages {
        if !ident_ok(&n.id) {
            return Err(format!("node id '{}' is not a lower-case identifier (it names a slot)", n.id));
        }
        if n.id == "prelude" || n.id == "loop_open" || n.id == "loop_close" || n.id.starts_with("store_") || n.id.starts_with("load_") {
            return Err(format!("node id '{}' collides with a generated slot name", n.id));
        }
    }
    let group_of = |id: &str| g.groups.iter().find(|gr| gr.nodes.iter().any(|m| m == id));
    for gr in &g.groups {
        let inside = gr.nodes.iter().filter(|m| nodes.contains(&m.as_str())).count();
        if inside != 0 && inside != gr.nodes.len() {
            return Err(format!("weave group '{}' is split by the cut", gr.name));
        }
        if !ident_ok(&gr.name) {
            return Err(format!("weave group name '{}' is not a lower-case identifier", gr.name));
        }
    }
    let in_k = |id: &str| nodes.contains(&id);
    // edges cut away from this kernel
    let cut_in: Vec<&Edge> = g.edges.iter().filter(|e| e.from.as_deref().map_or(false, |f| !in_k(f)) && e.to.iter().any(|t| in_k(t))).collect();
    let cut_out: Vec<&Edge> = g.edges.iter().filter(|e| e.from.as_deref().map_or(false, in_k) && e.to.iter().any(|t| !in_k(t))).collect();
    let edge_doc = |e: &Edge| format!("tensor {} : {} {:?} {}", e.name, e.dtype, e.shape, e.layout);

    let mut slots: Vec<Slot> = Vec::new();
    let mut done_groups: Vec<&str> = Vec::new();
    let mut done_load: Vec<&str> = Vec::new();
    let node_io = |n: &Node| -> Vec<String> {
        let mut d = Vec::new();
        for e in &g.edges {
            if e.to.iter().any(|t| t == &n.id) {
                d.push(format!("  in : {} ({})", e.name, e.from.as_deref().unwrap_or("hbm")));
            }
            if e.from.as_deref() == Some(n.id.as_str()) {
                d.push(format!("  out: {}", e.name));
            }
        }
        d
    };
    for n in &stages {
        let members: Vec<&Node> = match group_of(&n.id) {
            Some(gr) => {
                if done_groups.contains(&gr.name.as_str()) {
                    continue;
                }
                done_groups.push(&gr.name);
                gr.nodes.iter().map(|m| g.node(m).unwrap()).collect()
            }
            None => vec![*n],
        };
        // loads of the cut edges these stages consume
        for e in &cut_in {
            if !done_load.contains(&e.name.as_str()) && e.to.iter().any(|t| members.iter().any(|m| &m.id == t)) {
                done_load.push(&e.name);
                let key = suggest(e, true);
                let mut doc = vec![format!("slot `load_{}`: {} arrives through HBM (cut edge from {}, class C4); issue its loads before `{}`.", e.name, edge_doc(e), e.from.as_deref().unwrap_or("-"), members[0].id)];
                if let Some(k) = key {
                    doc.push(format!("  suggested stage: `{k}`"));
                }
                slots.push(Slot { name: format!("load_{}", e.name), phase: n.phase, doc, body: Some(format!("todo!(\"slot load_{}: fill with {} (or override)\")", e.name, key.map_or("a move stage".to_string(), |k| format!("a `{k}` stage")))), extra: Vec::new() });
            }
        }
        if members.len() == 1 {
            let mut doc = vec![format!("slot `{}`: stage `{}` [{}]{}. Construct the stage instance and emit it with the kernel's orchestration (`emit` / `emit_with`, `declare_region`).", n.id, n.stage, n.prim.code(), params_text(n))];
            doc.extend(node_io(n));
            slots.push(Slot { name: n.id.clone(), phase: n.phase, doc, body: None, extra: Vec::new() });
        } else {
            let gr = group_of(&n.id).unwrap();
            let mut doc = vec![format!("slot `{}`: weave group (phase label `{}`) of {} stages woven by ONE weaver; the default body composes the members in declaration order (`stage::compose` + `seq`). Override to give the weave an order / strategy / `declare_region`.", gr.name, gr.label, members.len())];
            let mut extra = Vec::new();
            let mut ctor = Vec::new();
            for m in &members {
                doc.push(format!("  member `{}`: stage `{}` [{}]{}", m.id, m.stage, m.prim.code(), params_text(m)));
                doc.extend(node_io(m).into_iter().map(|l| format!("  {l}")));
                extra.push(format!("    /// the stage instance of member `{}` of group `{}` (key `{}`)\n    fn {}_stage<'a>(&'a self, cx: &'a Self::Cx) -> amdgpu_tile::stage::StageInst<'a>;\n", m.id, gr.name, m.stage, m.id));
                ctor.push(format!("self.{}_stage(&*cx)", m.id));
            }
            let body = format!("let (w, _members) = amdgpu_tile::stage::compose({:?}, vec![{}]);\n        w.seq().emit(b);", gr.label, ctor.join(", "));
            slots.push(Slot { name: gr.name.clone(), phase: n.phase, doc, body: Some(body), extra });
        }
        // stores of the cut edges these stages produce
        for e in &cut_out {
            if e.from.as_deref().map_or(false, |f| members.iter().any(|m| m.id == f)) {
                let key = suggest(e, false);
                let mut doc = vec![format!("slot `store_{}`: {} leaves the kernel through HBM (cut edge to {}, class C4).", e.name, edge_doc(e), e.to.join(","))];
                if let Some(k) = key {
                    doc.push(format!("  suggested stage: `{k}`"));
                }
                slots.push(Slot { name: format!("store_{}", e.name), phase: n.phase, doc, body: Some(format!("todo!(\"slot store_{}: fill with {} (or override)\")", e.name, key.map_or("a store stage".to_string(), |k| format!("a `{k}` stage")))), extra: Vec::new() });
            }
        }
    }
    let of = |p: Phase| -> Vec<&Slot> { slots.iter().filter(|s| s.phase == p).collect() };
    let (pro, lp, epi) = (of(Phase::Prologue), of(Phase::Loop), of(Phase::Epilogue));
    let tname = format!("{}Slots", camel(kernel));
    let mut s = String::new();
    s.push_str(&format!("// @generated by amdgpu_tile::opgraph::skeleton from graph \"{}\", kernel \"{kernel}\": regenerate, do not edit (fill the slots in an implementation of `{tname}`).\n", g.name));
    s.push_str("//\n");
    s.push_str(&format!("// kernel {kernel}: {} stage(s), {} threads\n", stages.len(), threads));
    for n in &stages {
        s.push_str(&format!("//   {} [{}: {}]{}  phase {}\n", n.id, n.prim.code(), n.stage, params_text(n), n.phase.name()));
    }
    for gr in g.groups.iter().filter(|gr| gr.nodes.iter().all(|m| in_k(m))) {
        s.push_str(&format!("//   weave group {} (phase label {}): {}\n", gr.name, gr.label, gr.nodes.join(" + ")));
    }
    for e in &rep.internal {
        s.push_str(&format!("//   on-chip {} {} -> {} : {} ({})\n", e.tensor, e.from.as_deref().unwrap_or("-"), e.to.join(","), e.class.name(), e.scope.name()));
    }
    for e in &cut_in {
        s.push_str(&format!("//   cut in : {} from {} (C4)\n", e.name, e.from.as_deref().unwrap_or("-")));
    }
    for e in &cut_out {
        s.push_str(&format!("//   cut out: {} to {} (C4)\n", e.name, e.to.join(",")));
    }
    s.push_str(&format!("//   hbm in : {}\n//   hbm out: {}\n", rep.inputs.join(", "), rep.outputs.join(", ")));
    s.push_str(&format!("//   est: {} VGPR, {} AGPR, {} B LDS, {} barrier(s)\n\n", rep.resources.vgprs, rep.resources.agprs, rep.resources.lds_bytes, rep.resources.barriers));
    s.push_str("use amdgpu_dsl::{FnBuilder, Function};\n\n");
    s.push_str("/// The orchestration slots of the kernel: everything the graph does NOT decide (pipeline cadence, emission order, weave orders and regions, scheduler tokens, lane / role setup).\n");
    s.push_str(&format!("pub trait {tname} {{\n"));
    s.push_str("    /// values shared between the slots (kernarg values, lane ids, LDS tiles, descriptors, loop variables)\n    type Cx;\n");
    s.push_str("    /// slot `prelude`: right after `FnBuilder::new`: `compile_profile`, kernel arguments, symbol ranges, grid, LDS plan, buffer descriptors.\n");
    for e in g.edges.iter().filter(|e| rep.inputs.contains(&e.name) || rep.outputs.contains(&e.name)) {
        let abi = e.abi.map_or(String::new(), |(o, n)| format!(" @{o}+{n}"));
        s.push_str(&format!("    ///   tensor {} : {} {:?} ({}){}\n", e.name, e.dtype, e.shape, if rep.inputs.contains(&e.name) { "in" } else { "out" }, abi));
    }
    s.push_str("    fn prelude(&mut self, b: &mut FnBuilder) -> Self::Cx;\n");
    let render = |s: &mut String, sl: &Slot| {
        for l in &sl.doc {
            s.push_str(&format!("    /// {l}\n"));
        }
        match &sl.body {
            None => s.push_str(&format!("    fn {}(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);\n", sl.name)),
            Some(body) => s.push_str(&format!("    fn {}(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx) {{\n        {body}\n    }}\n", sl.name)),
        }
        for x in &sl.extra {
            s.push_str(x);
        }
    };
    for sl in &pro {
        render(&mut s, sl);
    }
    if !lp.is_empty() {
        s.push_str("    /// slot `loop_open`: `begin_loop` with its carried values, the per-iteration scalars the loop stages share.\n    fn loop_open(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);\n");
        for sl in &lp {
            render(&mut s, sl);
        }
        s.push_str("    /// slot `loop_close`: the carried values' updates, the exit test, `end_loop`.\n    fn loop_close(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx);\n");
    }
    for sl in &epi {
        render(&mut s, sl);
    }
    s.push_str("}\n\n");
    let node_list = stages.iter().map(|n| format!("{:?}", n.id)).collect::<Vec<_>>().join(", ");
    let expect = rep.internal.iter().map(|e| format!("({:?}, {:?})", e.tensor, e.class.name())).collect::<Vec<_>>().join(", ");
    s.push_str(&format!("/// Builds kernel `{kernel}` of graph `{}`: the structure comes from the graph, everything else from the slots.\n", g.name));
    s.push_str(&format!("pub fn build_{kernel}<S: {tname}>(name: &str, slots: &mut S) -> Function {{\n"));
    s.push_str(&format!("    amdgpu_tile::opgraph::skeleton::check_kernel(include_str!({graph_file:?}), &[{node_list}], &[{expect}]).unwrap_or_else(|e| panic!(\"{{name}}: {{e}}\"));\n"));
    s.push_str(&format!("    let mut fb = FnBuilder::new(name, {threads});\n    let b = &mut fb;\n    let mut cx = slots.prelude(b);\n"));
    for sl in &pro {
        s.push_str(&format!("    slots.{}(b, &mut cx);\n", sl.name));
    }
    if !lp.is_empty() {
        s.push_str("    slots.loop_open(b, &mut cx);\n");
        for sl in &lp {
            s.push_str(&format!("    slots.{}(b, &mut cx);\n", sl.name));
        }
        s.push_str("    slots.loop_close(b, &mut cx);\n");
    }
    for sl in &epi {
        s.push_str(&format!("    slots.{}(b, &mut cx);\n", sl.name));
    }
    s.push_str("    fb.finish()\n}\n");
    Ok(s)
}