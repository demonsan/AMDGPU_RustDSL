//! `amdc` - command line driver for the AMDGPU Rust DSL compiler.

#![forbid(unsafe_code)]

use amdc::batch;
use amdc::hostgen;

use amdgpu_codegen::backend::{self, probe, LlvmTail, LlvmTools, TailBackend};
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!("usage:");
    eprintln!("  amdc list");
    eprintln!("  amdc dispatch decode16 <B> <HK> <CTX> [--page P]   (the paged-decode launch plan as JSON)");
    eprintln!("  amdc graph plan <graph.json | -> [--cus N] [--no-fuse] (the L1 launch plan of an op graph, docs/l1-graph.md)");
    eprintln!("  amdc print-ir <kernel>|--from-ir <file> [--stage dsl|mir|mir-ra] [--out <file>]");
    eprintln!("  amdc ladder-hints [--out FILE] [--jobs N]   the retry-ladder rung each kernel compiles at (crates/amdgpu-kernels/src/ladder_hints.rs)");
    eprintln!("  amdc compile-batch [--jobs N] [--names a,b,c] [--no-verify] [--assemble] [--repeat R] [--time]   compile many kernels on N threads and report where the time goes (docs/compile-time.md)");
    eprintln!("  amdc emit <kernel> [--out <file.s>] [--time] [--assemble] [--backend llvm-tail|own] [--mir-out <file.mir>] [--sched | --sched=ld] [--specialize name=value[,..]]");
    eprintln!("      --specialize binds scalar kernargs to constants (shape specialization; docs/jit.md); the specialized kernel must verify strictly");
    eprintln!("  amdc regplan <kernel> [--specialize ..]   per weave-region / stage register peaks (allocator truth) + register-plan report");
    eprintln!("  amdc ir-check <file.ir> [--verify]      parse + check an irv1 DSL IR file (docs/ir-text.md); --from-ir <file> works with emit / gpu-verify / jit-info / print-ir / regplan");
    eprintln!("  amdc jit-info <kernel> [--specialize name=value[,..]] [--backend ..]   JSON metadata for the Python JIT (docs/jit.md)");
    eprintln!("  amdc gpu-verify <kernel>|--all|--shipped [-v] [--strict] [--specialize ..]");
    eprintln!("      --backend llvm-tail runs the pinned llc tail (ROCM_LLVM_BIN overrides the tool dir);");
    eprintln!("      --mir-out writes the exported post-RA MIR only (no llc needed) and exits.");
    eprintln!("  amdc gen-host <kernel>|--shipped [--out <dir>]   generate C++ ABI/validate/launch header + JSON metadata");
    eprintln!("  amdc opgraph <graph.json> [--cut k1=n1+n2,k2=n3]   review text of an op graph and its cut (kernels, stages, on-chip edges with classes, HBM tensors, estimates)");
    eprintln!("  amdc skeleton <graph.json> <kernel> [--nodes n1,n2|all] [--graph-file <path>] [--out <file.rs>]   generate the Rust skeleton (orchestration slots) of one kernel of the graph");
    eprintln!("  amdc backend probe|check|write-lock   version probe / verify / rewrite backend.lock
  amdc weave-report <kernel>|--from-ir <file>   the back-end Weaver unit graph of the final MIR (docs/two-weavers.md)
  amdc passes [--target gfx950] [--evidence]   the compile pipeline: named passes, stage, target dependence, gate, evidence; the retry ladders
  amdc md5-manifest [--out FILE | --check FILE] [--jobs N]   the golden asm md5 of every reachable kernel (tests/golden/README.md)");
}

/// The `--specialize` bindings of the command line (`--specialize a=1,b=2` or `--specialize=a=1`, repeatable).
fn spec_bindings(args: &[String]) -> Result<Vec<(String, String)>, String> {
    let mut s: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--specialize" {
            s.push(args.get(i + 1).cloned().ok_or("--specialize needs name=value[,..]")?);
            i += 2;
        } else if let Some(v) = args[i].strip_prefix("--specialize=") {
            s.push(v.to_string());
            i += 1;
        } else {
            i += 1;
        }
    }
    amdgpu_dsl::specialize::parse_bindings(&s.join(","))
}

/// The kernel `name`, specialized by the command line's bindings (exit code 2 on a bad binding).
type Spec = (Vec<(String, u32, amdgpu_dsl::specialize::SpecValue)>, Vec<(u32, i64)>);

/// `--from-ir <file>` / `--from-ir=<file>`: the kernel comes from an irv1 text file instead of the registry (docs/ir-text.md).
fn from_ir_path(args: &[String]) -> Option<String> {
    args.windows(2).find(|w| w[0] == "--from-ir").map(|w| w[1].clone()).or_else(|| args.iter().find_map(|a| a.strip_prefix("--from-ir=").map(|s| s.to_string())))
}

/// The kernel of the command line: `--from-ir <file>` (parsed and checked) or the registered kernel `name`.
fn load_kernel(name: &str, args: &[String]) -> Result<amdgpu_dsl::Function, ExitCode> {
    if let Some(path) = from_ir_path(args) {
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                return Err(ExitCode::from(2));
            }
        };
        return amdgpu_dsl::ir_text::parse(&text).map_err(|e| {
            eprintln!("{path}: {e}");
            ExitCode::from(2)
        });
    }
    amdgpu_kernels::kernel_by_name(name).ok_or_else(|| {
        eprintln!("unknown kernel: {name}");
        ExitCode::from(2)
    })
}

/// `--from-ir` kernels and specialized kernels are verified strictly before use (unless `--no-verify`).
fn must_verify(args: &[String], spec: &Spec) -> bool {
    (from_ir_path(args).is_some() || !spec.0.is_empty()) && !args.iter().any(|a| a == "--no-verify")
}

fn kernel_spec(name: &str, args: &[String]) -> Result<(amdgpu_dsl::Function, Spec), ExitCode> {
    let f = load_kernel(name, args)?;
    let binds = match spec_bindings(args) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return Err(ExitCode::from(2));
        }
    };
    if binds.is_empty() {
        return Ok((f, (Vec::new(), Vec::new())));
    }
    let vals = match amdgpu_dsl::specialize::check(&f, &binds) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return Err(ExitCode::from(2));
        }
    };
    let derived = match amdgpu_dsl::specialize::close(&f, &vals) {
        Ok(c) => c.derived,
        Err(e) => {
            eprintln!("{e}");
            return Err(ExitCode::from(2));
        }
    };
    match amdgpu_dsl::specialize::specialize(&f, &binds) {
        Ok(g) => Ok((g, (vals, derived))),
        Err(e) => {
            eprintln!("{e}");
            Err(ExitCode::from(2))
        }
    }
}

/// The strict certificate of a specialized kernel (stderr); Err when it does not verify.
fn certify_specialized(f: &amdgpu_dsl::Function) -> Result<(), ExitCode> {
    match amdgpu_codegen::gpu_verify::analyse(f) {
        Ok(r) => {
            let l = r.ledger();
            let c = &f.contract;
            let contracts = c.extents.len() + c.ka_ranges.len() + c.grid_defs.len() + c.ka_defs.len() + c.ka_sets.len() + c.sym_ranges.len();
            let errs = r.errors();
            eprintln!("verified {}: certificate: {} trusted assumes, {contracts} trusted contracts, {} unproven", f.name, l.total(), errs.len());
            if errs.is_empty() {
                Ok(())
            } else {
                for e in errs {
                    eprintln!("  [{}] {}", e.code, e.msg);
                }
                Err(ExitCode::from(1))
            }
        }
        Err(e) => {
            eprintln!("compile error:\n{e}");
            Err(ExitCode::from(1))
        }
    }
}

fn backend_cmd(args: &[String]) -> ExitCode {
    let tools = LlvmTools::locate();
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let report = match probe::probe(&tools) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("probe failed: {e}");
            return ExitCode::from(1);
        }
    };
    match sub {
        "probe" => {
            print!("{}", report.render_lock());
            match report.validate() {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("UNSAFE TAIL: {e}");
                    ExitCode::from(1)
                }
            }
        }
        "check" => {
            if let Err(e) = report.validate() {
                eprintln!("UNSAFE TAIL: {e}");
                return ExitCode::from(1);
            }
            let lock = match probe::Lock::find_and_load() {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(1);
                }
            };
            match lock.check(&report) {
                Ok(()) => {
                    println!("backend.lock OK: {} start={} tail={} passes", report.version, report.start_before, report.tail.len());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("BACKEND LOCK MISMATCH: {e}");
                    ExitCode::from(1)
                }
            }
        }
        "write-lock" => {
            if let Err(e) = report.validate() {
                eprintln!("UNSAFE TAIL: {e}");
                return ExitCode::from(1);
            }
            let path = probe::Lock::find().unwrap_or_else(|| "backend.lock".into());
            if let Err(e) = fs::write(&path, report.render_lock()) {
                eprintln!("cannot write {}: {e}", path.display());
                return ExitCode::from(1);
            }
            eprintln!("wrote {}", path.display());
            ExitCode::SUCCESS
        }
        _ => {
            usage();
            ExitCode::from(2)
        }
    }
}

/// `amdc mir-roundtrip <in.mir> <out.mir>`: parse all `body: |` sections and print them back.
fn mir_roundtrip_cmd(args: &[String]) -> ExitCode {
    let (Some(i), Some(o)) = (args.get(1), args.get(2)) else {
        usage();
        return ExitCode::from(2);
    };
    let text = match fs::read_to_string(i) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{i}: {e}");
            return ExitCode::from(1);
        }
    };
    use amdgpu_codegen::backend::mir_roundtrip::Level;
    let lv = match args.get(3).map(|s| s.as_str()) {
        Some("--level=flags") => Level::Flags,
        Some("--level=reflag") => Level::Reflag,
        Some("--level=all") => Level::All,
        _ => Level::Identity,
    };
    let (out, st) = amdgpu_codegen::backend::mir_roundtrip::roundtrip_text_level(&text, lv);
    if let Err(e) = fs::write(o, out) {
        eprintln!("{o}: {e}");
        return ExitCode::from(1);
    }
    println!(
        "STATS files=1 bodies={} insts={} parsed={} opaque={} bundles={} changed={}",
        st.bodies, st.insts, st.parsed, st.opaque, st.bundles, st.changed_lines
    );
    ExitCode::SUCCESS
}

/// `amdc mir-table-report <files...>`: how well does our opcode table match LLVM's own MIR?
fn mir_table_report_cmd(args: &[String]) -> ExitCode {
    let mut files = Vec::new();
    for f in &args[1..] {
        match fs::read_to_string(f) {
            Ok(t) => files.push((f.clone(), t)),
            Err(e) => eprintln!("{f}: {e}"),
        }
    }
    let t = amdgpu_codegen::backend::llvm20::Table::load();
    let r = amdgpu_codegen::backend::mir_roundtrip::table_report(&files, &t);
    println!("instructions={} matched_exact={} implicit_operands_differ={} explicit_shape_mismatch={} unknown_opcode={}", r.total, r.matched, r.implicit_differs, r.shape_mismatch, r.unknown_opcode);
    println!("matched opcodes: {}", r.matched_by_opcode.len());
    println!("-- explicit operand shape differs from the table row (opcode: example)");
    for (o, ex) in &r.mismatch_examples {
        println!("  {o}: {ex}");
    }
    let mut unk: Vec<_> = r.unknown_by_opcode.iter().collect();
    unk.sort_by(|a, b| b.1.cmp(a.1));
    println!("-- explicit operands match but the implicit operand set differs (opcode: instruction | implicit set)");
    for (o, ex) in &r.implicit_examples {
        println!("  {o}: {ex}");
    }
    println!("-- top unknown opcodes (not in the table)");
    for (o, n) in unk.iter().take(40) {
        println!("  {n:>6} {o}");
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "gen-host" => {
            // amdc gen-host <kernel>|--shipped [--out <dir>]
            let mut out = "build/hostgen".to_string();
            let mut i = 1usize;
            let mut which: Vec<String> = Vec::new();
            while i < args.len() {
                match args[i].as_str() {
                    "--out" => { i += 1; if let Some(d) = args.get(i) { out = d.clone(); } }
                    "--shipped" => which = ["fmha_fwd_hd128_mxfp8_v3_vs128", "fmha_fwd_hd128_mxfp8_v3_t128", "fmha_fwd_hd128_mxfp8_v3_kvsplit", "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"].iter().map(|s| s.to_string()).collect(),
                    k => which.push(k.to_string()),
                }
                i += 1;
            }
            if which.is_empty() { usage(); return ExitCode::from(2); }
            if let Err(e) = fs::create_dir_all(&out) { eprintln!("{out}: {e}"); return ExitCode::from(1); }
            let mut includes = String::from("// Generated by amdc gen-host --shipped. DO NOT EDIT.\n#pragma once\n");
            let mut aggregate = String::from("[\n");
            for (n, k) in which.iter().enumerate() {
                let Some(f) = amdgpu_kernels::kernel_by_name(k) else { eprintln!("unknown kernel: {k}"); return ExitCode::from(2); };
                let g = match hostgen::generate(&f) { Ok(g) => g, Err(e) => { eprintln!("{e}"); return ExitCode::from(1); } };
                let h = format!("{out}/{}_abi.h", g.file_stem);
                let j = format!("{out}/{}_abi.json", g.file_stem);
                if let Err(e) = fs::write(&h, &g.header).and_then(|_| fs::write(&j, &g.json)) { eprintln!("{e}"); return ExitCode::from(1); }
                includes.push_str(&format!("#include \"{}_abi.h\"\n", g.file_stem));
                if n > 0 { aggregate.push_str(",\n"); }
                aggregate.push_str(g.json.trim_end());
                println!("wrote {h} and {j}");
            }
            aggregate.push_str("\n]\n");
            let sj = format!("{out}/shipped_abi.json");
            let sh = format!("{out}/shipped_abi.h");
            if let Err(e) = fs::write(&sj, &aggregate).and_then(|_| fs::write(&sh, &includes)) { eprintln!("{e}"); return ExitCode::from(1); }
            println!("wrote {sh} and {sj}");
            ExitCode::SUCCESS
        }
        "opgraph" | "skeleton" => {
            use amdgpu_graph::opgraph::{skeleton, Budget, Cut, Graph, KernelCut};
            let Some(path) = args.get(1) else { usage(); return ExitCode::from(2); };
            let text = match fs::read_to_string(path) { Ok(t) => t, Err(e) => { eprintln!("{path}: {e}"); return ExitCode::from(1); } };
            let g = match Graph::from_json_str(&text) { Ok(g) => g, Err(e) => { eprintln!("{path}: {e}"); return ExitCode::from(1); } };
            let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
            let all: Vec<String> = g.nodes.iter().map(|n| n.id.clone()).collect();
            if args[0] == "opgraph" {
                let cut = match flag("--cut") {
                    None => Cut { kernels: vec![KernelCut { name: "all".into(), nodes: all }] },
                    Some(spec) => Cut { kernels: spec.split(',').filter_map(|k| k.split_once('=')).map(|(n, ns)| KernelCut { name: n.to_string(), nodes: ns.split('+').map(|s| s.to_string()).collect() }).collect() },
                };
                return match g.evaluate_cut(&cut, Budget::default()) {
                    Ok(r) => { print!("{}", r.text(&g)); ExitCode::SUCCESS }
                    Err(e) => { eprintln!("cut: {e}"); ExitCode::from(1) }
                };
            }
            let Some(kernel) = args.get(2) else { usage(); return ExitCode::from(2); };
            let nodes: Vec<String> = match flag("--nodes").as_deref() { None | Some("all") => all, Some(l) => l.split(',').map(|s| s.to_string()).collect() };
            let refs: Vec<&str> = nodes.iter().map(|s| s.as_str()).collect();
            let gf = flag("--graph-file").unwrap_or_else(|| format!("{}.graph.json", g.name));
            match skeleton::generate(&g, kernel, &refs, &gf, amdgpu_kernels::stage_registry::registry()) {
                Ok(src) => match flag("--out") {
                    Some(o) => match fs::write(&o, &src) { Ok(_) => { println!("wrote {o}"); ExitCode::SUCCESS } Err(e) => { eprintln!("{o}: {e}"); ExitCode::from(1) } },
                    None => { print!("{src}"); ExitCode::SUCCESS }
                },
                Err(e) => { eprintln!("skeleton: {e}"); ExitCode::from(1) }
            }
        }
        "graph" => {
            // amdc graph plan <graph.json | -> [--cus N] [--no-fuse]: the L1 launch plan of a model-level op graph (docs/l1-graph.md): fused kernels chosen by
            // the fusion rules, per-shape parameters from the dispatch rules. Exit 1 when an op has no kernel.
            if args.get(1).map(|s| s.as_str()) != Some("plan") { usage(); return ExitCode::from(2); }
            let Some(path) = args.get(2) else { usage(); return ExitCode::from(2); };
            let text = if path == "-" {
                let mut s = String::new();
                if let Err(e) = std::io::Read::read_to_string(&mut std::io::stdin(), &mut s) { eprintln!("stdin: {e}"); return ExitCode::from(1); }
                s
            } else {
                match fs::read_to_string(path) { Ok(t) => t, Err(e) => { eprintln!("{path}: {e}"); return ExitCode::from(1); } }
            };
            let mut g = match amdgpu_graph::fusion::ModelGraph::from_json_str(&text) { Ok(g) => g, Err(e) => { eprintln!("{path}: {e}"); return ExitCode::from(1); } };
            if let Some(c) = args.windows(2).find(|w| w[0] == "--cus").and_then(|w| w[1].parse::<u64>().ok()) { g.cus = c; }
            let p = amdgpu_graph::fusion::plan_with(&g, !args.iter().any(|a| a == "--no-fuse"));
            print!("{}", p.to_json().to_string_pretty());
            if p.errors.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) }
        }        "compose-facts" => {
            // amdc compose-facts [--out FILE]: the generated `amdgpu_graph::compose_facts` source (the CTA shape and rings of every kernel that declares a plan)
            let src = amdgpu_kernels::compose_facts_gen::source();
            match args.windows(2).find(|w| w[0] == "--out") {
                Some(w) => match std::fs::write(&w[1], src) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(e) => {
                        eprintln!("compose-facts: {}: {e}", w[1]);
                        ExitCode::from(1)
                    }
                },
                None => {
                    print!("{src}");
                    ExitCode::SUCCESS
                }
            }
        }
        "compose-search" => {
            // amdc compose-search gemm <kind> <M> <N> <K> [--cus N] [--margin X]: the feasible candidates of a GEMM launch with the coarse model's time, the conservative pruning, and the recorded rule's choice, as JSON
            // (docs/compose.md; the measured top-k is `tools/compose_search.py`)
            use amdgpu_graph::compose_search as cs;
            let num = |i: usize| args.get(i).and_then(|s| s.parse::<u64>().ok());
            let opt = |name: &str, d: f64| args.windows(2).find(|w| w[0] == name).and_then(|w| w[1].parse::<f64>().ok()).unwrap_or(d);
            match (args.get(1).map(|s| s.as_str()), args.get(2), num(3), num(4), num(5)) {
                (Some("gemm"), Some(kind), Some(m), Some(n), Some(k)) => {
                    let cus = opt("--cus", 256.0) as u64;
                    let cands = cs::gemm_candidates(kind, m, n, k, cus);
                    let kept = cs::prune(&cands, opt("--margin", 1.5));
                    let rule = amdgpu_graph::dispatch::gemm_kernels(kind, m, n, k, cus).map(|(kn, s, red)| format!("{{\"kernel\":\"{kn}\",\"splits\":{s},\"reduce\":{}}}", red.map(|r| format!("\"{r}\"")).unwrap_or_else(|| "null".into()))).unwrap_or_else(|| "null".into());
                    let rows: Vec<String> = cands.iter().zip(&kept).map(|(c, k)| c.to_json(*k)).collect();
                    println!("{{\"kind\":\"{kind}\",\"m\":{m},\"n\":{n},\"k\":{k},\"cus\":{cus},\"rule\":{rule},\"candidates\":[{}]}}", rows.join(","));
                    ExitCode::SUCCESS
                }
                (Some(fam @ ("moe" | "mla")), _, _, _, _) => {
                    // amdc compose-search moe <bf16|fp8|fp4> <M> <TOPK> <E> | mla <B> <H> <CTX> [--page P]
                    let cus = opt("--cus", 256.0) as u64;
                    let a = |i: usize| args.get(i).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
                    let (cands, rule, shape) = if fam == "moe" {
                        let dt = args.get(2).cloned().unwrap_or_default();
                        let (m, topk, e) = (a(3), a(4), a(5));
                        let bm = amdgpu_graph::dispatch::moe_block_m(m, topk, e, opt("--inter", 256.0) as u64, cus);
                        let sfx = if bm == 64 { "_bm64" } else { "" };
                        (cs::moe_candidates(&dt, m, topk, e, cus), format!("{{\"kernel\":\"gemm_moe1_{dt}_silu{sfx}\",\"splits\":1,\"reduce\":\"gemm_moe2_{dt}{sfx}\"}}"), format!("\"dt\":\"{dt}\",\"m\":{m},\"topk\":{topk},\"e\":{e}"))
                    } else {
                        let (b, h, ctx, page) = (a(2), a(3), a(4), opt("--page", 64.0) as u32);
                        let (hp, s) = amdgpu_graph::dispatch::mla_config(b, h, ctx);
                        let rk = format!("mla_decode_bf16_{}p{page}", if hp { "hp_" } else { "" });
                        (cs::mla_candidates(b, h, ctx, page, cus), format!("{{\"kernel\":\"{rk}\",\"splits\":{s},\"reduce\":\"attn_combine_bf16_hd512\"}}"), format!("\"b\":{b},\"h\":{h},\"ctx\":{ctx},\"page\":{page}"))
                    };
                    let kept = cs::prune(&cands, opt("--margin", 1.5));
                    let rows: Vec<String> = cands.iter().zip(&kept).map(|(c, k)| c.to_json(*k)).collect();
                    println!("{{\"kind\":\"{fam}\",{shape},\"cus\":{cus},\"rule\":{rule},\"candidates\":[{}]}}", rows.join(","));
                    ExitCode::SUCCESS
                }
                (Some(fam @ ("decode16" | "prefill_bf16")), _, _, _, _) => {
                    // amdc compose-search decode16 <B> <HK> <CTX> [--page P] | prefill_bf16 <B> <H> <SQ> <SK> <causal 0|1>
                    let cus = opt("--cus", 256.0) as u64;
                    let a = |i: usize| args.get(i).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
                    let (cands, rule, shape) = if fam == "decode16" {
                        let (b, hk, ctx, page) = (a(2), a(3), a(4), opt("--page", 16.0) as u32);
                        let (rk, rs) = amdgpu_graph::dispatch::decode16_kernel(page, amdgpu_graph::dispatch::decode16_plan(b, hk, ctx));
                        (cs::decode16_candidates(b, hk, ctx, page, cus), format!("{{\"kernel\":\"{rk}\",\"splits\":{rs},\"reduce\":null}}"), format!("\"b\":{b},\"hk\":{hk},\"ctx\":{ctx},\"page\":{page}"))
                    } else {
                        let (b, h, sq, sk, c) = (a(2), a(3), a(4), a(5), a(6) != 0);
                        let (rk, rs) = amdgpu_graph::dispatch::bf16_prefill_kernel(b, h, sq, sk, c, cus);
                        let red = if rs > 1 { "\"attn_combine_bf16_hd128_q16\"" } else { "null" };
                        (cs::prefill_bf16_candidates(b, h, sq, sk, c, cus), format!("{{\"kernel\":\"{rk}\",\"splits\":{rs},\"reduce\":{red}}}"), format!("\"b\":{b},\"h\":{h},\"sq\":{sq},\"sk\":{sk},\"causal\":{c}"))
                    };
                    let kept = cs::prune(&cands, opt("--margin", 1.5));
                    let rows: Vec<String> = cands.iter().zip(&kept).map(|(c, k)| c.to_json(*k)).collect();
                    println!("{{\"kind\":\"{fam}\",{shape},\"cus\":{cus},\"rule\":{rule},\"candidates\":[{}]}}", rows.join(","));
                    ExitCode::SUCCESS
                }
                _ => {
                    eprintln!("usage: amdc compose-search gemm <bf16|i8_rowcol|fp8_rowcol|fp8_blockscale> <M> <N> <K> [--cus N] [--margin X] | decode16 <B> <HK> <CTX> [--page P] | prefill_bf16 <B> <H> <SQ> <SK> <causal>");
                    ExitCode::from(2)
                }
            }
        }
        "compose" => {
            // amdc compose <family> <shape>: the composition of a launch (docs/compose.md): kernels, splits, tile and the plan facts (CTA shape, rings) of the main kernel, as JSON
            use amdgpu_graph::compose as cp;
            let num = |i: usize| args.get(i).and_then(|s| s.parse::<u64>().ok());
            let opt = |name: &str, d: u64| args.windows(2).find(|w| w[0] == name).and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(d);
            let c = match (args.get(1).map(|s| s.as_str()), num(2), num(3), num(4)) {
                (Some("decode16"), Some(b), Some(hk), Some(ctx)) => Some(cp::decode16(b, hk, ctx, opt("--page", 16) as u32)),
                (Some("prefill_bf16"), Some(b), Some(h), Some(sq)) => Some(cp::prefill_bf16(b, h, sq, num(5).unwrap_or(sq), num(6).unwrap_or(0) != 0, opt("--cus", 256))),
                (Some("fp8pt"), Some(b), Some(h), Some(sq)) => Some(cp::fp8pt(b, h, sq, num(5).unwrap_or(sq), num(6).unwrap_or(0) != 0, opt("--cus", 256))),
                (Some("hd192"), Some(b), Some(h), Some(sq)) => Some(cp::hd192(b, h, sq, num(5).unwrap_or(sq), num(6).unwrap_or(0) != 0, opt("--cus", 256))),
                (Some("mxfp8"), Some(b), Some(h), Some(sq)) => Some(cp::mxfp8(b, h, sq, num(5).unwrap_or(sq))),
                (Some("mla"), Some(b), Some(h), _) => Some(cp::mla(b, h, args.windows(2).find(|w| w[0] == "--ctx").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(u64::MAX), opt("--page", 64) as u32)),
                (Some("bwd_causal"), _, _, _) => Some(cp::bwd_causal()),
                (Some("gemm"), _, _, _) => match (args.get(2), num(3), num(4), num(5)) {
                    (Some(kind), Some(m), Some(n), Some(k)) => cp::gemm(kind, m, n, k, opt("--cus", 256)),
                    _ => None,
                },
                (Some("moe"), _, _, _) => match (args.get(2), num(3), num(4), num(5)) {
                    (Some(dt), Some(m), Some(topk), Some(e)) => Some(cp::moe(dt, m, topk, e, args.windows(2).find(|w| w[0] == "--inter").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(u64::MAX))),
                    _ => None,
                },
                _ => None,
            };
            match c {
                Some(c) => {
                    println!("{}", c.to_json());
                    ExitCode::SUCCESS
                }
                None => {
                    eprintln!("usage: amdc compose decode16 <B> <HK> <CTX> [--page P] | prefill_bf16|fp8pt|hd192 <B> <H> <SQ> <SK> <causal> [--cus N] | mxfp8 <B> <H> <S> [SK] | gemm <kind> <M> <N> <K> [--cus N] | mla <B> <H> [--page P] | moe <bf16|fp8|fp4> <M> <TOPK> <E> | bwd_causal");
                    ExitCode::from(2)
                }
            }
        }
        "dispatch" => {
            // amdc dispatch decode16 <B> <HK> <CTX> [--page P]: the launch plan of the paged decode (dispatch::decode16_plan) as JSON - the single
            // rule the hosts (tools/pa_auto.sh, python amdgpu_jit.decode16_plan) follow
            let num = |i: usize| args.get(i).and_then(|s| s.parse::<u64>().ok());
            match (args.get(1).map(|s| s.as_str()), num(2), num(3), num(4)) {
                (Some("decode16"), Some(b), Some(hk), Some(ctx)) => {
                    let page = args.windows(2).find(|w| w[0] == "--page").and_then(|w| w[1].parse::<u32>().ok()).unwrap_or(16);
                    let plan = amdgpu_graph::dispatch::decode16_plan(b, hk, ctx);
                    let (kernel, s) = amdgpu_graph::dispatch::decode16_kernel(page, plan);
                    let form = match plan {
                        amdgpu_graph::dispatch::Decode16Plan::Fused => "fused",
                        amdgpu_graph::dispatch::Decode16Plan::LastC(_) => "lastc",
                    };
                    println!("{{\"kernel\":\"{kernel}\",\"form\":\"{form}\",\"splits\":{s},\"b\":{b},\"hk\":{hk},\"ctx\":{ctx},\"page\":{page}}}");
                    ExitCode::SUCCESS
                }
                (Some("prefill_bf16"), Some(b), Some(h), Some(sq)) => {
                    // amdc dispatch prefill_bf16 <B> <H> <SQ> <SK> <causal 0|1> [--cus N]
                    let (sk, causal) = (num(5).unwrap_or(sq), num(6).unwrap_or(0) != 0);
                    let cus = args.windows(2).find(|w| w[0] == "--cus").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(256);
                    let (kernel, s) = amdgpu_graph::dispatch::bf16_prefill_kernel(b, h, sq, sk, causal, cus);
                    println!("{{\"kernel\":\"{kernel}\",\"splits\":{s}}}");
                    ExitCode::SUCCESS
                }
                (Some(fam @ ("fp8pt" | "hd192")), Some(b), Some(h), Some(sq)) => {
                    // amdc dispatch fp8pt|hd192 <B> <H> <SQ> <SK> <causal 0|1> [--cus N]: the per-tensor FP8 / hd192 (MLA prefill) kernel and key splits
                    let (sk, causal) = (num(5).unwrap_or(sq), num(6).unwrap_or(0) != 0);
                    let cus = args.windows(2).find(|w| w[0] == "--cus").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(256);
                    let (kernel, s) = if fam == "fp8pt" { amdgpu_graph::dispatch::fp8pt_prefill_kernel(b, h, sq, sk, causal, cus) } else { amdgpu_graph::dispatch::hd192_prefill_kernel(b, h, sq, sk, causal, cus) };
                    println!("{{\"kernel\":\"{kernel}\",\"splits\":{s}}}");
                    ExitCode::SUCCESS
                }
                (Some("gemm_blockscale"), Some(m), Some(n), Some(k)) => {
                    // amdc dispatch gemm_blockscale <M> <N> <K> [--cus N]: the split-K factor of gemm_a8w8_blockscale
                    let cus = args.windows(2).find(|w| w[0] == "--cus").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(256);
                    { let (rows, s) = amdgpu_graph::dispatch::blockscale_config(m, n, k, cus); println!("{{\"rows\":{},\"splits\":{}}}", rows, s); }
                    ExitCode::SUCCESS
                }
                (Some("mxfp8"), Some(b), Some(h), Some(s)) => {
                    // amdc dispatch mxfp8 <B> <H> <S> <SK>: the MXFP8 prefill path and splits (dispatch::mxfp8_prefill_path)
                    let sk = num(5).unwrap_or(s);
                    let (path, sp) = amdgpu_graph::dispatch::mxfp8_prefill_path(b, h, s, sk);
                    let ks: Vec<String> = amdgpu_graph::dispatch::mxfp8_prefill_kernels(path).iter().map(|k| format!("\"{k}\"")).collect();
                    println!("{{\"path\":\"{path}\",\"splits\":{sp},\"kernels\":[{}]}}", ks.join(","));
                    ExitCode::SUCCESS
                }                (Some("gemm"), _, _, _) => {
                    // amdc dispatch gemm <kind> <M> <N> <K> [--cus N]: the dense GEMM launch (dispatch::gemm_kernels: kernel, split-K, reduce) -
                    // the one rule of the Python gemm() and the L1 graph plans
                    let cus = args.windows(2).find(|w| w[0] == "--cus").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(256);
                    let kind = args.get(2).map(|s| s.as_str()).unwrap_or("");
                    match (num(3), num(4), num(5)) {
                        (Some(m), Some(n), Some(k)) => match amdgpu_graph::dispatch::gemm_kernels(kind, m, n, k, cus) {
                            Some((kernel, s, reduce)) => {
                                let red = reduce.map(|r| format!("\"{r}\"")).unwrap_or_else(|| "null".into());
                                println!("{{\"kernel\":\"{kernel}\",\"splits\":{s},\"reduce\":{red}}}");
                                ExitCode::SUCCESS
                            }
                            None => { eprintln!("dispatch gemm: unknown kind `{kind}`"); ExitCode::from(2) }
                        },
                        _ => { usage(); ExitCode::from(2) }
                    }
                }                (Some("mla"), Some(b), Some(h), _) => {
                    // amdc dispatch mla <B> <H> [--ctx C]: (head-parallel form?, splits) of the MLA decode; C = the largest context (the block table's capacity); omitted = unknown
                    let ctx = args.windows(2).find(|w| w[0] == "--ctx").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(u64::MAX);
                    let (hp, s) = amdgpu_graph::dispatch::mla_config(b, h, ctx);
                    println!("{{\"hp\":{hp},\"splits\":{s}}}");
                    ExitCode::SUCCESS
                }
                (Some("bwd_causal"), _, _, _) => {
                    // amdc dispatch bwd_causal: the causal attention-backward dq / dkdv kernels (longest-first grids)
                    let (dq, dkdv) = amdgpu_graph::dispatch::bwd_causal_kernels();
                    println!("{{\"dq\":\"{dq}\",\"dkdv\":\"{dkdv}\"}}");
                    ExitCode::SUCCESS
                }
                (Some("moe"), Some(m), Some(topk), Some(e)) => {
                    // amdc dispatch moe <M> <TOPK> <E> [--inter I] [--cus N]: block_m of the routing sort and the gemm_moe kernels (I omitted = unknown: the measured band is off)
                    println!("{{\"block_m\":{}}}", amdgpu_graph::dispatch::moe_block_m(m, topk, e, args.windows(2).find(|w| w[0] == "--inter").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(u64::MAX), args.windows(2).find(|w| w[0] == "--cus").and_then(|w| w[1].parse::<u64>().ok()).unwrap_or(256)));
                    ExitCode::SUCCESS
                }
                _ => {
                    eprintln!("usage: amdc dispatch decode16 <B> <HK> <CTX> [--page P] | prefill_bf16|fp8pt|hd192 <B> <H> <SQ> <SK> <causal> [--cus N] | gemm_blockscale <M> <N> <K> [--cus N] | moe <M> <TOPK> <E> | mla <B> <H> | bwd_causal");
                    ExitCode::from(2)
                }
            }
        }
        "list" => {
            for k in amdgpu_kernels::kernel_names() {
                println!("{k}");
            }
            ExitCode::SUCCESS
        }
        "gpu-verify" => {
            // amdc gpu-verify <kernel|--all> [--strict]: the GPU-semantics report (docs/gpu-semantics-verifier.md); exit 1 when --strict and an error was found
            let strict = args.iter().any(|a| a == "--strict");
            let names: Vec<String> = match args.get(1).map(|s| s.as_str()) {
                Some("--all") => amdgpu_kernels::kernel_names().iter().map(|s| s.to_string()).collect(),
                // the shipped default path (`fmha_fwd_hd128_mxfp8_auto`): the four v3 kernels (crates/amdgpu-tile/src/fmha.rs)
                Some("--shipped") => ["fmha_fwd_hd128_mxfp8_v3_vs128", "fmha_fwd_hd128_mxfp8_v3_t128", "fmha_fwd_hd128_mxfp8_v3_kvsplit", "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"].iter().map(|s| s.to_string()).collect(),
                Some(n) => vec![n.to_string()],
                None => {
                    usage();
                    return ExitCode::from(2);
                }
            };
            let mut bad = false;
            let (mut tot_trusted, mut tot_contracts, mut tot_unproven) = (0usize, 0usize, 0usize);
            if args.iter().any(|a| a == "--json") {
                // structured findings (docs/diagnostics.md): one JSON object per kernel
                let mut objs = Vec::new();
                for name in names {
                    let (f, _) = match kernel_spec(&name, &args) {
                        Ok(x) => x,
                        Err(c) => return c,
                    };
                    let ds = match amdgpu_codegen::gpu_verify::analyse(&f) {
                        Ok(r) => {
                            bad |= !r.errors().is_empty();
                            amdgpu_codegen::diag::from_report(&f, &r)
                        }
                        Err(e) => {
                            bad = true;
                            amdgpu_codegen::diag::from_compile_error(&f, &e)
                        }
                    };
                    objs.push(format!("{{\"kernel\":\"{}\",\"diagnostics\":{}}}", f.name, amdgpu_codegen::diag::to_json(&ds)));
                }
                println!("[{}]", objs.join(","));
                return if strict && bad { ExitCode::from(1) } else { ExitCode::SUCCESS };
            }
            for name in names {
                let (f, _) = match kernel_spec(&name, &args) {
                    Ok(x) => x,
                    Err(c) => return c,
                };
                match amdgpu_codegen::gpu_verify::analyse(&f) {
                    Ok(r) => {
                        let errs = r.errors().len();
                        println!("== {}: {} finding(s), {errs} error(s); proven {:?}", f.name, r.findings.len(), r.proven);
                        for fd in &r.findings {
                            if args.iter().any(|a| a == "-v") || fd.severity != amdgpu_codegen::gpu_verify::Severity::Info {
                                println!("  [{}{}] {:?}: {}", fd.code, if fd.alias.is_empty() || fd.alias == fd.code { String::new() } else { format!("|{}", fd.alias) }, fd.severity, fd.msg);
                            }
                        }
                        // the trust LEDGER: every escape is counted by kind
                        let l = r.ledger();
                        let trusted = l.total();
                        let c = &f.contract;
                        let contracts = c.extents.len() + c.ka_ranges.len() + c.grid_defs.len() + c.ka_defs.len() + c.ka_sets.len() + c.sym_ranges.len();
                        println!("   certificate: {trusted} trusted assumes ({} assume_trusted, {} tile_trust_periodic, {} region_trust, {} asm_opaque, {} timing), {contracts} trusted contracts, {errs} unproven", l.assume_trusted, l.tile_trust, l.region_trust, l.asm_opaque, l.timing_trust);
                        if !r.unchecked.is_empty() {
                            println!("   bounds-checked, not race-checked (docs/trust-inventory.md): {:?}", r.unchecked);
                        }                        tot_trusted += trusted;
                        tot_contracts += contracts;
                        tot_unproven += errs;
                        bad |= errs > 0;
                    }
                    Err(e) => {
                        println!("== {name}: compile error: {e}");
                        bad = true;
                    }
                }
            }
            println!("certificate (all kernels listed): {tot_trusted} trusted assumes, {tot_contracts} trusted contracts, {tot_unproven} unproven");
            if strict && bad {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        "srcmap" => {
            // amdc srcmap <kernel> [--line N]: own-backend asm lines -> IR value -> Rust builder call (JSON; docs/diagnostics.md)
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, _) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            let only = args.windows(2).find(|w| w[0] == "--line").and_then(|w| w[1].parse::<usize>().ok());
            match amdgpu_codegen::diag::asm_map_json(&f, only) {
                Ok(j) => {
                    println!("{j}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    println!("{{\"kernel\":\"{}\",\"diagnostics\":{}}}", f.name, amdgpu_codegen::diag::to_json(&amdgpu_codegen::diag::from_compile_error(&f, &e)));
                    ExitCode::from(1)
                }
            }
        }
        "exact-report" => {
            // amdc exact-report <kernel> [--region <name>]: per IR statement, the final instructions it produced, with the ones the back end
            // added on its own marked (docs/exact-mode.md, step E1)
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, _) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            let region = args.iter().position(|a| a == "--region").and_then(|i| args.get(i + 1)).map(|s| s.as_str());
            match amdgpu_codegen::compile_mir_alloc(&f) {
                Ok((m, a)) => {
                    let stmts = amdgpu_codegen::lower::last_stmts();
                    print!("{}", amdgpu_codegen::exact::report(&f, &stmts, &m, &a, region));
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error: {e}");
                    ExitCode::from(1)
                }
            }
        }
        "weave-report" => {
            // amdc weave-report <kernel>|--from-ir <file>: the back-end Weaver's unit graph of the final MIR (docs/two-weavers.md, step B1)
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, _) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            match amdgpu_codegen::compile_mir(&f) {
                Ok(m) => {
                    let g = amdgpu_codegen::bweave::build(&m);
                    print!("{}", amdgpu_codegen::bweave::report(&m, &g, f.plan.as_ref()));
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::from(1)
                }
            }
        }
        "passes" => {
            // amdc passes [--target T] [--evidence]: the compile pipeline of a target (named passes, stage, target dependence, gate, evidence; the retry ladders)
            let target = args.windows(2).find(|w| w[0] == "--target").map(|w| w[1].clone()).unwrap_or_else(|| amdgpu_codegen::target::T.name.to_string());
            match amdgpu_codegen::pass::describe(&target, args.iter().any(|a| a == "--evidence")) {
                Ok(s) => {
                    print!("{s}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::from(2)
                }
            }
        }
        "md5-manifest" => {
            // amdc md5-manifest [--out FILE] [--check FILE] [--jobs N]: the own-backend asm md5 of every reachable kernel (the refactor's golden oracle)
            let opt = |flag: &str| args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone());
            let jobs = opt("--jobs").and_then(|j| j.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
            let now = match amdc::manifest::compute(jobs) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            if let Some(p) = opt("--check") {
                let golden = match fs::read_to_string(&p) {
                    Ok(s) => amdc::manifest::parse(&s),
                    Err(e) => {
                        eprintln!("cannot read {p}: {e}");
                        return ExitCode::from(2);
                    }
                };
                let d = amdc::manifest::diff(&golden, &now);
                for l in &d {
                    println!("{l}");
                }
                println!("{} kernels, {} difference(s) against {p}", now.len(), d.len());
                return if d.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) };
            }
            let text = amdc::manifest::render(&now);
            match opt("--out") {
                Some(p) => {
                    if let Err(e) = fs::write(&p, &text) {
                        eprintln!("cannot write {p}: {e}");
                        return ExitCode::from(1);
                    }
                    eprintln!("wrote {p} ({} kernels, {} not compiling)", now.len(), now.iter().filter(|(_, v)| v.len() != 32).count());
                }
                None => print!("{text}"),
            }
            ExitCode::SUCCESS
        }
        "check-launch" => {
            // amdc check-launch <kernel> [--ka OFF=V,...] [--grid X,Y,Z] [--size ARG=BYTES,...] [--sym NAME=V,...]: the host contract of the
            // kernel (the facts its certificate assumes) evaluated for one concrete launch - for launchers that do not go through Python
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, _) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            let opt = |flag: &str| args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone()).unwrap_or_default();
            let parse_i = |s: &str| -> Option<i64> {
                let s = s.trim();
                if let Some(h) = s.strip_prefix("0x") { i64::from_str_radix(h, 16).ok() } else { s.parse().ok() }
            };
            let mut ka: std::collections::HashMap<u32, i64> = Default::default();
            for kv in opt("--ka").split(',').filter(|s| !s.is_empty()) {
                let Some((k, v)) = kv.split_once('=') else { eprintln!("--ka: OFF=V"); return ExitCode::from(2) };
                match (parse_i(k), parse_i(v)) {
                    (Some(k), Some(v)) => {
                        ka.insert(k as u32, v);
                    }
                    _ => {
                        eprintln!("--ka: bad entry {kv}");
                        return ExitCode::from(2);
                    }
                }
            }
            let mut grid = [1i64; 3];
            for (i, g) in opt("--grid").split(',').filter(|s| !s.is_empty()).enumerate().take(3) {
                grid[i] = parse_i(g).unwrap_or(1);
            }
            let mut sizes: std::collections::HashMap<String, i64> = Default::default();
            for kv in opt("--size").split(',').filter(|s| !s.is_empty()) {
                if let Some((k, v)) = kv.split_once('=') {
                    sizes.insert(k.to_string(), parse_i(v).unwrap_or(-1));
                }
            }
            let mut syms: std::collections::HashMap<String, i64> = Default::default();
            for kv in opt("--sym").split(',').filter(|s| !s.is_empty()) {
                if let Some((k, v)) = kv.split_once('=') {
                    syms.insert(k.to_string(), parse_i(v).unwrap_or(-1));
                }
            }
            let c = &f.contract;
            // symbols a definition names directly (grid_is(axis, B), ka_is(off, G)) are read off the launch
            for (ax, e) in &c.grid_defs {
                if let amdgpu_dsl::Ext::Sym(n) = e {
                    syms.entry(n.clone()).or_insert(grid[*ax as usize]);
                }
            }
            // ... and `sym * c` / `c * sym` definitions are solved for the symbol
            fn solve(e: &amdgpu_dsl::Ext, v: i64, syms: &mut std::collections::HashMap<String, i64>) {
                use amdgpu_dsl::Ext as E;
                match e {
                    E::Sym(n) => {
                        syms.entry(n.clone()).or_insert(v);
                    }
                    E::Mul(a, b) => match (a.as_ref(), b.as_ref()) {
                        (x, E::Const(k)) | (E::Const(k), x) if *k != 0 && v % k == 0 => solve(x, v / k, syms),
                        _ => {}
                    },
                    _ => {}
                }
            }
            for (off, e) in &c.ka_defs {
                if let Some(v) = ka.get(off) {
                    solve(e, *v, &mut syms);
                }
            }
            for (ax, e) in &c.grid_defs {
                solve(e, grid[*ax as usize], &mut syms);
            }
            fn ev(e: &amdgpu_dsl::Ext, ka: &std::collections::HashMap<u32, i64>, grid: &[i64; 3], syms: &std::collections::HashMap<String, i64>) -> Result<i64, String> {
                use amdgpu_dsl::Ext as E;
                Ok(match e {
                    E::Const(c) => *c,
                    E::Ka(o) => *ka.get(o).ok_or(format!("kernarg 0x{o:x} not given (--ka 0x{o:x}=V)"))?,
                    E::Grid(a) => grid[*a as usize],
                    E::Sym(n) => *syms.get(n).ok_or(format!("symbol {n} not given (--sym {n}=V)"))?,
                    E::Add(a, b) => ev(a, ka, grid, syms)? + ev(b, ka, grid, syms)?,
                    E::Sub(a, b) => ev(a, ka, grid, syms)? - ev(b, ka, grid, syms)?,
                    E::Mul(a, b) => ev(a, ka, grid, syms)? * ev(b, ka, grid, syms)?,
                    E::CeilDiv(a, k) => (ev(a, ka, grid, syms)? + k - 1).div_euclid(*k),
                    E::FloorDiv(a, k) => ev(a, ka, grid, syms)?.div_euclid(*k),
                })
            }
            let mut bad: Vec<String> = Vec::new();
            let mut ok = 0usize;
            let mut skipped: Vec<String> = Vec::new();
            for (n, lo, hi) in &c.sym_ranges {
                match syms.get(n) {
                    Some(v) if v < lo || v > hi => bad.push(format!("symbol {n} = {v} outside [{lo}, {hi}]")),
                    Some(_) => ok += 1,
                    None => skipped.push(format!("symbol {n} (not given)")),
                }
            }
            for (ax, e) in &c.grid_defs {
                match ev(e, &ka, &grid, &syms) {
                    Ok(w) if w != grid[*ax as usize] => bad.push(format!("grid[{ax}] = {}, the contract requires {e:?} = {w}", grid[*ax as usize])),
                    Ok(_) => ok += 1,
                    Err(m) => skipped.push(format!("grid[{ax}]: {m}")),
                }
            }
            for (off, e) in &c.ka_defs {
                match (ka.get(off), ev(e, &ka, &grid, &syms)) {
                    (Some(v), Ok(w)) if *v != w => bad.push(format!("kernarg 0x{off:x} = {v}, the contract requires {e:?} = {w}")),
                    (Some(_), Ok(_)) => ok += 1,
                    (None, _) => skipped.push(format!("kernarg 0x{off:x} (not given)")),
                    (_, Err(m)) => skipped.push(format!("kernarg 0x{off:x}: {m}")),
                }
            }
            for (off, lo, hi) in &c.ka_ranges {
                match ka.get(off) {
                    Some(v) if v < lo || v > hi => bad.push(format!("kernarg 0x{off:x} = {v} outside [{lo}, {hi}]")),
                    Some(_) => ok += 1,
                    None => skipped.push(format!("kernarg 0x{off:x} (not given)")),
                }
            }
            for (off, vals) in &c.ka_sets {
                match ka.get(off) {
                    Some(v) if !vals.contains(v) => bad.push(format!("kernarg 0x{off:x} = {v} not one of {vals:?}")),
                    Some(_) => ok += 1,
                    None => skipped.push(format!("kernarg 0x{off:x} (not given)")),
                }
            }
            for x in &c.extents {
                match (sizes.get(&x.arg), ev(&x.alloc, &ka, &grid, &syms)) {
                    (Some(s), Ok(need)) if *s < need => bad.push(format!("argument `{}`: {s} bytes allocated, the contract needs {:?} = {need}", x.arg, x.alloc)),
                    (Some(_), Ok(_)) => ok += 1,
                    (None, _) => skipped.push(format!("argument `{}` size (--size {}=BYTES)", x.arg, x.arg)),
                    (_, Err(m)) => skipped.push(format!("argument `{}`: {m}", x.arg)),
                }
            }
            for b in &bad {
                println!("VIOLATED  {b}");
            }
            for s in &skipped {
                println!("unchecked {s}");
            }
            println!("{}: {ok} contract fact(s) hold, {} violated, {} not checked (missing inputs)", f.name, bad.len(), skipped.len());
            if bad.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) }
        }
        "nops" => {
            // amdc nops <kernel> [--sites]: every s_nop of the final code classified by hazard rule / producer -> consumer (counts, wait states)
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, _) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            match amdgpu_codegen::compile_mir_alloc(&f) {
                Ok((m, _)) => {
                    let sites = amdgpu_codegen::hazards::classify_nops(&m);
                    let mut agg: std::collections::BTreeMap<(String, String, String), (usize, usize)> = std::collections::BTreeMap::new();
                    for s in &sites {
                        let e = agg.entry((s.rule.to_string(), s.producer.map_or("-".into(), |x| format!("{x:?}")), s.consumer.map_or("-".into(), |x| format!("{x:?}")))).or_insert((0, 0));
                        e.0 += 1;
                        e.1 += s.states;
                    }
                    let mut rows: Vec<_> = agg.into_iter().collect();
                    rows.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
                    let (n, w): (usize, usize) = (sites.len(), sites.iter().map(|s| s.states).sum());
                    println!("{}: {n} s_nop, {w} wait states (static)", f.name);
                    println!("  {:>5} {:>6}  {:<40} {:<26} {}", "nops", "states", "rule", "producer", "consumer");
                    for ((rule, p, c), (k, st)) in rows {
                        println!("  {k:>5} {st:>6}  {rule:<40} {p:<26} {c}");
                    }
                    if args.iter().any(|a| a == "--sites") {
                        for s in &sites {
                            let loc = if s.origin != u32::MAX { f.value_loc(s.origin).unwrap_or_default() } else { String::new() };
                            println!("  site b{} i{} s_nop {}: {} {:?} -> {:?} {loc}", s.block, s.index, s.states - 1, s.rule, s.producer, s.consumer);
                        }
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::from(1)
                }
            }
        }
        "summary" => {
            // amdc summary <kernel> [--json]: resources, per-region instruction mix, waits / nops, certificate counts (docs/diagnostics.md)
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, _) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            match amdgpu_codegen::diag::summary_json(&f) {
                Ok(j) => {
                    println!("{j}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    println!("{{\"kernel\":\"{}\",\"diagnostics\":{}}}", f.name, amdgpu_codegen::diag::to_json(&amdgpu_codegen::diag::from_compile_error(&f, &e)));
                    ExitCode::from(1)
                }
            }
        }
        "mops" => {
            // amdc mops <kernel>: the distinct machine opcodes (MOp variants) of the final MIR of a kernel, one per line (the coverage sweep tools/llvm_opcode_sweep.sh uses it)
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let Some(f) = amdgpu_kernels::kernel_by_name(name) else {
                eprintln!("unknown kernel: {name}");
                return ExitCode::from(2);
            };
            match amdgpu_codegen::compile_mir(&f) {
                Ok(m) => {
                    let mut set = std::collections::BTreeSet::new();
                    for b in &m.blocks {
                        for i in &b.insts {
                            set.insert(format!("{:?}", i.op));
                        }
                    }
                    for s in set {
                        println!("{s}");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        "backend" => backend_cmd(&args),
        "mir-roundtrip" => mir_roundtrip_cmd(&args),
        "mir-table-report" => mir_table_report_cmd(&args),
        "simmodel" => {
            // amdc simmodel <kernel> [--sched] : model cycles/iteration of every loop for 1,2 waves per SIMD
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let Some(f) = amdgpu_kernels::kernel_by_name(name) else {
                eprintln!("unknown kernel: {name}");
                return ExitCode::from(2);
            };
            if args.iter().any(|a| a == "--sched") {
                env::set_var("AMDC_SCHED", "1");
            }
            match amdgpu_codegen::compile_mir(&f) {
                Ok(m) => {
                    for r in amdgpu_codegen::sched::sim::report(&m, &[1, 2]) {
                        println!("SIM {} loop@{} blocks={} insts={} {}", name, r.header_label, r.blocks, r.insts,
                            r.cycles_per_iter.iter().map(|(w, c)| format!("waves{}={:.1}", w, c)).collect::<Vec<_>>().join(" "));
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        "print-ir" => {
            let name = args.get(1).cloned().unwrap_or_default();
            if name.is_empty() || (name.starts_with("--") && from_ir_path(&args).is_none()) { usage(); return ExitCode::from(2); }
            let stage = args.windows(2).find(|w| w[0] == "--stage").map(|w| w[1].as_str()).unwrap_or("dsl");
            let f = match load_kernel(&name, &args) { Ok(f) => f, Err(c) => return c };
            let text = match stage {
                "dsl" => amdgpu_codegen::ir_print::print_dsl(&f),
                "mir" => match amdgpu_codegen::lower::lower(&f) {
                    Ok(m) => amdgpu_codegen::ir_print::print_mir(&m, None),
                    Err(e) => { eprintln!("lower failed: {e}"); return ExitCode::from(1) }
                },
                "mir-ra" => match amdgpu_codegen::compile_mir_alloc(&f) {
                    Ok((m, a)) => amdgpu_codegen::ir_print::print_mir(&m, Some(&a)),
                    Err(e) => { eprintln!("compile failed: {e}"); return ExitCode::from(1) }
                },
                _ => { eprintln!("unknown stage {stage} (dsl|mir|mir-ra)"); return ExitCode::from(2) }
            };
            // --out <file>: write the file directly (large kernels through a container's stdout pipe can be cut short)
            match args.windows(2).find(|w| w[0] == "--out").map(|w| w[1].clone()) {
                Some(p) => {
                    if let Err(e) = fs::write(&p, &text) {
                        eprintln!("cannot write {p}: {e}");
                        return ExitCode::from(1);
                    }
                }
                None => {
                    use std::io::Write;
                    let mut o = std::io::stdout().lock();
                    let _ = o.write_all(text.as_bytes());
                    let _ = o.flush();
                }
            }
            ExitCode::SUCCESS
        }
        "ir-check" => {
            // amdc ir-check <file.ir> [--verify]: parse + check an irv1 file, re-print it (text fixpoint); --verify adds the strict GPU-semantics certificate
            let Some(path) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let text = match fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::from(2);
                }
            };
            let json = args.iter().any(|a| a == "--json");
            let f = match amdgpu_dsl::ir_text::parse(&text) {
                Ok(f) => f,
                Err(e) => {
                    if json {
                        let d = amdgpu_codegen::diag::Diag { code: "IR-PARSE".into(), severity: "error".into(), message: format!("{path}: {e}"), ..Default::default() };
                        println!("{{\"file\":\"{path}\",\"ok\":false,\"diagnostics\":{}}}", amdgpu_codegen::diag::to_json(&[d]));
                    } else {
                        eprintln!("{path}: {e}");
                    }
                    return ExitCode::from(2);
                }
            };
            if json {
                // structured: the IR verifier's findings (statement-localized) and the strict certificate's when --verify
                let mut ds: Vec<amdgpu_codegen::diag::Diag> = amdgpu_dsl::verify(&f)
                    .into_iter()
                    .map(|m| amdgpu_codegen::diag::localize(&f, amdgpu_codegen::diag::Diag { code: if m.contains("not provably wave-uniform") { "IR-UNIFORM-BRANCH".into() } else { "IR-VERIFY".into() }, severity: "error".into(), message: m, ..Default::default() }))
                    .collect();
                if args.iter().any(|a| a == "--verify") {
                    match amdgpu_codegen::gpu_verify::analyse(&f) {
                        Ok(r) => ds.extend(amdgpu_codegen::diag::from_report(&f, &r)),
                        Err(e) => ds.extend(amdgpu_codegen::diag::from_compile_error(&f, &e)),
                    }
                }
                let ok = !ds.iter().any(|d| d.severity == "error");
                println!("{{\"file\":\"{path}\",\"kernel\":\"{}\",\"ok\":{ok},\"statements\":{},\"values\":{},\"diagnostics\":{}}}", f.name, f.stmts().len(), f.value_tys.len(), amdgpu_codegen::diag::to_json(&ds));
                return if ok { ExitCode::SUCCESS } else { ExitCode::from(1) };
            }
            let again = amdgpu_dsl::ir_text::print(&f);
            let fix = amdgpu_dsl::ir_text::parse(&again).map(|g| format!("{g:?}") == format!("{f:?}")).unwrap_or(false);
            println!(
                "{path}: ok - kernel `{}`, {} statements, {} values, {} loop variables, {} args{}{}",
                f.name,
                f.stmts().len(),
                f.value_tys.len(),
                f.var_tys.len(),
                f.args.len(),
                if fix { "; re-print round-trips" } else { "; WARNING: re-print does not round-trip" },
                if f.compile_profile.is_none() { "; note: profile None - the compile policy is derived from the kernel NAME" } else { "" }
            );
            if args.iter().any(|a| a == "--verify") {
                if let Err(c) = certify_specialized(&f) {
                    return c;
                }
            }
            if fix {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        "regplan" => {
            // amdc regplan <kernel> [--specialize ..]: per weave-region / stage register peaks of the allocated kernel + the plan's report
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, _) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            let json = args.iter().any(|a| a == "--json");
            match amdgpu_codegen::compile_mir_alloc(&f) {
                Ok((m, a)) if json => {
                    // virtual register -> the IR value whose lowering defined it -> the builder call (diagnostics origin)
                    let mut origin: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
                    for blk in &m.blocks {
                        for i in &blk.insts {
                            if let (Some(d), true) = (i.dst, i.origin != u32::MAX) {
                                origin.entry(m.canon(d)).or_insert(i.origin);
                            }
                        }
                    }
                    let src_of = |s: &str| -> String {
                        let r: Option<u32> = s.strip_prefix('r').and_then(|x| x.split(|c: char| !c.is_ascii_digit()).next()).and_then(|x| x.parse().ok());
                        match r.and_then(|r| origin.get(&r).copied()) {
                            Some(v) => format!("v{v}@{}", f.value_loc(v).unwrap_or_default()),
                            None => String::new(),
                        }
                    };
                    let regs: Vec<String> = amdgpu_codegen::regplan::region_report(&m, &a)
                        .iter()
                        .map(|p| {
                            format!(
                                "{{\"name\":\"{}\",\"insts\":{},\"vgpr_footprint_peak\":{},\"vgpr_live_peak\":{},\"agpr_footprint_peak\":{},\"peak_at\":[{},{}],\"live_at_peak\":[{}]}}",
                                p.name,
                                p.insts,
                                p.vgpr_footprint_peak,
                                p.vgpr_live_peak,
                                p.agpr_footprint_peak,
                                p.at.0,
                                p.at.1,
                                p.live_at_peak.iter().take(64).map(|s| format!("\"{} {}\"", s.replace('"', "'"), src_of(s))).collect::<Vec<_>>().join(",")
                            )
                        })
                        .collect();
                    println!("{{\"kernel\":\"{}\",\"vgpr\":{},\"agpr\":{},\"regions\":[{}]}}", f.name, a.max_v, amdgpu_codegen::regalloc::agpr_count(&m, &a), regs.join(","));
                    ExitCode::SUCCESS
                }
                Err(e) if json => {
                    println!("{{\"kernel\":\"{}\",\"ok\":false,\"diagnostics\":{}}}", f.name, amdgpu_codegen::diag::to_json(&amdgpu_codegen::diag::from_compile_error(&f, &e)));
                    ExitCode::from(1)
                }
                Ok((m, a)) => {
                    println!("{}: arch VGPRs {} AGPRs {}", f.name, a.max_v, amdgpu_codegen::regalloc::agpr_count(&m, &a));
                    for p in amdgpu_codegen::regplan::region_report(&m, &a) {
                        println!(
                            "region {:<28} insts {:>5}  vgpr footprint {:>3}  live dwords {:>3}  agpr footprint {:>3}  peak at b{} i{}: {}",
                            p.name,
                            p.insts,
                            p.vgpr_footprint_peak,
                            p.vgpr_live_peak,
                            p.agpr_footprint_peak,
                            p.at.0,
                            p.at.1,
                            p.live_at_peak.iter().take(12).cloned().collect::<Vec<_>>().join(" ")
                        );
                    }
                    for l in amdgpu_codegen::regplan::report(&m, &a) {
                        println!("plan: {l}");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        "jit-info" => {
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, spec) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            if must_verify(&args, &spec) {
                if let Err(c) = certify_specialized(&f) {
                    return c;
                }
            }
            let req = args.windows(2).find(|w| w[0] == "--backend").map(|w| w[1].clone()).or_else(|| args.iter().find_map(|a| a.strip_prefix("--backend=").map(|s| s.to_string()))).unwrap_or_else(|| "auto".to_string());
            let (resolved, _) = amdgpu_codegen::resolve_backend(&req, &f);
            let Some(tail) = backend::backend_by_name(&resolved) else {
                eprintln!("unknown backend: {resolved} (own | llvm-tail | auto)");
                return ExitCode::from(2);
            };
            match amdgpu_codegen::compile_with(&f, tail.as_ref()) {
                Ok(c) => {
                    let source = match from_ir_path(&args).and_then(|p| fs::read_to_string(p).ok()) {
                        Some(t) => format!("ir:{}", amdc::jit::fnv_hex(&t)),
                        None => "registry".to_string(),
                    };
                    println!("{}", amdc::jit::info_json_src(&f, &spec.0, &spec.1, tail.name(), amdc::jit::resources(&c.asm), &source));
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        "compile-batch" => batch::compile_batch(&args),
        "ladder-hints" => batch::ladder_hints(&args),
        "emit" => {
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            // --time: per-phase timing of this compile (amdgpu_codegen::timing; stderr), --assemble: also run the external clang assembler and time it
            let want_time = args.iter().any(|a| a == "--time");
            let want_assemble = args.iter().any(|a| a == "--assemble");
            if want_time {
                amdgpu_codegen::timing::enable(true);
            }
            let t_build = std::time::Instant::now();
            let (f, spec) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            let build_d = t_build.elapsed();
            if must_verify(&args, &spec) {
                if let Err(c) = certify_specialized(&f) {
                    return c;
                }
            }
            let mut out_path: Option<String> = None;
            let mut mir_out: Option<String> = None;
            let mut backend_name = env::var("AMDC_BACKEND").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "auto".to_string());
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--out" {
                    out_path = args.get(i + 1).cloned();
                    i += 2;
                } else if args[i] == "--mir-out" {
                    mir_out = args.get(i + 1).cloned();
                    i += 2;
                } else if let Some(b) = args[i].strip_prefix("--backend=") {
                    backend_name = b.to_string();
                    i += 1;
                } else if args[i] == "--sched=ld" {
                    env::set_var("AMDC_SCHED", "ld");
                    i += 1;
                } else if args[i] == "--sched" {
                    env::set_var("AMDC_SCHED", "1");
                    i += 1;
                } else if args[i] == "--backend" {
                    backend_name = args.get(i + 1).cloned().unwrap_or_default();
                    i += 2;
                } else if args[i] == "--specialize" {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if let Some(p) = mir_out {
                return match amdgpu_codegen::export_mir(&f, &LlvmTail::new()) {
                    Ok(mir) => match fs::write(&p, mir) {
                        Ok(()) => {
                            eprintln!("wrote {p}");
                            ExitCode::SUCCESS
                        }
                        Err(e) => {
                            eprintln!("cannot write {p}: {e}");
                            ExitCode::from(1)
                        }
                    },
                    Err(e) => {
                        eprintln!("compile error:\n{e}");
                        ExitCode::from(1)
                    }
                };
            }
            let (resolved, note) = amdgpu_codegen::resolve_backend(&backend_name, &f);
            if let Some(n) = note {
                eprintln!("note: {n}");
            }
            let backend_name = resolved;
            let Some(tail) = backend::backend_by_name(&backend_name) else {
                eprintln!("unknown backend: {backend_name} (own | llvm-tail | auto)");
                return ExitCode::from(2);
            };
            let tail: &dyn TailBackend = tail.as_ref();
            let json = args.iter().any(|a| a == "--json");
            let t_compile = std::time::Instant::now();
            let res = amdgpu_codegen::compile_with(&f, tail);
            let compile_d = t_compile.elapsed();
            if want_time {
                let ents = amdgpu_codegen::timing::take();
                batch::emit_report(&f.name, build_d, compile_d, res.as_ref().ok().map(|c| c.asm.as_str()), want_assemble, ents);
            }
            match res {
                Ok(c) if json => {
                    if let Some(p) = &out_path {
                        if let Err(e) = fs::write(p, &c.asm) {
                            eprintln!("cannot write {p}: {e}");
                            return ExitCode::from(1);
                        }
                    }
                    println!("{{\"kernel\":\"{}\",\"ok\":true,\"backend\":\"{}\",\"vgpr\":{},\"sgpr\":{},\"out\":{},\"diagnostics\":[]}}", f.name, tail.name(), c.vgpr, c.sgpr, out_path.as_ref().map(|p| format!("\"{p}\"")).unwrap_or_else(|| "null".into()));
                    ExitCode::SUCCESS
                }
                Err(e) if json => {
                    println!("{{\"kernel\":\"{}\",\"ok\":false,\"diagnostics\":{}}}", f.name, amdgpu_codegen::diag::to_json(&amdgpu_codegen::diag::from_compile_error(&f, &e)));
                    ExitCode::from(1)
                }
                Ok(c) => {
                    match out_path {
                        Some(p) => {
                            if let Err(e) = fs::write(&p, &c.asm) {
                                eprintln!("cannot write {p}: {e}");
                                return ExitCode::from(1);
                            }
                            eprintln!("wrote {p} (backend={} vgpr={} sgpr={})", tail.name(), c.vgpr, c.sgpr);
                        }
                        None => print!("{}", c.asm),
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        other => {
            eprintln!("unknown command: {other}");
            usage();
            ExitCode::from(2)
        }
    }
}