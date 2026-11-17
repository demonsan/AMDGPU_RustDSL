//! gfx1250 hang-safety verifier (G5 of docs/gfx1250-plan.md, rules in docs/gfx1250-hang-rules.md).
//!
//! A hang takes a GPU away from everybody who shares it, so every rule here is a COMPILE ERROR, never a warning and never switchable by the environment: a kernel that the rules
//! reject is not emitted, and a deliberately faulty kernel is never run on a GPU (the mutation tests of `crates/amdgpu-kernels/tests/hang_rules_1250.rs` stop at the compiler).
//!
//! Three layers, all run by `gfx1250::compile_with` for both tails:
//! * `verify_ir`  : the kernel IR (statements, descriptors built from constants / kernarg words, barrier structure, the launch facts the kernel declares);
//! * `verify_asm` : the final instruction text (what the hardware will execute, including exact replays and the LLVM tail's output): clauses, indefinite sleeps, barrier operands,
//!                  DPP, privileged state, the dynamic VGPR instruction, hazards that stall;
//! * `enforce` / `enforce_asm` : the policy: any violation is an error naming the rule id (`GV-1250-*`).
//!
//! Rule ids are stable names (`RULES`); each has a mutation test (`mut_<rule>`: rejected with the rule on, accepted with `with_rules_disabled`, so the test is red when the rule is
//! removed) and a positive twin (the legal variant compiles).

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::fxhash::FxMap as HashMap;
use crate::fxhash::FxSet as HashSet;
use crate::mir::MFunction;
use crate::regalloc::Alloc;
use amdgpu_dsl::{Function, Op, Opnd, Stmt, ValueId, VarId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Ir,
    Asm,
}

#[derive(Clone, Copy, Debug)]
pub struct Rule {
    pub id: &'static str,
    pub layer: Layer,
    /// What the rule keeps the hardware from doing, in generic terms.
    pub what: &'static str,
}

pub const RULES: &[Rule] = &[
    Rule { id: "GV-1250-TDM-CLAUSE", layer: Layer::Asm, what: "a tensor operation is never inside an instruction clause" },
    Rule { id: "GV-1250-SLEEP", layer: Layer::Asm, what: "no indefinite sleep: a wave that sleeps forever may never wake" },
    Rule { id: "GV-1250-MCAST", layer: Layer::Ir, what: "a tensor operation's multicast mask is provably zero unless the launch declares a cluster of at least two workgroups" },
    Rule { id: "GV-1250-BAR-ID", layer: Layer::Ir, what: "barrier ids are in range and allocated (work-group, cluster only in a cluster launch, named ones within the declared count)" },
    Rule { id: "GV-1250-BAR-WAIT-IMM", layer: Layer::Asm, what: "the operand of a barrier wait is an inline constant" },
    Rule { id: "GV-1250-TRAP-BAR", layer: Layer::Ir, what: "the barriers reserved for the trap handler are not used by kernels" },
    Rule { id: "GV-1250-BAR-REACH", layer: Layer::Ir, what: "every barrier is reached by every wave of its group: no barrier under wave-divergent control flow with unequal arrivals, none in a loop with a wave-divergent exit, no wait without a signal" },
    Rule { id: "GV-1250-DPP", layer: Layer::Asm, what: "no DPP modifier together with a literal operand, nor on a 64-bit integer operation" },
    Rule { id: "GV-1250-PERMLANE-CMPX", layer: Layer::Asm, what: "a lane permute does not directly follow a v_cmpx" },
    Rule { id: "GV-1250-DVGPR", layer: Layer::Asm, what: "no dynamic VGPR allocation" },
    Rule { id: "GV-1250-PRIV", layer: Layer::Asm, what: "privileged state is not written from a kernel" },
    Rule { id: "GV-1250-SCHED-MODE", layer: Layer::Asm, what: "the relaxed dependency-checking mode is never entered" },
    Rule { id: "GV-1250-SETREG-MSB", layer: Layer::Asm, what: "a mode register write is not directly followed by a VGPR MSB write" },
];

#[derive(Clone, Debug)]
pub struct Violation {
    pub rule: &'static str,
    pub msg: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.rule, self.msg)
    }
}

thread_local! {
    static OFF: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Runs `body` with the named rules switched off on this thread: ONLY for the mutation tests (a rule's test must be red without the rule).
pub fn with_rules_disabled<R>(ids: &[&str], body: impl FnOnce() -> R) -> R {
    let saved = OFF.with(|o| o.replace(ids.iter().map(|s| s.to_string()).collect()));
    let r = body();
    OFF.with(|o| *o.borrow_mut() = saved);
    r
}

fn on(id: &str) -> bool {
    !OFF.with(|o| o.borrow().iter().any(|x| x == id))
}

fn viol(v: &mut Vec<Violation>, rule: &'static str, msg: String) {
    if on(rule) {
        v.push(Violation { rule, msg });
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// IR layer

/// Workgroups per cluster the kernel's launch declares (`kd` entry `cluster_dims`); 0 / 1 = not a cluster launch.
pub fn cluster_wgs(f: &Function) -> i64 {
    f.kd.iter().rev().find(|(k, _)| k == "cluster_dims").map(|(_, v)| *v).unwrap_or(0)
}

fn kd_val(f: &Function, key: &str) -> i64 {
    f.kd.iter().rev().find(|(k, _)| k == key).map(|(_, v)| *v).unwrap_or(0)
}

fn defs_of(f: &Function) -> HashMap<ValueId, Op> {
    let mut m = HashMap::default();
    for st in f.stmts() {
        if let Stmt::Let { dst, op } = st {
            m.insert(*dst, op.clone());
        }
    }
    m
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mask {
    /// provably zero in the low 16 bits
    Zero,
    /// a known non-zero constant
    Const(u32),
    /// not provable
    Unknown,
}

fn const_of(o: &Opnd, defs: &HashMap<ValueId, Op>) -> Option<i64> {
    match o {
        Opnd::I32(c) => Some(*c as i64),
        Opnd::Val(v) => match defs.get(v) {
            Some(Op::ConstI32(c)) => Some(*c as i64),
            _ => None,
        },
        _ => None,
    }
}

/// The multicast mask (group 1, word 0, bits 15:0) of a tensor operation's group-1 operand.
fn mcast_mask(f: &Function, defs: &HashMap<ValueId, Op>, g1: &Opnd) -> Mask {
    let Opnd::Val(v) = g1 else { return Mask::Unknown };
    let Some(Op::SgprCat8 { words }) = defs.get(v) else { return Mask::Unknown };
    let w0 = &words[0];
    if let Some(c) = const_of(w0, defs) {
        let m = (c as u32) & 0xffff;
        return if m == 0 { Mask::Zero } else { Mask::Const(m) };
    }
    // a word the host supplies in the kernarg segment: provable only through the host contract (a fixed value / a set of values whose low 16 bits are zero)
    if let Opnd::Val(x) = w0 {
        if let Some(Op::KernargI32 { offset }) = defs.get(x) {
            let c = &f.contract;
            if let Some((_, vals)) = c.ka_sets.iter().find(|(o, _)| o == offset) {
                if vals.iter().all(|v| (*v as u32) & 0xffff == 0) {
                    return Mask::Zero;
                }
            }
            if let Some((_, lo, hi)) = c.ka_ranges.iter().find(|(o, _, _)| o == offset) {
                if lo == hi && (*lo as u32) & 0xffff == 0 {
                    return Mask::Zero;
                }
            }
        }
    }
    Mask::Unknown
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Ev {
    Full,
    Signal(i32),
    Wait(i32),
}

fn ev_of(st: &Stmt) -> Option<Ev> {
    match st {
        Stmt::Barrier => Some(Ev::Full),
        Stmt::BarrierSignal { id } => Some(Ev::Signal(*id)),
        Stmt::BarrierWait { id } => Some(Ev::Wait(*id)),
        _ => None,
    }
}

fn events(stmts: &[Stmt], lo: usize, hi: usize) -> BTreeMap<Ev, i32> {
    let mut m = BTreeMap::new();
    for st in &stmts[lo..hi] {
        if let Some(e) = ev_of(st) {
            *m.entry(e).or_insert(0) += 1;
        }
    }
    m
}

/// Values / variables that differ between the waves of a work-group (derived from the work-item id), as in the barrier balance of `gpu_verify`.
fn wave_varying(f: &Function) -> (HashSet<ValueId>, HashSet<VarId>) {
    let mut tv: HashSet<ValueId> = HashSet::default();
    let mut tvars: HashSet<VarId> = HashSet::default();
    let tainted = |o: &Opnd, tv: &HashSet<ValueId>, tvars: &HashSet<VarId>| match o {
        Opnd::Val(v) => tv.contains(v),
        Opnd::Var(v) => tvars.contains(v),
        _ => false,
    };
    loop {
        let mut changed = false;
        for st in f.stmts() {
            match st {
                Stmt::Let { dst, op } => {
                    let t = matches!(op, Op::WorkitemId) || op.operands().iter().any(|o| tainted(o, &tv, &tvars));
                    if t && tv.insert(*dst) {
                        changed = true;
                    }
                }
                Stmt::SetVar { var, value } => {
                    if tainted(value, &tv, &tvars) && tvars.insert(*var) {
                        changed = true;
                    }
                }
                Stmt::BeginLoop { vars, init } | Stmt::DeclareVars { vars, init } => {
                    for (v, i0) in vars.iter().zip(init.iter()) {
                        if tainted(i0, &tv, &tvars) && tvars.insert(*v) {
                            changed = true;
                        }
                    }
                }
                _ => {}
            }
        }
        if !changed {
            break;
        }
    }
    (tv, tvars)
}

fn bar_reach(f: &Function, v: &mut Vec<Violation>) {
    let stmts = f.stmts();
    let (tv, tvars) = wave_varying(f);
    let tainted = |o: &Opnd| match o {
        Opnd::Val(x) => tv.contains(x),
        Opnd::Var(x) => tvars.contains(x),
        _ => false,
    };
    // structure: matching positions of if / else / endif and loop / endloop
    let mut if_stack: Vec<(usize, Option<usize>)> = Vec::new();
    let mut loop_stack: Vec<usize> = Vec::new();
    let mut ifs: Vec<(usize, Option<usize>, usize)> = Vec::new();
    let mut loops: Vec<(usize, usize)> = Vec::new();
    for (i, st) in stmts.iter().enumerate() {
        match st {
            Stmt::BeginIf { .. } => if_stack.push((i, None)),
            Stmt::Else => {
                if let Some(t) = if_stack.last_mut() {
                    t.1 = Some(i);
                }
            }
            Stmt::EndIf => {
                if let Some((b, e)) = if_stack.pop() {
                    ifs.push((b, e, i));
                }
            }
            Stmt::BeginLoop { .. } => loop_stack.push(i),
            Stmt::EndLoop => {
                if let Some(b) = loop_stack.pop() {
                    loops.push((b, i));
                }
            }
            _ => {}
        }
    }
    // (1) wave-divergent if: both arms must arrive at the same barriers
    for &(b, e, end) in &ifs {
        let Stmt::BeginIf { cond } = &stmts[b] else { continue };
        if !tainted(cond) {
            continue;
        }
        let (then_hi, els) = match e {
            Some(e) => (e, events(stmts, e + 1, end)),
            None => (end, BTreeMap::new()),
        };
        let then = events(stmts, b + 1, then_hi);
        if then != els {
            viol(
                v,
                "GV-1250-BAR-REACH",
                format!("statement {b}: a barrier operation under a condition that differs between the waves of a work-group: the arms arrive at different barriers (then {then:?}, else {els:?}); a wave that does not arrive leaves the others waiting forever"),
            );
        }
    }
    // (2) a loop with a wave-divergent exit must not contain a barrier operation
    for &(b, end) in &loops {
        let mut diverges = false;
        let mut depth = 0;
        for st in &stmts[b + 1..end] {
            match st {
                Stmt::BeginLoop { .. } => depth += 1,
                Stmt::EndLoop => depth -= 1,
                Stmt::BreakIf { cond } if depth == 0 && tainted(cond) => diverges = true,
                _ => {}
            }
        }
        if diverges {
            let ev = events(stmts, b + 1, end);
            if !ev.is_empty() {
                viol(
                    v,
                    "GV-1250-BAR-REACH",
                    format!("statement {b}: a loop whose exit differs between the waves of a work-group contains barrier operations ({ev:?}): waves leave at different iterations while others wait"),
                );
            }
        }
    }
    // (3) a wait needs a signal of the same barrier: before it, or later in the same loop (a software-pipelined signal of the next iteration)
    for (i, st) in stmts.iter().enumerate() {
        let Stmt::BarrierWait { id } = st else { continue };
        let before = stmts[..i].iter().any(|s| matches!(s, Stmt::BarrierSignal { id: j } if j == id));
        let in_loop_later = loops.iter().any(|&(b, end)| b < i && i < end && stmts[i + 1..end].iter().any(|s| matches!(s, Stmt::BarrierSignal { id: j } if j == id)));
        if !before && !in_loop_later {
            viol(v, "GV-1250-BAR-REACH", format!("statement {i}: a wait on barrier {id} with no signal of that barrier before it (or later in its loop): the wait never completes"));
        }
    }
}

/// The IR-level rules.
pub fn verify_ir(f: &Function) -> Vec<Violation> {
    let mut v = Vec::new();
    let defs = defs_of(f);
    let cluster = cluster_wgs(f);
    let named = kd_val(f, "named_barrier_count");
    for (pos, st) in f.stmts().iter().enumerate() {
        match st {
            Stmt::TensorLoadToLds { g1, .. } | Stmt::TensorStoreFromLds { g1, .. } => {
                let m = mcast_mask(f, &defs, g1);
                if m != Mask::Zero && cluster < 2 {
                    let what = match m {
                        Mask::Const(c) => format!("the multicast mask is {c:#x}"),
                        _ => "the multicast mask is not provably zero".to_string(),
                    };
                    viol(&mut v, "GV-1250-MCAST", format!("statement {pos}: {what}, but the launch declares no cluster (cluster_dims >= 2): a multicast outside a cluster may hang or corrupt data"));
                }
            }
            Stmt::BarrierSignal { id } | Stmt::BarrierWait { id } => {
                let id = *id;
                match id {
                    -2 | -4 => viol(&mut v, "GV-1250-TRAP-BAR", format!("statement {pos}: barrier {id} is reserved for the trap handler")),
                    -1 => {}
                    -3 => {
                        if cluster < 2 {
                            viol(&mut v, "GV-1250-BAR-ID", format!("statement {pos}: the cluster barrier (-3) in a launch that declares no cluster"));
                        }
                    }
                    1..=16 => {
                        if (id as i64) > named {
                            viol(&mut v, "GV-1250-BAR-ID", format!("statement {pos}: named barrier {id} is not allocated (named_barrier_count = {named}): an unallocated barrier is silently a no-op"));
                        }
                    }
                    _ => viol(&mut v, "GV-1250-BAR-ID", format!("statement {pos}: barrier id {id} is not a valid barrier (valid: -1, -3 in a cluster, 1..=16 named)")),
                }
            }
            _ => {}
        }
    }
    bar_reach(f, &mut v);
    v
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// asm layer

struct Inst {
    line: usize,
    mn: String,
    ops: String,
}

fn parse_int(s: &str) -> Option<i64> {
    let s = s.trim();
    let (neg, t) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let v = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) { i64::from_str_radix(h, 16).ok()? } else { t.parse::<i64>().ok()? };
    Some(if neg { -v } else { v })
}

fn lex(asm: &str) -> (Vec<Inst>, Vec<(usize, u32)>) {
    let mut insts = Vec::new();
    let mut raw = Vec::new();
    for (i, l) in asm.lines().enumerate() {
        let l = l.split("//").next().unwrap_or("").trim();
        if l.is_empty() || l.ends_with(':') {
            continue;
        }
        if let Some(r) = l.strip_prefix(".long") {
            for w in r.split(',') {
                if let Some(x) = parse_int(w) {
                    raw.push((i + 1, x as u32));
                }
            }
            continue;
        }
        if l.starts_with('.') {
            continue;
        }
        let (mn, ops) = match l.split_once(char::is_whitespace) {
            Some((a, b)) => (a, b.trim()),
            None => (l, ""),
        };
        insts.push(Inst { line: i + 1, mn: mn.to_string(), ops: ops.to_string() });
    }
    (insts, raw)
}

fn is_dpp(ops: &str) -> bool {
    ["quad_perm", "row_shl", "row_shr", "row_ror", "row_mirror", "row_half_mirror", "row_bcast", "row_share", "row_xmask", "row_newbcast", "wave_shl", "wave_shr", "wave_rol", "wave_ror", "dpp8", "dpp16"].iter().any(|k| ops.contains(k))
}

/// A literal operand (not an inline constant): a hex / decimal integer outside -16..=64, or a float other than the inline ones.
fn has_literal(ops: &str) -> bool {
    // drop the bracketed modifier arguments (`quad_perm:[0,1,2,3]`, `dpp8:[...]`) and the `name:value` modifiers
    let mut s = String::new();
    let mut depth = 0;
    for c in ops.chars() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            _ if depth == 0 => s.push(c),
            _ => {}
        }
    }
    for t in s.split(|c: char| c == ',' || c.is_whitespace()).filter(|t| !t.is_empty() && !t.contains(':')) {
        if let Some(x) = parse_int(t) {
            if !(-16..=64).contains(&x) {
                return true;
            }
        } else if t.parse::<f64>().is_ok() {
            let x: f64 = t.parse().unwrap();
            if ![0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0, 0.0, 0.15915494].iter().any(|k| (x - k).abs() < 1e-7) {
                return true;
            }
        }
    }
    false
}

const BARRIER_MNEMONICS: &[&str] = &["s_barrier_signal", "s_barrier_signal_isfirst", "s_barrier_wait", "s_barrier_join", "s_barrier_init", "s_wakeup_barrier", "s_get_barrier_state"];

/// The final-instruction-stream rules (the text of a kernel as the assembler will see it).
pub fn verify_asm(asm: &str) -> Vec<Violation> {
    let (insts, raw) = lex(asm);
    let mut v = Vec::new();
    // clauses: the N + 1 instructions after `s_clause N`
    let mut remaining = 0usize;
    for (k, ins) in insts.iter().enumerate() {
        if remaining > 0 {
            if ins.mn.starts_with("tensor_") {
                viol(&mut v, "GV-1250-TDM-CLAUSE", format!("line {}: `{}` inside an instruction clause", ins.line, ins.mn));
            }
            remaining -= 1;
        }
        if ins.mn == "s_clause" {
            remaining = parse_int(&ins.ops).map(|n| (n as usize) + 1).unwrap_or(0);
        }
        // indefinite sleeps
        if ins.mn == "s_sleep" || ins.mn == "s_monitor_sleep" {
            if parse_int(&ins.ops).map(|x| x as u64 & 0x8000 != 0).unwrap_or(false) {
                viol(&mut v, "GV-1250-SLEEP", format!("line {}: `{} {}` is an indefinite sleep", ins.line, ins.mn, ins.ops));
            }
        }
        // barrier operands
        if BARRIER_MNEMONICS.contains(&ins.mn.as_str()) && ins.mn != "s_get_barrier_state" {
            let first = ins.ops.split(',').next_back().unwrap_or("").trim();   // the barrier operand is the last one (`s_get_barrier_state sdst, id` is excluded above)
            match parse_int(first) {
                Some(-2) | Some(-4) => viol(&mut v, "GV-1250-TRAP-BAR", format!("line {}: `{} {}` uses a trap-handler barrier", ins.line, ins.mn, ins.ops)),
                Some(id) if id == -1 || id == -3 || (1..=16).contains(&id) => {}
                Some(id) => viol(&mut v, "GV-1250-BAR-ID", format!("line {}: `{} {}`: barrier id {id} is not a valid barrier", ins.line, ins.mn, ins.ops)),
                None => {
                    if ins.mn == "s_barrier_wait" {
                        viol(&mut v, "GV-1250-BAR-WAIT-IMM", format!("line {}: `{} {}`: the barrier operand of a wait must be an inline constant", ins.line, ins.mn, ins.ops));
                    }
                }
            }
        }
        // DPP
        if is_dpp(&ins.ops) {
            if has_literal(&ins.ops) {
                viol(&mut v, "GV-1250-DPP", format!("line {}: `{} {}`: a DPP modifier with a literal operand", ins.line, ins.mn, ins.ops));
            }
            if ins.mn.contains("_u64") || ins.mn.contains("_i64") || ins.mn.contains("_b64") {
                viol(&mut v, "GV-1250-DPP", format!("line {}: `{}`: a DPP modifier on a 64-bit integer operation", ins.line, ins.mn));
            }
        }
        // hazards that stall
        if k > 0 && ins.mn.starts_with("v_permlane") && insts[k - 1].mn.starts_with("v_cmpx") {
            viol(&mut v, "GV-1250-PERMLANE-CMPX", format!("line {}: `{}` directly after `{}`", ins.line, ins.mn, insts[k - 1].mn));
        }
        // dynamic VGPRs
        if ins.mn == "s_alloc_vgpr" {
            viol(&mut v, "GV-1250-DVGPR", format!("line {}: `s_alloc_vgpr`: dynamic VGPR allocation is not supported", ins.line));
        }
        // privileged state, mode registers
        if ins.mn == "s_sethalt" {
            viol(&mut v, "GV-1250-PRIV", format!("line {}: `s_sethalt`: privileged state", ins.line));
        }
        if ins.mn.starts_with("s_setreg") {
            let up = ins.ops.to_uppercase();
            if ["PRIV", "TRAP", "TTMP", "POPS"].iter().any(|k| up.contains(k)) {
                viol(&mut v, "GV-1250-PRIV", format!("line {}: `{} {}`: privileged register", ins.line, ins.mn, ins.ops));
            }
            if up.contains("SCHED_MODE") {
                viol(&mut v, "GV-1250-SCHED-MODE", format!("line {}: `{} {}`: the relaxed dependency-checking mode", ins.line, ins.mn, ins.ops));
            } else if up.contains("MODE") && insts.get(k + 1).map(|n| n.mn == "s_set_vgpr_msb").unwrap_or(false) {
                viol(&mut v, "GV-1250-SETREG-MSB", format!("line {}: a mode register write directly followed by `s_set_vgpr_msb`", ins.line));
            }
        }
    }
    // raw words (exact replays): indefinite sleep encodings
    for (line, w) in raw {
        if w & 0xffff_0000 == 0xbf83_0000 && w & 0x8000 != 0 {
            viol(&mut v, "GV-1250-SLEEP", format!("line {line}: raw word {w:#010x} is an indefinite sleep"));
        }
    }
    v
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// policy

fn fail(name: &str, v: &[Violation]) -> Result<(), String> {
    if v.is_empty() {
        return Ok(());
    }
    let mut s = format!("{name}: gfx1250 hang-safety verifier: {} violation(s)", v.len());
    for x in v {
        s.push_str(&format!("\n  {x}"));
    }
    Err(s)
}

/// IR-level enforcement, run right after `lower_and_allocate` by `gfx1250::compile_with` (the final MIR is accepted for rules that need it; the current rules read the IR).
pub fn enforce(f: &Function, _m: &MFunction, _a: &Alloc) -> Result<(), String> {
    fail(&f.name, &verify_ir(f))
}

/// Final-instruction-stream enforcement, run on the emitted text of both tails.
pub fn enforce_asm(name: &str, asm: &str) -> Result<(), String> {
    fail(name, &verify_asm(asm))
}
