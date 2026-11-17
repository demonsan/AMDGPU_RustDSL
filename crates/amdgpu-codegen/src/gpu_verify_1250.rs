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
    Rule { id: "GV-1250-NBAR-COUNT", layer: Layer::Ir, what: "a named barrier's member count is a constant in the field's range (1..=63); completion is count-based, so the count is the number of signals per phase, joined or not" },
    Rule { id: "GV-1250-BAR-JOIN", layer: Layer::Ir, what: "named barriers are joined and left in a consistent discipline: one joined barrier at a time, a leave only while joined, the same join state on every path, a wait on a named barrier only while joined to it (any wave may signal any barrier)" },
    Rule { id: "GV-1250-BAR-INIT", layer: Layer::Ir, what: "a named barrier is initialized before it is used, and a work-group barrier orders the initialization before the first use (a re-initialization after another wave signalled loses the count)" },
    Rule { id: "GV-1250-BAR-COUNT", layer: Layer::Ir, what: "the cluster barrier is signalled by exactly one wave per work-group per phase: proved statically when the signalling condition is a function of the wave id, otherwise an explicit, named launch fact that is listed as a trusted item (never an implicit pass)" },
    Rule { id: "GV-1250-BAR-WAIT-IMM", layer: Layer::Asm, what: "the operand of a barrier wait is an inline constant" },
    Rule { id: "GV-1250-TRAP-BAR", layer: Layer::Ir, what: "the barriers reserved for the trap handler are not used by kernels" },
    Rule { id: "GV-1250-BAR-REACH", layer: Layer::Ir, what: "every barrier is reached by every wave of its group: no barrier under wave-divergent control flow with unequal arrivals, none in a loop with a wave-divergent exit, no wait without a signal" },
    Rule { id: "GV-1250-TDM-DESC", layer: Layer::Ir, what: "a tensor descriptor whose compile-time-known fields are invalid for a kernel: the context-restore / store / volatile bits set, a wrong type, a valid count that is not 0 or 1, or a tile that does not fit in the declared LDS (a descriptor word the host supplies is the host contract's business, see GV-1250-MCAST)" },
    Rule { id: "GV-1250-TDM-DRAIN", layer: Layer::Asm, what: "a tensor operation never issues with more than 10 tensor operations outstanding (a sound static bound over all paths; waits lower it, loops that issue without draining are unbounded)" },
    Rule { id: "GV-1250-DPP", layer: Layer::Asm, what: "no DPP modifier together with a literal operand, nor on a 64-bit integer operation" },
    Rule { id: "GV-1250-PERMLANE-CMPX", layer: Layer::Asm, what: "a lane permute does not directly follow a v_cmpx" },
    Rule { id: "GV-1250-DVGPR", layer: Layer::Asm, what: "no dynamic VGPR allocation" },
    Rule { id: "GV-1250-PRIV", layer: Layer::Asm, what: "privileged state is not written from a kernel" },
    Rule { id: "GV-1250-SCHED-MODE", layer: Layer::Asm, what: "the full relaxed dependency-checking mode (and the reserved value) of the scheduling-mode register is never selected; the partial mode used by exact replays is accepted" },
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

/// The compile-time-known words of a descriptor group (`None` per word the host / the program supplies).
fn group_words(defs: &HashMap<ValueId, Op>, o: &Opnd) -> Vec<Option<u32>> {
    let Opnd::Val(v) = o else { return Vec::new() };
    match defs.get(v) {
        Some(Op::RsrcCat { words }) => words.iter().map(|w| const_of(w, defs).map(|c| c as u32)).collect(),
        Some(Op::SgprCat8 { words }) => words.iter().map(|w| const_of(w, defs).map(|c| c as u32)).collect(),
        _ => Vec::new(),
    }
}

/// GV-1250-TDM-DESC: the fields of a tensor descriptor that are constants must be valid for a kernel (group 0: bits 1:0 valid count 0 / 1, bits 2 / 3 / 4 context-restore / store / volatile
/// zero, bits 127:126 type 2; the tile written to LDS fits in the declared LDS).
fn desc_check(f: &Function, defs: &HashMap<ValueId, Op>, pos: usize, g0: &Opnd, g1: &Opnd, v: &mut Vec<Violation>) {
    let w0 = group_words(defs, g0);
    let w1 = group_words(defs, g1);
    if let Some(Some(x)) = w0.first() {
        if x & 3 > 1 {
            viol(v, "GV-1250-TDM-DESC", format!("statement {pos}: descriptor group 0 valid count {} (must be 0 or 1)", x & 3));
        }
        if x & 0b11100 != 0 {
            viol(v, "GV-1250-TDM-DESC", format!("statement {pos}: descriptor group 0 context-restore / store / volatile bits {:#x} must be zero in a kernel", x & 0b11100));
        }
    }
    if let Some(Some(x)) = w0.get(3) {
        if x >> 30 != 2 {
            viol(v, "GV-1250-TDM-DESC", format!("statement {pos}: descriptor group 0 type field {} (must be 2)", x >> 30));
        }
    }
    // the tile (no padding, 1..3 dimensions) in LDS
    if let (Some(Some(lds)), true, Some(Some(a0)), Some(Some(a3)), Some(Some(a4))) = (w0.get(1), w1.len() == 8, w1.first(), w1.get(3), w1.get(4)) {
        let pad_enable = a0 >> 20 & 1 != 0;
        let esize = 1u64 << (a0 >> 16 & 3);
        let (t0, t1, t2) = ((a3 >> 16) as u64, (a4 & 0xffff) as u64, (a4 >> 16) as u64);
        let elems = t0 * t1.max(1) * t2.max(1);
        if !pad_enable && t0 > 0 && f.lds_bytes > 0 && (*lds as u64) + elems * esize > f.lds_bytes as u64 {
            viol(v, "GV-1250-TDM-DESC", format!("statement {pos}: the tile ({elems} elements of {esize} bytes at LDS address {lds}) does not fit in the declared {} bytes of LDS", f.lds_bytes));
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Ev {
    Full,
    Signal(i32),
    Wait(i32),
    Join(i32),
    Leave,
    Cluster,
}

fn ev_of(st: &Stmt) -> Option<Ev> {
    match st {
        Stmt::Barrier => Some(Ev::Full),
        // the cluster barrier is signalled by ONE wave per work-group (any of them): its signals are not arrivals every wave must make; GV-1250-BAR-COUNT counts them
        Stmt::BarrierSignal { id: -3 } => None,
        Stmt::BarrierSignal { id } => Some(Ev::Signal(*id)),
        Stmt::BarrierWait { id } => Some(Ev::Wait(*id)),
        Stmt::BarrierJoin { id } => Some(Ev::Join(*id)),
        Stmt::BarrierLeave => Some(Ev::Leave),
        Stmt::ClusterBarrier => Some(Ev::Cluster),
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

// --------------------------------------------------------------------------------------------------------------------------------------------
// control flow of the statement list: structured if / loop plus the exact-mode forward gotos (blocks of a region form a DAG)

type Sig = BTreeMap<Ev, i32>;

fn sig_add(a: &Sig, b: &Sig) -> Sig {
    let mut r = a.clone();
    for (k, n) in b {
        *r.entry(*k).or_insert(0) += n;
    }
    r
}

/// The most distinct barrier paths a region may have before the verifier gives up (and rejects: not provable).
const MAX_PATH_SIGS: usize = 512;

struct Shape {
    /// BeginIf -> (Else, EndIf)
    if_else: HashMap<usize, (Option<usize>, usize)>,
    /// Else -> EndIf
    else_end: HashMap<usize, usize>,
    /// BeginLoop -> EndLoop, EndLoop -> BeginLoop
    loop_end: HashMap<usize, usize>,
    loop_begin: HashMap<usize, usize>,
    /// BreakIf -> the BeginLoop it leaves
    break_loop: HashMap<usize, usize>,
    /// LayoutKey key -> statement
    key_pos: HashMap<u32, usize>,
}

fn shape_of(stmts: &[Stmt]) -> Shape {
    enum Open {
        If(usize, Option<usize>),
        Loop(usize),
    }
    let mut sh = Shape { if_else: HashMap::default(), else_end: HashMap::default(), loop_end: HashMap::default(), loop_begin: HashMap::default(), break_loop: HashMap::default(), key_pos: HashMap::default() };
    let mut stack: Vec<Open> = Vec::new();
    for (i, st) in stmts.iter().enumerate() {
        match st {
            Stmt::BeginIf { .. } => stack.push(Open::If(i, None)),
            Stmt::Else => {
                if let Some(Open::If(_, e)) = stack.last_mut() {
                    *e = Some(i);
                }
            }
            Stmt::EndIf => {
                if let Some(Open::If(b, e)) = stack.pop() {
                    sh.if_else.insert(b, (e, i));
                    if let Some(e) = e {
                        sh.else_end.insert(e, i);
                    }
                }
            }
            Stmt::BeginLoop { .. } => stack.push(Open::Loop(i)),
            Stmt::EndLoop => {
                if let Some(Open::Loop(b)) = stack.pop() {
                    sh.loop_end.insert(b, i);
                    sh.loop_begin.insert(i, b);
                }
            }
            Stmt::BreakIf { .. } => {
                if let Some(b) = stack.iter().rev().find_map(|o| if let Open::Loop(b) = o { Some(*b) } else { None }) {
                    sh.break_loop.insert(i, b);
                }
            }
            Stmt::LayoutKey { key } => {
                sh.key_pos.insert(*key, i);
            }
            _ => {}
        }
    }
    sh
}

/// Control-flow successors of statement `i` (the statement list is the CFG: if / else / loops structured, gotos forward to a LayoutKey).
fn succs(stmts: &[Stmt], sh: &Shape, i: usize) -> Vec<usize> {
    let n = stmts.len();
    let mut v = Vec::new();
    match &stmts[i] {
        Stmt::BeginIf { .. } => {
            v.push(i + 1);
            if let Some(&(e, end)) = sh.if_else.get(&i) {
                v.push(match e {
                    Some(e) => e + 1,
                    None => end,
                });
            }
        }
        Stmt::Else => {
            if let Some(&end) = sh.else_end.get(&i) {
                v.push(end);
            }
        }
        Stmt::EndLoop => {
            if let Some(&b) = sh.loop_begin.get(&i) {
                v.push(b + 1);
            }
            v.push(i + 1);
        }
        Stmt::BreakIf { .. } => {
            v.push(i + 1);
            if let Some(&b) = sh.break_loop.get(&i) {
                if let Some(&e) = sh.loop_end.get(&b) {
                    v.push(e + 1);
                }
            }
        }
        Stmt::GotoIf { key, .. } => {
            v.push(i + 1);
            if let Some(&t) = sh.key_pos.get(key) {
                v.push(t);
            }
        }
        Stmt::Goto { key } => {
            if let Some(&t) = sh.key_pos.get(key) {
                v.push(t);
            }
        }
        _ => v.push(i + 1),
    }
    v.retain(|&t| t <= n);
    v
}

/// The facts tracked along every path of the CFG.
#[derive(Clone, PartialEq, Eq, Default, Debug)]
struct Flow {
    /// the named barrier the wave is joined to
    joined: Option<i32>,
    /// named barriers initialized on EVERY path
    inited: std::collections::BTreeSet<i32>,
    /// named barriers initialized on SOME path with no work-group barrier since
    unordered: std::collections::BTreeSet<i32>,
}

/// Joins two path states; `.1` is true when the join states of the two paths differ.
fn flow_merge(a: &Flow, b: &Flow) -> (Flow, bool) {
    let r = Flow {
        joined: a.joined,
        inited: a.inited.intersection(&b.inited).copied().collect(),
        unordered: a.unordered.union(&b.unordered).copied().collect(),
    };
    (r, a.joined != b.joined)
}

fn flow_step(st: &Stmt, f: &Flow) -> Flow {
    let mut r = f.clone();
    match st {
        Stmt::BarrierInit { id, .. } => {
            r.inited.insert(*id);
            r.unordered.insert(*id);
        }
        Stmt::Barrier | Stmt::BarrierWait { id: -1 } => r.unordered.clear(),
        Stmt::BarrierJoin { id } => r.joined = Some(*id),
        Stmt::BarrierLeave => r.joined = None,
        _ => {}
    }
    r
}

/// GV-1250-BAR-JOIN, GV-1250-BAR-INIT and the wait-on-an-unsignalled-barrier part of GV-1250-BAR-REACH: a forward dataflow over the CFG (merges join the facts of every incoming path, so a goto's target
/// sees the facts of the fall-through and of every goto naming it).
fn flow_checks(f: &Function, sh: &Shape, v: &mut Vec<Violation>) {
    let stmts = f.stmts();
    let n = stmts.len();
    let mut state: Vec<Option<Flow>> = vec![None; n + 1];
    let mut conflict: HashSet<usize> = HashSet::default();
    state[0] = Some(Flow::default());
    let mut work = vec![0usize];
    while let Some(i) = work.pop() {
        if i >= n {
            continue;
        }
        let out = flow_step(&stmts[i], state[i].as_ref().unwrap());
        for t in succs(stmts, sh, i) {
            let (new, differ) = match &state[t] {
                None => (out.clone(), false),
                Some(old) => flow_merge(old, &out),
            };
            if differ {
                conflict.insert(t);
            }
            if state[t].as_ref() != Some(&new) {
                state[t] = Some(new);
                work.push(t);
            }
        }
    }
    let mut at: Vec<usize> = conflict.into_iter().collect();
    at.sort();
    for t in at {
        viol(v, "GV-1250-BAR-JOIN", format!("statement {t}: control-flow paths meet with different named-barrier join states (a branch, a goto target or a loop back edge that joins / leaves on one path only)"));
    }
    let mut all_signals: HashSet<i32> = stmts.iter().filter_map(|s| if let Stmt::BarrierSignal { id } = s { Some(*id) } else { None }).collect();
    if stmts.iter().any(|s| matches!(s, Stmt::ClusterBarrier | Stmt::Barrier)) {
        all_signals.insert(-1);
    }
    if stmts.iter().any(|s| matches!(s, Stmt::ClusterBarrier)) {
        all_signals.insert(-3);
    }
    for (i, st) in stmts.iter().enumerate() {
        let Some(fl) = &state[i] else { continue };
        match st {
            Stmt::BarrierJoin { id } => {
                if let Some(j) = fl.joined {
                    viol(v, "GV-1250-BAR-JOIN", format!("statement {i}: join of barrier {id} while still joined to barrier {j} (one joined barrier at a time)"));
                }
            }
            Stmt::BarrierLeave => {
                if fl.joined.is_none() {
                    viol(v, "GV-1250-BAR-JOIN", format!("statement {i}: leave with no barrier joined"));
                }
            }
            _ => {}
        }
        match st {
            Stmt::BarrierWait { id } if (1..=16).contains(id) => {
                if fl.joined != Some(*id) {
                    viol(v, "GV-1250-BAR-JOIN", format!("statement {i}: wait on named barrier {id} while {} (a wave waits only on the named barrier it joined; signalling any barrier is allowed)", match fl.joined { Some(j) => format!("joined to barrier {j}"), None => "not joined".to_string() }));
                }
            }
            _ => {}
        }
        if let Stmt::BarrierJoin { id } | Stmt::BarrierSignal { id } | Stmt::BarrierWait { id } = st {
            if (1..=16).contains(id) {
                if !fl.inited.contains(id) {
                    viol(v, "GV-1250-BAR-INIT", format!("statement {i}: named barrier {id} used with no initialization before it on every path (its member count is unset: it never completes)"));
                } else if fl.unordered.contains(id) {
                    viol(v, "GV-1250-BAR-INIT", format!("statement {i}: named barrier {id} used with no work-group barrier after its initialization (another wave may still be initializing it)"));
                }
            }
        }
        if let Stmt::BarrierWait { id } = st {
            // completion is count-based and any wave may signal: the signals can come from other waves (a cluster barrier is signalled by one wave per workgroup, waited on by all), so the
            // only provable hang is a barrier that NOTHING in the kernel signals
            if !all_signals.contains(id) {
                viol(v, "GV-1250-BAR-REACH", format!("statement {i}: a wait on barrier {id}, which no statement of the kernel signals: the wait never completes"));
            }
        }
    }
}

/// The barrier signatures (events per path) of the statements in `lo..hi` (one region: the top level, an if arm, a loop body): a set with one signature per distinct path. Nested ifs are
/// choices between their arms' signatures, loops contribute their body once, the blocks of a region with gotos form a DAG. Checks on the way: GV-1250-BAR-REACH for a branch / goto whose
/// condition differs between waves and whose sides do not arrive at the same barriers, and for a loop with a wave-divergent exit that contains barrier operations.
fn region_sigs(stmts: &[Stmt], sh: &Shape, lo: usize, hi: usize, tainted: &dyn Fn(&Opnd) -> bool, v: &mut Vec<Violation>) -> std::collections::BTreeSet<Sig> {
    use std::collections::BTreeSet;
    struct Block {
        pre: BTreeSet<Sig>,
        succ: Vec<Result<usize, u32>>,   // Ok(block) / Err(key)
        diverging_goto: Option<usize>,   // the statement of a wave-divergent GotoIf that ends this block
    }
    let one = || -> BTreeSet<Sig> { std::iter::once(Sig::new()).collect() };
    let cross = |a: &BTreeSet<Sig>, b: &BTreeSet<Sig>| -> BTreeSet<Sig> { a.iter().flat_map(|x| b.iter().map(move |y| sig_add(x, y))).collect() };
    let mut blocks: Vec<Block> = vec![Block { pre: one(), succ: vec![], diverging_goto: None }];
    let mut key_block: HashMap<u32, usize> = HashMap::default();
    let mut too_many = false;
    let mut i = lo;
    while i < hi {
        let cur = blocks.len() - 1;
        match &stmts[i] {
            Stmt::BeginIf { cond } => {
                let &(e, end) = sh.if_else.get(&i).unwrap_or(&(None, i));
                let (then_hi, els) = match e {
                    Some(e) => (e, region_sigs(stmts, sh, e + 1, end, tainted, v)),
                    None => (end, one()),
                };
                let then = region_sigs(stmts, sh, i + 1, then_hi, tainted, v);
                if tainted(cond) && then != els {
                    viol(v, "GV-1250-BAR-REACH", format!("statement {i}: a branch on a condition that differs between the waves of a work-group whose sides arrive at different barriers (then {then:?}, else {els:?}): a wave that does not arrive leaves the others waiting forever"));
                }
                let choice: BTreeSet<Sig> = then.union(&els).cloned().collect();
                let pre = cross(&blocks[cur].pre, &choice);
                blocks[cur].pre = pre;
                i = end + 1;
                continue;
            }
            Stmt::BeginLoop { .. } => {
                let end = *sh.loop_end.get(&i).unwrap_or(&i);
                let body = region_sigs(stmts, sh, i + 1, end, tainted, v);
                let mut diverges = false;
                let mut depth = 0;
                for st in &stmts[i + 1..end] {
                    match st {
                        Stmt::BeginLoop { .. } => depth += 1,
                        Stmt::EndLoop => depth -= 1,
                        Stmt::BreakIf { cond } if depth == 0 && tainted(cond) => diverges = true,
                        _ => {}
                    }
                }
                if diverges && body.iter().any(|s| !s.is_empty()) {
                    viol(v, "GV-1250-BAR-REACH", format!("statement {i}: a loop whose exit differs between the waves of a work-group contains barrier operations: waves leave at different iterations while others wait"));
                }
                let pre = cross(&blocks[cur].pre, &body);
                blocks[cur].pre = pre;
                i = end + 1;
                continue;
            }
            Stmt::LayoutKey { key } => {
                let nb = blocks.len();
                blocks[cur].succ.push(Ok(nb));
                blocks.push(Block { pre: one(), succ: vec![], diverging_goto: None });
                key_block.insert(*key, nb);
            }
            Stmt::GotoIf { cond, key } => {
                let nb = blocks.len();
                blocks[cur].succ.push(Err(*key));
                blocks[cur].succ.push(Ok(nb));
                if tainted(cond) {
                    blocks[cur].diverging_goto = Some(i);
                }
                blocks.push(Block { pre: one(), succ: vec![], diverging_goto: None });
            }
            Stmt::Goto { key } => {
                blocks[cur].succ.push(Err(*key));
                blocks.push(Block { pre: one(), succ: vec![], diverging_goto: None });
            }
            st => {
                if let Some(e) = ev_of(st) {
                    let one_ev: Sig = std::iter::once((e, 1)).collect();
                    let pre: BTreeSet<Sig> = blocks[cur].pre.iter().map(|x| sig_add(x, &one_ev)).collect();
                    blocks[cur].pre = pre;
                }
            }
        }
        i += 1;
    }
    // resolve keys, then the signatures of every block from the last one back (all edges go forward)
    let nblocks = blocks.len();
    let mut sigs: Vec<BTreeSet<Sig>> = vec![BTreeSet::new(); nblocks];
    for bi in (0..nblocks).rev() {
        let targets: Vec<usize> = blocks[bi].succ.iter().filter_map(|s| match s { Ok(b) => Some(*b), Err(k) => key_block.get(k).copied() }).filter(|&t| t > bi).collect();
        let cont: BTreeSet<Sig> = if targets.is_empty() { one() } else { targets.iter().flat_map(|&t| sigs[t].iter().cloned()).collect() };
        sigs[bi] = cross(&blocks[bi].pre, &cont);
        if sigs[bi].len() > MAX_PATH_SIGS {
            too_many = true;
            sigs[bi] = sigs[bi].iter().take(MAX_PATH_SIGS).cloned().collect();
        }
    }
    for bi in 0..nblocks {
        if let Some(at) = blocks[bi].diverging_goto {
            let side = |s: &Result<usize, u32>| -> BTreeSet<Sig> {
                let t = match s { Ok(b) => Some(*b), Err(k) => key_block.get(k).copied() };
                t.map(|t| sigs[t].clone()).unwrap_or_else(one)
            };
            let (taken, fall) = (side(&blocks[bi].succ[0]), side(&blocks[bi].succ[1]));
            if taken != fall {
                viol(v, "GV-1250-BAR-REACH", format!("statement {at}: a goto on a condition that differs between the waves of a work-group whose two sides arrive at different barriers (taken {taken:?}, fall-through {fall:?})"));
            }
        }
    }
    if too_many {
        viol(v, "GV-1250-BAR-REACH", format!("statements {lo}..{hi}: more than {MAX_PATH_SIGS} distinct barrier paths: not provable"));
    }
    sigs.into_iter().next().unwrap_or_else(one)
}

/// The barrier-related IR rules over the whole body.
fn bar_structure(f: &Function, v: &mut Vec<Violation>) {
    let stmts = f.stmts();
    let sh = shape_of(stmts);
    let (tv, tvars) = wave_varying(f);
    let tainted = |o: &Opnd| match o {
        Opnd::Val(x) => tv.contains(x),
        Opnd::Var(x) => tvars.contains(x),
        _ => false,
    };
    region_sigs(stmts, &sh, 0, stmts.len(), &tainted, v);
    flow_checks(f, &sh, v);
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// GV-1250-BAR-COUNT: per-work-group signal accounting of the cluster barrier

/// The kernel's launch fact "barrier `id` is signalled by `per_wg` waves of every work-group per phase" (`FnBuilder::barrier_signal_count`): a TRUSTED item (see `trusted_items`).
fn signal_count_fact(f: &Function, id: i32) -> Option<i64> {
    let key = format!("g5:signals_per_wg:{id}");
    f.kd.iter().rev().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// What the guard evaluator reads: the defining op of every value and every operand a variable is ever assigned (declaration, loop init, `SetVar`).
struct EvCtx<'a> {
    defs: &'a HashMap<ValueId, Op>,
    vars: &'a HashMap<VarId, Vec<Opnd>>,
}

fn var_assignments(f: &Function) -> HashMap<VarId, Vec<Opnd>> {
    let mut m: HashMap<VarId, Vec<Opnd>> = HashMap::default();
    for st in f.stmts() {
        match st {
            Stmt::DeclareVars { vars, init } | Stmt::BeginLoop { vars, init } => {
                for (v, i) in vars.iter().zip(init.iter()) {
                    m.entry(*v).or_default().push(*i);
                }
            }
            Stmt::SetVar { var, value } => m.entry(*var).or_default().push(*value),
            _ => {}
        }
    }
    m
}

/// The value of an operand for the work-item `tid`, when it is a function of constants and the work-item id only (the wave id and lane arithmetic the guards of leader arms use).
fn eval_i(o: &Opnd, cx: &EvCtx, tid: i32, depth: u32) -> Option<i64> {
    if depth > 24 {
        return None;
    }
    let defs = cx.defs;
    let go = |x: &Opnd| eval_i(x, cx, tid, depth + 1).map(|v| v as i32);
    match o {
        Opnd::I32(c) => Some(*c as i64),
        // a variable whose every assignment evaluates to one value (a register the author copied once)
        Opnd::Var(v) => {
            let asg = cx.vars.get(v)?;
            let mut val: Option<i32> = None;
            for a in asg {
                // an undefined initial value (a register declared before its first assignment) constrains nothing
                if matches!(a, Opnd::Val(u) if matches!(defs.get(u), Some(Op::Undef { .. }))) {
                    continue;
                }
                let x = go(a)?;
                match val {
                    None => val = Some(x),
                    Some(y) if y == x => {}
                    _ => return None,
                }
            }
            val.map(|x| x as i64)
        }
        Opnd::Val(v) => {
            let r: i32 = match defs.get(v)? {
                Op::ConstI32(c) => *c,
                Op::WorkitemId => tid,
                Op::AddI32(a, b) => go(a)?.wrapping_add(go(b)?),
                Op::MulI32(a, b) => go(a)?.wrapping_mul(go(b)?),
                Op::ShlI32(a, b) => ((go(a)? as u32).checked_shl(go(b)? as u32).unwrap_or(0)) as i32,
                Op::LshrI32(a, b) => ((go(a)? as u32).checked_shr(go(b)? as u32).unwrap_or(0)) as i32,
                Op::AndI32(a, b) => go(a)? & go(b)?,
                Op::XorI32(a, b) => go(a)? ^ go(b)?,
                Op::MinI32(a, b) => go(a)?.min(go(b)?),
                Op::CmpLtI32(a, b) => (go(a)? < go(b)?) as i32,
                Op::CmpGeI32(a, b) => (go(a)? >= go(b)?) as i32,
                Op::NotI1(a) => (go(a)? == 0) as i32,
                Op::Cmp { pred, ty, a, b } => {
                    use amdgpu_dsl::{CmpPred as P, CmpTy as T};
                    let (x, y) = (go(a)?, go(b)?);
                    let (xu, yu) = (x as u32, y as u32);
                    let r = match (ty, pred) {
                        (T::I32, P::Lt) => x < y,
                        (T::I32, P::Le) => x <= y,
                        (T::I32, P::Gt) => x > y,
                        (T::I32, P::Ge) => x >= y,
                        (T::U32, P::Lt) => xu < yu,
                        (T::U32, P::Le) => xu <= yu,
                        (T::U32, P::Gt) => xu > yu,
                        (T::U32, P::Ge) => xu >= yu,
                        (T::I32 | T::U32, P::Eq) => x == y,
                        (T::I32 | T::U32, P::Ne | P::Lg) => x != y,
                        _ => return None,
                    };
                    r as i32
                }
                Op::Select { cond, a, b, .. } => {
                    if go(cond)? != 0 {
                        go(a)?
                    } else {
                        go(b)?
                    }
                }
                // TTMP8 = {debug mark, grid-YZ valid, waveIDinGroup[4:0] (bits 29:25), dispatch index[24:0]}: only the wave id field is a function of the work-item id, so a bit-field
                // extract that lies entirely inside bits 29:25 is evaluated; any other use of the register is not
                Op::Salu { op: amdgpu_dsl::SaluOp::BfeU32, a, b } if is_ttmp8(a, cx, 0) => {
                    let bw = go(b)? as u32;
                    let (off, w) = (bw & 31, (bw >> 16) & 0x7f);
                    if off < 25 || w == 0 || off + w > 30 {
                        return None;
                    }
                    let wave = (tid as u32) >> 5;
                    ((wave >> (off - 25)) & ((1u32 << w) - 1)) as i32
                }
                Op::Salu { op, a, b } if !matches!(op, amdgpu_dsl::SaluOp::AddNcU64) => op.eval(go(a)? as u32 as u64, go(b)? as u32 as u64) as u32 as i32,
                _ => return None,
            };
            Some(r as i64)
        }
        _ => None,
    }
}

/// Is the operand TTMP8 itself (the register read, or a variable whose only assignment is that read)?
fn is_ttmp8(o: &Opnd, cx: &EvCtx, depth: u32) -> bool {
    if depth > 8 {
        return false;
    }
    match o {
        Opnd::Val(v) => matches!(cx.defs.get(v), Some(Op::ReadTtmp { idx: 8 })),
        Opnd::Var(v) => match cx.vars.get(v) {
            Some(a) if !a.is_empty() => a.iter().all(|x| matches!(x, Opnd::Val(u) if matches!(cx.defs.get(u), Some(Op::Undef { .. }))) || is_ttmp8(x, cx, depth + 1)),
            _ => false,
        },
        _ => false,
    }
}

/// The value for the wave `w` (work-items 32 w ..): every lane of the wave must agree, else not a wave-level fact.
fn eval_wave(o: &Opnd, cx: &EvCtx, w: i32) -> Option<i64> {
    let a = eval_i(o, cx, 32 * w, 0)?;
    let b = eval_i(o, cx, 32 * w + 31, 0)?;
    let c = eval_i(o, cx, 32 * w + 15, 0)?;
    if a == b && b == c {
        Some(a)
    } else {
        None
    }
}

/// One region's cluster-barrier accounting over every resolution of the wave-uniform conditions the verifier cannot evaluate.
#[derive(Debug)]
struct CountItem {
    region: usize,
    id: i32,
    /// None: not provable (a path depends on something wave-divergent the verifier cannot evaluate, or too many resolutions). Some: for every resolution, (signals per work-group, waits per wave).
    worlds: Option<Vec<(i64, i64)>>,
    /// when `worlds` is None: the statement whose condition stopped the proof
    why: String,
}

/// The most resolutions of unknown uniform conditions the verifier enumerates.
const MAX_WORLDS: usize = 4096;

/// The statements wave `w` reaches, deciding every branch / goto condition for that wave: a condition the verifier can evaluate is decided; a wave-UNIFORM condition it cannot evaluate takes the
/// value of `assign` (every wave takes the same side; `Err(stmt)` asks the caller to assign it); a wave-divergent one it cannot evaluate: `Err(usize::MAX)`. Loops are entered and left (a loop
/// exit is not decided: statements after the loop are reached when the loop is).
fn wave_reach(f: &Function, sh: &Shape, cx: &EvCtx, uniform: &dyn Fn(&Opnd) -> bool, rel: &[u32], assign: &HashMap<usize, bool>, w: i32) -> Result<Vec<bool>, (usize, bool)> {
    let stmts = f.stmts();
    let n = stmts.len();
    let mut reach = vec![false; n + 1];
    let mut work = vec![0usize];
    reach[0] = true;
    while let Some(i) = work.pop() {
        if i >= n {
            continue;
        }
        // the statements the two sides of the branch at `i` differ by: (from, to); a uniform condition the verifier cannot evaluate is only enumerated when a relevant statement
        // (a signal / wait of the barrier being counted) lies there, otherwise both sides count alike and the fall-through side is taken
        let differ = |from: usize, to: usize| rel[to.min(rel.len() - 1)] > rel[from.min(rel.len() - 1)];
        let decide = |cond: &Opnd, lo: usize, hi: usize, dflt: bool| -> Result<bool, (usize, bool)> {
            match eval_wave(cond, cx, w) {
                Some(v) => Ok(v != 0),
                None if !uniform(cond) => Err((i, true)),
                None if !differ(lo, hi) => Ok(dflt),
                None => assign.get(&i).copied().ok_or((i, false)),
            }
        };
        let mut next: Vec<usize> = Vec::new();
        match &stmts[i] {
            Stmt::BeginIf { cond } => {
                let &(e, end) = sh.if_else.get(&i).ok_or((i, true))?;
                let v = decide(cond, i, end, true)?;
                if v {
                    next.push(i + 1);
                } else {
                    next.push(match e {
                        Some(e) => e + 1,
                        None => end,
                    });
                }
            }
            Stmt::GotoIf { cond, key } => {
                let tgt = *sh.key_pos.get(key).ok_or((i, true))?;
                let v = decide(cond, i, tgt, false)?;
                if v {
                    next.push(*sh.key_pos.get(key).ok_or((i, true))?);
                } else {
                    next.push(i + 1);
                }
            }
            _ => next = succs(stmts, sh, i),
        }
        for t in next {
            if t <= n && !reach[t] {
                reach[t] = true;
                work.push(t);
            }
        }
    }
    Ok(reach)
}

fn count_items(f: &Function) -> Vec<CountItem> {
    let stmts = f.stmts();
    let defs = defs_of(f);
    let vars = var_assignments(f);
    let cx = EvCtx { defs: &defs, vars: &vars };
    let sh = shape_of(stmts);
    let nw = (f.workgroup_size / 32).max(1) as i32;
    let (uv, uvars) = amdgpu_dsl::uniform_values(f);
    let uniform = |o: &Opnd| match o {
        Opnd::Val(x) => uv.contains(x),
        Opnd::Var(x) => uvars.contains(x),
        _ => true,
    };
    // the region of every statement: its innermost loop (the top level otherwise): the signals and waits of one pass
    let mut region: Vec<usize> = Vec::with_capacity(stmts.len());
    let mut regions: Vec<usize> = vec![usize::MAX];
    for (i, st) in stmts.iter().enumerate() {
        match st {
            Stmt::BeginLoop { .. } => regions.push(i),
            Stmt::EndLoop => {
                regions.pop();
            }
            _ => {}
        }
        region.push(*regions.last().unwrap());
    }
    let mut all_regions: Vec<usize> = region.clone();
    all_regions.sort();
    all_regions.dedup();
    let id = -3;
    let mut items: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();   // (region, signal statements, wait statements)
    let mut unpaired: Vec<usize> = Vec::new();
    for &r in &all_regions {
        let sig: Vec<usize> = (0..stmts.len()).filter(|&i| region[i] == r && matches!(stmts[i], Stmt::BarrierSignal { id: j } if j == id)).collect();
        let wts: Vec<usize> = (0..stmts.len()).filter(|&i| region[i] == r && matches!(stmts[i], Stmt::BarrierWait { id: j } if j == id)).collect();
        // a region that waits on the barrier but never signals it: the signal is issued outside the region (a split barrier: signal early, wait late), which a per-region count cannot see
        if sig.is_empty() && !wts.is_empty() {
            unpaired.push(r);
        }
        if !(sig.is_empty() && wts.is_empty()) {
            items.push((r, sig, wts));
        }
    }
    if items.is_empty() {
        return Vec::new();
    }
    // prefix counts of the relevant statements (the signals and waits being counted)
    let mut rel: Vec<u32> = vec![0; stmts.len() + 1];
    for (i, st) in stmts.iter().enumerate() {
        rel[i + 1] = rel[i] + matches!(st, Stmt::BarrierSignal { id: -3 } | Stmt::BarrierWait { id: -3 }) as u32;
    }
    // enumerate the resolutions of the unknown wave-uniform conditions (each world: one assignment; every wave takes the assigned side)
    let mut results: Vec<Vec<(i64, i64)>> = vec![Vec::new(); items.len()];
    let mut stack: Vec<HashMap<usize, bool>> = vec![HashMap::default()];
    let mut seen_worlds = 0usize;
    while let Some(assign) = stack.pop() {
        seen_worlds += 1;
        if seen_worlds > MAX_WORLDS {
            return items.into_iter().map(|(r, _, _)| CountItem { region: r, id, worlds: None, why: format!("more than {MAX_WORLDS} resolutions of the uniform conditions") }).collect();
        }
        let reach: Result<Vec<Vec<bool>>, (usize, bool)> = (0..nw).map(|w| wave_reach(f, &sh, &cx, &uniform, &rel, &assign, w)).collect();
        match reach {
            Err((at, true)) => {
                let why = format!("statement {at} ({:.90?}) branches on a condition that is neither evaluable nor wave-uniform", stmts[at]);
                return items.into_iter().map(|(r, _, _)| CountItem { region: r, id, worlds: None, why: why.clone() }).collect();
            }
            Err((at, false)) => {
                for v in [true, false] {
                    let mut a = assign.clone();
                    a.insert(at, v);
                    stack.push(a);
                }
            }
            Ok(rc) => {
                for (k, (_, sig, wts)) in items.iter().enumerate() {
                    // signals per work-group: every (wave, signal statement) the wave reaches; waits per wave: the most any wave reaches
                    let signals: i64 = rc.iter().map(|rw| sig.iter().filter(|&&i| rw[i]).count() as i64).sum();
                    let waits = rc.iter().map(|rw| wts.iter().filter(|&&i| rw[i]).count() as i64).max().unwrap_or(0);
                    results[k].push((signals, waits));
                }
            }
        }
    }
    items
        .into_iter()
        .zip(results)
        .map(|((r, _, _), w)| {
            let mut w = w;
            w.sort();
            w.dedup();
            if unpaired.contains(&r) {
                CountItem { region: r, id, worlds: None, why: "the region waits on the barrier but contains no signal of it: the matching signal is issued outside the region".to_string() }
            } else {
                CountItem { region: r, id, worlds: Some(w), why: String::new() }
            }
        })
        .collect()
}

fn bar_count(f: &Function, v: &mut Vec<Violation>) {
    for it in count_items(f) {
        let where_ = if it.region == usize::MAX { "the top level".to_string() } else { format!("the loop at statement {}", it.region) };
        match &it.worlds {
            Some(ws) => {
                for &(sig, waits) in ws {
                    if waits > 0 && sig != waits {
                        viol(v, "GV-1250-BAR-COUNT", format!("{where_}: barrier {} is signalled by {sig} wave(s) per work-group and waited on {waits} time(s) per wave (on some resolution of the kernel's uniform conditions): the cluster barrier needs exactly one signalling wave per work-group per wait (the count is by signals, not by waves)", it.id));
                    }
                }
            }
            None => {
                match signal_count_fact(f, it.id) {
                    None => viol(v, "GV-1250-BAR-COUNT", format!("{where_}: cannot prove how many waves per work-group signal barrier {} ({}): declare the launch fact with `barrier_signal_count({}, per_wg)` (a trusted item), or restructure so the verifier can count", it.id, it.why, it.id)),
                    Some(k) if it.id == -3 && k != 1 => viol(v, "GV-1250-BAR-COUNT", format!("the launch fact says {k} signal(s) per work-group for the cluster barrier: exactly 1 is required")),
                    Some(_) => {}
                }
            }
        }
    }
}

/// The launch facts the verifier relied on instead of proving them: listed in the compile's certificate / diagnostics so a trusted assumption is never silent.
pub fn trusted_items(f: &Function) -> Vec<String> {
    let mut v = Vec::new();
    for it in count_items(f) {
        if it.worlds.is_none() {
            if let Some(k) = signal_count_fact(f, it.id) {
                v.push(format!("barrier {}: {k} signalling wave(s) per work-group per phase (declared launch fact, not proved)", it.id));
            }
        }
    }
    v
}

/// The IR-level rules.
pub fn verify_ir(f: &Function) -> Vec<Violation> {
    let mut v = Vec::new();
    let defs = defs_of(f);
    let cluster = cluster_wgs(f);
    let named = kd_val(f, "named_barrier_count");
    for (pos, st) in f.stmts().iter().enumerate() {
        match st {
            Stmt::TensorLoadToLds { g0, g1, .. } | Stmt::TensorStoreFromLds { g0, g1, .. } => {
                desc_check(f, &defs, pos, g0, g1, &mut v);
                let m = mcast_mask(f, &defs, g1);
                if m != Mask::Zero && cluster < 2 {
                    let what = match m {
                        Mask::Const(c) => format!("the multicast mask is {c:#x}"),
                        _ => "the multicast mask is not provably zero".to_string(),
                    };
                    viol(&mut v, "GV-1250-MCAST", format!("statement {pos}: {what}, but the launch declares no cluster (cluster_dims >= 2): a multicast outside a cluster may hang or corrupt data"));
                }
            }
            Stmt::BarrierInit { id, members } => {
                if !(1..=16).contains(id) || (*id as i64) > named {
                    viol(&mut v, "GV-1250-BAR-ID", format!("statement {pos}: barrier init of id {id}: named barriers are 1..=16 and must be allocated (named_barrier_count = {named})"));
                }
                if !(1..=63).contains(&(*members as i64)) {
                    viol(&mut v, "GV-1250-NBAR-COUNT", format!("statement {pos}: named barrier {id} with member count {members}: the count must be between 1 and 63"));
                }
            }
            Stmt::BarrierJoin { id } => {
                if !(1..=16).contains(id) || (*id as i64) > named {
                    viol(&mut v, "GV-1250-BAR-ID", format!("statement {pos}: barrier join of id {id}: named barriers are 1..=16 and must be allocated (named_barrier_count = {named})"));
                }
            }
            Stmt::ClusterBarrier => {
                if cluster < 2 {
                    viol(&mut v, "GV-1250-BAR-ID", format!("statement {pos}: the cluster barrier in a launch that declares no cluster"));
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
    bar_structure(f, &mut v);
    bar_count(f, &mut v);
    v
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// asm layer

struct Inst {
    line: usize,
    mn: String,
    ops: String,
    /// byte address and size when the text is a disassembly (`// 000000001400: D0710001 7C000000 ...`)
    addr: Option<(u64, u64)>,
}

/// `// 0000000000001400: BFA20003` -> (address, bytes of the encoding)
fn disasm_addr(comment: &str) -> Option<(u64, u64)> {
    let c = comment.trim();
    let (a, rest) = c.split_once(':')?;
    let addr = u64::from_str_radix(a.trim(), 16).ok()?;
    let words = rest.split_whitespace().filter(|w| w.len() == 8 && w.chars().all(|c| c.is_ascii_hexdigit())).count() as u64;
    if words == 0 {
        return None;
    }
    Some((addr, 4 * words))
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

fn lex(asm: &str) -> (Vec<Inst>, Vec<(usize, u32)>, HashMap<String, usize>) {
    let mut insts = Vec::new();
    let mut raw = Vec::new();
    let mut labels: HashMap<String, usize> = HashMap::default();
    for (i, l) in asm.lines().enumerate() {
        let (l, comment) = match l.split_once("//") {
            Some((a, b)) => (a.trim(), b),
            None => (l.trim(), ""),
        };
        if l.is_empty() {
            continue;
        }
        if l.ends_with(':') {
            // a label (a disassembler prints `<name>:` or `address <name>:`): the index of the next instruction
            let name = l.trim_end_matches(':').rsplit(|c: char| c == '<' || c == '>' || c.is_whitespace()).find(|x| !x.is_empty()).unwrap_or("");
            labels.insert(name.to_string(), insts.len());
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
        insts.push(Inst { line: i + 1, mn: mn.to_string(), ops: ops.to_string(), addr: disasm_addr(comment) });
    }
    (insts, raw, labels)
}

/// The most tensor operations that may be outstanding when another one issues (so at most 11 in flight): the invariant the public compiler's hazard pass for this target maintains ("keep the
/// tensor count low") by inserting `s_wait_tensorcnt 10` before a tensor operation that an earlier one can reach.
pub const TDM_MAX_OUTSTANDING_AT_ISSUE: u32 = 10;
/// "Unbounded": the tensor counter has 6 bits; a bound that reaches this is not a bound.
const TDM_UNBOUNDED: u32 = 64;

/// GV-1250-TDM-DRAIN: a sound static upper bound of the tensor operations outstanding at every tensor issue, path-sensitive over the control-flow graph of the text. An issue adds one, an
/// `s_wait_tensorcnt N` caps the bound at N, joins take the maximum, loops iterate to a fixpoint (a loop that issues without draining saturates: unbounded). A tensor operation that can issue with
/// more than `TDM_MAX_OUTSTANDING_AT_ISSUE` outstanding is a violation.
fn tdm_drain(insts: &[Inst], labels: &HashMap<String, usize>, v: &mut Vec<Violation>) {
    if !insts.iter().any(|i| i.mn.starts_with("tensor_")) {
        return;
    }
    let n = insts.len();
    let by_addr: HashMap<u64, usize> = insts.iter().enumerate().filter_map(|(i, x)| x.addr.map(|(a, _)| (a, i))).collect();
    // bound[i]: an upper bound of the outstanding tensor operations at the entry of instruction i over all paths (None: not reached yet)
    let mut bound: Vec<Option<u32>> = vec![None; n + 1];
    bound[0] = Some(0);
    let mut work = vec![0usize];
    while let Some(i) = work.pop() {
        if i >= n {
            continue;
        }
        let ins = &insts[i];
        let d = bound[i].unwrap_or(0);
        let out = if ins.mn.starts_with("tensor_") {
            (d + 1).min(TDM_UNBOUNDED)
        } else if ins.mn == "s_wait_tensorcnt" {
            match parse_int(&ins.ops) {
                Some(k) if k >= 0 => d.min(k as u32),
                _ => d,
            }
        } else {
            d
        };
        let mut succ: Vec<usize> = Vec::new();
        let branch = ins.mn == "s_branch" || ins.mn.starts_with("s_cbranch");
        let falls = !(ins.mn == "s_branch" || ins.mn == "s_endpgm" || ins.mn == "s_setpc_b64");
        if falls {
            succ.push(i + 1);
        }
        if branch {
            let target = ins.ops.split(',').next().unwrap_or("").trim();
            if let Some(t) = labels.get(target) {
                succ.push(*t);
            } else if let (Some(off), Some((a, size))) = (parse_int(target), ins.addr) {
                // a disassembly: the branch offset counts 4-byte words from the next instruction
                let dest = (a + size) as i64 + off * 4;
                if let Some(t) = by_addr.get(&(dest as u64)) {
                    succ.push(*t);
                }
            }
        }
        for t in succ {
            if t > n {
                continue;
            }
            // join = the larger bound; revisit when it grows
            if bound[t].map(|b| out > b).unwrap_or(true) {
                bound[t] = Some(out);
                work.push(t);
            }
        }
    }
    for (i, ins) in insts.iter().enumerate() {
        if ins.mn.starts_with("tensor_") {
            if let Some(d) = bound[i] {
                if d > TDM_MAX_OUTSTANDING_AT_ISSUE {
                    let how = if d >= TDM_UNBOUNDED { "an unbounded number".to_string() } else { format!("up to {d}") };
                    viol(v, "GV-1250-TDM-DRAIN", format!("line {}: `{}` can issue with {how} of tensor operations outstanding (the invariant: at most {TDM_MAX_OUTSTANDING_AT_ISSUE} at an issue)", ins.line, ins.mn));
                }
            }
        }
    }
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

/// A write to the scheduling-mode register that can leave the full relaxed dependency-checking mode (value 1) or the reserved value (3) selected, or whose value is not a constant.
/// The partial mode (2: the hardware stops checking two dependency classes, the program owns those waits) and the normal mode (0) are accepted: exact replays of vendor code use the partial mode.
fn sched_mode_problem(mn: &str, ops: &str) -> Option<String> {
    let open = ops.find('(')?;
    let close = ops.rfind(')')?;
    let parts: Vec<&str> = ops[open + 1..close].split(',').map(|x| x.trim()).collect();
    let (off, size) = if parts.len() >= 3 { (parse_int(parts[1])?, parse_int(parts[2])?) } else { (0, 32) };
    // the value operand after the closing parenthesis
    let val = ops[close + 1..].trim_start_matches(',').trim();
    let imm = if mn.contains("imm") { parse_int(val) } else { None };
    let Some(imm) = imm else { return Some("a write to the scheduling-mode register whose value is not a constant".into()) };
    let bit = |r: i64| -> Option<i64> { if r >= off && r < off + size { Some((imm >> (r - off)) & 1) } else { None } };
    match (bit(0), bit(1)) {
        (Some(b0), Some(b1)) => {
            let mode = b0 | (b1 << 1);
            if mode == 1 || mode == 3 { Some(format!("scheduling mode {mode}: the full relaxed dependency-checking mode (1) and the reserved value (3) are not allowed; the partial mode (2) is")) } else { None }
        }
        (Some(1), None) => Some("a write that can select scheduling mode 1 or 3".into()),
        (None, Some(1)) => Some("a write of mode bit 1 alone: the resulting mode may be the reserved value (3)".into()),
        _ => None,
    }
}

const BARRIER_MNEMONICS: &[&str] = &["s_barrier_signal", "s_barrier_signal_isfirst", "s_barrier_wait", "s_barrier_join", "s_barrier_init", "s_wakeup_barrier", "s_get_barrier_state"];

/// The final-instruction-stream rules (the text of a kernel as the assembler will see it).
pub fn verify_asm(asm: &str) -> Vec<Violation> {
    let (insts, raw, labels) = lex(asm);
    let mut v = Vec::new();
    tdm_drain(&insts, &labels, &mut v);
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
            // a disassembler prints the 16-bit immediate of a negative id as unsigned (0xffff = -1, 0xfffd = -3)
            let id16 = parse_int(first).map(|x| if (0x8000..=0xffff).contains(&x) { x - 0x1_0000 } else { x });
            if ins.mn == "s_barrier_signal_isfirst" && id16 == Some(-3) {
                viol(&mut v, "GV-1250-BAR-ID", format!("line {}: `{} {}`: the first-arriver form on the cluster barrier (only the work-group barrier idiom uses it)", ins.line, ins.mn, ins.ops));
            }
            match id16 {
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
                if let Some(why) = sched_mode_problem(&ins.mn, &ins.ops) {
                    viol(&mut v, "GV-1250-SCHED-MODE", format!("line {}: `{} {}`: {why}", ins.line, ins.mn, ins.ops));
                }
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
