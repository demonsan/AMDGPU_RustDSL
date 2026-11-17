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
