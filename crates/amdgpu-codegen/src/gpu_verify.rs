//! GPU-semantics verifier (v1): bounds / lifetime / legality proofs over the DSL IR and the final MIR. Design: `docs/gpu-semantics-verifier.md`.
//!
//! * `verify_ir`  : LDS budget, global + buffer bounds (symbolic interval analysis with the kernel's declared `Contract`), role-barrier balance.
//! * `verify_mir` : register-class legality (limits, tuple / descriptor alignment, MFMA contiguity) and path-sensitive wait / lifetime checks.
//! * `enforce`    : the `AMDC_GPU_VERIFY=off|warn|strict` policy wired into `compile_with`.
//!
//! Soundness statement: every `Error` finding is a property that could NOT be proven; a kernel without errors satisfies, under the host contract
//! (declared extents / ranges / grid relations, trusted `assume`s, no i32 wrap-around), the checked properties. Trusted assumptions are listed as `Info`.

use std::collections::{BTreeMap, HashMap, HashSet};

use amdgpu_dsl::{Ext, Function, Node, Op, Opnd, RegionKind, Stmt, ValueId, VarId};
use amdgpu_sym::{cap, Ctx, Itv, Poly, Sym, Q};

use crate::mir::{is_agpr_phys, MFunction, MOp, RClass, Src, AGPR_TAG};
use crate::regalloc::Alloc;

// --------------------------------------------------------------------------------------------------------------------------------------------
// findings

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warn,
    Info,
}

/// The counted trust LEDGER of a report: every escape hatch the certificate relies on, by kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    /// `assume_trusted` loop-counter ranges + unproven value assumes
    pub assume_trusted: usize,
    /// `tile_trust_periodic`
    pub tile_trust: usize,
    /// `buf_trust` (region trust)
    pub region_trust: usize,
    /// safe-asm blocks (`b.asm`, var asm): their internal memory accesses are not modelled (GV-ASM)
    pub asm_opaque: usize,
    /// exact mode `TrustWait`: a read the wait counters do not guarantee complete (a stated timing reliance, GV-TRUST-TIMING)
    pub timing_trust: usize,
}

impl Ledger {
    pub fn total(&self) -> usize {
        self.assume_trusted + self.tile_trust + self.region_trust + self.asm_opaque + self.timing_trust
    }
}

#[derive(Clone, Debug)]
pub struct Finding {
    pub code: &'static str,
    pub severity: Severity,
    pub msg: String,
    /// The v4 protocol-model name of this verdict (GV-PROTO-*; empty when the finding is not a protocol obligation).
    pub alias: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    /// Number of accesses proven in bounds, by space.
    pub proven: BTreeMap<&'static str, usize>,
    /// (marker statement, reused private region): every access of a tile_over marker avoids the region or stays in the writer's own slice
    pub own_reuse: BTreeMap<(usize, String), bool>,
    /// marker statement of an `lds_owned` tile -> every access of the scope is proven to be in the issuing thread's own words
    pub owned: BTreeMap<usize, bool>,
    /// Accesses that are bounds-checked but NOT race-checked, by kind (docs/trust-inventory.md): `lds-untracked` (an LDS access outside every
    /// tile marker), `global-write-untracked` (a global / buffer store outside a `buf_region` marker: cross-workgroup disjointness is trusted)
    pub unchecked: BTreeMap<&'static str, usize>,
    /// > 0 while the analyzer iterates a loop to a fixpoint: nothing is recorded.
    quiet: u32,
}

impl Report {
    pub(crate) fn add(&mut self, code: &'static str, severity: Severity, msg: String) {
        if self.quiet == 0 {
            let alias = crate::gpu_protocol::proto_alias(code, &msg);
            self.findings.push(Finding { code, severity, msg, alias });
        }
    }
    pub(crate) fn unchecked_access(&mut self, what: &'static str) {
        if self.quiet == 0 {
            *self.unchecked.entry(what).or_insert(0) += 1;
        }
    }
    pub(crate) fn ok(&mut self, what: &'static str) {
        if self.quiet == 0 {
            *self.proven.entry(what).or_insert(0) += 1;
        }
    }
    pub fn errors(&self) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.severity == Severity::Error).collect()
    }
    pub fn merge(&mut self, o: Report) {
        self.findings.extend(o.findings);
        for (k, v) in o.proven {
            *self.proven.entry(k).or_insert(0) += v;
        }
    }
    /// The trust ledger: how many escapes (by kind) this report relies on.
    pub fn ledger(&self) -> Ledger {
        let mut l = Ledger::default();
        for f in &self.findings {
            if f.code == "GV-ASM" {
                l.asm_opaque += 1;
                continue;
            }
            if f.code == "GV-TRUST-TIMING" {
                l.timing_trust += 1;
                continue;
            }
            if f.code != "GV-ASSUME" || !f.msg.contains("TRUSTED") {
                continue;
            }
            if f.msg.contains("tile_trust_periodic") {
                l.tile_trust += 1;
            } else if f.msg.contains("buf_trust") {
                l.region_trust += 1;
            } else {
                l.assume_trusted += 1;
            }
        }
        l
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for f in &self.findings {
            if f.alias.is_empty() || f.alias == f.code {
                s.push_str(&format!("[{}] {:?}: {}\n", f.code, f.severity, f.msg));
            } else {
                s.push_str(&format!("[{}|{}] {:?}: {}\n", f.code, f.alias, f.severity, f.msg));
            }
        }
        s.push_str(&format!("proven accesses: {:?}\n", self.proven));
        s
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// symbolic arithmetic: the shared `amdgpu-sym` crate (rationals, polynomials, constraint context, intervals)

/// The constraint context of a kernel's contract: `ka_range` / `ka_set` / `sym_range` bounds and constants, `ka_is` / `grid_is` definitions.
fn contract_ctx(f: &Function) -> Ctx {
    let mut c = Ctx::new();
    for (off, lo, hi) in &f.contract.ka_ranges {
        let s = Sym::Ka(*off);
        let e = c.lb.entry(s.clone()).or_insert(0);
        *e = (*e).max(*lo);
        if *hi < i64::MAX {
            let u = c.ub.entry(s.clone()).or_insert(*hi);
            *u = (*u).min(*hi);
        }
        if lo == hi {
            c.konst.insert(s, *lo);
        }
    }
    for (off, vals) in &f.contract.ka_sets {
        if let (Some(lo), Some(hi)) = (vals.iter().min(), vals.iter().max()) {
            let s = Sym::Ka(*off);
            let e = c.lb.entry(s.clone()).or_insert(0);
            *e = (*e).max(*lo);
            let u = c.ub.entry(s.clone()).or_insert(*hi);
            *u = (*u).min(*hi);
            if lo == hi {
                c.konst.insert(s, *lo);
            }
        }
    }
    for (n, lo, hi) in &f.contract.sym_ranges {
        let s = Sym::Named(n.clone());
        c.lb.insert(s.clone(), *lo);
        if *hi < i64::MAX {
            c.ub.insert(s.clone(), *hi);
        }
        if lo == hi {
            c.konst.insert(s, *lo);
        }
    }
    // kernarg definitions: substituted to a fixpoint, so the result does not depend on the declaration order (an acyclic set of n definitions is fully
    // substituted after n passes; for definitions already in dependency order every pass after the first reproduces the same polynomials)
    for _ in 0..f.contract.ka_defs.len().max(1) {
        for (off, e) in &f.contract.ka_defs {
            if let Some(p) = e.to_poly(&c) {
                c.subst.insert(Sym::Ka(*off), p);
            }
        }
    }
    // grid definitions: substitute (the right-hand side is converted with the substitutions known so far)
    for (axis, e) in &f.contract.grid_defs {
        if let Some(p) = e.to_poly(&c) {
            c.subst.insert(Sym::Grid(*axis), p);
        } else if let (true, Ext::CeilDiv(inner, k)) = (f.exact, e) {
            // exact (lifted) kernels: grid = ceil(P / k) = (P + r) / k with a fresh remainder symbol r in [0, k - 1] (the relation a lifted
            // address needs: the last tile's rows end at most k - 1 past P). Shipped kernels keep the grid symbol (their proofs use it).
            if let (Some(p), true) = (inner.to_poly(&c), *k > 0) {
                let r = Sym::Named(format!("ceil_rem_grid{axis}"));
                c.lb.insert(r.clone(), 0);
                c.ub.insert(r.clone(), *k as i64 - 1);
                // P a single kernarg symbol: keep the grid symbol (>= 1) and write the kernarg as k * grid - r (both facts survive: grid >= 1
                // and k * grid <= P + k - 1); otherwise substitute the grid as (P + r) / k
                let single = p.0.len() == 1 && p.0.iter().next().map_or(false, |(m, q)| m.len() == 1 && matches!(m[0], Sym::Ka(_)) && *q == Q::int(1));
                if single {
                    let s = p.0.keys().next().unwrap()[0].clone();
                    let q = Poly::sym(Sym::Grid(*axis)).scale(Q::int(*k as i128)).sub(&Poly::sym(r));
                    for v in c.subst.values_mut() {
                        if let Some(n) = subst_sym(v, &s, &q) {
                            *v = n;
                        }
                    }
                    c.subst.insert(s, q);
                } else {
                    let q = p.add(&Poly::sym(r)).scale(Q::new(1, *k as i128));
                    c.subst.insert(Sym::Grid(*axis), q);
                }
            }
        }
    }
    c
}

fn ext_itv(cx: &Ctx, e: &Ext) -> Itv {
    e.to_itv(cx)
}
// --------------------------------------------------------------------------------------------------------------------------------------------
// IR analysis: LDS budget, global / buffer bounds

#[derive(Clone, Debug)]
enum AV {
    Int(Itv),
    /// A global pointer: the kernarg it derives from (index into `f.args`) and the byte offset interval from its base.
    Ptr { arg: Option<usize>, off: Itv },
    /// A buffer descriptor: the kernarg it was built from and its `num_records` (None = the 0xFFFFFFFF "no bounds" sentinel).
    /// a buffer descriptor: allocation, num_records, and the byte offset of its base inside the allocation (moved bases for > 4 GiB tensors)
    Rsrc { arg: Option<usize>, num: Option<Itv>, base: Itv },
    /// A wave-uniform select of pointers / descriptors with different provenance: every access through it is checked against EACH arm.
    Either(Vec<AV>),
    Other,
}

struct An<'a> {
    f: &'a Function,
    cx: Ctx,
    env: HashMap<ValueId, AV>,
    vars: HashMap<VarId, Itv>,
    /// pointer / descriptor variables: their provenance (the offset part lives in `vars`); `AV::Other` = conflicting provenances
    var_prov: HashMap<VarId, AV>,
    /// proven ring counters (var -> [min, max] of the reachable values) and their shadows (var -> (init X, ring init c, min, max): the var
    /// is X - c + ring at every point)
    rings: RingFacts,
    /// value -> the values whose definitions read it (for re-evaluation under branch facts)
    users: HashMap<ValueId, Vec<ValueId>>,
    /// per active branch fact on a value: the dependents it re-evaluated and their previous values
    dep_undo: Vec<Vec<(ValueId, Option<AV>)>>,
    /// active lanes (highest active lane + 1) under the author's EXEC (exact mode `exec_set` / save / restore); 64 otherwise
    exec_lanes: u32,
    exec_saved: HashMap<u32, u32>,
    var_init: HashMap<VarId, Itv>,
    /// E5 scope variables (`DeclareVars`): tracked flow-sensitively (a `set_var` takes effect at once; `if` arms join; a loop that
    /// writes one runs it through its fixpoint). Absent from `vars` = not yet defined on any path (an `Undef` initial value).
    scope: std::collections::HashSet<VarId>,
    /// E5 scope variables whose current value is one known IR value on every path here (flow-sensitive, like `vars`): the
    /// linear-form analysis (cross-workgroup disjointness) reads through them
    var_sym: std::cell::RefCell<HashMap<VarId, Opnd>>,
    assumed: HashMap<VarId, Itv>,
    assumed_trusted: HashMap<VarId, String>,
    tag: Option<(usize, Itv, Opnd)>,
    /// statement index of the open tile marker (reuse proofs are recorded per marker)
    tag_stmt: usize,
    /// plain global / buffer stores outside `buf_region` markers (statement -> pointer kernarg, symbolic linear byte offset, size): the
    /// cross-workgroup disjointness proof runs over them at the end
    gstores: BTreeMap<usize, (u32, Option<SLin>, i128, ARanges)>,
    /// atom key -> the operands it stands for (structurally equal values share a key)
    atom_inst: std::cell::RefCell<HashMap<(bool, u64), Vec<Opnd>>>,
    sig_memo: std::cell::RefCell<HashMap<ValueId, String>>,
    /// caps of the store being linearized: a masked store's live arm runs only when `x < bound` (x linear, bound uniform)
    caps_cur: std::cell::RefCell<Vec<(SLin, Poly, bool)>>,
    gcaps: BTreeMap<usize, Vec<(SLin, Poly, bool)>>,
    /// set_vars waiting for the end of their statement list (stmt index, var, value)
    pending_sets: Vec<(usize, VarId, Opnd)>,
    /// grid axes of the `wg_winner` branches enclosing the current statement
    winner_axes: Vec<u8>,
    /// store stmt -> the winner axes active there
    gwin: BTreeMap<usize, Vec<u8>>,
    /// trusted loop-variable assumes already counted in the ledger
    trusted_counted: std::collections::HashSet<VarId>,
    group_caps: std::cell::RefCell<BTreeMap<u32, Vec<Vec<(SLin, Poly, bool)>>>>,
    group_win: std::cell::RefCell<BTreeMap<u32, Vec<Vec<u8>>>>,
    /// the address operand (and constant offset) of the LDS access being checked: private-region checks need it symbolically
    cur_addr: Option<(Opnd, i128)>,
    acc: HashMap<VarId, Itv>,
    defs: HashMap<ValueId, Op>,
    /// values constrained by `assume`: their interval is a contract fact, so `recompute` (select-arm refinement) must not re-derive them from their operands
    assumed_vals: std::collections::HashSet<ValueId>,
    ovr: Option<(Opnd, Itv)>,
    rep: Report,
    idx: usize,
    has_asm: bool,
}

fn is_int_ty(t: amdgpu_dsl::DType) -> bool {
    matches!(t, amdgpu_dsl::DType::I32 | amdgpu_dsl::DType::U32 | amdgpu_dsl::DType::I1)
}

impl<'a> An<'a> {
    fn int_of(&self, o: &Opnd) -> Itv {
        if let Some((t, i)) = &self.ovr {
            if t == o {
                return i.clone();
            }
        }
        match o {
            Opnd::I32(c) => Itv::konst(*c as i128),
            Opnd::Val(v) => match self.env.get(v) {
                Some(AV::Int(i)) => i.clone(),
                _ => Itv::top(),
            },
            // a proven ring counter / shadow (ring_facts): its value set holds at every program point
            Opnd::Var(v) if self.rings.rings.contains_key(v) => {
                let (lo, hi) = self.rings.rings[v];
                Itv::range(Poly::int(lo), Poly::int(hi))
            }
            Opnd::Var(v) if self.rings.sym.contains_key(v) => {
                let x = self.int_of(&self.rings.sym[v]);
                let lo_c = x.lo.iter().filter_map(|p| p.as_const()).map(|q| (q.n + q.d - 1).div_euclid(q.d)).max();
                let hi_c = x.hi.iter().filter_map(|p| p.as_const()).map(|q| q.n.div_euclid(q.d)).min();
                let r = match (lo_c, hi_c) {
                    (Some(lo), Some(hi)) => self.rings.orbit_range(*v, lo, hi),
                    _ => None,
                };
                match r {
                    Some((lo, hi)) => Itv::range(Poly::int(lo), Poly::int(hi)),
                    None => self.vars.get(v).cloned().unwrap_or_default(),
                }
            }
            Opnd::Var(v) if self.rings.shadows.contains_key(v) => {
                let (x, c, lo, hi, _) = self.rings.shadows[v].clone();
                self.int_of(&x).add(&Itv::range(Poly::int(lo - c), Poly::int(hi - c)))
            }
            Opnd::Var(v) => self.vars.get(v).cloned().unwrap_or_default(),
            Opnd::F32(_) => Itv::top(),
        }
    }
    fn av_of(&self, o: &Opnd) -> AV {
        match o {
            Opnd::Val(v) => self.env.get(v).cloned().unwrap_or(AV::Other),
            // a pointer / descriptor variable: its provenance (fixed over the variable's life) with the offset tracked like an integer variable
            Opnd::Var(v) => match self.var_prov.get(v) {
                Some(AV::Ptr { arg, .. }) => AV::Ptr { arg: *arg, off: self.vars.get(v).cloned().unwrap_or_else(Itv::top) },
                Some(AV::Rsrc { arg, num, .. }) => AV::Rsrc { arg: *arg, num: num.clone(), base: self.vars.get(v).cloned().unwrap_or_else(Itv::top) },
                Some(_) => AV::Rsrc { arg: None, num: None, base: Itv::top() },
                None => AV::Other,
            },
            _ => AV::Other,
        }
    }
    /// Strip authored moves / `+ 0` / `| 0` from a word operand.
    fn word_root(&self, o: &Opnd) -> Opnd {
        let mut o = *o;
        for _ in 0..8 {
            let Opnd::Val(v) = o else { return o };
            match self.defs.get(&v) {
                Some(Op::MovS { a, .. } | Op::MovV { a, .. }) => o = *a,
                Some(Op::AddI32(a, Opnd::I32(0))) => o = *a,
                Some(Op::Bin(amdgpu_dsl::BinOp::OrB32, a, Opnd::I32(0))) => o = *a,
                _ => return o,
            }
        }
        o
    }
    /// A word that is `PtrLo(p)` (false) or `PtrHi(p)` (true): the pointer operand and which half.
    /// The pointer under a hi word `PtrHi(p) & 0xffff`.
    fn masked_hi_of(&self, o: &Opnd) -> Option<Opnd> {
        let Opnd::Val(v) = self.word_root(o) else { return None };
        match self.defs.get(&v) {
            Some(Op::AndI32(a, Opnd::I32(0xffff)) | Op::AndI32(Opnd::I32(0xffff), a)) => match self.word_of(a) {
                Some((p, true)) => Some(p),
                _ => None,
            },
            _ => None,
        }
    }
    /// A canonical spelling of a pointer value: through `PtrAdd` (base and offset) and `PtrCat` of a pointer's own words, so two SSA values
    /// computing the same pointer compare equal.
    fn ptr_key(&self, o: &Opnd, depth: usize) -> String {
        if depth == 0 {
            return format!("{o:?}");
        }
        let key_off = |me: &Self, x: &Opnd| match me.word_root(x) {
            Opnd::Val(v) => match me.defs.get(&v) {
                Some(Op::ConstI32(k)) => format!("#{k}"),
                _ => format!("v{v}"),
            },
            Opnd::I32(k) => format!("#{k}"),
            other => format!("{other:?}"),
        };
        match o {
            Opnd::Val(v) => match self.defs.get(v) {
                Some(Op::PtrAdd { base, byte_off }) => format!("add({},{})", self.ptr_key(base, depth - 1), key_off(self, byte_off)),
                Some(Op::PtrCat { lo, hi }) => match (self.word_of(lo), self.word_of(hi).or_else(|| self.masked_hi_of(hi).map(|p| (p, true)))) {
                    (Some((pl, false)), Some((ph, true))) if self.ptr_key(&pl, depth - 1) == self.ptr_key(&ph, depth - 1) => self.ptr_key(&pl, depth - 1),
                    _ => format!("v{v}"),
                },
                _ => format!("v{v}"),
            },
            other => format!("{other:?}"),
        }
    }
    /// `D + w` is the next value of ring / shadow `w`: D is a `Select` by the ring's own rule on the ring's current value (no set of the
    /// ring or of `w` between D's compare and this statement).
    fn ring_step_add(&self, d: ValueId, w: VarId) -> bool {
        let ring = if self.rings.rings.contains_key(&w) || self.rings.sym.contains_key(&w) { w } else if let Some(r) = self.ring_of_shadow(w) { r } else { return false };
        if self.rings.steps.get(&d) == Some(&ring) {
            return true;
        }
        let Some(&(pred, ty, var_left, k, ka, kb)) = self.rings.rules.get(&ring) else { return false };
        let konst = |o: &Opnd| -> Option<i128> {
            match self.word_root(o) {
                Opnd::I32(x) => Some(x as i128),
                Opnd::Val(v) => match self.defs.get(&v) {
                    Some(Op::ConstI32(x)) => Some(*x as i128),
                    _ => None,
                },
                _ => None,
            }
        };
        let Some(Op::Select { cond: Opnd::Val(c), a, b, .. }) = self.defs.get(&d) else { return false };
        if konst(a) != Some(ka) || konst(b) != Some(kb) {
            return false;
        }
        let Some(Op::Cmp { pred: p2, ty: t2, a: ca, b: cb }) = self.defs.get(c) else { return false };
        let side_ok = match (ca, cb) {
            (Opnd::Var(x), o) if *x == ring && var_left => konst(o) == Some(k),
            (o, Opnd::Var(x)) if *x == ring && !var_left => konst(o) == Some(k),
            _ => false,
        };
        if !(side_ok && *p2 == pred && *t2 == ty) {
            return false;
        }
        let Some(&pc) = self.rings.let_pos.get(c) else { return false };
        let now = self.idx;
        let moved = |v: VarId| self.rings.set_pos.get(&v).map_or(false, |l| l.iter().any(|&j| j > pc && j < now));
        pc < now && !moved(ring) && !moved(w)
    }
    /// The ring a shadow follows (its range and init are recorded in `rings.1`; the ring is the one whose steps update it).
    fn ring_of_shadow(&self, w: VarId) -> Option<VarId> {
        self.rings.shadows.get(&w).map(|s| s.4)
    }
    fn same_ptr(&self, a: &Opnd, b: &Opnd) -> bool {
        a == b || self.ptr_key(a, 8) == self.ptr_key(b, 8)
    }
    fn word_of(&self, o: &Opnd) -> Option<(Opnd, bool)> {
        let Opnd::Val(v) = self.word_root(o) else { return None };
        match self.defs.get(&v) {
            Some(Op::PtrLo(p)) => Some((*p, false)),
            Some(Op::PtrHi(p)) => Some((*p, true)),
            _ => None,
        }
    }
    /// [lo(p), hi(p) & 0xffff, nr, 0x00020000] -> (p, nr); else what does not match.
    fn rsrc_words(&self, w: &[Opnd; 4]) -> Result<(Opnd, Opnd), String> {
        let (p, hi) = self.word_of(&w[0]).ok_or("word 0 is not the low half of a pointer")?;
        if hi {
            return Err("word 0 is the HIGH half of a pointer".into());
        }
        let w1 = self.word_root(&w[1]);
        let masked = match w1 {
            Opnd::Val(v) => match self.defs.get(&v) {
                Some(Op::AndI32(a, Opnd::I32(0xffff)) | Op::AndI32(Opnd::I32(0xffff), a)) => Some(*a),
                Some(Op::Bin(amdgpu_dsl::BinOp::OrB32, _, Opnd::I32(k))) if *k != 0 => return Err(format!("word 1 sets a stride ({k:#x}): not supported")),
                _ => None,
            },
            _ => None,
        };
        // (a valid pointer is a canonical 48-bit address: its hi word already has bits 16..31 clear, so the mask may be absent)
        match masked.and_then(|a| self.word_of(&a)).or_else(|| self.word_of(&w1)) {
            Some((p1, true)) if self.same_ptr(&p1, &p) => {}
            _ => return Err("word 1 is not (high half of the same pointer) & 0xffff".into()),
        }
        let fmt = match self.word_root(&w[3]) {
            Opnd::I32(k) => Some(k),
            Opnd::Val(v) => match self.defs.get(&v) {
                Some(Op::ConstI32(k)) => Some(*k),
                _ => None,
            },
            _ => None,
        };
        if fmt != Some(0x0002_0000) {
            return Err(format!("word 3 is not the raw-buffer format constant 0x00020000 ({fmt:?})"));
        }
        Ok((p, self.word_root(&w[2])))
    }
    /// The integer part a variable carries for a value: the value itself, or the byte offset of a pointer / the base offset of a descriptor
    /// (whose provenance is recorded; two different provenances for one variable make it unknown, GV-GLB-PROV at its use).
    fn var_part(&mut self, v: VarId, o: &Opnd) -> Itv {
        let (prov, part) = match self.av_of(o) {
            AV::Ptr { arg, off } => (Some(AV::Ptr { arg, off: Itv::konst(0) }), off),
            AV::Rsrc { arg, num, base } => (Some(AV::Rsrc { arg, num, base: Itv::konst(0) }), base),
            _ => (None, self.int_of(o)),
        };
        if let Some(p) = prov {
            let same = |a: &AV, b: &AV| match (a, b) {
                (AV::Ptr { arg: x, .. }, AV::Ptr { arg: y, .. }) => x == y,
                (AV::Rsrc { arg: x, num: n1, .. }, AV::Rsrc { arg: y, num: n2, .. }) => x == y && n1 == n2,
                _ => false,
            };
            match self.var_prov.get(&v) {
                Some(old) if !same(old, &p) => {
                    self.var_prov.insert(v, AV::Other);
                }
                Some(_) => {}
                None => {
                    self.var_prov.insert(v, p);
                }
            }
        }
        part
    }
    fn arg_extent(&self, arg: usize) -> Option<&amdgpu_dsl::ArgExtent> {
        let name = &self.f.args.get(arg)?.name;
        self.f.contract.extents.iter().find(|e| &e.arg == name)
    }
    fn arg_name(&self, arg: usize) -> String {
        self.f.args.get(arg).map(|a| a.name.clone()).unwrap_or_else(|| "?".to_string())
    }

    fn nonneg(&self, i: &Itv) -> bool {
        i.nonneg(&self.cx)
    }

    fn shape(&self, o: &Opnd, depth: usize) -> String {
        match o {
            Opnd::I32(c) => format!("{c}"),
            Opnd::Var(v) => format!("var{}", v.0),
            Opnd::Val(v) => match self.defs.get(v) {
                Some(Op::ConstI32(k)) => format!("{k}"),
                Some(op) if depth > 0 => {
                    let name = format!("{op:?}");
                    let name = name.split(|c: char| c == '(' || c == ' ').next().unwrap_or("?").to_string();
                    let ops: Vec<String> = op.operands().iter().map(|x| self.shape(x, depth - 1)).collect();
                    format!("{name}({})", ops.join(", "))
                }
                _ => format!("v{v}"),
            },
            Opnd::F32(_) => "f".to_string(),
        }
    }

    /// Where does the lack of a bound come from? Follows the first unbounded operand down to a leaf.
    fn top_origin(&self, o: &Opnd, depth: usize) -> String {
        match o {
            Opnd::Var(v) => {
                let upd: Vec<String> = self.f.stmts().iter().filter_map(|s| if let Stmt::SetVar { var, value } = s { if var == v { Some(self.shape(value, 3)) } else { None } } else { None }).take(2).collect();
                let init: Vec<String> = self.f.stmts().iter().filter_map(|s| if let Stmt::BeginLoop { vars, init } | Stmt::DeclareVars { vars, init } = s { vars.iter().position(|x| x == v).map(|k| self.shape(&init[k], 3)) } else { None }).take(1).collect();
                format!("loop variable var{} = {} (not bounded by the fixpoint; initial {} = {}; updated by {}; add b.assume(var, lo, hi) at the top of the loop body)", v.0, self.vars.get(v).map(|i| i.show()).unwrap_or_default(), init.join(""), self.var_init.get(v).map(|i| i.show()).unwrap_or_default(), upd.join(" / "))
            }
            Opnd::Val(v) => {
                let Some(op) = self.defs.get(v) else { return format!("v{v}") };
                let name = format!("{op:?}");
                let name = name.split(|c: char| c == '(' || c == ' ').next().unwrap_or("?").to_string();
                if depth == 0 {
                    return format!("v{v} = {name} ...");
                }
                for x in op.operands() {
                    let unbounded = match &x {
                        Opnd::Val(y) => !matches!(self.env.get(y), Some(AV::Int(i)) if !i.lo.is_empty() && !i.hi.is_empty()),
                        Opnd::Var(y) => self.vars.get(y).map_or(true, |i| i.lo.is_empty() || i.hi.is_empty()),
                        _ => false,
                    };
                    if unbounded {
                        return format!("v{v} = {name} <- {}", self.top_origin(&x, depth - 1));
                    }
                }
                format!("v{v} = {name} (no interval rule for this operation)")
            }
            _ => "constant".to_string(),
        }
    }

    /// `lo >= 0 && hi + size <= limit`; returns the failing reason.
    fn prove_range(&self, addr: &Itv, size: i128, limit: &Itv) -> Result<(), String> {
        if !self.nonneg(addr) {
            return Err(format!("lower bound not provably >= 0: address {}", addr.show()));
        }
        let ok = addr.hi.iter().any(|h| limit.lo.iter().any(|l| self.cx.nonneg(&l.sub(&h.add(&Poly::int(size))))));
        if ok {
            Ok(())
        } else {
            Err(format!("upper bound not provably <= limit - {size}: address {} vs limit {}", addr.show(), limit.show()))
        }
    }

    fn check_lds_at(&mut self, what: &str, o: &Opnd, size: i128) {
        self.cur_addr = Some((*o, 0));
        let a = self.int_of(o);
        let before = self.rep.findings.len();
        self.check_lds(what, &a, size);
        if self.rep.findings.len() > before {
            let why = if a.lo.is_empty() { format!(" | unbounded because: {}", self.top_origin(o, 6)) } else { String::new() };
            let sh = self.shape(o, 5);
            if let Some(last) = self.rep.findings.last_mut() {
                last.msg.push_str(&format!("{why} | address = {sh}"));
            }
        }
    }

    /// Under a marker of an `lds_owned` tile: is this access (the current address operand) in the issuing thread's own words? The address must be
    /// `base + col_stride tid + k row_stride + r` with 0 <= r <= col_stride - size, every other term a multiple of row_stride, and the columns of
    /// all threads inside one row (row_stride >= threads col_stride).
    fn note_owned(&mut self, size: i128) {
        let Some((ti, _, _)) = self.tag.clone() else { return };
        let t = &self.f.contract.tiles[ti];
        let Some((col, row)) = t.owned else { return };
        let (col, row, base) = (col as i128, row as i128, t.base as i128);
        let ok = (|| {
            let (o, off) = self.cur_addr?;
            let (atoms, c) = self.lin(&o)?;
            let c = c + off;
            if row < self.f.workgroup_size as i128 * col {
                return None;
            }
            let mut tid_seen = false;
            for ((is_var, id), k) in &atoms {
                let is_tid = !is_var && matches!(self.defs.get(id), Some(Op::WorkitemId));
                if is_tid {
                    if *k != col || tid_seen {
                        return None;
                    }
                    tid_seen = true;
                } else if k % row != 0 {
                    return None;
                }
            }
            let r = (c - base).rem_euclid(row);
            (tid_seen && r + size <= col).then_some(())
        })()
        .is_some();
        if self.rep.quiet == 0 {
            let e = self.rep.owned.entry(self.tag_stmt).or_insert(true);
            *e &= ok;
        }
    }

    // ---------------------------------------------------------------------------------------------------------------- cross-workgroup stores
    /// Symbolic linear form: atom -> coefficient (a polynomial over the contract symbols), plus a constant polynomial. Uniform values with an
    /// exact (point) interval are constants; workgroup ids, lane-derived values and loop variables are atoms.
    fn slin(&self, o: &Opnd, depth: usize) -> Option<SLin> {
        if depth > 32 {
            return None;
        }
        let atom = |k: (bool, u64), o: Opnd| {
            self.atom_inst.borrow_mut().entry(k).or_default().push(o);
            Some((BTreeMap::from([(k, Poly::int(1))]), Poly::int(0)))
        };
        match o {
            Opnd::I32(c) => Some((BTreeMap::new(), Poly::int(*c as i128))),
            Opnd::F32(_) => None,
            Opnd::Var(v) => {
                let s = self.var_sym.borrow().get(v).copied();
                match s {
                    Some(x) if depth < 24 => self.slin(&x, depth + 1),
                    _ => atom((true, v.0 as u64), *o),
                }
            }
            Opnd::Val(v) => {
                let def = self.defs.get(v);
                if let Some(Op::ConstI32(c)) = def {
                    return Some((BTreeMap::new(), Poly::int(*c as i128)));
                }
                // an output of a declared workgroup bijection is a digit of its own
                if self.f.contract.wg_maps.iter().any(|m| m.outs.contains(v)) {
                    return atom(self.atom_key(*v), *o);
                }
                if !matches!(def, Some(Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ | Op::WorkitemId)) {
                    if let Some(p) = self.point_of(o) {
                        return Some((BTreeMap::new(), p));
                    }
                }
                match def {
                    Some(Op::AddI32(a, b)) => Some(slin_add(self.slin(a, depth + 1)?, self.slin(b, depth + 1)?)),
                    // min(x, y) with y a uniform point: the clamp cap x < y + 1 (see `disjoint_group_why`)
                    Some(Op::MinI32(a, b)) if self.point_of(b).is_some() => {
                        let bound = self.point_of(b).unwrap().add(&Poly::int(1));
                        if let Some(l) = self.slin(a, depth + 1) {
                            self.caps_cur.borrow_mut().push((l, bound, true));
                        }
                        self.slin(a, depth + 1)
                    }
                    // readfirstlane of a wave-uniform value is that value
                    Some(Op::ReadFirstLane(a)) if self.uniform(a, 0) => self.slin(a, depth + 1),
                    Some(Op::MovV { a, .. } | Op::MovS { a, .. }) => self.slin(a, depth + 1),
                    Some(Op::Tri(amdgpu_dsl::TriOp::Add3U32, a, b, c)) => {
                        Some(slin_add(slin_add(self.slin(a, depth + 1)?, self.slin(b, depth + 1)?), self.slin(c, depth + 1)?))
                    }
                    Some(Op::Tri(amdgpu_dsl::TriOp::LshlAddU32, a, b, c)) => match self.int_of(b).point_const() {
                        Some(s) if s.d == 1 && (0..31).contains(&s.n) => {
                            let (x, k) = self.slin(a, depth + 1)?;
                            let f = Q::int(1i128 << s.n);
                            Some(slin_add((x.into_iter().map(|(kk, p)| (kk, p.scale(f))).collect(), k.scale(f)), self.slin(c, depth + 1)?))
                        }
                        _ => atom(self.atom_key(*v), *o),
                    },
                    Some(Op::MulI32(a, b)) => {
                        let (x, y) = (self.slin(a, depth + 1)?, self.slin(b, depth + 1)?);
                        let (k, (atoms, c)) = if x.0.is_empty() {
                            (x.1, y)
                        } else if y.0.is_empty() {
                            (y.1, x)
                        } else {
                            return atom(self.atom_key(*v), *o);
                        };
                        let mut m = BTreeMap::new();
                        for (a, p) in atoms {
                            m.insert(a, p.mul(&k)?);
                        }
                        Some((m, c.mul(&k)?))
                    }
                    Some(Op::ShlI32(a, b)) => match self.int_of(b).point_const() {
                        Some(s) if s.d == 1 && (0..31).contains(&s.n) => {
                            let (x, c) = self.slin(a, depth + 1)?;
                            let f = Q::int(1i128 << s.n);
                            Some((x.into_iter().map(|(k, p)| (k, p.scale(f))).collect(), c.scale(f)))
                        }
                        _ => atom(self.atom_key(*v), *o),
                    },
                    // a masked store: the sentinel arm (an offset >= 2^30, dropped by the descriptor's range check, which the bounds proof
                    // establishes) writes nothing - only the live arm matters
                    Some(Op::Select { cond, a, b, .. }) => {
                        // (a negative constant is a huge unsigned buffer offset)
                        let big = |o: &Opnd| matches!(o, Opnd::I32(c) if *c < 0) || self.int_of(o).lo.iter().any(|l| self.cx.le(&Poly::int(1 << 30), l));
                        if big(b) {
                            self.note_cap(cond, true, depth, false);
                            self.slin(a, depth + 1)
                        } else if big(a) {
                            self.note_cap(cond, false, depth, false);
                            self.slin(b, depth + 1)
                        } else if self.is_clamp(cond, a, b) {
                            // min(x, bound - 1) as `x < bound ? x : bound - 1`: a clamp cap (see `disjoint_group_why`)
                            self.note_cap(cond, true, depth, true);
                            self.slin(a, depth + 1)
                        } else {
                            atom(self.atom_key(*v), *o)
                        }
                    }
                    _ => atom(self.atom_key(*v), *o),
                }
            }
        }
    }

    /// The key of a value atom: workgroup ids get fixed keys (u64::MAX - axis); any other value the hash of its structural signature, so the
    /// same expression recomputed for another store is the same digit (sound: the proof only needs every instance inside the key's range).
    fn atom_key(&self, v: ValueId) -> (bool, u64) {
        // the outputs of a `wg_union` share one key (one index computed per store class)
        if let Some(i) = self.f.contract.wg_maps.iter().position(|m| m.union && m.outs.contains(&v)) {
            return (false, (1u64 << 62) + i as u64);
        }
        match self.defs.get(&v) {
            Some(Op::WorkgroupIdX) => return (false, u64::MAX),
            Some(Op::WorkgroupIdY) => return (false, u64::MAX - 1),
            Some(Op::WorkgroupIdZ) => return (false, u64::MAX - 2),
            _ => {}
        }
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let s = self.sig(v, 0);
        s.hash(&mut h);
        let k = (false, h.finish() & !(1u64 << 63));
        if std::env::var("AMDC_GW_DEBUG").map_or(false, |v| !v.is_empty()) {
            self.sig_memo.borrow_mut().insert(u32::MAX - (k.1 % 1_000_000) as u32, s.chars().take(300).collect());
        }
        k
    }

    /// Structural signature of a value: its defining op with every value operand replaced by that operand's signature.
    fn sig(&self, v: ValueId, depth: usize) -> String {
        if let Some(s) = self.sig_memo.borrow().get(&v) {
            return s.clone();
        }
        let s = match self.defs.get(&v) {
            None => format!("v{v}"),
            Some(_) if depth > 16 => format!("v{v}"),
            Some(op) => {
                let d = format!("{op:?}");
                let mut out = String::new();
                let mut rest = d.as_str();
                while let Some(i) = rest.find("Val(") {
                    out.push_str(&rest[..i]);
                    let tail = &rest[i + 4..];
                    let j = tail.find(')').unwrap_or(0);
                    match tail[..j].parse::<ValueId>() {
                        Ok(x) => {
                            out.push('[');
                            out.push_str(&self.sig(x, depth + 1));
                            out.push(']');
                        }
                        Err(_) => out.push_str(&rest[i..i + 4 + j]),
                    }
                    rest = &tail[j..];
                }
                out.push_str(rest);
                out
            }
        };
        self.sig_memo.borrow_mut().insert(v, s.clone());
        s
    }

    /// The live arm of a masked store runs only where `cond == live`: a condition `x < bound` (bound a uniform value with an exact interval)
    /// caps the linear value x - recorded for the store being linearized (`disjoint_group_why` turns it into one composite digit).
    fn note_cap(&self, cond: &Opnd, live: bool, depth: usize, clamp: bool) {
        let Some((x, y)) = self.lt_of(cond, live) else { return };
        let it = self.int_of(&y);
        if !(it.lo.len() == 1 && it.hi.len() == 1 && it.lo[0] == it.hi[0]) {
            return;
        }
        if let Some(l) = self.slin(&x, depth + 1) {
            self.caps_cur.borrow_mut().push((l, it.lo[0].clone(), clamp));
        }
    }

    /// `Some(axis)` when loop variable `v` starts at the workgroup id of `axis` and every `set_var` adds exactly the grid size of that axis
    /// (`v = wg + k grid`): the values of different workgroups are disjoint residue classes.
    fn persistent_axis(&self, v: VarId) -> Option<u8> {
        let stmts = self.f.stmts();
        let init = stmts.iter().find_map(|s| match s {
            Stmt::BeginLoop { vars, init } | Stmt::DeclareVars { vars, init } => vars.iter().position(|x| *x == v).map(|k| init[k]),
            _ => None,
        })?;
        let ax = match init {
            Opnd::Val(x) => match self.defs.get(&x)? {
                Op::WorkgroupIdX => 0u8,
                Op::WorkgroupIdY => 1,
                Op::WorkgroupIdZ => 2,
                _ => return None,
            },
            _ => return None,
        };
        let grid = self.cx.sym_poly(Sym::Grid(ax));
        let mut any = false;
        for s in stmts.iter() {
            if let Stmt::SetVar { var, value } = s {
                if *var != v {
                    continue;
                }
                any = true;
                let Opnd::Val(x) = value else { return None };
                let step = match self.defs.get(x)? {
                    Op::AddI32(Opnd::Var(w), s) | Op::AddI32(s, Opnd::Var(w)) if *w == v => *s,
                    _ => return None,
                };
                let p = self.point_of(&step)?;
                if !(self.cx.le(&p, &grid) && self.cx.le(&grid, &p)) {
                    return None;
                }
            }
        }
        any.then_some(ax)
    }

    /// Known trailing zero bits of an integer operand (its values are multiples of 2^tz), capped at 30.
    fn tz(&self, o: &Opnd, depth: usize) -> u32 {
        if depth > 16 {
            return 0;
        }
        let tzc = |c: i128| if c == 0 { 30 } else { (c.trailing_zeros()).min(30) };
        match o {
            Opnd::I32(c) => tzc(*c as i128),
            Opnd::Val(v) => match self.defs.get(v) {
                Some(Op::ConstI32(c)) => tzc(*c as i128),
                Some(Op::ShlI32(a, b)) => match self.int_of(b).point_const() {
                    Some(s) if s.d == 1 && (0..31).contains(&s.n) => (self.tz(a, depth + 1) + s.n as u32).min(30),
                    _ => 0,
                },
                Some(Op::AndI32(a, b)) => self.tz(a, depth + 1).max(self.tz(b, depth + 1)),
                Some(Op::AddI32(a, b)) => self.tz(a, depth + 1).min(self.tz(b, depth + 1)),
                Some(Op::MulI32(a, b)) => (self.tz(a, depth + 1) + self.tz(b, depth + 1)).min(30),
                _ => 0,
            },
            _ => 0,
        }
    }

    /// The exact (point) value of a uniform operand, if its interval is one polynomial.
    fn point_of(&self, o: &Opnd) -> Option<Poly> {
        // a lower bound that is also an upper bound (an interval keeps several candidate bounds)
        let it = self.int_of(o);
        it.lo.iter().find(|l| it.hi.contains(l)).cloned()
    }

    /// `cond == live` as `x < y`.
    fn lt_of(&self, cond: &Opnd, live: bool) -> Option<(Opnd, Opnd)> {
        let Opnd::Val(cv) = cond else { return None };
        match (self.defs.get(cv), live) {
            (Some(Op::CmpLtI32(x, y)), true) | (Some(Op::CmpGeI32(x, y)), false) => Some((*x, *y)),
            (Some(Op::NotI1(c)), l) => self.lt_of(c, !l),
            _ => None,
        }
    }

    /// `cond ? a : b` is `min(a, y - 1)`: cond is `a < y` and b is exactly y - 1 (both uniform points).
    fn is_clamp(&self, cond: &Opnd, a: &Opnd, b: &Opnd) -> bool {
        let Some((x, y)) = self.lt_of(cond, true) else { return false };
        let same = x == *a || matches!((x, a), (Opnd::Val(p), Opnd::Val(q)) if self.sig(p, 0) == self.sig(*q, 0));
        if !same {
            return false;
        }
        let (iy, ib) = (self.int_of(&y), self.int_of(b));
        let point = |i: &Itv| i.lo.len() == 1 && i.hi.len() == 1 && i.lo[0] == i.hi[0];
        point(&iy) && point(&ib) && ib.lo[0] == iy.lo[0].sub(&Poly::int(1))
    }

    /// Wave-uniform (every lane holds the same value): built from constants, kernargs, workgroup ids and readfirstlane results only.
    fn uniform(&self, o: &Opnd, depth: usize) -> bool {
        if depth > 24 {
            return false;
        }
        match o {
            Opnd::I32(_) | Opnd::F32(_) => true,
            Opnd::Var(_) => false,
            Opnd::Val(v) => match self.defs.get(v) {
                None | Some(Op::WorkitemId) => false,
                Some(Op::ReadFirstLane(_) | Op::MovS { .. } | Op::KernargI32 { .. } | Op::ConstI32(_) | Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ) => true,
                Some(Op::MovV { a, .. }) => self.uniform(a, depth + 1),
                Some(op @ (Op::AddI32(..) | Op::MulI32(..) | Op::ShlI32(..) | Op::LshrI32(..) | Op::AndI32(..) | Op::XorI32(..) | Op::MinI32(..) | Op::Select { .. } | Op::Bin(..) | Op::Tri(..) | Op::CmpLtI32(..) | Op::CmpGeI32(..) | Op::NotI1(..))) => {
                    op.operands().iter().all(|x| self.uniform(x, depth + 1))
                }
                _ => false,
            },
        }
    }

    /// The pointer kernarg a pointer value derives from and its byte offset as a symbolic linear form.
    fn ptr_slin(&self, o: &Opnd, depth: usize) -> Option<(u32, SLin)> {
        if depth > 32 {
            return None;
        }
        // a pointer scope variable (exact mode): its recorded provenance, when its offset is one known constant on every path
        if let Opnd::Var(x) = o {
            if let Some(AV::Ptr { arg: Some(a), .. }) = self.var_prov.get(x) {
                let c = self.vars.get(x)?.point_const().filter(|q| q.d == 1)?;
                let off = self.f.args.get(*a)?.kernarg_offset;
                return Some((off, (BTreeMap::new(), Poly::int(c.n))));
            }
            return None;
        }
        let Opnd::Val(v) = o else { return None };
        match self.defs.get(v)? {
            Op::KernargPtr { offset } => Some((*offset, (BTreeMap::new(), Poly::int(0)))),
            Op::PtrAdd { base, byte_off } => {
                let (arg, l) = self.ptr_slin(base, depth + 1)?;
                Some((arg, slin_add(l, self.slin(byte_off, depth + 1)?)))
            }
            // a pointer reassembled from its own words (exact mode: the author's SGPR pair, the high word possibly masked to 48 bits)
            Op::PtrCat { lo, .. } => {
                let p = self.ptr_of_word(lo)?;
                if self.ptr_equiv_words(o) {
                    self.ptr_slin(&p, depth + 1)
                } else {
                    None
                }
            }
            Op::MovS { a, .. } => self.ptr_slin(a, depth + 1),
            Op::PtrMadU32 { base, a, b } => {
                let (arg, l) = self.ptr_slin(base, depth + 1)?;
                let (x, y) = (self.slin(a, depth + 1)?, self.slin(b, depth + 1)?);
                let (k, (atoms, c2)) = if x.0.is_empty() {
                    (x.1, y)
                } else if y.0.is_empty() {
                    (y.1, x)
                } else {
                    return None;
                };
                let mut m = BTreeMap::new();
                for (at, p) in atoms {
                    m.insert(at, p.mul(&k)?);
                }
                Some((arg, slin_add(l, (m, c2.mul(&k)?))))
            }
            _ => None,
        }
    }

    /// Records a plain global store (not under a `buf_region` marker) for the cross-workgroup disjointness proof.
    fn record_store(&mut self, base: Option<(u32, SLin)>, extra: Option<SLin>, imm: i128, size: i128) {
        if self.rep.quiet > 0 || self.tag.as_ref().map_or(false, |(ti, _, _)| self.f.contract.tiles[*ti].global) {
            return;
        }
        let rec = match (base, extra) {
            (Some((arg, l)), Some(e)) => {
                let (a, c) = slin_add(l, e);
                (arg, Some((a, c.add(&Poly::int(imm)))))
            }
            (Some((arg, _)), None) => (arg, None),
            _ => (u32::MAX, None),
        };
        let ranges = rec.1.as_ref().and_then(|(atoms, _)| {
            let mut m = BTreeMap::new();
            for k in atoms.keys() {
                // a `wg_union` index: the fact's range
                if !k.0 && k.1 >= (1u64 << 62) && k.1 < (1u64 << 62) + 1024 {
                    let wm = &self.f.contract.wg_maps[(k.1 - (1u64 << 62)) as usize];
                    if let Some((lo, hi)) = &wm.range {
                        let (il, ih) = (ext_itv(&self.cx, lo), ext_itv(&self.cx, hi));
                        m.insert(*k, (il.lo.first()?.clone(), ih.hi.first()?.clone()));
                        continue;
                    }
                }
                let insts = self.atom_inst.borrow().get(k).cloned().unwrap_or_default();
                let mut r: Option<(Poly, Poly)> = None;
                for o in insts {
                    let it = self.int_of(&o);
                    // the tightest comparable bounds of the instance (an interval keeps several candidate bounds)
                    // a persistent-loop variable (wg + k grid, k >= 0) is >= 0 even when the loop widened its interval
                    let persistent = matches!(o, Opnd::Var(v) if self.persistent_axis(v).is_some());
                    let mut l = match it.lo.first() {
                        Some(x) => x.clone(),
                        None if persistent => Poly::int(0),
                        None => return None,
                    };
                    for x in it.lo.iter().skip(1) {
                        if self.cx.le(&l, x) {
                            l = x.clone();
                        }
                    }
                    // no upper bound: a huge sentinel (only the TOP digit of the radix may be unbounded - its span is never compared)
                    let mut h = it.hi.first().cloned().unwrap_or(Poly::int(1i128 << 100));
                    for x in it.hi.iter().skip(1) {
                        if self.cx.le(x, &h) {
                            h = x.clone();
                        }
                    }
                    r = Some(match r {
                        None => (l, h),
                        Some((l0, h0)) => (self.pmin(&l0, &l)?, self.pmax(&h0, &h)?),
                    });
                }
                m.insert(*k, r?);
            }
            Some(m)
        });
        self.gstores.insert(self.idx, (rec.0, rec.1, size, ranges));
        self.gwin.insert(self.idx, self.winner_axes.clone());
        let caps = std::mem::take(&mut *self.caps_cur.borrow_mut());
        self.gcaps.insert(self.idx, caps);
    }

    fn pmin(&self, a: &Poly, b: &Poly) -> Option<Poly> {
        if self.cx.le(a, b) {
            Some(a.clone())
        } else if self.cx.le(b, a) {
            Some(b.clone())
        } else {
            None
        }
    }
    fn pmax(&self, a: &Poly, b: &Poly) -> Option<Poly> {
        if self.cx.le(a, b) {
            Some(b.clone())
        } else if self.cx.le(b, a) {
            Some(a.clone())
        } else {
            None
        }
    }

    fn record_global_store(&mut self, base: &Opnd, byte_off: i128, size: i128) {
        self.caps_cur.borrow_mut().clear();
        let b = self.ptr_slin(base, 0);
        let e = b.as_ref().map(|_| (BTreeMap::new(), Poly::int(0)));
        self.record_store(b, e, byte_off, size);
    }

    /// The pointer whose low word `w` is (through authored moves): `PtrLo(p)` -> p.
    fn ptr_of_word(&self, w: &Opnd) -> Option<Opnd> {
        let r = self.word_root(w);
        let Opnd::Val(v) = r else { return None };
        match self.defs.get(&v) {
            Some(Op::PtrLo(p)) => Some(*p),
            _ => None,
        }
    }
    /// A PtrCat whose high word is the same pointer's high word (or that word masked to its low 16 bits, the identity on a canonical
    /// 48-bit address).
    fn ptr_equiv_words(&self, cat: &Opnd) -> bool {
        let Opnd::Val(v) = cat else { return false };
        let Some(Op::PtrCat { lo, hi }) = self.defs.get(v) else { return false };
        let Some(p) = self.ptr_of_word(lo) else { return false };
        let h = self.word_root(hi);
        let Opnd::Val(hv) = h else { return false };
        match self.defs.get(&hv) {
            Some(Op::PtrHi(q)) => *q == p,
            Some(Op::AndI32(a, Opnd::I32(0xffff))) | Some(Op::AndI32(Opnd::I32(0xffff), a)) => {
                matches!(self.word_root(a), Opnd::Val(x) if matches!(self.defs.get(&x), Some(Op::PtrHi(q)) if *q == p))
            }
            _ => false,
        }
    }

    fn record_buffer_store(&mut self, rsrc: &Opnd, voff: &Option<Opnd>, soff: &Opnd, imm: i128, size: i128) {
        self.caps_cur.borrow_mut().clear();
        let base = match rsrc {
            Opnd::Val(v) => match self.defs.get(v) {
                Some(Op::BufferRsrc { base, .. } | Op::BufferRsrcDyn { base, .. }) => self.ptr_slin(base, 0),
                // exact mode: a descriptor assembled from words - its base is the pointer of word 0 (word 1 its high half, checked by the
                // descriptor recognition)
                Some(Op::RsrcCat { words }) => self.ptr_of_word(&words[0]).and_then(|p| self.ptr_slin(&p, 0)),
                _ => None,
            },
            _ => None,
        };
        let extra = (|| {
            let a = match voff {
                Some(o) => self.slin(o, 0)?,
                None => (BTreeMap::new(), Poly::int(0)),
            };
            Some(slin_add(a, self.slin(soff, 0)?))
        })();
        self.record_store(base, extra, imm, size);
    }

    /// Cross-workgroup disjointness of the recorded stores, per pointer argument: the offset map (workgroup ids, lane values, loop variables
    /// and the per-store constants as digits) is injective - sorted by coefficient, every coefficient is at least the span of the smaller digits
    /// plus the access size (a mixed radix) - and every workgroup axis that can exceed 1 is one of its digits, so two different workgroups can
    /// never write the same byte. Returns the number of stores proven / not proven.
    fn global_disjointness(&self) -> (usize, usize) {
        let mut by_arg: BTreeMap<u32, Vec<&(u32, Option<SLin>, i128, ARanges)>> = BTreeMap::new();
        let mut caps_by_arg: BTreeMap<u32, Vec<Vec<(SLin, Poly, bool)>>> = BTreeMap::new();
        for (i, s) in &self.gstores {
            by_arg.entry(s.0).or_default().push(s);
            caps_by_arg.entry(s.0).or_default().push(self.gcaps.get(i).cloned().unwrap_or_default());
        }
        *self.group_caps.borrow_mut() = caps_by_arg;
        let mut win_by_arg: BTreeMap<u32, Vec<Vec<u8>>> = BTreeMap::new();
        for (i, s) in &self.gstores {
            win_by_arg.entry(s.0).or_default().push(self.gwin.get(i).cloned().unwrap_or_default());
        }
        *self.group_win.borrow_mut() = win_by_arg;
        let (mut ok, mut bad) = (0, 0);
        for (arg, ss) in by_arg {
            if arg != u32::MAX && self.disjoint_group(&ss) {
                ok += ss.len();
            } else {
                bad += ss.len();
            }
        }
        (ok, bad)
    }

    fn disjoint_group(&self, ss: &[&(u32, Option<SLin>, i128, ARanges)]) -> bool {
        let r = self.disjoint_group_why(ss);
        if std::env::var("AMDC_GW_DEBUG").map_or(false, |v| !v.is_empty()) {
            let show = |l: &Option<SLin>| match l {
                Some((a, c)) => format!("{} + {}", a.iter().map(|(k, p)| format!("{}{:x}*({})", if k.0 { "var" } else { "a" }, k.1, p.show())).collect::<Vec<_>>().join(" + "), c.show()),
                None => "<no linear form>".into(),
            };
            eprintln!("gw: arg 0x{:x}: {} stores: {:?}; first: {}", ss[0].0, ss.len(), r, show(&ss[0].1));
            let mut keys: std::collections::BTreeSet<(bool, u64)> = Default::default();
            for s in ss {
                if let Some((a, _)) = &s.1 {
                    keys.extend(a.keys().cloned());
                }
            }
            for k in keys.iter().take(12) {
                let n = ss.iter().filter(|s| s.1.as_ref().map_or(false, |(a, _)| a.contains_key(k))).count();
                let sg = if k.0 { format!("var{}", k.1) } else { self.sig_memo.borrow().get(&(u32::MAX - (k.1 % 1_000_000) as u32)).cloned().unwrap_or_default() };
                eprintln!("gw:   atom {:x} in {n}/{} stores: {}", k.1, ss.len(), sg);
            }
        }
        r.is_ok()
    }

    fn disjoint_group_why(&self, ss: &[&(u32, Option<SLin>, i128, ARanges)]) -> Result<(), String> {
        let cx = &self.cx;
        // 1. the union of the stores' atoms: an atom's coefficient must agree among the stores that use it; a store without it has digit value 0
        let mut atoms: BTreeMap<(bool, u64), Poly> = BTreeMap::new();
        let mut consts: Vec<Poly> = Vec::new();
        let mut size: i128 = 0;
        for s in ss {
            let Some((a, c)) = s.1.as_ref() else { return Err("no linear form".into()) };
            for (key, coef) in a {
                match atoms.get(key) {
                    Some(c0) if c0 != coef => return Err("an atom with different coefficients".into()),
                    _ => {
                        atoms.insert(*key, coef.clone());
                    }
                }
            }
            consts.push(c.clone());
            size = size.max(s.2);
        }
        // 2. every atom's range: hull over every store's instances (0 for a store that does not use it)
        let mut ranges: BTreeMap<(bool, u64), (Poly, Poly)> = BTreeMap::new();
        for k in atoms.keys() {
            let mut r: Option<(Poly, Poly)> = None;
            for s in ss {
                let (l, h) = match s.3.as_ref() {
                    None => return Err("atom without a range".into()),
                    Some(m) => match m.get(k) {
                        Some(x) => x.clone(),
                        None => (Poly::int(0), Poly::int(0)),
                    },
                };
                r = Some(match r {
                    None => (l, h),
                    Some((l0, h0)) => match (self.pmin(&l0, &l), self.pmax(&h0, &h)) {
                        (Some(a), Some(b)) => (a, b),
                        _ => return Err("incomparable atom ranges".into()),
                    },
                });
            }
            ranges.insert(*k, r.ok_or("no atom range")?);
        }
        let is_wg = |k: &(bool, u64)| (!k.0 && k.1 >= u64::MAX - 2) || self.f.contract.wg_maps.iter().any(|m| m.outs.iter().any(|v| self.atom_key(*v) == *k));
        // 3. caps (`x < bound` on the live arm of a mask, or a clamp min(x, bound - 1)) present in every store with the same atoms (constants may
        // differ): when the offset holds k * x, x's atoms become ONE digit (coefficient k, values [min x, bound - 1]) and each store's constant
        // loses its k * (x constant) - provided x is injective in its atoms
        let mut digits: Vec<(Poly, Poly)> = Vec::new();
        let mut covered: std::collections::BTreeSet<(bool, u64)> = Default::default();
        let caps_all = self.group_caps.borrow().get(&ss[0].0).cloned().unwrap_or_default();
        let dbg = std::env::var("AMDC_GW_DEBUG").map_or(false, |v| !v.is_empty());
        if dbg {
            eprintln!("gw:   caps per store: {:?}", caps_all.iter().map(|c| c.len()).collect::<Vec<_>>());
        }
        if caps_all.len() == ss.len() {
            let mut used: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); ss.len()];
            for (ci, cap0) in caps_all[0].iter().enumerate() {
                let ((latoms, _), bound, clamp) = cap0;
                if latoms.is_empty() || !latoms.keys().all(|k| atoms.contains_key(k)) {
                    { if dbg { eprintln!("gw:   cap {ci} rejected at 1"); } continue; }
                }
                // the matching cap of every store (same atoms, bound and kind)
                let mut picks: Vec<usize> = vec![ci];
                for cs in caps_all.iter().skip(1) {
                    match cs.iter().position(|c| c.0 .0 == *latoms && c.1 == *bound && c.2 == *clamp) {
                        Some(p) => picks.push(p),
                        None => break,
                    }
                }
                if picks.len() != ss.len() {
                    { if dbg { eprintln!("gw:   cap {ci} rejected at 2"); } continue; }
                }
                // k = offset coefficient / x coefficient, the same for every atom of x (x coefficients positive constants)
                let mut kf: Option<Poly> = None;
                let mut ok = true;
                for (a, cl) in latoms {
                    let Some(q) = cl.as_const() else {
                        ok = false;
                        break;
                    };
                    if q.n <= 0 {
                        ok = false;
                        break;
                    }
                    let kk = atoms[a].scale(Q::new(q.d, q.n));
                    match &kf {
                        None => kf = Some(kk),
                        Some(k0) if *k0 == kk => {}
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                let Some(kf) = kf else { continue };
                if !ok || !cx.nonneg(&kf) {
                    { if dbg { eprintln!("gw:   cap {ci} rejected at 3"); } continue; }
                }
                // x injective: its atoms sorted by coefficient form a radix with unit size
                let mut xd: Vec<(Q, Poly)> = latoms.iter().map(|(a, cl)| (cl.as_const().unwrap(), ranges[a].1.sub(&ranges[a].0))).collect();
                xd.sort_by(|p, q| (p.0.n * q.0.d).cmp(&(q.0.n * p.0.d)));
                let mut acc = Poly::int(1);
                let mut inj = true;
                for (c, m) in &xd {
                    if !cx.le(&acc, &Poly::konst(*c)) {
                        inj = false;
                        break;
                    }
                    acc = acc.add(&m.scale(*c));
                }
                if !inj {
                    { if dbg { eprintln!("gw:   cap {ci} rejected at 4"); } continue; }
                }
                // a clamp loses the clamped lanes' x: sound when x's only workgroup digit u is its top digit and every x >= bound - 1 has u
                // at its maximum (c_u hi_u <= bound - 1): the clamped value then stays in that same workgroup's range
                if *clamp {
                    let wgs: Vec<&(bool, u64)> = latoms.keys().filter(|k| is_wg(k)).collect();
                    let top = latoms.iter().max_by(|p, q| {
                        let (a, b) = (p.1.as_const().unwrap(), q.1.as_const().unwrap());
                        (a.n * b.d).cmp(&(b.n * a.d))
                    });
                    let ok_clamp = match (wgs.as_slice(), top) {
                        ([w], Some((tk, tc))) if *w == tk => {
                            let c = tc.as_const().unwrap();
                            cx.le(&ranges[tk].1.scale(c), &bound.sub(&Poly::int(1))) || {
                                // a workgroup axis with grid = ceil(bound / c): c (grid - 1) <= bound - 1 always
                                !tk.0 && tk.1 >= u64::MAX - 2 && c.d == 1 && {
                                    let ax = (u64::MAX - tk.1) as u8;
                                    self.f.contract.grid_defs.iter().any(|(a2, e)| {
                                        *a2 == ax
                                            && matches!(e, Ext::CeilDiv(inner, kk) if *kk as i128 == c.n && match inner.as_ref() {
                                                Ext::Ka(off) => *bound == Poly::sym(Sym::Ka(*off)),
                                                Ext::Sym(n) => *bound == Poly::sym(Sym::Named(n.clone())),
                                                _ => false,
                                            })
                                    })
                                }
                            }
                        }
                        ([], _) => true,
                        _ => false,
                    };
                    if dbg && !ok_clamp {
                        let tops = top.map(|(k, c)| format!("{:x} c={} hi={}", k.1, c.show(), ranges[k].1.show()));
                        eprintln!("gw:   clamp: wg atoms {:?}, top {:?}, bound {}", wgs.iter().map(|k| format!("{:x}", k.1)).collect::<Vec<_>>(), tops, bound.show());
                    }
                    if !ok_clamp {
                        { if dbg { eprintln!("gw:   cap {ci} rejected at 5"); } continue; }
                    }
                }
                // x's minimum over the stores (their own constants) and the per-store constant correction
                let mut xmin: Option<Poly> = None;
                for (si, p) in picks.iter().enumerate() {
                    let lc = &caps_all[si][*p].0 .1;
                    let mut m = lc.clone();
                    for (a, cl) in latoms {
                        m = m.add(&ranges[a].0.scale(cl.as_const().unwrap()));
                    }
                    xmin = match xmin {
                        None => Some(m),
                        Some(x0) => match self.pmin(&x0, &m) {
                            Some(v) => Some(v),
                            None => {
                                ok = false;
                                Some(x0)
                            }
                        },
                    };
                }
                let Some(xmin) = xmin else { continue };
                let span = bound.sub(&Poly::int(1)).sub(&xmin);
                if !ok || !cx.nonneg(&span) {
                    { if dbg { eprintln!("gw:   cap {ci} rejected at 6"); } continue; }
                }
                let mut fixed = Vec::with_capacity(ss.len());
                for (si, p) in picks.iter().enumerate() {
                    let Some(kd) = kf.mul(&caps_all[si][*p].0 .1) else {
                        ok = false;
                        break;
                    };
                    fixed.push(consts[si].sub(&kd));
                }
                if !ok || picks.iter().enumerate().any(|(si, p)| used[si].contains(p)) {
                    { if dbg { eprintln!("gw:   cap {ci} rejected at 7"); } continue; }
                }
                for (si, p) in picks.iter().enumerate() {
                    used[si].insert(*p);
                }
                consts = fixed;
                for a in latoms.keys() {
                    atoms.remove(a);
                    covered.insert(*a);
                }
                digits.push((kf, span));
            }
        }
        // 4. the per-store constants: differences to the first, split by monomial -> one pseudo digit per monomial (gcd stride)
        let c0 = consts[0].clone();
        let mut by_mono: BTreeMap<Vec<Sym>, Vec<Q>> = BTreeMap::new();
        for c in &consts {
            for mono in c.sub(&c0).0.keys() {
                by_mono.entry(mono.clone()).or_default();
            }
        }
        for (mono, qs) in by_mono.iter_mut() {
            for c in &consts {
                qs.push(c.sub(&c0).0.get(mono).copied().unwrap_or(Q::int(0)));
            }
        }
        for (mono, qs) in &by_mono {
            if qs.iter().any(|q| q.d != 1) {
                return Err("a fractional store constant".into());
            }
            let ns: Vec<i128> = qs.iter().map(|q| q.n).collect();
            let lo = *ns.iter().min().unwrap();
            let hi = *ns.iter().max().unwrap();
            if hi > lo {
                let g = ns.iter().fold(0i128, |acc, d| gcd(acc, d - lo));
                let mut unit = std::collections::BTreeMap::new();
                unit.insert(mono.clone(), Q::int(g));
                let unit = Poly(unit);
                if !cx.nonneg(&unit) {
                    return Err("a store-constant stride of unknown sign".into());
                }
                digits.push((unit, Poly::int((hi - lo) / g)));
            }
        }
        // 5. the remaining atoms
        let mut wg_axes = [false; 3];
        for k in atoms.keys().chain(covered.iter()) {
            if !k.0 && k.1 >= u64::MAX - 2 {
                wg_axes[(u64::MAX - k.1) as usize] = true;
            }
            // a persistent-loop variable (starts at the workgroup id, steps by the grid size): distinct workgroups visit disjoint residue
            // classes, so the variable is a workgroup digit of that axis
            if k.0 {
                if let Some(ax) = self.persistent_axis(VarId(k.1 as u32)) {
                    wg_axes[ax as usize] = true;
                }
            }
        }
        for (k, c) in &atoms {
            let (l, h) = &ranges[k];
            let m = h.sub(l);
            if !cx.nonneg(&m) {
                if std::env::var("AMDC_GW_DEBUG").map_or(false, |v| !v.is_empty()) {
                    eprintln!("gw:   atom {:?}: range [{}, {}] (hi - lo not provably >= 0)", k, l.show(), h.show());
                }
                return Err("an atom range of unknown sign".into());
            }
            let c = if cx.nonneg(c) {
                c.clone()
            } else if cx.nonneg(&c.scale(Q::int(-1))) {
                c.scale(Q::int(-1))
            } else {
                return Err("a coefficient of unknown sign".into());
            };
            // an atom whose values are multiples of 2^z (known trailing zeros): the digit steps by c 2^z over (h - l) / 2^z steps
            let z = if k.0 { 0 } else { self.atom_inst.borrow().get(k).and_then(|v| v.first().cloned()).map_or(0, |o| self.tz(&o, 0)) };
            let (c, m) = match m.as_const() {
                Some(q) if z > 0 && q.d == 1 => (c.scale(Q::int(1i128 << z)), Poly::int(q.n >> z)),
                _ => (c, m),
            };
            digits.push((c, m));
        }
        // 6. every workgroup axis that can hold more than one workgroup must be a digit
        for (ax, seen) in wg_axes.iter().enumerate() {
            // an axis the contract leaves undefined and the kernel never reads is taken as one workgroup (a host contract like the grid
            // extents: docs/trust-inventory.md); an axis the kernel reads but the contract leaves open may hold several
            let defined = self.f.contract.grid_defs.iter().any(|(a, _)| *a as usize == ax);
            let read = self.defs.values().any(|op| matches!((ax, op), (0, Op::WorkgroupIdX) | (1, Op::WorkgroupIdY) | (2, Op::WorkgroupIdZ)));
            // ... or covered by a declared bijection whose outputs are all digits
            let mapped = self.f.contract.wg_maps.iter().any(|m| {
                m.axes.contains(&(ax as u8))
                    && m.outs.iter().all(|v| {
                        let k = self.atom_key(*v);
                        atoms.contains_key(&k) || covered.contains(&k)
                    })
            });
            // ... or every store of the group runs in a `wg_winner` branch over this axis (one workgroup per group)
            let winner = self.group_win.borrow().get(&ss[0].0).map_or(false, |ws| !ws.is_empty() && ws.iter().all(|w| w.contains(&(ax as u8))));
            if !seen && !mapped && !winner && (defined || read) && cx.sym_poly(Sym::Grid(ax as u8)).as_const() != Some(Q::int(1)) {
                return Err("a workgroup axis that is not a digit".into());
            }
        }
        // 7. sort by coefficient (every pair must be comparable) and check the mixed radix
        let n = digits.len();
        for i in 0..n {
            for j in i + 1..n {
                if cx.le(&digits[i].0, &digits[j].0) {
                    continue;
                }
                if cx.le(&digits[j].0, &digits[i].0) {
                    digits.swap(i, j);
                } else {
                    return Err(format!("incomparable coefficients {} vs {}", digits[i].0.show(), digits[j].0.show()));
                }
            }
        }
        let mut acc = Poly::int(size);
        for (c, m) in &digits {
            if m.0.is_empty() {
                continue; // a single-valued digit
            }
            if !cx.le(&acc, c) {
                return Err(format!("digits overlap (radix): {} > {}", acc.show(), c.show()));
            }
            let Some(cm) = c.mul(m) else { return Err("polynomial too large".into()) };
            acc = acc.add(&cm);
        }
        Ok(())
    }


    /// Under a tile marker, the access must lie inside the marked tile / ring (hull over the slot interval).
    fn tile_cover(&mut self, what: &str, addr: &Itv, size: i128) {
        let Some((ti, slot, slot_o)) = self.tag.clone() else {
            // untagged: must provably stay out of every private (per-wave) region
            for t in self.f.contract.tiles.iter().filter(|t| t.private) {
                let (lo, hi) = (t.base as i128, (t.base + t.slot_size * t.slots as i64) as i128);
                let below = addr.hi.iter().any(|h| self.cx.le(&h.add(&Poly::int(size)), &Poly::int(lo)));
                let above = addr.lo.iter().any(|l| self.cx.le(&Poly::int(hi), l));
                if !(below || above) {
                    let m = format!("stmt {}: {what}: LDS access {} (+{size}) outside any tile marker may touch the private per-wave region `{}` [{lo}, {hi}): an unmarked access cannot be proven to be the owning wave", self.idx, addr.show(), t.name);
                    self.rep.add("GV-TILE-PRIVATE-ESCAPE", Severity::Error, m);
                }
            }
            return;
        };
        let t = &self.f.contract.tiles[ti];
        if t.global {
            return; // an LDS access under a global-region marker is reported by the protocol engine (GV-TILE-KIND)
        }
        if t.private {
            // the access must lie in the slice of THIS marker's wave key: addr - key*size - base in [0, size - accsize]
            let (key_atoms, key_c) = self.lin(&slot_o).unwrap_or((std::collections::BTreeMap::new(), 0));
            let ok = match self.cur_addr.clone().and_then(|(o, imm)| self.lin(&o).map(|l| (l, imm))) {
                Some(((mut atoms, c), imm)) => {
                    for (k, v) in key_atoms {
                        *atoms.entry(k).or_insert(0) -= v * t.slot_size as i128;
                    }
                    atoms.retain(|_, v| *v != 0);
                    let mut r = Itv::konst(c + imm - key_c * t.slot_size as i128 - t.base as i128);
                    for ((is_var, id), coef) in atoms {
                        let o = if is_var { Opnd::Var(amdgpu_dsl::VarId(id)) } else { Opnd::Val(id) };
                        r = r.add(&self.int_of(&o).scale(Q::int(coef)));
                    }
                    let limit = Itv::konst(t.slot_size as i128);
                    match self.prove_range(&r, size, &limit) {
                        Ok(()) => true,
                        Err(e) => {
                            let m = format!("stmt {}: {what}: LDS access is not provably inside the private slice of wave key {slot_o:?} of `{}` (slice size {}): another wave's slice may be touched: {e}", self.idx, t.name, t.slot_size);
                            self.rep.add("GV-TILE-PRIVATE-ESCAPE", Severity::Error, m);
                            false
                        }
                    }
                }
                None => {
                    let m = format!("stmt {}: {what}: the address of an access to the private region `{}` is not a linear function of the wave key {slot_o:?} (cannot prove it stays in the own-wave slice)", self.idx, t.name);
                    self.rep.add("GV-TILE-PRIVATE-ESCAPE", Severity::Error, m);
                    false
                }
            };
            if ok {
                self.rep.ok("tile-cover");
            }
            return;
        }
        // a tile that REUSES private regions (tile_over): record, per reused region, whether every access of this marker either avoids the region
        // or lies in the slice of the wave key the region's own markers use (this wave's own, dead-for-itself storage) - gpu_protocol exempts such writes
        let reused: Vec<usize> = self
            .f
            .contract
            .tile_reuses
            .iter()
            .filter(|(n, _)| *n == t.name)
            .filter_map(|(_, o)| self.f.contract.tiles.iter().position(|x| &x.name == o && x.private))
            .collect();
        for pi in reused {
            let p = self.f.contract.tiles[pi].clone();
            let own = self.own_slice_of(&p, addr, size);
            let e = self.rep.own_reuse.entry((self.tag_stmt, p.name.clone())).or_insert(true);
            *e &= own;
        }
        let reg = slot.scale(Q::int(t.slot_size as i128)).add_const(t.base as i128);
        let lo_ok = reg.lo.iter().any(|r| addr.lo.iter().any(|a| self.cx.le(r, a)));
        let hi_ok = addr.hi.iter().any(|h| reg.hi.iter().any(|rh| self.cx.le(&h.add(&Poly::int(size)), &rh.add(&Poly::int(t.slot_size as i128)))));
        if lo_ok && hi_ok {
            self.rep.ok("tile-cover");
        } else {
            let m = format!("stmt {}: {what}: LDS access {} (+{size}) is not provably inside tile `{}` (slot interval {}, slot size {}, base {})", self.idx, addr.show(), t.name, slot.show(), t.slot_size, t.base);
            self.rep.add("GV-TILE-ESCAPE", Severity::Error, m);
        }
    }

    /// The current access (`cur_addr`, interval `addr`) avoids the private region `p` entirely, or lies inside the slice of the wave key its own markers
    /// use (one key operand for all of them; else false).
    fn own_slice_of(&self, p: &amdgpu_dsl::TileDecl, addr: &Itv, size: i128) -> bool {
        let (lo, hi) = (p.base as i128, (p.base + p.slot_size * p.slots as i64) as i128);
        let below = addr.hi.iter().any(|h| self.cx.le(&h.add(&Poly::int(size)), &Poly::int(lo)));
        let above = addr.lo.iter().any(|l| self.cx.le(&Poly::int(hi), l));
        if below || above {
            return true;
        }
        let mut keys: Vec<Opnd> = Vec::new();
        for s in self.f.stmts() {
            if let Stmt::TileProduce { tile, slot, .. } | Stmt::TileConsume { tile, slot, .. } = s {
                if *tile == p.name && !keys.contains(slot) {
                    keys.push(*slot);
                }
            }
        }
        let [key] = keys.as_slice() else { return false };
        let Some((key_atoms, key_c)) = self.lin(key) else { return false };
        let Some(((mut atoms, c), imm)) = self.cur_addr.clone().and_then(|(o, imm)| self.lin(&o).map(|l| (l, imm))) else { return false };
        for (k, v) in key_atoms {
            *atoms.entry(k).or_insert(0) -= v * p.slot_size as i128;
        }
        atoms.retain(|_, v| *v != 0);
        let mut r = Itv::konst(c + imm - key_c * p.slot_size as i128 - p.base as i128);
        for ((is_var, id), coef) in atoms {
            let o = if is_var { Opnd::Var(amdgpu_dsl::VarId(id)) } else { Opnd::Val(id) };
            r = r.add(&self.int_of(&o).scale(Q::int(coef)));
        }
        self.prove_range(&r, size, &Itv::konst(p.slot_size as i128)).is_ok()
    }

    /// Linear form over atoms (values / loop variables that are not +, constant*, << constant of others): atom coefficients and a constant.
    fn lin(&self, o: &Opnd) -> Option<(std::collections::BTreeMap<(bool, u32), i128>, i128)> {
        use std::collections::BTreeMap;
        fn go(s: &An, o: &Opnd, depth: usize) -> Option<(BTreeMap<(bool, u32), i128>, i128)> {
            if depth > 24 {
                return None;
            }
            match o {
                Opnd::I32(c) => Some((BTreeMap::new(), *c as i128)),
                Opnd::Var(v) => Some((BTreeMap::from([((true, v.0), 1)]), 0)),
                Opnd::Val(v) => match s.defs.get(v) {
                    Some(Op::ConstI32(c)) => Some((BTreeMap::new(), *c as i128)),
                    Some(Op::AddI32(a, b)) => {
                        let (mut x, c1) = go(s, a, depth + 1)?;
                        let (y, c2) = go(s, b, depth + 1)?;
                        for (k, v) in y {
                            *x.entry(k).or_insert(0) += v;
                        }
                        x.retain(|_, v| *v != 0);
                        Some((x, c1 + c2))
                    }
                    Some(Op::MulI32(a, b)) => {
                        let (x, c1) = go(s, a, depth + 1)?;
                        let (y, c2) = go(s, b, depth + 1)?;
                        let (m, (atoms, c), k) = if x.is_empty() { (c1, (y, c2), 0) } else if y.is_empty() { (c2, (x, c1), 0) } else { return Some((BTreeMap::from([((false, *v), 1)]), 0)) };
                        let _ = k;
                        Some((atoms.into_iter().map(|(a, v)| (a, v * m)).filter(|(_, v)| *v != 0).collect(), c * m))
                    }
                    Some(Op::ShlI32(a, b)) => {
                        let (x, c1) = go(s, a, depth + 1)?;
                        let (y, c2) = go(s, b, depth + 1)?;
                        if y.is_empty() && (0..31).contains(&c2) {
                            let m = 1i128 << c2;
                            Some((x.into_iter().map(|(a, v)| (a, v * m)).collect(), c1 * m))
                        } else {
                            Some((BTreeMap::from([((false, *v), 1)]), 0))
                        }
                    }
                    _ => Some((BTreeMap::from([((false, *v), 1)]), 0)),
                },
                Opnd::F32(_) => None,
            }
        }
        go(self, o, 0)
    }

    fn check_lds(&mut self, what: &str, addr: &Itv, size: i128) {
        // AMDC_GV_LDS_TRACE=1: every LDS access (statement, kind, address interval, size, address shape) for deriving a lift's tile map
        if std::env::var("AMDC_GV_LDS_TRACE").is_ok() && self.rep.quiet == 0 {
            let sh = self.cur_addr.map(|(o, k)| format!("{} + {k}", self.shape(&o, 6))).unwrap_or_default();
            eprintln!("LDS stmt {} | {what} | addr {} | size {size} | {sh}", self.idx, addr.show());
        }
        if self.tag.is_none() {
            self.rep.unchecked_access("lds-untracked");
        }
        self.note_owned(size);
        self.tile_cover(what, addr, size);
        self.cur_addr = None;
        let limit = Itv::konst(self.f.lds_bytes as i128);
        match self.prove_range(addr, size, &limit) {
            Ok(()) => self.rep.ok("lds"),
            Err(e) => {
                let m = format!("stmt {}: {what}: LDS access [addr, addr+{size}) not provably inside the declared {}-byte segment: {e}", self.idx, self.f.lds_bytes);
                self.rep.add("GV-LDS-OOB", Severity::Error, m)
            }
        }
    }

    fn check_global(&mut self, what: &str, base: &AV, imm: i128, size: i128, store: bool) {
        if let AV::Either(arms) = base {
            for x in arms.clone() {
                self.check_global(what, &x, imm, size, store);
            }
            return;
        }
        let (arg, off) = match base {
            AV::Ptr { arg: Some(a), off } => (*a, off.clone()),
            _ => {
                let m = format!("stmt {}: {what}: pointer provenance unknown (not derived from a kernarg pointer by +offsets): cannot bound the access", self.idx);
                self.rep.add("GV-GLB-PROV", Severity::Error, m);
                return;
            }
        };
        let Some(ext) = self.arg_extent(arg).cloned() else {
            let m = format!("stmt {}: {what}: kernarg pointer `{}` has no declared extent (arg_tensor / arg_tensor_slack)", self.idx, self.arg_name(arg));
            self.rep.add("GV-GLB-NOEXT", Severity::Error, m);
            return;
        };
        // under a global-region marker the access must also lie inside the declared region of THAT argument
        if let Some((ti, _, _)) = self.tag.clone() {
            let tname = self.f.contract.tiles[ti].name.clone();
            if self.f.contract.tiles[ti].global {
                if let Some(br) = self.f.contract.buf_regions.iter().find(|r| r.name == tname).cloned() {
                    if br.arg != self.arg_name(arg) {
                        let m = format!("stmt {}: {what}: the access goes through `{}` but the marked global region `{}` belongs to `{}`", self.idx, self.arg_name(arg), tname, br.arg);
                        self.rep.add("GV-TILE-ESCAPE", Severity::Error, m);
                    } else {
                        let (bi, si) = (ext_itv(&self.cx, &br.base), ext_itv(&self.cx, &br.size));
                        let a = off.add_const(imm);
                        let lo_ok = bi.hi.iter().any(|bh| a.lo.iter().any(|al| self.cx.le(bh, al)));
                        let hi_ok = a.hi.iter().any(|ah| bi.lo.iter().any(|bl| si.lo.iter().any(|sl| self.cx.le(&ah.add(&Poly::int(size)), &bl.add(sl)))));
                        if lo_ok && hi_ok {
                            self.rep.ok("region-cover");
                        } else {
                            let m = format!("stmt {}: {what}: access {} (+{size}) is not provably inside the global region `{}` [{}, +{}) of `{}`", self.idx, a.show(), tname, bi.show(), si.show(), br.arg);
                            self.rep.add("GV-TILE-ESCAPE", Severity::Error, m);
                        }
                    }
                }
            }
        }
        let lim_e = if store { ext.logical.clone().unwrap_or(ext.alloc.clone()) } else { ext.alloc.clone() };
        let limit = ext_itv(&self.cx, &lim_e);
        let addr = off.add_const(imm);
        match self.prove_range(&addr, size, &limit) {
            Ok(()) => self.rep.ok("global"),
            Err(e) => {
                let m = format!("stmt {}: {what}: {} of {size} bytes through `{}` not provably inside its {} extent: {e}", self.idx, if store { "store" } else { "load" }, self.arg_name(arg), if store && ext.logical.is_some() { "logical" } else { "allocated" });
                self.rep.add("GV-GLB-OOB", Severity::Error, m)
            }
        }
    }

    fn check_buffer(&mut self, what: &str, rsrc: &AV, voff: &Option<Opnd>, soff: &Opnd, imm: i128, size: i128, store: bool) {
        if let AV::Either(arms) = rsrc {
            for x in arms.clone() {
                self.check_buffer(what, &x, voff, soff, imm, size, store);
            }
            return;
        }
        let _ = store;
        let AV::Rsrc { arg, num, base: boff } = rsrc else {
            let m = format!("stmt {}: {what}: descriptor operand is not a known buffer descriptor", self.idx);
            self.rep.add("GV-GLB-PROV", Severity::Error, m);
            return;
        };
        if num.is_some() {
            // hardware range check against num_records (loads read 0, stores are dropped): accepted; overstatement is checked at the descriptor.
            // The range check sees the EFFECTIVE offset voffset + soffset + imm in 32 bits: a masking SENTINEL voffset (a negative constant chosen
            // by a select, e.g. -16 for "this lane is out of range") plus a positive instruction offset wraps back to a small in-range offset
            // (-16 + 16 = 0) and the access lands inside the buffer instead of being dropped. Reject (stores) / warn (loads) such sentinels.
            // (A register offset that is merely negative with a folded immediate - a legitimate address - is not a sentinel and is not flagged.)
            if imm > 0 {
                let soff_k = match soff {
                    Opnd::I32(c) => Some(*c as i128),
                    o => self.int_of(o).point_const().filter(|q| q.d == 1).map(|q| q.n),
                };
                let mut wraps = Vec::new();
                if let (Some(o), Some(sk)) = (voff, soff_k) {
                    for c in self.select_consts(o, 6) {
                        let e = c + sk + imm;
                        if c + sk < 0 && e >= 0 {
                            wraps.push((c, e));
                        }
                    }
                }
                if let Some((c, e)) = wraps.first() {
                    let m = format!(
                        "stmt {}: {what}: the masking sentinel voffset {c} plus the instruction offset {imm} is the 32-bit effective offset {e}: the \
                         masked lanes access the buffer at {e} instead of being dropped by the range check (use a sentinel that stays >= num_records after \
                         the immediate, e.g. 0x7fffff00, or fold the immediate into the register offset)",
                        self.idx
                    );
                    self.rep.add("GV-BUF-WRAP", if store { Severity::Error } else { Severity::Warn }, m);
                    return;
                }
            }
            self.rep.ok("buffer(hw-guarded)");
            return;
        }
        let v = match voff {
            Some(o) => self.int_of(o),
            None => Itv::konst(0),
        };
        let off = boff.add(&v.add(&self.int_of(soff)));
        let base = AV::Ptr { arg: *arg, off };
        self.check_global(what, &base, imm, size, store);
    }

    /// POSSIBLE BITS of an i32 value seen as a 32-bit pattern (a superset of the bits any execution can set; `u32::MAX` = unknown). Structural on the
    /// value's definition (constants, and / xor / shifts / add / mul / select / min) intersected with the interval bound (a non-negative value <= h has
    /// no bit above h's). Sound for wrap-around: shifts truncate, sums and products that may exceed 32 bits give `u32::MAX`.
    fn bits(&self, o: &Opnd, depth: usize) -> u32 {
        let all = u32::MAX;
        let from_itv = |i: &Itv| -> u32 { amdgpu_sym::bits::itv_bits32(i, &self.cx) };
        let itv_m = from_itv(&self.int_of(o));
        let v = match o {
            Opnd::I32(c) => return *c as u32,
            Opnd::Val(v) => *v,
            _ => return itv_m,
        };
        if depth == 0 || self.assumed_vals.contains(&v) {
            return itv_m;
        }
        let Some(op) = self.defs.get(&v) else { return itv_m };
        let k = |o: &Opnd| -> Option<u32> {
            match o {
                Opnd::I32(c) => Some(*c as u32),
                Opnd::Val(x) => match self.defs.get(x) {
                    Some(Op::ConstI32(c)) => Some(*c as u32),
                    _ => None,
                },
                _ => None,
            }
        };
        let st = match op {
            Op::ConstI32(c) => *c as u32,
            Op::AndI32(a, b) => self.bits(a, depth - 1) & self.bits(b, depth - 1),
            Op::XorI32(a, b) => self.bits(a, depth - 1) | self.bits(b, depth - 1),
            Op::ShlI32(a, b) => match k(b) {
                Some(s) if s < 32 => self.bits(a, depth - 1).wrapping_shl(s),
                _ => all,
            },
            Op::LshrI32(a, b) => match k(b) {
                Some(s) if s < 32 => self.bits(a, depth - 1) >> s,
                _ => all,
            },
            Op::AddI32(a, b) => amdgpu_sym::bits::add32(self.bits(a, depth - 1), self.bits(b, depth - 1)),
            Op::MulI32(a, b) => amdgpu_sym::bits::mul32(self.bits(a, depth - 1), self.bits(b, depth - 1)),
            Op::Select { a, b, .. } | Op::MinI32(a, b) => self.bits(a, depth - 1) | self.bits(b, depth - 1),
            _ => all,
        };
        st & itv_m
    }

    /// Narrows a bitwise result by its possible bits: a pattern with bit 31 clear and no bit above `m` is in `[0, m]`.
    fn narrow_bits(&self, r: AV, m: u32) -> AV {
        match r {
            AV::Int(mut i) if m >> 31 == 0 => {
                i.lo.push(Poly::int(0));
                i.hi.push(Poly::int(m as i128));
                AV::Int(i)
            }
            other => other,
        }
    }

    /// Re-evaluates an integer value under the current override (used for the arms of a `Select` whose condition refines a variable).
    fn recompute(&mut self, o: &Opnd, depth: usize) -> Itv {
        if depth == 0 {
            return self.int_of(o);
        }
        // the refined operand of a select may be a loop variable (a select-cycled ring slot: `idx < 2 ? idx + 1 : 0`)
        if let Some((t, i)) = &self.ovr {
            if t == o {
                return i.clone();
            }
        }
        let Opnd::Val(v) = o else { return self.int_of(o) };
        if self.assumed_vals.contains(v) {
            return self.int_of(o);
        }
        let Some(op) = self.defs.get(v).cloned() else { return self.int_of(o) };
        if !is_int_ty(op.result_ty()) || matches!(op, Op::Asm { .. }) {
            return self.int_of(o);
        }
        let mut saved: Vec<(ValueId, Option<AV>)> = Vec::new();
        let mut saved_vars: Vec<(VarId, Option<Itv>)> = Vec::new();
        for ch in op.operands() {
            if let Opnd::Val(c) = ch {
                if self.defs.get(&c).map_or(false, |d| is_int_ty(d.result_ty())) {
                    let ci = self.recompute(&ch, depth - 1);
                    saved.push((c, self.env.insert(c, AV::Int(ci))));
                }
            }
            if let Opnd::Var(x) = ch {
                if let Some((t, i)) = self.ovr.clone() {
                    if t == ch {
                        saved_vars.push((x, self.vars.insert(x, i)));
                    }
                }
            }
        }
        let r = match self.eval(&op) {
            AV::Int(i) => i,
            _ => Itv::top(),
        };
        for (c, old) in saved.into_iter().rev() {
            match old {
                Some(a) => {
                    self.env.insert(c, a);
                }
                None => {
                    self.env.remove(&c);
                }
            }
        }
        for (x, old) in saved_vars.into_iter().rev() {
            match old {
                Some(a) => {
                    self.vars.insert(x, a);
                }
                None => {
                    self.vars.remove(&x);
                }
            }
        }
        r
    }

    /// `Select(cond, a, b)` where `cond` compares a value/variable `x` with a bound: evaluate the arms under the refined `x`.
    fn eval_select(&mut self, ty: amdgpu_dsl::DType, cond: &Opnd, a: &Opnd, b: &Opnd) -> AV {
        if !is_int_ty(ty) {
            let (va, vb) = (self.av_of(a), self.av_of(b));
            if let (AV::Ptr { arg: x, off: ox }, AV::Ptr { arg: y, off: oy }) = (&va, &vb) {
                if x == y {
                    return AV::Ptr { arg: *x, off: ox.join(oy, &self.cx) };
                }
            }
            // pointers / descriptors of different provenance: keep both arms (flattened)
            if matches!(ty, amdgpu_dsl::DType::Ptr | amdgpu_dsl::DType::Rsrc) {
                let mut arms = Vec::new();
                for v in [va, vb] {
                    match v {
                        AV::Either(xs) => arms.extend(xs),
                        AV::Ptr { .. } | AV::Rsrc { .. } => arms.push(v),
                        _ => return AV::Other,
                    }
                }
                return AV::Either(arms);
            }
            return AV::Other;
        }
        let refine = match cond {
            Opnd::Val(c) => match self.defs.get(c) {
                Some(Op::CmpLtI32(x, k)) => {
                    // `c < v` with a constant on the left refines `v` (mode 2: v >= c + 1 / v <= c); otherwise `x < k` refines x (mode 0)
                    // a literal constant on the left (its interval may carry redundant symbolic bounds, so ask its definition, not
                    // the interval) refines the right operand: `0 < idx ? idx - 1 : 2` (the select-cycled ring slot)
                    let lit = |o: &Opnd| match o {
                        Opnd::I32(c) => Some(*c as i128),
                        Opnd::Val(v) => match self.defs.get(v) {
                            Some(Op::ConstI32(c)) => Some(*c as i128),
                            _ => None,
                        },
                        _ => None,
                    };
                    let xi = match lit(x) {
                        Some(c) => Itv::konst(c),
                        None => self.int_of(x),
                    };
                    let x_const = lit(x).is_some() || xi.point_const().is_some();
                    let k_const = lit(k).is_some() || self.int_of(k).point_const().is_some();
                    if x_const && !k_const { Some((*k, xi, 2u8)) } else { Some((*x, self.int_of(k), 0u8)) }
                }
                Some(Op::CmpGeI32(x, k)) => Some((*x, self.int_of(k), 1u8)),
                _ => None,
            },
            _ => None,
        };
        let (ia, ib) = if let Some((x, k, mode)) = refine {
            let xi = self.int_of(&x);
            // x < k : x <= k.hi - 1 ; x >= k : x >= k.lo
            let below = Itv { lo: Vec::new(), hi: k.hi.iter().map(|p| p.add(&Poly::int(-1))).collect() };
            let above = Itv { lo: k.lo.clone(), hi: Vec::new() };
            let (t_itv, f_itv) = match mode {
                0 => (xi.meet(&below), xi.meet(&above)),
                1 => (xi.meet(&above), xi.meet(&below)),
                _ => {
                    // k is the constant c: true  => x >= c + 1 ; false => x <= c
                    let ge = Itv { lo: k.lo.iter().map(|p| p.add(&Poly::int(1))).collect(), hi: Vec::new() };
                    let le = Itv { lo: Vec::new(), hi: k.hi.clone() };
                    (xi.meet(&ge), xi.meet(&le))
                }
            };
            self.ovr = Some((x, t_itv));
            let ia = self.recompute(a, 3);
            self.ovr = Some((x, f_itv));
            let ib = self.recompute(b, 3);
            self.ovr = None;
            (ia, ib)
        } else {
            (self.int_of(a), self.int_of(b))
        };
        AV::Int(ia.join(&ib, &self.cx).norm(&self.cx))
    }

    fn eval(&mut self, op: &Op) -> AV {
        if let Op::Select { ty, cond, a, b } = op {
            return self.eval_select(*ty, cond, a, b);
        }
        // bitwise ops: the interval rules below, narrowed by the possible bits of the operands (xor / and of bounded patterns, masks of any sign)
        if let Op::XorI32(a, b) | Op::AndI32(a, b) = op {
            let (ma, mb) = (self.bits(a, 6), self.bits(b, 6));
            let m = if matches!(op, Op::XorI32(..)) { ma | mb } else { ma & mb };
            let r = self.eval_core(op);
            return self.narrow_bits(r, m);
        }
        self.eval_core(op)
    }

    fn eval_core(&mut self, op: &Op) -> AV {
        let cx = &self.cx;
        match op {
            Op::ConstI32(c) => AV::Int(Itv::konst(*c as i128)),
            Op::KernargI32 { offset } => AV::Int(Itv::point(cx.sym_poly(Sym::Ka(*offset)))),
            Op::KernargPtr { offset } => {
                let arg = self.f.args.iter().position(|a| a.kernarg_offset == *offset);
                AV::Ptr { arg, off: Itv::konst(0) }
            }
            // the kernarg segment pointer: no tensor provenance (an access through it is GV-GLB-PROV)
            Op::KernargSegment => AV::Ptr { arg: None, off: Itv::konst(0) },
            Op::WorkitemId => AV::Int(Itv::range(Poly::int(0), Poly::int(self.f.workgroup_size as i128 - 1))),
            Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ => {
                let axis = match op {
                    Op::WorkgroupIdX => 0u8,
                    Op::WorkgroupIdY => 1,
                    _ => 2,
                };
                AV::Int(Itv::range(Poly::int(0), cx.sym_poly(Sym::Grid(axis)).sub(&Poly::int(1))))
            }
            // the next value of a ring (or of its shadow) computed ahead of its set_var: `D + var` with D a step of that ring lies in the
            // ring's (shadow's) proven range, like the variable itself
            Op::AddI32(Opnd::Val(d), Opnd::Var(w)) | Op::AddI32(Opnd::Var(w), Opnd::Val(d)) if self.ring_step_add(*d, *w) =>
            {
                AV::Int(self.int_of(&Opnd::Var(*w)))
            }
            Op::AddI32(a, b) => AV::Int(self.int_of(a).add(&self.int_of(b))),
            Op::MulI32(a, b) => AV::Int(self.int_of(a).mul(&self.int_of(b), cx)),
            Op::ShlI32(a, b) => match self.int_of(b).point_const() {
                Some(k) if k.d == 1 && (0..31).contains(&k.n) => AV::Int(self.int_of(a).scale(Q::int(1i128 << k.n))),
                _ => AV::Int(Itv::top()),
            },
            Op::LshrI32(a, b) => match self.int_of(b).point_const() {
                Some(k) if k.d == 1 && (0..31).contains(&k.n) => {
                    let x = self.int_of(a);
                    if !x.nonneg(cx) {
                        return AV::Int(Itv::top());
                    }
                    let div = Q::new(1, 1i128 << k.n);
                    let mut lo: Vec<Poly> = vec![Poly::int(0)];
                    // (values are integers: a constant lower bound rounds up)
                    lo.extend(x.lo.iter().map(|p| {
                        let q = p.add(&Poly::int(-((1i128 << k.n) - 1))).scale(div);
                        match q.as_const() {
                            Some(c) => Poly::int((c.n + c.d - 1).div_euclid(c.d)),
                            None => q,
                        }
                    }));
                    AV::Int(Itv { lo: cap(lo), hi: cap(x.hi.iter().map(|p| { let q = p.scale(div); match q.floor_const() { Some(k) => Poly::int(k), None => q } }).collect()) })
                }
                _ => AV::Int(Itv::top()),
            },
            Op::AndI32(a, b) => {
                let (x, y) = (self.int_of(a), self.int_of(b));
                let mut hi: Vec<Poly> = Vec::new();
                let mut ok = false;
                for (p, q) in [(&x, &y), (&y, &x)] {
                    if let Some(m) = p.point_const() {
                        if m.d == 1 && m.n >= 0 {
                            hi.push(Poly::int(m.n));
                            ok = true;
                            if q.nonneg(cx) {
                                hi.extend(q.hi.iter().cloned());
                            }
                        }
                    }
                }
                if !ok && x.nonneg(cx) && y.nonneg(cx) {
                    hi.extend(x.hi.iter().cloned());
                    hi.extend(y.hi.iter().cloned());
                    ok = true;
                }
                if ok {
                    AV::Int(Itv { lo: vec![Poly::int(0)], hi: cap(hi) })
                } else {
                    AV::Int(Itv::top())
                }
            }
            Op::XorI32(a, b) => {
                let (x, y) = (self.int_of(a), self.int_of(b));
                let num = |i: &Itv| -> Option<i128> {
                    if !i.nonneg(cx) || i.hi.is_empty() {
                        return None;
                    }
                    i.hi.iter().filter_map(|p| p.floor_const()).min()
                };
                // x ^ c for a constant c (any sign): the bits above the width of x are those of c, the low bits are free
                let with_const = |p: &Itv, q: &Itv| -> Option<Itv> {
                    let c = p.point_const()?;
                    if c.d != 1 {
                        return None;
                    }
                    let m = num(q)?;
                    let mut p2 = 1i128;
                    while p2 <= m {
                        p2 <<= 1;
                        if p2 > (1i128 << 100) {
                            return None; // an unbounded interval (long add / xor chains): no mask bound, never loop on i128 overflow
                        }
                    }
                    let c_hi = c.n - c.n.rem_euclid(p2);
                    Some(Itv::range(Poly::int(c_hi), Poly::int(c_hi + p2 - 1)))
                };
                if let Some(r) = with_const(&y, &x).or_else(|| with_const(&x, &y)) {
                    return AV::Int(r);
                }
                match (num(&x), num(&y)) {
                    (Some(m1), Some(m2)) => {
                        let m = m1.max(m2);
                        let mut p2 = 1i128;
                        while p2 <= m {
                            p2 <<= 1;
                            if p2 > (1i128 << 100) {
                                return AV::Int(Itv::top()); // see above: the shift would overflow i128 and never terminate
                            }
                        }
                        AV::Int(Itv::range(Poly::int(0), Poly::int(p2 - 1)))
                    }
                    _ => AV::Int(Itv::top()),
                }
            }
            Op::MinI32(a, b) => AV::Int(self.int_of(a).min(&self.int_of(b), cx)),
            // P1 integer ops (docs/op-coverage-tileops.md): the address-relevant ones get interval rules (no wrap-around: assumption A3)
            Op::Bin(o, a, b) => {
                use amdgpu_dsl::BinOp as B;
                let (x, y) = (self.int_of(a), self.int_of(b));
                let hi_const = |i: &Itv| i.hi.iter().filter_map(|p| p.floor_const()).min();
                match o {
                    B::MinU32 if x.nonneg(cx) && y.nonneg(cx) => AV::Int(x.min(&y, cx)),
                    B::MaxI32 | B::MaxU32 if !matches!(o, B::MaxU32) || (x.nonneg(cx) && y.nonneg(cx)) => {
                        let mut lo = x.lo.clone();
                        lo.extend(y.lo.iter().cloned());
                        let hi = match (hi_const(&x), hi_const(&y)) {
                            (Some(p), Some(q)) => vec![Poly::int(p.max(q))],
                            _ => vec![],
                        };
                        AV::Int(Itv { lo: cap(lo), hi })
                    }
                    B::SubU32 => AV::Int(x.add(&y.scale(Q::int(-1)))),
                    B::MulU32U24 if x.nonneg(cx) && y.nonneg(cx) && hi_const(&x).map_or(false, |h| h < 1 << 24) && hi_const(&y).map_or(false, |h| h < 1 << 24) => AV::Int(x.mul(&y, cx)),
                    // v_mul_i32_i24: exact product of two operands inside the signed 24-bit range
                    B::MulI32I24 if {
                        let lo_const = |i: &Itv| i.lo.iter().filter_map(|p| p.floor_const()).max();
                        // (symbolic bounds too: kernarg / grid symbols within their contract ranges)
                        let i24 = |i: &Itv| (lo_const(i).map_or(false, |l| l >= -(1 << 23)) && hi_const(i).map_or(false, |h| h < 1 << 23)) || i.within(&Itv::range(Poly::int(-(1 << 23)), Poly::int((1 << 23) - 1)), cx);
                        i24(&x) && i24(&y)
                    } => AV::Int(x.mul(&y, cx)),
                    // v_mbcnt_lo / hi: the accumulator plus the count of set mask bits of 32 lanes
                    B::MbcntLo | B::MbcntHi => AV::Int(y.add(&Itv::range(Poly::int(0), Poly::int(32)))),
                    // or of two non-negative bounded values: below the next power of two of the larger bound
                    B::OrB32 if x.nonneg(cx) && y.nonneg(cx) => match (hi_const(&x), hi_const(&y)) {
                        (Some(p), Some(q)) if p.max(q) < 1 << 40 => {
                            let m = p.max(q);
                            let mut p2 = 1i128;
                            while p2 <= m {
                                p2 <<= 1;
                            }
                            AV::Int(Itv::range(Poly::int(0), Poly::int(p2 - 1)))
                        }
                        _ => AV::Int(Itv::top()),
                    },
                    _ => AV::Int(Itv::top()),
                }
            }
            Op::Tri(o, a, b, c) => {
                use amdgpu_dsl::TriOp as T;
                let (x, y, z) = (self.int_of(a), self.int_of(b), self.int_of(c));
                let hi_const = |i: &Itv| i.hi.iter().filter_map(|p| p.floor_const()).min();
                let small = |i: &Itv| i.nonneg(cx) && hi_const(i).map_or(false, |h| h < 1 << 24);
                match o {
                    T::Add3U32 => AV::Int(x.add(&y).add(&z)),
                    T::LshlAddU32 => match y.point_const() {
                        Some(k) if k.d == 1 && (0..31).contains(&k.n) => AV::Int(x.scale(Q::int(1i128 << k.n)).add(&z)),
                        _ => AV::Int(Itv::top()),
                    },
                    T::MadU32U24 if small(&x) && small(&y) => AV::Int(x.mul(&y, cx).add(&z)),
                    T::Min3I32 => AV::Int(x.min(&y, cx).min(&z, cx)),
                    T::Min3U32 if x.nonneg(cx) && y.nonneg(cx) && z.nonneg(cx) => AV::Int(x.min(&y, cx).min(&z, cx)),
                    T::BfeU32 => match z.point_const() {
                        Some(w) if w.d == 1 && (1..32).contains(&w.n) => AV::Int(Itv::range(Poly::int(0), Poly::int((1i128 << w.n) - 1))),
                        _ => AV::Int(Itv::top()),
                    },
                    _ => AV::Int(Itv::top()),
                }
            }
            Op::ReadFirstLane(a) => AV::Int(self.int_of(a)),
            // (s) descriptor / pointer words: a pointer re-assembled from its own two words is that pointer; a descriptor assembled as
            // [lo(p), hi(p) & 0xffff (| 0), num_records, 0x00020000] is BufferRsrc(p, num_records); anything else has unknown provenance
            Op::PtrCat { lo, hi } => match (self.word_of(lo), self.word_of(hi)) {
                (Some((pl, false)), Some((ph, true))) if self.same_ptr(&pl, &ph) => self.av_of(&pl),
                // a hi word with its top 16 bits cleared: the same pointer (a canonical 48-bit address has them clear)
                (Some((pl, false)), None) if self.masked_hi_of(hi).map_or(false, |ph| self.same_ptr(&pl, &ph)) => self.av_of(&pl),
                _ => AV::Ptr { arg: None, off: Itv::top() },
            },
            Op::RsrcCat { words } => match self.rsrc_words(words) {
                Ok((p, nr)) => match self.av_of(&p) {
                    AV::Ptr { arg, off } => AV::Rsrc { arg, num: Some(self.int_of(&nr)), base: off },
                    _ => AV::Rsrc { arg: None, num: None, base: Itv::top() },
                },
                Err(why) => {
                    let m = format!("stmt {}: descriptor words not recognised ({why})", self.idx);
                    self.rep.add("GV-GLB-PROV", Severity::Error, m);
                    AV::Rsrc { arg: None, num: None, base: Itv::top() }
                }
            },
            Op::MovV { a, .. } | Op::MovS { a, .. } => self.av_of(a),
            Op::AddK { a, k } => AV::Int(self.int_of(a).add(&Itv::konst(*k as i128))),
            // another lane's value of the same expression: the per-lane interval (lane-independent bounds) still holds
            Op::DppQuadPerm { x, .. } => AV::Int(self.int_of(x)),
            // a clock reading: any 32-bit value
            Op::MemTimeLo { .. } => AV::Int(Itv::top()),
            Op::PtrAdd { base, byte_off } => match self.av_of(base) {
                AV::Ptr { arg, off } => AV::Ptr { arg, off: off.add(&self.int_of(byte_off)) },
                AV::Either(arms) => {
                    let o = self.int_of(byte_off);
                    AV::Either(arms.into_iter().map(|x| match x { AV::Ptr { arg, off } => AV::Ptr { arg, off: off.add(&o) }, _ => AV::Ptr { arg: None, off: Itv::top() } }).collect())
                }
                _ => AV::Ptr { arg: None, off: Itv::top() },
            },
            // base + (u64) a * b: exact when both factors are provably non-negative (then u32 = the value); otherwise the offset is unknown
            Op::PtrMadU32 { base, a, b } => {
                let (ia, ib) = (self.int_of(a), self.int_of(b));
                let ok = ia.nonneg(&self.cx) && ib.nonneg(&self.cx);
                let prod = if ok { Some(ia.mul(&ib, &self.cx)) } else { None };
                match (self.av_of(base), prod) {
                    (AV::Ptr { arg, off }, Some(p)) => AV::Ptr { arg, off: off.add(&p) },
                    _ => AV::Ptr { arg: None, off: Itv::top() },
                }
            }
            Op::LoadU16 { .. } => AV::Int(Itv::range(Poly::int(0), Poly::int(65535))),
            // P1 narrow loads: their value ranges (indices / scale bytes read from memory are still bounded by the width)
            Op::LoadGlobalN { ty, .. } | Op::BufferLoadN { ty, .. } | Op::LoadLdsN { ty, .. } => match ty {
                amdgpu_dsl::MemTy::U8 => AV::Int(Itv::range(Poly::int(0), Poly::int(255))),
                amdgpu_dsl::MemTy::I8 => AV::Int(Itv::range(Poly::int(-128), Poly::int(127))),
                amdgpu_dsl::MemTy::U16 => AV::Int(Itv::range(Poly::int(0), Poly::int(65535))),
                amdgpu_dsl::MemTy::I16 => AV::Int(Itv::range(Poly::int(-32768), Poly::int(32767))),
                amdgpu_dsl::MemTy::B32 => AV::Int(Itv::top()),
                _ => AV::Other,
            },
            Op::BufferRsrc { base, num_records } => {
                let (arg, boff) = match self.av_of(base) {
                    AV::Ptr { arg, off } => (arg, off),
                    _ => (None, Itv::konst(0)),
                };
                let num = if *num_records == 0xFFFF_FFFF { None } else { Some(Itv::konst(*num_records as i128)) };
                AV::Rsrc { arg, num, base: boff }
            }
            Op::BufferRsrcDyn { base, num_records } => {
                let (arg, boff) = match self.av_of(base) {
                    AV::Ptr { arg, off } => (arg, off),
                    _ => (None, Itv::konst(0)),
                };
                AV::Rsrc { arg, num: Some(self.int_of(num_records)), base: boff }
            }
            Op::Asm { .. } => {
                self.has_asm = true;
                AV::Other
            }
            other => {
                if is_int_ty(other.result_ty()) {
                    AV::Int(Itv::top())
                } else {
                    AV::Other
                }
            }
        }
    }

    fn global_load_size(op: &Op) -> Option<i128> {
        Some(match op {
            Op::LoadU16 { .. } => 2,
            Op::LoadF32 { .. } | Op::LoadF32Coherent { .. } => 4,
            Op::LoadGlobalB128 { .. } => 16,
            _ => return None,
        })
    }

    fn check_op(&mut self, op: &Op) {
        if let Some(sz) = Self::global_load_size(op) {
            let (base, off) = match op {
                Op::LoadU16 { base, byte_off } | Op::LoadF32 { base, byte_off } | Op::LoadGlobalB128 { base, byte_off } | Op::LoadF32Coherent { base, byte_off } => (base, *byte_off),
                _ => return,
            };
            let b = self.av_of(base);
            self.check_global("global load", &b, off as i128, sz, false);
            return;
        }
        match op {
            Op::LoadScalar { base, byte_off } => {
                let b = self.av_of(base);
                self.check_global("s_load_dword", &b, *byte_off as i128, 4, false);
            }
            Op::AtomicAddU32 { base, byte_off, .. } => {
                let b = self.av_of(base);
                self.check_global("global atomic add", &b, *byte_off as i128, 4, true);
            }
            Op::LoadLdsB32 { addr } => self.check_lds_at("ds_read_b32", addr, 4),
            Op::LoadLdsB128 { addr } => self.check_lds_at("ds_read_b128", addr, 16),
            Op::LoadLdsB64 { addr } => self.check_lds_at("ds_read_b64", addr, 8),
            Op::LoadLdsTrB8 { addr } => self.check_lds_at("ds_read_b64_tr_b8", addr, 8),
            Op::LoadLdsTr4B8 { addrs } => {
                for ad in addrs.iter() {
                    self.check_lds_at("ds_read_b64_tr_b8 (x4)", ad, 8);
                }
            }
            Op::LoadLdsB128x2 { lo, hi } => {
                for ad in [lo, hi] {
                    self.check_lds_at("ds_read_b128 (x2)", ad, 16);
                }
            }
            Op::BufferLoad { rsrc, voff, soff, imm, dwords } => {
                let r = self.av_of(rsrc);
                self.check_buffer("buffer load", &r, voff, soff, *imm as i128, 4 * *dwords as i128, false);
            }
            Op::BufferLoadB128x2 { rsrc, voff, soff, imm_lo, imm_hi } => {
                let r = self.av_of(rsrc);
                for imm in [*imm_lo, *imm_hi] {
                    self.check_buffer("buffer load (b128x2)", &r, voff, soff, imm as i128, 16, false);
                }
            }
            Op::BufferRsrc { .. } | Op::BufferRsrcDyn { .. } => {}
            // P1 memory loads
            Op::LoadGlobalN { base, byte_off, ty, .. } => {
                let b = self.av_of(base);
                self.check_global("global load", &b, *byte_off as i128, ty.bytes() as i128, false);
            }
            Op::LoadGlobalD16 { base, byte_off, ty, .. } => {
                let b = self.av_of(base);
                self.check_global("global load d16", &b, *byte_off as i128, ty.bytes() as i128, false);
            }
            Op::BufferLoadN { rsrc, voff, soff, imm, ty, .. } => {
                let r = self.av_of(rsrc);
                self.check_buffer("buffer load", &r, voff, soff, *imm as i128, ty.bytes() as i128, false);
            }
            Op::LoadLdsN { addr, ty } => self.check_lds_at("ds_read", addr, ty.bytes() as i128),
            Op::LoadLdsD16 { addr, ty, .. } => self.check_lds_at("ds_read d16", addr, ty.bytes() as i128),
            Op::LoadLdsTrB16 { addr } => self.check_lds_at("ds_read_b64_tr_b16", addr, 8),
            Op::AtomicLds { addr, .. } => self.check_lds_at("ds atomic (returning)", addr, 4),
            Op::BufferAtomicRet { rsrc, voff, soff, imm, .. } => {
                let r = self.av_of(rsrc);
                self.check_buffer("buffer atomic (returning)", &r, voff, soff, *imm as i128, 4, true);
            }
            Op::AtomicGlobal { base, byte_off, .. } => {
                let b = self.av_of(base);
                self.check_global("global atomic (returning)", &b, *byte_off as i128, 4, true);
            }
            Op::LoadLds2 { addr, off0, off1, b64, st64 } => self.check_lds2("ds_read2", addr, *off0, *off1, *b64, *st64),
            _ => {}
        }
    }

    /// Both elements of a two-address DS access (`addr + off * stride`, stride = element size, x 64 for st64).
    fn check_lds2(&mut self, what: &str, addr: &Opnd, off0: u8, off1: u8, b64: bool, st64: bool) {
        let eb: i128 = if b64 { 8 } else { 4 };
        let stride = eb * if st64 { 64 } else { 1 };
        for o in [off0, off1] {
            self.cur_addr = Some((*addr, o as i128 * stride));
            let a = self.int_of(addr).add_const(o as i128 * stride);
            self.check_lds(what, &a, eb);
        }
    }

    /// The constants an operand may take through selects (the arms of `Select`s, recursively, and constants): masking sentinels.
    fn select_consts(&self, o: &Opnd, depth: usize) -> Vec<i128> {
        match o {
            Opnd::I32(c) => vec![*c as i128],
            Opnd::Val(v) if depth > 0 => match self.defs.get(v) {
                Some(Op::ConstI32(c)) => vec![*c as i128],
                // a sentinel shifted by a constant (sentinel + k) is still a sentinel
                Some(Op::AddI32(x, Opnd::I32(k))) | Some(Op::AddI32(Opnd::I32(k), x)) => self.select_consts(x, depth - 1).into_iter().map(|c| c + *k as i128).collect(),
                Some(Op::Select { a, b, .. }) => {
                    let mut r = self.select_consts(a, depth - 1);
                    r.extend(self.select_consts(b, depth - 1));
                    r
                }
                _ => Vec::new(),
            },
            _ => Vec::new(),
        }
    }

    /// A descriptor must not overstate the allocation it was built from.
    fn check_descriptor(&mut self, dst: &AV) {
        if let AV::Rsrc { arg: Some(a), num: Some(n), base } = dst {
            let Some(ext) = self.arg_extent(*a).cloned() else {
                let m = format!("stmt {}: buffer descriptor over `{}` has no declared extent to check num_records against", self.idx, self.arg_name(*a));
                self.rep.add("GV-GLB-NOEXT", Severity::Error, m);
                return;
            };
            let limit = ext_itv(&self.cx, &ext.alloc);
            // a moved base (`ptr_mad`, > 4 GiB tensors): base >= 0 and base + num_records <= the allocation
            let end = base.add(n);
            let ok = base.nonneg(&self.cx) && end.hi.iter().any(|h| limit.lo.iter().any(|l| self.cx.le(h, l)));
            if ok {
                self.rep.ok("descriptor");
            } else {
                let m = format!("stmt {}: buffer descriptor over `{}` declares num_records {} which is not provably <= the allocation {} (an overstated descriptor turns out-of-bounds reads into wrong data instead of zeros)", self.idx, self.arg_name(*a), n.show(), limit.show());
                self.rep.add("GV-BUF-OVER", Severity::Error, m);
            }
        } else if let AV::Rsrc { arg: None, .. } = dst {
            let m = format!("stmt {}: buffer descriptor not built from a plain kernarg pointer (provenance unknown)", self.idx);
            self.rep.add("GV-GLB-PROV", Severity::Error, m);
        }
    }

    fn assume(&mut self, target: &Opnd, lo: &Ext, hi: &Ext, trusted: &Option<String>) {
        let a = Itv { lo: ext_itv(&self.cx, lo).lo, hi: ext_itv(&self.cx, hi).hi };
        match target {
            Opnd::Var(v) => {
                let init_ok = match self.var_init.get(v) {
                    Some(i) => i.within(&a, &self.cx),
                    None => false,
                };
                self.assumed.insert(*v, a.clone());
                self.vars.insert(*v, a.clone());
                if let Some(r) = trusted {
                    self.assumed_trusted.insert(*v, r.clone());
                    // a trusted loop-variable range the analysis proves anyway (initial value inside, every set_var inside - checked at the
                    // set_vars) costs no trust: only an unproven initial value or step is counted (TRUSTED)
                    if init_ok {
                        let m = format!("stmt {}: assume_trusted var{} in {} ({r}): initial value proven inside; counted only if a step is not proven", self.idx, v.0, a.show());
                        self.rep.add("GV-ASSUME", Severity::Info, m);
                    } else {
                        let m = format!("stmt {}: assume_trusted var{} in {} ({r}): TRUSTED, the initial value is not proven inside", self.idx, v.0, a.show());
                        self.rep.add("GV-ASSUME", Severity::Info, m);
                        self.trusted_counted.insert(*v);
                    }
                } else if init_ok {
                    let m = format!("stmt {}: assume var{} in {}: initial value proven inside; every set_var is checked against it (an unproven step is an error)", self.idx, v.0, a.show());
                    self.rep.add("GV-ASSUME", Severity::Info, m);
                } else {
                    let init = self.var_init.get(v).map(|i| i.show()).unwrap_or_default();
                    let m = format!("stmt {}: assume var{} in {}: the initial value {init} is not provably inside the assumed range (use assume_trusted(.., reason) if it holds by a guard the analyzer cannot see)", self.idx, v.0, a.show());
                    self.rep.add("GV-ASSUME-VIOLATED", Severity::Error, m);
                }
            }
            Opnd::Val(v) => {
                let cur = self.int_of(target);
                let proved = cur.within(&a, &self.cx);
                let refined = cur.meet(&a);
                self.env.insert(*v, AV::Int(refined));
                self.assumed_vals.insert(*v);
                let m = format!("stmt {}: assume v{v} in {}: {}", self.idx, a.show(), if proved { "already implied by the analysis (redundant)" } else { "TRUSTED" });
                self.rep.add("GV-ASSUME", Severity::Info, m);
            }
            _ => {}
        }
    }

    fn run(&mut self) {
        let f = self.f;
        let mut idx = 0usize;
        self.walk(f.body(), &mut idx);
    }

    /// Walks the region tree; `idx` is the statement index of the linearization (what findings print as `stmt N`).
    fn walk(&mut self, nodes: &[Node], idx: &mut usize) {
        // loop-exit facts: after `break_if(c)` the rest of this body runs only when c is false (a break condition is wave-uniform), so the
        // operand of a `x < k` / `x >= k` condition is refined until the end of the enclosing node list (`AMDC_NO_BREAK_FACTS=1`: off)
        let mut break_facts: Vec<Option<(Opnd, Option<AV>, Option<Itv>)>> = Vec::new();
        let pend0 = self.pending_sets.len();
        let use_break = std::env::var("AMDC_NO_BREAK_FACTS").map_or(true, |v| v.is_empty());
        for n in nodes {
            match n {
                Node::S(st) => {
                    self.idx = *idx;
                    self.exec_stmt(st);
                    *idx += 1;
                    if let (true, Stmt::BreakIf { cond }) = (use_break, st) {
                        let u = self.branch_fact(cond, false);
                        break_facts.push(u);
                    }
                }
                Node::Loop { vars, init, body } => {
                    let begin = *idx;
                    self.idx = begin;
                    self.walk_loop(vars, init, body, begin);
                    // BeginLoop + body + EndLoop (the end marker has no effect of its own)
                    *idx = begin + 1 + flat_len(body) + 1;
                }
                Node::If { cond, then, els } => {
                    self.idx = *idx;
                    self.exec_stmt(&Stmt::BeginIf { cond: *cond });
                    *idx += 1;
                    // branch facts (scoped): inside `then` the condition holds, inside `else` it does not (`if` is wave-level:
                    // every lane of a wave that runs the branch satisfies it only for wave-uniform conditions - see `branch_fact`)
                    let undo = self.branch_fact(cond, true);
                    // a `wg_winner` branch: one workgroup per group along its axes
                    let wn = self.winner_axes.len();
                    for w in &self.f.contract.wg_winners {
                        if matches!(cond, Opnd::Val(c) if *c == w.cond) {
                            self.winner_axes.extend(w.axes.iter().copied());
                        }
                    }
                    let before = self.scope_state();
                    let sym_before = self.var_sym.borrow().clone();
                    self.walk(then, idx);
                    let sym_then = self.var_sym.borrow().clone();
                    *self.var_sym.borrow_mut() = sym_before;
                    self.winner_axes.truncate(wn);
                    self.undo_fact(undo);
                    let after_then = self.scope_state();
                    self.set_scope_state(&before);
                    if let Some(e) = els {
                        self.idx = *idx;
                        self.exec_stmt(&Stmt::Else);
                        *idx += 1;
                        let undo = self.branch_fact(cond, false);
                        self.walk(e, idx);
                        self.undo_fact(undo);
                    }
                    // the join of the two paths (an arm that leaves the loop contributes too: an over-approximation)
                    let after_else = self.scope_state();
                    let joined = self.join_scope(&after_then, &after_else);
                    self.set_scope_state(&joined);
                    // a symbolic value survives the join only when both paths agree
                    {
                        let mut se = self.var_sym.borrow_mut();
                        se.retain(|k, v| sym_then.get(k) == Some(v));
                    }
                    self.idx = *idx;
                    self.exec_stmt(&Stmt::EndIf);
                    *idx += 1;
                }
                Node::Region { kind, tile, slot, version, body } => {
                    self.idx = *idx;
                    let open = match kind {
                        RegionKind::Produce => Stmt::TileProduce { tile: tile.clone(), slot: *slot, version: *version },
                        RegionKind::Consume => Stmt::TileConsume { tile: tile.clone(), slot: *slot, version: *version },
                    };
                    self.exec_stmt(&open);
                    *idx += 1;
                    self.walk(body, idx);
                    self.idx = *idx;
                    self.exec_stmt(&Stmt::TileEnd);
                    *idx += 1;
                }
                // a pipeline stage (fusion-g): StageBegin + body + StageEnd in the linearization; the markers do not change the analysis state
                // (GV-STAGE, `verify_stages`, checks the contracts on the linear view)
                Node::Stage { body, .. } => {
                    *idx += 1;
                    self.walk(body, idx);
                    *idx += 1;
                }
            }
        }
        // this list's set_vars, under its loop-exit facts
        let sets: Vec<(usize, VarId, Opnd)> = self.pending_sets.drain(pend0..).collect();
        for (i, var, value) in sets {
            let saved = self.idx;
            self.idx = i;
            self.apply_set(i, &var, &value);
            self.idx = saved;
        }
        for u in break_facts.into_iter().rev() {
            self.undo_fact(u);
        }
    }

    /// The fact a branch adds about a WAVE-UNIFORM operand of its condition: `x < k` (or its negation) refines the interval of `x` for the
    /// branch. `begin_if` is a wave-level branch (taken when ANY lane's condition holds), so a per-lane value proves nothing about the other
    /// lanes: only operands that are the same in every lane (loop variables, which the DSL keeps scalar, and values built from kernargs /
    /// constants / workgroup ids) are refined. Returns how to undo it.
    fn branch_fact(&mut self, cond: &Opnd, positive: bool) -> Option<(Opnd, Option<AV>, Option<Itv>)> {
        let Opnd::Val(c) = cond else { return None };
        // (x, k, x < k holds on this arm, k is offset by +1: `x <= k` is `x < k + 1`)
        let (x, k, lt, plus1) = match self.defs.get(c)?.clone() {
            Op::NotI1(n) => return self.branch_fact(&n, !positive),
            Op::CmpLtI32(x, k) => (x, k, positive, 0),
            Op::CmpGeI32(x, k) => (x, k, !positive, 0),
            // the generic integer compare (exact-mode / lifted IR): x OP k or k OP x; an unsigned compare only of a provably non-negative x
            Op::Cmp { pred, ty: ty @ (amdgpu_dsl::CmpTy::I32 | amdgpu_dsl::CmpTy::U32), a, b } => {
                use amdgpu_dsl::CmpPred as P;
                // normalise to `x pred k` with x the non-constant side
                let (x, k, pred) = if self.int_of(&b).point_const().is_some() || matches!(b, Opnd::I32(_)) {
                    (a, b, pred)
                } else {
                    let flip = match pred {
                        P::Lt => P::Gt,
                        P::Le => P::Ge,
                        P::Gt => P::Lt,
                        P::Ge => P::Le,
                        other => other,
                    };
                    (b, a, flip)
                };
                if ty == amdgpu_dsl::CmpTy::U32 && !(self.int_of(&x).nonneg(&self.cx) && self.int_of(&k).nonneg(&self.cx)) {
                    return None;
                }
                match pred {
                    P::Lt => (x, k, positive, 0),
                    P::Ge => (x, k, !positive, 0),
                    P::Le => (x, k, positive, 1),
                    P::Gt => (x, k, !positive, 1),
                    _ => return None,
                }
            }
            _ => return None,
        };
        if !self.uniform_opnd(&x, 6, &mut Vec::new()) || !self.uniform_opnd(&k, 6, &mut Vec::new()) {
            return None;
        }
        let ki = self.int_of(&k).add_const(plus1);
        let add = if lt { Itv { lo: Vec::new(), hi: ki.hi.iter().map(|p| p.add(&Poly::int(-1))).collect() } } else { Itv { lo: ki.lo.clone(), hi: Vec::new() } };
        let xi = self.int_of(&x);
        // the branch bound first (a constant bound from the branch beats a symbolic candidate the cap would keep), then the existing ones
        // (the strongest existing candidate first, then the branch bound, then the rest: the cap never drops either)
        let merge = |old: &Vec<Poly>, new: &Vec<Poly>| -> Vec<Poly> {
            let mut v: Vec<Poly> = old.iter().take(1).cloned().collect();
            v.extend(new.iter().cloned());
            v.extend(old.iter().skip(1).cloned());
            v
        };
        let lo = merge(&xi.lo, &add.lo);
        let hi = merge(&xi.hi, &add.hi);
        let refined = Itv { lo: cap(lo), hi: cap(hi) };
        match x {
            Opnd::Var(v) => {
                let old = self.vars.insert(v, refined);
                Some((x, None, old))
            }
            Opnd::Val(v) => match self.env.get(&v) {
                Some(AV::Int(_)) => {
                    let old = self.env.insert(v, AV::Int(refined));
                    // values computed EARLIER from v (e.g. a prelude address) are re-evaluated under the fact (undone with it)
                    let mut saved: Vec<(ValueId, Option<AV>)> = Vec::new();
                    // (breadth first: the nearest dependents first, within a generous bound)
                    let mut work: std::collections::VecDeque<ValueId> = std::collections::VecDeque::from(vec![v]);
                    let mut deps: Vec<(usize, ValueId)> = Vec::new();
                    let mut seen: HashSet<ValueId> = HashSet::new();
                    while let Some(u) = work.pop_front() {
                        for &d in self.users.get(&u).map(|x| x.as_slice()).unwrap_or(&[]) {
                            let Some(&pd) = self.rings.let_pos.get(&d) else { continue };
                            // (only pure functions of values: a definition that reads a variable read it at ITS point, not now)
                            let reads_var = self.defs.get(&d).map_or(true, |op| op.operands().iter().any(|o| matches!(o, Opnd::Var(_))));
                            if pd < self.idx && !reads_var && self.env.contains_key(&d) && seen.len() < 65536 && seen.insert(d) {
                                deps.push((pd, d));
                                work.push_back(d);
                            }
                        }
                    }
                    deps.sort();
                    self.rep.quiet += 1;
                    for (_, d) in deps {
                        if let Some(op) = self.defs.get(&d).cloned() {
                            let nv = self.eval(&op);
                            // both the old and the re-evaluated interval hold: keep the candidates of both (the new first)
                            let nv = match (nv, self.env.get(&d)) {
                                (AV::Int(n), Some(AV::Int(o))) => {
                                    let mut lo = n.lo.clone();
                                    lo.extend(o.lo.iter().cloned());
                                    let mut hi = n.hi.clone();
                                    hi.extend(o.hi.iter().cloned());
                                    AV::Int(Itv { lo: cap(lo), hi: cap(hi) })
                                }
                                (n, _) => n,
                            };
                            saved.push((d, self.env.insert(d, nv)));
                        }
                    }
                    self.rep.quiet -= 1;
                    self.dep_undo.push(saved);
                    Some((x, old, None))
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn undo_fact(&mut self, undo: Option<(Opnd, Option<AV>, Option<Itv>)>) {
        match undo {
            Some((Opnd::Var(v), _, old)) => match old {
                Some(i) => {
                    self.vars.insert(v, i);
                }
                None => {
                    self.vars.remove(&v);
                }
            },
            Some((Opnd::Val(v), old, _)) => {
                for (d, o) in self.dep_undo.pop().unwrap_or_default().into_iter().rev() {
                    match o {
                        Some(a) => {
                            self.env.insert(d, a);
                        }
                        None => {
                            self.env.remove(&d);
                        }
                    }
                }
                match old {
                    Some(a) => {
                        self.env.insert(v, a);
                    }
                    None => {
                        self.env.remove(&v);
                    }
                }
            }
            _ => {}
        }
    }

    /// Is the operand the same in every lane? Constants, kernargs, workgroup ids, readfirstlane results, integer arithmetic of uniform operands,
    /// and loop variables whose initial value and every set value are uniform.
    fn uniform_opnd(&self, o: &Opnd, depth: usize, visiting: &mut Vec<VarId>) -> bool {
        match o {
            Opnd::I32(_) => true,
            Opnd::F32(_) => false,
            Opnd::Var(v) => {
                // coinductive: a loop variable is uniform when every value it is set to is uniform assuming it is
                if visiting.contains(v) {
                    return true;
                }
                if depth == 0 {
                    return false;
                }
                let f = self.f;
                let mut sets = f.stmts().iter().filter_map(|s| match s {
                    Stmt::SetVar { var, value } if var == v => Some(*value),
                    _ => None,
                });
                let inits = f.stmts().iter().filter_map(|s| match s {
                    Stmt::BeginLoop { vars, init } | Stmt::DeclareVars { vars, init } => vars.iter().position(|x| x == v).map(|k| init[k]),
                    _ => None,
                });
                let all: Vec<Opnd> = inits.chain(&mut sets).collect();
                visiting.push(*v);
                let r = !all.is_empty() && all.iter().all(|x| self.uniform_opnd(x, depth - 1, visiting));
                visiting.pop();
                r
            }
            Opnd::Val(v) => {
                if depth == 0 {
                    return false;
                }
                match self.defs.get(v) {
                    Some(Op::ConstI32(_) | Op::KernargI32 { .. } | Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ) => true,
                    Some(Op::AddI32(a, b) | Op::MulI32(a, b) | Op::ShlI32(a, b) | Op::LshrI32(a, b) | Op::AndI32(a, b) | Op::XorI32(a, b) | Op::MinI32(a, b)) => {
                        self.uniform_opnd(a, depth - 1, visiting) && self.uniform_opnd(b, depth - 1, visiting)
                    }
                    Some(Op::ReadFirstLane(_) | Op::MovS { .. }) => true,
                    Some(Op::AddK { a, .. }) => self.uniform_opnd(a, depth - 1, visiting),
                    Some(Op::MovV { a, .. }) => self.uniform_opnd(a, depth - 1, visiting),
                    _ => false,
                }
            }
        }
    }

    /// A loop: fixpoint of the loop variables over the body, then the reporting pass.
    /// The E5 scope variables' current values (None = undefined on every path so far).
    fn scope_state(&self) -> Vec<(VarId, Option<Itv>)> {
        let mut v: Vec<(VarId, Option<Itv>)> = self.scope.iter().map(|x| (*x, self.vars.get(x).cloned())).collect();
        v.sort_by_key(|(x, _)| x.0);
        v
    }
    fn set_scope_state(&mut self, s: &[(VarId, Option<Itv>)]) {
        for (x, it) in s {
            match it {
                Some(i) => {
                    self.vars.insert(*x, i.clone());
                }
                None => {
                    self.vars.remove(x);
                }
            }
        }
    }
    fn join_scope(&self, a: &[(VarId, Option<Itv>)], b: &[(VarId, Option<Itv>)]) -> Vec<(VarId, Option<Itv>)> {
        let bm: HashMap<VarId, Option<Itv>> = b.iter().cloned().collect();
        a.iter()
            .map(|(x, ia)| {
                let ib = bm.get(x).cloned().flatten();
                let j = match (ia, ib) {
                    (Some(p), Some(q)) => Some(p.join(&q, &self.cx).norm(&self.cx)),
                    (Some(p), None) => Some(p.clone()),
                    (None, q) => q,
                };
                (*x, j)
            })
            .collect()
    }
    /// E5 scope variables a loop body writes (`set_var`): they take part in the loop's fixpoint.
    fn scope_written(&self, body: &[Node]) -> Vec<VarId> {
        fn scan(nodes: &[Node], out: &mut Vec<VarId>) {
            for n in nodes {
                match n {
                    Node::S(Stmt::SetVar { var, .. }) => out.push(*var),
                    Node::S(_) => {}
                    Node::Loop { body, .. } | Node::Region { body, .. } | Node::Stage { body, .. } => scan(body, out),
                    Node::If { then, els, .. } => {
                        scan(then, out);
                        if let Some(e) = els {
                            scan(e, out);
                        }
                    }
                }
            }
        }
        let mut out = Vec::new();
        scan(body, &mut out);
        out.retain(|v| self.scope.contains(v));
        out.sort_by_key(|v| v.0);
        out.dedup();
        out
    }

    fn walk_loop(&mut self, vars: &[VarId], init: &[Opnd], body: &[Node], begin: usize) {
        self.tag = None;
        // E5 scope variables written in the body: their entry value joined with the body's end value, to a fixpoint (widened to
        // unbounded after 5 rounds), before the reporting walk
        let sw = self.scope_written(body);
        // a scope variable written in the loop has no single symbolic value in it or after it
        for v in &sw {
            self.var_sym.borrow_mut().remove(v);
        }
        if !sw.is_empty() {
            let mut cur: Vec<Option<Itv>> = sw.iter().map(|v| self.vars.get(v).cloned()).collect();
            let mut ok = false;
            for round in 0..6 {
                for (v, it) in sw.iter().zip(cur.iter()) {
                    match it {
                        Some(i) => {
                            self.vars.insert(*v, i.clone());
                        }
                        None => {
                            self.vars.remove(v);
                        }
                    }
                }
                // a quiet walk of the loop as a whole (its own loop variables included)
                self.rep.quiet += 1;
                let saved_sets = self.pending_sets.len();
                self.walk_loop_inner(vars, init, body, begin);
                self.pending_sets.truncate(saved_sets);
                self.rep.quiet -= 1;
                let mut changed = false;
                for (k, v) in sw.iter().enumerate() {
                    if self.assumed.contains_key(v) {
                        continue;
                    }
                    let end = self.vars.get(v).cloned();
                    let j = match (&cur[k], end) {
                        (Some(p), Some(q)) => Some(p.join(&q, &self.cx).norm(&self.cx)),
                        (Some(p), None) => Some(p.clone()),
                        (None, q) => q,
                    };
                    if j != cur[k] {
                        changed = true;
                        cur[k] = if round >= 4 { Some(Itv::top()) } else { j };
                    }
                }
                if !changed {
                    ok = true;
                    break;
                }
            }
            if !ok {
                for (k, v) in sw.iter().enumerate() {
                    if !self.assumed.contains_key(v) {
                        cur[k] = Some(Itv::top());
                    }
                }
            }
            for (v, it) in sw.iter().zip(cur.iter()) {
                match it {
                    Some(i) => {
                        self.vars.insert(*v, i.clone());
                    }
                    None => {
                        self.vars.remove(v);
                    }
                }
            }
            // the reporting walk starts from the fixpoint; after the loop the variables hold the hull
            self.walk_loop_inner(vars, init, body, begin);
            for v in &sw {
                self.var_sym.borrow_mut().remove(v);
            }
            for (v, it) in sw.iter().zip(cur.iter()) {
                let end = self.vars.get(v).cloned();
                let j = match (it, end) {
                    (Some(p), Some(q)) => Some(p.join(&q, &self.cx).norm(&self.cx)),
                    (Some(p), None) => Some(p.clone()),
                    (None, q) => q,
                };
                match j {
                    Some(i) => {
                        self.vars.insert(*v, i);
                    }
                    None => {
                        self.vars.remove(v);
                    }
                }
            }
            return;
        }
        self.walk_loop_inner(vars, init, body, begin);
    }

    fn walk_loop_inner(&mut self, vars: &[VarId], init: &[Opnd], body: &[Node], begin: usize) {
        self.tag = None;
        let mut cur: Vec<Itv> = Vec::new();
        for (v, i0) in vars.iter().zip(init.iter()) {
            let it = self.var_part(*v, i0);
            self.var_init.insert(*v, it.clone());
            self.assumed.remove(v);
            cur.push(it.norm(&self.cx));
        }
        // fixpoint of the loop variables over the body (join of the initial value and every set_var value), widened to unbounded after 5 rounds
        let mut converged = false;
        for round in 0..6 {
            for (v, it) in vars.iter().zip(cur.iter()) {
                self.vars.insert(*v, it.clone());
                self.acc.remove(v);
            }
            self.rep.quiet += 1;
            {
            let mut j = begin + 1;
            self.walk(body, &mut j);
        }
            self.rep.quiet -= 1;
            let mut changed = false;
            for (k, v) in vars.iter().enumerate() {
                // a variable restricted by `assume` keeps its assumed range (checked in the reporting pass)
                if let Some(a) = self.assumed.get(v) {
                    if cur[k] != *a {
                        cur[k] = a.clone();
                        changed = true;
                    }
                    continue;
                }
                let joined = match self.acc.get(v) {
                    Some(x) => cur[k].join(x, &self.cx).norm(&self.cx),
                    None => cur[k].clone(),
                };
                if joined != cur[k] {
                    changed = true;
                    cur[k] = if round >= 4 { Itv::top() } else { joined };
                }
            }
            if !changed {
                converged = true;
                break;
            }
        }
        if !converged {
            for (k, v) in vars.iter().enumerate() {
                if !self.assumed.contains_key(v) {
                    cur[k] = Itv::top();
                }
            }
        }
        for (v, it) in vars.iter().zip(cur.iter()) {
            self.vars.insert(*v, it.clone());
            self.acc.remove(v);
        }
        {
            let mut j = begin + 1;
            self.walk(body, &mut j);
        }
        // after the loop a variable holds its initial value or a set_var value: the loop hull still bounds it
        for (v, it) in vars.iter().zip(cur.iter()) {
            self.vars.insert(*v, it.clone());
        }
    }

    /// One straight-line statement (or a structure marker of an if / tile region) at statement index `self.idx`.
    fn exec_stmt(&mut self, st: &Stmt) {
        let f = self.f;
        let i = self.idx;
        let _ = i;
        match st {
            Stmt::Let { dst, op } => {
                self.check_op(op);
                let mut v = self.eval(op);
                if matches!(op, Op::BufferRsrc { .. } | Op::BufferRsrcDyn { .. }) {
                    self.check_descriptor(&v);
                }
                // an output of a (non-union) workgroup map is a coordinate of a bijection onto the grid: it lies in [0, Grid(axis) - 1] (part of
                // the map's fact, counted with its reason)
                if let Some((m, k)) = self.f.contract.wg_maps.iter().filter(|m| !m.union).find_map(|m| m.outs.iter().position(|o| o == dst).map(|k| (m, k))) {
                    if let Some(&ax) = m.axes.get(k) {
                        let g = Itv::range(Poly::int(0), self.cx.sym_poly(Sym::Grid(ax)).sub(&Poly::int(1)));
                        v = match v {
                            AV::Int(i) => {
                                let mut lo = i.lo.clone();
                                lo.extend(g.lo);
                                let mut hi = i.hi.clone();
                                hi.extend(g.hi);
                                AV::Int(Itv { lo: cap(lo), hi: cap(hi) })
                            }
                            _ => AV::Int(g),
                        };
                    }
                }
                self.env.insert(*dst, v);
            }
            Stmt::Assume { target, lo, hi, trusted } => self.assume(target, lo, hi, trusted),
            // the author's EXEC: the lanes that execute (a DMA writes only theirs)
            Stmt::ExecAll => self.exec_lanes = 64,
            Stmt::ExecSet { mask } => self.exec_lanes = 64 - mask.leading_zeros(),
            Stmt::ExecSave { sgpr } => {
                self.exec_saved.insert(*sgpr, self.exec_lanes);
            }
            Stmt::ExecRestore { sgpr } => self.exec_lanes = self.exec_saved.get(sgpr).copied().unwrap_or(64),
            Stmt::TrustWait { reason } => {
                let sev = if f.exact { Severity::Warn } else { Severity::Error };
                self.rep.add("GV-TRUST-TIMING", sev, format!("stmt {}: TRUSTED timing (the next read is not guaranteed by the wait counters): {reason}{}", self.idx, if f.exact { "" } else { " - only exact (non-dispatched) kernels may state it" }));
            }
            Stmt::TileProduce { tile, slot, .. } | Stmt::TileConsume { tile, slot, .. } => {
                self.tag = f.contract.tiles.iter().position(|t| &t.name == tile).map(|ti| (ti, self.int_of(slot), *slot));
                self.tag_stmt = self.idx;
            }
            Stmt::TileEnd | Stmt::Barrier | Stmt::EndLoop | Stmt::Else | Stmt::EndIf => self.tag = None,
            Stmt::StoreU16 { base, byte_off, .. } => {
                self.record_global_store(base, *byte_off as i128, 2);
                let b = self.av_of(base);
                self.check_global("store_u16", &b, *byte_off as i128, 2, true)
            }
            Stmt::StoreF32 { base, byte_off, .. } | Stmt::StoreF32Wt { base, byte_off, .. } => {
                self.record_global_store(base, *byte_off as i128, 4);
                let b = self.av_of(base);
                self.check_global("store_f32", &b, *byte_off as i128, 4, true)
            }
            Stmt::StoreF32x2 { base, byte_off, .. } => {
                self.record_global_store(base, *byte_off as i128, 8);
                let b = self.av_of(base);
                self.check_global("store_f32x2", &b, *byte_off as i128, 8, true)
            }
            Stmt::StoreF32x4 { base, byte_off, .. } => {
                self.record_global_store(base, *byte_off as i128, 16);
                let b = self.av_of(base);
                self.check_global("store_f32x4", &b, *byte_off as i128, 16, true)
            }
            Stmt::BufferStore { rsrc, voff, soff, imm, dwords, .. } => {
                self.record_buffer_store(rsrc, voff, soff, *imm as i128, 4 * *dwords as i128);
                let r = self.av_of(rsrc);
                self.check_buffer("buffer store", &r, voff, soff, *imm as i128, 4 * *dwords as i128, true);
            }
            Stmt::BufferLoadLds { rsrc, voff, soff, lds_addr, imm, dwords } => {
                let r = self.av_of(rsrc);
                self.check_buffer("buffer load ... lds (global side)", &r, voff, soff, *imm as i128, 4 * *dwords as i128, false);
                let a = self.int_of(lds_addr).add_const(*imm as i128);
                self.cur_addr = Some((*lds_addr, *imm as i128));
                let lanes = self.exec_lanes as i128;
                let before = self.rep.findings.len();
                self.check_lds(&format!("buffer load ... lds (LDS side: M0 + imm + lane*size, {lanes} lanes)"), &a, lanes * 4 * *dwords as i128);
                if self.rep.findings.len() > before {
                    let sh = self.shape(lds_addr, 6);
                    if let Some(last) = self.rep.findings.last_mut() {
                        last.msg.push_str(&format!(" | M0 = {sh} ; imm {imm}"));
                    }
                }
            }
            Stmt::LoadGlobalLds { base, lds_addr, byte_off, dwords } => {
                let b = self.av_of(base);
                self.check_global("global_load_lds (global side)", &b, *byte_off as i128, 4 * *dwords as i128, false);
                let a = self.int_of(lds_addr).add_const(*byte_off as i128);
                self.cur_addr = Some((*lds_addr, *byte_off as i128));
                let before = self.rep.findings.len();
                let lanes = self.exec_lanes as i128;
                self.check_lds(&format!("global_load_lds (LDS side: M0 + imm + lane*size, {lanes} lanes)"), &a, lanes * 4 * *dwords as i128);
                if self.rep.findings.len() > before {
                    let sh = self.shape(lds_addr, 5);
                    if let Some(last) = self.rep.findings.last_mut() {
                        last.msg.push_str(&format!(" | M0 = {sh} ; imm {byte_off}"));
                    }
                }
            }
            Stmt::StoreLdsB32 { addr, .. } => self.check_lds_at("ds_write_b32", addr, 4),
            Stmt::StoreLdsB128 { addr, .. } => self.check_lds_at("ds_write_b128", addr, 16),
        Stmt::StoreLdsB64 { addr, .. } => self.check_lds_at("ds_write_b64", addr, 8),
            Stmt::StoreGlobalN { base, byte_off, ty, .. } => {
                self.record_global_store(base, *byte_off as i128, ty.bytes() as i128);
                let b = self.av_of(base);
                self.check_global("global store", &b, *byte_off as i128, ty.bytes() as i128, true)
            }
            Stmt::BufferStoreN { rsrc, voff, soff, imm, ty, .. } => {
                self.record_buffer_store(rsrc, voff, soff, *imm as i128, ty.bytes() as i128);
                let r = self.av_of(rsrc);
                self.check_buffer("buffer store", &r, voff, soff, *imm as i128, ty.bytes() as i128, true);
            }
            Stmt::StoreLdsN { addr, ty, .. } => self.check_lds_at("ds_write", addr, ty.bytes() as i128),
            Stmt::AtomicLdsNoRet { addr, .. } => self.check_lds_at("ds atomic", addr, 4),
            Stmt::BufferAtomicNoRet { rsrc, voff, soff, imm, .. } => {
                let r = self.av_of(rsrc);
                self.check_buffer("buffer atomic", &r, voff, soff, *imm as i128, 4, true);
            }
            Stmt::AtomicGlobalNoRet { base, byte_off, .. } => {
                let b = self.av_of(base);
                self.check_global("global atomic", &b, *byte_off as i128, 4, true)
            }
            Stmt::StoreLds2 { addr, off0, off1, b64, st64, .. } => self.check_lds2("ds_write2", addr, *off0, *off1, *b64, *st64),
            Stmt::BeginLoop { .. } => unreachable!("loops are walked by `walk_loop`"),
            Stmt::SetVar { var, value } => {
                // applied at the end of the enclosing statement list, so a later `break_if` of the same list refines the value (the back
                // edge is reached only when the break was not taken)
                self.pending_sets.push((i, *var, *value));
                // an E5 scope variable holds the value from here on (flow-sensitive; `if` joins and loop fixpoints cover the paths)
                if self.scope.contains(var) {
                    // the symbolic value (through authored moves)
                    let mut r = *value;
                    for _ in 0..8 {
                        match r {
                            Opnd::Val(v) => match self.defs.get(&v) {
                                Some(Op::MovS { a, .. } | Op::MovV { a, .. }) => r = *a,
                                _ => break,
                            },
                            _ => break,
                        }
                    }
                    if r == Opnd::Var(*var) {
                        // a copy of itself: unchanged
                    } else {
                        self.var_sym.borrow_mut().insert(*var, r);
                    }
                    let it = self.var_part(*var, value);
                    let it = match self.assumed.get(var) {
                        Some(a) => a.clone(),
                        None => it,
                    };
                    self.vars.insert(*var, it);
                }
            }
            // E5 scope variables: defined by their init (absent = `Undef`: no value on any path yet)
            Stmt::DeclareVars { vars, init } => {
                for (v, i0) in vars.iter().zip(init.iter()) {
                    self.scope.insert(*v);
                    let undef = matches!(i0, Opnd::Val(x) if matches!(self.defs.get(x), Some(Op::Undef { .. })));
                    self.var_sym.borrow_mut().remove(v);
                    if undef {
                        self.vars.remove(v);
                        continue;
                    }
                    self.var_sym.borrow_mut().insert(*v, *i0);
                    let it = self.var_part(*v, i0);
                    self.var_init.insert(*v, it.clone());
                    let cur = self.assumed.get(v).cloned().unwrap_or(it);
                    self.vars.insert(*v, cur);
                }
            }
            Stmt::BeginIf { .. } => self.tag = None,
            Stmt::VarAsm { .. } | Stmt::VarAsmN { .. } => self.has_asm = true,
            _ => {}
        }
    }

    fn apply_set(&mut self, i: usize, var: &VarId, value: &Opnd) {
        {
            {
                let nv = self.var_part(*var, value);
                if self.rep.quiet > 0 {
                    let e = match self.acc.get(var) {
                        Some(x) => x.join(&nv, &self.cx),
                        None => nv.clone(),
                    };
                    self.acc.insert(*var, e);
                }
                if let Some(a) = self.assumed.get(var).cloned() {
                    if !nv.within(&a, &self.cx) {
                        if let Some(r) = self.assumed_trusted.get(var).cloned() {
                            // counted once per variable
                            if self.rep.quiet == 0 && self.trusted_counted.insert(*var) {
                                let m = format!("stmt {i}: set_var var{} = {} is not provably inside the assumed range {} (TRUSTED: {r})", var.0, nv.show(), a.show());
                                self.rep.add("GV-ASSUME", Severity::Info, m);
                            }
                        } else {
                            let m = format!("stmt {i}: set_var var{} = {} is not provably inside the assumed range {}: the assumption is not inductive (a loop-carried assume that the loop itself can violate is a proof hole; weaken it, or use assume_trusted(.., reason) when a guard the analyzer cannot see bounds it)", var.0, nv.show(), a.show());
                            self.rep.add("GV-ASSUME-VIOLATED", Severity::Error, m);
                        }
                    }
                }
            }
        }
    }
}

/// Number of statements of the linearization of `nodes` (structure markers included).
fn flat_len(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|n| match n {
            Node::S(_) => 1,
            Node::Loop { body, .. } | Node::Region { body, .. } | Node::Stage { body, .. } => 2 + flat_len(body),
            Node::If { then, els, .. } => 2 + flat_len(then) + els.as_ref().map_or(0, |e| 1 + flat_len(e)),
        })
        .sum()
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// role / barrier protocol

/// Dynamic barrier count of a statement range as a polynomial in the loop trip counts: index d = coefficient of the product of the trip counts of d nested loops
/// (all loops at the same nesting depth are assumed to run the same number of iterations in both roles, which is how the role kernels are built).
/// `deltas` collects, per nested if in order, the barrier difference (else arm - then arm; an if without else counts as an empty else arm): the
/// totals follow the then-sides only and would not see a barrier missing from an else arm (a per-exit drain, the full-tile arm of a tail test). Two
/// roles built alike must have equal totals AND equal deltas, i.e. the same barrier count on every combination of (uniform) branch outcomes.
/// Does the then-arm of the `BeginIf` at `pos` end with a `break_if` (E5: a loop exit with its own body)?
fn arm_exits(stmts: &[Stmt], pos: usize) -> bool {
    let mut depth = 0i32;
    let mut i = pos + 1;
    while i < stmts.len() {
        match &stmts[i] {
            Stmt::BeginIf { .. } | Stmt::BeginLoop { .. } => depth += 1,
            Stmt::EndIf | Stmt::EndLoop if depth > 0 => depth -= 1,
            Stmt::Else | Stmt::EndIf if depth == 0 => return i > pos + 1 && matches!(stmts[i - 1], Stmt::BreakIf { .. }),
            _ => {}
        }
        i += 1;
    }
    false
}

/// Does the loop body starting at `pos` (after `BeginLoop`) leave other than through E5 exit arms (a `break_if` that is not the last
/// statement of an exit arm)?
fn loop_has_plain_break(stmts: &[Stmt], pos: usize) -> bool {
    let mut depth = 0i32;
    let mut i = pos;
    while i < stmts.len() {
        match &stmts[i] {
            Stmt::BeginLoop { .. } => depth += 1,
            Stmt::EndLoop if depth == 0 => return false,
            Stmt::EndLoop => depth -= 1,
            Stmt::BreakIf { .. } if depth == 0 => {
                let last_of_arm = matches!(stmts.get(i + 1), Some(Stmt::Else | Stmt::EndIf));
                if !last_of_arm {
                    return true;
                }
            }
            _ => {}
        }
        i += 1;
    }
    false
}

fn add_poly(a: &mut Vec<i64>, b: &[i64]) {
    if a.len() < b.len() {
        a.resize(b.len(), 0);
    }
    for (d, c) in b.iter().enumerate() {
        a[d] += c;
    }
}

/// `bar_counts` with the E5 loop exits: an arm ending in `break_if` runs once, on its exit path - its barriers (with the iteration
/// prefix before it) go to `exits`, one polynomial per exit path (the barriers before the loop + N iterations + the prefix + the arm +
/// what follows the loop), in program order, instead of into the per-iteration count.
fn bar_counts(stmts: &[Stmt], pos: &mut usize, until: &dyn Fn(&Stmt) -> bool, deltas: &mut Vec<Vec<i64>>) -> Vec<i64> {
    let mut exits = Vec::new();
    let mut dead = false;
    bar_counts_x(stmts, pos, until, deltas, &mut exits, &mut dead)
}

/// `dead` is set when the range ends behind a loop that only leaves through exit arms (its fall-through count is no path).
fn bar_counts_x(stmts: &[Stmt], pos: &mut usize, until: &dyn Fn(&Stmt) -> bool, deltas: &mut Vec<Vec<i64>>, exits: &mut Vec<Vec<i64>>, dead: &mut bool) -> Vec<i64> {
    let mut out: Vec<i64> = vec![0];
    // exits recorded at this level: (path count so far, `out` when recorded) - the barriers after the record are added at the end
    let mut mine: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
    let finish = |out: &Vec<i64>, mine: Vec<(Vec<i64>, Vec<i64>)>, exits: &mut Vec<Vec<i64>>| {
        for (mut e, snap) in mine {
            let mut tail = out.clone();
            let neg: Vec<i64> = snap.iter().map(|c| -c).collect();
            add_poly(&mut tail, &neg);
            add_poly(&mut e, &tail);
            exits.push(trim(e));
        }
    };
    while *pos < stmts.len() {
        if until(&stmts[*pos]) {
            finish(&out, mine, exits);
            return out;
        }
        match &stmts[*pos] {
            Stmt::Barrier => {
                out[0] += 1;
                *pos += 1;
            }
            Stmt::BeginLoop { .. } => {
                *pos += 1;
                let plain = loop_has_plain_break(stmts, *pos);
                let mut inner_exits = Vec::new();
                let mut inner_dead = false;
                let inner = bar_counts_x(stmts, pos, &|s| matches!(s, Stmt::EndLoop), deltas, &mut inner_exits, &mut inner_dead);
                *pos += 1;
                if !plain && !inner_exits.is_empty() {
                    *dead = true;
                }
                let mut iter = vec![0];
                for (d, c) in inner.iter().enumerate() {
                    if iter.len() < d + 2 {
                        iter.resize(d + 2, 0);
                    }
                    iter[d + 1] += c;
                }
                // an exit path: the barriers so far + N iterations + its own (prefix + arm)
                for e in inner_exits {
                    let mut p = out.clone();
                    add_poly(&mut p, &iter);
                    add_poly(&mut p, &e);
                    mine.push((p, {
                        let mut s = out.clone();
                        add_poly(&mut s, &iter);
                        s
                    }));
                }
                add_poly(&mut out, &iter);
            }
            Stmt::BeginIf { .. } if arm_exits(stmts, *pos) => {
                // E5 exit with its body: the iteration prefix + the arm, once (an else arm, if any, continues the iteration)
                *pos += 1;
                let mut arm_exits_inner = Vec::new();
                let mut d2 = false;
                let t = bar_counts_x(stmts, pos, &|s| matches!(s, Stmt::Else | Stmt::EndIf), deltas, &mut arm_exits_inner, &mut d2);
                let mut e = out.clone();
                add_poly(&mut e, &t);
                exits.push(trim(e));
                exits.extend(arm_exits_inner);
                if matches!(stmts.get(*pos), Some(Stmt::Else)) {
                    *pos += 1;
                    let mut d3 = false;
                    let el = bar_counts_x(stmts, pos, &|s| matches!(s, Stmt::EndIf), deltas, exits, &mut d3);
                    add_poly(&mut out, &el);
                }
                *pos += 1;
            }
            Stmt::BeginIf { .. } => {
                *pos += 1;
                let t = bar_counts(stmts, pos, &|s| matches!(s, Stmt::Else | Stmt::EndIf), deltas);
                let mut e = vec![0];
                if matches!(stmts.get(*pos), Some(Stmt::Else)) {
                    *pos += 1;
                    e = bar_counts(stmts, pos, &|s| matches!(s, Stmt::EndIf), deltas);
                }
                let n = t.len().max(e.len());
                deltas.push(trim((0..n).map(|d| e.get(d).copied().unwrap_or(0) - t.get(d).copied().unwrap_or(0)).collect()));
                *pos += 1;
                // a nested if contributes its then-side (nested divergent ifs are checked on their own)
                if out.len() < t.len() {
                    out.resize(t.len(), 0);
                }
                for (d, c) in t.iter().enumerate() {
                    out[d] += c;
                }
            }
            _ => *pos += 1,
        }
    }
    finish(&out, mine, exits);
    out
}

fn trim(mut v: Vec<i64>) -> Vec<i64> {
    while v.len() > 1 && *v.last().unwrap() == 0 {
        v.pop();
    }
    v
}
fn barrier_balance(f: &Function, rep: &mut Report) {
    // lane/wave-dependent values: derived from the workitem id (different waves of a workgroup take different paths)
    let mut tv: HashSet<ValueId> = HashSet::new();
    let mut tvars: HashSet<VarId> = HashSet::new();
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
    // every wave-divergent if/else must have the same barrier signature on both sides
    let mut pos = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    while pos < f.stmts().len() {
        if let Stmt::BeginIf { cond } = &f.stmts()[pos] {
            let divergent = tainted(cond, &tv, &tvars);
            let mut p = pos + 1;
            let (mut te, mut ee) = (Vec::new(), Vec::new());
            let (mut tx, mut ex) = (Vec::new(), Vec::new());
            let (mut tdead, mut edead) = (false, false);
            let t = trim(bar_counts_x(f.stmts(), &mut p, &|s| matches!(s, Stmt::Else | Stmt::EndIf), &mut te, &mut tx, &mut tdead));
            let mut e = vec![0];
            if matches!(f.stmts().get(p), Some(Stmt::Else)) {
                p += 1;
                e = trim(bar_counts_x(f.stmts(), &mut p, &|s| matches!(s, Stmt::EndIf), &mut ee, &mut ex, &mut edead));
            }
            // both sides end in loops left only through exits: the fall-through counts are no path, the exit paths are compared below
            let (t, e) = if tdead && edead && !tx.is_empty() { (vec![0], vec![0]) } else { (t, e) };
            // E5 loop exits: the two sides must leave by the same exits in the same order, each with the same barrier count
            if divergent && tx != ex {
                rep.add(
                    "GV-BAR",
                    Severity::Error,
                    format!("stmt {pos}: wave-divergent if/else (condition depends on the workitem id): the loop exit paths run different numbers of barriers: then {tx:?} vs else {ex:?} (per exit, coefficients of 1, N, ...) - waves that run fewer barriers deadlock the workgroup"),
                );
                pos += 1;
                continue;
            }
            // nested ifs whose arms run the same barriers change nothing (and the two sides need not have the same nesting for them)
            te.retain(|d| d.iter().any(|&x| x != 0));
            ee.retain(|d| d.iter().any(|&x| x != 0));
            if divergent && t == e && te != ee {
                rep.add(
                    "GV-BAR",
                    Severity::Error,
                    format!("stmt {pos}: wave-divergent if/else (condition depends on the workitem id): a nested if changes the barrier count differently in the two sides (else - then per nested if: then side {te:?} vs else side {ee:?}), so some combination of branch outcomes runs different numbers of barriers - waves that run fewer barriers deadlock the workgroup"),
                );
            } else if divergent && t != e {
                rep.add(
                    "GV-BAR",
                    Severity::Error,
                    format!("stmt {pos}: wave-divergent if/else (condition depends on the workitem id) executes a different number of barriers: then {t:?} vs else {e:?} (coefficients of 1, N, N*M, ... for the nested loop trip counts) - waves that run fewer barriers deadlock the workgroup"),
                );
            } else if divergent {
                rep.ok("barrier-balanced-ifs");
            }
        }
        stack.clear();
        pos += 1;
    }
}

/// IR-level verification: LDS budget, global / buffer bounds, role-barrier balance, contract sanity.
pub fn verify_ir(f: &Function) -> Report {
    let mut an = An { f, cx: contract_ctx(f), env: HashMap::new(), vars: HashMap::new(), var_prov: HashMap::new(), rings: ring_facts(f), users: value_users(f), dep_undo: Vec::new(), exec_lanes: 64, exec_saved: HashMap::new(), var_init: HashMap::new(), scope: Default::default(), var_sym: Default::default(), assumed: HashMap::new(), assumed_trusted: HashMap::new(), tag: None, tag_stmt: 0, gstores: BTreeMap::new(), atom_inst: Default::default(), sig_memo: Default::default(), caps_cur: Default::default(), gcaps: BTreeMap::new(), pending_sets: Vec::new(), winner_axes: Vec::new(), gwin: BTreeMap::new(), trusted_counted: Default::default(), group_caps: Default::default(), group_win: Default::default(), cur_addr: None, acc: HashMap::new(), ovr: None, assumed_vals: Default::default(), defs: f.stmts().iter().filter_map(|s| if let Stmt::Let { dst, op } = s { Some((*dst, op.clone())) } else { None }).collect(), rep: Report::default(), idx: 0, has_asm: false };
    if std::env::var("AMDC_GV_DEBUG").is_ok() || std::env::var("AMDC_GV_LDS_TRACE").is_ok() {
        for (v, r) in &an.rings.rings {
            eprintln!("RING var{} range [{}, {}] starts {:?} rule {:?}", v.0, r.0, r.1, an.rings.starts.get(v), an.rings.rules.get(v));
        }
        for (v, x) in &an.rings.sym {
            eprintln!("RING var{} start {:?} (expression; orbits enumerated at use) rule {:?}", v.0, x, an.rings.rules.get(v));
        }
        for (w, s) in &an.rings.shadows {
            eprintln!("SHADOW var{} = {:?} - {} + ring var{}", w.0, s.0, s.1, s.4 .0);
        }
        eprintln!("rings {:?}", an.rings.rings);
        eprintln!("shadows {:?}", an.rings.shadows.iter().map(|(w, s)| (w, s.1, s.2, s.3)).collect::<Vec<_>>());
    }
    for e in &f.contract.extents {
        if !f.args.iter().any(|a| a.name == e.arg && a.ty == amdgpu_dsl::DType::Ptr) {
            an.rep.add("GV-CONTRACT", Severity::Error, format!("contract: extent declared for `{}` which is not a pointer kernarg of `{}`", e.arg, f.name));
        }
    }
    for r in &f.contract.buf_regions {
        if !f.args.iter().any(|a| a.name == r.arg && a.ty == amdgpu_dsl::DType::Ptr) {
            an.rep.add("GV-CONTRACT", Severity::Error, format!("contract: buf_region `{}` names `{}` which is not a pointer kernarg of `{}`", r.name, r.arg, f.name));
        }
    }
    for la in &f.contract.last_arrivers {
        if !f.args.iter().any(|a| a.name == la.counter_arg && a.ty == amdgpu_dsl::DType::Ptr) {
            an.rep.add("GV-CONTRACT", Severity::Error, format!("contract: last-arriver counter `{}` is not a pointer kernarg of `{}`", la.counter_arg, f.name));
        }
    }
    an.run();
    for w in &f.contract.wg_winners {
        an.rep.add("GV-ASSUME", Severity::Info, format!("wg_winner v{} over grid axes {:?}: TRUSTED ({})", w.cond, w.axes, w.reason));
    }
    for m in &f.contract.wg_maps {
        an.rep.add("GV-ASSUME", Severity::Info, format!("wg_bijection {:?} over grid axes {:?}: TRUSTED ({})", m.outs, m.axes, m.reason));
    }
    let (gw_ok, gw_bad) = an.global_disjointness();
    for _ in 0..gw_ok {
        an.rep.ok("global-write-disjoint");
    }
    for _ in 0..gw_bad {
        an.rep.unchecked_access("global-write-untracked");
    }
    let has_asm = an.has_asm;
    let mut rep = an.rep;
    if has_asm {
        rep.add("GV-ASM", Severity::Info, "safe-asm blocks present: their internal memory accesses are NOT bounds-checked by v1 (address inputs come from IR values that are)".to_string());
    }
    barrier_balance(f, &mut rep);
    verify_stages(f, &mut rep);
    crate::gpu_protocol::verify_tiles(f, &mut rep);
    crate::gpu_protocol::verify_last_arrivers(f, &mut rep);
    rep
}

/// The barrier / tile-protocol part of `verify_ir` alone (role barrier balance `GV-BAR`, the LDS / global tile protocol `GV-TILE-*` / `GV-PROTO-*`,
/// last arrivers): the same verdicts for those codes as `verify_ir`, without the value / bounds analysis. For mutation tests that re-verify a kernel
/// many times (`gpu_tiles.rs`: every barrier removed in turn).
pub fn verify_ir_protocol(f: &Function) -> Report {
    verify_ir_protocol_cap(f, None)
}

/// `verify_ir_protocol` simulating at most `cap` loop iterations (see `gpu_protocol::verify_tiles_cap`): a mutation test's fast check.
pub fn verify_ir_protocol_cap(f: &Function, cap: Option<usize>) -> Report {
    let mut rep = Report::default();
    barrier_balance(f, &mut rep);
    crate::gpu_protocol::verify_tiles_cap(f, &mut rep, cap);
    crate::gpu_protocol::verify_last_arrivers(f, &mut rep);
    rep
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// MIR level: register legality + path-sensitive wait / lifetime

/// Symbolic linear form of an i32 value: atom (is_var, id) -> coefficient polynomial, plus a constant polynomial.
type SLin = (std::collections::BTreeMap<(bool, u64), Poly>, Poly);
/// Per-atom value range (lo, hi) of a recorded store (None: not boundable).
type ARanges = Option<std::collections::BTreeMap<(bool, u64), (Poly, Poly)>>;

fn slin_add(x: SLin, y: SLin) -> SLin {
    let (mut a, c1) = x;
    for (k, v) in y.0 {
        let e = a.entry(k).or_insert(Poly::int(0));
        *e = e.add(&v);
    }
    a.retain(|_, v| !v.0.is_empty());
    (a, c1.add(&y.1))
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

const VGPR_LIMIT: u32 = crate::target::T.verify.vgpr_limit;
const SGPR_LIMIT: u32 = crate::target::T.verify.sgpr_limit;
const LDS_LIMIT: u32 = crate::target::T.verify.lds_limit;

fn is_matrix_op(op: MOp) -> bool {
    // every MFMA except the sparse smfmac (op_attrs)
    crate::agpr::is_mfma(op)
}

/// Final-MIR verification: every allocated register obeys the class rules and the declared limits; matrix operands do not overlap their destination;
/// no instruction touches a register with a load in flight on ANY control-flow path (loop back edges included).
pub fn verify_mir(f: &Function, m: &MFunction, a: &Alloc) -> Report {
    let mut rep = Report::default();
    let mut used: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for b in &m.blocks {
        for i in &b.insts {
            if let Some(d) = i.dst {
                used.insert(d);
            }
            for s in &i.srcs {
                if let Src::Reg(r) = s {
                    used.insert(*r);
                }
            }
        }
    }
    for &r in &used {
        let c: RClass = m.vreg_class(r);
        let p = a.phys[r as usize];
        let n = c.dwords();
        if c.is_agpr() {
            // AGPR file (gfx90a+/gfx950): index = phys - AGPR_TAG; tuples start at EVEN indices like VGPR tuples (assembler-measured: a[1:4], a[1:2],
            // a[1:8] are rejected for ds_read / MFMA operands, a[2:5], a[2:3], a[2:9] accepted); 256 AGPRs per wave.
            if !is_agpr_phys(p) {
                rep.add("GV-REG-CLASS", Severity::Error, format!("vreg {r} ({c:?}) is an AGPR-class value but was allocated to a non-AGPR register {p}"));
                continue;
            }
            let ai = p - AGPR_TAG;
            if n >= 2 && ai % 2 != 0 {
                rep.add("GV-REG-ALIGN", Severity::Error, format!("vreg {r} ({c:?}) at a{ai}: a {n}-dword AGPR tuple must start at an even register (ISA, assembler-measured)"));
            } else if n >= 2 && ai % n != 0 {
                rep.add("GV-REG-POLICY", Severity::Info, format!("vreg {r} ({c:?}) at a{ai}: ISA-legal, but not aligned to the tuple width {n}"));
            }
            if ai + n > 256 {
                rep.add("GV-REG-LIMIT", Severity::Error, format!("vreg {r} ({c:?}) at a{ai}..a{}: exceeds the 256-AGPR per-wave limit", ai + n - 1));
            }
            continue;
        }
        if is_agpr_phys(p) {
            rep.add("GV-REG-CLASS", Severity::Error, format!("vreg {r} ({c:?}) is not an AGPR-class value but was allocated to AGPR a{}", p - AGPR_TAG));
            continue;
        }
        if c.is_vector() {
            // exact mode: the author may use v0 / s0..s7 once their hardware values are dead (the clobber check covers it)
            if p < 1 && !f.exact {
                rep.add("GV-REG-RESERVED", Severity::Error, format!("vreg {r} ({c:?}) allocated to v{p}: v0 holds the workitem id"));
            }
            // ISA (measured with the gfx950 assembler, 2025-11: every multi-dword VGPR tuple - MFMA A/B/C/D of any width, global/ds/buffer data, packed ops -
            // is accepted at EVEN start registers and rejected at odd ones: v[2:17], v[6:21], v[14:29] are legal V16 tuples). The allocator additionally aligns to the
            // tuple width: reported as a policy deviation (Info), not an error.
            let need = 2;
            if n >= 2 && p % need != 0 {
                rep.add("GV-REG-ALIGN", Severity::Error, format!("vreg {r} ({c:?}) at v{p}: a {n}-dword VGPR tuple must start at an even register (ISA, assembler-measured)"));
            } else if n >= 2 && p % n != 0 {
                rep.add("GV-REG-POLICY", Severity::Info, format!("vreg {r} ({c:?}) at v{p}: ISA-legal, but not aligned to the tuple width {n} (the allocator normally aligns to the width)"));
            }
            if p + n > VGPR_LIMIT {
                rep.add("GV-REG-LIMIT", Severity::Error, format!("vreg {r} ({c:?}) at v{p}..v{}: exceeds the {VGPR_LIMIT}-VGPR per-wave limit", p + n - 1));
            }
        } else {
            if p < 8 && !f.exact {
                rep.add("GV-REG-RESERVED", Severity::Error, format!("vreg {r} ({c:?}) allocated to s{p}: s0..s7 hold the kernarg pointer / workgroup ids"));
            }
            // measured: pairs at even registers; quads (buffer descriptors, s_load_dwordx4) and octets (s_load_dwordx8) at multiples of 4
            let need = if n >= 4 { 4 } else { 2 };
            if n >= 2 && p % need != 0 {
                rep.add("GV-REG-ALIGN", Severity::Error, format!("vreg {r} ({c:?}) at s{p}: an SGPR {} must start at a multiple of {need} (64-bit pairs even, 128-bit descriptors / wider multiple of 4; assembler-measured)", if n == 4 { "descriptor quad" } else { "tuple" }));
            }
            if p + n > SGPR_LIMIT && !(f.exact && ((p == crate::mir::VCC_PIN && n == 2) || (p == crate::mir::SCC_PIN && n == 1))) {
                rep.add("GV-REG-LIMIT", Severity::Error, format!("vreg {r} ({c:?}) at s{p}..s{}: exceeds the {SGPR_LIMIT}-SGPR limit", p + n - 1));
            }
        }
    }
    // unified register file: accum_offset (arch VGPRs rounded to 4) + AGPRs must fit the 512 registers of a wave
    let n_agpr = crate::regalloc::agpr_count(m, a);
    if n_agpr > 0 {
        let (accum, total, _) = crate::regalloc::unified_budget(a.max_v, n_agpr);
        if total > crate::regalloc::UNIFIED_FILE {
            rep.add("GV-REG-LIMIT", Severity::Error, format!("unified register file: {accum} arch VGPRs (accum_offset) + {n_agpr} AGPRs = {total} > {}", crate::regalloc::UNIFIED_FILE));
        } else {
            rep.ok("agpr-unified-budget");
        }
    }
    if a.max_v > VGPR_LIMIT {
        rep.add("GV-REG-LIMIT", Severity::Error, format!("kernel uses {} VGPRs (limit {VGPR_LIMIT})", a.max_v));
    }
    if let Err(e) = crate::regalloc::check_workgroup_budget(m, a) {
        rep.add("GV-REG-OCCUPANCY", Severity::Error, e);
    }
    if a.max_s > SGPR_LIMIT {
        rep.add("GV-REG-LIMIT", Severity::Error, format!("kernel uses {} SGPRs (limit {SGPR_LIMIT})", a.max_s));
    }
    if m.lds_bytes > LDS_LIMIT {
        rep.add("GV-LDS-OOB", Severity::Error, format!("declared + spill LDS {} bytes exceeds the {LDS_LIMIT}-byte per-workgroup limit", m.lds_bytes));
    }
    if m.lds_bytes < f.lds_bytes {
        rep.add("GV-LDS-OOB", Severity::Error, format!("emitted LDS segment {} bytes is smaller than the DSL-declared {}", m.lds_bytes, f.lds_bytes));
    }
    // matrix operand / destination overlap
    let range = |r: u32| -> (u32, u32) {
        let p = a.phys[r as usize];
        (p, p + m.vreg_class(r).dwords())
    };
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, i) in b.insts.iter().enumerate() {
            if !is_matrix_op(i.op) {
                continue;
            }
            let Some(d) = i.dst else { continue };
            let rd = range(d);
            for k in 0..2usize {
                if let Some(Src::Reg(s)) = i.srcs.get(k) {
                    let rs = range(*s);
                    if rs.0 < rd.1 && rd.0 < rs.1 {
                        rep.add("GV-REG-MFMA", Severity::Error, format!("block {bi} inst {ii} ({:?}): multiplicand {} v[{}..{}) overlaps the destination v[{}..{})", i.op, ["A", "B"][k], rs.0, rs.1, rd.0, rd.1));
                    }
                }
            }
        }
    }
    // MFMA vdst and srcC must be in the same register file (arch VGPR / AGPR) - the acc_cd bit selects one file for both
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, i) in b.insts.iter().enumerate() {
            if !crate::agpr::is_mfma(i.op) || crate::op_attrs::attrs(i.op).mfma.map_or(false, |m| m.zero_c) {
                continue;
            }
            if let (Some(d), Some(Src::Reg(c))) = (i.dst, i.srcs.get(2)) {
                if is_agpr_phys(a.phys[d as usize]) != is_agpr_phys(a.phys[*c as usize]) {
                    rep.add("GV-REG-MFMA-FILE", Severity::Error, format!("block {bi} inst {ii} ({:?}): vdst and srcC are in different register files", i.op));
                }
            }
        }
    }    // an exact kernel with `TrustWait` waivers: exact::finalize verified every wait site itself (the waived ones counted as GV-TRUST-TIMING)
    let waived = f.exact && f.stmts().iter().any(|s| matches!(s, Stmt::TrustWait { .. }));
    if waived {
        rep.ok("waits(exact, per site)");
    } else if let Err(e) = crate::sched::waits::verify_waits(m) {
        rep.add("GV-WAIT", Severity::Error, format!("{e} (on at least one control-flow path, including loop back edges)"));
    } else {
        rep.ok("wait-paths");
    }
    rep
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// policy

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    /// Collect and print findings, never fail.
    Warn,
    /// Any `Error` finding fails the compile.
    Strict,
    /// Default: findings are collected silently; MIR-level errors (registers, waits) fail the compile, IR-level ones do not (kernels need their contract first).
    Default,
}

pub fn mode_from_env() -> Mode {
    match std::env::var("AMDC_GPU_VERIFY").ok().as_deref() {
        Some("off") | Some("0") => Mode::Off,
        Some("warn") => Mode::Warn,
        Some("strict") | Some("1") => Mode::Strict,
        _ => Mode::Default,
    }
}

/// Full analysis of a kernel (compiles it to the final MIR first).
pub fn analyse(f: &Function) -> Result<Report, String> {
    let (m, a) = crate::compile_mir_alloc(f)?;
    let mut rep = verify_ir(f);
    rep.merge(verify_mir(f, &m, &a));
    Ok(rep)
}

/// Called by `compile_with` on the final MIR.
pub fn enforce(f: &Function, m: &MFunction, a: &Alloc) -> Result<(), String> {
    let mode = mode_from_env();
    if mode == Mode::Off {
        return Ok(());
    }
    let mut rep = verify_ir(f);
    let mir = verify_mir(f, m, a);
    let mir_errors: Vec<String> = mir.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect();
    rep.merge(mir);
    if mode == Mode::Warn && !rep.findings.is_empty() {
        eprintln!("gpu_verify({}):\n{}", f.name, rep.render());
    }
    match mode {
        Mode::Strict => {
            let errs: Vec<String> = rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect();
            if errs.is_empty() {
                Ok(())
            } else {
                Err(format!("gpu_verify (strict) rejected `{}`: {} error(s)\n{}", f.name, errs.len(), errs.join("\n")))
            }
        }
        Mode::Default => {
            if mir_errors.is_empty() {
                Ok(())
            } else {
                Err(format!("gpu_verify rejected `{}` (register / wait legality): {} error(s)\n{}", f.name, mir_errors.len(), mir_errors.join("\n")))
            }
        }
        _ => Ok(()),
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// Stage boundaries (Node::Stage, docs/fusion-layout-handoff.md §3): the resource contract checked at the region edge

/// GV-STAGE: every stage's contract against the program.
/// * a consumed set (`live_in`) is handed over by an EARLIER stage's `live_out` with the same values, element table (layout fingerprint) and
///   element width, and the same wave roles;
/// * every `live_out` value is defined before the stage ends;
/// * a value defined inside one stage and read inside another must be declared in the reader's `live_in`;
/// * `lds_reused` tiles were freed by an earlier stage, and no stage writes a tile an earlier stage freed unless it declares the reuse;
/// * `barrier_phase` equals the number of barriers before the stage in program order;
/// * `layout` / `storage` of a consumed set match the hand-over when both sides record them (VGPR <-> AGPR allowed);
/// * `scalars_in` are handed over by an earlier stage's `scalars_out` with the same values and owner;
/// * `outstanding`: the vm / lgkm operations in flight at the stage entry (from earlier code) and at its exit (linear view, explicit waits
///   applied) do not exceed the declared tolerances;
/// * `resources.barriers` equals the number of barriers inside the stage.
pub fn verify_stages(f: &Function, rep: &mut Report) {
    use amdgpu_dsl::{stmt_operands, StageContract};
    let stmts = f.stmts();
    if !stmts.iter().any(|s| matches!(s, Stmt::StageBegin { .. })) {
        return;
    }
    let mut closed: Vec<(String, StageContract)> = Vec::new();
    let mut open: Vec<(usize, String, StageContract)> = Vec::new();
    let mut def_stage: HashMap<u32, usize> = HashMap::new();
    let mut defined: std::collections::HashSet<u32> = Default::default();
    let mut barriers = 0u32;
    let mut sid = 0usize;
    let mut freed: Vec<String> = Vec::new();
    let mut ok = 0usize;
    // linear in-flight counters (vm, lgkm) and the barrier count at each open stage's entry
    let (mut vm, mut lgkm) = (0u32, 0u32);
    let mut stage_barriers: Vec<u32> = Vec::new();
    for (i, st) in stmts.iter().enumerate() {
        let (dvm, dlgkm) = mem_counts(st);
        vm += dvm;
        lgkm += dlgkm;
        match st {
            Stmt::WaitVm { count } => vm = vm.min(*count as u32),
            Stmt::WaitLds { count } => lgkm = lgkm.min(*count as u32),
            _ => {}
        }
        match st {
            Stmt::StageBegin { name, contract } => {
                sid += 1;
                stage_barriers.push(barriers);
                let o = &contract.outstanding;
                if o.vm_in.is_some_and(|n| vm > n) || o.lgkm_in.is_some_and(|n| lgkm > n) {
                    rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: {vm} vm / {lgkm} lgkm operations in flight at the entry exceed the tolerance {:?} / {:?}", o.vm_in, o.lgkm_in));
                }
                for want in &contract.scalars_in {
                    match closed.iter().rev().find_map(|(pn, c)| c.scalars_out.iter().find(|x| x.name == want.name).map(|x| (pn, x))) {
                        // values bound outside every stage (kernel glue) are visible to all stages: no producer needed
                        None if want.values.iter().all(|v| defined.contains(&(*v as u32)) && !def_stage.contains_key(&(*v as u32))) => ok += 1,
                        None => rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: scalar `{}` is not handed over by an earlier stage", want.name)),
                        Some((pn, have)) => {
                            if have.values != want.values || have.owner != want.owner {
                                rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: scalar `{}` (owner {}) differs from stage {pn}'s hand-over (owner {})", want.name, want.owner, have.owner));
                            } else {
                                ok += 1;
                            }
                        }
                    }
                }
                if contract.barrier_phase != StageContract::ANY_PHASE && contract.barrier_phase != barriers {
                    rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: barrier phase {} declared, {barriers} barriers precede it", contract.barrier_phase));
                }
                for want in &contract.live_in {
                    match closed.iter().rev().find_map(|(pn, c)| c.live_out.iter().find(|x| x.name == want.name).map(|x| (pn, c, x))) {
                        None => rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: `{}` is not handed over by an earlier stage", want.name)),
                        Some((pn, pc, have)) => {
                            if have.values != want.values {
                                rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: `{}` values differ from stage {pn}'s hand-over", want.name));
                            } else if !have.layout.is_empty() && !want.layout.is_empty() && have.layout != want.layout {
                                rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: `{}` TV layout {} differs from stage {pn}'s {}", want.name, want.layout, have.layout));
                            } else if have.storage != want.storage && !matches!((have.storage.as_str(), want.storage.as_str()), ("vgpr", "agpr") | ("agpr", "vgpr")) {
                                rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: `{}` storage {} vs stage {pn}'s {}", want.name, want.storage, have.storage));
                            } else if have.table != want.table || have.elem_bits != want.elem_bits {
                                rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: `{}` layout / width differs from stage {pn}'s hand-over (a conversion must be code inside a stage)", want.name));
                            } else if pc.roles != contract.roles {
                                rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: roles {:?} vs stage {pn}'s {:?}", contract.roles, pc.roles));
                            } else {
                                ok += 1;
                            }
                        }
                    }
                }
                for r in &contract.lds_reused {
                    if !freed.contains(r) {
                        rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: reuses LDS tile `{r}` that no earlier stage freed"));
                    }
                }
                open.push((sid, name.clone(), (**contract).clone()));
            }
            Stmt::StageEnd => {
                if let Some((_, name, c)) = open.pop() {
                    let b0 = stage_barriers.pop().unwrap_or(0);
                    if let Some(n) = c.resources.barriers {
                        if barriers - b0 != n {
                            rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: {} barriers inside, {n} declared", barriers - b0));
                        }
                    }
                    let o = &c.outstanding;
                    if o.vm_out.is_some_and(|n| vm > n) || o.lgkm_out.is_some_and(|n| lgkm > n) {
                        rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: {vm} vm / {lgkm} lgkm operations in flight at the exit exceed the declared {:?} / {:?}", o.vm_out, o.lgkm_out));
                    }
                    for v in c.scalars_out.iter().flat_map(|x| x.values.iter()) {
                        if !defined.contains(&(*v as u32)) {
                            rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: hands over scalar v{v} which is not defined"));
                        }
                    }
                    for v in c.live_out.iter().flat_map(|x| x.values.iter()) {
                        if !defined.contains(&(*v as u32)) {
                            rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name}: hands over v{v} which is not defined"));
                        }
                    }
                    freed.extend(c.lds_freed.iter().cloned());
                    closed.push((name, c));
                }
            }
            Stmt::Barrier => barriers += 1,
            Stmt::TileProduce { tile, .. } => {
                if freed.contains(tile) && !open.last().is_some_and(|(_, _, c)| c.lds_reused.contains(tile)) {
                    rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: writes LDS tile `{tile}` freed by an earlier stage without declaring the reuse"));
                }
            }
            _ => {}
        }
        if let Stmt::Let { dst, op } = st {
            defined.insert(*dst as u32);
            // constants are not hand-overs (the builder pools them: a constant first used in one stage is reused by later ones)
            let constant = matches!(op, amdgpu_dsl::Op::ConstI32(_) | amdgpu_dsl::Op::ConstF32(_));
            if let (Some((s, _, _)), false) = (open.last(), constant) {
                def_stage.insert(*dst as u32, *s);
            }
        }
        if let Some((s, name, c)) = open.last() {
            let ops: Vec<Opnd> = match st {
                Stmt::Let { op, .. } => op.operands(),
                other => stmt_operands(other),
            };
            for o in ops {
                if let Opnd::Val(v) = o {
                    if let Some(ds) = def_stage.get(&(v as u32)) {
                        if ds != s && !c.live_in.iter().any(|x| x.values.contains(&v)) && !c.scalars_in.iter().any(|x| x.values.contains(&v)) {
                            rep.add("GV-STAGE", Severity::Error, format!("stmt {i}: stage {name} reads v{v} of another stage without declaring it live-in"));
                        }
                    }
                }
            }
        }
    }
    for _ in 0..ok {
        rep.ok("stage-handoff");
    }
}

/// (vm, lgkm) operations a statement issues (the counters `s_waitcnt vmcnt / lgkmcnt` track; linear-view accounting for GV-STAGE).
fn mem_counts(st: &Stmt) -> (u32, u32) {
    use amdgpu_dsl::Op;
    match st {
        Stmt::StoreU16 { .. } | Stmt::StoreF32 { .. } | Stmt::StoreF32x4 { .. } | Stmt::StoreF32x2 { .. } | Stmt::BufferStore { .. } | Stmt::StoreF32Wt { .. } => (1, 0),
        Stmt::BufferLoadLds { .. } | Stmt::LoadGlobalLds { .. } => (1, 0),
        Stmt::StoreLdsB32 { .. } | Stmt::StoreLdsB128 { .. } | Stmt::StoreLdsB64 { .. } => (0, 1),
        Stmt::StoreGlobalN { .. } | Stmt::BufferStoreN { .. } => (1, 0),
        Stmt::StoreLdsN { .. } | Stmt::StoreLds2 { .. } | Stmt::AtomicLdsNoRet { .. } => (0, 1),
        Stmt::AtomicGlobalNoRet { .. } | Stmt::BufferAtomicNoRet { .. } => (1, 0),
        Stmt::Let { op, .. } => match op {
            Op::LoadU16 { .. } | Op::LoadF32 { .. } | Op::LoadGlobalB128 { .. } | Op::LoadF32Coherent { .. } | Op::BufferLoad { .. } => (1, 0),
            Op::BufferLoadB128x2 { .. } => (2, 0),
            Op::LoadLdsB32 { .. } | Op::LoadLdsB128 { .. } | Op::LoadLdsB64 { .. } | Op::LoadLdsTrB8 { .. } => (0, 1),
            Op::LoadLdsB128x2 { .. } => (0, 2),
            Op::LoadLdsTr4B8 { .. } => (0, 4),
            Op::LoadGlobalN { .. } | Op::LoadGlobalD16 { .. } | Op::BufferLoadN { .. } => (1, 0),
            Op::LoadLdsN { .. } | Op::LoadLdsD16 { .. } | Op::LoadLds2 { .. } | Op::LoadLdsTrB16 { .. } | Op::AtomicLds { .. } | Op::Bpermute { .. } | Op::Permute { .. } | Op::Swizzle { .. } => (0, 1),
            Op::AtomicGlobal { .. } | Op::BufferAtomicRet { .. } => (1, 0),
            _ => (0, 0),
        },
        _ => (0, 0),
    }
}

/// Ring counters and their shadows, proven from the IR alone (gpu_verify uses the ranges as facts, nothing is trusted):
/// * a ring `v`: every value it is set to (an init or a `set_var`) is a constant or `v + Select{Cmp(v, K), a, b}` with constants (through
///   `MovS` / `MovV`), the compare reading `v` with no set of `v` in between; the reachable values are enumerated from the constants;
/// * a shadow `w` of ring `v`: `w` is set once to some X in the straight-line block where `v` is set to a constant c, and otherwise only to
///   `w + D` where the same `D` value updates `v` in the same block: then `w = X - c + v` at every point after both, so `w` lies in
///   `X - c + [min, max]`.
/// Proven ring counters (see `ring_facts`).
#[derive(Default)]
struct RingFacts {
    /// ring -> [min, max] of its reachable values
    rings: HashMap<VarId, (i128, i128)>,
    /// shadow -> (its init X, the ring's init c, min, max, the ring): the shadow is X - c + ring
    shadows: HashMap<VarId, (Opnd, i128, i128, i128, VarId)>,
    /// step values (the `Select`s the rings are updated with) -> their ring
    steps: HashMap<ValueId, VarId>,
    /// ring -> its step rule (pred, ty, var on the left, K, a, b)
    rules: HashMap<VarId, (amdgpu_dsl::CmpPred, amdgpu_dsl::CmpTy, bool, i128, i128, i128)>,
    /// statement positions of every set of a variable, and of every `Let`
    set_pos: HashMap<VarId, Vec<usize>>,
    let_pos: HashMap<ValueId, usize>,
    /// rings whose single non-constant start value is an expression (e.g. a per-wave base): the start's interval is enumerated at use
    sym: HashMap<VarId, Opnd>,
    /// constant-start rings: their constant starts
    starts: HashMap<VarId, Vec<i128>>,
    sym_cache: std::cell::RefCell<HashMap<(VarId, i128, i128), Option<(i128, i128)>>>,
}

impl RingFacts {
    /// The reachable values of a ring started anywhere in [lo, hi] (None: too many starts or a run that does not cycle).
    fn orbit_range(&self, v: VarId, lo: i128, hi: i128) -> Option<(i128, i128)> {
        if let Some(r) = self.sym_cache.borrow().get(&(v, lo, hi)) {
            return *r;
        }
        let r = (|| {
            let &(pred, ty, var_left, k, ka, kb) = self.rules.get(&v)?;
            if hi < lo || hi - lo > 1 << 17 {
                return None;
            }
            let (mut mn, mut mx) = (i128::MAX, i128::MIN);
            for x0 in lo..=hi {
                let mut x = x0;
                let mut seen = std::collections::BTreeSet::new();
                while seen.insert(x) {
                    if seen.len() > 64 {
                        return None;
                    }
                    mn = mn.min(x);
                    mx = mx.max(x);
                    x = ((x + if ring_cmp(pred, ty, var_left, k, x) { ka } else { kb }) as i64 as i32) as i128;
                }
            }
            Some((mn, mx))
        })();
        self.sym_cache.borrow_mut().insert((v, lo, hi), r);
        r
    }
}

/// The period of a proven ring counter (or of a shadow: its ring's) when every constant start lies on one pure cycle; for the tile simulator.
pub(crate) fn ring_period(f: &Function, x: VarId) -> Option<usize> {
    let rf = ring_facts(f);
    let v = if rf.rings.contains_key(&x) { x } else { rf.shadows.get(&x)?.4 };
    let &(pred, ty, var_left, k, ka, kb) = rf.rules.get(&v)?;
    let starts = rf.starts.get(&v)?;
    let mut period = 0usize;
    for &x0 in starts {
        let mut cur = x0;
        let mut p = 0usize;
        loop {
            cur = ((cur + if ring_cmp(pred, ty, var_left, k, cur) { ka } else { kb }) as i64 as i32) as i128;
            p += 1;
            if cur == x0 {
                break;
            }
            if p > 64 {
                return None;
            }
        }
        if period != 0 && period != p {
            return None;
        }
        period = p;
    }
    (period > 0).then_some(period)
}

fn ring_cmp(pred: amdgpu_dsl::CmpPred, ty: amdgpu_dsl::CmpTy, var_left: bool, k: i128, x: i128) -> bool {
    let (l, r) = if var_left { (x, k) } else { (k, x) };
    let (l, r) = match ty {
        amdgpu_dsl::CmpTy::U32 => ((l as i64 as u32) as i128, (r as i64 as u32) as i128),
        _ => ((l as i64 as i32) as i128, (r as i64 as i32) as i128),
    };
    use amdgpu_dsl::CmpPred as P;
    match pred {
        P::Lt => l < r,
        P::Le => l <= r,
        P::Gt => l > r,
        P::Ge => l >= r,
        P::Eq => l == r,
        _ => l != r,
    }
}

fn ring_facts(f: &Function) -> RingFacts {
    let stmts = f.stmts();
    let defs: HashMap<ValueId, Op> = stmts.iter().filter_map(|s| if let Stmt::Let { dst, op } = s { Some((*dst, op.clone())) } else { None }).collect();
    let pos: HashMap<ValueId, usize> = stmts.iter().enumerate().filter_map(|(k, s)| if let Stmt::Let { dst, .. } = s { Some((*dst, k)) } else { None }).collect();
    fn konst(defs: &HashMap<ValueId, Op>, o: &Opnd, depth: usize) -> Option<i128> {
        match o {
            Opnd::I32(k) => Some(*k as i128),
            Opnd::Val(v) if depth > 0 => match defs.get(v) {
                Some(Op::ConstI32(k)) => Some(*k as i128),
                Some(Op::MovS { a, .. } | Op::MovV { a, .. }) => konst(defs, a, depth - 1),
                _ => None,
            },
            _ => None,
        }
    }
    let control = |s: &Stmt| {
        let d = format!("{s:?}");
        ["BeginIf", "Else", "EndIf", "BeginLoop", "EndLoop", "Break", "Continue", "BranchPlace", "LayoutAnchor", "DeclareVars"].iter().any(|p| d.starts_with(p))
    };
    let straight = |a: usize, b: usize| {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        hi <= lo + 1 || !stmts[lo + 1..hi].iter().any(|s| control(s))
    };
    // every (position, value) a variable is set to: set_vars, plus non-Undef inits of loops / scopes
    let mut sets: HashMap<VarId, Vec<(usize, Opnd)>> = HashMap::new();
    for (k, s) in stmts.iter().enumerate() {
        match s {
            Stmt::SetVar { var, value } => sets.entry(*var).or_default().push((k, *value)),
            Stmt::BeginLoop { vars, init } | Stmt::DeclareVars { vars, init } => {
                for (v, i) in vars.iter().zip(init.iter()) {
                    let undef = matches!(i, Opnd::Val(x) if matches!(defs.get(x), Some(Op::Undef { .. })));
                    if !undef {
                        sets.entry(*v).or_default().push((k, *i));
                    }
                }
            }
            _ => {}
        }
    }
    // `value` = Var(v) + D (either order): D
    let step_of = |v: VarId, value: &Opnd| -> Option<ValueId> {
        let Opnd::Val(x) = value else { return None };
        match defs.get(x) {
            Some(Op::AddI32(Opnd::Var(w), Opnd::Val(d)) | Op::AddI32(Opnd::Val(d), Opnd::Var(w))) if *w == v => Some(*d),
            _ => None,
        }
    };
    let mut rings: HashMap<VarId, (i128, i128)> = HashMap::new();
    let mut ring_step: HashMap<VarId, Vec<ValueId>> = HashMap::new();
    let mut ring_const: HashMap<VarId, Vec<(usize, i128)>> = HashMap::new();
    let mut ring_rule: HashMap<VarId, (amdgpu_dsl::CmpPred, amdgpu_dsl::CmpTy, bool, i128, i128, i128)> = HashMap::new();
    let mut sym: HashMap<VarId, Opnd> = HashMap::new();
    'var: for (v, list) in &sets {
        let mut consts = Vec::new();
        let mut steps = Vec::new();
        let mut rule: Option<(amdgpu_dsl::CmpPred, amdgpu_dsl::CmpTy, bool, i128, i128, i128)> = None;
        let mut sym_init: Option<Opnd> = None;
        for (k, value) in list {
            if let Some(c) = konst(&defs, value, 4) {
                consts.push((*k, c));
                continue;
            }
            let Some(d) = step_of(*v, value) else {
                if sym_init.is_some() {
                    continue 'var;
                }
                sym_init = Some(*value);
                continue;
            };
            let Some(Op::Select { cond: Opnd::Val(c), a, b, .. }) = defs.get(&d) else { continue 'var };
            let (Some(ka), Some(kb)) = (konst(&defs, a, 4), konst(&defs, b, 4)) else { continue 'var };
            let Some(Op::Cmp { pred, ty, a: ca, b: cb }) = defs.get(c) else { continue 'var };
            let (var_left, kk) = match (ca, cb) {
                (Opnd::Var(w), o) if w == v => (true, konst(&defs, o, 4)),
                (o, Opnd::Var(w)) if w == v => (false, konst(&defs, o, 4)),
                _ => continue 'var,
            };
            let Some(kk) = kk else { continue 'var };
            // no set of v between the compare and this set
            let pc = pos[c];
            if pc > *k || list.iter().any(|(j, _)| *j > pc && *j < *k) {
                continue 'var;
            }
            let r = (*pred, *ty, var_left, kk, ka, kb);
            if rule.map_or(false, |x| x != r) {
                continue 'var;
            }
            rule = Some(r);
            steps.push(d);
        }
        let Some((pred, ty, var_left, k, ka, kb)) = rule else { continue };
        if let Some(x) = sym_init {
            if !consts.is_empty() {
                continue;
            }
            ring_rule.insert(*v, (pred, ty, var_left, k, ka, kb));
            ring_step.insert(*v, steps);
            sym.insert(*v, x);
            continue;
        }
        if consts.is_empty() {
            continue;
        }
        let holds = |x: i128| -> bool {
            let (l, r) = if var_left { (x, k) } else { (k, x) };
            let (l, r) = match ty {
                amdgpu_dsl::CmpTy::U32 => ((l as i64 as u32) as i128, (r as i64 as u32) as i128),
                _ => ((l as i64 as i32) as i128, (r as i64 as i32) as i128),
            };
            use amdgpu_dsl::CmpPred as P;
            match pred {
                P::Lt => l < r,
                P::Le => l <= r,
                P::Gt => l > r,
                P::Ge => l >= r,
                P::Eq => l == r,
                _ => l != r,
            }
        };
        let mut seen: std::collections::BTreeSet<i128> = std::collections::BTreeSet::new();
        let mut work: Vec<i128> = consts.iter().map(|x| x.1).collect();
        while let Some(x) = work.pop() {
            if !seen.insert(x) {
                continue;
            }
            if seen.len() > 256 {
                continue 'var;
            }
            let n = ((x + if holds(x) { ka } else { kb }) as i64 as i32) as i128;
            work.push(n);
        }
        rings.insert(*v, (*seen.first().unwrap(), *seen.last().unwrap()));
        ring_rule.insert(*v, (pred, ty, var_left, k, ka, kb));
        ring_step.insert(*v, steps);
        ring_const.insert(*v, consts);
    }
    let mut shadows: HashMap<VarId, (Opnd, i128, i128, i128, VarId)> = HashMap::new();
    'w: for (w, list) in &sets {
        if rings.contains_key(w) {
            continue;
        }
        let mut init: Option<(usize, Opnd)> = None;
        let mut ring: Option<VarId> = None;
        for (k, value) in list {
            if let Some(d) = step_of(*w, value) {
                // the same D updates a ring in the same straight-line block
                let Some((v, _)) = ring_step.iter().find(|(_, ds)| ds.contains(&d)) else { continue 'w };
                let Some((kv, _)) = sets[v].iter().find(|(_, val)| step_of(*v, val) == Some(d)) else { continue 'w };
                if !straight(*k, *kv) || ring.map_or(false, |r| r != *v) {
                    continue 'w;
                }
                ring = Some(*v);
            } else {
                if init.is_some() {
                    continue 'w;
                }
                init = Some((*k, *value));
            }
        }
        let (Some(v), Some((ki, x))) = (ring, init) else { continue };
        // the ring's single constant set in the same straight-line block as w's init
        // (a shadow of a ring with an expression start is not tracked)
        let Some(cs) = ring_const.get(&v) else { continue };
        let [(kc, c)] = cs.as_slice() else { continue };
        if !straight(ki, *kc) {
            continue;
        }
        let (lo, hi) = rings[&v];
        shadows.insert(*w, (x, *c, lo, hi, v));
    }
    let steps: HashMap<ValueId, VarId> = ring_step.iter().flat_map(|(v, ds)| ds.iter().map(move |d| (*d, *v))).collect();
    let set_pos: HashMap<VarId, Vec<usize>> = sets.iter().map(|(v, l)| (*v, l.iter().map(|x| x.0).collect())).collect();
    let starts: HashMap<VarId, Vec<i128>> = ring_const.iter().map(|(v, l)| (*v, l.iter().map(|x| x.1).collect())).collect();
    RingFacts { rings, shadows, steps, rules: ring_rule, set_pos, let_pos: pos, sym, starts, sym_cache: Default::default() }
}

/// value -> the `Let` values whose operations read it.
fn value_users(f: &Function) -> HashMap<ValueId, Vec<ValueId>> {
    let mut m: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for s in f.stmts() {
        if let Stmt::Let { dst, op } = s {
            for o in op.operands() {
                if let Opnd::Val(v) = o {
                    m.entry(v).or_default().push(*dst);
                }
            }
        }
    }
    m
}

/// `p` with the symbol `s` replaced by `q` (None: a product overflowed the polynomial representation).
fn subst_sym(p: &Poly, s: &Sym, q: &Poly) -> Option<Poly> {
    let mut out = Poly::int(0);
    for (mono, c) in &p.0 {
        let mut term = Poly::konst(*c);
        for x in mono {
            let f = if x == s { q.clone() } else { Poly::sym(x.clone()) };
            term = term.mul(&f)?;
        }
        out = out.add(&term);
    }
    Some(out)
}
