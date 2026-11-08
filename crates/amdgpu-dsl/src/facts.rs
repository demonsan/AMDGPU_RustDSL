//! Dependency facts of DSL statements and region nodes (docs/weave-capture.md): what a scheduler / weaver working ABOVE the IR needs to
//! reorder captured statements legally — SSA defs / uses, loop-variable reads / writes (`set_var` semantics), an effect set, a resource
//! class with a nominal latency, and the register footprint of the results.
//!
//! Every classification is an exhaustive `match` (a new `Op` / `Stmt` variant fails to compile until it is classified here); ids are
//! collected through the generated id visitor (`ir_codec.rs`), so operands are never missed.

use crate::ir_text::IrIds;
use crate::*;

/// Execution resource of a statement (nominal, for cost models; `amdgpu-isa` atoms give the MFMA passes — cross-checked by a test).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResClass {
    /// matrix core instruction; `passes` = XDL passes (issue cycles = 4 x passes)
    Mfma { passes: u8 },
    /// vector ALU
    Valu,
    /// transcendental VALU (exp / log / rcp)
    Trans,
    /// scalar ALU / uniform values (kernarg loads, workgroup ids, constants)
    Salu,
    LdsRead,
    LdsWrite,
    VmemLoad,
    VmemStore,
    /// global -> LDS DMA
    Dma,
    /// barriers, waits, fences, priority, sched groups, timers
    Sync,
    /// no-code markers (assume, weave-region marks)
    Marker,
    /// a structured region (loop / if / tile region / stage) taken as a whole
    Region,
    /// a safe-asm block (opaque)
    Asm,
}

impl ResClass {
    /// Nominal latency in cycles from issue to a usable result (cost-model input, not a hazard model).
    pub fn latency(self) -> u32 {
        match self {
            ResClass::Mfma { passes } => 4 * passes as u32 + 4,
            ResClass::Valu => 8,
            ResClass::Trans => 16,
            ResClass::Salu => 4,
            ResClass::LdsRead => 128,
            ResClass::LdsWrite => 64,
            ResClass::VmemLoad => 600,
            ResClass::VmemStore => 200,
            ResClass::Dma => 600,
            ResClass::Sync | ResClass::Marker | ResClass::Region | ResClass::Asm => 0,
        }
    }
    /// Issue cycles the statement occupies its unit for (MFMA: 4 x passes; everything else nominal 4 / 0).
    pub fn issue(self) -> u32 {
        match self {
            ResClass::Mfma { passes } => 4 * passes as u32,
            ResClass::Marker | ResClass::Region => 0,
            _ => 4,
        }
    }
}

/// The effect set of a statement (a region node: the union of its body).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    pub lds_read: bool,
    pub lds_write: bool,
    pub global_read: bool,
    pub global_write: bool,
    /// global -> LDS DMA (also sets `global_read` and `lds_write`)
    pub dma: bool,
    pub barrier: bool,
    /// explicit `wait_lds(n)` / `wait_vm(n)`
    pub wait_lds: Option<u8>,
    pub wait_vm: Option<u8>,
    /// memory fences (release / acquire)
    pub fence: bool,
    /// tile-protocol / stage / weave-region markers (structure the verifier and the region strategies rely on)
    pub marker: bool,
    /// `set_prio` / `sched_group` (scheduling directives) and the cycle timer
    pub directive: bool,
    /// a wave-uniform branch region (`if` / `loop`: `break_if` is a branch too); the DSL has no exec-mask changes, so `exec_change` stays false
    pub branch: bool,
    pub exec_change: bool,
    /// opaque safe-asm block
    pub asm: bool,
}

impl Effects {
    fn union(&mut self, o: &Effects) {
        self.lds_read |= o.lds_read;
        self.lds_write |= o.lds_write;
        self.global_read |= o.global_read;
        self.global_write |= o.global_write;
        self.dma |= o.dma;
        self.barrier |= o.barrier;
        self.wait_lds = match (self.wait_lds, o.wait_lds) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.wait_vm = match (self.wait_vm, o.wait_vm) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.fence |= o.fence;
        self.marker |= o.marker;
        self.directive |= o.directive;
        self.branch |= o.branch;
        self.exec_change |= o.exec_change;
        self.asm |= o.asm;
    }
    /// Touches memory (LDS or global) in any way.
    pub fn memory(&self) -> bool {
        self.lds_read || self.lds_write || self.global_read || self.global_write || self.dma
    }
    /// An ordering anchor: nothing with an effect may cross it (barriers, waits, fences, markers, directives, asm).
    pub fn anchor(&self) -> bool {
        self.barrier || self.wait_lds.is_some() || self.wait_vm.is_some() || self.fence || self.marker || self.directive || self.asm
    }
}

/// Dependency facts of one statement or region node.
#[derive(Clone, Debug, PartialEq)]
pub struct Facts {
    pub class: ResClass,
    /// values defined (a `Let`'s destination; a region: every value defined inside)
    pub defs: Vec<ValueId>,
    /// values read that are not defined by the same node
    pub uses: Vec<ValueId>,
    pub var_reads: Vec<VarId>,
    pub var_writes: Vec<VarId>,
    pub eff: Effects,
    /// the LDS tile region the node is (tile name, slot) when it is a tile region node
    pub tile: Option<(String, Opnd)>,
    /// result dwords of the defined values (register footprint; `Rsrc` / `I1` count 0 — they live in SGPRs / VCC)
    pub footprint: u32,
}

/// Dwords of a value of type `t` held in vector registers.
pub fn dwords(t: DType) -> u32 {
    match t {
        DType::F16 | DType::F32 | DType::I32 | DType::U32 => 1,
        DType::Ptr | DType::Frag2 => 2,
        DType::Frag4 => 4,
        DType::Frag8 => 8,
        DType::Frag16 => 16,
        DType::I1 | DType::Rsrc => 0,
    }
}

/// Resource class and effects of an op.
pub fn op_class(op: &Op) -> (ResClass, Effects) {
    let mut e = Effects::default();
    let c = match op {
        Op::ConstI32(_) | Op::ConstF32(_) | Op::KernargI32 { .. } | Op::KernargF32 { .. } | Op::KernargPtr { .. } | Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ => ResClass::Salu,
        Op::BufferRsrc { .. } | Op::BufferRsrcDyn { .. } | Op::ReadFirstLane(..) => ResClass::Salu,
        Op::MemTimeLo { .. } => {
            e.directive = true;
            ResClass::Sync
        }
        Op::WorkitemId
        | Op::AddI32(..)
        | Op::MulI32(..)
        | Op::ShlI32(..)
        | Op::LshrI32(..)
        | Op::AndI32(..)
        | Op::MinI32(..)
        | Op::XorI32(..)
        | Op::AddF32(..)
        | Op::SubF32(..)
        | Op::MulF32(..)
        | Op::MaxF32(..)
        | Op::Max3F32(..)
        | Op::FmaF32(..)
        | Op::CvtF16ToF32(..)
        | Op::CvtF32ToF16(..)
        | Op::CmpLtI32(..)
        | Op::CmpGeI32(..)
        | Op::CmpLtF32(..)
        | Op::NotI1(..)
        | Op::Select { .. }
        | Op::PtrAdd { .. }
        | Op::DppQuadPerm { .. }
        | Op::Frag2(..)
        | Op::Frag2Lane { .. }
        | Op::VarSlice8 { .. }
        | Op::Permlane32Swap { .. }
        | Op::Frag4(..)
        | Op::Frag4Lane { .. }
        | Op::Frag8(..)
        | Op::CatFrag8(..)
        | Op::CvtPknormU16(..)
        | Op::Frag16Lane { .. }
        | Op::FragZero { .. }
        | Op::BitcastF32(..)
        | Op::BitcastI32(..)
        | Op::CvtPkBf16F32(..)
        | Op::Frag2Dup(..)
        | Op::PkAddF32(..)
        | Op::PkMulF32(..)
        | Op::CvtPkFp8x4F32(..)
        | Op::PartnerMaxF32(..)
        | Op::CvtPknormP8x4(..)
        | Op::PermB32(..)
        | Op::PkSubU16(..)
        | Op::CvtPknormPair(..)
        | Op::CvtP8x4FromU16(..) => ResClass::Valu,
        Op::ExpF32(..) | Op::RcpF32(..) | Op::Exp2F32(..) | Op::Log2F32(..) => ResClass::Trans,
        Op::LoadU16 { .. } | Op::LoadF32 { .. } | Op::LoadGlobalB128 { .. } | Op::LoadF32Coherent { .. } | Op::BufferLoad { .. } | Op::BufferLoadB128x2 { .. } => {
            e.global_read = true;
            ResClass::VmemLoad
        }
        Op::AtomicAddU32 { .. } => {
            e.global_read = true;
            e.global_write = true;
            ResClass::VmemLoad
        }
        Op::LoadLdsB32 { .. } | Op::LoadLdsB128 { .. } | Op::LoadLdsB64 { .. } | Op::LoadLdsTrB8 { .. } | Op::LoadLdsTr4B8 { .. } | Op::LoadLdsB128x2 { .. } => {
            e.lds_read = true;
            ResClass::LdsRead
        }
        Op::MfmaF16 { .. } => ResClass::Mfma { passes: 4 },
        Op::MfmaF16Wide { .. } => ResClass::Mfma { passes: 8 },
        Op::MfmaScaleFp8_32x32x64 { .. } | Op::MfmaFp8_32x32x64 { .. } => ResClass::Mfma { passes: 16 },
        Op::SmfmacFp8_16x16x128 { .. } => ResClass::Mfma { passes: 8 },
        Op::Asm { .. } => {
            e.asm = true;
            ResClass::Asm
        }
    };
    (c, e)
}

/// Resource class, effects and the loop variables a statement WRITES (a `Var*` in-place update also reads its variable; the zero-C MFMA
/// forms and `set_var` do not).
fn stmt_class(st: &Stmt) -> (ResClass, Effects, Vec<VarId>, bool) {
    let mut e = Effects::default();
    // (class, writes, the written variable's old value is read)
    let (c, w, rmw): (ResClass, Vec<VarId>, bool) = match st {
        Stmt::Let { op, .. } => {
            let (c, oe) = op_class(op);
            e = oe;
            (c, vec![], false)
        }
        Stmt::StageBegin { .. } | Stmt::StageEnd | Stmt::RegionMark { .. } | Stmt::TileProduce { .. } | Stmt::TileConsume { .. } | Stmt::TileEnd => {
            e.marker = true;
            (ResClass::Marker, vec![], false)
        }
        Stmt::Assume { .. } => (ResClass::Marker, vec![], false),
        Stmt::StoreU16 { .. } | Stmt::StoreF32 { .. } | Stmt::StoreF32x4 { .. } | Stmt::StoreF32x2 { .. } | Stmt::BufferStore { .. } | Stmt::StoreF32Wt { .. } => {
            e.global_write = true;
            (ResClass::VmemStore, vec![], false)
        }
        Stmt::BufferLoadLds { .. } | Stmt::LoadGlobalLds { .. } => {
            e.dma = true;
            e.global_read = true;
            e.lds_write = true;
            (ResClass::Dma, vec![], false)
        }
        Stmt::FenceRelease { .. } | Stmt::FenceAcquire { .. } => {
            e.fence = true;
            (ResClass::Sync, vec![], false)
        }
        Stmt::StoreLdsB32 { .. } | Stmt::StoreLdsB128 { .. } | Stmt::StoreLdsB64 { .. } => {
            e.lds_write = true;
            (ResClass::LdsWrite, vec![], false)
        }
        Stmt::WaitLds { count } => {
            e.wait_lds = Some(*count);
            (ResClass::Sync, vec![], false)
        }
        Stmt::WaitVm { count } => {
            e.wait_vm = Some(*count);
            (ResClass::Sync, vec![], false)
        }
        Stmt::Barrier => {
            e.barrier = true;
            (ResClass::Sync, vec![], false)
        }
        Stmt::SetPrio { .. } | Stmt::SchedGroup { .. } => {
            e.directive = true;
            (ResClass::Sync, vec![], false)
        }
        Stmt::SetVar { var, .. } => (ResClass::Valu, vec![*var], false),
        Stmt::VarScaleF32 { var, .. } | Stmt::VarScalePk { var, .. } | Stmt::VarAddPk { var, .. } | Stmt::VarScaleLane { var, .. } | Stmt::VarSetLane { var, .. } => {
            (ResClass::Valu, vec![*var], true)
        }
        Stmt::VarMfmaF16Z { var, .. } => (ResClass::Mfma { passes: 8 }, vec![*var], false),
        Stmt::VarMfmaScale { var, .. } | Stmt::VarMfmaFp8 { var, .. } => (ResClass::Mfma { passes: 16 }, vec![*var], true),
        Stmt::VarMfmaScaleC { var, .. } | Stmt::VarMfmaFp8C { var, .. } => (ResClass::Mfma { passes: 16 }, vec![*var], false),
        Stmt::VarMfmaFp8Z { var, .. } => (ResClass::Mfma { passes: 16 }, vec![*var], false),
        Stmt::VarSmfmac { var, .. } => (ResClass::Mfma { passes: 8 }, vec![*var], true),
        Stmt::VarMfmaScale16 { var, .. } => (ResClass::Mfma { passes: 8 }, vec![*var], true),
        Stmt::VarAsm { var, .. } => {
            e.asm = true;
            (ResClass::Asm, vec![*var], true)
        }
        Stmt::VarAsmN { vars, .. } => {
            e.asm = true;
            (ResClass::Asm, vars.clone(), true)
        }
        Stmt::BeginLoop { vars, .. } => {
            e.branch = true;
            (ResClass::Region, vars.clone(), false)
        }
        Stmt::BreakIf { .. } | Stmt::BeginIf { .. } | Stmt::Else | Stmt::EndIf | Stmt::EndLoop => {
            e.branch = true;
            (ResClass::Region, vec![], false)
        }
    };
    (c, e, w, rmw)
}

fn push_unique<T: PartialEq + Copy>(v: &mut Vec<T>, x: T) {
    if !v.contains(&x) {
        v.push(x);
    }
}

/// Facts of one (non-region) statement. `tys`: the function's value types (`Function::value_tys`).
pub fn stmt_facts(st: &Stmt, tys: &[DType]) -> Facts {
    let (class, eff, writes, rmw) = stmt_class(st);
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    let mut vars = Vec::new();
    let mut s = st.clone();
    if let Stmt::Let { dst, .. } = st {
        defs.push(*dst);
    }
    s.ids(
        &mut |v| {
            if !defs.contains(v) {
                push_unique(&mut uses, *v);
            }
        },
        &mut |r| push_unique(&mut vars, *r),
    );
    // the written variable is mentioned by the statement itself: it is a READ only for read-modify-write forms
    let var_reads: Vec<VarId> = vars.iter().copied().filter(|r| rmw || !writes.contains(r)).collect();
    let footprint = defs.iter().map(|d| tys.get(*d as usize).map_or(0, |t| dwords(*t))).sum();
    Facts { class, defs, uses, var_reads, var_writes: writes, eff, tile: None, footprint }
}

/// Facts of a node: a statement, or a region taken as a whole (the union of its body; values defined inside are its defs, so a use
/// inside of a value defined inside is not a use of the node).
pub fn node_facts(n: &Node, tys: &[DType]) -> Facts {
    match n {
        Node::S(st) => stmt_facts(st, tys),
        _ => {
            let mut stmts = Vec::new();
            flatten_into(std::slice::from_ref(n), &mut stmts);
            let mut f = Facts { class: ResClass::Region, defs: vec![], uses: vec![], var_reads: vec![], var_writes: vec![], eff: Effects::default(), tile: None, footprint: 0 };
            let mut all_uses = Vec::new();
            for st in &stmts {
                let s = stmt_facts(st, tys);
                f.defs.extend(s.defs.iter().copied());
                for u in s.uses {
                    push_unique(&mut all_uses, u);
                }
                for r in s.var_reads {
                    push_unique(&mut f.var_reads, r);
                }
                for w in s.var_writes {
                    push_unique(&mut f.var_writes, w);
                }
                f.eff.union(&s.eff);
                f.footprint += s.footprint;
            }
            f.uses = all_uses.into_iter().filter(|u| !f.defs.contains(u)).collect();
            if let Node::Region { tile, slot, .. } = n {
                f.tile = Some((tile.clone(), *slot));
            }
            // a tile region's / stage's own markers are structure, not an anchor against the work around it: keep `marker` only when the body has one
            if matches!(n, Node::Region { .. } | Node::Stage { .. }) {
                let mut inner = Vec::new();
                match n {
                    Node::Region { body, .. } | Node::Stage { body, .. } => flatten_into(body, &mut inner),
                    _ => {}
                }
                f.eff.marker = inner.iter().any(|s| stmt_class(s).1.marker);
            }
            f
        }
    }
}

/// Provably disjoint LDS footprints: two tile regions of the same tile with different constant slots.
fn lds_disjoint(a: &Facts, b: &Facts) -> bool {
    match (&a.tile, &b.tile) {
        (Some((ta, Opnd::I32(sa))), Some((tb, Opnd::I32(sb)))) => ta == tb && sa != sb,
        _ => false,
    }
}

/// Must `a` stay before `b` (`a` precedes `b` in the original order)? The legality rule of reordering captured statements:
///  * data: `b` uses a value `a` defines; loop variables: write->read, read->write, write->write on the same variable;
///  * anchors (barriers, waits, fences, markers, directives, asm) keep their order with every statement that has an effect and with each other;
///    a pure statement may cross an anchor (the compiler re-inserts the waits it needs, and barriers order memory, not ALU work);
///  * memory: LDS write vs LDS read/write (DMA writes LDS) unless provably disjoint slots of one tile; global write vs global read/write;
///  * branch regions (if / loop) are ordered against every effectful statement (their bodies are not inspected for disjointness).
pub fn must_order(a: &Facts, b: &Facts) -> bool {
    if b.uses.iter().any(|u| a.defs.contains(u)) {
        return true;
    }
    let hits = |x: &[VarId], y: &[VarId]| x.iter().any(|v| y.contains(v));
    if hits(&a.var_writes, &b.var_reads) || hits(&a.var_reads, &b.var_writes) || hits(&a.var_writes, &b.var_writes) {
        return true;
    }
    let (ea, eb) = (&a.eff, &b.eff);
    let effectful = |e: &Effects| e.memory() || e.anchor() || e.branch;
    if (ea.anchor() && effectful(eb)) || (eb.anchor() && effectful(ea)) {
        return true;
    }
    if (ea.branch && effectful(eb)) || (eb.branch && effectful(ea)) {
        return true;
    }
    let lds = (ea.lds_write && (eb.lds_read || eb.lds_write)) || (eb.lds_write && ea.lds_read);
    if lds && !lds_disjoint(a, b) {
        return true;
    }
    (ea.global_write && (eb.global_read || eb.global_write)) || (eb.global_write && ea.global_read)
}

/// Statements captured by `FnBuilder::capture` (top-level nodes in emission order; regions opened and closed inside stay one node).
#[derive(Clone, Debug, Default)]
pub struct Captured {
    pub nodes: Vec<Node>,
}

impl Captured {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    /// Facts of every top-level node (`FnBuilder::facts` adds the variables read through register views; lowering keeps views SSA-correct
    /// regardless: `stale_view_roots`).
    pub fn facts(&self, f: &Function) -> Vec<Facts> {
        self.nodes.iter().map(|n| node_facts(n, &f.value_tys)).collect()
    }
    /// Facts of the whole capture as one unit (defs / uses / variables / effects / footprint of all its nodes).
    pub fn summary(&self, f: &Function) -> Facts {
        let mut s = Facts { class: ResClass::Region, defs: vec![], uses: vec![], var_reads: vec![], var_writes: vec![], eff: Effects::default(), tile: None, footprint: 0 };
        let mut uses = Vec::new();
        for n in &self.nodes {
            let x = node_facts(n, &f.value_tys);
            s.defs.extend(x.defs.iter().copied());
            for u in x.uses {
                push_unique(&mut uses, u);
            }
            for r in x.var_reads {
                push_unique(&mut s.var_reads, r);
            }
            for w in x.var_writes {
                push_unique(&mut s.var_writes, w);
            }
            s.eff.union(&x.eff);
            s.footprint += x.footprint;
        }
        s.uses = uses.into_iter().filter(|u| !s.defs.contains(u)).collect();
        s
    }
    /// The order constraints between the top-level nodes: `(i, j)` with `i < j` when node `i` must stay before node `j`.
    pub fn order_edges(&self, f: &Function) -> Vec<(usize, usize)> {
        let fs = self.facts(f);
        let mut e = Vec::new();
        for j in 0..fs.len() {
            for i in 0..j {
                if must_order(&fs[i], &fs[j]) {
                    e.push((i, j));
                }
            }
        }
        e
    }
    /// Index ranges of top-level nodes that must not be split: a weave-region `RegionMark` begin .. its end (inclusive); every other node is
    /// its own group (regions are already single nodes).
    pub fn groups(&self) -> Vec<std::ops::Range<usize>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.nodes.len() {
            if let Node::S(Stmt::RegionMark { begin: true, .. }) = &self.nodes[i] {
                let mut depth = 0i32;
                let mut j = i;
                while j < self.nodes.len() {
                    if let Node::S(Stmt::RegionMark { begin, .. }) = &self.nodes[j] {
                        depth += if *begin { 1 } else { -1 };
                        if depth == 0 {
                            break;
                        }
                    }
                    j += 1;
                }
                let end = (j + 1).min(self.nodes.len());
                out.push(i..end);
                i = end;
            } else {
                out.push(i..i + 1);
                i += 1;
            }
        }
        out
    }
}

/// The operand a register-VIEW op reinterprets (no instruction: the result is the operand's register or a sub-register of it).
pub fn view_source(op: &Op) -> Option<Opnd> {
    match op {
        Op::VarSlice8 { var, .. } => Some(Opnd::Var(*var)),
        Op::Frag2Lane { frag, .. } | Op::Frag4Lane { frag, .. } | Op::Frag16Lane { frag, .. } => Some(*frag),
        Op::BitcastF32(a) | Op::BitcastI32(a) => Some(*a),
        _ => None,
    }
}

/// The values a statement reads (every ValueId it mentions except a `Let`'s destination).
pub fn stmt_value_uses(st: &Stmt) -> Vec<ValueId> {
    let mut out = Vec::new();
    let dst = if let Stmt::Let { dst, .. } = st { Some(*dst) } else { None };
    let mut s = st.clone();
    s.ids(
        &mut |v| {
            if Some(*v) != dst {
                push_unique(&mut out, *v);
            }
        },
        &mut |_| {},
    );
    out
}

/// value -> (loop variable, root view value) for every value that is a register view of a loop variable (directly, or a view of a view).
pub fn var_views<'a>(stmts: impl IntoIterator<Item = &'a Stmt>) -> std::collections::HashMap<ValueId, (VarId, ValueId)> {
    let mut m: std::collections::HashMap<ValueId, (VarId, ValueId)> = std::collections::HashMap::new();
    for st in stmts {
        if let Stmt::Let { dst, op } = st {
            match view_source(op) {
                Some(Opnd::Var(v)) => {
                    m.insert(*dst, (v, *dst));
                }
                Some(Opnd::Val(x)) => {
                    if let Some(&(v, root)) = m.get(&x) {
                        m.insert(*dst, (v, root));
                    }
                }
                _ => {}
            }
        }
    }
    m
}

/// The root views (`var_views`) that must be COPIED at their definition for SSA semantics: some use of the view (or of a view derived from it)
/// executes after a write of its loop variable that follows the view's definition — later in program order, or anywhere in a loop that
/// contains the use but not the definition (the next iteration). Program order is the flat order (a write in one `if` arm and a use in the
/// other is counted conservatively). Every other view stays a zero-cost alias.
pub fn stale_view_roots(f: &Function) -> std::collections::HashSet<ValueId> {
    let stmts = f.stmts();
    let views = var_views(stmts.iter());
    let mut out = std::collections::HashSet::new();
    if views.is_empty() {
        return out;
    }
    // loop nesting of every statement (ids of the enclosing loops)
    let mut loops: Vec<Vec<usize>> = Vec::with_capacity(stmts.len());
    let mut stack: Vec<usize> = Vec::new();
    let mut next = 0usize;
    for st in stmts {
        if let Stmt::EndLoop = st {
            stack.pop();
        }
        loops.push(stack.clone());
        if let Stmt::BeginLoop { .. } = st {
            stack.push(next);
            next += 1;
        }
    }
    let mut def_pos: std::collections::HashMap<ValueId, usize> = std::collections::HashMap::new();
    let mut writes: std::collections::HashMap<VarId, Vec<usize>> = std::collections::HashMap::new();
    for (i, st) in stmts.iter().enumerate() {
        if let Stmt::Let { dst, .. } = st {
            def_pos.insert(*dst, i);
        }
        for w in stmt_class(st).2 {
            if !matches!(st, Stmt::BeginLoop { .. }) {
                writes.entry(w).or_default().push(i);
            }
        }
    }
    for (r, st) in stmts.iter().enumerate() {
        if matches!(st, Stmt::Let { op, .. } if view_source(op).is_some()) {
            continue; // forwarding a view is not a read
        }
        for u in stmt_value_uses(st) {
            let Some(&(var, root)) = views.get(&u) else { continue };
            if out.contains(&root) {
                continue;
            }
            let Some(&p) = def_pos.get(&root) else { continue };
            let stale = writes.get(&var).map_or(false, |ws| {
                ws.iter().any(|&q| q > p && (q < r || loops[q].iter().any(|l| loops[r].contains(l) && !loops[p].contains(l))))
            });
            if stale {
                out.insert(root);
            }
        }
    }
    out
}
