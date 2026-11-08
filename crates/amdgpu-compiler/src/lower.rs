//! Lowering: typed DSL IR -> virtual-register MIR.
//!
//! The lowering is deliberately conservative:
//! * program order is preserved (no scheduling yet),
//! * every immediate is materialised into a register except where the ISA
//!   allows an in-line literal (shift amounts, 64-bit adds, SALU),
//! * `s_waitcnt` is inserted before the first consumer of a pending load, so
//!   the emitted code is safe without relying on hardware scoreboards for VMEM,
//! * loop-carried variables become plain virtual registers, so no phi
//!   elimination is needed downstream.

use std::collections::{HashMap, HashSet};

use amdgpu_dsl::{DType, Function, Node, Op, Opnd, RegionKind, Stmt, ValueId};
use crate::mir::*;

// Compiler optimisations that reduce dynamic VALU count (accuracy-preserving: same values, same
// arithmetic). Selected per kernel profile (`lib.rs`, `with_opts`); DEFAULT `dce,cmpfold`.
//  * `imm`     - constants become in-line / literal operands (VOP2 src0, shift amounts, cndmask false arm)
//                instead of `v_mov` into a register;
//  * `sgpr`    - scalar (SGPR) values are used directly as VALU source operands instead of a `v_mov` copy;
//  * `shl`     - multiplication by a power-of-two constant becomes a shift;
//  * `scalarize` - wave-uniform integer arithmetic (workgroup ids, kernarg values, constants and loop counters /
//                 toggles derived from them with add/mul/shl/lshr/and/xor) is computed with SALU instructions in SGPRs;
//                 a value needed by a VALU instruction is copied with one `v_mov` (cached per region);
//  * `addrfold` - an address `(base + k1) + (y + k2)` used once is formed as `base + y` with `k1+k2` in the DS
//                 offset field, so the `base + k` variants no longer stay live in VGPRs;
//  * `saddr`   - global loads whose address is `kernarg_ptr + vector offset` use the scalar-base form (no 64-bit add);
//  * `dce`     - dead address / clamp chains (`v_add_co`/`v_addc`, `v_cmp`+`v_cndmask`, pointer adds) are removed;
//  * `cmpfold` - `select(cmp(x,y), ..)` re-issues the compare instead of the 0/1 materialisation +
//                `v_cmp_ne`, and the dead materialisation is removed.
thread_local! {
    static OVERRIDE: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` with the lowering options forced to `opts` on this thread (tests; the environment is not consulted).
pub fn with_opts<R>(opts: &[&str], f: impl FnOnce() -> R) -> R {
    OVERRIDE.with(|o| *o.borrow_mut() = Some(opts.iter().map(|s| s.to_string()).collect()));
    let r = f();
    OVERRIDE.with(|o| *o.borrow_mut() = None);
    r
}

/// The lowering options in effect on this thread (override, environment or default).
pub fn current_opts() -> Vec<String> {
    if let Some(v) = OVERRIDE.with(|o| o.borrow().clone()) {
        return v;
    }
    vec!["dce".to_string(), "cmpfold".to_string()]
}

/// `lower` with extra options on top of the current ones (used by the register-pressure retries).
pub fn lower_with_extra(f: &Function, extra: &[&str]) -> Result<MFunction, String> {
    let mut v = current_opts();
    v.extend(extra.iter().map(|s| s.to_string()));
    let refs: Vec<&str> = v.iter().map(|s| s.as_str()).collect();
    with_opts(&refs, || lower(f))
}

pub fn opt_on(name: &str) -> bool {
    if let Some(v) = OVERRIDE.with(|o| o.borrow().clone()) {
        return v.iter().any(|x| x == name);
    }
    // default: the optimisations measured to help (value-preserving, bit-exact on all shapes)
    name == "dce" || name == "cmpfold"
}

/// Is `bits` an in-line constant of the ISA (integers -16..64 and the f32 constants 0, +-0.5, +-1, +-2, +-4, 1/(2pi))?
/// Literals are not allowed where the instruction also reads the constant bus (e.g. `v_cndmask` reads vcc).
fn is_inline_const(bits: i64) -> bool {
    let u = bits as i32 as u32;
    (-16..=64).contains(&bits) || matches!(u, 0x3f00_0000 | 0xbf00_0000 | 0x3f80_0000 | 0xbf80_0000 | 0x4000_0000 | 0xc000_0000 | 0x4080_0000 | 0xc080_0000 | 0x3e22_f983)
}

/// May `bits` be the constant source operand of `op` (placed in src0)?
fn imm_ok(op: MOp, bits: i64) -> bool {
    match op {
        MOp::VMulLoU32 => (-16..=64).contains(&bits), // VOP3 on gfx9: in-line constants only
        MOp::VAddU32 | MOp::VAddF32 | MOp::VSubF32 | MOp::VMulF32 | MOp::VMaxF32 | MOp::VAndB32 | MOp::VXorB32 | MOp::VOrB32 | MOp::VMinI32 => true,
        _ => false,
    }
}

pub fn lower(f: &Function) -> Result<MFunction, String> {
    let errs = amdgpu_dsl::verify(f);
    if !errs.is_empty() {
        return Err(format!(
            "IR verification failed:\n  {}\nIR:\n{}",
            errs.join("\n  "),
            amdgpu_dsl::print::print_dsl(f)
        ));
    }
    let mut lw = Lowerer::new(f);
    if opt_on("scalarize") {
        let (u, v) = uniform_analysis(f);
        lw.uniform = u;
        lw.uvars = v;
    }
    lw.run(f)?;
    let mut m = lw.finish()?;
    if opt_on("kabatch") {
        batch_kernarg_loads(&mut m);
    }
    Ok(m)
}

/// `kabatch`: kernarg loads of the entry block that fall into the same 16-byte-aligned window (inside the kernarg segment) become ONE
/// `s_load_dwordx4` into an SGPR quad; the original destinations become views of it (`m.alias`). Issued at the first load's position (earlier
/// is always valid: the kernarg segment is read-only and every use already waits for `lgkmcnt(0)`).
fn batch_kernarg_loads(m: &mut MFunction) {
    let Some(b0) = m.blocks.first() else { return };
    let size = m.kernarg_size as i64;
    // (inst index, byte offset, dwords, dst)
    let loads: Vec<(usize, i64, i64, VReg)> = b0
        .insts
        .iter()
        .enumerate()
        .filter_map(|(i, x)| match (x.op, x.dst) {
            (MOp::SLoadDword, Some(d)) => Some((i, x.imm, 1, d)),
            (MOp::SLoadDwordx2, Some(d)) => Some((i, x.imm, 2, d)),
            _ => None,
        })
        .collect();
    let mut windows: Vec<i64> = loads.iter().map(|l| l.1 / 16).collect();
    windows.sort();
    windows.dedup();
    let mut remove: Vec<usize> = Vec::new();
    let mut insert: Vec<(usize, MInst)> = Vec::new();
    for w in windows {
        let lo = 16 * w;
        if lo + 16 > size {
            continue;
        }
        let group: Vec<&(usize, i64, i64, VReg)> = loads.iter().filter(|l| l.1 >= lo && l.1 + 4 * l.2 <= lo + 16 && l.1 % 4 == 0).collect();
        if group.len() < 2 || group.iter().any(|l| l.2 == 2 && (l.1 - lo) % 8 != 0) {
            continue;
        }
        let q = m.vregs.len() as VReg;
        m.vregs.push(RClass::S4);
        for l in &group {
            let off = ((l.1 - lo) / 4) as u8;
            // views of an original destination follow it into the quad
            let views: Vec<VReg> = m.alias.iter().filter(|(_, (t, _))| *t == l.3).map(|(a, _)| *a).collect();
            for v in views {
                let (_, o) = m.alias[&v];
                m.alias.insert(v, (q, off + o));
            }
            m.alias.insert(l.3, (q, off));
            remove.push(l.0);
        }
        let first = group.iter().map(|l| l.0).min().unwrap();
        insert.push((first, MInst::new(MOp::SLoadDwordx4).def(q).imm(lo)));
    }
    if insert.is_empty() {
        return;
    }
    let old = std::mem::take(&mut m.blocks[0].insts);
    let mut out = Vec::with_capacity(old.len());
    for (i, x) in old.into_iter().enumerate() {
        for (at, ins) in &insert {
            if *at == i {
                out.push(ins.clone());
            }
        }
        if !remove.contains(&i) {
            out.push(x);
        }
    }
    m.blocks[0].insts = out;
}

struct Lowerer {
    mir: MFunction,
    val: HashMap<ValueId, VReg>,
    /// Values that were converted from a scalar register into a vector one.
    converted: HashMap<ValueId, VReg>,
    vars: Vec<VReg>,
    consts: HashMap<i64, VReg>,
    /// Constant / conversion caches saved at control-flow entry; anything materialised
    /// inside a branch or loop body must not be assumed by code after it or on the sibling path.
    scope_snaps: Vec<(HashMap<i64, VReg>, HashMap<ValueId, VReg>)>,
    pending_vmem: Vec<VReg>,
    pending_smem: Vec<VReg>,
    pending_lds: Vec<VReg>,
    break_stack: Vec<(u32, u32)>,
    if_stack: Vec<(u32, u32)>,
    /// Pending-load sets at if entry (vmem, smem, lds) and whether the then-branch changed them.
    if_snaps: Vec<((Vec<VReg>, Vec<VReg>, Vec<VReg>), bool)>,
    /// `v_readfirstlane` results of the current block (LDS-DMA bases), keyed by the vector source register.
    rfl_cache: HashMap<VReg, VReg>,
    cur: usize,
    next_label: u32,
    defs: HashMap<ValueId, Op>,
    /// How many times each value is used as an operand anywhere in the function.
    use_counts: HashMap<ValueId, usize>,
    /// every operand use of every value (`amdgpu_dsl::stmt_operands`: exhaustive), for rewrites that must see all uses (`var_set_lane`)
    exact_uses: HashMap<ValueId, usize>,
    /// Cache of sub-register views: (tuple, dword offset, class) -> alias vreg.
    alias_cache: HashMap<(VReg, u8, RClass), VReg>,
    /// root register views of loop variables that must be copied at their definition (a use would otherwise see a later write of the variable: `facts::stale_view_roots`)
    view_copy: std::collections::HashSet<ValueId>,
    /// Labels of else-blocks that were actually started (the others are synthesised at `endif`).
    else_started: HashSet<u32>,
    /// AGPR requests from the IR (resolved to vregs in inish).
    agpr_marks: Vec<amdgpu_dsl::AgprMark>,
    /// register plan of the kernel (resolved to vregs in `finish`) and the stack of open weave / stage regions (ids into `mir.reg_hints.regions`)
    reg_plan: Vec<amdgpu_dsl::RegMark>,
    region_stack: Vec<u16>,
    /// `scalarize`: DSL values / loop variables that are wave-uniform integers (computed in SGPRs).
    uniform: HashSet<ValueId>,
    uvars: HashSet<amdgpu_dsl::VarId>,
    /// `addrfold`: integer values held as `reg + constant` without an instruction for the constant (value -> (reg, k)).
    lazy: HashMap<ValueId, (VReg, i64)>,
}

impl Lowerer {
    fn new(f: &Function) -> Self {
        let mut mir = MFunction {
            name: f.name.clone(),
            args: f
                .args
                .iter()
                .map(|a| MArg {
                    name: a.name.clone(),
                    dsl_ty: a.ty,
                    offset: a.kernarg_offset,
                    size: a.size,
                })
                .collect(),
            kernarg_size: f.kernarg_size,
            workgroup_size: f.workgroup_size,
            lds_bytes: f.lds_bytes,
            blocks: Vec::new(),
            vregs: Vec::new(),
            loops: Vec::new(),
            alias: HashMap::new(),
            elide_fallthrough: false,
            edge_pad: opt_on("edgepad"),
            agpr_pin: Default::default(), reg_hints: Default::default(),
        };
        mir.blocks.push(Block::new(0));

        let mut defs = HashMap::new();
        let mut use_counts: HashMap<ValueId, usize> = HashMap::new();
        let mut count = |o: &Opnd| {
            if let Opnd::Val(v) = o {
                *use_counts.entry(*v).or_insert(0) += 1;
            }
        };
        for st in f.stmts() {
            match st {
                Stmt::Let { dst, op } => {
                    defs.insert(*dst, op.clone());
                    for o in op.operands() {
                        count(&o);
                    }
                }
                Stmt::StoreU16 { base, value, .. } | Stmt::StoreF32 { base, value, .. } | Stmt::StoreF32x4 { base, value, .. } | Stmt::StoreF32x2 { base, value, .. } => {
                    count(base);
                    count(value);
                }
                Stmt::StoreLdsB32 { addr, value } | Stmt::StoreLdsB128 { addr, value } | Stmt::StoreLdsB64 { addr, value } => {
                    count(addr);
                    count(value);
                }
                Stmt::BufferStore { rsrc, voff, soff, value, .. } => {
                    count(rsrc);
                    for o in voff.iter() {
                        count(o);
                    }
                    count(soff);
                    count(value);
                }
                Stmt::BufferLoadLds { rsrc, voff, soff, lds_addr, .. } => {
                    count(rsrc);
                    for o in voff.iter() {
                        count(o);
                    }
                    count(soff);
                    count(lds_addr);
                }
                Stmt::SetVar { value, .. } => count(value),
                Stmt::StoreGlobalN { .. } | Stmt::BufferStoreN { .. } | Stmt::StoreLdsN { .. } | Stmt::StoreLds2 { .. } | Stmt::VarMfma { .. } => {
                    for o in amdgpu_dsl::stmt_operands(st) {
                        count(&o);
                    }
                }
                Stmt::BeginLoop { init, .. } => {
                    for o in init {
                        count(o);
                    }
                }
                Stmt::BreakIf { cond } | Stmt::BeginIf { cond } => count(cond),
                _ => {}
            }
        }

        Lowerer {
            mir,
            val: HashMap::new(),
            converted: HashMap::new(),
            vars: Vec::new(),
            consts: HashMap::new(),
            scope_snaps: Vec::new(),
            pending_vmem: Vec::new(),
            pending_smem: Vec::new(),
            pending_lds: Vec::new(),
            break_stack: Vec::new(),
            if_stack: Vec::new(),
            if_snaps: Vec::new(),
            rfl_cache: HashMap::new(),
            cur: 0,
            next_label: 1,
            defs,
            use_counts,
            exact_uses: {
                let mut m: HashMap<ValueId, usize> = HashMap::new();
                for st in f.stmts() {
                    for o in amdgpu_dsl::stmt_operands(st) {
                        if let Opnd::Val(v) = o {
                            *m.entry(v).or_insert(0) += 1;
                        }
                    }
                }
                m
            },
            alias_cache: HashMap::new(),
            view_copy: amdgpu_dsl::facts::stale_view_roots(f),
            else_started: HashSet::new(),
            agpr_marks: f.agpr.clone(),
            reg_plan: f.reg_plan.clone(),
            region_stack: Vec::new(),
            uniform: HashSet::new(),
            uvars: HashSet::new(),
            lazy: HashMap::new(),
        }
    }

    fn finish(mut self) -> Result<MFunction, String> {
        if !self.pending_vmem.is_empty() {
            self.push(MInst::new(MOp::SWaitVm).auto());
        }
        if !self.pending_smem.is_empty() || !self.pending_lds.is_empty() {
            self.push(MInst::new(MOp::SWaitLgkm).imm(0).auto());
        }
        self.set_term(Term::Endpgm);
        // AGPR requests: record the canonical vregs now (before the clean-up passes can rename anything)
        for mk in std::mem::take(&mut self.agpr_marks) {
            let r = match mk.target {
                amdgpu_dsl::AgprTarget::Var(v) => self.vars.get(v.0 as usize).copied(),
                amdgpu_dsl::AgprTarget::Val(v) => self.val.get(&v).copied(),
            };
            if let Some(r) = r {
                let c = self.mir.canon(r);
                // an exact pin is never weakened by a later ny request on the same register (e.g. `agpr(load)` coalesced into a `pin_agpr_at` variable)
                let want = mk.pin.unwrap_or(crate::agpr::AGPR_ANY);
                match self.mir.agpr_pin.get(&c) {
                    Some(old) if *old != crate::agpr::AGPR_ANY && want == crate::agpr::AGPR_ANY => {}
                    Some(old) if *old != crate::agpr::AGPR_ANY && want != *old => return Err(format!("conflicting exact AGPR pins a{old} and a{want} on one register")),
                    _ => {
                        self.mir.agpr_pin.insert(c, want);
                    }
                }
            }
        }
        self.resolve_reg_plan();
        if opt_on("cmpfold") {
            fold_select_cmp(&mut self.mir);
            dce_cmp_pairs(&mut self.mir);
        }
        // Always: an unused load must not stay (its destination would get a zero-length live range while the load is
        // still in flight and could be reused by a live value: found with sv+dma+kh0 under `dce`).
        dce_full(&mut self.mir, true);
        if opt_on("dce") {
            dce_full(&mut self.mir, false);
        }
        // rematerialise loop-invariant constants next to their loop uses (register pressure; `remat1/2/3` = at most 1/2/any
        // number of loop blocks using the constant)
        let spill_n: usize = ["ldsspill8", "ldsspill16", "ldsspill24", "ldsspill32", "ldsspill48", "ldsspill64"].iter().find(|n| opt_on(n)).map(|n| n.trim_start_matches("ldsspill").parse().unwrap()).unwrap_or(0);
        if opt_on("remat4") {
            remat_exprs(&mut self.mir);
        } else {
            for (name, k) in [("remat1", 1usize), ("remat2", 2), ("remat3", usize::MAX)] {
                if opt_on(name) {
                    remat_consts(&mut self.mir, k);
                    break;
                }
            }
        }
        if spill_n > 0 {
            let (n, bytes) = lds_spill(&mut self.mir, spill_n);
            if n > 0 {
                eprintln!("lds-spill: {n} loop-invariant values moved to LDS, {} bytes per value per workgroup, {bytes} bytes total", bytes / (n.max(1) as u32));
                crate::sched::waits::insert_waits(&mut self.mir);
            }
        }
        let renames = coalesce(&mut self.mir);
        if self.mir.reg_hints.active() {
            // a planned value whose producer was coalesced into another register (e.g. a loop variable) is planned as that register
            let rn = |r: VReg| -> VReg {
                let mut r = r;
                while let Some((_, d)) = renames.iter().find(|(s, _)| *s == r) {
                    r = *d;
                }
                r
            };
            let h = &mut self.mir.reg_hints;
            for (r, _) in h.pins_v.iter_mut() {
                *r = rn(*r);
            }
            for g in h.share.iter_mut().chain(h.separate.iter_mut()).chain(h.rings.iter_mut().map(|x| &mut x.1)) {
                for r in g.iter_mut() {
                    *r = rn(*r);
                }
            }
        }
        dce(&mut self.mir);
        Ok(self.mir)
    }

    /// The DSL register plan at vreg level (`MFunction::reg_hints`): every RegRef becomes the canonical vreg of the value / variable.
    fn resolve_reg_plan(&mut self) {
        use amdgpu_dsl::{RegMark, RegRef};
        let plan = std::mem::take(&mut self.reg_plan);
        if plan.is_empty() {
            return;
        }
        let mut vals: Vec<(ValueId, VReg)> = self.val.iter().map(|(v, r)| (*v, *r)).collect();
        vals.sort_unstable();
        for (v, r) in vals {
            let c = self.mir.canon(r);
            self.mir.reg_hints.names.entry(c).or_insert_with(|| format!("%v{v}"));
            self.mir.reg_hints.by_name.insert(format!("%v{v}"), c);
        }
        for (i, r) in self.vars.iter().enumerate() {
            let c = self.mir.canon(*r);
            self.mir.reg_hints.names.insert(c, format!("%r{i}"));
            self.mir.reg_hints.by_name.insert(format!("%r{i}"), c);
        }
        let res = |s: &Self, r: &RegRef| -> Option<VReg> {
            match r {
                RegRef::Val(v) => s.val.get(v).map(|x| s.mir.canon(*x)),
                RegRef::Var(v) => s.vars.get(v.0 as usize).map(|x| s.mir.canon(*x)),
            }
        };
        let mut unresolved = Vec::new();
        let group = |s: &Self, rs: &[RegRef], unresolved: &mut Vec<String>| -> Vec<VReg> {
            rs.iter()
                .filter_map(|r| {
                    let x = res(s, r);
                    if x.is_none() {
                        unresolved.push(format!("{r:?}"));
                    }
                    x
                })
                .collect()
        };
        for mk in &plan {
            match mk {
                RegMark::PinV { r, base } => match res(self, r) {
                    Some(x) => self.mir.reg_hints.pins_v.push((x, *base)),
                    None => unresolved.push(format!("{r:?}")),
                },
                RegMark::Share(rs) => {
                    let g = group(self, rs, &mut unresolved);
                    self.mir.reg_hints.share.push(g);
                }
                RegMark::Separate(rs) => {
                    let g = group(self, rs, &mut unresolved);
                    self.mir.reg_hints.separate.push(g);
                }
                RegMark::Cap { region, vgprs, agprs } => self.mir.reg_hints.caps.push((region.clone(), *vgprs, *agprs)),
                RegMark::Ring { name, bufs } => {
                    let g = group(self, bufs, &mut unresolved);
                    self.mir.reg_hints.rings.push((name.clone(), g));
                }
            }
        }
        self.mir.reg_hints.unresolved = unresolved;
    }

    /// Is the operand a provably all-zero fragment (`frag_zero`)? An MFMA value form with such a C uses the inline-0 src2 encoding.
    fn is_zero_frag(&self, o: &Opnd) -> bool {
        matches!(o, Opnd::Val(v) if matches!(self.defs.get(v), Some(Op::FragZero { .. })))
    }

    fn fresh_label(&mut self) -> u32 {
        let l = self.next_label;
        self.next_label += 1;
        l
    }

    fn new_vreg(&mut self, c: RClass) -> VReg {
        let r = self.mir.vregs.len() as VReg;
        self.mir.vregs.push(c);
        r
    }

    fn push(&mut self, mut inst: MInst) {
        inst.region = self.region_stack.last().copied().unwrap_or(0);
        self.mir.blocks[self.cur].insts.push(inst);
    }

    fn set_term(&mut self, t: Term) {
        self.mir.blocks[self.cur].term = t;
    }

    fn start_block(&mut self, label: u32) {
        self.rfl_cache.clear();
        self.mir.blocks.push(Block::new(label));
        self.cur = self.mir.blocks.len() - 1;
    }

    /// Drain every outstanding load before control flow diverges or joins: the pending
    /// set is a linear-lowering notion, so a wait emitted on one path must not be assumed
    /// by the code on another.
    fn flush_pending(&mut self) {
        if !self.pending_vmem.is_empty() {
            self.push(MInst::new(MOp::SWaitVm).auto());
            self.pending_vmem.clear();
        }
        if !self.pending_smem.is_empty() || !self.pending_lds.is_empty() {
            self.push(MInst::new(MOp::SWaitLgkm).imm(0).auto());
            self.pending_smem.clear();
            self.pending_lds.clear();
        }
    }
    /// Wait for any pending load feeding `srcs`, then clear the pending set.
    fn touch(&mut self, srcs: &[Src]) {
        // Global loads complete in issue order, so waiting for the newest load that feeds `srcs`
        // only needs `vmcnt(n)` where n is the number of loads issued after it: loads issued later
        // (a prefetch, for instance) stay in flight.
        let mut vm_newest: Option<usize> = None;
        let mut need_lgkm = false;
        let mut lds_newest: Option<usize> = None;
        for s in srcs {
            if let Src::Reg(r) = s {
                let r = &self.mir.canon(*r);
                if let Some(pos) = self.pending_vmem.iter().rposition(|x| x == r) {
                    vm_newest = Some(vm_newest.map_or(pos, |p| p.max(pos)));
                }
                if self.pending_smem.contains(r) || self.pending_lds.contains(r) {
                    need_lgkm = true;
                }
                if let Some(pos) = self.pending_lds.iter().rposition(|x| x == r) {
                    lds_newest = Some(lds_newest.map_or(pos, |p| p.max(pos)));
                }
            }
        }
        // `lgkmn` (opt-in, the GEMM kernels): LDS reads return in issue order: with no scalar load in flight, `lgkmcnt(n)` (n = LDS read INSTRUCTIONS issued after the newest one
        // feeding `srcs`; `pending_lds` holds one entry per instruction) keeps later reads (prefetches) in flight. Untracked LDS writes only make
        // the count conservative (issued earlier: they complete first; issued later: they are not counted in n). The 4-bit counter caps n at 15.
        if need_lgkm && self.pending_smem.is_empty() && opt_on("lgkmn") {
            if let Some(pos) = lds_newest {
                let after = self.pending_lds.len() - 1 - pos;
                if after <= 15 {
                    self.push(MInst::new(MOp::SWaitLgkm).imm(after as i64).auto());
                    self.pending_lds.drain(..=pos);
                    need_lgkm = false;
                }
            }
        }
        if let Some(pos) = vm_newest {
            let after = self.pending_vmem.len() - 1 - pos;
            self.push(MInst::new(MOp::SWaitVm).imm(after.min(63) as i64).auto());
            if after > 63 {
                // beyond the counter range: fall back to draining everything
                self.mir.blocks[self.cur].insts.last_mut().unwrap().imm = 0;
                self.pending_vmem.clear();
            } else {
                self.pending_vmem.drain(..=pos);
            }
        }
        if need_lgkm {
            self.push(MInst::new(MOp::SWaitLgkm).imm(0).auto());
            self.pending_smem.clear();
            self.pending_lds.clear();
        }
    }

    /// A sub-register view of `tuple` at dword `off` (no instruction, no register of its own).
    fn alias_vreg(&mut self, tuple: VReg, off: u8, class: RClass) -> VReg {
        let tuple = self.mir.canon(tuple);
        if let Some(a) = self.alias_cache.get(&(tuple, off, class)) {
            return *a;
        }
        let a = self.new_vreg(class);
        self.mir.alias.insert(a, (tuple, off));
        self.alias_cache.insert((tuple, off, class), a);
        a
    }

    /// Alias of dwords `off..` of `r` (which may itself be an alias) with class `class`.
    fn sub_alias(&mut self, r: VReg, off: u8, class: RClass) -> VReg {
        let (base, o) = self.mir.alias.get(&r).copied().unwrap_or((r, 0));
        self.alias_vreg(base, o + off, class)
    }

    /// Rewrite the `ds_read_b128`s that produced `v` (`LoadLdsB128` / `LoadLdsB128x2`, current block) to write the register of loop variable
    /// `dst` directly. Legal only when no instruction after the first of those reads issues a read of `dst` (the old value) or writes it.
    fn try_coalesce_lds_reload(&mut self, v: ValueId, dst: VReg) -> bool {
        let want = match self.defs.get(&v) {
            Some(Op::LoadLdsB128 { .. }) => (RClass::V4, 1usize),
            Some(Op::LoadLdsB128x2 { .. }) => (RClass::V8, 2usize),
            _ => return false,
        };
        let Some(&d) = self.val.get(&v) else { return false };
        if self.mir.vreg_class(d) != want.0 || self.mir.vreg_class(dst) != want.0 || self.mir.alias.contains_key(&d) || self.mir.alias.contains_key(&dst) {
            return false;
        }
        let insts = &self.mir.blocks[self.cur].insts;
        let prod: Vec<usize> = insts.iter().enumerate().filter(|(_, i)| i.op == MOp::LdsReadB128 && i.dst.map_or(false, |x| self.mir.canon(x) == d)).map(|(k, _)| k).collect();
        if prod.len() != want.1 {
            return false;
        }
        let first = prod[0];
        for (k, i) in insts.iter().enumerate().skip(first) {
            if prod.contains(&k) {
                continue;
            }
            if i.uses().iter().any(|u| self.mir.canon(*u) == dst) || i.dst.map_or(false, |x| self.mir.canon(x) == dst) {
                return false;
            }
        }
        // new destinations (the variable's tuple or its halves)
        let mut news: Vec<(usize, VReg)> = Vec::new();
        for &k in &prod {
            let old = self.mir.blocks[self.cur].insts[k].dst.unwrap();
            let off = self.mir.alias.get(&old).map(|a| a.1).unwrap_or(0);
            let nd = if want.0 == RClass::V4 { dst } else { self.alias_vreg(dst, off, RClass::V4) };
            news.push((k, nd));
        }
        for (k, nd) in news {
            self.mir.blocks[self.cur].insts[k].dst = Some(nd);
        }
        for e in self.pending_lds.iter_mut() {
            if *e == d {
                *e = dst;
            }
        }
        self.val.insert(v, dst);
        true
    }
    /// Make the instruction that produced value `v` write `new_dst` directly (instead of a fresh
    /// register that would then be moved into a fragment). Only for single-use results of plain
    /// ALU ops defined earlier in the current block.
    fn retarget_producer(&mut self, v: ValueId, new_dst: VReg) -> bool {
        self.retarget_producer_inner(v, new_dst)
    }

    fn retarget_producer_inner(&mut self, v: ValueId, new_dst: VReg) -> bool {
        let mut cur = v;
        loop {
            if self.use_counts.get(&cur).copied() != Some(1) {
                return false;
            }
            match self.defs.get(&cur) {
                Some(Op::BitcastF32(Opnd::Val(i))) | Some(Op::BitcastI32(Opnd::Val(i))) => cur = *i,
                Some(Op::MulF32(..))
                | Some(Op::AddF32(..))
                | Some(Op::SubF32(..))
                | Some(Op::AddI32(..))
                | Some(Op::ShlI32(..))
                | Some(Op::XorI32(..))
                | Some(Op::MulI32(..)) => break,
                // wider producer set (`retarget2`): single-use results of any plain ALU / conversion op
                Some(Op::MaxF32(..))
                | Some(Op::Max3F32(..))
                | Some(Op::FmaF32(..))
                | Some(Op::ExpF32(..))
                | Some(Op::Exp2F32(..))
                | Some(Op::Log2F32(..))
                | Some(Op::RcpF32(..))
                | Some(Op::CvtF16ToF32(..))
                | Some(Op::CvtF32ToF16(..))
                | Some(Op::CvtPkFp8x4F32(..))
                | Some(Op::CvtPkBf16F32(..))
                | Some(Op::CvtPknormU16(..))
                | Some(Op::CvtP8x4FromU16(..))
                | Some(Op::CvtPknormP8x4(..))
                | Some(Op::CvtPknormPair(..))
                | Some(Op::PermB32(..))
                | Some(Op::LshrI32(..))
                | Some(Op::AndI32(..))
                | Some(Op::MinI32(..))
                | Some(Op::PkSubU16(..))
                | Some(Op::Select { .. })
                    if opt_on("retarget2") =>
                {
                    break
                }
                _ => return false,
            }
        }
        let r = match self.val.get(&cur) {
            Some(r) => *r,
            None => return false,
        };
        if self.mir.vreg_class(r) != RClass::V1 || self.mir.alias.contains_key(&r) {
            return false;
        }
        let insts = &mut self.mir.blocks[self.cur].insts;
        // every instruction of the producing chain (a partial-write form such as the cvt_pk high half reads and writes the
        // same register): collect first, then rewrite, so a failed attempt changes nothing
        let mut chain: Vec<usize> = Vec::new();
        let mut complete = false;
        for (k, inst) in insts.iter().enumerate().rev() {
            let writes = inst.dst == Some(r);
            let reads = inst.uses().contains(&r);
            if writes {
                chain.push(k);
                if reads {
                    continue;
                }
                complete = true;
                break;
            }
            if reads {
                return false;
            }
        }
        if !complete {
            return false;
        }
        for k in chain {
            let inst = &mut insts[k];
            inst.dst = Some(new_dst);
            for s in inst.srcs.iter_mut() {
                if let Src::Reg(x) = s {
                    if *x == r {
                        *x = new_dst;
                    }
                }
            }
        }
        self.val.insert(cur, new_dst);
        true
    }

    /// Lower a safe-asm block (`amdgpu-safeasm`) into ordinary MIR: ports bind to DSL values, constants join the region's constant cache,
    /// everything else becomes fresh virtual registers; the instructions then go through the normal scheduling / allocation / hazard pipeline.
    fn lower_asm(&mut self, dst: Option<ValueId>, block: &amdgpu_safeasm::AsmBlock, inputs: &[Opnd], inouts: &[VReg]) -> Result<(), String> {
        use amdgpu_safeasm::{AClass, ASrc};
        let rc = |c: AClass| match c {
            AClass::V1 => RClass::V1,
            AClass::V2 => RClass::V2,
            AClass::V4 => RClass::V4,
            AClass::V8 => RClass::V8,
            AClass::V16 => RClass::V16,
            AClass::S1 => RClass::S1,
            AClass::S2 => RClass::S2,
        };
        // belt and braces: the block was verified when it was built, a lowered block is re-verified (it may come from another build)
        block.verify().map_err(|e| format!("asm block `{}` fails the hazard verifier: {}", block.name(), e.join("; ")))?;
        let n_bound = block.inputs().iter().filter(|p| !p.mutable).count();
        let n_mut = block.inputs().iter().filter(|p| p.mutable).count();
        if inputs.len() != n_bound || n_mut != inouts.len() {
            return Err(format!("asm block `{}`: {} inputs bound for {} read-only ports; {} in/out variables bound for {} in/out ports", block.name(), inputs.len(), n_bound, inouts.len(), n_mut));
        }
        let mut inout_iter = inouts.iter();
        let mut map: Vec<Option<VReg>> = vec![None; block.regs().len()];
        let n_before_inputs = self.mir.blocks[self.cur].insts.len();
        let mut bound = inputs.iter();
        for p in block.inputs().iter() {
            if p.mutable {
                map[p.reg as usize] = inout_iter.next().copied();
                continue;
            }
            let o = bound.next().expect("input count checked");
            let r = match p.class {
                AClass::V1 => self.as_v1(*o)?,
                AClass::V2 => self.as_v2(*o)?,
                AClass::V4 => self.as_v4(*o)?,
                AClass::V8 => self.as_v8(*o)?,
                AClass::V16 => self.as_v16(*o)?,
                other => return Err(format!("asm block: input port class {other:?} is not supported")),
            };
            self.touch(&[Src::Reg(r)]);
            map[p.reg as usize] = Some(r);
        }
        if block.is_pinned() {
            // the waits for loads the block's inputs depend on must stay at the START of the pinned run: make them explicit so the wait pass keeps them
            // there instead of re-deriving them in front of the first (possibly mid-run) use
            for i in &mut self.mir.blocks[self.cur].insts[n_before_inputs..] {
                if matches!(i.op, MOp::SWaitVm | MOp::SWaitLgkm) {
                    i.auto = false;
                }
            }
        }
        // constants: a one-dword register written once by `v_mov_b32 imm` is the region's cached constant register
        let mut ndefs = vec![0u32; block.regs().len()];
        for i in block.insts() {
            if let Some(d) = i.dst {
                ndefs[d as usize] += 1;
            }
        }
        let aliased: std::collections::HashSet<u32> = block.alias().iter().map(|a| a.0).collect();
        let mut skip = vec![false; block.insts().len()];
        let is_port = |r: u32| block.inputs().iter().chain(block.outputs().iter()).any(|p| p.reg == r);
        let mut const_of: Vec<Option<(usize, i64)>> = vec![None; block.regs().len()];
        for (k, i) in block.insts().iter().enumerate() {
            if let ("VMovB32", Some(d), true) = (i.op, i.dst, i.srcs.is_empty()) {
                let d = d as usize;
                if ndefs[d] == 1 && block.regs()[d] == AClass::V1 && map[d].is_none() && !aliased.contains(&(d as u32)) && !is_port(d as u32) {
                    const_of[d] = Some((k, i.imm));
                    skip[k] = true;
                }
            }
        }
        // the constant cache removes those moves: the wait-state distances inside the block must still hold without them. If the
        // verifier (which counts every instruction) would reject the stream with the moves removed, keep the moves instead.
        if skip.iter().any(|s| *s) {
            let kept: Vec<amdgpu_safeasm::AInst> = block.insts().iter().enumerate().filter(|(k, _)| !skip[*k]).map(|(_, i)| i.clone()).collect();
            if amdgpu_safeasm::verify_insts(block, kept).is_err() {
                skip.iter_mut().for_each(|s| *s = false);
                const_of.iter_mut().for_each(|c| *c = None);
            }
        }
        for (d, c) in const_of.iter().enumerate() {
            if let Some((_, k)) = c {
                map[d] = Some(self.const_vreg(*k));
            }
        }
        for (i, c) in block.regs().iter().enumerate() {
            if map[i].is_none() && !aliased.contains(&(i as u32)) {
                map[i] = Some(self.new_vreg(rc(*c)));
            }
        }
        for (a, base, off) in block.alias() {
            let b = map[*base as usize].ok_or("asm block: alias of an unmapped register")?;
            map[*a as usize] = Some(self.sub_alias(b, *off, rc(block.regs()[*a as usize])));
        }
        let m = |r: u32| map[r as usize].expect("unmapped asm register");
        let mut n_pushed = 0usize;
        for (k, i) in block.insts().iter().enumerate() {
            if skip[k] {
                continue;
            }
            let op = mop_by_name(i.op).ok_or_else(|| format!("asm block: unknown op `{}`", i.op))?;
            let mut mi = MInst::new(op);
            if let Some(d) = i.dst {
                mi = mi.def(m(d));
            }
            for s in &i.srcs {
                mi = match s {
                    ASrc::Reg(r) => mi.srcreg(m(*r)),
                    ASrc::Imm(v) => mi.src(Src::Imm(*v)),
                };
            }
            let mut mi = mi.imm(i.imm);
            mi.pin = block.is_pinned();
            self.push(mi);
            n_pushed += 1;
        }
        let out = block.outputs().first().ok_or("asm block has no output port")?;
        if let Some(dst) = dst {
            self.val.insert(dst, m(out.reg));
        }
        // F8: the lowered MIR must not write a register the kernel handed in (in/out ports excepted)
        let inputs_c: Vec<VReg> = block.inputs().iter().filter(|p| !p.mutable).map(|p| self.mir.canon(m(p.reg))).collect();
        let start = self.mir.blocks[self.cur].insts.len() - n_pushed;
        for i in &self.mir.blocks[self.cur].insts[start..] {
            if let Some(d) = i.dst {
                if inputs_c.contains(&self.mir.canon(d)) {
                    return Err(format!("asm block `{}` writes one of its input registers", block.name()));
                }
            }
        }
        Ok(())
    }

    /// Bit pattern of a constant operand (integer or f32), for in-line operands.    /// `base` = `PtrAdd{ scalar kernarg pointer, vector byte offset }`: (scalar pair, 32-bit vector offset) for the SADDR form.
    fn saddr_parts_force(&mut self, base: Opnd) -> Result<Option<(VReg, VReg)>, String> {
        self.saddr_parts_impl(base)
    }
    fn saddr_parts_impl(&mut self, base: Opnd) -> Result<Option<(VReg, VReg)>, String> {
        let Opnd::Val(v) = base else { return Ok(None) };
        let Some(Op::PtrAdd { base: pb, byte_off }) = self.defs.get(&v).cloned() else { return Ok(None) };
        let Opnd::Val(pv) = pb else { return Ok(None) };
        let Some(&sreg) = self.val.get(&pv) else { return Ok(None) };
        if self.mir.vreg_class(sreg) != RClass::S2 {
            return Ok(None);
        }
        let off = self.as_v1(byte_off)?;
        self.touch(&[Src::Reg(sreg), Src::Reg(off)]);
        Ok(Some((sreg, off)))
    }

    /// MUBUF operands: (descriptor S4 register, voffset source, soffset source). `voffset` None = the `off` form (encoded as `Src::Imm(0)`); `soffset` is an
    /// SGPR value (wave-uniform integer, loop variable allowed) or a small constant (0..64, an inline constant of the encoding).
    fn buffer_operands(&mut self, rsrc: Opnd, voff: Option<Opnd>, soff: Opnd) -> Result<(VReg, Src, Src), String> {
        let rr = match rsrc {
            Opnd::Val(v) => *self.val.get(&v).ok_or_else(|| format!("descriptor v{v} not lowered"))?,
            _ => return Err("buffer descriptor operand must be a value".to_string()),
        };
        if self.mir.vreg_class(rr) != RClass::S4 {
            return Err("buffer access: the descriptor operand is not an S4 register".to_string());
        }
        let vs = match voff {
            Some(o) => Src::Reg(self.as_v1(o)?),
            None => Src::Imm(0),
        };
        let ss = match self.const_bits(soff) {
            Some(k) if (0..=64).contains(&k) => Src::Imm(k),
            Some(k) => return Err(format!("buffer soffset constant {k} must be 0..64 (an inline constant); put larger values in an SGPR")),
            None => self.s_operand(soff)?,
        };
        let mut t = vec![Src::Reg(rr), vs];
        t.push(ss);
        self.touch(&t);
        Ok((rr, vs, ss))
    }

    /// The `M0` source of an LDS-DMA load: a wave-uniform SGPR (or constant moved into one), else the first lane of a VGPR.
    fn m0_source(&mut self, lds_addr: Opnd) -> Result<VReg, String> {
        let r = match lds_addr {
            Opnd::Var(v) => self.vars.get(v.0 as usize).copied().filter(|r| self.mir.vreg_class(*r) == RClass::S1),
            o => self.scalar_reg(o),
        };
        match (r, self.const_bits(lds_addr)) {
            (Some(r), _) => {
                self.touch(&[Src::Reg(r)]);
                Ok(r)
            }
            (None, Some(k)) => {
                let t = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::SMovB32).def(t).imm(k));
                Ok(t)
            }
            _ => {
                let lv = self.as_v1(lds_addr)?;
                self.touch(&[Src::Reg(lv)]);
                let s = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::VReadFirstLane).def(s).srcreg(lv));
                Ok(s)
            }
        }
    }

    /// A scalar source operand: an immediate (SALU takes 32-bit literals) or an SGPR value / scalar loop variable.
    fn s_operand(&mut self, o: Opnd) -> Result<Src, String> {
        if let Some(k) = self.const_bits(o) {
            return Ok(Src::Imm(k));
        }
        let r = match o {
            Opnd::Val(v) => *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?,
            Opnd::Var(v) => *self.vars.get(v.0 as usize).ok_or("var not lowered")?,
            _ => return Err("scalar operand must be a value, variable or immediate".to_string()),
        };
        if self.mir.vreg_class(r) != RClass::S1 {
            return Err(format!("operand {o:?} is not in an SGPR (uniform analysis disagrees with lowering)"));
        }
        Ok(Src::Reg(r))
    }

    /// `dst = cond ? a : b` in SALU: `cond` is a (possibly negated) signed compare of uniform integers, re-emitted as `s_cmp_lt_i32` right before
    /// `s_cselect_b32` (SCC is not kept live across other SALU work).
    fn scalar_select(&mut self, dst: ValueId, cond: Opnd, a: Opnd, b: Opnd) -> Result<(), String> {
        let mut neg = false;
        let mut c = cond;
        let (x, y) = loop {
            let Opnd::Val(v) = c else { return Err("scalar select: the condition is not a value".into()) };
            match self.defs.get(&v).cloned() {
                Some(Op::NotI1(n)) => {
                    neg = !neg;
                    c = n;
                }
                Some(Op::CmpLtI32(x, y)) => break (x, y),
                Some(Op::CmpGeI32(x, y)) => {
                    neg = !neg;
                    break (x, y);
                }
                other => return Err(format!("scalar select: condition {other:?} is not a uniform integer compare")),
            }
        };
        let (t, f) = if neg { (b, a) } else { (a, b) };
        // operands first (literals into SGPRs), then the compare immediately followed by the select
        let mut ops = Vec::with_capacity(4);
        for o in [x, y, t, f] {
            ops.push(self.s_operand(o)?);
        }
        for i in [0usize, 2] {
            if let (Src::Imm(k), Src::Imm(_)) = (ops[i], ops[i + 1]) {
                let r = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::SMovB32).def(r).imm(k));
                ops[i] = Src::Reg(r);
            }
        }
        let regs: Vec<Src> = ops.iter().copied().filter(|s| matches!(s, Src::Reg(_))).collect();
        self.touch(&regs);
        self.push(MInst::new(MOp::SCmpLtI32).src(ops[0]).src(ops[1]));
        let d = self.new_vreg(RClass::S1);
        self.push(MInst::new(MOp::SCselectB32).def(d).src(ops[2]).src(ops[3]));
        self.val.insert(dst, d);
        Ok(())
    }

    /// `dst = cond ? a : b` for a 64-bit pointer (S2) or a 128-bit buffer descriptor (S4) held in SGPRs: `s_cmp_lt_i32` + `s_cselect_b64` (x2 for the
    /// descriptor) - no branch, no VGPR round trip. `cond` must be a (possibly negated) signed compare of wave-uniform integers.
    fn scalar_select_wide(&mut self, dst: ValueId, ty: DType, cond: Opnd, a: Opnd, b: Opnd) -> Result<(), String> {
        let mut neg = false;
        let mut c = cond;
        let (x, y) = loop {
            let Opnd::Val(v) = c else { return Err("uniform select: the condition is not a value".into()) };
            match self.defs.get(&v).cloned() {
                Some(Op::NotI1(n)) => {
                    neg = !neg;
                    c = n;
                }
                Some(Op::CmpLtI32(x, y)) => break (x, y),
                Some(Op::CmpGeI32(x, y)) => {
                    neg = !neg;
                    break (x, y);
                }
                other => return Err(format!("uniform select: condition {other:?} is not a signed compare of wave-uniform integers")),
            }
        };
        let (t, f) = if neg { (b, a) } else { (a, b) };
        let cls = if ty == DType::Rsrc { RClass::S4 } else { RClass::S2 };
        let arm = |o: Opnd| -> Result<VReg, String> {
            let r = match o {
                Opnd::Val(v) => *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?,
                _ => return Err("uniform select: an arm is not a value".into()),
            };
            if self.mir.vreg_class(r) != cls {
                return Err(format!("uniform select of {}: arm {o:?} is in {:?}, not an SGPR {:?} (only wave-uniform pointers / descriptors)", ty.name(), self.mir.vreg_class(r), cls));
            }
            Ok(r)
        };
        let (rt, rf) = (arm(t)?, arm(f)?);
        let (mut sx, sy) = (self.s_operand(x)?, self.s_operand(y)?);
        if let (Src::Imm(k), Src::Imm(_)) = (sx, sy) {
            let r = self.new_vreg(RClass::S1);
            self.push(MInst::new(MOp::SMovB32).def(r).imm(k));
            sx = Src::Reg(r);
        }
        let regs: Vec<Src> = [sx, sy, Src::Reg(rt), Src::Reg(rf)].into_iter().filter(|s| matches!(s, Src::Reg(_))).collect();
        self.touch(&regs);
        self.push(MInst::new(MOp::SCmpLtI32).src(sx).src(sy));
        let d = self.new_vreg(cls);
        let op = if cls == RClass::S4 { MOp::SCselectB128 } else { MOp::SCselectB64 };
        self.push(MInst::new(op).def(d).srcreg(rt).srcreg(rf));
        self.val.insert(dst, d);
        Ok(())
    }

    /// A scalar source when the operand already is wave-uniform: a literal, an SGPR value or a scalar loop variable (None otherwise; never copies).
    fn scalar_src(&self, o: Opnd) -> Option<Src> {
        if let Some(k) = self.const_bits(o) {
            return Some(Src::Imm(k));
        }
        let r = match o {
            Opnd::Val(v) => *self.val.get(&v)?,
            Opnd::Var(v) => *self.vars.get(v.0 as usize)?,
            _ => return None,
        };
        (self.mir.vreg_class(r) == RClass::S1).then_some(Src::Reg(r))
    }

    /// `dst = a <op> b` in SALU (both operands uniform).
    fn scalar_bin(&mut self, dst: ValueId, op: MOp, a: Opnd, b: Opnd) -> Result<(), String> {
        let (mut sa, mut sb) = (self.s_operand(a)?, self.s_operand(b)?);
        let commutative = matches!(op, MOp::SAddU32 | MOp::SMulI32 | MOp::SAndB32 | MOp::SXorB32);
        if matches!(sa, Src::Imm(_)) && matches!(sb, Src::Reg(_)) && commutative {
            std::mem::swap(&mut sa, &mut sb);
        }
        if let Src::Imm(k) = sa {
            let t = self.new_vreg(RClass::S1);
            self.push(MInst::new(MOp::SMovB32).def(t).imm(k));
            sa = Src::Reg(t);
        }
        let regs: Vec<Src> = [sa, sb].into_iter().filter(|s| matches!(s, Src::Reg(_))).collect();
        self.touch(&regs);
        let d = self.new_vreg(RClass::S1);
        self.push(MInst::new(op).def(d).src(sa).src(sb));
        self.val.insert(dst, d);
        Ok(())
    }

    fn const_bits(&self, o: Opnd) -> Option<i64> {
        match o {
            Opnd::I32(i) => Some(i as i64),
            Opnd::F32(f) => Some((f.to_bits() as i32) as i64),
            Opnd::Val(v) => match self.defs.get(&v) {
                Some(Op::ConstI32(i)) => Some(*i as i64),
                Some(Op::ConstF32(f)) => Some((f.to_bits() as i32) as i64),
                _ => None,
            },
            _ => None,
        }
    }

    fn const_of(&self, o: Opnd) -> Option<i64> {
        match o {
            Opnd::I32(i) => Some(i as i64),
            Opnd::Val(v) => match self.defs.get(&v) {
                Some(Op::ConstI32(i)) => Some(*i as i64),
                _ => None,
            },
            _ => None,
        }
    }

    /// An LDS address operand split into a base register and the constant byte offset that the DS
    /// instruction can take in its immediate field (chains of `x + const` are flattened).
    fn lds_addr(&mut self, o: Opnd) -> Result<(VReg, i64), String> {
        let mut cur = o;
        let mut off: i64 = 0;
        loop {
            if let Opnd::Val(v) = cur {
                if let Some(&(t, k)) = self.lazy.get(&v) {
                    if  (0..65536).contains(&(off + k)) {
                        self.touch(&[Src::Reg(t)]);
                        return Ok((t, off + k));
                    }
                }
                if let Some(Op::AddI32(x, y)) = self.defs.get(&v).cloned() {
                    if let Some(k) = self.const_of(y) {
                        if k >= 0 && off + k < 65536 {
                            off += k;
                            cur = x;
                            continue;
                        }
                    }
                    if let Some(k) = self.const_of(x) {
                        if k >= 0 && off + k < 65536 {
                            off += k;
                            cur = y;
                            continue;
                        }
                    }
                }
            }
            break;
        }
        Ok((self.as_v1(cur)?, off))
    }

    /// Is `r` a loop variable's register or a view (lane / sub-register) of one? Such registers are overwritten in place.
    fn var_backed(&self, r: VReg) -> bool {
        let c = self.mir.canon(r);
        self.vars.iter().any(|v| *v != u32::MAX && self.mir.canon(*v) == c)
    }

    /// (register, constant) view of an integer operand for `addrfold`.
    fn lazy_core(&mut self, o: Opnd) -> Result<(VReg, i64), String> {
        if let Opnd::Val(v) = o {
            if let Some(&(t, k)) = self.lazy.get(&v) {
                return Ok((t, k));
            }
        }
        let (c, k) = self.split_const(o);
        if let Opnd::Val(v) = c {
            if let Some(&(t, k2)) = self.lazy.get(&v) {
                return Ok((t, k + k2));
            }
        }
        Ok((self.as_v1(c)?, k))
    }

    /// `dst = a + b` kept as `reg + k` when a constant is involved; emits at most the register add. Returns false
    /// when nothing is gained (the caller emits the ordinary add).
    fn lazy_add(&mut self, dst: ValueId, a: Opnd, b: Opnd) -> Result<bool, String> {
        if self.uniform.contains(&dst) {
            return Ok(false);
        }
        // A loop variable is a mutable register: `var + k` must be evaluated now, never kept as a reference to the
        // register (set_var may overwrite it before the value is used).
        for o in [a, b] {
            if self.const_bits(o).is_none() && matches!(self.split_const(o).0, Opnd::Var(_)) {
                return Ok(false);
            }
        }
        let (kb, ka) = (self.const_bits(b), self.const_bits(a));
        match (ka, kb) {
            (None, Some(k)) | (Some(k), None) => {
                let other = if kb.is_some() { a } else { b };
                let (t, k0) = self.lazy_core(other)?;
                if self.var_backed(t) {
                    return Ok(false);
                }
                self.lazy.insert(dst, (t, k0 + k));
                Ok(true)
            }
            (None, None) => {
                // only worth it when a side already carries a constant
                let sa = self.split_const(a).1 != 0 || matches!(a, Opnd::Val(v) if self.lazy.contains_key(&v));
                let sb = self.split_const(b).1 != 0 || matches!(b, Opnd::Val(v) if self.lazy.contains_key(&v));
                if !(sa || sb) {
                    return Ok(false);
                }
                let (ta, k1) = self.lazy_core(a)?;
                let (tb, k2) = self.lazy_core(b)?;
                if self.var_backed(ta) || self.var_backed(tb) {
                    return Ok(false);
                }
                self.touch(&[Src::Reg(ta), Src::Reg(tb)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VAddU32).def(d).srcreg(ta).srcreg(tb));
                self.lazy.insert(dst, (d, k1 + k2));
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// ``o = core + k`` through chains of ``x + const``: (core, k).
    fn split_const(&self, o: Opnd) -> (Opnd, i64) {
        let mut cur = o;
        let mut k = 0i64;
        loop {
            if let Opnd::Val(v) = cur {
                if let Some(Op::AddI32(x, y)) = self.defs.get(&v) {
                    if let Some(c) = self.const_of(*y) {
                        k += c;
                        cur = *x;
                        continue;
                    }
                    if let Some(c) = self.const_of(*x) {
                        k += c;
                        cur = *y;
                        continue;
                    }
                }
            }
            return (cur, k);
        }
    }

    /// A freshly zeroed sixteen-dword fragment.
    fn fresh_zero16(&mut self) -> VReg {
        let d = self.new_vreg(RClass::V16);
        let zero = self.const_vreg(0);
        for i in 0..16 {
            self.push(MInst::new(MOp::VWriteSub).def(d).srcreg(zero).imm(i as i64));
        }
        d
    }

    fn const_vreg(&mut self, bits: i64) -> VReg {
        if let Some(r) = self.consts.get(&bits) {
            return *r;
        }
        let r = self.new_vreg(RClass::V1);
        self.push(MInst::new(MOp::VMovB32).def(r).imm(bits));
        self.consts.insert(bits, r);
        r
    }

    fn as_v1(&mut self, o: Opnd) -> Result<VReg, String> {
        match o {
            Opnd::Val(v) => {
                // Constants are rematerialised per region: a constant defined before a loop
                // must not stay live across the whole loop just because the body uses it.
                match self.defs.get(&v).cloned() {
                    Some(Op::ConstI32(i)) => return Ok(self.const_vreg(i as i64)),
                    Some(Op::ConstF32(f)) => return Ok(self.const_vreg((f.to_bits() as i32) as i64)),
                    _ => {}
                }
                if let Some(r) = self.converted.get(&v) {
                    let r = *r;
                    self.touch(&[Src::Reg(r)]);
                    return Ok(r);
                }
                if let Some(&(t, k)) = self.lazy.get(&v) {
                    // materialise `t + k` (cached for the region)
                    if k == 0 {
                        self.touch(&[Src::Reg(t)]);
                        return Ok(t);
                    }
                    let d = self.new_vreg(RClass::V1);
                    self.touch(&[Src::Reg(t)]);
                    self.push(MInst::new(MOp::VAddU32).def(d).src(Src::Imm(k)).srcreg(t));
                    self.converted.insert(v, d);
                    return Ok(d);
                }
                let r = *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?;
                match self.mir.vreg_class(r) {
                    RClass::V1 => {
                        self.touch(&[Src::Reg(r)]);
                        Ok(r)
                    }
                    RClass::S1 => {
                        let d = self.new_vreg(RClass::V1);
                        self.touch(&[Src::Reg(r)]);
                        self.push(MInst::new(MOp::VMovFromS).def(d).srcreg(r));
                        self.converted.insert(v, d);
                        Ok(d)
                    }
                    other => Err(format!("value v{v} has class {other:?}, expected scalar/vector")),
                }
            }
            Opnd::Var(v) => {
                let r = self.vars.get(v.0 as usize).copied().ok_or_else(|| format!("var{v} not lowered"))?;
                if self.mir.vreg_class(r) == RClass::S1 {
                    // uniform (SGPR) loop variable read by a VALU consumer
                    let d = self.new_vreg(RClass::V1);
                    self.touch(&[Src::Reg(r)]);
                    self.push(MInst::new(MOp::VMovFromS).def(d).srcreg(r));
                    return Ok(d);
                }
                Ok(r)
            }
            Opnd::I32(i) => Ok(self.const_vreg(i as i64)),
            Opnd::F32(f) => Ok(self.const_vreg((f.to_bits() as i32) as i64)),
        }
    }

    fn as_v2(&mut self, o: Opnd) -> Result<VReg, String> {
        match o {
            Opnd::Val(v) => {
                if let Some(r) = self.converted.get(&v) {
                    let r = *r;
                    self.touch(&[Src::Reg(r)]);
                    return Ok(r);
                }
                if let Some(&(t, k)) = self.lazy.get(&v) {
                    // materialise `t + k` (cached for the region)
                    if k == 0 {
                        self.touch(&[Src::Reg(t)]);
                        return Ok(t);
                    }
                    let d = self.new_vreg(RClass::V1);
                    self.touch(&[Src::Reg(t)]);
                    self.push(MInst::new(MOp::VAddU32).def(d).src(Src::Imm(k)).srcreg(t));
                    self.converted.insert(v, d);
                    return Ok(d);
                }
                let r = *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?;
                match self.mir.vreg_class(r) {
                    RClass::V2 => {
                        self.touch(&[Src::Reg(r)]);
                        Ok(r)
                    }
                    RClass::S2 => {
                        let d = self.new_vreg(RClass::V2);
                        self.touch(&[Src::Reg(r)]);
                        self.push(MInst::new(MOp::VMovFromSPair).def(d).srcreg(r));
                        self.converted.insert(v, d);
                        Ok(d)
                    }
                    other => Err(format!("value v{v} has class {other:?}, expected pointer")),
                }
            }
            Opnd::Var(v) => {
                let r = self
                    .vars
                    .get(v.0 as usize)
                    .copied()
                    .ok_or_else(|| format!("var{v} not lowered"))?;
                if self.mir.vreg_class(r) == RClass::V2 {
                    self.touch(&[Src::Reg(r)]);
                    Ok(r)
                } else {
                    Err(format!("var{v} is not a pointer"))
                }
            }
            _ => Err("pointer operand must be a value or variable".to_string()),
        }
    }

    /// Eight-dword fragment operand.
    fn as_v8(&mut self, o: Opnd) -> Result<VReg, String> {
        match o {
            Opnd::Val(v) => {
                let r = *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?;
                if self.mir.vreg_class(r) == RClass::V8 {
                    self.touch(&[Src::Reg(r)]);
                    Ok(r)
                } else {
                    Err(format!("value v{v} is not an eight-dword fragment"))
                }
            }
            Opnd::Var(v) => {
                let r = *self.vars.get(v.0 as usize).ok_or_else(|| format!("var{} not lowered", v.0))?;
                if self.mir.vreg_class(r) == RClass::V8 {
                    self.touch(&[Src::Reg(r)]);
                    Ok(r)
                } else {
                    Err(format!("var{} is not an eight-dword fragment", v.0))
                }
            }
            _ => Err("eight-dword fragment operand must be an SSA value or a loop variable".to_string()),
        }
    }

    /// Copy a whole register (all dwords of its class) into `dst`.
    fn copy_reg(&mut self, dst: VReg, src: VReg) -> Result<(), String> {
        if dst == src {
            return Ok(());
        }
        let class = self.mir.vreg_class(src);
        let width = class.dwords() as usize;
        if width == 1 {
            self.push(MInst::new(MOp::VMovB32).def(dst).srcreg(src));
            return Ok(());
        }
        if self.mir.vreg_class(dst) != class {
            return Err(format!("copy_reg: class mismatch {:?} -> {:?}", class, self.mir.vreg_class(dst)));
        }
        for i in 0..width {
            let s = self.alias_vreg(src, i as u8, RClass::V1);
            let d = self.alias_vreg(dst, i as u8, RClass::V1);
            self.push(MInst::new(MOp::VMovB32).def(d).srcreg(s));
        }
        Ok(())
    }

    /// Sixteen-dword fragment operand.
    fn as_v16(&mut self, o: Opnd) -> Result<VReg, String> {
        match o {
            Opnd::Val(v) if matches!(self.defs.get(&v), Some(Op::FragZero { ty: DType::Frag16 })) => {
                Ok(self.fresh_zero16())
            }
            Opnd::Val(v) => {
                let r = *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?;
                if self.mir.vreg_class(r) == RClass::V16 {
                    self.touch(&[Src::Reg(r)]);
                    Ok(r)
                } else {
                    Err(format!("value v{v} is not a sixteen-dword fragment"))
                }
            }
            Opnd::Var(v) => {
                let r = self
                    .vars
                    .get(v.0 as usize)
                    .copied()
                    .ok_or_else(|| format!("var{v} not lowered"))?;
                if self.mir.vreg_class(r) == RClass::V16 {
                    self.touch(&[Src::Reg(r)]);
                    Ok(r)
                } else {
                    Err(format!("var{v} is not a sixteen-dword fragment"))
                }
            }
            _ => Err("fragment operand must be a value or variable".to_string()),
        }
    }

    /// Pointer or two-dword fragment operand, whichever the value holds.
    fn as_v4(&mut self, o: Opnd) -> Result<VReg, String> {
        match o {
            Opnd::Val(v) => {
                let r = *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?;
                if self.mir.vreg_class(r) == RClass::V4 {
                    self.touch(&[Src::Reg(r)]);
                    Ok(r)
                } else {
                    Err(format!("value v{v} is not a four-dword fragment"))
                }
            }
            Opnd::Var(v) => {
                let r = self
                    .vars
                    .get(v.0 as usize)
                    .copied()
                    .ok_or_else(|| format!("var{v} not lowered"))?;
                if self.mir.vreg_class(r) == RClass::V4 {
                    self.touch(&[Src::Reg(r)]);
                    Ok(r)
                } else {
                    Err(format!("var{v} is not a four-dword fragment"))
                }
            }
            _ => Err("fragment operand must be a value or variable".to_string()),
        }
    }

    /// Lowers the region tree: every node is handed to `lower_stmt` in program order, the structure markers of loops / ifs / tile
    /// regions at their boundaries (the handlers keep the label / snapshot stacks), so the emitted code is that of the linear walk.
    fn run(&mut self, f: &Function) -> Result<(), String> {
        self.walk(f, f.body())
    }

    fn walk(&mut self, f: &Function, nodes: &[Node]) -> Result<(), String> {
        for n in nodes {
            match n {
                Node::S(st) => self.lower_stmt(f, st)?,
                Node::Loop { vars, init, body } => {
                    self.lower_stmt(f, &Stmt::BeginLoop { vars: vars.clone(), init: init.clone() })?;
                    self.walk(f, body)?;
                    self.lower_stmt(f, &Stmt::EndLoop)?;
                }
                Node::If { cond, then, els } => {
                    self.lower_stmt(f, &Stmt::BeginIf { cond: *cond })?;
                    self.walk(f, then)?;
                    if let Some(e) = els {
                        self.lower_stmt(f, &Stmt::Else)?;
                        self.walk(f, e)?;
                    }
                    self.lower_stmt(f, &Stmt::EndIf)?;
                }
                Node::Region { kind, tile, slot, version, body } => {
                    let open = match kind {
                        RegionKind::Produce => Stmt::TileProduce { tile: tile.clone(), slot: *slot, version: *version },
                        RegionKind::Consume => Stmt::TileConsume { tile: tile.clone(), slot: *slot, version: *version },
                    };
                    self.lower_stmt(f, &open)?;
                    self.walk(f, body)?;
                    self.lower_stmt(f, &Stmt::TileEnd)?;
                }
                // a stage is transparent to lowering (its contract is checked by gpu_verify)
                Node::Stage { body, .. } => self.walk(f, body)?,
            }
        }
        Ok(())
    }

    /// One statement (structure markers included: they open / close the CFG regions).
    fn lower_stmt(&mut self, f: &Function, st: &Stmt) -> Result<(), String> {
        {
            match st {
                Stmt::Let { dst, op } => {
                    self.lower_op(*dst, op)?;
                    if self.view_copy.contains(dst) {
                        // SSA semantics for a view of a loop variable used after the variable is written: snapshot it now
                        let r = *self.val.get(dst).ok_or("view not lowered")?;
                        let c = self.new_vreg(self.mir.vreg_class(r));
                        self.touch(&[Src::Reg(r)]);
                        if self.mir.vreg_class(r) == RClass::S1 {
                            self.push(MInst::new(MOp::SMovB32).def(c).srcreg(r));
                        } else {
                            self.copy_reg(c, r)?;
                        }
                        self.val.insert(*dst, c);
                    }
                }
                Stmt::StoreU16 { base, byte_off, value } => {
                    let b = self.as_v2(*base)?;
                    let v = self.as_v1(*value)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(
                        MInst::new(MOp::GlobalStoreU16)
                            .srcreg(b)
                            .srcreg(v)
                            .imm(*byte_off as i64),
                    );
                }
                Stmt::StoreF32 { base, byte_off, value } => {
                    let b = self.as_v2(*base)?;
                    let v = self.as_v1(*value)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(
                        MInst::new(MOp::GlobalStoreF32)
                            .srcreg(b)
                            .srcreg(v)
                            .imm(*byte_off as i64),
                    );
                }
                Stmt::StoreF32Wt { base, byte_off, value } => {
                    let b = self.as_v2(*base)?;
                    let v = self.as_v1(*value)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(MInst::new(MOp::GlobalStoreF32Sc1).srcreg(b).srcreg(v).imm(*byte_off as i64));
                }
                Stmt::FenceRelease { wb } => {
                    // everything issued so far must be complete (stores acknowledged) before the L2 write-back and before the releasing atomic
                    self.push(MInst::new(MOp::SWaitVm).imm(0));
                    self.push(MInst::new(MOp::SWaitLgkm).imm(0));
                    if *wb {
                        self.push(MInst::new(MOp::BufferWbl2));
                        self.push(MInst::new(MOp::SWaitVm).imm(0));
                    }
                    self.pending_vmem.clear();
                    self.pending_lds.clear();
                    self.pending_smem.clear();
                }
                Stmt::FenceAcquire { inv } => {
                    // the ticket (atomic result) must have returned; then drop stale cache lines before the loads that follow
                    self.push(MInst::new(MOp::SWaitVm).imm(0));
                    self.pending_vmem.clear();
                    if *inv {
                        self.push(MInst::new(MOp::BufferInv));
                        // BUFFER_INV returns an ACK through VM_CNT: wait for it before the first coherent load (strict full sequence)
                        self.push(MInst::new(MOp::SWaitVm).imm(0));
                    }
                }
                Stmt::BufferStore { rsrc, voff, soff, imm, dwords, value } => {
                    let (rr, vs, ss) = self.buffer_operands(*rsrc, *voff, *soff)?;
                    let (v, mop) = match dwords {
                        1 => (self.as_v1(*value)?, MOp::BufferStoreDwordX1),
                        2 => (self.as_v2(*value)?, MOp::BufferStoreDwordX2),
                        _ => (self.as_v4(*value)?, MOp::BufferStoreDwordX4),
                    };
                    self.touch(&[Src::Reg(v)]);
                    self.push(MInst::new(mop).srcreg(v).srcreg(rr).src(vs).src(ss).imm(*imm as i64));
                }
                Stmt::BufferLoadLds { rsrc, voff, soff, lds_addr, imm, dwords } => {
                    let (rr, vs, ss) = self.buffer_operands(*rsrc, *voff, *soff)?;
                    let m0 = self.m0_source(*lds_addr)?;
                    let mop = if *dwords == 4 { MOp::BufferLoadDwordX4Lds } else { MOp::BufferLoadDwordX1Lds };
                    self.push(MInst::new(mop).srcreg(rr).src(vs).src(ss).srcreg(m0).imm(*imm as i64));
                }
                Stmt::StoreF32x2 { base, byte_off, value } => {
                    let b = self.as_v2(*base)?;
                    let v = self.as_v2(*value)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(
                        MInst::new(MOp::GlobalStoreF32x2)
                            .srcreg(b)
                            .srcreg(v)
                            .imm(*byte_off as i64),
                    );
                }
                Stmt::StoreF32x4 { base, byte_off, value } => {
                    let b = self.as_v2(*base)?;
                    let v = self.as_v4(*value)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(
                        MInst::new(MOp::GlobalStoreF32x4)
                            .srcreg(b)
                            .srcreg(v)
                            .imm(*byte_off as i64),
                    );
                }
                Stmt::SetVar { var, value } => {
                    let dst = self.vars[var.0 as usize];
                    let class = self.mir.vreg_class(dst);
                    if class == RClass::S1 {
                        let s = self.s_operand(*value)?;
                        if let Src::Reg(x) = s {
                            self.touch(&[Src::Reg(x)]);
                        }
                        self.push(MInst::new(MOp::SMovB32).def(dst).src(s));
                        return Ok(());
                    }
                    if class == RClass::V16 && opt_on("zeroinit") && matches!(value, Opnd::Val(v) if matches!(self.defs.get(v), Some(Op::FragZero { ty: DType::Frag16 }))) {
                        // zero accumulator: zero the variable's own register instead of building a zero tuple and copying it
                        let zero = self.const_vreg(0);
                        for i in 0..16 {
                            self.push(MInst::new(MOp::VWriteSub).def(dst).srcreg(zero).imm(i as i64));
                        }
                        return Ok(());
                    }
                    // in-place reload: set_var(var, load_lds(..)) where the load is the value's only use and nothing reads the variable after the load
                    // was issued - the ds_read writes the variable's own registers (no copy, no second fragment; the AGPR pipelining pattern)
                    if matches!(class, RClass::V4 | RClass::V8) {
                        if let Opnd::Val(v) = value {
                            if self.use_counts.get(v).copied() == Some(1) && self.try_coalesce_lds_reload(*v, dst) {
                                return Ok(());
                            }
                        }
                    }
                    let src = match class {
                        RClass::V1 => self.as_v1(*value)?,
                        RClass::V2 => self.as_v2(*value)?,
                        RClass::V4 => self.as_v4(*value)?,
                        RClass::V8 => self.as_v8(*value)?,
                        RClass::V16 => self.as_v16(*value)?,
                        other => return Err(format!("loop variable class {other:?} is not supported")),
                    };
                    self.touch(&[Src::Reg(src)]);
                    self.copy_reg(dst, src)?;
                }
                Stmt::VarScaleF32 { var, scalar } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let s = self.as_v1(*scalar)?;
                    self.touch(&[Src::Reg(v), Src::Reg(s)]);
                    for i in 0..16u8 {
                        let a = self.alias_vreg(v, i, RClass::V1);
                        self.push(MInst::new(MOp::VMulF32).def(a).srcreg(a).srcreg(s));
                    }
                }
                Stmt::VarSetLane { var, index, value } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let a = self.sub_alias(v, *index, RClass::V1);
                    // the producer writes the variable's dword directly when the value has no other use and was computed in this block
                    let mut done = false;
                    if let Opnd::Val(x) = value {
                        let is_const = matches!(self.defs.get(x), Some(Op::ConstI32(_) | Op::ConstF32(_)));
                        if let (false, Some(1), Some(&r)) = (is_const, self.exact_uses.get(x).copied(), self.val.get(x)) {
                            let cur = self.cur;
                            let plain = self.mir.vreg_class(r) == RClass::V1 && !self.mir.alias.contains_key(&r) && !self.mir.alias.values().any(|(t, _)| *t == r);
                            let def_here = self.mir.blocks[cur].insts.iter().any(|i| i.dst == Some(r))
                                && self.mir.blocks[cur].insts.iter().filter(|i| i.dst == Some(r)).all(|i| { let at = crate::op_attrs::attrs(i.op); at.mem.is_none() && !at.smem })
                                && !self.consts.values().any(|q| *q == r)
                                && !self.converted.values().any(|q| *q == r);
                            let elsewhere = self.mir.blocks.iter().enumerate().any(|(bi, b)| bi != cur && b.insts.iter().any(|i| i.dst == Some(r) || i.srcs.iter().any(|s| matches!(s, Src::Reg(q) if *q == r))));
                            if plain && def_here && !elsewhere {
                                for i in self.mir.blocks[cur].insts.iter_mut() {
                                    if i.dst == Some(r) {
                                        i.dst = Some(a);
                                    }
                                    for s in i.srcs.iter_mut() {
                                        if let Src::Reg(q) = s {
                                            if *q == r {
                                                *q = a;
                                            }
                                        }
                                    }
                                }
                                self.touch(&[Src::Reg(v)]);
                                done = true;
                            }
                        }
                    }
                    if !done {
                        let s = self.as_v1(*value)?;
                        self.touch(&[Src::Reg(v), Src::Reg(s)]);
                        self.push(MInst::new(MOp::VMovB32).def(a).srcreg(s));
                    }
                }
                Stmt::VarScaleLane { var, index, scalar } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let s = self.as_v1(*scalar)?;
                    self.touch(&[Src::Reg(v), Src::Reg(s)]);
                    let a = self.sub_alias(v, *index, RClass::V1);
                    self.push(MInst::new(MOp::VMulF32).def(a).srcreg(a).srcreg(s));
                }
                Stmt::VarAddPk { var, pair } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let p = self.as_v2(*pair)?;
                    self.touch(&[Src::Reg(v), Src::Reg(p)]);
                    for i in 0..8u8 {
                        let a = self.sub_alias(v, 2 * i, RClass::V2);
                        self.push(MInst::new(MOp::VPkAddF32).def(a).srcreg(a).srcreg(p));
                    }
                }
                Stmt::VarMfmaF16Z { var, a, b } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v2(*a)?;
                    let rb = self.as_v2(*b)?;
                    self.push(MInst::new(MOp::VMfma32x32x8F16Z).def(v).srcreg(ra).srcreg(rb));
                }
                Stmt::VarScalePk { var, pair } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let p = self.as_v2(*pair)?;
                    self.touch(&[Src::Reg(v), Src::Reg(p)]);
                    for i in 0..8u8 {
                        let a = self.sub_alias(v, 2 * i, RClass::V2);
                        self.push(MInst::new(MOp::VPkMulF32).def(a).srcreg(a).srcreg(p));
                    }
                }
                Stmt::VarMfmaFp8Z { var, a, b } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v8(*a)?;
                    let rb = self.as_v8(*b)?;
                    self.push(MInst::new(MOp::VMfma32x32x64Fp8Z).def(v).srcreg(ra).srcreg(rb));
                }
                Stmt::VarMfmaFp8C { var, a, b, c } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v8(*a)?;
                    let rb = self.as_v8(*b)?;
                    let rc = self.as_v16(*c)?;
                    self.push(MInst::new(MOp::VMfma32x32x64Fp8).def(v).srcreg(ra).srcreg(rb).srcreg(rc));
                }
                Stmt::VarMfmaFp8 { var, a, b } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v8(*a)?;
                    let rb = self.as_v8(*b)?;
                    self.push(MInst::new(MOp::VMfma32x32x64Fp8).def(v).srcreg(ra).srcreg(rb).srcreg(v));
                }
                Stmt::VarMfmaScaleC { var, a, b, c, scale_a, scale_b } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v8(*a)?;
                    let rb = self.as_v8(*b)?;
                    let rc = self.as_v16(*c)?;
                    let sa = self.as_v1(*scale_a)?;
                    let sb = self.as_v1(*scale_b)?;
                    self.push(MInst::new(MOp::VMfmaScale32x32x64Fp8).def(v).srcreg(ra).srcreg(rb).srcreg(rc).srcreg(sa).srcreg(sb));
                }
                Stmt::VarMfmaScale { var, a, b, scale_a, scale_b, opsel } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v8(*a)?;
                    let rb = self.as_v8(*b)?;
                    let sa = self.as_v1(*scale_a)?;
                    let sb = self.as_v1(*scale_b)?;
                    self.push(
                        MInst::new(MOp::VMfmaScale32x32x64Fp8)
                            .def(v)
                            .srcreg(ra)
                            .srcreg(rb)
                            .srcreg(v)
                            .srcreg(sa)
                            .srcreg(sb)
                            .imm(*opsel as i64),
                    );
                }
                Stmt::VarMfmaScale16 { var, a, b, scale_a, scale_b, opsel } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v8(*a)?;
                    let rb = self.as_v8(*b)?;
                    let sa = self.as_v1(*scale_a)?;
                    let sb = self.as_v1(*scale_b)?;
                    self.push(MInst::new(MOp::VMfmaScale16x16x128Fp8).def(v).srcreg(ra).srcreg(rb).srcreg(v).srcreg(sa).srcreg(sb).imm(*opsel as i64));
                }
                Stmt::VarSmfmac { var, a, b, idx } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v4(*a)?;
                    let rb = self.as_v8(*b)?;
                    let ri = self.as_v1(*idx)?;
                    // C is the destination in place; it is also listed as the 4th (hazard-visible) source
                    self.push(MInst::new(MOp::VSmfmac16x16x128Fp8).def(v).srcreg(ra).srcreg(rb).srcreg(ri).srcreg(v));
                }
                Stmt::VarAsm { var, block, inputs } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    self.lower_asm(None, block, inputs, &[v])?;
                }
                Stmt::VarAsmN { vars, block, inputs } => {
                    let regs: Vec<VReg> = vars.iter().map(|v| self.vars.get(v.0 as usize).copied().ok_or("var not lowered")).collect::<Result<_, _>>()?;
                    self.lower_asm(None, block, inputs, &regs)?;
                }
                Stmt::BeginLoop { vars, init } => {
                    for (i, var) in vars.iter().enumerate() {
                        let ty = f.var_ty(*var).ok_or("loop var has no type")?;
                        let class = match ty {
                            DType::Ptr => return Err("pointer loop variables are not supported".to_string()),
                            DType::Frag2 => RClass::V2,
                            DType::Frag4 => RClass::V4,
                            DType::Frag8 => RClass::V8,
                            DType::Frag16 => RClass::V16,
                            _ if self.uvars.contains(var) => RClass::S1,
                            _ => RClass::V1,
                        };
                        let r = self.new_vreg(class);
                        if self.vars.len() <= var.0 as usize {
                            self.vars.resize(var.0 as usize + 1, u32::MAX);
                        }
                        self.vars[var.0 as usize] = r;
                        if class == RClass::S1 {
                            let s = self.s_operand(init[i])?;
                            if let Src::Reg(x) = s {
                                self.touch(&[Src::Reg(x)]);
                            }
                            self.push(MInst::new(MOp::SMovB32).def(r).src(s));
                            continue;
                        }
                        if class == RClass::V16 && opt_on("zeroinit") && matches!(init[i], Opnd::Val(v) if matches!(self.defs.get(&v), Some(Op::FragZero { ty: DType::Frag16 }))) {
                            let zero = self.const_vreg(0);
                            for k in 0..16 {
                                self.push(MInst::new(MOp::VWriteSub).def(r).srcreg(zero).imm(k as i64));
                            }
                            continue;
                        }
                        // loop-entry in-place load: init = load_lds(..) (single use) writes the variable's own registers instead of a second fragment + copy
                        if matches!(class, RClass::V4 | RClass::V8) {
                            if let Opnd::Val(v) = init[i] {
                                if self.use_counts.get(&v).copied() == Some(1) && self.try_coalesce_lds_reload(v, r) {
                                    continue;
                                }
                            }
                        }
                        let src = match class {
                            RClass::V1 => self.as_v1(init[i])?,
                            RClass::V2 => self.as_v2(init[i])?,
                            RClass::V4 => self.as_v4(init[i])?,
                            RClass::V8 => self.as_v8(init[i])?,
                            RClass::V16 => self.as_v16(init[i])?,
                            _ => return Err("unsupported loop variable class".to_string()),
                        };
                        self.touch(&[Src::Reg(src)]);
                        self.copy_reg(r, src)?;
                    }
                    self.flush_pending();
                    self.scope_snaps.push((self.consts.clone(), self.converted.clone()));
                    self.consts.clear();
                    self.converted.clear();
                    let header = self.fresh_label();
                    let exit = self.fresh_label();
                    self.set_term(Term::Goto(header));
                    self.start_block(header);
                    self.break_stack.push((exit, header));
                }
                Stmt::BreakIf { cond } => {
                    let (exit, _) = *self.break_stack.last().ok_or("break outside loop")?;
                    self.flush_pending();
                    let (on, srcs) = self.lower_condition(*cond, false)?;
                    self.touch(&srcs);
                    let fall = self.fresh_label();
                    self.set_term(Term::CondGoto { on, target: exit, fall });
                    self.start_block(fall);
                }
                Stmt::EndLoop => {
                    let (exit, header) = self.break_stack.pop().ok_or("endloop without loop")?;
                    self.flush_pending();
                    if let Some((c, v)) = self.scope_snaps.pop() {
                        self.consts = c;
                        self.converted = v;
                    }
                    let latch = self.mir.blocks[self.cur].label;
                    self.mir.loops.push((header, latch));
                    self.set_term(Term::Goto(header));
                    self.start_block(exit);
                }
                Stmt::BeginIf { cond } => {
                    // Outstanding loads stay pending across the if (a prefetch must not be drained at
                    // every branch): the sets are restored/compared at else/endif, which drain
                    // conservatively whenever a branch changed them.
                    self.scope_snaps.push((self.consts.clone(), self.converted.clone()));
                    let (on, srcs) = self.lower_condition(*cond, false)?;
                    self.touch(&srcs);
                    self.if_snaps.push(((self.pending_vmem.clone(), self.pending_smem.clone(), self.pending_lds.clone()), false));
                    let then_l = self.fresh_label();
                    let else_l = self.fresh_label();
                    let end_l = self.fresh_label();
                    self.set_term(Term::CondGoto { on, target: then_l, fall: else_l });
                    self.start_block(then_l);
                    self.if_stack.push((else_l, end_l));
                }
                Stmt::Else => {
                    {
                        let cur = (self.pending_vmem.clone(), self.pending_smem.clone(), self.pending_lds.clone());
                        let (snap, dirty) = {
                            let e = self.if_snaps.last_mut().ok_or("else without if")?;
                            if cur != e.0 {
                                e.1 = true;
                            }
                            (e.0.clone(), e.1)
                        };
                        if dirty {
                            // the then-path drains everything; the else path starts from the entry state
                            self.flush_pending();
                        }
                        self.pending_vmem = snap.0;
                        self.pending_smem = snap.1;
                        self.pending_lds = snap.2;
                    }
                    if let Some((else_l, _)) = self.if_stack.last() {
                        self.else_started.insert(*else_l);
                    }
                    if let Some((c, v)) = self.scope_snaps.last() {
                        self.consts = c.clone();
                        self.converted = v.clone();
                    }
                    let (else_l, end_l) = *self.if_stack.last().ok_or("else without if")?;
                    self.set_term(Term::Goto(end_l));
                    self.start_block(else_l);
                }
                Stmt::StoreLdsB128 { addr, value } => {
                    let (a, off) = self.lds_addr(*addr)?;
                    let v = self.as_v4(*value)?;
                    self.push(MInst::new(MOp::LdsWriteB128).srcreg(a).srcreg(v).imm(off));
                }
                Stmt::VarMfma { var, kind, a, b, scale: Some(s), zero } if kind.scaled => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let (mop, zmop, imm) = scaled_mfma_mop(*kind, s.opsel)?;
                    let ra = self.mma_operand(*kind, kind.a, *a)?;
                    let rb = self.mma_operand(*kind, kind.b, *b)?;
                    let sa = self.as_v1(s.sa)?;
                    let sb = self.as_v1(s.sb)?;
                    self.touch(&[Src::Reg(sa), Src::Reg(sb)]);
                    if *zero {
                        self.push(MInst::new(zmop).def(v).srcreg(ra).srcreg(rb).srcreg(sa).srcreg(sb).imm(imm));
                    } else {
                        self.push(MInst::new(mop).def(v).srcreg(ra).srcreg(rb).srcreg(v).srcreg(sa).srcreg(sb).imm(imm));
                    }
                }
                Stmt::VarMfma { var, kind, a, b, scale: _, zero } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let (mop, zmop) = mfma_mop(*kind)?;
                    let ra = self.as_v4(*a)?;
                    let rb = self.as_v4(*b)?;
                    if *zero {
                        self.push(MInst::new(zmop).def(v).srcreg(ra).srcreg(rb));
                    } else {
                        self.push(MInst::new(mop).def(v).srcreg(ra).srcreg(rb).srcreg(v));
                    }
                }
                Stmt::StoreGlobalN { base, byte_off, value, ty, hi, cache } => {
                    let b = self.as_v2(*base)?;
                    let (v, mop) = self.store_value(*value, *ty, *hi, global_store_mop)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(MInst::new(mop).srcreg(b).srcreg(v).imm(*byte_off as i64).cpol(cache.bits()));
                }
                Stmt::BufferStoreN { rsrc, voff, soff, imm, value, ty, hi, cache } => {
                    let (rr, vs, ss) = self.buffer_operands(*rsrc, *voff, *soff)?;
                    let (v, mop) = self.store_value(*value, *ty, *hi, buffer_store_mop)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(MInst::new(mop).srcreg(v).srcreg(rr).src(vs).src(ss).imm(*imm as i64).cpol(cache.bits()));
                }
                Stmt::StoreLdsN { addr, value, ty, hi } => {
                    let (a, off) = self.lds_addr(*addr)?;
                    let (v, mop) = self.store_value(*value, *ty, *hi, lds_store_mop)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(MInst::new(mop).srcreg(a).srcreg(v).imm(off));
                }
                Stmt::StoreLds2 { addr, v0, v1, off0, off1, b64, st64 } => {
                    let (a, imm) = self.lds2_addr(*addr, *off0, *off1, *b64, *st64)?;
                    let (r0, r1) = if *b64 { (self.as_v2(*v0)?, self.as_v2(*v1)?) } else { (self.as_v1(*v0)?, self.as_v1(*v1)?) };
                    self.touch(&[Src::Reg(r0), Src::Reg(r1)]);
                    let mop = match (b64, st64) {
                        (false, false) => MOp::LdsWrite2B32,
                        (true, false) => MOp::LdsWrite2B64,
                        (false, true) => MOp::LdsWrite2St64B32,
                        (true, true) => MOp::LdsWrite2St64B64,
                    };
                    self.push(MInst::new(mop).srcreg(a).srcreg(r0).srcreg(r1).imm(imm));
                }
                Stmt::StoreLdsB64 { addr, value } => {
                    let (a, off) = self.lds_addr(*addr)?;
                    let v = self.as_v2(*value)?;
                    self.push(MInst::new(MOp::LdsWriteB64).srcreg(a).srcreg(v).imm(off));
                }
                Stmt::StoreLdsB32 { addr, value } => {
                    let (a, off) = self.lds_addr(*addr)?;
                    let v = self.as_v1(*value)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(MInst::new(MOp::LdsWriteB32).srcreg(a).srcreg(v).imm(off));
                }
                Stmt::WaitLds { count } => {
                    self.push(MInst::new(MOp::SWaitLgkm).imm(*count as i64));
                }
                Stmt::Assume { .. } | Stmt::TileProduce { .. } | Stmt::TileConsume { .. } | Stmt::TileEnd => {} // contract markers for gpu_verify: no code
                // stages and weave regions: no code; the instructions lowered inside are tagged with the region (register planning)
                Stmt::StageBegin { name, .. } | Stmt::RegionMark { name, begin: true } => {
                    let parent = self.region_stack.last().copied().unwrap_or(0);
                    self.mir.reg_hints.regions.push((name.clone(), parent));
                    let id = self.mir.reg_hints.regions.len() as u16;
                    self.region_stack.push(id);
                }
                Stmt::StageEnd | Stmt::RegionMark { begin: false, .. } => {
                    self.region_stack.pop();
                }
                Stmt::Barrier => {
                    self.push(MInst::new(MOp::SBarrier));
                }
                Stmt::WaitVm { count } => {
                    self.push(MInst::new(MOp::SWaitVm).imm(*count as i64));
                }
                Stmt::LoadGlobalLds { base, lds_addr, byte_off, dwords } => {
                    let sad = if opt_on("saddrlds") { self.saddr_parts_force(*base)? } else { None };
                    let rb = if sad.is_some() { u32::MAX } else { self.as_v2(*base)? };
                    // scalarize: a wave-uniform LDS base already lives in an SGPR (or is a constant): no v_mov + v_readfirstlane
                    let direct: Option<VReg> = if opt_on("scalarize") {
                        let r = match *lds_addr {
                            Opnd::Var(v) => self.vars.get(v.0 as usize).copied().filter(|r| self.mir.vreg_class(*r) == RClass::S1),
                            o => self.scalar_reg(o),
                        };
                        match (r, self.const_bits(*lds_addr)) {
                            (Some(r), _) => Some(r),
                            (None, Some(k)) => {
                                let t = self.new_vreg(RClass::S1);
                                self.push(MInst::new(MOp::SMovB32).def(t).imm(k));
                                Some(t)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let lv = if direct.is_some() { if let Some((_, off)) = sad { off } else { rb } } else { self.as_v1(*lds_addr)? };
                    if let Some((sr0, off0)) = sad { self.touch(&[Src::Reg(sr0), Src::Reg(off0), Src::Reg(lv)]); } else { self.touch(&[Src::Reg(rb), Src::Reg(lv)]); }
                    let sreg = if let Some(s) = direct {
                        self.touch(&[Src::Reg(s)]);
                        s
                    } else { match self.rfl_cache.get(&lv) {
                        Some(s) => *s,
                        None => {
                            let s = self.new_vreg(RClass::S1);
                            self.push(MInst::new(MOp::VReadFirstLane).def(s).srcreg(lv));
                            self.rfl_cache.insert(lv, s);
                            s
                        }
                    } };
                    // imm = instruction offset: added to the global address AND to the LDS destination (M0 + imm + 16 lane)
                    if let Some((sr0, off0)) = sad {
                        let mop = if *dwords == 4 { MOp::GlobalLoadLdsB128S } else { MOp::GlobalLoadLdsB32S };
                        self.push(MInst::new(mop).srcreg(sr0).srcreg(off0).srcreg(sreg).imm(*byte_off as i64));
                    } else {
                        let mop = if *dwords == 4 { MOp::GlobalLoadLdsB128 } else { MOp::GlobalLoadLdsB32 };
                        self.push(MInst::new(mop).srcreg(rb).srcreg(sreg).imm(*byte_off as i64));
                    }
                }
                Stmt::SchedGroup { mask, size, sync } => {
                    self.push(MInst::new(MOp::SchedGroup).imm((*mask as i64 & 0xffff) | ((*size as i64 & 0xffff) << 16) | ((*sync as i64) << 32)));
                }
                Stmt::SetPrio { prio } => {
                    self.push(MInst::new(MOp::SSetPrio).imm(*prio as i64));
                }
                Stmt::EndIf => {
                    let (snap, dirty_then) = self.if_snaps.pop().ok_or("endif without if")?;
                    let cur = (self.pending_vmem.clone(), self.pending_smem.clone(), self.pending_lds.clone());
                    let differs = cur != snap;
                    if let Some((c, v)) = self.scope_snaps.pop() {
                        self.consts = c;
                        self.converted = v;
                    }
                    let (else_l, end_l) = self.if_stack.pop().ok_or("endif without if")?;
                    if self.else_started.remove(&else_l) {
                        // current block ends the else branch
                        if dirty_then || differs {
                            self.flush_pending();
                        }
                        self.set_term(Term::Goto(end_l));
                    } else {
                        // `if` without `else`: current block ends the then branch; the not-taken edge
                        // still needs its target block (and the same drain if the then path drained)
                        if differs {
                            self.flush_pending();
                        }
                        self.set_term(Term::Goto(end_l));
                        self.start_block(else_l);
                        if differs {
                            self.pending_vmem = snap.0.clone();
                            self.pending_smem = snap.1.clone();
                            self.pending_lds = snap.2.clone();
                            self.flush_pending();
                        }
                        self.set_term(Term::Goto(end_l));
                    }
                    if dirty_then || differs {
                        self.pending_vmem.clear();
                        self.pending_smem.clear();
                        self.pending_lds.clear();
                    } else {
                        self.pending_vmem = snap.0;
                        self.pending_smem = snap.1;
                        self.pending_lds = snap.2;
                    }
                    self.start_block(end_l);
                }
            }
                }
        Ok(())
    }

    /// Lowers an `i1` value into a branch condition.
    ///
    /// Peephole: a comparison (possibly negated) branches directly on `vcc`,
    /// avoiding materialising a 0/1 register.
    fn lower_condition(&mut self, cond: Opnd, negate: bool) -> Result<(Pred, Vec<Src>), String> {
        if let Opnd::Val(v) = cond {
            if let Some(op) = self.defs.get(&v).cloned() {
                // `scalarize2`: a signed compare of two wave-uniform integers (SGPR values / scalar loop variables / literals) branches on SCC:
                // `s_cmp_lt_i32` + `s_cbranch_scc0/1` - no VGPR copies, no VALU compare, no vcc hazard padding
                if opt_on("scalarize2") || opt_on("salubr") {
                    if let Op::CmpLtI32(a, b) | Op::CmpGeI32(a, b) = op {
                        if let (Some(mut sa), Some(sb)) = (self.scalar_src(a), self.scalar_src(b)) {
                            if !(matches!(sa, Src::Imm(_)) && matches!(sb, Src::Imm(_))) {
                                if let Src::Imm(k) = sa {
                                    let r = self.new_vreg(RClass::S1);
                                    self.push(MInst::new(MOp::SMovB32).def(r).imm(k));
                                    sa = Src::Reg(r);
                                }
                                let regs: Vec<Src> = [sa, sb].into_iter().filter(|s| matches!(s, Src::Reg(_))).collect();
                                self.touch(&regs);
                                self.push(MInst::new(MOp::SCmpLtI32).src(sa).src(sb));
                                // SCC = a < b; Ge branches on its complement
                                let taken_on_lt = matches!(op, Op::CmpLtI32(..)) != negate;
                                let on = if taken_on_lt { Pred::Scc1 } else { Pred::Scc0 };
                                return Ok((on, regs));
                            }
                        }
                    }
                }
                match op {
                    Op::CmpLtI32(a, b) => {
                        // src0 of the e32 compare may be an SGPR (sgpr option); src1 must be a VGPR
                        let sa = if opt_on("sgpr") { self.sgpr_src(a) } else { None };
                        let ra = match sa {
                            Some(s) => s,
                            None => self.as_v1(a)?,
                        };
                        let rb = self.as_v1(b)?;
                        self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
                        self.push(MInst::new(MOp::VCmpLtI32).srcreg(ra).srcreg(rb));
                        let on = if negate { Pred::Vccz } else { Pred::Vccnz };
                        return Ok((on, vec![Src::Reg(ra), Src::Reg(rb)]));
                    }
                    Op::CmpGeI32(a, b) => {
                        // src0 of the e32 compare may be an SGPR (sgpr option); src1 must be a VGPR
                        let sa = if opt_on("sgpr") { self.sgpr_src(a) } else { None };
                        let ra = match sa {
                            Some(s) => s,
                            None => self.as_v1(a)?,
                        };
                        let rb = self.as_v1(b)?;
                        self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
                        self.push(MInst::new(MOp::VCmpGeI32).srcreg(ra).srcreg(rb));
                        let on = if negate { Pred::Vccz } else { Pred::Vccnz };
                        return Ok((on, vec![Src::Reg(ra), Src::Reg(rb)]));
                    }
                    Op::CmpLtF32(a, b) => {
                        // src0 of the e32 compare may be an SGPR (sgpr option); src1 must be a VGPR
                        let sa = if opt_on("sgpr") { self.sgpr_src(a) } else { None };
                        let ra = match sa {
                            Some(s) => s,
                            None => self.as_v1(a)?,
                        };
                        let rb = self.as_v1(b)?;
                        self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
                        self.push(MInst::new(MOp::VCmpLtF32).srcreg(ra).srcreg(rb));
                        let on = if negate { Pred::Vccz } else { Pred::Vccnz };
                        return Ok((on, vec![Src::Reg(ra), Src::Reg(rb)]));
                    }
                    Op::NotI1(inner) => return self.lower_condition(inner, !negate),
                    _ => {}
                }
            }
        }
        let r = self.as_v1(cond)?;
        let zero = self.const_vreg(0);
        self.push(MInst::new(MOp::VCmpNeI32).srcreg(r).srcreg(zero));
        let on = if negate { Pred::Vccz } else { Pred::Vccnz };
        Ok((on, vec![Src::Reg(r), Src::Reg(zero)]))
    }

    fn lower_op(&mut self, dst: ValueId, op: &Op) -> Result<(), String> {
        match op {
            Op::ConstI32(v) => {
                let r = self.const_vreg(*v as i64);
                self.val.insert(dst, r);
            }
            Op::ConstF32(v) => {
                let r = self.const_vreg((v.to_bits() as i32) as i64);
                self.val.insert(dst, r);
            }
            Op::KernargI32 { offset } | Op::KernargF32 { offset } => {
                let d = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::SLoadDword).def(d).imm(*offset as i64));
                self.pending_smem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::MemTimeLo { .. } => {
                let t = self.new_vreg(RClass::S2);
                self.push(MInst::new(MOp::SMemTime).def(t));
                self.pending_smem.push(self.mir.canon(t));
                self.touch(&[Src::Reg(t)]); // the stamp is read right away: this emits the `s_waitcnt lgkmcnt(0)` (SMEM returns out of order, so it is a full drain)
                let lo = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::SLoPart).def(lo).srcreg(t));
                self.val.insert(dst, lo);
            }
            Op::KernargPtr { offset } => {
                let d = self.new_vreg(RClass::S2);
                self.push(MInst::new(MOp::SLoadDwordx2).def(d).imm(*offset as i64));
                self.pending_smem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::WorkitemId => {
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VReadPhys).def(d).imm(0));
                self.val.insert(dst, d);
            }
            Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ => {
                let phys = match op {
                    Op::WorkgroupIdX => 2,
                    Op::WorkgroupIdY => 3,
                    _ => 4,
                };
                let d = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::SReadPhys).def(d).imm(phys));
                self.val.insert(dst, d);
            }
            Op::AddI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SAddU32, *a, *b)?,
            Op::MulI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SMulI32, *a, *b)?,
            Op::ShlI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SShlB32, *a, *b)?,
            Op::LshrI32(Opnd::Val(a), b) if self.uniform.contains(&dst) && matches!(self.defs.get(a), Some(Op::WorkitemId)) => {
                // wave id: v_lshr then one v_readfirstlane into an SGPR (the value is the same in every lane of the wave)
                let k = self.const_bits(*b).ok_or("wave id shift must be constant")?;
                let tid = self.as_v1(Opnd::Val(*a))?;
                self.touch(&[Src::Reg(tid)]);
                let t = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VLshrB32).def(t).src(Src::Imm(k)).srcreg(tid));
                let s = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::VReadFirstLane).def(s).srcreg(t));
                self.val.insert(dst, s);
            }
            Op::LshrI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SLshrB32, *a, *b)?,
            Op::AndI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SAndB32, *a, *b)?,
            Op::XorI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SXorB32, *a, *b)?,
            Op::MinI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SMinI32, *a, *b)?,
            Op::Select { ty: DType::I32, cond, a, b } if self.uniform.contains(&dst) => self.scalar_select(dst, *cond, *a, *b)?,
            Op::Select { ty: ty @ (DType::Ptr | DType::Rsrc), cond, a, b } => self.scalar_select_wide(dst, *ty, *cond, *a, *b)?,
            Op::ReadFirstLane(Opnd::Val(a)) if self.uniform.contains(a) && self.val.get(a).map_or(false, |r| self.mir.vreg_class(*r) == RClass::S1) => {
                // already wave-uniform in an SGPR: no v_mov + v_readfirstlane round trip
                let r = self.val[a];
                self.val.insert(dst, r);
            }
            Op::AddI32(a, b) if opt_on("addrfold") && self.lazy_add(dst, *a, *b)? => {}
            Op::AddI32(a, b) => self.bin_v1(dst, MOp::VAddU32, *a, *b)?,
            Op::AddF32(a, b) => self.bin_v1(dst, MOp::VAddF32, *a, *b)?,
            Op::SubF32(a, b) => self.bin_v1(dst, MOp::VSubF32, *a, *b)?,
            Op::MulF32(a, b) => self.bin_v1(dst, MOp::VMulF32, *a, *b)?,
            Op::MaxF32(a, b) => self.bin_v1(dst, MOp::VMaxF32, *a, *b)?,
            Op::AndI32(a, b) => self.bin_v1(dst, MOp::VAndB32, *a, *b)?,
            Op::MinI32(a, b) => self.bin_v1(dst, MOp::VMinI32, *a, *b)?,
            Op::Un(o, a) => self.un_v1(dst, un_mop(*o), *a)?,
            Op::Bin(amdgpu_dsl::BinOp::AshrI32, a, b) => self.v_plain(dst, MOp::VAshrI32, &[*b, *a])?,
            Op::Bin(o, a, b) => self.v_plain(dst, bin_mop(*o), &[*a, *b])?,
            Op::Tri(o, a, b, c) => self.v_plain(dst, tri_mop(*o), &[*a, *b, *c])?,
            Op::Cmp { pred, ty, a, b } => self.cmp_i1(dst, cmp_mop(*pred, *ty), *a, *b)?,
            Op::Cvt { op, srcs, sel } => {
                let (n, ty, _) = op.sig();
                let (mop, partial) = cvt_mop(*op);
                let mut rs = Vec::with_capacity(n);
                for s in &srcs[..n] {
                    rs.push(self.as_v1(*s)?);
                }
                let touched: Vec<Src> = rs.iter().map(|r| Src::Reg(*r)).collect();
                self.touch(&touched);
                let d = self.new_vreg(if ty == DType::Frag2 { RClass::V2 } else { RClass::V1 });
                let mut inst = MInst::new(mop).def(d).imm(*sel as i64);
                if partial {
                    // the instruction keeps the destination's other bits: start from the old value, tie it as the last source
                    let old = rs.pop().expect("partial-write conversion has an old operand");
                    self.copy_reg(d, old)?;
                    for r in rs {
                        inst = inst.srcreg(r);
                    }
                    inst = inst.srcreg(d);
                } else {
                    for r in rs {
                        inst = inst.srcreg(r);
                    }
                }
                self.push(inst);
                self.val.insert(dst, d);
            }
            Op::LoadGlobalN { base, byte_off, ty, cache } => {
                let (cls, mop) = global_load_mop(*ty);
                let d = self.new_vreg(cls);
                let rb = self.as_v2(*base)?;
                self.push(MInst::new(mop).def(d).srcreg(rb).imm(*byte_off as i64).cpol(cache.bits()));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadGlobalD16 { base, byte_off, ty, hi, cache } => {
                // the d16 forms are partial writes in the ISA; gfx950 zeroes the other half anyway: start from 0 so both readings agree
                let rb = self.as_v2(*base)?;
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VMovB32).def(d).src(Src::Imm(0)));
                self.push(MInst::new(global_d16_mop(*ty, *hi)).def(d).srcreg(rb).srcreg(d).imm(*byte_off as i64).cpol(cache.bits()));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::BufferLoadN { rsrc, voff, soff, imm, ty, cache } => {
                let (rr, vs, ss) = self.buffer_operands(*rsrc, *voff, *soff)?;
                let (cls, mop) = buffer_load_mop(*ty);
                let d = self.new_vreg(cls);
                self.push(MInst::new(mop).def(d).srcreg(rr).src(vs).src(ss).imm(*imm as i64).cpol(cache.bits()));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadLdsN { addr, ty } => {
                let (a, off) = self.lds_addr(*addr)?;
                let (cls, mop) = lds_load_mop(*ty);
                let d = self.new_vreg(cls);
                self.push(MInst::new(mop).def(d).srcreg(a).imm(off));
                self.pending_lds.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadLdsD16 { addr, ty, hi } => {
                let (a, off) = self.lds_addr(*addr)?;
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VMovB32).def(d).src(Src::Imm(0)));
                self.push(MInst::new(lds_d16_mop(*ty, *hi)).def(d).srcreg(a).srcreg(d).imm(off));
                self.pending_lds.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadLds2 { addr, off0, off1, b64, st64 } => {
                let (a, imm) = self.lds2_addr(*addr, *off0, *off1, *b64, *st64)?;
                let (cls, mop) = match (b64, st64) {
                    (false, false) => (RClass::V2, MOp::LdsRead2B32),
                    (true, false) => (RClass::V4, MOp::LdsRead2B64),
                    (false, true) => (RClass::V2, MOp::LdsRead2St64B32),
                    (true, true) => (RClass::V4, MOp::LdsRead2St64B64),
                };
                let d = self.new_vreg(cls);
                self.push(MInst::new(mop).def(d).srcreg(a).imm(imm));
                self.pending_lds.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::Mfma { kind, a, b, c, scale: Some(s) } if kind.scaled => {
                let (mop, zmop, imm) = scaled_mfma_mop(*kind, s.opsel)?;
                let ra = self.mma_operand(*kind, kind.a, *a)?;
                let rb = self.mma_operand(*kind, kind.b, *b)?;
                let sa = self.as_v1(s.sa)?;
                let sb = self.as_v1(s.sb)?;
                self.touch(&[Src::Reg(sa), Src::Reg(sb)]);
                let n = kind.acc_dwords();
                let cls = if n == 4 { RClass::V4 } else { RClass::V16 };
                match c {
                    Some(c) if !self.is_zero_frag(c) => {
                        let rc = if n == 4 { self.as_v4(*c)? } else { self.as_v16(*c)? };
                        let reusable = matches!(c, Opnd::Val(cv) if self.use_counts.get(cv).copied() == Some(1) && matches!(self.defs.get(cv), Some(Op::Mfma { .. })));
                        let d = if reusable {
                            rc
                        } else {
                            let d = self.new_vreg(cls);
                            self.copy_reg(d, rc)?;
                            d
                        };
                        self.push(MInst::new(mop).def(d).srcreg(ra).srcreg(rb).srcreg(d).srcreg(sa).srcreg(sb).imm(imm));
                        self.val.insert(dst, d);
                    }
                    _ => {
                        let d = self.new_vreg(cls);
                        self.push(MInst::new(zmop).def(d).srcreg(ra).srcreg(rb).srcreg(sa).srcreg(sb).imm(imm));
                        self.val.insert(dst, d);
                    }
                }
            }
            Op::Mfma { kind, a, b, c, scale: _ } => {
                let (mop, zmop) = mfma_mop(*kind)?;
                let ra = self.as_v4(*a)?;
                let rb = self.as_v4(*b)?;
                let n = kind.acc_dwords();
                let cls = if n == 4 { RClass::V4 } else { RClass::V16 };
                match c {
                    Some(c) if !self.is_zero_frag(c) => {
                        let rc = if n == 4 { self.as_v4(*c)? } else { self.as_v16(*c)? };
                        let reusable = matches!(c, Opnd::Val(cv) if self.use_counts.get(cv).copied() == Some(1) && matches!(self.defs.get(cv), Some(Op::Mfma { .. })));
                        let d = if reusable {
                            rc
                        } else {
                            let d = self.new_vreg(cls);
                            self.copy_reg(d, rc)?;
                            d
                        };
                        self.push(MInst::new(mop).def(d).srcreg(ra).srcreg(rb).srcreg(d));
                        self.val.insert(dst, d);
                    }
                    _ => {
                        let d = self.new_vreg(cls);
                        self.push(MInst::new(zmop).def(d).srcreg(ra).srcreg(rb));
                        self.val.insert(dst, d);
                    }
                }
            }
            Op::LoadLdsTrB16 { addr } => {
                let (a, off) = self.lds_addr(*addr)?;
                let d = self.new_vreg(RClass::V2);
                self.push(MInst::new(MOp::LdsReadB64TrB16).def(d).srcreg(a).imm(off));
                self.pending_lds.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::XorI32(a, b) => self.bin_v1(dst, MOp::VXorB32, *a, *b)?,
            Op::MulI32(a, b) => self.bin_v1(dst, MOp::VMulLoU32, *a, *b)?,
            Op::ShlI32(a, b) => {
                // lshlrev encodes the shift in src0, where a literal is legal.
                let shift = self.shift_operand(*b)?;
                let value = self.as_v1(*a)?;
                self.touch(&[Src::Reg(value)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VLshlB32).def(d).src(shift).srcreg(value));
                self.val.insert(dst, d);
            }
            Op::LshrI32(a, b) => {
                let shift = self.shift_operand(*b)?;
                let value = self.as_v1(*a)?;
                self.touch(&[Src::Reg(value)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VLshrB32).def(d).src(shift).srcreg(value));
                self.val.insert(dst, d);
            }
            Op::FmaF32(a, b, c) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                let rc = self.as_v1(*c)?;
                self.touch(&[Src::Reg(ra), Src::Reg(rb), Src::Reg(rc)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VFmaF32).def(d).srcreg(ra).srcreg(rb).srcreg(rc));
                self.val.insert(dst, d);
            }
            Op::PkSubU16(a, b) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VPkSubU16).def(d).srcreg(ra).srcreg(rb));
                self.val.insert(dst, d);
            }
            Op::PermB32(a, b, c) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                let rc = self.as_v1(*c)?;
                self.touch(&[Src::Reg(ra), Src::Reg(rb), Src::Reg(rc)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VPermB32).def(d).srcreg(ra).srcreg(rb).srcreg(rc));
                self.val.insert(dst, d);
            }
            Op::CvtPknormPair(a, b) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VCvtPknormU16).def(d).srcreg(ra).srcreg(rb));
                self.val.insert(dst, d);
            }
            Op::CvtP8x4FromU16(u0, u1, sc) => {
                let r0 = self.as_v1(*u0)?;
                let r1 = self.as_v1(*u1)?;
                let rs = self.as_v1(*sc)?;
                self.touch(&[Src::Reg(r0), Src::Reg(r1), Src::Reg(rs)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VCvtPkFp8Lo).def(d).srcreg(r0).srcreg(rs));
                self.push(MInst::new(MOp::VCvtPkFp8Hi).def(d).srcreg(r1).srcreg(rs).srcreg(d));
                self.val.insert(dst, d);
            }
            Op::Max3F32(a, b, c) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                let rc = self.as_v1(*c)?;
                self.touch(&[Src::Reg(ra), Src::Reg(rb), Src::Reg(rc)]);
                let d = self.new_vreg(RClass::V1);
                self.push(
                    MInst::new(MOp::VMax3F32)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(rc),
                );
                self.val.insert(dst, d);
            }
            Op::ExpF32(a) => {
                // The hardware instruction is exp2, so natural exp is
                // exp2(x * log2(e)).
                let ra = self.as_v1(*a)?;
                self.touch(&[Src::Reg(ra)]);
                let scaled = self.new_vreg(RClass::V1);
                if opt_on("imm") {
                    self.push(MInst::new(MOp::VMulF32).def(scaled).src(Src::Imm(0x3FB8AA3B_u32 as i32 as i64)).srcreg(ra));
                } else {
                    let log2e = self.const_vreg(0x3FB8AA3B_u32 as i32 as i64);
                    self.push(MInst::new(MOp::VMulF32).def(scaled).srcreg(ra).srcreg(log2e));
                }
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VExpF32).def(d).srcreg(scaled));
                self.val.insert(dst, d);
            }
            Op::RcpF32(a) => self.un_v1(dst, MOp::VRcpF32, *a)?,
            Op::Exp2F32(a) => self.un_v1(dst, MOp::VExpF32, *a)?,
            Op::Log2F32(a) => self.un_v1(dst, MOp::VLogF32, *a)?,
            Op::CvtPkBf16F32(a, b) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VCvtPkBf16F32).def(d).srcreg(ra).srcreg(rb));
                self.val.insert(dst, d);
            }
            Op::CvtF16ToF32(a) => self.un_v1(dst, MOp::VCvtF32F16, *a)?,
            Op::CvtF32ToF16(a) => self.un_v1(dst, MOp::VCvtF16F32, *a)?,
            Op::CmpLtI32(a, b) => self.cmp_i1(dst, MOp::VCmpLtI32, *a, *b)?,
            Op::CmpLtF32(a, b) => self.cmp_i1(dst, MOp::VCmpLtF32, *a, *b)?,
            Op::CmpGeI32(a, b) => self.cmp_i1(dst, MOp::VCmpGeI32, *a, *b)?,
            Op::NotI1(a) => {
                let r = self.as_v1(*a)?;
                let one = self.const_vreg(1);
                self.touch(&[Src::Reg(r)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VXorB32).def(d).srcreg(r).srcreg(one));
                self.val.insert(dst, d);
            }
            Op::Select { ty, cond, a, b } => {
                let rc = self.as_v1(*cond)?;
                let zero = self.const_vreg(0);
                self.touch(&[Src::Reg(rc)]);
                self.push(MInst::new(MOp::VCmpNeI32).srcreg(rc).srcreg(zero));
                let ra = match ty {
                    DType::Ptr => return Err("select of pointers is not supported".to_string()),
                    _ => self.as_v1(*a)?,
                };
                // (no SGPR operand here: v_cndmask_b32_e32 reads vcc through the constant bus, so an SGPR source would be a second read)
                let b_src = match (opt_on("imm"), self.const_bits(*b)) {
                    (true, Some(k)) if is_inline_const(k) => Src::Imm(k),
                    _ => Src::Reg(self.as_v1(*b)?),
                };
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VCndmask).def(d).src(b_src).srcreg(ra));
                self.val.insert(dst, d);
            }
            Op::PtrAdd { base, byte_off } => {
                let rb = self.as_v2(*base)?;
                let ro = self.as_v1(*byte_off)?;
                self.touch(&[Src::Reg(ro)]);
                let d = self.new_vreg(RClass::V2);
                self.push(MInst::new(MOp::VPtrAdd).def(d).srcreg(rb).srcreg(ro));
                self.val.insert(dst, d);
            }
            Op::LoadU16 { base, byte_off } => {
                let rb = self.as_v2(*base)?;
                let d = self.new_vreg(RClass::V1);
                self.push(
                    MInst::new(MOp::GlobalLoadU16).def(d).srcreg(rb).imm(*byte_off as i64),
                );
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadF32 { base, byte_off } => {
                let d = self.new_vreg(RClass::V1);
                let rb = self.as_v2(*base)?;
                self.push(MInst::new(MOp::GlobalLoadF32).def(d).srcreg(rb).imm(*byte_off as i64));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadF32Coherent { base, byte_off } => {
                let d = self.new_vreg(RClass::V1);
                let rb = self.as_v2(*base)?;
                self.push(MInst::new(MOp::GlobalLoadF32Sc1).def(d).srcreg(rb).imm(*byte_off as i64));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::BufferRsrc { base, num_records } => {
                let Opnd::Val(bv) = *base else { return Err("buffer descriptor base must be a kernarg pointer value".to_string()) };
                let br = *self.val.get(&bv).ok_or_else(|| format!("value v{bv} not lowered"))?;
                if self.mir.vreg_class(br) != RClass::S2 {
                    return Err("buffer descriptor base must be a scalar (kernarg) pointer: a per-lane pointer cannot form a descriptor".to_string());
                }
                self.touch(&[Src::Reg(br)]);
                let d = self.new_vreg(RClass::S4);
                self.push(MInst::new(MOp::SBufferRsrc).def(d).srcreg(br).imm(*num_records as i64));
                self.val.insert(dst, d);
            }
            Op::BufferRsrcDyn { base, num_records } => {
                let Opnd::Val(bv) = *base else { return Err("buffer descriptor base must be a kernarg pointer value".to_string()) };
                let br = *self.val.get(&bv).ok_or_else(|| format!("value v{bv} not lowered"))?;
                if self.mir.vreg_class(br) != RClass::S2 {
                    return Err("buffer descriptor base must be a scalar (kernarg) pointer".to_string());
                }
                self.touch(&[Src::Reg(br)]);
                let d = self.new_vreg(RClass::S4);
                match self.s_operand(*num_records)? {
                    Src::Imm(k) => self.push(MInst::new(MOp::SBufferRsrc).def(d).srcreg(br).imm(k)),
                    Src::Reg(n) => {
                        self.touch(&[Src::Reg(n)]);
                        self.push(MInst::new(MOp::SBufferRsrcDyn).def(d).srcreg(br).srcreg(n));
                    }
                }
                self.val.insert(dst, d);
            }
            Op::BufferLoad { rsrc, voff, soff, imm, dwords } => {
                let (rr, vs, ss) = self.buffer_operands(*rsrc, *voff, *soff)?;
                let (cls, mop) = match dwords {
                    1 => (RClass::V1, MOp::BufferLoadDwordX1),
                    2 => (RClass::V2, MOp::BufferLoadDwordX2),
                    _ => (RClass::V4, MOp::BufferLoadDwordX4),
                };
                let d = self.new_vreg(cls);
                self.push(MInst::new(mop).def(d).srcreg(rr).src(vs).src(ss).imm(*imm as i64));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::BufferLoadB128x2 { rsrc, voff, soff, imm_lo, imm_hi } => {
                let (rr, vs, ss) = self.buffer_operands(*rsrc, *voff, *soff)?;
                let d = self.new_vreg(RClass::V8);
                for (k, imm) in [(0u8, *imm_lo), (4u8, *imm_hi)] {
                    let part = self.alias_vreg(d, k, RClass::V4);
                    self.push(MInst::new(MOp::BufferLoadDwordX4).def(part).srcreg(rr).src(vs).src(ss).imm(imm as i64));
                }
                // one vmcnt entry per half
                self.pending_vmem.push(self.mir.canon(d));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::AtomicAddU32 { base, byte_off, value } => {
                let d = self.new_vreg(RClass::V1);
                let rb = self.as_v2(*base)?;
                let rv = self.as_v1(*value)?;
                self.touch(&[Src::Reg(rv)]);
                self.push(MInst::new(MOp::GlobalAtomicAddRet).def(d).srcreg(rb).srcreg(rv).imm(*byte_off as i64));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadGlobalB128 { base, byte_off } => {
                let d = self.new_vreg(RClass::V4);
                let rb = self.as_v2(*base)?;
                self.push(MInst::new(MOp::GlobalLoadB128).def(d).srcreg(rb).imm(*byte_off as i64));
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::Frag2(a, b) => {
                let d = self.new_vreg(RClass::V2);
                for (i, o) in [*a, *b].into_iter().enumerate() {
                    let r = self.as_v1(o)?;
                    self.touch(&[Src::Reg(r)]);
                    self.push(MInst::new(MOp::VWriteSub).def(d).srcreg(r).imm(i as i64));
                }
                self.val.insert(dst, d);
            }
            Op::Permlane32Swap { a, b } => {
                let (ra, rb) = (self.as_v1(*a)?, self.as_v1(*b)?);
                self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
                let d = self.new_vreg(RClass::V2);
                let lo = self.alias_vreg(d, 0, RClass::V1);
                let hi = self.alias_vreg(d, 1, RClass::V1);
                self.push(MInst::new(MOp::VMovB32).def(lo).srcreg(ra));
                self.push(MInst::new(MOp::VMovB32).def(hi).srcreg(rb));
                // both operands are written: lo = [a 0..31, b 0..31], hi = [a 32..63, b 32..63]
                self.push(MInst::new(MOp::VPermlane32Swap).def(lo).srcreg(lo).srcreg(hi));
                self.val.insert(dst, d);
            }
            Op::VarSlice4 { var, idx } => {
                let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                self.touch(&[Src::Reg(v)]);
                let d = self.sub_alias(v, 4 * *idx, RClass::V4);
                self.val.insert(dst, d);
            }
            Op::VarSlice8 { var, half } => {
                let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                self.touch(&[Src::Reg(v)]);
                let d = self.sub_alias(v, 8 * *half, RClass::V8);
                self.val.insert(dst, d);
            }
            Op::Frag2Lane { frag, index } => {
                let f = self.as_v2(*frag)?;
                let d = self.alias_vreg(f, *index, RClass::V1);
                self.val.insert(dst, d);
            }
            Op::Frag4(a, b, c, e) => {
                let d = self.new_vreg(RClass::V4);
                for (i, o) in [*a, *b, *c, *e].into_iter().enumerate() {
                    let r = self.as_v1(o)?;
                    self.touch(&[Src::Reg(r)]);
                    self.push(MInst::new(MOp::VWriteSub).def(d).srcreg(r).imm(i as i64));
                }
                self.val.insert(dst, d);
            }
            Op::Frag4Lane { frag, index } => {
                let f = self.as_v4(*frag)?;
                let d = self.alias_vreg(f, *index, RClass::V1);
                self.val.insert(dst, d);
            }
            Op::MfmaF16 { a, b, c } => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                let rc = self.as_v4(*c)?;
                let d = self.new_vreg(RClass::V4);
                // gfx9 MFMA accumulates in place: D and C are the same registers,
                // so materialise the incoming accumulator into D first.
                self.copy_reg(d, rc)?;
                self.push(
                    MInst::new(MOp::VMfma16x16x16F16)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(d),
                );
                self.val.insert(dst, d);
            }
            Op::LoadLdsTrB8 { addr } => {
                let (a, off) = self.lds_addr(*addr)?;
                let d = self.new_vreg(RClass::V2);
                self.push(MInst::new(MOp::LdsReadB64TrB8).def(d).srcreg(a).imm(off));
                self.pending_lds.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadLdsB128 { addr } => {
                let (a, off) = self.lds_addr(*addr)?;
                let d = self.new_vreg(RClass::V4);
                self.push(MInst::new(MOp::LdsReadB128).def(d).srcreg(a).imm(off));
                self.pending_lds.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::DppQuadPerm { x, perm } => {
                let r = self.as_v1(*x)?;
                self.touch(&[Src::Reg(r)]);
                let d = self.new_vreg(RClass::V1);
                let ctl = perm[0] as i64 | (perm[1] as i64) << 2 | (perm[2] as i64) << 4 | (perm[3] as i64) << 6;
                self.push(MInst::new(MOp::VMovDpp).def(d).srcreg(r).imm(ctl));
                self.val.insert(dst, d);
            }
            Op::ReadFirstLane(a) => {
                let r = self.as_v1(*a)?;
                self.touch(&[Src::Reg(r)]);
                let d = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::VReadFirstLane).def(d).srcreg(r));
                self.val.insert(dst, d);
            }
            Op::LoadLdsB64 { addr } => {
                let (a, off) = self.lds_addr(*addr)?;
                let d = self.new_vreg(RClass::V2);
                self.push(MInst::new(MOp::LdsReadB64).def(d).srcreg(a).imm(off));
                self.pending_lds.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadLdsB32 { addr } => {
                let (a, off) = self.lds_addr(*addr)?;
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::LdsReadB32).def(d).srcreg(a).imm(off));
                self.pending_lds.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::MfmaF16Wide { a, b, c } => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                if self.is_zero_frag(c) {
                    // D = A x B: the accumulator is the inline constant 0 (no zero fragment, no copy)
                    let d = self.new_vreg(RClass::V16);
                    self.push(MInst::new(MOp::VMfma32x32x8F16Z).def(d).srcreg(ra).srcreg(rb));
                    self.val.insert(dst, d);
                    return Ok(());
                }
                let rc = self.as_v16(*c)?;
                let d = self.new_vreg(RClass::V16);
                self.copy_reg(d, rc)?;
                self.push(
                    MInst::new(MOp::VMfma32x32x8F16)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(d),
                );
                self.val.insert(dst, d);
            }
            Op::CvtPknormU16(a, b) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                let d = self.new_vreg(RClass::V1);
                self.push(
                    MInst::new(MOp::VCvtPknormU16)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb),
                );
                self.val.insert(dst, d);
            }
            Op::CatFrag8(lo, hi) => {
                let lo_r = self.as_v4(*lo)?;
                let hi_r = self.as_v4(*hi)?;
                let d = self.new_vreg(RClass::V8);
                for i in 0..4u8 {
                    let s = self.alias_vreg(lo_r, i, RClass::V1);
                    let a = self.alias_vreg(d, i, RClass::V1);
                    self.push(MInst::new(MOp::VMovB32).def(a).srcreg(s));
                }
                for i in 0..4u8 {
                    let s = self.alias_vreg(hi_r, i, RClass::V1);
                    let a = self.alias_vreg(d, i + 4, RClass::V1);
                    self.push(MInst::new(MOp::VMovB32).def(a).srcreg(s));
                }
                self.val.insert(dst, d);
            }
            Op::LoadLdsTr4B8 { addrs } => {
                let d = self.new_vreg(RClass::V8);
                for (r, ad) in addrs.iter().enumerate() {
                    let (a, off) = self.lds_addr(*ad)?;
                    let part = self.alias_vreg(d, (2 * r) as u8, RClass::V2);
                    self.push(MInst::new(MOp::LdsReadB64TrB8).def(part).srcreg(a).imm(off));
                }
                for _ in 0..addrs.len() {
                    self.pending_lds.push(d);
                }
                self.val.insert(dst, d);
            }
            Op::LoadLdsB128x2 { lo, hi } => {
                let d = self.new_vreg(RClass::V8);
                let (a_lo, off_lo) = self.lds_addr(*lo)?;
                let part_lo = self.alias_vreg(d, 0, RClass::V4);
                self.push(MInst::new(MOp::LdsReadB128).def(part_lo).srcreg(a_lo).imm(off_lo));
                let (a_hi, off_hi) = self.lds_addr(*hi)?;
                let part_hi = self.alias_vreg(d, 4, RClass::V4);
                self.push(MInst::new(MOp::LdsReadB128).def(part_hi).srcreg(a_hi).imm(off_hi));
                self.pending_lds.push(d);
                self.pending_lds.push(d);
                self.val.insert(dst, d);
            }            Op::Frag8(a, b, c, d, e, f, g, h) => {
                let v = self.new_vreg(RClass::V8);
                let ops8 = [*a, *b, *c, *d, *e, *f, *g, *h];
                for (i, o) in ops8.into_iter().enumerate() {
                    let al = self.alias_vreg(v, i as u8, RClass::V1);
                    let retargeted = match o {
                        Opnd::Val(ov) => self.retarget_producer(ov, al),
                        _ => false,
                    };
                    if !retargeted {
                        let r = self.as_v1(o)?;
                        self.touch(&[Src::Reg(r)]);
                        self.push(MInst::new(MOp::VMovB32).def(al).srcreg(r));
                    }
                }
                self.val.insert(dst, v);
            }
            Op::MfmaFp8_32x32x64 { a, b, c } => {
                let ra = self.as_v8(*a)?;
                let rb = self.as_v8(*b)?;
                if self.is_zero_frag(c) {
                    let d = self.new_vreg(RClass::V16);
                    self.push(MInst::new(MOp::VMfma32x32x64Fp8Z).def(d).srcreg(ra).srcreg(rb));
                    self.val.insert(dst, d);
                    return Ok(());
                }
                let rc = self.as_v16(*c)?;
                let reusable = matches!(
                    c,
                    Opnd::Val(cv) if self.use_counts.get(cv).copied() == Some(1)
                        && matches!(self.defs.get(cv), Some(Op::MfmaScaleFp8_32x32x64 { .. }) | Some(Op::MfmaFp8_32x32x64 { .. }))
                );
                let d = if reusable {
                    rc
                } else {
                    let d = self.new_vreg(RClass::V16);
                    self.copy_reg(d, rc)?;
                    d
                };
                self.push(MInst::new(MOp::VMfma32x32x64Fp8).def(d).srcreg(ra).srcreg(rb).srcreg(d));
                self.val.insert(dst, d);
            }
            Op::MfmaScaleFp8_32x32x64 { a, b, c, scale_a, scale_b } => {
                let ra = self.as_v8(*a)?;
                let rb = self.as_v8(*b)?;
                let c_is_zero = self.is_zero_frag(c);
                let sa = self.as_v1(*scale_a)?;
                let sb = self.as_v1(*scale_b)?;
                if c_is_zero {
                    // D = A x B: the accumulator is the inline constant 0, nothing to initialise
                    let d = self.new_vreg(RClass::V16);
                    self.push(
                        MInst::new(MOp::VMfmaScale32x32x64Fp8Z)
                            .def(d)
                            .srcreg(ra)
                            .srcreg(rb)
                            .srcreg(sa)
                            .srcreg(sb),
                    );
                    self.val.insert(dst, d);
                    return Ok(());
                }
                let d = if c_is_zero {
                    self.fresh_zero16()
                } else {
                    let rc = self.as_v16(*c)?;
                    // A single-use accumulator produced by an MFMA or a fragment build can be
                    // updated in place: no copy and no second sixteen-dword live range.
                    let reusable = matches!(
                        c,
                        Opnd::Val(cv)
                            if self.use_counts.get(cv).copied() == Some(1)
                                && matches!(
                                    self.defs.get(cv),
                                    Some(Op::MfmaScaleFp8_32x32x64 { .. })
                                )
                    );
                    if reusable {
                        rc
                    } else {
                        let d = self.new_vreg(RClass::V16);
                        self.copy_reg(d, rc)?;
                        d
                    }
                };
                self.push(
                    MInst::new(MOp::VMfmaScale32x32x64Fp8)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(d)
                        .srcreg(sa)
                        .srcreg(sb),
                );
                self.val.insert(dst, d);
            }
            Op::Asm { block, inputs } => {
                self.lower_asm(Some(dst), block, inputs, &[])?;
            }
            Op::SmfmacFp8_16x16x128 { a, b, idx, c } => {
                // Sparse MFMA: the accumulator D is also the input C. A (4 dwords) is the
                // 2:4-compressed operand, B (8 dwords) the dense matrix, idx a single dword.
                let ra = self.as_v4(*a)?;
                let rb = self.as_v8(*b)?;
                let ri = self.as_v1(*idx)?;
                let c_is_zero = matches!(
                    c,
                    Opnd::Val(cv) if matches!(self.defs.get(cv), Some(Op::FragZero { ty: DType::Frag4 }))
                );
                let d = if c_is_zero {
                    let d = self.new_vreg(RClass::V4);
                    let zero = self.const_vreg(0);
                    for i in 0..4i64 {
                        self.push(MInst::new(MOp::VWriteSub).def(d).srcreg(zero).imm(i));
                    }
                    d
                } else {
                    let rc = self.as_v4(*c)?;
                    // In-place accumulation when the accumulator has no other use.
                    let reusable = matches!(
                        c,
                        Opnd::Val(cv)
                            if self.use_counts.get(cv).copied() == Some(1)
                                && matches!(
                                    self.defs.get(cv),
                                    Some(Op::SmfmacFp8_16x16x128 { .. }) | Some(Op::Frag4(..))
                                )
                    );
                    if reusable {
                        rc
                    } else {
                        let d = self.new_vreg(RClass::V4);
                        self.copy_reg(d, rc)?;
                        d
                    }
                };
                self.push(
                    // srcs: A, B, SRC2 (sparse index) and the implicit in-place accumulator C (== D), so that liveness,
                    // the scheduler and the hazard passes see that the instruction reads D.
                    MInst::new(MOp::VSmfmac16x16x128Fp8)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(ri)
                        .srcreg(d),
                );
                self.val.insert(dst, d);
            }
            Op::Frag16Lane { frag, index } => {
                // Any multi-dword fragment: read the requested dword.
                let f = match frag {
                    Opnd::Val(v) => {
                        let r = *self.val.get(v).ok_or_else(|| format!("value v{v} not lowered"))?;
                        self.touch(&[Src::Reg(r)]);
                        r
                    }
                    other => self.as_v16(*other)?,
                };
                let d = self.alias_vreg(f, *index, RClass::V1);
                self.val.insert(dst, d);
            }
            Op::Frag2Dup(a) => {
                let ra = self.as_v1(*a)?;
                self.touch(&[Src::Reg(ra)]);
                let d = self.new_vreg(RClass::V2);
                let d0 = self.sub_alias(d, 0, RClass::V1);
                let d1 = self.sub_alias(d, 1, RClass::V1);
                self.push(MInst::new(MOp::VMovB32).def(d0).srcreg(ra));
                self.push(MInst::new(MOp::VMovB32).def(d1).srcreg(ra));
                self.val.insert(dst, d);
            }
            Op::CvtPknormP8x4(a, b, c, e, sc) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                let rc = self.as_v1(*c)?;
                let re = self.as_v1(*e)?;
                let rs = self.as_v1(*sc)?;
                self.touch(&[Src::Reg(ra), Src::Reg(rb), Src::Reg(rc), Src::Reg(re), Src::Reg(rs)]);
                let u0 = self.new_vreg(RClass::V1);
                let u1 = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VCvtPknormU16).def(u0).srcreg(ra).srcreg(rb));
                self.push(MInst::new(MOp::VCvtPknormU16).def(u1).srcreg(rc).srcreg(re));
                let d = self.new_vreg(RClass::V1);
                // low-half form writes bits 0..15; the high-half form keeps them
                self.push(MInst::new(MOp::VCvtPkFp8Lo).def(d).srcreg(u0).srcreg(rs));
                self.push(MInst::new(MOp::VCvtPkFp8Hi).def(d).srcreg(u1).srcreg(rs).srcreg(d));
                self.val.insert(dst, d);
            }
            Op::PartnerMaxF32(a) => {
                let ra = self.as_v1(*a)?;
                self.touch(&[Src::Reg(ra)]);
                let d1 = self.new_vreg(RClass::V1);
                let d2 = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VMovB32).def(d1).srcreg(ra));
                self.push(MInst::new(MOp::VMovB32).def(d2).srcreg(ra));
                // afterwards d1 = [lo, lo], d2 = [hi, hi]
                self.push(MInst::new(MOp::VPermlane32Swap).def(d1).srcreg(d1).srcreg(d2));
                let r = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VMaxF32).def(r).srcreg(d1).srcreg(d2));
                self.val.insert(dst, r);
            }
            Op::PkAddF32(a, b) | Op::PkMulF32(a, b) => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
                let d = self.new_vreg(RClass::V2);
                let mop = if matches!(op, Op::PkAddF32(..)) { MOp::VPkAddF32 } else { MOp::VPkMulF32 };
                self.push(MInst::new(mop).def(d).srcreg(ra).srcreg(rb));
                self.val.insert(dst, d);
            }
            Op::CvtPkFp8x4F32(a, b, c, e) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                let rc = self.as_v1(*c)?;
                let re = self.as_v1(*e)?;
                self.touch(&[Src::Reg(ra), Src::Reg(rb), Src::Reg(rc), Src::Reg(re)]);
                let d = self.new_vreg(RClass::V1);
                // the low-half form writes only 16 bits; the high-half form keeps them
                self.push(MInst::new(MOp::VCvtPkFp8F32).def(d).srcreg(ra).srcreg(rb));
                self.push(MInst::new(MOp::VCvtPkFp8F32Hi).def(d).srcreg(rc).srcreg(re).srcreg(d));
                self.val.insert(dst, d);
            }
            Op::FragZero { ty } => {
                // A zero Frag16 is rematerialised at each use (see as_v16) so that it never
                // stays live across a loop just because every iteration starts from it.
                if matches!(ty, DType::Frag16) {
                    return Ok(());
                }
                let class = match ty {
                    DType::Frag2 => RClass::V2,
                    DType::Frag4 => RClass::V4,
                    DType::Frag16 => RClass::V16,
                    other => return Err(format!("frag_zero: unsupported type {}", other.name())),
                };
                let d = self.new_vreg(class);
                let zero = self.const_vreg(0);
                for i in 0..class.dwords() {
                    self.push(MInst::new(MOp::VWriteSub).def(d).srcreg(zero).imm(i as i64));
                }
                self.val.insert(dst, d);
            }
            Op::BitcastF32(a) | Op::BitcastI32(a) => {
                // Pure register reinterpretation: no instruction.
                let r = self.as_v1(*a)?;
                self.val.insert(dst, r);
            }
        }
        Ok(())
    }

    /// Shift amounts can be an in-line literal or a register.
    fn shift_operand(&mut self, o: Opnd) -> Result<Src, String> {
        match o {
            Opnd::I32(i) => Ok(Src::Imm(i as i64)),
            other => {
                if opt_on("imm") {
                    if let Some(k) = self.const_of(other) {
                        return Ok(Src::Imm(k));
                    }
                }
                Ok(Src::Reg(self.as_v1(other)?))
            }
        }
    }

    /// The SGPR an operand already lives in: an S1 value or a scalar (uniform) loop variable. gfx9 VALU instructions read at most
    /// ONE distinct SGPR (constant bus limit 1) and only in src0 of the e32 forms; every user of this helper keeps to that.
    fn sgpr_src(&self, o: Opnd) -> Option<VReg> {
        match o {
            Opnd::Var(v) => self.vars.get(v.0 as usize).copied().filter(|r| self.mir.vreg_class(*r) == RClass::S1),
            o => self.scalar_reg(o),
        }
    }

    /// The SGPR a value already lives in (S1 class), if any.
    fn scalar_reg(&self, o: Opnd) -> Option<VReg> {
        let Opnd::Val(v) = o else { return None };
        if self.converted.contains_key(&v) {
            return None;
        }
        let r = *self.val.get(&v)?;
        (self.mir.vreg_class(r) == RClass::S1).then_some(r)
    }

    fn bin_v1(&mut self, dst: ValueId, op: MOp, a: Opnd, b: Opnd) -> Result<(), String> {
        if opt_on("sgpr") {
            let vop3 = op == MOp::VMulLoU32;
            let commutative = op != MOp::VSubF32;
            let (sa, sb) = (self.sgpr_src(a), self.sgpr_src(b));
            let plan = match (sa, sb) {
                // both scalar: one operand stays in its SGPR, the other is moved (one v_mov instead of two)
                (Some(s), _) if self.const_bits(b).is_none() => Some((s, b, true)),
                (None, Some(s)) if self.const_bits(a).is_none() && (commutative || vop3) => Some((s, a, false)),
                _ => None,
            };
            if let Some((s, other, scalar_first)) = plan {
                let r = self.as_v1(other)?;
                self.touch(&[Src::Reg(r), Src::Reg(s)]);
                let d = self.new_vreg(RClass::V1);
                let inst = if scalar_first || !commutative {
                    MInst::new(op).def(d).srcreg(s).srcreg(r)
                } else {
                    MInst::new(op).def(d).srcreg(s).srcreg(r)
                };
                self.push(inst);
                self.val.insert(dst, d);
                return Ok(());
            }
        }
        if opt_on("imm") {
            let (ca, cb) = (self.const_bits(a), self.const_bits(b));
            let commutative = op != MOp::VSubF32;
            let pick = match (ca, cb) {
                (None, Some(k)) if commutative && imm_ok(op, k) => Some((k, a)),
                (Some(k), None) if imm_ok(op, k) => Some((k, b)),
                _ => None,
            };
            if let Some((k, other)) = pick {
                let r = self.as_v1(other)?;
                self.touch(&[Src::Reg(r)]);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(op).def(d).src(Src::Imm(k)).srcreg(r));
                self.val.insert(dst, d);
                return Ok(());
            }
        }
        let ra = self.as_v1(a)?;
        let rb = self.as_v1(b)?;
        self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
        let d = self.new_vreg(RClass::V1);
        self.push(MInst::new(op).def(d).srcreg(ra).srcreg(rb));
        self.val.insert(dst, d);
        Ok(())
    }

    /// The register of an MFMA A / B operand of element type `t` (fp4 / 16-bit / i8: 4 dwords; fp6 / bf6: 6 dwords held in a Frag8; fp8 / bf8: 8).
    fn mma_operand(&mut self, k: amdgpu_dsl::MmaKind, t: amdgpu_dsl::MmaTy, o: Opnd) -> Result<VReg, String> {
        if k.operand_dwords(t) == 4 { self.as_v4(o) } else { self.as_v8(o) }
    }

    /// The data register of a store of `ty` (`hi`: the `_d16_hi` forms of 8 / 16-bit stores) and its MOp.
    fn store_value(&mut self, value: Opnd, ty: amdgpu_dsl::MemTy, hi: bool, pick: fn(amdgpu_dsl::MemTy, bool) -> Result<MOp, String>) -> Result<(VReg, MOp), String> {
        use amdgpu_dsl::MemTy as M;
        let mop = pick(ty, hi)?;
        let v = match ty {
            M::B64 => self.as_v2(value)?,
            M::B96 | M::B128 => self.as_v4(value)?,
            _ => self.as_v1(value)?,
        };
        Ok((v, mop))
    }

    /// Address register and `offset0 | offset1 << 8` of a two-address DS access: a constant part of the address is folded into the offsets
    /// when it is a multiple of the stride and both offsets stay within 8 bits, else it is added to the address.
    fn lds2_addr(&mut self, addr: Opnd, off0: u8, off1: u8, b64: bool, st64: bool) -> Result<(VReg, i64), String> {
        let (a, off) = self.lds_addr(addr)?;
        let unit = (if b64 { 8 } else { 4 }) * (if st64 { 64 } else { 1 });
        let (o0, o1) = (off0 as i64, off1 as i64);
        if off == 0 {
            return Ok((a, o0 | (o1 << 8)));
        }
        if off % unit == 0 && o0 + off / unit <= 255 && o1 + off / unit <= 255 && off > 0 {
            let k = off / unit;
            return Ok((a, (o0 + k) | ((o1 + k) << 8)));
        }
        let d = self.new_vreg(RClass::V1);
        self.touch(&[Src::Reg(a)]);
        self.push(MInst::new(MOp::VAddU32).def(d).src(Src::Imm(off)).srcreg(a));
        Ok((d, o0 | (o1 << 8)))
    }

    /// A VALU op with plain V1 operands (constants materialised; e32 / VOP3 forms both accept them).
    fn v_plain(&mut self, dst: ValueId, op: MOp, srcs: &[Opnd]) -> Result<(), String> {
        let mut rs = Vec::with_capacity(srcs.len());
        for s in srcs {
            rs.push(self.as_v1(*s)?);
        }
        let touched: Vec<Src> = rs.iter().map(|r| Src::Reg(*r)).collect();
        self.touch(&touched);
        let d = self.new_vreg(RClass::V1);
        let mut inst = MInst::new(op).def(d);
        for r in rs {
            inst = inst.srcreg(r);
        }
        self.push(inst);
        self.val.insert(dst, d);
        Ok(())
    }

    fn un_v1(&mut self, dst: ValueId, op: MOp, a: Opnd) -> Result<(), String> {
        let ra = self.as_v1(a)?;
        self.touch(&[Src::Reg(ra)]);
        let d = self.new_vreg(RClass::V1);
        self.push(MInst::new(op).def(d).srcreg(ra));
        self.val.insert(dst, d);
        Ok(())
    }

    fn cmp_i1(&mut self, dst: ValueId, op: MOp, a: Opnd, b: Opnd) -> Result<(), String> {
        if opt_on("imm") {
            if let (Some(k), None) = (self.const_bits(a), self.const_bits(b)) {
                let rb = self.as_v1(b)?;
                self.touch(&[Src::Reg(rb)]);
                self.push(MInst::new(op).src(Src::Imm(k)).srcreg(rb));
                let one = self.const_vreg(1);
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VCndmask).def(d).src(Src::Imm(0)).srcreg(one));
                self.val.insert(dst, d);
                return Ok(());
            }
        }
        let sa = if opt_on("sgpr") { self.sgpr_src(a) } else { None };
        let ra = match sa {
            Some(s) => s,
            None => self.as_v1(a)?,
        };
        let rb = self.as_v1(b)?;
        self.touch(&[Src::Reg(ra), Src::Reg(rb)]);
        self.push(MInst::new(op).srcreg(ra).srcreg(rb));
        let one = self.const_vreg(1);
        let d = self.new_vreg(RClass::V1);
        self.push(MInst::new(MOp::VCndmask).def(d).src(Src::Imm(0)).srcreg(one));
        self.val.insert(dst, d);
        Ok(())
    }
}

/// Copy coalescing: `mov a, b` where `b` is produced by the immediately
/// preceding instruction and has no other use is folded into the producer.
/// This keeps loop-carried state from doubling register pressure.
fn coalesce(m: &mut MFunction) -> Vec<(VReg, VReg)> {
    let mut renames: Vec<(VReg, VReg)> = Vec::new();
    let classes: Vec<RClass> = m.vregs.clone();
    let aliased = m.alias.clone();
    let apin = m.agpr_pin.clone();
    let mut uses: HashMap<VReg, u32> = HashMap::new();
    for b in &m.blocks {
        for inst in &b.insts {
            for r in inst.uses() {
                *uses.entry(r).or_insert(0) += 1;
            }
        }
    }
    for b in &mut m.blocks {
        let mut i = 1usize;
        while i < b.insts.len() {
            let (op, dst, srcs) = {
                let inst = &b.insts[i];
                (inst.op, inst.dst, inst.srcs.clone())
            };
            if op == MOp::VMovB32 && srcs.len() == 1 && !b.insts[i].pin && !b.insts[i - 1].pin {
                if let (Some(d), Src::Reg(s)) = (dst, srcs[0]) {
                    let prev = &b.insts[i - 1];
                    let prev_defs_s = prev.dst == Some(s);
                    let single_use = uses.get(&s).copied().unwrap_or(0) == 1;
                    let same_class = classes[d as usize] == classes[s as usize];
                    let prev_is_move_to_s = prev.op == MOp::VMovB32;
                    // A lane of a fragment is not a standalone value: the whole tuple may still be read.
                    let s_is_alias = aliased.contains_key(&s);
                    // an AGPR-requested vreg keeps its own mov (it becomes a v_accvgpr_* op)
                    let agpr_involved = apin.contains_key(&aliased.get(&d).map(|a| a.0).unwrap_or(d)) || apin.contains_key(&aliased.get(&s).map(|a| a.0).unwrap_or(s));
                    if prev_defs_s && single_use && same_class && !prev_is_move_to_s && !s_is_alias && !agpr_involved {
                        if let Some(pd) = b.insts[i - 1].dst.as_mut() {
                            *pd = d;
                        }
                        renames.push((s, d));
                        b.insts.remove(i);
                        continue;
                    }
                }
            }
            i += 1;
        }
    }
    renames
}

/// Dead-code elimination of pure single-result ALU instructions whose result is never read
/// (the folded `x + const` chains of LDS addresses, for instance).
fn dce(m: &mut MFunction) {
    fn pure_alu(op: MOp) -> bool {
        matches!(
            op,
            MOp::VMovB32
                | MOp::VAddU32
                | MOp::VMulLoU32
                | MOp::VAndB32
                | MOp::VMinI32
                | MOp::VOrB32
                | MOp::VXorB32
                | MOp::VLshrB32
                | MOp::VLshlB32
                | MOp::VAddF32
                | MOp::VSubF32
                | MOp::VMulF32
        )
    }
    loop {
        let mut used: HashMap<VReg, u32> = HashMap::new();
        for b in &m.blocks {
            for inst in &b.insts {
                for r in inst.uses() {
                    *used.entry(r).or_insert(0) += 1;
                }
            }
        }
        let mut removed = false;
        let alias = m.alias.clone();
        for b in &mut m.blocks {
            let before = b.insts.len();
            b.insts.retain(|inst| {
                if let (true, Some(d)) = (pure_alu(inst.op), inst.dst) {
                    if !alias.contains_key(&d) && used.get(&d).copied().unwrap_or(0) == 0 {
                        return false;
                    }
                }
                true
            });
            if b.insts.len() != before {
                removed = true;
            }
        }
        if !removed {
            break;
        }
    }
}
/// `select(cmp(x, y), ..)`: lowering materialises `cmp` as a 0/1 value (`v_cmp` + `v_cndmask t, 0, 1`) and
/// the select compares it against zero again. Re-issue the original compare in place of the second one
/// when its operands are unchanged in between; the materialisation then usually becomes dead.
fn fold_select_cmp(m: &mut MFunction) {
    let alias = m.alias.clone();
    let canon = |r: VReg| alias.get(&r).map(|a| a.0).unwrap_or(r);
    for b in &mut m.blocks {
        // t -> (compare op, srcs, instruction index of the compare)
        let mut mat: HashMap<VReg, (MOp, Vec<Src>, usize)> = HashMap::new();
        for i in 0..b.insts.len() {
            let inst = b.insts[i].clone();
            // any redefinition of a recorded 0/1 value invalidates its record (loop variables are not SSA)
            if let Some(d) = inst.dst {
                mat.remove(&d);
                mat.remove(&canon(d));
            }
            // materialisation: v_cndmask t, 0, one  right after a compare (a const mov may sit in between)
            if inst.op == MOp::VCndmask && matches!(inst.srcs.first(), Some(Src::Imm(0))) && inst.dst.is_some() {
                let mut j = i;
                while j > 0 {
                    j -= 1;
                    if b.insts[j].op == MOp::VMovB32 {
                        continue;
                    }
                    break;
                }
                let c = &b.insts[j];
                if matches!(c.op, MOp::VCmpLtI32 | MOp::VCmpLtF32 | MOp::VCmpGeI32 | MOp::VCmpGtI32) && c.dst.is_none() {
                    mat.insert(inst.dst.unwrap(), (c.op, c.srcs.clone(), j));
                }
                continue;
            }
            // consumer: v_cmp_ne t, zero
            if inst.op == MOp::VCmpNeI32 && inst.srcs.len() == 2 {
                if let Src::Reg(t) = inst.srcs[0] {
                    if let Some((op, srcs, cidx)) = mat.get(&t).cloned() {
                        let regs: Vec<VReg> = srcs.iter().filter_map(|s| if let Src::Reg(r) = s { Some(canon(*r)) } else { None }).collect();
                        let redefined = b.insts[cidx + 1..i].iter().any(|x| x.dst.map_or(false, |d| regs.contains(&canon(d))));
                        if !redefined {
                            b.insts[i].op = op;
                            b.insts[i].srcs = srcs;
                        }
                    }
                }
            }
        }
    }
}

/// Remove `v_cmp ; v_cndmask t, 0, one` pairs whose result `t` is never read.
fn dce_cmp_pairs(m: &mut MFunction) {
    let mut used: HashMap<VReg, u32> = HashMap::new();
    for b in &m.blocks {
        for inst in &b.insts {
            for r in inst.uses() {
                *used.entry(r).or_insert(0) += 1;
            }
        }
    }
    let alias = m.alias.clone();
    for b in &mut m.blocks {
        let mut dead = vec![false; b.insts.len()];
        for i in 0..b.insts.len() {
            let inst = &b.insts[i];
            if inst.op == MOp::VCndmask && matches!(inst.srcs.first(), Some(Src::Imm(0))) {
                if let Some(d) = inst.dst {
                    if used.get(&d).copied().unwrap_or(0) == 0 && !alias.contains_key(&d) {
                        let mut j = i;
                        while j > 0 {
                            j -= 1;
                            if b.insts[j].op == MOp::VMovB32 {
                                continue;
                            }
                            break;
                        }
                        if matches!(b.insts[j].op, MOp::VCmpLtI32 | MOp::VCmpLtF32 | MOp::VCmpGeI32 | MOp::VCmpGtI32) && b.insts[j].dst.is_none() {
                            dead[i] = true;
                            dead[j] = true;
                        }
                    }
                }
            }
        }
        let mut k = 0;
        b.insts.retain(|_| {
            k += 1;
            !dead[k - 1]
        });
    }
}

fn dce_full(m: &mut MFunction, loads_only: bool) {
    // op_attrs: `dce` (loads are removable in both modes, pure ALU only in the full mode)
    let removable = |op: MOp| -> bool {
        match crate::op_attrs::attrs(op).dce {
            crate::op_attrs::Dce::Load => true,
            crate::op_attrs::Dce::Pure => !loads_only,
            crate::op_attrs::Dce::Keep => false,
        }
    };
    let defines_vcc = |op: MOp| op.defines_vcc() || op == MOp::VPtrAdd;
    loop {
        let mut used: HashMap<VReg, u32> = HashMap::new();
        for b in &m.blocks {
            for inst in &b.insts {
                for r in inst.uses() {
                    *used.entry(m.canon(r)).or_insert(0) += 1;
                }
            }
        }
        let mut changed = false;
        // SCC read on entry of a block before any SCC write (instructions, then the branch)
        let scc_term = |t: &Term| matches!(t, Term::CondGoto { on: Pred::Scc0 | Pred::Scc1, .. });
        let scc_in: Vec<bool> = m
            .blocks
            .iter()
            .map(|b| {
                for i in &b.insts {
                    let a = crate::op_attrs::attrs(i.op);
                    if a.scc_read {
                        return true;
                    }
                    if a.scc_write {
                        return false;
                    }
                }
                scc_term(&b.term)
            })
            .collect();
        let label_idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
        for bi in 0..m.blocks.len() {
            let mut vcc_live = matches!(m.blocks[bi].term, Term::CondGoto { on: Pred::Vccnz | Pred::Vccz, .. });
            // SCC: a dead SALU result is removable only if no later instruction (or the branch) of the block reads the SCC it writes
            let succ_scc = match &m.blocks[bi].term {
                Term::Goto(t) => label_idx.get(t).map_or(true, |&s| scc_in[s]),
                Term::CondGoto { target, fall, .. } => [target, fall].iter().any(|t| label_idx.get(t).map_or(true, |&s| scc_in[s])),
                Term::Endpgm => false,
            };
            let mut scc_live = scc_term(&m.blocks[bi].term) || succ_scc;
            let n = m.blocks[bi].insts.len();
            let mut dead = vec![false; n];
            for i in (0..n).rev() {
                let inst = &m.blocks[bi].insts[i];
                let dst_unused = match inst.dst {
                    Some(d) => !m.alias.contains_key(&d) && used.get(&m.canon(d)).copied().unwrap_or(0) == 0,
                    None => true,
                };
                let rm = removable(inst.op)
                    && dst_unused
                    && (!defines_vcc(inst.op) || !vcc_live)
                    && (!crate::op_attrs::attrs(inst.op).scc_write || !scc_live)
                    // a v_cndmask/v_addc consumes vcc: removing it is fine only if nothing else depends on its dst (checked above)
                    && !(inst.dst.is_none() && !defines_vcc(inst.op));
                if rm {
                    dead[i] = true;
                    // its own uses no longer count; handled by the outer fixed point
                    continue;
                }
                if defines_vcc(inst.op) {
                    vcc_live = false;
                }
                if inst.op.consumes_vcc() {
                    vcc_live = true;
                }
                let at = crate::op_attrs::attrs(inst.op);
                if at.scc_write {
                    scc_live = false;
                }
                if at.scc_read {
                    scc_live = true;
                }
            }
            if dead.iter().any(|d| *d) {
                changed = true;
                let mut k = 0;
                m.blocks[bi].insts.retain(|_| {
                    k += 1;
                    !dead[k - 1]
                });
            }
        }
        if !changed {
            break;
        }
    }
}
/// Wave-uniform integer values and loop variables (see the `scalarize` option). Control flow is wave-uniform by
/// construction (branches are taken on `vcc != 0`), so a loop variable is uniform when its initial value and every
/// value assigned to it are uniform.
fn uniform_analysis(f: &Function) -> (HashSet<ValueId>, HashSet<amdgpu_dsl::VarId>) {
    let mut uvars: HashSet<amdgpu_dsl::VarId> = HashSet::new();
    for (i, t) in f.var_tys.iter().enumerate() {
        if *t == DType::I32 {
            uvars.insert(amdgpu_dsl::VarId(i as u32));
        }
    }
    loop {
        let mut u: HashSet<ValueId> = HashSet::new();
        let mut wi: HashSet<ValueId> = HashSet::new(); // workitem id values (waveuni)
        let mut cst: HashMap<ValueId, i64> = HashMap::new();
        // `scalarize2`: uniform integer compares (only consumed by uniform selects: the select re-emits them as s_cmp + s_cselect)
        let ext = opt_on("scalarize2");
        let mut ucond: HashSet<ValueId> = HashSet::new();
        let is_u = |o: &Opnd, u: &HashSet<ValueId>, uvars: &HashSet<amdgpu_dsl::VarId>| match o {
            Opnd::I32(_) => true,
            Opnd::Val(v) => u.contains(v),
            Opnd::Var(v) => uvars.contains(v),
            _ => false,
        };
        let mut changed = false;
        for st in f.stmts() {
            match st {
                Stmt::Let { dst, op } => {
                    if matches!(op, Op::WorkitemId) {
                        wi.insert(*dst);
                    }
                    if let Op::ConstI32(k) = op {
                        cst.insert(*dst, *k as i64);
                    }
                    let uni = match op {
                        Op::ConstI32(_) | Op::KernargI32 { .. } | Op::MemTimeLo { .. } | Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ => true,
                        // waveuni: `workitem_id >> k` (k >= 6) is constant within a 64-lane wave (1-D workgroups)
                        Op::LshrI32(Opnd::Val(a), b) if wi.contains(a) && opt_on("waveuni") => match b {
                            Opnd::I32(k) => *k >= 6,
                            Opnd::Val(v) => cst.get(v).map_or(false, |k| *k >= 6),
                            _ => false,
                        },
                        Op::AddI32(a, b) | Op::MulI32(a, b) | Op::ShlI32(a, b) | Op::LshrI32(a, b) | Op::AndI32(a, b) | Op::XorI32(a, b) => {
                            is_u(a, &u, &uvars) && is_u(b, &u, &uvars)
                        }
                        Op::MinI32(a, b) if ext => is_u(a, &u, &uvars) && is_u(b, &u, &uvars),
                        Op::CmpLtI32(a, b) | Op::CmpGeI32(a, b) if ext => {
                            if is_u(a, &u, &uvars) && is_u(b, &u, &uvars) {
                                ucond.insert(*dst);
                            }
                            false
                        }
                        Op::NotI1(Opnd::Val(c)) if ext => {
                            if ucond.contains(c) {
                                ucond.insert(*dst);
                            }
                            false
                        }
                        Op::Select { ty: DType::I32, cond: Opnd::Val(c), a, b } if ext => ucond.contains(c) && is_u(a, &u, &uvars) && is_u(b, &u, &uvars),
                        // the result of v_readfirstlane is wave-uniform by definition
                        Op::ReadFirstLane(_) if ext => true,
                        _ => false,
                    };
                    if uni {
                        u.insert(*dst);
                    }
                }
                Stmt::SetVar { var, value } => {
                    if uvars.contains(var) && !is_u(value, &u, &uvars) {
                        uvars.remove(var);
                        changed = true;
                    }
                }
                Stmt::BeginLoop { vars, init } => {
                    for (v, i) in vars.iter().zip(init.iter()) {
                        if uvars.contains(v) && !is_u(i, &u, &uvars) {
                            uvars.remove(v);
                            changed = true;
                        }
                    }
                }
                // any other statement that writes a variable in place makes it non-scalar
                Stmt::VarScaleF32 { var, .. } | Stmt::VarScalePk { var, .. } | Stmt::VarAddPk { var, .. } | Stmt::VarMfmaF16Z { var, .. }
                | Stmt::VarScaleLane { var, .. } | Stmt::VarMfmaScale { var, .. } | Stmt::VarMfmaScale16 { var, .. } | Stmt::VarMfmaScaleC { var, .. } | Stmt::VarMfmaFp8 { var, .. } | Stmt::VarMfmaFp8Z { var, .. } | Stmt::VarMfmaFp8C { var, .. } | Stmt::VarSmfmac { var, .. } | Stmt::VarMfma { var, .. } | Stmt::VarAsm { var, .. } => {
                    if uvars.remove(var) {
                        changed = true;
                    }
                }
                Stmt::VarAsmN { vars, .. } => {
                    for var in vars {
                        if uvars.remove(var) {
                            changed = true;
                        }
                    }
                }
                _ => {}
            }
        }
        if !changed {
            return (u, uvars);
        }
    }
}
/// Rematerialise constants (`v_mov_b32 d, imm`, single definition) that are defined outside a loop and used inside it:
/// every loop block that uses the constant gets its own `v_mov` right before the first use, so the constant no longer
/// occupies a VGPR across the whole loop. Costs one VALU per (block, constant) per iteration; `max_blocks` limits the
/// constants to those used by at most that many loop blocks.
fn remat_consts(m: &mut MFunction, max_blocks: usize) {
    let idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut in_loop = vec![false; m.blocks.len()];
    for (h, l) in &m.loops {
        if let (Some(&hi), Some(&li)) = (idx.get(h), idx.get(l)) {
            for b in hi.min(li)..=hi.max(li) {
                in_loop[b] = true;
            }
        }
    }
    // single-definition constants
    let mut defs: HashMap<VReg, (usize, i64, u32)> = HashMap::new(); // vreg -> (def block, imm, #defs)
    let mut ndefs: HashMap<VReg, u32> = HashMap::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        for inst in &b.insts {
            if let Some(d) = inst.dst {
                *ndefs.entry(d).or_insert(0) += 1;
                if inst.op == MOp::VMovB32 && inst.srcs.is_empty() {
                    defs.insert(d, (bi, inst.imm, 0));
                }
            }
        }
    }
    let consts: HashMap<VReg, (usize, i64)> = defs
        .into_iter()
        .filter(|(r, (b, _, _))| ndefs.get(r) == Some(&1) && !in_loop[*b] && !m.alias.contains_key(r) && m.vreg_class(*r) == RClass::V1)
        .map(|(r, (b, k, _))| (r, (b, k)))
        .collect();
    // loop blocks using each constant
    let mut users: HashMap<VReg, Vec<usize>> = HashMap::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        if !in_loop[bi] {
            continue;
        }
        for inst in &b.insts {
            for s in &inst.srcs {
                if let Src::Reg(r) = s {
                    if consts.contains_key(r) {
                        let e = users.entry(*r).or_default();
                        if e.last() != Some(&bi) {
                            e.push(bi);
                        }
                    }
                }
            }
        }
    }
    for (c, blocks) in users {
        if blocks.len() > max_blocks {
            continue;
        }
        let k = consts[&c].1;
        for bi in blocks {
            let nr = m.vregs.len() as VReg;
            m.vregs.push(RClass::V1);
            let b = &mut m.blocks[bi];
            let first = b.insts.iter().position(|i| i.srcs.iter().any(|s| matches!(s, Src::Reg(r) if *r == c))).unwrap();
            for inst in b.insts.iter_mut().skip(first) {
                for s in inst.srcs.iter_mut() {
                    if let Src::Reg(r) = s {
                        if *r == c {
                            *r = nr;
                        }
                    }
                }
            }
            b.insts.insert(first, MInst::new(MOp::VMovB32).def(nr).imm(k));
        }
    }
}
/// Rematerialise whole expressions: a pure single-definition V1 value defined outside a loop whose expression tree
/// (add/mul/shift/and/or/xor/mov, depth <= 6) has only constants, single-definition SGPRs and the workitem id as leaves is
/// recomputed in every loop block that uses it, right before the first use, instead of staying in a VGPR across the loop.
/// Last-resort register relief (`remat4`): costs the expression's VALU instructions per block per iteration.
fn remat_exprs(m: &mut MFunction) {
    let idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut in_loop = vec![false; m.blocks.len()];
    for (h, l) in &m.loops {
        if let (Some(&hi), Some(&li)) = (idx.get(h), idx.get(l)) {
            for b in hi.min(li)..=hi.max(li) {
                in_loop[b] = true;
            }
        }
    }
    let mut ndefs: HashMap<VReg, u32> = HashMap::new();
    let mut def: HashMap<VReg, (usize, usize)> = HashMap::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            if let Some(d) = inst.dst {
                *ndefs.entry(d).or_insert(0) += 1;
                def.insert(d, (bi, ii));
            }
        }
    }
    fn pure(op: MOp) -> bool {
        matches!(
            op,
            MOp::VMovB32 | MOp::VAddU32 | MOp::VMulLoU32 | MOp::VAndB32 | MOp::VOrB32 | MOp::VXorB32 | MOp::VLshrB32 | MOp::VLshlB32
        )
    }
    // expression tree of `r` as a list of (vreg, instruction), children first; None if a leaf is not allowed
    fn expr(
        m: &MFunction,
        ndefs: &HashMap<VReg, u32>,
        def: &HashMap<VReg, (usize, usize)>,
        in_loop: &[bool],
        r: VReg,
        depth: u32,
        out: &mut Vec<(VReg, MInst)>,
    ) -> Option<()> {
        if depth > 6 || m.alias.contains_key(&r) || m.vreg_class(r) != RClass::V1 || ndefs.get(&r) != Some(&1) {
            return None;
        }
        let (b, i) = *def.get(&r)?;
        if in_loop[b] {
            return None;
        }
        let inst = m.blocks[b].insts[i].clone();
        if inst.pin || !pure(inst.op) {
            return None;
        }
        for s in &inst.srcs {
            if let Src::Reg(x) = s {
                let xc = m.vreg_class(*x);
                if xc == RClass::S1 {
                    if ndefs.get(x) != Some(&1) {
                        return None; // a loop variable: its value changes
                    }
                    continue;
                }
                // the workitem id: a VReadPhys, kept live by everything else anyway
                if let Some(&(xb, xi)) = def.get(x) {
                    if ndefs.get(x) == Some(&1) && m.blocks[xb].insts[xi].op == MOp::VReadPhys {
                        continue;
                    }
                }
                expr(m, ndefs, def, in_loop, *x, depth + 1, out)?;
            }
        }
        if !out.iter().any(|(v, _)| *v == r) {
            out.push((r, inst));
        }
        Some(())
    }
    let mut cands: Vec<(VReg, Vec<(VReg, MInst)>)> = Vec::new();
    let mut used_in_loop: std::collections::HashSet<VReg> = std::collections::HashSet::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        if in_loop[bi] {
            for inst in &b.insts {
                for r in inst.uses() {
                    used_in_loop.insert(r);
                }
            }
        }
    }
    let mut regs: Vec<VReg> = used_in_loop.into_iter().collect();
    regs.sort();
    for r in regs {
        let mut e = Vec::new();
        if expr(m, &ndefs, &def, &in_loop, r, 0, &mut e).is_some() {
            cands.push((r, e));
        }
    }
    for (c, e) in cands {
        let blocks: Vec<usize> = (0..m.blocks.len())
            .filter(|&bi| in_loop[bi] && m.blocks[bi].insts.iter().any(|i| i.srcs.iter().any(|s| matches!(s, Src::Reg(r) if *r == c))))
            .collect();
        for bi in blocks {
            // fresh copies of the expression, renamed
            let mut map: HashMap<VReg, VReg> = HashMap::new();
            let mut clones: Vec<MInst> = Vec::new();
            for (v, inst) in &e {
                let nr = m.vregs.len() as VReg;
                m.vregs.push(RClass::V1);
                map.insert(*v, nr);
                let mut ni = inst.clone();
                ni.dst = Some(nr);
                for s in ni.srcs.iter_mut() {
                    if let Src::Reg(x) = s {
                        if let Some(n) = map.get(x) {
                            *x = *n;
                        }
                    }
                }
                clones.push(ni);
            }
            let nc = map[&c];
            let b = &mut m.blocks[bi];
            let first = b.insts.iter().position(|i| i.srcs.iter().any(|s| matches!(s, Src::Reg(r) if *r == c))).unwrap();
            for inst in b.insts.iter_mut().skip(first) {
                for s in inst.srcs.iter_mut() {
                    if let Src::Reg(r) = s {
                        if *r == c {
                            *r = nc;
                        }
                    }
                }
            }
            for (k, ci) in clones.into_iter().enumerate() {
                b.insts.insert(first + k, ci);
            }
        }
    }
}
/// LDS spilling of loop-invariant V1 values (last-resort register relief, `ldsspillN` / `AMDC_RA_LDS_SPILL=N`).
/// The N invariants that are used in the fewest loop blocks are stored once, right after their definition, into a per-wave
/// LDS area (`ds_write_b32 [lane*4 + wave_base] offset:k*256`, one 256-byte row per value so consecutive lanes hit consecutive
/// banks) and reloaded with `ds_read_b32` into a short-lived temporary before the first use in every loop block. The value
/// then only lives from its definition to its last use outside loops. Costs one DS read per (value, loop block) per
/// iteration and `256 * values * waves` bytes of LDS (appended after the kernel's own LDS). Returns (#values, total bytes).
fn lds_spill(m: &mut MFunction, count: usize) -> (usize, u32) {
    // stop when the true peak is this low (alignment / reserved registers need slack); AMDC_RA_LDS_SPILL_TARGET lowers it for the regression probe
    let target: u32 = 236;
    let nwaves = (m.workgroup_size / 64).max(1);
    let base = (m.lds_bytes + 15) / 16 * 16;
    let newv = |m: &mut MFunction| -> VReg {
        let r = m.vregs.len() as VReg;
        m.vregs.push(RClass::V1);
        r
    };
    let (tid, wave, creg, wbase, lane, lane4, a1, addr) = (newv(m), newv(m), newv(m), newv(m), newv(m), newv(m), newv(m), newv(m));
    let pre = vec![
        MInst::new(MOp::VReadPhys).def(tid).imm(0),
        MInst::new(MOp::VLshrB32).def(wave).src(Src::Imm(6)).srcreg(tid),
        MInst::new(MOp::VMovB32).def(creg).imm(0), // patched below: bytes per wave
        MInst::new(MOp::VMulLoU32).def(wbase).srcreg(wave).srcreg(creg),
        MInst::new(MOp::VAndB32).def(lane).src(Src::Imm(63)).srcreg(tid),
        MInst::new(MOp::VLshlB32).def(lane4).src(Src::Imm(2)).srcreg(lane),
        MInst::new(MOp::VAddU32).def(a1).srcreg(wbase).srcreg(lane4),
        MInst::new(MOp::VAddU32).def(addr).src(Src::Imm(base as i64)).srcreg(a1),
    ];
    let npre = pre.len();
    for (k, p) in pre.into_iter().enumerate() {
        m.blocks[0].insts.insert(k, p);
    }
    let mut spilled = 0usize;
    for _ in 0..count {
        let (peak, ppos, pblk) = crate::regalloc::pressure_profile(m);
        if peak <= target {
            break;
        }
        let pidx = ppos - m.blocks[..pblk].iter().map(|b| b.insts.len()).sum::<usize>();
        let idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
        let mut in_loop = vec![false; m.blocks.len()];
        for (h, l) in &m.loops {
            if let (Some(&hi), Some(&li)) = (idx.get(h), idx.get(l)) {
                for b in hi.min(li)..=hi.max(li) {
                    in_loop[b] = true;
                }
            }
        }
        let mut ndefs: HashMap<VReg, u32> = HashMap::new();
        let mut def: HashMap<VReg, (usize, usize)> = HashMap::new();
        for (bi, b) in m.blocks.iter().enumerate() {
            for (ii, inst) in b.insts.iter().enumerate() {
                if let Some(d) = inst.dst {
                    *ndefs.entry(d).or_insert(0) += 1;
                    def.insert(d, (bi, ii));
                }
            }
        }
        let mut loop_blocks: HashMap<VReg, std::collections::BTreeSet<usize>> = HashMap::new();
        let mut loop_uses: HashMap<VReg, usize> = HashMap::new();
        for (bi, b) in m.blocks.iter().enumerate() {
            if !in_loop[bi] {
                continue;
            }
            for inst in &b.insts {
                for r in inst.uses() {
                    loop_blocks.entry(r).or_default().insert(bi);
                    *loop_uses.entry(r).or_insert(0) += 1;
                }
            }
        }
        let mut cands: Vec<(usize, usize, VReg)> = Vec::new();
        for (r, blocks) in &loop_blocks {
            let r = *r;
            if r == addr || m.vreg_class(r) != RClass::V1 || m.alias.contains_key(&r) || ndefs.get(&r) != Some(&1) {
                continue;
            }
            let Some(&(db, di)) = def.get(&r) else { continue };
            if in_loop[db] {
                continue;
            }
            if blocks.contains(&pblk) {
                // the reload temporary lives from the first to the last use in the peak block: it only helps if that range
                // does not contain the peak position
                // (straddling values are reloaded before each use instead: per_use below; costs one DS read per use)
            }
            if matches!(m.blocks[db].insts[di].op, MOp::VReadPhys) {
                continue;
            }
            cands.push((blocks.len(), loop_uses[&r], r));
        }
        cands.sort();
        let Some(&(_, _, v)) = cands.first() else { break };
        let off = (spilled as i64) * 256;
        let (db, di) = def[&v];
        m.blocks[db].insts.insert(di + 1, MInst::new(MOp::LdsWriteB32).srcreg(addr).srcreg(v).imm(off));
        let blocks: Vec<usize> = loop_blocks[&v].iter().copied().collect();
        for bi in blocks {
            if bi == pblk {
                let us: Vec<usize> = m.blocks[bi].insts.iter().enumerate().filter(|(_, i)| i.uses().contains(&v)).map(|(k, _)| k).collect();
                if us.first().map_or(false, |f| *f <= pidx) && us.last().map_or(false, |l| *l >= pidx) {
                    // per-use reloads: each temporary lives for one instruction
                    for k in us.into_iter().rev() {
                        let t = newv(m);
                        for s in m.blocks[bi].insts[k].srcs.iter_mut() {
                            if let Src::Reg(r) = s {
                                if *r == v {
                                    *r = t;
                                }
                            }
                        }
                        m.blocks[bi].insts.insert(k, MInst::new(MOp::LdsReadB32).def(t).srcreg(addr).imm(off));
                    }
                    continue;
                }
            }
            let t = newv(m);
            let b = &mut m.blocks[bi];
            let first = b.insts.iter().position(|i| i.srcs.iter().any(|s| matches!(s, Src::Reg(r) if *r == v))).unwrap();
            for inst in b.insts.iter_mut().skip(first) {
                for s in inst.srcs.iter_mut() {
                    if let Src::Reg(r) = s {
                        if *r == v {
                            *r = t;
                        }
                    }
                }
            }
            b.insts.insert(first, MInst::new(MOp::LdsReadB32).def(t).srcreg(addr).imm(off));
        }
        spilled += 1;
    }
    let _ = npre;
    if spilled == 0 {
        // nothing spilled: remove the address prologue again
        for _ in 0..npre {
            m.blocks[0].insts.remove(0);
        }
        return (0, 0);
    }
    let area = (spilled as u32) * 256;
    m.blocks[0].insts[2].imm = area as i64;
    let total = area * nwaves;
    m.lds_bytes = base + total;
    (spilled, total)
}
use crate::asm::mop_by_name;

fn un_mop(o: amdgpu_dsl::UnOp) -> MOp {
    use amdgpu_dsl::UnOp as U;
    match o {
        U::FloorF32 => MOp::VFloorF32,
        U::CeilF32 => MOp::VCeilF32,
        U::TruncF32 => MOp::VTruncF32,
        U::RndneF32 => MOp::VRndneF32,
        U::FractF32 => MOp::VFractF32,
        U::RsqF32 => MOp::VRsqF32,
        U::SqrtF32 => MOp::VSqrtF32,
        U::SinF32 => MOp::VSinF32,
        U::CosF32 => MOp::VCosF32,
        U::CvtI32F32 => MOp::VCvtI32F32,
        U::CvtU32F32 => MOp::VCvtU32F32,
        U::CvtF32I32 => MOp::VCvtF32I32,
        U::CvtF32U32 => MOp::VCvtF32U32,
        U::CvtF32Ubyte0 => MOp::VCvtF32Ubyte0,
        U::CvtF32Ubyte1 => MOp::VCvtF32Ubyte1,
        U::CvtF32Ubyte2 => MOp::VCvtF32Ubyte2,
        U::CvtF32Ubyte3 => MOp::VCvtF32Ubyte3,
    }
}

fn bin_mop(o: amdgpu_dsl::BinOp) -> MOp {
    use amdgpu_dsl::BinOp as B;
    match o {
        B::MinF32 => MOp::VMinF32,
        B::MaxI32 => MOp::VMaxI32,
        B::MaxU32 => MOp::VMaxU32,
        B::MinU32 => MOp::VMinU32,
        B::SubU32 => MOp::VSubU32,
        B::MulU32U24 => MOp::VMulU32U24,
        B::AshrI32 => MOp::VAshrI32,
        B::MulHiU32 => MOp::VMulHiU32,
        B::MulHiI32 => MOp::VMulHiI32,
        B::OrB32 => MOp::VOrB32,
        B::PkAddF16 => MOp::VPkAddF16,
        B::PkMulF16 => MOp::VPkMulF16,
        B::PkMaxF16 => MOp::VPkMaxF16,
        B::PkMinF16 => MOp::VPkMinF16,
    }
}

fn tri_mop(o: amdgpu_dsl::TriOp) -> MOp {
    use amdgpu_dsl::TriOp as T;
    match o {
        T::Min3F32 => MOp::VMin3F32,
        T::Med3F32 => MOp::VMed3F32,
        T::Min3I32 => MOp::VMin3I32,
        T::Max3I32 => MOp::VMax3I32,
        T::Med3I32 => MOp::VMed3I32,
        T::Min3U32 => MOp::VMin3U32,
        T::Max3U32 => MOp::VMax3U32,
        T::Med3U32 => MOp::VMed3U32,
        T::MadU32U24 => MOp::VMadU32U24,
        T::LshlAddU32 => MOp::VLshlAddU32,
        T::Add3U32 => MOp::VAdd3U32,
        T::BfeU32 => MOp::VBfeU32,
        T::BfeI32 => MOp::VBfeI32,
        T::BfiB32 => MOp::VBfiB32,
        T::PkFmaF16 => MOp::VPkFmaF16,
    }
}

/// vcc-writing compare of `pred` on `ty` (`Ne` on integers is `v_cmp_ne_u32` for both signednesses).
fn cmp_mop(p: amdgpu_dsl::CmpPred, t: amdgpu_dsl::CmpTy) -> MOp {
    use amdgpu_dsl::{CmpPred as P, CmpTy as T};
    match (t, p) {
        (T::F32, P::Lt) => MOp::VCmpLtF32,
        (T::F32, P::Le) => MOp::VCmpLeF32,
        (T::F32, P::Gt) => MOp::VCmpGtF32,
        (T::F32, P::Ge) => MOp::VCmpGeF32,
        (T::F32, P::Eq) => MOp::VCmpEqF32,
        (T::F32, P::Ne) => MOp::VCmpNeF32,
        (T::F32, P::Lg) => MOp::VCmpLgF32,
        (T::F32, P::Uno) => MOp::VCmpUF32,
        (T::F32, P::Ord) => MOp::VCmpOF32,
        (T::I32, P::Lt) => MOp::VCmpLtI32,
        (T::I32, P::Le) => MOp::VCmpLeI32,
        (T::I32, P::Gt) => MOp::VCmpGtI32,
        (T::I32, P::Ge) => MOp::VCmpGeI32,
        (T::I32, P::Eq) => MOp::VCmpEqI32,
        (T::U32, P::Lt) => MOp::VCmpLtU32,
        (T::U32, P::Le) => MOp::VCmpLeU32,
        (T::U32, P::Gt) => MOp::VCmpGtU32,
        (T::U32, P::Ge) => MOp::VCmpGeU32,
        (T::U32, P::Eq) => MOp::VCmpEqU32,
        (_, P::Ne) => MOp::VCmpNeI32,
        // float-only predicates on integers are rejected by the IR verifier
        (_, P::Lg | P::Uno | P::Ord) => MOp::VCmpNeI32,
    }
}

fn global_load_mop(t: amdgpu_dsl::MemTy) -> (RClass, MOp) {
    use amdgpu_dsl::MemTy as M;
    match t {
        M::U8 => (RClass::V1, MOp::GlobalLoadU8),
        M::I8 => (RClass::V1, MOp::GlobalLoadI8),
        M::U16 => (RClass::V1, MOp::GlobalLoadU16),
        M::I16 => (RClass::V1, MOp::GlobalLoadI16),
        M::B32 => (RClass::V1, MOp::GlobalLoadF32),
        M::B64 => (RClass::V2, MOp::GlobalLoadB64),
        M::B96 => (RClass::V4, MOp::GlobalLoadB96),
        M::B128 => (RClass::V4, MOp::GlobalLoadB128),
    }
}

fn global_d16_mop(t: amdgpu_dsl::D16Ty, hi: bool) -> MOp {
    use amdgpu_dsl::D16Ty as D;
    match (t, hi) {
        (D::U8, false) => MOp::GlobalLoadD16U8,
        (D::U8, true) => MOp::GlobalLoadD16U8Hi,
        (D::I8, false) => MOp::GlobalLoadD16I8,
        (D::I8, true) => MOp::GlobalLoadD16I8Hi,
        (D::B16, false) => MOp::GlobalLoadD16B16,
        (D::B16, true) => MOp::GlobalLoadD16B16Hi,
    }
}

fn buffer_load_mop(t: amdgpu_dsl::MemTy) -> (RClass, MOp) {
    use amdgpu_dsl::MemTy as M;
    match t {
        M::U8 => (RClass::V1, MOp::BufferLoadU8),
        M::I8 => (RClass::V1, MOp::BufferLoadI8),
        M::U16 => (RClass::V1, MOp::BufferLoadU16),
        M::I16 => (RClass::V1, MOp::BufferLoadI16),
        M::B32 => (RClass::V1, MOp::BufferLoadDwordX1),
        M::B64 => (RClass::V2, MOp::BufferLoadDwordX2),
        M::B96 => (RClass::V4, MOp::BufferLoadB96),
        M::B128 => (RClass::V4, MOp::BufferLoadDwordX4),
    }
}

fn lds_load_mop(t: amdgpu_dsl::MemTy) -> (RClass, MOp) {
    use amdgpu_dsl::MemTy as M;
    match t {
        M::U8 => (RClass::V1, MOp::LdsReadU8),
        M::I8 => (RClass::V1, MOp::LdsReadI8),
        M::U16 => (RClass::V1, MOp::LdsReadU16),
        M::I16 => (RClass::V1, MOp::LdsReadI16),
        M::B32 => (RClass::V1, MOp::LdsReadB32),
        M::B64 => (RClass::V2, MOp::LdsReadB64),
        M::B96 => (RClass::V4, MOp::LdsReadB96),
        M::B128 => (RClass::V4, MOp::LdsReadB128),
    }
}

fn lds_d16_mop(t: amdgpu_dsl::D16Ty, hi: bool) -> MOp {
    use amdgpu_dsl::D16Ty as D;
    match (t, hi) {
        (D::U8, false) => MOp::LdsReadD16U8,
        (D::U8, true) => MOp::LdsReadD16U8Hi,
        (D::I8, false) => MOp::LdsReadD16I8,
        (D::I8, true) => MOp::LdsReadD16I8Hi,
        (D::B16, false) => MOp::LdsReadD16B16,
        (D::B16, true) => MOp::LdsReadD16B16Hi,
    }
}

fn global_store_mop(t: amdgpu_dsl::MemTy, hi: bool) -> Result<MOp, String> {
    use amdgpu_dsl::MemTy as M;
    Ok(match (t, hi) {
        (M::U8 | M::I8, false) => MOp::GlobalStoreB8,
        (M::U8 | M::I8, true) => MOp::GlobalStoreB8Hi,
        (M::U16 | M::I16, false) => MOp::GlobalStoreU16,
        (M::U16 | M::I16, true) => MOp::GlobalStoreB16Hi,
        (M::B32, false) => MOp::GlobalStoreF32,
        (M::B64, false) => MOp::GlobalStoreF32x2,
        (M::B96, false) => MOp::GlobalStoreB96,
        (M::B128, false) => MOp::GlobalStoreF32x4,
        (t, true) => return Err(format!("a d16_hi store of {t:?}: only 8 / 16-bit stores have a hi form")),
    })
}

fn buffer_store_mop(t: amdgpu_dsl::MemTy, hi: bool) -> Result<MOp, String> {
    use amdgpu_dsl::MemTy as M;
    Ok(match (t, hi) {
        (M::U8 | M::I8, false) => MOp::BufferStoreB8,
        (M::U8 | M::I8, true) => MOp::BufferStoreB8Hi,
        (M::U16 | M::I16, false) => MOp::BufferStoreB16,
        (M::U16 | M::I16, true) => MOp::BufferStoreB16Hi,
        (M::B32, false) => MOp::BufferStoreDwordX1,
        (M::B64, false) => MOp::BufferStoreDwordX2,
        (M::B96, false) => MOp::BufferStoreB96,
        (M::B128, false) => MOp::BufferStoreDwordX4,
        (t, true) => return Err(format!("a d16_hi store of {t:?}: only 8 / 16-bit stores have a hi form")),
    })
}

fn lds_store_mop(t: amdgpu_dsl::MemTy, hi: bool) -> Result<MOp, String> {
    use amdgpu_dsl::MemTy as M;
    Ok(match (t, hi) {
        (M::U8 | M::I8, false) => MOp::LdsWriteB8,
        (M::U8 | M::I8, true) => MOp::LdsWriteB8Hi,
        (M::U16 | M::I16, false) => MOp::LdsWriteB16,
        (M::U16 | M::I16, true) => MOp::LdsWriteB16Hi,
        (M::B32, false) => MOp::LdsWriteB32,
        (M::B64, false) => MOp::LdsWriteB64,
        (M::B96, false) => MOp::LdsWriteB96,
        (M::B128, false) => MOp::LdsWriteB128,
        (t, true) => return Err(format!("a d16_hi store of {t:?}: only 8 / 16-bit stores have a hi form")),
    })
}

/// (C form, zero-C form) MOps of an MFMA kind.
fn mfma_mop(k: amdgpu_dsl::MmaKind) -> Result<(MOp, MOp), String> {
    use amdgpu_dsl::{MmaShape as S, MmaTy as T};
    if k.scaled || k.a != k.b {
        return Err(format!("mfma {k:?}: not supported yet"));
    }
    Ok(match (k.shape, k.a) {
        (S::M16N16K32, T::Bf16) => (MOp::VMfma16x16x32Bf16, MOp::VMfma16x16x32Bf16Z),
        (S::M32N32K16, T::Bf16) => (MOp::VMfma32x32x16Bf16, MOp::VMfma32x32x16Bf16Z),
        (S::M16N16K32, T::F16) => (MOp::VMfma16x16x32F16, MOp::VMfma16x16x32F16Z),
        (S::M32N32K16, T::F16) => (MOp::VMfma32x32x16F16, MOp::VMfma32x32x16F16Z),
        (S::M16N16K64, T::I8) => (MOp::VMfma16x16x64I8, MOp::VMfma16x16x64I8Z),
        (S::M32N32K32, T::I8) => (MOp::VMfma32x32x32I8, MOp::VMfma32x32x32I8Z),
        _ => return Err(format!("mfma {k:?}: not supported yet")),
    })
}

/// (C form, zero-C form, imm) of a block-scaled f8f6f4 MFMA with A / B format select (imm = opsel | cbsz << 8 | blgp << 12).
fn scaled_mfma_mop(k: amdgpu_dsl::MmaKind, opsel: u8) -> Result<(MOp, MOp, i64), String> {
    use amdgpu_dsl::{MmaKind, MmaShape as S};
    let (Some(ca), Some(cb)) = (MmaKind::f8f6f4_code(k.a), MmaKind::f8f6f4_code(k.b)) else {
        return Err(format!("mfma {k:?}: scaled MFMA needs f8f6f4 operand types"));
    };
    let size = |t| k.operand_dwords(t);
    let imm = (opsel as i64 & 0x33) | ((ca as i64) << 8) | ((cb as i64) << 12);
    let m = |c: MOp, z: MOp| Ok((c, z, imm));
    match (k.shape, size(k.a), size(k.b)) {
        (S::M16N16K128, 4, 4) => m(MOp::VMfmaScale16x16x128A4B4, MOp::VMfmaScale16x16x128A4B4Z),
        (S::M16N16K128, 4, 6) => m(MOp::VMfmaScale16x16x128A4B6, MOp::VMfmaScale16x16x128A4B6Z),
        (S::M16N16K128, 4, 8) => m(MOp::VMfmaScale16x16x128A4B8, MOp::VMfmaScale16x16x128A4B8Z),
        (S::M16N16K128, 6, 4) => m(MOp::VMfmaScale16x16x128A6B4, MOp::VMfmaScale16x16x128A6B4Z),
        (S::M16N16K128, 6, 6) => m(MOp::VMfmaScale16x16x128A6B6, MOp::VMfmaScale16x16x128A6B6Z),
        (S::M16N16K128, 6, 8) => m(MOp::VMfmaScale16x16x128A6B8, MOp::VMfmaScale16x16x128A6B8Z),
        (S::M16N16K128, 8, 4) => m(MOp::VMfmaScale16x16x128A8B4, MOp::VMfmaScale16x16x128A8B4Z),
        (S::M16N16K128, 8, 6) => m(MOp::VMfmaScale16x16x128A8B6, MOp::VMfmaScale16x16x128A8B6Z),
        (S::M16N16K128, 8, 8) => m(MOp::VMfmaScale16x16x128A8B8, MOp::VMfmaScale16x16x128A8B8Z),
        (S::M32N32K64, 4, 4) => m(MOp::VMfmaScale32x32x64A4B4, MOp::VMfmaScale32x32x64A4B4Z),
        (S::M32N32K64, 4, 6) => m(MOp::VMfmaScale32x32x64A4B6, MOp::VMfmaScale32x32x64A4B6Z),
        (S::M32N32K64, 4, 8) => m(MOp::VMfmaScale32x32x64A4B8, MOp::VMfmaScale32x32x64A4B8Z),
        (S::M32N32K64, 6, 4) => m(MOp::VMfmaScale32x32x64A6B4, MOp::VMfmaScale32x32x64A6B4Z),
        (S::M32N32K64, 6, 6) => m(MOp::VMfmaScale32x32x64A6B6, MOp::VMfmaScale32x32x64A6B6Z),
        (S::M32N32K64, 6, 8) => m(MOp::VMfmaScale32x32x64A6B8, MOp::VMfmaScale32x32x64A6B8Z),
        (S::M32N32K64, 8, 4) => m(MOp::VMfmaScale32x32x64A8B4, MOp::VMfmaScale32x32x64A8B4Z),
        (S::M32N32K64, 8, 6) => m(MOp::VMfmaScale32x32x64A8B6, MOp::VMfmaScale32x32x64A8B6Z),
        (S::M32N32K64, 8, 8) => m(MOp::VMfmaScale32x32x64A8B8, MOp::VMfmaScale32x32x64A8B8Z),
        _ => Err(format!("mfma {k:?}: not a scaled f8f6f4 shape")),
    }
}

/// (MOp, partial write with the old value as the last operand) of a conversion.
fn cvt_mop(o: amdgpu_dsl::CvtOp) -> (MOp, bool) {
    use amdgpu_dsl::CvtOp as C;
    match o {
        C::PkF32Fp8 => (MOp::VCvtPkF32Fp8, false),
        C::PkF32Bf8 => (MOp::VCvtPkF32Bf8, false),
        C::F32Fp8 => (MOp::VCvtF32Fp8, false),
        C::F32Bf8 => (MOp::VCvtF32Bf8, false),
        C::PkBf8F32 => (MOp::VCvtPkBf8F32, true),
        C::ScalePkBf16Fp8 => (MOp::VCvtScalePkBf16Fp8, false),
        C::ScalePkF16Fp8 => (MOp::VCvtScalePkF16Fp8, false),
        C::ScalePkBf16Bf8 => (MOp::VCvtScalePkBf16Bf8, false),
        C::ScalePkF16Bf8 => (MOp::VCvtScalePkF16Bf8, false),
        C::ScalePkF32Fp8 => (MOp::VCvtScalePkF32Fp8, false),
        C::ScalePkF32Bf8 => (MOp::VCvtScalePkF32Bf8, false),
        C::ScalePkF32Fp4 => (MOp::VCvtScalePkF32Fp4, false),
        C::ScalePkBf16Fp4 => (MOp::VCvtScalePkBf16Fp4, false),
        C::ScalePkF16Fp4 => (MOp::VCvtScalePkF16Fp4, false),
        C::ScalePkFp8F32 => (MOp::VCvtScalePkFp8F32, true),
        C::ScalePkBf8F32 => (MOp::VCvtScalePkBf8F32, true),
        C::ScalePkFp4F32 => (MOp::VCvtScalePkFp4F32, true),
        C::PkU8F32 => (MOp::VCvtPkU8F32, false),
        C::PkI16I32 => (MOp::VCvtPkI16I32, false),
        C::PkU16U32 => (MOp::VCvtPkU16U32, false),
    }
}
