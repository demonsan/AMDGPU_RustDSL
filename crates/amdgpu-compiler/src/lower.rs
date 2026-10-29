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

use amdgpu_dsl::{DType, Function, Op, Opnd, Stmt, ValueId};
use crate::mir::*;

/// Compiler optimisations that reduce dynamic VALU count (accuracy-preserving: same values, same
/// arithmetic). Selected with `AMDC_LOWER_OPTS=imm,cmpfold` (`all`, `none`); DEFAULT `dce,cmpfold`.
///  * `imm`     - constants become in-line / literal operands (VOP2 src0, shift amounts, cndmask false arm)
///                instead of `v_mov` into a register;
///  * `sgpr`    - scalar (SGPR) values are used directly as VALU source operands instead of a `v_mov` copy;
///  * `shl`     - multiplication by a power-of-two constant becomes a shift;
///  * `scalarize` - wave-uniform integer arithmetic (workgroup ids, kernarg values, constants and loop counters /
///                 toggles derived from them with add/mul/shl/lshr/and/xor) is computed with SALU instructions in SGPRs;
///                 a value needed by a VALU instruction is copied with one `v_mov` (cached per region);
///  * `addrfold` - an address `(base + k1) + (y + k2)` used once is formed as `base + y` with `k1+k2` in the DS
///                 offset field, so the `base + k` variants no longer stay live in VGPRs;
///  * `saddr`   - global loads whose address is `kernarg_ptr + vector offset` use the scalar-base form (no 64-bit add);
///  * `dce`     - dead address / clamp chains (`v_add_co`/`v_addc`, `v_cmp`+`v_cndmask`, pointer adds) are removed;
///  * `cmpfold` - `select(cmp(x,y), ..)` re-issues the compare instead of the 0/1 materialisation +
///                `v_cmp_ne`, and the dead materialisation is removed.
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

pub fn opt_on(name: &str) -> bool {
    if let Some(v) = OVERRIDE.with(|o| o.borrow().clone()) {
        return v.iter().any(|x| x == name || x == "all");
    }
    static OPTS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let v = OPTS.get_or_init(|| match std::env::var("AMDC_LOWER_OPTS") {
        Ok(s) if !s.is_empty() => s.split(',').map(|x| x.trim().to_string()).collect(),
        Ok(_) => vec!["dce".to_string(), "cmpfold".to_string()], // cargo.sh forwards an empty value when unset
        // default: the optimisations measured to help (value-preserving, bit-exact on all shapes); `none` disables all
        Err(_) => vec!["dce".to_string(), "cmpfold".to_string()],
    });
    v.iter().any(|x| x == name || x == "all")
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
        MOp::VAddU32 | MOp::VAddF32 | MOp::VSubF32 | MOp::VMulF32 | MOp::VMaxF32 | MOp::VAndB32 | MOp::VXorB32 | MOp::VOrB32 => true,
        _ => false,
    }
}

pub fn lower(f: &Function) -> Result<MFunction, String> {
    let errs = amdgpu_dsl::verify(f);
    if !errs.is_empty() {
        return Err(format!(
            "IR verification failed:\n  {}\nIR:\n{}",
            errs.join("\n  "),
            f.dump()
        ));
    }
    let mut lw = Lowerer::new(f);
    if opt_on("scalarize") {
        let (u, v) = uniform_analysis(f);
        lw.uniform = u;
        lw.uvars = v;
    }
    lw.run(f)?;
    Ok(lw.finish())
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
    /// Cache of sub-register views: (tuple, dword offset, class) -> alias vreg.
    alias_cache: HashMap<(VReg, u8, RClass), VReg>,
    /// Labels of else-blocks that were actually started (the others are synthesised at `endif`).
    else_started: HashSet<u32>,
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
        };
        mir.blocks.push(Block::new(0));

        let mut defs = HashMap::new();
        let mut use_counts: HashMap<ValueId, usize> = HashMap::new();
        let mut count = |o: &Opnd| {
            if let Opnd::Val(v) = o {
                *use_counts.entry(*v).or_insert(0) += 1;
            }
        };
        for st in &f.stmts {
            match st {
                Stmt::Let { dst, op } => {
                    defs.insert(*dst, op.clone());
                    for o in op.operands() {
                        count(&o);
                    }
                }
                Stmt::StoreU16 { base, value, .. } | Stmt::StoreF32 { base, value, .. } => {
                    count(base);
                    count(value);
                }
                Stmt::StoreLdsB32 { addr, value } | Stmt::StoreLdsB128 { addr, value } => {
                    count(addr);
                    count(value);
                }
                Stmt::SetVar { value, .. } => count(value),
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
            alias_cache: HashMap::new(),
            else_started: HashSet::new(),
            uniform: HashSet::new(),
            uvars: HashSet::new(),
            lazy: HashMap::new(),
        }
    }

    fn finish(mut self) -> MFunction {
        if !self.pending_vmem.is_empty() {
            self.push(MInst::new(MOp::SWaitVm).auto());
        }
        if !self.pending_smem.is_empty() || !self.pending_lds.is_empty() {
            self.push(MInst::new(MOp::SWaitLgkm).imm(0).auto());
        }
        self.set_term(Term::Endpgm);
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
        coalesce(&mut self.mir);
        dce(&mut self.mir);
        self.mir
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

    fn push(&mut self, inst: MInst) {
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
        for s in srcs {
            if let Src::Reg(r) = s {
                let r = &self.mir.canon(*r);
                if let Some(pos) = self.pending_vmem.iter().rposition(|x| x == r) {
                    vm_newest = Some(vm_newest.map_or(pos, |p| p.max(pos)));
                }
                if self.pending_smem.contains(r) || self.pending_lds.contains(r) {
                    need_lgkm = true;
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

    /// Make the instruction that produced value `v` write `new_dst` directly (instead of a fresh
    /// register that would then be moved into a fragment). Only for single-use results of plain
    /// ALU ops defined earlier in the current block.
    fn retarget_producer(&mut self, v: ValueId, new_dst: VReg) -> bool {
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
        for inst in insts.iter_mut().rev() {
            if inst.dst == Some(r) {
                inst.dst = Some(new_dst);
                self.val.insert(cur, new_dst);
                return true;
            }
            if inst.uses().contains(&r) {
                return false;
            }
        }
        false
    }

    /// Bit pattern of a constant operand (integer or f32), for in-line operands.
    /// `base` = `PtrAdd{ scalar kernarg pointer, vector byte offset }`: (scalar pair, 32-bit vector offset) for the SADDR form.
    fn saddr_parts(&mut self, base: Opnd) -> Result<Option<(VReg, VReg)>, String> {
        if !opt_on("saddr") {
            return Ok(None);
        }
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
                    if std::env::var("AMDC_ADDRFOLD_NODS").map(|v| v.is_empty()).unwrap_or(true) && (0..65536).contains(&(off + k)) {
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
        let mode = std::env::var("AMDC_ADDRFOLD_MODE").unwrap_or_default();
        if let Some(lim) = std::env::var("AMDC_ADDRFOLD_LIMIT").ok().and_then(|v| v.parse::<usize>().ok()) {
            if self.lazy.len() >= lim {
                return Ok(false);
            }
        }
        match (ka, kb) {
            (None, None) if mode == "c" => Ok(false),
            (None, Some(_)) | (Some(_), None) if mode == "n" => Ok(false),
            (None, Some(k)) | (Some(k), None) => {
                let other = if kb.is_some() { a } else { b };
                let (t, k0) = self.lazy_core(other)?;
                if self.var_backed(t) {
                    return Ok(false);
                }
                if std::env::var("AMDC_ADDRFOLD_TRACE").map(|v| !v.is_empty()).unwrap_or(false) {
                    eprintln!("LAZY#{} v{dst} = {:?} + {:?}: core r{t} class {:?} k={}", self.lazy.len(), a, b, self.mir.vreg_class(t), k0 + k);
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
            _ => Err("eight-dword fragment operand must be an SSA value".to_string()),
        }
    }

    /// Scalar operand for instructions that take SGPRs (e.g. MFMA scales).
    fn as_s1(&mut self, o: Opnd) -> Result<VReg, String> {
        match o {
            Opnd::Val(v) => {
                let r = *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?;
                if self.mir.vreg_class(r) == RClass::S1 {
                    Ok(r)
                } else {
                    Err(format!("value v{v} is not a scalar register"))
                }
            }
            Opnd::I32(i) => {
                let d = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::SMovB32).def(d).imm(i as i64));
                Ok(d)
            }
            _ => Err("scalar operand must be a value or immediate".to_string()),
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

    fn run(&mut self, f: &Function) -> Result<(), String> {
        for st in &f.stmts {
            match st {
                Stmt::Let { dst, op } => self.lower_op(*dst, op)?,
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
                Stmt::SetVar { var, value } => {
                    let dst = self.vars[var.0 as usize];
                    let class = self.mir.vreg_class(dst);
                    if class == RClass::S1 {
                        let s = self.s_operand(*value)?;
                        if let Src::Reg(x) = s {
                            self.touch(&[Src::Reg(x)]);
                        }
                        self.push(MInst::new(MOp::SMovB32).def(dst).src(s));
                        continue;
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
                Stmt::VarLoadB128 { var, base, byte_off } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    if let Some((sr, off)) = self.saddr_parts(*base)? {
                        self.push(MInst::new(MOp::GlobalLoadB128S).def(v).srcreg(sr).srcreg(off).imm(*byte_off as i64));
                    } else {
                        let rb = self.as_v2(*base)?;
                        self.push(MInst::new(MOp::GlobalLoadB128).def(v).srcreg(rb).imm(*byte_off as i64));
                    }
                    self.pending_vmem.push(self.mir.canon(v));
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
                Stmt::VarMfmaScaleZ { var, a, b, scale_a, scale_b } => {
                    let v = *self.vars.get(var.0 as usize).ok_or("var not lowered")?;
                    let ra = self.as_v8(*a)?;
                    let rb = self.as_v8(*b)?;
                    let sa = self.as_v1(*scale_a)?;
                    let sb = self.as_v1(*scale_b)?;
                    self.push(MInst::new(MOp::VMfmaScale32x32x64Fp8Z).def(v).srcreg(ra).srcreg(rb).srcreg(sa).srcreg(sb));
                }
                Stmt::VarMfmaScale { var, a, b, scale_a, scale_b } => {
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
                            .srcreg(sb),
                    );
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
                Stmt::StoreLdsB32 { addr, value } => {
                    let (a, off) = self.lds_addr(*addr)?;
                    let v = self.as_v1(*value)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(MInst::new(MOp::LdsWriteB32).srcreg(a).srcreg(v).imm(off));
                }
                Stmt::WaitLds { count } => {
                    self.push(MInst::new(MOp::SWaitLgkm).imm(*count as i64));
                }
                Stmt::Barrier => {
                    self.push(MInst::new(MOp::SBarrier));
                }
                Stmt::WaitVm { count } => {
                    self.push(MInst::new(MOp::SWaitVm).imm(*count as i64));
                }
                Stmt::LoadGlobalLds { base, lds_addr, byte_off, dwords } => {
                    let rb = self.as_v2(*base)?;
                    let lv = self.as_v1(*lds_addr)?;
                    self.touch(&[Src::Reg(rb), Src::Reg(lv)]);
                    let sreg = match self.rfl_cache.get(&lv) {
                        Some(s) => *s,
                        None => {
                            let s = self.new_vreg(RClass::S1);
                            self.push(MInst::new(MOp::VReadFirstLane).def(s).srcreg(lv));
                            self.rfl_cache.insert(lv, s);
                            s
                        }
                    };
                    let mop = if *dwords == 4 { MOp::GlobalLoadLdsB128 } else { MOp::GlobalLoadLdsB32 };
                    // imm = instruction offset: added to the global address AND to the LDS destination (M0 + imm + 16 lane)
                    self.push(MInst::new(mop).srcreg(rb).srcreg(sreg).imm(*byte_off as i64));
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
                match op {
                    Op::CmpLtI32(a, b) => {
                        let ra = self.as_v1(a)?;
                        let rb = self.as_v1(b)?;
                        self.push(MInst::new(MOp::VCmpLtI32).srcreg(ra).srcreg(rb));
                        let on = if negate { Pred::Vccz } else { Pred::Vccnz };
                        return Ok((on, vec![Src::Reg(ra), Src::Reg(rb)]));
                    }
                    Op::CmpGeI32(a, b) => {
                        let ra = self.as_v1(a)?;
                        let rb = self.as_v1(b)?;
                        self.push(MInst::new(MOp::VCmpGeI32).srcreg(ra).srcreg(rb));
                        let on = if negate { Pred::Vccz } else { Pred::Vccnz };
                        return Ok((on, vec![Src::Reg(ra), Src::Reg(rb)]));
                    }
                    Op::CmpLtF32(a, b) => {
                        let ra = self.as_v1(a)?;
                        let rb = self.as_v1(b)?;
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
            Op::LshrI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SLshrB32, *a, *b)?,
            Op::AndI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SAndB32, *a, *b)?,
            Op::XorI32(a, b) if self.uniform.contains(&dst) => self.scalar_bin(dst, MOp::SXorB32, *a, *b)?,
            Op::AddI32(a, b) if opt_on("addrfold") && self.lazy_add(dst, *a, *b)? => {
                if std::env::var("AMDC_ADDRFOLD_EAGER").map(|v| !v.is_empty()).unwrap_or(false) {
                    let _ = self.as_v1(Opnd::Val(dst))?;
                }
            }
            Op::AddI32(a, b) => self.bin_v1(dst, MOp::VAddU32, *a, *b)?,
            Op::AddF32(a, b) => self.bin_v1(dst, MOp::VAddF32, *a, *b)?,
            Op::SubF32(a, b) => self.bin_v1(dst, MOp::VSubF32, *a, *b)?,
            Op::MulF32(a, b) => self.bin_v1(dst, MOp::VMulF32, *a, *b)?,
            Op::MaxF32(a, b) => self.bin_v1(dst, MOp::VMaxF32, *a, *b)?,
            Op::AndI32(a, b) => self.bin_v1(dst, MOp::VAndB32, *a, *b)?,
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
                if let Some((sr, off)) = self.saddr_parts(*base)? {
                    self.push(MInst::new(MOp::GlobalLoadF32S).def(d).srcreg(sr).srcreg(off).imm(*byte_off as i64));
                } else {
                    let rb = self.as_v2(*base)?;
                    self.push(MInst::new(MOp::GlobalLoadF32).def(d).srcreg(rb).imm(*byte_off as i64));
                }
                self.pending_vmem.push(self.mir.canon(d));
                self.val.insert(dst, d);
            }
            Op::LoadGlobalB128 { base, byte_off } => {
                let d = self.new_vreg(RClass::V4);
                if let Some((sr, off)) = self.saddr_parts(*base)? {
                    self.push(MInst::new(MOp::GlobalLoadB128S).def(d).srcreg(sr).srcreg(off).imm(*byte_off as i64));
                } else {
                    let rb = self.as_v2(*base)?;
                    self.push(MInst::new(MOp::GlobalLoadB128).def(d).srcreg(rb).imm(*byte_off as i64));
                }
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
            Op::ReadFirstLane(a) => {
                let r = self.as_v1(*a)?;
                self.touch(&[Src::Reg(r)]);
                let d = self.new_vreg(RClass::S1);
                self.push(MInst::new(MOp::VReadFirstLane).def(d).srcreg(r));
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
            Op::MfmaFp8_16x16x32 { a, b, c } => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                let rc = self.as_v4(*c)?;
                let d = self.new_vreg(RClass::V4);
                self.copy_reg(d, rc)?;
                self.push(
                    MInst::new(MOp::VMfma16x16x32Fp8)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(d),
                );
                self.val.insert(dst, d);
            }
            Op::MfmaFp8_32x32x16 { a, b, c } => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                let rc = self.as_v16(*c)?;
                let d = self.new_vreg(RClass::V16);
                self.copy_reg(d, rc)?;
                self.push(
                    MInst::new(MOp::VMfma32x32x16Fp8)
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
            Op::CvtPkFp8F32(a, b) => {
                let ra = self.as_v1(*a)?;
                let rb = self.as_v1(*b)?;
                let d = self.new_vreg(RClass::V1);
                // The instruction only writes the low 16 bits, so start from 0.
                let z = self.const_vreg(0);
                self.push(MInst::new(MOp::VMovB32).def(d).srcreg(z));
                self.push(
                    MInst::new(MOp::VCvtPkFp8F32)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb),
                );
                self.val.insert(dst, d);
            }
            Op::CvtScalef32PkFp8Bf16 { old, src, scale, high } => {
                let ro = self.as_v1(*old)?;
                let rs = self.as_v1(*src)?;
                let rsc = self.as_v1(*scale)?;
                // The instruction writes the 16 bits selected by OPSEL[3]. The
                // low-half form is verified; the high-half encoding is still
                // being pinned down, so build it explicitly: pack into the low
                // half, shift it up, then OR in the preserved low half of old.
                let packed = self.new_vreg(RClass::V1);
                self.push(
                    MInst::new(MOp::VCvtPkFp8Lo)
                        .def(packed)
                        .srcreg(rs)
                        .srcreg(rsc),
                );
                let d = self.new_vreg(RClass::V1);
                if *high {
                    let c16 = self.const_vreg(16);
                    let shifted = self.new_vreg(RClass::V1);
                    self.push(
                        MInst::new(MOp::VLshlB32)
                            .def(shifted)
                            .srcreg(c16)
                            .srcreg(packed),
                    );
                    let low_kept = self.new_vreg(RClass::V1);
                    let mask = self.const_vreg(0xFFFF);
                    self.push(
                        MInst::new(MOp::VAndB32)
                            .def(low_kept)
                            .srcreg(ro)
                            .srcreg(mask),
                    );
                    self.push(
                        MInst::new(MOp::VOrB32)
                            .def(d)
                            .srcreg(shifted)
                            .srcreg(low_kept),
                    );
                } else {
                    // The low-half write preserves the upper 16 bits of the
                    // destination, so seed it with the previous value.
                    self.push(MInst::new(MOp::VMovB32).def(d).srcreg(ro));
                    self.push(
                        MInst::new(MOp::VCvtPkFp8Lo)
                            .def(d)
                            .srcreg(rs)
                            .srcreg(rsc),
                    );
                }
                self.val.insert(dst, d);
            }
            Op::Frag16Build(lanes) => {
                let v = self.new_vreg(RClass::V16);
                for (i, o) in lanes.iter().enumerate() {
                    let a = self.alias_vreg(v, i as u8, RClass::V1);
                    let retargeted = match o {
                        Opnd::Val(ov) => self.retarget_producer(*ov, a),
                        _ => false,
                    };
                    if !retargeted {
                        let r = self.as_v1(*o)?;
                        self.push(MInst::new(MOp::VMovB32).def(a).srcreg(r));
                    }
                }
                self.val.insert(dst, v);
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
                self.pending_lds.push(d);
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
                self.val.insert(dst, d);
            }            Op::Frag8(a, b, c, d, e, f, g, h) => {
                let v = self.new_vreg(RClass::V8);
                for (i, o) in [*a, *b, *c, *d, *e, *f, *g, *h].into_iter().enumerate() {
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
            Op::MfmaScaleFp8_32x32x64 { a, b, c, scale_a, scale_b } => {
                let ra = self.as_v8(*a)?;
                let rb = self.as_v8(*b)?;
                let c_is_zero = matches!(
                    c,
                    Opnd::Val(cv) if matches!(self.defs.get(cv), Some(Op::FragZero { ty: DType::Frag16 }))
                );
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
                                    Some(Op::MfmaScaleFp8_32x32x64 { .. }) | Some(Op::Frag16Build(_))
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
            Op::Frag16Pair { frag, index } => {
                let f = match frag {
                    Opnd::Val(v) => *self.val.get(v).ok_or_else(|| format!("value v{v} not lowered"))?,
                    other => self.as_v16(*other)?,
                };
                self.touch(&[Src::Reg(f)]);
                let d = self.sub_alias(f, *index, RClass::V2);
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
        if opt_on("shl") && op == MOp::VMulLoU32 {
            for (c, other) in [(b, a), (a, b)] {
                if let Some(k) = self.const_bits(c) {
                    if k > 0 && (k & (k - 1)) == 0 && k < (1 << 31) && self.const_bits(other).is_none() {
                        let r = self.as_v1(other)?;
                        self.touch(&[Src::Reg(r)]);
                        let d = self.new_vreg(RClass::V1);
                        self.push(MInst::new(MOp::VLshlB32).def(d).src(Src::Imm(k.trailing_zeros() as i64)).srcreg(r));
                        self.val.insert(dst, d);
                        return Ok(());
                    }
                }
            }
        }
        if opt_on("sgpr") {
            let vop3 = op == MOp::VMulLoU32;
            let commutative = op != MOp::VSubF32;
            let (sa, sb) = (self.scalar_reg(a), self.scalar_reg(b));
            let plan = match (sa, sb) {
                (Some(s), None) if self.const_bits(b).is_none() => Some((s, b, true)),
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
        let ra = self.as_v1(a)?;
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
fn coalesce(m: &mut MFunction) {
    let classes: Vec<RClass> = m.vregs.clone();
    let aliased = m.alias.clone();
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
            if op == MOp::VMovB32 && srcs.len() == 1 {
                if let (Some(d), Src::Reg(s)) = (dst, srcs[0]) {
                    let prev = &b.insts[i - 1];
                    let prev_defs_s = prev.dst == Some(s);
                    let single_use = uses.get(&s).copied().unwrap_or(0) == 1;
                    let same_class = classes[d as usize] == classes[s as usize];
                    let prev_is_move_to_s = prev.op == MOp::VMovB32;
                    // A lane of a fragment is not a standalone value: the whole tuple may still be read.
                    let s_is_alias = aliased.contains_key(&s);
                    if prev_defs_s && single_use && same_class && !prev_is_move_to_s && !s_is_alias {
                        if let Some(pd) = b.insts[i - 1].dst.as_mut() {
                            *pd = d;
                        }
                        b.insts.remove(i);
                        continue;
                    }
                }
            }
            i += 1;
        }
    }
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
                if matches!(c.op, MOp::VCmpLtI32 | MOp::VCmpLtF32 | MOp::VCmpGeI32) && c.dst.is_none() {
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
                        if matches!(b.insts[j].op, MOp::VCmpLtI32 | MOp::VCmpLtF32 | MOp::VCmpGeI32) && b.insts[j].dst.is_none() {
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
/// Dead-code elimination that understands `vcc`: a pure instruction whose result is never read is removed, and a
/// compare / carry producer is removed when no later instruction of the block reads the `vcc` it writes
/// (the terminator may read it). The lowering emits address / clamp chains whose consumers were dropped by other
/// options (e.g. the global scale loads replaced by LDS scale reads); this removes them. Same values, same
/// arithmetic for everything that remains.
fn dce_full(m: &mut MFunction, loads_only: bool) {
    fn load_op(op: MOp) -> bool {
        matches!(
            op,
            MOp::SLoadDword | MOp::SLoadDwordx2 | MOp::LdsReadB32 | MOp::LdsReadB128 | MOp::LdsReadB64TrB8 | MOp::GlobalLoadU16
                | MOp::GlobalLoadF32 | MOp::GlobalLoadB128 | MOp::GlobalLoadB128S | MOp::GlobalLoadF32S
        )
    }
    let removable = |op: MOp| -> bool {
        if load_op(op) {
            return true;
        }
        if loads_only {
            return false;
        }
        matches!(
            op,
            MOp::VMovB32 | MOp::VMovFromS | MOp::VMovFromSPair | MOp::VPtrAdd | MOp::VAddU32 | MOp::VAddCoU32 | MOp::VAddcCoU32
                | MOp::VAddF32 | MOp::VSubF32 | MOp::VMulF32 | MOp::VMaxF32 | MOp::VMax3F32 | MOp::VFmaF32 | MOp::VExpF32
                | MOp::VRcpF32 | MOp::VAndB32 | MOp::VOrB32 | MOp::VXorB32 | MOp::VLshrB32 | MOp::VLshlB32 | MOp::VMulLoU32
                | MOp::VCndmask | MOp::VCvtF32F16 | MOp::VCvtF16F32 | MOp::VReadPhys | MOp::VCmpLtI32 | MOp::VCmpGeI32
                | MOp::VCmpNeI32 | MOp::VCmpLtF32
        )
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
        for bi in 0..m.blocks.len() {
            let mut vcc_live = matches!(m.blocks[bi].term, Term::CondGoto { on: Pred::Vccnz | Pred::Vccz, .. });
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
        let is_u = |o: &Opnd, u: &HashSet<ValueId>, uvars: &HashSet<amdgpu_dsl::VarId>| match o {
            Opnd::I32(_) => true,
            Opnd::Val(v) => u.contains(v),
            Opnd::Var(v) => uvars.contains(v),
            _ => false,
        };
        let mut changed = false;
        for st in &f.stmts {
            match st {
                Stmt::Let { dst, op } => {
                    let uni = match op {
                        Op::ConstI32(_) | Op::KernargI32 { .. } | Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ => true,
                        Op::AddI32(a, b) | Op::MulI32(a, b) | Op::ShlI32(a, b) | Op::LshrI32(a, b) | Op::AndI32(a, b) | Op::XorI32(a, b) => {
                            is_u(a, &u, &uvars) && is_u(b, &u, &uvars)
                        }
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
                | Stmt::VarScaleLane { var, .. } | Stmt::VarLoadB128 { var, .. } | Stmt::VarMfmaScale { var, .. } | Stmt::VarMfmaScaleZ { var, .. } => {
                    if uvars.remove(var) {
                        changed = true;
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