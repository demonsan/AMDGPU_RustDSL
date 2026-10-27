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

use std::collections::HashMap;

use amdgpu_dsl::{DType, Function, Op, Opnd, Stmt, ValueId};
use crate::mir::*;

pub fn lower(f: &Function) -> Result<MFunction, String> {
    let errs = amdgpu_dsl::verify(f);
    if !errs.is_empty() {
        return Err(format!("IR verification failed:\n  {}", errs.join("\n  ")));
    }
    let mut lw = Lowerer::new(f);
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
    pending_vmem: Vec<VReg>,
    pending_smem: Vec<VReg>,
    break_stack: Vec<(u32, u32)>,
    if_stack: Vec<(u32, u32)>,
    cur: usize,
    next_label: u32,
    defs: HashMap<ValueId, Op>,
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
            blocks: Vec::new(),
            vregs: Vec::new(),
            loops: Vec::new(),
        };
        mir.blocks.push(Block::new(0));

        let mut defs = HashMap::new();
        for st in &f.stmts {
            if let Stmt::Let { dst, op } = st {
                defs.insert(*dst, op.clone());
            }
        }

        Lowerer {
            mir,
            val: HashMap::new(),
            converted: HashMap::new(),
            vars: Vec::new(),
            consts: HashMap::new(),
            pending_vmem: Vec::new(),
            pending_smem: Vec::new(),
            break_stack: Vec::new(),
            if_stack: Vec::new(),
            cur: 0,
            next_label: 1,
            defs,
        }
    }

    fn finish(mut self) -> MFunction {
        if !self.pending_vmem.is_empty() {
            self.push(MInst::new(MOp::SWaitVm));
        }
        if !self.pending_smem.is_empty() {
            self.push(MInst::new(MOp::SWaitLgkm));
        }
        self.set_term(Term::Endpgm);
        coalesce(&mut self.mir);
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
        self.mir.blocks.push(Block::new(label));
        self.cur = self.mir.blocks.len() - 1;
    }

    /// Wait for any pending load feeding `srcs`, then clear the pending set.
    fn touch(&mut self, srcs: &[Src]) {
        let mut need_vm = false;
        let mut need_sm = false;
        for s in srcs {
            if let Src::Reg(r) = s {
                if self.pending_vmem.contains(r) {
                    need_vm = true;
                }
                if self.pending_smem.contains(r) {
                    need_sm = true;
                }
            }
        }
        if need_vm {
            self.push(MInst::new(MOp::SWaitVm));
            self.pending_vmem.clear();
        }
        if need_sm {
            self.push(MInst::new(MOp::SWaitLgkm));
            self.pending_smem.clear();
        }
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
                if let Some(r) = self.converted.get(&v) {
                    return Ok(*r);
                }
                let r = *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?;
                match self.mir.vreg_class(r) {
                    RClass::V1 => Ok(r),
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
            Opnd::Var(v) => self
                .vars
                .get(v.0 as usize)
                .copied()
                .ok_or_else(|| format!("var{v} not lowered")),
            Opnd::I32(i) => Ok(self.const_vreg(i as i64)),
            Opnd::F32(f) => Ok(self.const_vreg((f.to_bits() as i32) as i64)),
        }
    }

    fn as_v2(&mut self, o: Opnd) -> Result<VReg, String> {
        match o {
            Opnd::Val(v) => {
                if let Some(r) = self.converted.get(&v) {
                    return Ok(*r);
                }
                let r = *self.val.get(&v).ok_or_else(|| format!("value v{v} not lowered"))?;
                match self.mir.vreg_class(r) {
                    RClass::V2 => Ok(r),
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
                    Ok(r)
                } else {
                    Err(format!("var{v} is not a pointer"))
                }
            }
            _ => Err("pointer operand must be a value or variable".to_string()),
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
                    let src = self.as_v1(*value)?;
                    let dst = self.vars[var.0 as usize];
                    if self.mir.vreg_class(dst) != RClass::V1 {
                        return Err("pointer loop variables are not supported yet".to_string());
                    }
                    self.touch(&[Src::Reg(src)]);
                    if src != dst {
                        self.push(MInst::new(MOp::VMovB32).def(dst).srcreg(src));
                    }
                }
                Stmt::BeginLoop { vars, init } => {
                    for (i, var) in vars.iter().enumerate() {
                        let ty = f.var_ty(*var).ok_or("loop var has no type")?;
                        let class = match ty {
                            DType::Ptr => return Err("pointer loop variables are not supported".to_string()),
                            _ => RClass::V1,
                        };
                        let r = self.new_vreg(class);
                        if self.vars.len() <= var.0 as usize {
                            self.vars.resize(var.0 as usize + 1, u32::MAX);
                        }
                        self.vars[var.0 as usize] = r;
                        let src = self.as_v1(init[i])?;
                        self.touch(&[Src::Reg(src)]);
                        self.push(MInst::new(MOp::VMovB32).def(r).srcreg(src));
                    }
                    let header = self.fresh_label();
                    let exit = self.fresh_label();
                    self.set_term(Term::Goto(header));
                    self.start_block(header);
                    self.break_stack.push((exit, header));
                }
                Stmt::BreakIf { cond } => {
                    let (exit, _) = *self.break_stack.last().ok_or("break outside loop")?;
                    let (on, srcs) = self.lower_condition(*cond, false)?;
                    self.touch(&srcs);
                    let fall = self.fresh_label();
                    self.set_term(Term::CondGoto { on, target: exit, fall });
                    self.start_block(fall);
                }
                Stmt::EndLoop => {
                    let (exit, header) = self.break_stack.pop().ok_or("endloop without loop")?;
                    let latch = self.mir.blocks[self.cur].label;
                    self.mir.loops.push((header, latch));
                    self.set_term(Term::Goto(header));
                    self.start_block(exit);
                }
                Stmt::BeginIf { cond } => {
                    let (on, srcs) = self.lower_condition(*cond, false)?;
                    self.touch(&srcs);
                    let then_l = self.fresh_label();
                    let else_l = self.fresh_label();
                    let end_l = self.fresh_label();
                    self.set_term(Term::CondGoto { on, target: then_l, fall: else_l });
                    self.start_block(then_l);
                    self.if_stack.push((else_l, end_l));
                }
                Stmt::Else => {
                    let (else_l, end_l) = *self.if_stack.last().ok_or("else without if")?;
                    self.set_term(Term::Goto(end_l));
                    self.start_block(else_l);
                }
                Stmt::EndIf => {
                    let (_, end_l) = self.if_stack.pop().ok_or("endif without if")?;
                    self.set_term(Term::Goto(end_l));
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
                self.pending_smem.push(d);
                self.val.insert(dst, d);
            }
            Op::KernargPtr { offset } => {
                let d = self.new_vreg(RClass::S2);
                self.push(MInst::new(MOp::SLoadDwordx2).def(d).imm(*offset as i64));
                self.pending_smem.push(d);
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
            Op::AddI32(a, b) => self.bin_v1(dst, MOp::VAddU32, *a, *b)?,
            Op::AddF32(a, b) => self.bin_v1(dst, MOp::VAddF32, *a, *b)?,
            Op::SubF32(a, b) => self.bin_v1(dst, MOp::VSubF32, *a, *b)?,
            Op::MulF32(a, b) => self.bin_v1(dst, MOp::VMulF32, *a, *b)?,
            Op::MaxF32(a, b) => self.bin_v1(dst, MOp::VMaxF32, *a, *b)?,
            Op::AndI32(a, b) => self.bin_v1(dst, MOp::VAndB32, *a, *b)?,
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
            Op::ExpF32(a) => self.un_v1(dst, MOp::VExpF32, *a)?,
            Op::RcpF32(a) => self.un_v1(dst, MOp::VRcpF32, *a)?,
            Op::CvtF16ToF32(a) => self.un_v1(dst, MOp::VCvtF32F16, *a)?,
            Op::CvtF32ToF16(a) => self.un_v1(dst, MOp::VCvtF16F32, *a)?,
            Op::CmpLtI32(a, b) => self.cmp_i1(dst, MOp::VCmpLtI32, *a, *b)?,
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
                let rb = self.as_v1(*b)?;
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VCndmask).def(d).srcreg(rb).srcreg(ra));
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
                self.pending_vmem.push(d);
                self.val.insert(dst, d);
            }
            Op::LoadF32 { base, byte_off } => {
                let rb = self.as_v2(*base)?;
                let d = self.new_vreg(RClass::V1);
                self.push(
                    MInst::new(MOp::GlobalLoadF32).def(d).srcreg(rb).imm(*byte_off as i64),
                );
                self.pending_vmem.push(d);
                self.val.insert(dst, d);
            }
        }
        Ok(())
    }

    /// Shift amounts can be an in-line literal or a register.
    fn shift_operand(&mut self, o: Opnd) -> Result<Src, String> {
        match o {
            Opnd::I32(i) => Ok(Src::Imm(i as i64)),
            other => Ok(Src::Reg(self.as_v1(other)?)),
        }
    }

    fn bin_v1(&mut self, dst: ValueId, op: MOp, a: Opnd, b: Opnd) -> Result<(), String> {
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
                    if prev_defs_s && single_use && same_class && !prev_is_move_to_s {
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
