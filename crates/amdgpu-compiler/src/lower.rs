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
        return Err(format!(
            "IR verification failed:\n  {}\nIR:\n{}",
            errs.join("\n  "),
            f.dump()
        ));
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
    pending_lds: Vec<VReg>,
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
            lds_bytes: f.lds_bytes,
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
            pending_lds: Vec::new(),
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
        if !self.pending_smem.is_empty() || !self.pending_lds.is_empty() {
            self.push(MInst::new(MOp::SWaitLgkm).imm(0));
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
        let mut need_lgkm = false;
        for s in srcs {
            if let Src::Reg(r) = s {
                if self.pending_vmem.contains(r) {
                    need_vm = true;
                }
                if self.pending_smem.contains(r) || self.pending_lds.contains(r) {
                    need_lgkm = true;
                }
            }
        }
        if need_vm {
            self.push(MInst::new(MOp::SWaitVm));
            self.pending_vmem.clear();
        }
        if need_lgkm {
            self.push(MInst::new(MOp::SWaitLgkm).imm(0));
            self.pending_smem.clear();
            self.pending_lds.clear();
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
            let t = self.new_vreg(RClass::V1);
            self.push(MInst::new(MOp::VReadSub).def(t).srcreg(src).imm(i as i64));
            self.push(MInst::new(MOp::VWriteSub).def(dst).srcreg(t).imm(i as i64));
        }
        Ok(())
    }

    /// Sixteen-dword fragment operand.
    fn as_v16(&mut self, o: Opnd) -> Result<VReg, String> {
        match o {
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
                Stmt::BeginLoop { vars, init } => {
                    for (i, var) in vars.iter().enumerate() {
                        let ty = f.var_ty(*var).ok_or("loop var has no type")?;
                        let class = match ty {
                            DType::Ptr => return Err("pointer loop variables are not supported".to_string()),
                            DType::Frag2 => RClass::V2,
                            DType::Frag4 => RClass::V4,
                            DType::Frag8 => RClass::V8,
                            DType::Frag16 => RClass::V16,
                            _ => RClass::V1,
                        };
                        let r = self.new_vreg(class);
                        if self.vars.len() <= var.0 as usize {
                            self.vars.resize(var.0 as usize + 1, u32::MAX);
                        }
                        self.vars[var.0 as usize] = r;
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
                Stmt::StoreLdsB128 { addr, value } => {
                    let a = self.as_v1(*addr)?;
                    let v = self.as_v4(*value)?;
                    self.push(MInst::new(MOp::LdsWriteB128).srcreg(a).srcreg(v));
                }
                Stmt::StoreLdsB32 { addr, value } => {
                    let a = self.as_v1(*addr)?;
                    let v = self.as_v1(*value)?;
                    self.touch(&[Src::Reg(v)]);
                    self.push(MInst::new(MOp::LdsWriteB32).srcreg(a).srcreg(v));
                }
                Stmt::WaitLds { count } => {
                    self.push(MInst::new(MOp::SWaitLgkm).imm(*count as i64));
                }
                Stmt::Barrier => {
                    self.push(MInst::new(MOp::SBarrier));
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
                let log2e = self.const_vreg(0x3FB8AA3B_u32 as i32 as i64);
                let scaled = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VMulF32).def(scaled).srcreg(ra).srcreg(log2e));
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VExpF32).def(d).srcreg(scaled));
                self.val.insert(dst, d);
            }
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
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VReadSub).def(d).srcreg(f).imm(*index as i64));
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
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VReadSub).def(d).srcreg(f).imm(*index as i64));
                self.val.insert(dst, d);
            }
            Op::MfmaF16 { a, b, c } => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                let rc = self.as_v4(*c)?;
                let d = self.new_vreg(RClass::V4);
                self.push(
                    MInst::new(MOp::VMfma16x16x16F16)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(rc),
                );
                self.val.insert(dst, d);
            }
            Op::LoadLdsB128 { addr } => {
                let a = self.as_v1(*addr)?;
                let d = self.new_vreg(RClass::V4);
                self.push(MInst::new(MOp::LdsReadB128).def(d).srcreg(a));
                self.pending_lds.push(d);
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
                let a = self.as_v1(*addr)?;
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::LdsReadB32).def(d).srcreg(a));
                self.pending_lds.push(d);
                self.val.insert(dst, d);
            }
            Op::MfmaF16Wide { a, b, c } => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                let rc = self.as_v16(*c)?;
                let d = self.new_vreg(RClass::V16);
                self.push(
                    MInst::new(MOp::VMfma32x32x8F16)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(rc),
                );
                self.val.insert(dst, d);
            }
            Op::MfmaFp8_16x16x32 { a, b, c } => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                let rc = self.as_v4(*c)?;
                let d = self.new_vreg(RClass::V4);
                self.push(
                    MInst::new(MOp::VMfma16x16x32Fp8)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(rc),
                );
                self.val.insert(dst, d);
            }
            Op::MfmaFp8_32x32x16 { a, b, c } => {
                let ra = self.as_v2(*a)?;
                let rb = self.as_v2(*b)?;
                let rc = self.as_v16(*c)?;
                let d = self.new_vreg(RClass::V16);
                self.push(
                    MInst::new(MOp::VMfma32x32x16Fp8)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(rc),
                );
                self.val.insert(dst, d);
            }
            Op::CatFrag8(lo, hi) => {
                let lo_r = self.as_v4(*lo)?;
                let hi_r = self.as_v4(*hi)?;
                let d = self.new_vreg(RClass::V8);
                for i in 0..4u8 {
                    let sub = self.new_vreg(RClass::V1);
                    self.push(MInst::new(MOp::VReadSub).def(sub).srcreg(lo_r).imm(i as i64));
                    self.push(MInst::new(MOp::VWriteSub).def(d).srcreg(sub).imm(i as i64));
                }
                for i in 0..4u8 {
                    let sub = self.new_vreg(RClass::V1);
                    self.push(MInst::new(MOp::VReadSub).def(sub).srcreg(hi_r).imm(i as i64));
                    self.push(MInst::new(MOp::VWriteSub).def(d).srcreg(sub).imm((i + 4) as i64));
                }
                self.val.insert(dst, d);
            }
            Op::Frag8(a, b, c, d, e, f, g, h) => {
                let v = self.new_vreg(RClass::V8);
                for (i, o) in [*a, *b, *c, *d, *e, *f, *g, *h].into_iter().enumerate() {
                    let r = self.as_v1(o)?;
                    self.touch(&[Src::Reg(r)]);
                    self.push(MInst::new(MOp::VWriteSub).def(v).srcreg(r).imm(i as i64));
                }
                self.val.insert(dst, v);
            }
            Op::MfmaScaleFp8_32x32x64 { a, b, c, scale_a, scale_b } => {
                let ra = self.as_v8(*a)?;
                let rb = self.as_v8(*b)?;
                let rc = self.as_v16(*c)?;
                let sa = self.as_v1(*scale_a)?;
                let sb = self.as_v1(*scale_b)?;
                let d = self.new_vreg(RClass::V16);
                self.push(
                    MInst::new(MOp::VMfmaScale32x32x64Fp8)
                        .def(d)
                        .srcreg(ra)
                        .srcreg(rb)
                        .srcreg(rc)
                        .srcreg(sa)
                        .srcreg(sb),
                );
                self.val.insert(dst, d);
            }
            Op::Frag16Lane { frag, index } => {
                let f = self.as_v16(*frag)?;
                let d = self.new_vreg(RClass::V1);
                self.push(MInst::new(MOp::VReadSub).def(d).srcreg(f).imm(*index as i64));
                self.val.insert(dst, d);
            }
            Op::FragZero { ty } => {
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
            Op::BitcastF32(a) => {
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
