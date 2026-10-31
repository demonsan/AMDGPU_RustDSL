//! The sequence builder: records instructions as plain `AInst` data (lowered to MIR by the compiler, so register allocation, scheduling and the
//! hazard passes stay ours), tracks a wait-state clock and the DS/VMEM scoreboards, and hands out the type-state tokens that make hazards
//! impossible to forget.

use crate::block::*;
use crate::regs::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

/// Wait states (instructions or `s_nop` slots) that must separate a VALU write of a VGPR from a hazard-sensitive reader
/// (matrix operand, `v_readfirstlane`): "2 required waits", CDNA4 ISA 7.6, LLVM hazard recognizer.
pub const VALU_TO_MATRIX_READ: u32 = 2;
/// Wait states between a (scaled) MFMA / SMFMAC and the first reader of its result (hardware-confirmed: 19 is still stale).
pub const MATRIX_RESULT: u32 = 20;
/// Wait states between a transcendental op (`v_exp/rcp/log_f32`) and any reader of its result (hazards.rs TRANS_LATENCY = 5 slots).
pub const TRANS_RESULT: u32 = 4;
/// Wait states between a packed-fp32 op and a transcendental op that reads its result (hazards.rs `pkw`).
pub const PK_TO_TRANS: u32 = 1;
/// Wait states between a compare that writes `vcc` and the `v_cndmask` that reads it.
pub const VCC_TO_CNDMASK: u32 = 2;

#[derive(Clone, Copy, Debug)]
struct Write {
    /// Instruction slot of the write.
    at: u32,
    /// Wait states a hazard-sensitive reader needs after it.
    need: u32,
}

/// Marker: `vcc` holds no live mask.
#[derive(Debug)]
pub struct Free;
/// Marker: `vcc` holds a mask written by a compare and not yet consumed.
#[derive(Debug)]
pub struct Live;

/// The single `vcc` register as a type-state token: a compare turns `Vcc<Free>` into `Vcc<Live>`, a `v_cndmask` consumes it.
#[derive(Debug)]
pub struct Vcc<S> {
    pub(crate) _s: PhantomData<S>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Queue {
    /// LDS / scalar memory (`lgkmcnt`).
    Lgkm,
    /// Global / buffer loads (`vmcnt`).
    Vm,
}

/// A load in flight. Owns the destination register, so the value cannot be read (or the register reused) until `wait` returns it. The wait
/// emitted is the *partial* count derived from the scoreboard: `s_waitcnt lgkmcnt(k)` / `vmcnt(k)` with `k` = loads issued after this one
/// that are still outstanding, and nothing at all if an earlier wait already covered it.
#[must_use = "a load in flight must be waited for before its value is used"]
#[derive(Debug)]
pub struct Pending<T> {
    value: Option<T>,
    poison: Poison,
    id: u32,
    queue: Queue,
}

/// Problems recorded by `Drop` of tokens abandoned without being retired / waited for; reported by `finish`.
pub(crate) type Poison = Rc<RefCell<Vec<String>>>;

pub struct Seq {
    pub(crate) poison: Poison,
    pub(crate) insts: Vec<AInst>,
    pub(crate) regs: Vec<AClass>,
    pub(crate) alias: Vec<(u32, u32, u8)>,
    alias_base: HashMap<VReg, (VReg, u8)>,
    pub(crate) clock: u32,
    writes: HashMap<VReg, Write>,
    vcc_taken: bool,
    pk_writes: HashMap<VReg, u32>,
    vcc_written_at: u32,
    zero: Option<VReg>,
    inputs: Vec<AsmPort>,
    outputs: Vec<AsmPort>,
    exit_in_flight: Vec<(u32, u32)>,
    next_token: u32,
    loop_body: bool,
    pinned: bool,
    lgkm: Vec<u32>,
    vm: Vec<u32>,
}

impl Default for Seq {
    fn default() -> Self {
        Self::new()
    }
}

impl Seq {
    pub fn new() -> Seq {
        Seq {
            poison: Rc::new(RefCell::new(Vec::new())),
            insts: Vec::new(),
            regs: Vec::new(),
            alias: Vec::new(),
            alias_base: HashMap::new(),
            clock: 0,
            writes: HashMap::new(),
            vcc_taken: false,
            pk_writes: HashMap::new(),
            vcc_written_at: 0,
            zero: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            exit_in_flight: Vec::new(),
            next_token: 0,
            loop_body: false,
            pinned: false,
            lgkm: Vec::new(),
            vm: Vec::new(),
        }
    }

    // ---- registers -----------------------------------------------------------------------------------------------------------
    fn new_reg(&mut self, c: AClass) -> VReg {
        self.regs.push(c);
        (self.regs.len() - 1) as VReg
    }

    /// A fresh vector register tuple (contents undefined until written).
    pub fn vgpr<const N: usize>(&mut self) -> Vgpr<N>
    where
        Wd<N>: VectorWidth,
    {
        Vgpr { id: self.new_reg(<Wd<N> as VectorWidth>::CLASS) }
    }

    /// A fresh scalar register (`N` = 1) or pair (`N` = 2).
    pub fn sgpr<const N: usize>(&mut self) -> Sgpr<N>
    where
        Wd<N>: ScalarWidth,
    {
        Sgpr { id: self.new_reg(<Wd<N> as ScalarWidth>::CLASS) }
    }

    /// A register tuple provided by the surrounding kernel (a DSL value): a READ-ONLY handle (`In<N>`), it can never be a destination.
    pub fn input<const N: usize>(&mut self) -> In<N>
    where
        Wd<N>: VectorWidth,
    {
        let r = self.vgpr::<N>();
        self.inputs.push(AsmPort { reg: r.id, class: <Wd<N> as VectorWidth>::CLASS, mutable: false });
        In { id: r.id }
    }

    /// A register tuple the kernel owns and the sequence UPDATES IN PLACE (a loop variable, bound by `Stmt::VarAsm`). Writable, unlike `input`;
    /// hand it back with `output` / `output_exit`.
    pub fn inout<const N: usize>(&mut self) -> Vgpr<N>
    where
        Wd<N>: VectorWidth,
    {
        let r = self.vgpr::<N>();
        self.inputs.push(AsmPort { reg: r.id, class: <Wd<N> as VectorWidth>::CLASS, mutable: true });
        r
    }

    /// Hand a finished register tuple back to the kernel as the sequence's result (consumes it: it cannot be used afterwards).
    pub fn output<const N: usize>(&mut self, r: Vgpr<N>)
    where
        Wd<N>: VectorWidth,
    {
        self.outputs.push(AsmPort { reg: r.id, class: <Wd<N> as VectorWidth>::CLASS, mutable: false });
    }

    /// Hand back a result whose matrix write is still in flight; the consumer after the sequence is padded by the compiler's hazard pass.
    pub fn output_exit<const N: usize>(&mut self, e: Exit<N>)
    where
        Wd<N>: VectorWidth,
    {
        let owed = MATRIX_RESULT.saturating_sub(self.clock - (e.issued + 1));
        self.exit_in_flight.push((e.id, owed));
        self.outputs.push(AsmPort { reg: e.id, class: <Wd<N> as VectorWidth>::CLASS, mutable: false });
    }

    /// A one-dword register holding the constant `k` (`v_mov_b32 v, k`).
    pub fn splat(&mut self, k: i32) -> Vgpr<1> {
        let r = self.new_reg(AClass::V1);
        self.push(AInst { op: "VMovB32", dst: Some(r), srcs: vec![], imm: k as i64 });
        self.note_valu_write(r);
        Vgpr { id: r }
    }

    /// A tuple whose `N` dwords all hold the constant `k` (one `v_mov` plus the per-dword copies).
    pub fn filled<const N: usize>(&mut self, k: i32) -> Vgpr<N>
    where
        Wd<N>: VectorWidth,
    {
        let c = self.splat(k);
        let d = self.vgpr::<N>();
        for i in 0..N {
            self.push(AInst { op: "VWriteSub", dst: Some(d.id), srcs: vec![ASrc::Reg(c.id)], imm: i as i64 });
        }
        self.note_valu_write(d.id);
        d
    }

    /// A tuple whose dwords are all zero.
    pub fn zeroed<const N: usize>(&mut self) -> Vgpr<N>
    where
        Wd<N>: VectorWidth,
    {
        let d = self.vgpr::<N>();
        if N == 1 {
            self.push(AInst { op: "VMovB32", dst: Some(d.id), srcs: vec![], imm: 0 });
        } else {
            let z = match self.zero {
                Some(z) => z,
                None => {
                    let z = self.new_reg(AClass::V1);
                    self.push(AInst { op: "VMovB32", dst: Some(z), srcs: vec![], imm: 0 });
                    self.zero = Some(z);
                    z
                }
            };
            for i in 0..N {
                self.push(AInst { op: "VWriteSub", dst: Some(d.id), srcs: vec![ASrc::Reg(z)], imm: i as i64 });
            }
        }
        self.note_valu_write(d.id);
        d
    }

    /// Dword `I` of a tuple as a one-dword read-only view (bounds checked at compile time).
    pub fn lane<'a, const N: usize, const I: usize>(&mut self, r: &'a impl Readable<N>) -> Sub<'a, 1> {
        let () = LaneCheck::<I, N>::OK;
        Sub { id: self.alias_of(r.reg_id(), I as u8, AClass::V1), _p: PhantomData }
    }

    /// Dword `I` of a tuple as a one-dword writable view.
    pub fn lane_mut<'a, const N: usize, const I: usize>(&mut self, r: &'a mut Vgpr<N>) -> SubMut<'a, 1> {
        let () = LaneCheck::<I, N>::OK;
        SubMut { id: self.alias_of(r.id, I as u8, AClass::V1), _p: PhantomData }
    }

    /// Low (`HI` = false) or high half of an 8-dword tuple as a 4-dword view; the parent stays borrowed.
    pub fn half8<'a, const HI: bool>(&mut self, r: &'a impl Readable<8>) -> Sub<'a, 4> {
        Sub { id: self.alias_of(r.reg_id(), if HI { 4 } else { 0 }, AClass::V4), _p: PhantomData }
    }

    /// Dwords `2*I .. 2*I+2` of an 8-dword tuple as a writable pair (I in 0..4); the parent stays mutably borrowed while the view lives.
    pub fn pair_mut<'a, const I: usize>(&mut self, r: &'a mut Vgpr<8>) -> SubMut<'a, 2> {
        let () = ViewCheck::<I, 4>::OK;
        SubMut { id: self.alias_of(r.id, (2 * I) as u8, AClass::V2), _p: PhantomData }
    }

    /// All four dword pairs of an 8-dword tuple at once: DISJOINT writable views under ONE mutable borrow of the parent, so four DS reads can be in
    /// flight together; the parent is usable again (e.g. `ready`) only after every pending token has been waited and dropped.
    pub fn split_pairs_mut<'a>(&mut self, r: &'a mut Vgpr<8>) -> [SubMut<'a, 2>; 4] {
        let id = r.id;
        [0u8, 2, 4, 6].map(|o| SubMut { id: self.alias_of(id, o, AClass::V2), _p: PhantomData })
    }

    /// Both halves of an 8-dword tuple (disjoint writable 4-dword views).
    pub fn split_halves8_mut<'a>(&mut self, r: &'a mut Vgpr<8>) -> [SubMut<'a, 4>; 2] {
        let id = r.id;
        [0u8, 4].map(|o| SubMut { id: self.alias_of(id, o, AClass::V4), _p: PhantomData })
    }

    /// All four quarters of a 16-dword tuple (disjoint writable 4-dword views).
    pub fn split_quarters16_mut<'a>(&mut self, r: &'a mut Vgpr<16>) -> [SubMut<'a, 4>; 4] {
        let id = r.id;
        [0u8, 4, 8, 12].map(|o| SubMut { id: self.alias_of(id, o, AClass::V4), _p: PhantomData })
    }

    /// Low (`HI` = false) or high half of an 8-dword tuple as a writable 4-dword view.
    pub fn half8_mut<'a, const HI: bool>(&mut self, r: &'a mut Vgpr<8>) -> SubMut<'a, 4> {
        SubMut { id: self.alias_of(r.id, if HI { 4 } else { 0 }, AClass::V4), _p: PhantomData }
    }

    /// Dwords `4*I .. 4*I+4` of a 16-dword tuple as a writable quad (I in 0..4).
    pub fn quarter16_mut<'a, const I: usize>(&mut self, r: &'a mut Vgpr<16>) -> SubMut<'a, 4> {
        let () = ViewCheck::<I, 4>::OK;
        SubMut { id: self.alias_of(r.id, (4 * I) as u8, AClass::V4), _p: PhantomData }
    }

    fn alias_of(&mut self, parent: VReg, off: u8, class: AClass) -> VReg {
        let (base, o) = self.alias_base.get(&parent).copied().unwrap_or((parent, 0));
        let a = self.new_reg(class);
        self.alias_base.insert(a, (base, o + off));
        self.alias.push((a, base, o + off));
        a
    }

    fn canon(&self, r: VReg) -> VReg {
        self.alias_base.get(&r).map(|a| a.0).unwrap_or(r)
    }

    // ---- clock and hazard bookkeeping ---------------------------------------------------------------------------------------
    pub(crate) fn push(&mut self, i: AInst) {
        self.insts.push(i);
        self.clock += 1;
    }

    pub(crate) fn note_valu_write(&mut self, r: VReg) {
        let c = self.canon(r);
        let at = self.clock - 1;
        self.writes.insert(c, Write { at, need: VALU_TO_MATRIX_READ });
    }

    pub(crate) fn note_write(&mut self, r: VReg, need: u32) {
        let c = self.canon(r);
        let at = self.clock - 1;
        self.writes.insert(c, Write { at, need });
    }

    pub(crate) fn note_pk_write(&mut self, r: VReg) {
        let c = self.canon(r);
        let at = self.clock - 1;
        self.pk_writes.insert(c, at);
    }

    pub(crate) fn pad_for_pk(&mut self, r: VReg) {
        let c = self.canon(r);
        if let Some(at) = self.pk_writes.get(&c).copied() {
            self.pad_after(at, PK_TO_TRANS);
        }
    }

    pub(crate) fn clear_write(&mut self, r: VReg) {
        let c = self.canon(r);
        self.writes.remove(&c);
    }

    /// `n` wait states (`s_nop n-1`, repeated as needed).
    pub fn nop(&mut self, mut n: u32) {
        while n > 0 {
            let k = n.min(8);
            self.insts.push(AInst { op: "SNop", dst: None, srcs: vec![], imm: k as i64 - 1 });
            self.clock += k;
            n -= k;
        }
    }

    /// Pad until `need` wait states separate the write at slot `at` from the next instruction.
    pub(crate) fn pad_after(&mut self, at: u32, need: u32) {
        let have = self.clock - (at + 1);
        if have < need {
            self.nop(need - have);
        }
    }

    /// Hazard-sensitive read proof for a whole register: pads exactly the missing wait states since its last write (zero when
    /// enough instructions were already issued in between). The only way to obtain a `Ready`.
    pub fn ready<'a, const N: usize>(&mut self, r: &'a impl Readable<N>) -> Ready<'a, N> {
        let id = r.reg_id();
        self.make_ready(id);
        Ready { id, _p: PhantomData }
    }

    /// Same for a view.
    pub fn ready_sub<'a, const M: usize>(&mut self, r: Sub<'a, M>) -> Ready<'a, M> {
        self.make_ready(r.id);
        Ready { id: r.id, _p: PhantomData }
    }

    pub(crate) fn make_ready_pub(&mut self, id: VReg) {
        self.make_ready(id)
    }

    pub(crate) fn write_of(&self, id: VReg) -> Option<(u32, u32)> {
        self.writes.get(&self.canon(id)).map(|w| (w.at, w.need))
    }

    fn make_ready(&mut self, id: VReg) {
        let c = self.canon(id);
        if let Some(w) = self.writes.get(&c).copied() {
            self.pad_after(w.at, w.need);
        }
    }

    // ---- vcc ------------------------------------------------------------------------------------------------------------------
    /// The `vcc` token. There is exactly one per sequence (panics if taken twice: that is a sequence-construction bug, not a data bug).
    pub fn vcc(&mut self) -> Option<Vcc<Free>> {
        if self.vcc_taken {
            return None;
        }
        self.vcc_taken = true;
        Some(Vcc { _s: PhantomData })
    }

    pub(crate) fn vcc_written(&mut self) {
        self.vcc_written_at = self.clock - 1;
    }

    pub(crate) fn vcc_ready(&mut self) {
        let at = self.vcc_written_at;
        self.pad_after(at, VCC_TO_CNDMASK);
    }

    // ---- DS / VMEM scoreboards ---------------------------------------------------------------------------------------------------
    /// A store occupies a scoreboard slot but returns no value: nothing to wait for individually (`wait_lgkm_all` / a later load wait covers it).
    pub(crate) fn issue_store(&mut self, q: Queue) {
        let id = self.next_token;
        self.next_token += 1;
        match q {
            Queue::Lgkm => self.lgkm.push(id),
            Queue::Vm => self.vm.push(id),
        }
    }

    pub(crate) fn issue_load<T>(&mut self, q: Queue, value: T) -> Pending<T> {
        let id = self.next_token;
        self.next_token += 1;
        match q {
            Queue::Lgkm => self.lgkm.push(id),
            Queue::Vm => self.vm.push(id),
        }
        Pending { value: Some(value), poison: self.poison.clone(), id, queue: q }
    }

    /// Outstanding loads issued after `id` (the `k` of `s_waitcnt ...cnt(k)`), completing everything up to and including `id`.
    fn complete(&mut self, q: Queue, id: u32) -> Option<u32> {
        let list = match q {
            Queue::Lgkm => &mut self.lgkm,
            Queue::Vm => &mut self.vm,
        };
        let pos = list.iter().position(|x| *x == id)?;
        let remaining = (list.len() - pos - 1) as u32;
        list.drain(..=pos);
        Some(remaining)
    }

    /// Wait for everything outstanding on a queue (`lgkmcnt(0)` / `vmcnt(0)`), a no-op if nothing is outstanding.
    pub fn wait_lgkm_all(&mut self) {
        if !self.lgkm.is_empty() {
            self.lgkm.clear();
            self.push(AInst { op: "SWaitLgkm", dst: None, srcs: vec![], imm: 0 });
        }
    }

    pub fn wait_vm_all(&mut self) {
        if !self.vm.is_empty() {
            self.vm.clear();
            self.push(AInst { op: "SWaitVm", dst: None, srcs: vec![], imm: 0 });
        }
    }

    /// `s_waitcnt lgkmcnt(k)`: wait until at most `k` DS/scalar-memory operations are outstanding. A count that does not wait for anything (`k` >= the
    /// number outstanding) is an error, not a silent no-op: it means the author's model of the queue is wrong.
    pub fn wait_lgkm_cnt(&mut self, k: u32) -> Result<(), String> {
        if k as usize >= self.lgkm.len() {
            return Err(format!("lgkmcnt({k}) waits for nothing: only {} operation(s) outstanding", self.lgkm.len()));
        }
        let n = self.lgkm.len() - k as usize;
        self.lgkm.drain(..n);
        self.push(AInst { op: "SWaitLgkm", dst: None, srcs: vec![], imm: k as i64 });
        Ok(())
    }

    /// `s_waitcnt vmcnt(k)` (same rule).
    pub fn wait_vm_cnt(&mut self, k: u32) -> Result<(), String> {
        if k as usize >= self.vm.len() {
            return Err(format!("vmcnt({k}) waits for nothing: only {} operation(s) outstanding", self.vm.len()));
        }
        let n = self.vm.len() - k as usize;
        self.vm.drain(..n);
        self.push(AInst { op: "SWaitVm", dst: None, srcs: vec![], imm: k as i64 });
        Ok(())
    }

    /// Loads still outstanding (for tests and the exit contract): (lgkm, vm).
    pub fn outstanding(&self) -> (usize, usize) {
        (self.lgkm.len(), self.vm.len())
    }

    // ---- output ---------------------------------------------------------------------------------------------------------------
    /// The recorded instructions.
    pub fn insts(&self) -> &[AInst] {
        &self.insts
    }

    /// One op name per line (for tests and diffs).
    pub fn listing(&self) -> String {
        self.insts.iter().map(|i| i.op.to_string()).collect::<Vec<_>>().join("\n")
    }

    /// Pin the block: the hand-authored instruction order is preserved by the scheduler (no instruction moves into, out of or across it).
    pub fn pin(&mut self) {
        self.pinned = true;
    }

    /// Finish a LOOP BODY: the block runs once per iteration, so it must be DRAIN-CLOSED. It enters with nothing in flight (the previous iteration
    /// ended clean) and must exit the same way: every matrix result retired (20 wait states), no outstanding DS/VMEM loads, no matrix `output_exit`.
    /// The straight-line verifier is then sound for every iteration because the entry and exit wait clocks are defined (all waits satisfied).
    pub fn finish_loop_body(mut self) -> Result<AsmBlock, String> {
        self.loop_body = true;
        self.finish()
    }

    /// Finish the sequence into a verified plain-data block. Fails (with every reason) if a token was abandoned without being retired /
    /// waited for, loads are still outstanding, or the hazard verifier finds a violation of the wait-state model.
    pub fn finish(self) -> Result<AsmBlock, String> {
        let mut errs: Vec<String> = self.poison.borrow().clone();
        if !self.lgkm.is_empty() || !self.vm.is_empty() {
            errs.push("sequence ends with loads still outstanding: wait for them (Pending::wait / wait_*_all)".to_string());
        }
        let block = AsmBlock {
            name: "safeasm".into(),
            regs: self.regs,
            alias: self.alias,
            insts: self.insts,
            inputs: self.inputs,
            outputs: self.outputs,
            exit_in_flight: self.exit_in_flight,
            loop_body: self.loop_body,
            pinned: self.pinned,
        };
        if let Err(v) = block.verify() {
            errs.extend(v);
        }
        if errs.is_empty() {
            Ok(block)
        } else {
            Err(errs.join("; "))
        }
    }
}
struct ViewCheck<const I: usize, const N: usize>;
impl<const I: usize, const N: usize> ViewCheck<I, N> {
    const OK: () = assert!(I < N, "view index out of range for this register tuple");
}

struct LaneCheck<const I: usize, const N: usize>;
impl<const I: usize, const N: usize> LaneCheck<I, N> {
    const OK: () = assert!(I < N, "lane index out of range for this register tuple");
}

impl<T> Pending<T> {
    /// Wait for this load (and, in order, every load issued before it): emits `s_waitcnt lgkmcnt(k)` / `vmcnt(k)` with the partial count
    /// from the scoreboard (nothing if an earlier wait already covered it) and returns the register, now readable.
    pub fn wait(mut self, s: &mut Seq) -> T {
        if let Some(k) = s.complete(self.queue, self.id) {
            let op = match self.queue {
                Queue::Lgkm => "SWaitLgkm",
                Queue::Vm => "SWaitVm",
            };
            s.push(AInst { op, dst: None, srcs: vec![], imm: k as i64 });
        }
        self.value.take().expect("pending value")
    }
}

impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        if self.value.is_some() {
            self.poison.borrow_mut().push("a load was abandoned without wait(): its destination register is never made readable".to_string());
        }
    }
}

/// Opcode identities of the matrix tokens: `chain` is only defined for the accumulate opcodes (the hardware interlocks a dependent in-place op of
/// the SAME opcode; a different opcode, including the `_zero` form, needs the full result latency).
pub const OP_MFMA_SCALE: u32 = 0;
pub const OP_MFMA_SCALE_ZERO: u32 = 1;
pub const OP_SMFMAC_FP8: u32 = 2;
/// The f16 32x32x8 MFMA with an implicit zero accumulator (the bias MFMA). A different opcode from the scaled one: the scaled MFMA that accumulates
/// into its result must wait for the full result latency (`retire`), 12 wait states for this opcode (hazards.rs `mfma_latency`).
pub const OP_MFMA_F16_ZERO: u32 = 3;
/// The UNSCALED fp8 MFMA (accumulate form: chainable in place) and its zero-accumulator form (a different opcode: `retire` before an accumulating one).
pub const OP_MFMA_FP8: u32 = 4;
pub const OP_MFMA_FP8_ZERO: u32 = 5;
/// Result latency in wait states of the f16 32x32x8 MFMA (hazards.rs).
pub const MATRIX_RESULT_F16: u32 = 12;

/// An in-flight matrix instruction of opcode `OP`. Owns the accumulator and holds the borrows of every operand until it is retired / exited:
/// the result cannot be read and an operand the instruction is still reading cannot be overwritten (borrow-check errors). Abandoning the token
/// (`drop`, `let _ =`) loses the accumulator register, and `Seq::finish` reports it ("unretired matrix result").
#[must_use = "an in-flight MFMA must be retired (or chained, or exited) before its result is used"]
pub struct Mfma<'a, const OP: u32> {
    pub(crate) issued: u32,
    pub(crate) lat: u32,
    pub(crate) acc: Option<Vgpr<16>>,
    pub(crate) poison: Poison,
    pub(crate) _p: PhantomData<&'a ()>,
}

/// An in-flight SMFMAC (same discipline as `Mfma`).
#[must_use = "an in-flight SMFMAC must be retired (or chained, or exited) before its result is used"]
pub struct Smfmac<'a, const OP: u32> {
    pub(crate) issued: u32,
    pub(crate) acc: Option<Vgpr<4>>,
    pub(crate) poison: Poison,
    pub(crate) _p: PhantomData<&'a ()>,
}

impl<'a, const OP: u32> Mfma<'a, OP> {
    /// Wait until the result is readable (pads exactly the missing wait states), release the operands and give the accumulator back.
    pub fn retire(mut self, s: &mut Seq) -> Vgpr<16> {
        s.pad_after(self.issued, self.lat);
        let acc = self.acc.take().expect("accumulator");
        s.clear_write(acc.id);
        acc
    }

    /// Leave the sequence with the result still in flight (the compiler's hazard pass pads the consumer after the block).
    pub fn exit(mut self) -> Exit<16> {
        let acc = self.acc.take().expect("accumulator");
        Exit { id: acc.id, issued: self.issued }
    }
}

impl<'a, const OP: u32> Drop for Mfma<'a, OP> {
    fn drop(&mut self) {
        if self.acc.is_some() {
            self.poison.borrow_mut().push("unretired matrix result: an MFMA token was dropped without retire()/exit()".to_string());
        }
    }
}

impl<'a, const OP: u32> Smfmac<'a, OP> {
    pub fn retire(mut self, s: &mut Seq) -> Vgpr<4> {
        s.pad_after(self.issued, MATRIX_RESULT);
        let acc = self.acc.take().expect("accumulator");
        s.clear_write(acc.id);
        acc
    }

    pub fn exit(mut self) -> Exit<4> {
        let acc = self.acc.take().expect("accumulator");
        Exit { id: acc.id, issued: self.issued }
    }
}

impl<'a, const OP: u32> Drop for Smfmac<'a, OP> {
    fn drop(&mut self) {
        if self.acc.is_some() {
            self.poison.borrow_mut().push("unretired matrix result: an SMFMAC token was dropped without retire()/exit()".to_string());
        }
    }
}