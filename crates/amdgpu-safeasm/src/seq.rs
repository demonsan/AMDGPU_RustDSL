//! The sequence builder: records instructions as MIR (`MInst`) so register allocation, scheduling and the hazard passes stay ours,
//! tracks a wait-state clock, and hands out the type-state tokens that make hazards impossible to forget.

use crate::regs::*;
use amdgpu_compiler::mir::{Block, MFunction, MInst, MOp, RClass, Src, VReg};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Wait states (instructions or `s_nop` slots) that must separate a VALU write of a VGPR from a hazard-sensitive reader
/// (matrix operand, `v_readfirstlane`): "2 required waits", CDNA4 ISA 7.6, LLVM hazard recognizer.
pub const VALU_TO_MATRIX_READ: u32 = 2;
/// Wait states between a (scaled) MFMA / SMFMAC and the first reader of its result (hardware-confirmed: 19 is still stale).
pub const MATRIX_RESULT: u32 = 20;
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

/// The single `vcc` register as a type-state token: a compare turns `Vcc<Free>` into `Vcc<Live>`, a `v_cndmask` consumes it. A second
/// compare while a mask is live, or a `v_cndmask` without a compare, does not type-check.
#[derive(Debug)]
pub struct Vcc<S> {
    pub(crate) _s: PhantomData<S>,
}

pub struct Seq {
    pub(crate) insts: Vec<MInst>,
    pub(crate) vregs: Vec<RClass>,
    pub(crate) alias: HashMap<VReg, (VReg, u8)>,
    pub(crate) clock: u32,
    writes: HashMap<VReg, Write>,
    vcc_taken: bool,
    vcc_written_at: u32,
    zero: Option<VReg>,
}

impl Default for Seq {
    fn default() -> Self {
        Self::new()
    }
}

impl Seq {
    pub fn new() -> Seq {
        Seq { insts: Vec::new(), vregs: Vec::new(), alias: HashMap::new(), clock: 0, writes: HashMap::new(), vcc_taken: false, vcc_written_at: 0, zero: None }
    }

    // ---- registers -----------------------------------------------------------------------------------------------------------
    fn new_reg(&mut self, c: RClass) -> VReg {
        self.vregs.push(c);
        (self.vregs.len() - 1) as VReg
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

    /// A one-dword register holding the constant `k` (`v_mov_b32 v, k`).
    pub fn splat(&mut self, k: i32) -> Vgpr<1> {
        let r = self.new_reg(RClass::V1);
        self.push(MInst::new(MOp::VMovB32).def(r).imm(k as i64));
        self.note_valu_write(r);
        Vgpr { id: r }
    }

    /// A tuple whose dwords are all zero.
    pub fn zeroed<const N: usize>(&mut self) -> Vgpr<N>
    where
        Wd<N>: VectorWidth,
    {
        let d = self.vgpr::<N>();
        if N == 1 {
            self.push(MInst::new(MOp::VMovB32).def(d.id).imm(0));
        } else {
            let z = match self.zero {
                Some(z) => z,
                None => {
                    let z = self.new_reg(RClass::V1);
                    self.push(MInst::new(MOp::VMovB32).def(z).imm(0));
                    self.zero = Some(z);
                    z
                }
            };
            for i in 0..N {
                self.push(MInst::new(MOp::VWriteSub).def(d.id).srcreg(z).imm(i as i64));
            }
        }
        self.note_valu_write(d.id);
        d
    }

    /// Dword `I` of a tuple as a one-dword read-only view (bounds checked at compile time).
    pub fn lane<'a, const N: usize, const I: usize>(&mut self, r: &'a Vgpr<N>) -> Sub<'a, 1> {
        let () = LaneCheck::<I, N>::OK;
        Sub { id: self.alias_of(r.id, I as u8, RClass::V1), _p: PhantomData }
    }

    /// Dword `I` of a tuple as a one-dword writable view.
    pub fn lane_mut<'a, const N: usize, const I: usize>(&mut self, r: &'a mut Vgpr<N>) -> SubMut<'a, 1> {
        let () = LaneCheck::<I, N>::OK;
        SubMut { id: self.alias_of(r.id, I as u8, RClass::V1), _p: PhantomData }
    }

    /// Low (`HI` = false) or high half of an 8-dword tuple as a 4-dword view; the parent stays borrowed.
    pub fn half8<'a, const HI: bool>(&mut self, r: &'a Vgpr<8>) -> Sub<'a, 4> {
        Sub { id: self.alias_of(r.id, if HI { 4 } else { 0 }, RClass::V4), _p: PhantomData }
    }

    fn alias_of(&mut self, parent: VReg, off: u8, class: RClass) -> VReg {
        let (base, o) = self.alias.get(&parent).copied().unwrap_or((parent, 0));
        let a = self.new_reg(class);
        self.alias.insert(a, (base, o + off));
        a
    }

    fn canon(&self, r: VReg) -> VReg {
        self.alias.get(&r).map(|a| a.0).unwrap_or(r)
    }

    // ---- clock and hazard bookkeeping ---------------------------------------------------------------------------------------
    pub(crate) fn push(&mut self, i: MInst) {
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

    /// `n` wait states (`s_nop n-1`, repeated as needed).
    pub fn nop(&mut self, mut n: u32) {
        while n > 0 {
            let k = n.min(8);
            self.insts.push(MInst::new(MOp::SNop).imm(k as i64 - 1));
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
    pub fn ready<'a, const N: usize>(&mut self, r: &'a Vgpr<N>) -> Ready<'a, N> {
        self.make_ready(r.id);
        Ready { id: r.id, _p: PhantomData }
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
    pub fn vcc(&mut self) -> Vcc<Free> {
        assert!(!self.vcc_taken, "vcc already taken: there is one vcc per sequence");
        self.vcc_taken = true;
        Vcc { _s: PhantomData }
    }

    pub(crate) fn vcc_written(&mut self) {
        self.vcc_written_at = self.clock - 1;
    }

    pub(crate) fn vcc_ready(&mut self) {
        let at = self.vcc_written_at;
        self.pad_after(at, VCC_TO_CNDMASK);
    }

    // ---- output ---------------------------------------------------------------------------------------------------------------
    /// The recorded MIR block.
    pub fn insts(&self) -> &[MInst] {
        &self.insts
    }

    /// One MIR op per line (for tests and diffs).
    pub fn listing(&self) -> String {
        self.insts.iter().map(|i| format!("{:?}", i.op)).collect::<Vec<_>>().join("\n")
    }

    /// The whole sequence as a MIR function (register allocation and the rest of the pipeline are the existing ones).
    pub fn to_mfunction(&self, name: &str) -> MFunction {
        let mut b = Block::new(0);
        b.insts = self.insts.clone();
        MFunction {
            name: name.to_string(),
            args: Vec::new(),
            kernarg_size: 0,
            workgroup_size: 64,
            lds_bytes: 0,
            blocks: vec![b],
            vregs: self.vregs.clone(),
            loops: Vec::new(),
            alias: self.alias.clone(),
            elide_fallthrough: false,
        }
    }

    /// Textual assembly (own backend: register allocation + emission), used by the tests and for inspection only.
    pub fn to_asm(&self) -> Result<String, String> {
        let m = self.to_mfunction("safeasm_seq");
        let a = amdgpu_compiler::regalloc::allocate(&m)?;
        Ok(amdgpu_compiler::emit::emit(&m, &a))
    }
}

impl std::fmt::Display for Seq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.to_asm() {
            Ok(a) => write!(f, "{a}"),
            Err(e) => write!(f, "<unallocatable sequence: {e}>"),
        }
    }
}

struct LaneCheck<const I: usize, const N: usize>;
impl<const I: usize, const N: usize> LaneCheck<I, N> {
    const OK: () = assert!(I < N, "lane index out of range for this register tuple");
}

/// An in-flight scaled MFMA. Holds the borrows of every operand and the exclusive borrow of the accumulator until it is retired:
/// reading the accumulator, or overwriting an operand the instruction is still reading, is a borrow-check error.
#[must_use = "an in-flight MFMA must be retired (or chained) before its result is used"]
pub struct Mfma<'a> {
    pub(crate) issued: u32,
    pub(crate) _p: PhantomData<&'a mut ()>,
}

/// An in-flight SMFMAC (same discipline as `Mfma`).
#[must_use = "an in-flight SMFMAC must be retired (or chained) before its result is used"]
pub struct Smfmac<'a> {
    pub(crate) issued: u32,
    pub(crate) _p: PhantomData<&'a mut ()>,
}

impl<'a> Mfma<'a> {
    /// Wait until the result is readable (pads exactly the missing wait states) and release the accumulator and the operands.
    pub fn retire(self, s: &mut Seq) {
        s.pad_after(self.issued, MATRIX_RESULT);
    }
}

impl<'a> Smfmac<'a> {
    pub fn retire(self, s: &mut Seq) {
        s.pad_after(self.issued, MATRIX_RESULT);
    }
}

/// Used by `ops`: a matrix-op source operand turned into an MIR source.
pub(crate) fn rd<const N: usize>(r: Ready<'_, N>) -> Src {
    Src::Reg(r.id)
}