//! Partial-write destinations: `v_cvt_pk_fp8_f32` / `v_cvt_scalef32_pk_fp8_bf16` write ONE 16-bit half of a dword (`op_sel`) and keep the other
//! half (the destination is also a tied source). `Pw<L, H>` is a type-state wrapper that owns the register and records which halves have been
//! written; the register can only be taken back (`done`) when both are `Set`, so "read the unwritten half" does not type-check.

use crate::block::*;
use crate::regs::*;
use crate::seq::*;
use std::marker::PhantomData;

/// Half not written yet.
pub struct Unset;
/// Half written (or known defined on entry).
pub struct Set;

/// A one-dword register with per-half write state.
pub struct Pw<L, H> {
    reg: Vgpr<1>,
    _p: PhantomData<(L, H)>,
}

impl Pw<Set, Set> {
    /// Both halves are written: the register is an ordinary one-dword register again.
    pub fn done(self) -> Vgpr<1> {
        self.reg
    }
}

struct Bus1<A, B>(PhantomData<(A, B)>);
impl<A: Operand, B: Operand> Bus1<A, B> {
    const OK: () = assert!(A::BUS + B::BUS <= 1 && A::VOP3_OK && B::VOP3_OK, "gfx9 VOP3: at most one SGPR/literal source per instruction and no 32-bit literal");
}

impl Seq {
    /// Start writing a fresh (undefined) dword half by half.
    pub fn partial(&mut self, r: Vgpr<1>) -> Pw<Unset, Unset> {
        Pw { reg: r, _p: PhantomData }
    }

    fn pw_op<L, H, L2, H2, A: Operand, B: Operand>(&mut self, op: &'static str, pw: Pw<L, H>, a: A, b: B, tied: bool) -> Pw<L2, H2> {
        let () = Bus1::<A, B>::OK;
        let d = pw.reg.id;
        let mut srcs = vec![a.to_src(), b.to_src()];
        if tied {
            srcs.push(ASrc::Reg(d));
        }
        self.push(AInst { op, dst: Some(d), srcs, imm: 0 });
        self.note_valu_write(d);
        Pw { reg: pw.reg, _p: PhantomData }
    }

    /// `v_cvt_scalef32_pk_fp8_bf16` low half (op_sel 0): `data` = packed bf16 pair, `scale`.
    pub fn v_cvt_scalef32_pk_fp8_bf16_lo<H, A: Operand, B: Operand>(&mut self, pw: Pw<Unset, H>, data: A, scale: B) -> Pw<Set, H> {
        self.pw_op("VCvtPkFp8Lo", pw, data, scale, false)
    }
    /// ... high half (op_sel 8/4): keeps the low half (tied source).
    pub fn v_cvt_scalef32_pk_fp8_bf16_hi<L, A: Operand, B: Operand>(&mut self, pw: Pw<L, Unset>, data: A, scale: B) -> Pw<L, Set> {
        self.pw_op("VCvtPkFp8Hi", pw, data, scale, true)
    }
    /// `v_cvt_pk_fp8_f32` low half: two fp32 inputs -> two fp8 in the low 16 bits.
    pub fn v_cvt_pk_fp8_f32_lo<H, A: Operand, B: Operand>(&mut self, pw: Pw<Unset, H>, a: A, b: B) -> Pw<Set, H> {
        self.pw_op("VCvtPkFp8F32", pw, a, b, false)
    }
    /// ... high half.
    pub fn v_cvt_pk_fp8_f32_hi<L, A: Operand, B: Operand>(&mut self, pw: Pw<L, Unset>, a: A, b: B) -> Pw<L, Set> {
        self.pw_op("VCvtPkFp8F32Hi", pw, a, b, true)
    }
}