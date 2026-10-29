//! Typed register classes. Handles are plain `u32` slots into the sequence builder's register table: no pointers, no `unsafe`.
//!
//! * `Vgpr<N>` / `Sgpr<N>`: owners of a register tuple (N dwords). Not `Clone`/`Copy`: one owner, so Rust's borrow rules (`&` read,
//!   `&mut` write) are the aliasing rules of the register file.
//! * `Sub<'a, M>` / `SubMut<'a, M>`: views of a part of a tuple that keep the parent borrowed.
//! * `Imm<K>` (inline constant, range checked at compile time) and `Lit` (32-bit literal): operand-only kinds.
//! * `Ready<'a, N>`: proof that a register may be read by a hazard-sensitive instruction (see `seq`).

use amdgpu_compiler::mir::{RClass, VReg};
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}
pub(crate) use sealed::Sealed;

/// Width marker; only the widths the ISA has are implemented (sealed).
pub struct Wd<const N: usize>;
impl<const N: usize> Sealed for Wd<N> {}

pub trait VectorWidth: Sealed {
    const CLASS: RClass;
}
pub trait ScalarWidth: Sealed {
    const CLASS: RClass;
}
macro_rules! vwidth {
    ($($n:literal => $c:ident),*) => { $(impl VectorWidth for Wd<$n> { const CLASS: RClass = RClass::$c; })* };
}
vwidth!(1 => V1, 2 => V2, 4 => V4, 8 => V8, 16 => V16);
impl ScalarWidth for Wd<1> {
    const CLASS: RClass = RClass::S1;
}
impl ScalarWidth for Wd<2> {
    const CLASS: RClass = RClass::S2;
}

/// A vector register tuple of `N` dwords owned by the sequence.
#[derive(Debug)]
pub struct Vgpr<const N: usize> {
    pub(crate) id: VReg,
}

/// A scalar register tuple of `N` dwords (1 = `s`, 2 = pair).
#[derive(Debug)]
pub struct Sgpr<const N: usize> {
    pub(crate) id: VReg,
}

/// Shared view of `M` dwords of a vector register (keeps the parent borrowed).
#[derive(Debug, Clone, Copy)]
pub struct Sub<'a, const M: usize> {
    pub(crate) id: VReg,
    pub(crate) _p: PhantomData<&'a ()>,
}

/// Exclusive view of `M` dwords of a vector register (keeps the parent mutably borrowed).
#[derive(Debug)]
pub struct SubMut<'a, const M: usize> {
    pub(crate) id: VReg,
    pub(crate) _p: PhantomData<&'a mut ()>,
}

/// Inline constant operand (-16..=64). The range is checked at compile time.
#[derive(Debug, Clone, Copy)]
pub struct Imm<const K: i32>;
impl<const K: i32> Imm<K> {
    const OK: () = assert!(K >= -16 && K <= 64, "inline constants are -16..=64; use Lit for a 32-bit literal (VOP2/VOP1/VOPC src0 only)");
    pub const fn new() -> Self {
        let () = Self::OK;
        Imm
    }
}

/// 32-bit literal operand: encodable only in src0 of the e32 (VOP1/VOP2/VOPC) forms; costs a constant-bus slot.
#[derive(Debug, Clone, Copy)]
pub struct Lit(pub i32);

/// Proof that the register `id` may be read by a hazard-sensitive instruction (matrix operand, `v_readfirstlane`): the sequence
/// padded the required wait states when it issued this token. Borrows the register while it is alive.
#[derive(Debug, Clone, Copy)]
pub struct Ready<'a, const N: usize> {
    pub(crate) id: VReg,
    pub(crate) _p: PhantomData<&'a ()>,
}

/// A readable vector operand of width `M` (plain register or view).
pub trait Src<const M: usize>: Sealed {
    #[doc(hidden)]
    fn src_id(&self) -> VReg;
}
impl<const N: usize> Sealed for &Vgpr<N> {}
impl<const N: usize> Src<N> for &Vgpr<N> {
    fn src_id(&self) -> VReg {
        self.id
    }
}
impl<const M: usize> Sealed for Sub<'_, M> {}
impl<const M: usize> Src<M> for Sub<'_, M> {
    fn src_id(&self) -> VReg {
        self.id
    }
}

/// A writable vector destination of width `M`.
pub trait Dst<const M: usize>: Sealed {
    #[doc(hidden)]
    fn dst_id(&self) -> VReg;
}
impl<const N: usize> Sealed for &mut Vgpr<N> {}
impl<const N: usize> Dst<N> for &mut Vgpr<N> {
    fn dst_id(&self) -> VReg {
        self.id
    }
}
impl<const M: usize> Sealed for SubMut<'_, M> {}
impl<const M: usize> Dst<M> for SubMut<'_, M> {
    fn dst_id(&self) -> VReg {
        self.id
    }
}

/// Source operand kinds of a 32-bit VALU instruction, with their constant-bus cost (gfx9: at most ONE scalar/literal per instruction).
pub trait Operand: Sealed {
    /// Constant-bus slots this operand uses.
    const BUS: u8;
    /// Allowed in src1/src2 of an e32 form and in VOP3 slots (everything but a literal).
    const VOP3_OK: bool;
    #[doc(hidden)]
    fn to_src(&self) -> amdgpu_compiler::mir::Src;
}
impl Operand for &Vgpr<1> {
    const BUS: u8 = 0;
    const VOP3_OK: bool = true;
    fn to_src(&self) -> amdgpu_compiler::mir::Src {
        amdgpu_compiler::mir::Src::Reg(self.id)
    }
}
impl Operand for Sub<'_, 1> {
    const BUS: u8 = 0;
    const VOP3_OK: bool = true;
    fn to_src(&self) -> amdgpu_compiler::mir::Src {
        amdgpu_compiler::mir::Src::Reg(self.id)
    }
}
impl Sealed for &Sgpr<1> {}
impl Operand for &Sgpr<1> {
    const BUS: u8 = 1;
    const VOP3_OK: bool = true;
    fn to_src(&self) -> amdgpu_compiler::mir::Src {
        amdgpu_compiler::mir::Src::Reg(self.id)
    }
}
impl<const K: i32> Sealed for Imm<K> {}
impl<const K: i32> Operand for Imm<K> {
    const BUS: u8 = 0;
    const VOP3_OK: bool = true;
    fn to_src(&self) -> amdgpu_compiler::mir::Src {
        let () = Imm::<K>::OK;
        amdgpu_compiler::mir::Src::Imm(K as i64)
    }
}
impl Sealed for Lit {}
impl Operand for Lit {
    const BUS: u8 = 1;
    const VOP3_OK: bool = false;
    fn to_src(&self) -> amdgpu_compiler::mir::Src {
        amdgpu_compiler::mir::Src::Imm(self.0 as i64)
    }
}

/// A source that is a plain VGPR (the only kind allowed in src1 of VOP2/VOPC e32 forms and in matrix/DS data operands).
pub trait VgprOnly: Operand {}
impl VgprOnly for &Vgpr<1> {}
impl VgprOnly for Sub<'_, 1> {}