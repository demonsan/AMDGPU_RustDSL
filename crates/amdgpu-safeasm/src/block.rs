//! Plain-data export of a sequence: what the DSL embeds (`Op::Asm`) and the compiler lowers to MIR. Op names are the `MOp` variant names.

/// Register class of a sequence register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AClass {
    V1,
    V2,
    V4,
    V8,
    V16,
    S1,
    S2,
}

impl AClass {
    pub fn dwords(self) -> u32 {
        match self {
            AClass::V1 | AClass::S1 => 1,
            AClass::V2 | AClass::S2 => 2,
            AClass::V4 => 4,
            AClass::V8 => 8,
            AClass::V16 => 16,
        }
    }
    pub fn is_vector(self) -> bool {
        matches!(self, AClass::V1 | AClass::V2 | AClass::V4 | AClass::V8 | AClass::V16)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ASrc {
    Reg(u32),
    Imm(i64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AInst {
    /// `MOp` variant name, e.g. `"VMfmaScale32x32x64Fp8"`.
    pub op: &'static str,
    pub dst: Option<u32>,
    pub srcs: Vec<ASrc>,
    pub imm: i64,
}

/// A register the DSL provides (`input`, read only) or receives (`output`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AsmPort {
    pub reg: u32,
    pub class: AClass,
    /// An in/out port (`Seq::inout`): bound to a loop variable's own register, which the sequence updates in place.
    pub mutable: bool,
}

/// A verified sequence. Only `Seq::finish` can create one (the fields are private), and creation runs the hazard verifier, so a block
/// cannot carry a violation of the clock model; the compiler re-verifies it before lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsmBlock {
    pub(crate) name: String,
    pub(crate) regs: Vec<AClass>,
    pub(crate) alias: Vec<(u32, u32, u8)>,
    pub(crate) insts: Vec<AInst>,
    pub(crate) inputs: Vec<AsmPort>,
    pub(crate) outputs: Vec<AsmPort>,
    pub(crate) exit_in_flight: Vec<(u32, u32)>,
    /// Built with `Seq::finish_loop_body`: drain-closed at both ends (nothing in flight at entry or exit).
    pub(crate) loop_body: bool,
}

impl AsmBlock {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn regs(&self) -> &[AClass] {
        &self.regs
    }
    /// (alias register, base tuple register, dword offset): views of a tuple.
    pub fn alias(&self) -> &[(u32, u32, u8)] {
        &self.alias
    }
    pub fn insts(&self) -> &[AInst] {
        &self.insts
    }
    pub fn inputs(&self) -> &[AsmPort] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[AsmPort] {
        &self.outputs
    }
    /// Registers whose last matrix write is still in flight at block exit (the consumer after the block is padded by the compiler's
    /// hazard pass, which sees the whole instruction stream): `(register, wait states still owed)`.
    pub fn is_loop_body(&self) -> bool {
        self.loop_body
    }
    pub fn exit_in_flight(&self) -> &[(u32, u32)] {
        &self.exit_in_flight
    }

    /// Re-walk the instruction list with the crate's wait-state model and report every violation (see `verify`).
    pub fn verify(&self) -> Result<(), Vec<String>> {
        crate::verify::verify_block(self)
    }
}