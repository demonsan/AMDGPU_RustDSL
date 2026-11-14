//! In-process gfx950 assembler: the compiler's assembly text -> code object, without the external clang (docs/compile-time.md). `enc` encodes instructions, `tables` is generated from the llvm-mc oracle.

pub mod enc;
pub mod tables;