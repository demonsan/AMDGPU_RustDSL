//! `amdc` library part: the host-code generator (`amdc gen-host`), shared by the binary and its tests.

#![forbid(unsafe_code)]

pub mod hostgen;
pub mod jit;
pub mod manifest;
pub mod md5;
