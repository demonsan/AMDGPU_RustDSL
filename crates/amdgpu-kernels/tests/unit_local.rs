//! W3 B2 (docs/two-weavers.md section 3.2): with the lowering token `unitlocal`, the MIR of a plan unit (`Stmt::UnitMark`) uses no
//! lowering cache entry (constant, scalar->vector conversion, readfirstlane result) materialised by another unit, so the back end may
//! reorder units. Without the token (and for every kernel without unit marks) nothing changes.

use amdgpu_codegen::mir::{MFunction, MOp, Src};
use amdgpu_dsl::{ev, DType, Ext, FnBuilder, Function, Op};
use std::collections::HashMap;

/// Two units that both use the constant 7 and the same uniform kernarg converted to a vector value.
fn two_units(tokens: &[&str]) -> Function {
    let mut b = FnBuilder::new("probe_unit_local", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg("n", DType::I32, 0x08, 4);
    b.arg("pad", DType::I32, 0x0C, 4);
    b.arg_tensor("o", Ext::c(128 * 4));
    b.ka_range(0x08, 1, 64);
    b.lower_opts(tokens);
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let n = b.let_(Op::KernargI32 { offset: 0x08 });
    let tid = b.let_(Op::WorkitemId);
    for (u, base) in [(0u32, 0i32), (1, 256)] {
        b.unit_begin(u);
        let x = b.bind_i32(ev(tid) * 7 + n);
        let off = b.bind_i32((ev(tid) << 2) + base);
        let q = b.ptr_add(po, off);
        let f = b.let_(Op::BitcastF32(x.into()));
        b.store_f32(q, 0, f);
        b.unit_end(u);
    }
    b.finish()
}

fn lowered(tok: &[&str]) -> MFunction {
    let f = two_units(tok);
    let v: Vec<&str> = [&["dce", "cmpfold"][..], tok].concat();
    amdgpu_codegen::lower::with_opts(&v, || amdgpu_codegen::lower::lower(&f)).unwrap()
}

/// (consumer unit, producer unit) pairs where a unit reads a register that another unit's constant / conversion created.
fn cross_unit_cache_uses(m: &MFunction) -> Vec<(u32, u32)> {
    let mut def_unit: HashMap<u32, u32> = HashMap::new();
    for b in &m.blocks {
        for i in &b.insts {
            if let Some(d) = i.dst {
                if matches!(i.op, MOp::VMovB32 | MOp::SMovB32 | MOp::VMovFromS | MOp::VReadFirstLane) && i.unit > 0 {
                    def_unit.entry(m.canon(d)).or_insert(i.unit);
                }
            }
        }
    }
    let mut out = Vec::new();
    for b in &m.blocks {
        for i in &b.insts {
            if i.unit == 0 {
                continue;
            }
            for s in &i.srcs {
                if let Src::Reg(r) = s {
                    if let Some(&u) = def_unit.get(&m.canon(*r)) {
                        if u != i.unit {
                            out.push((i.unit, u));
                        }
                    }
                }
            }
        }
    }
    out
}

#[test]
fn without_the_token_the_second_unit_reuses_the_first_units_constants() {
    let m = lowered(&[]);
    assert!(m.blocks.iter().flat_map(|b| b.insts.iter()).any(|i| i.unit == 2), "units are tagged");
    assert!(!cross_unit_cache_uses(&m).is_empty(), "today's lowering shares caches across units");
}

#[test]
fn with_unitlocal_no_unit_uses_another_units_cache_entries() {
    let m = lowered(&["unitlocal"]);
    assert_eq!(cross_unit_cache_uses(&m), vec![]);
    // and the kernel still compiles end to end and verifies
    let f = two_units(&["unitlocal"]);
    amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
    let r = amdgpu_codegen::gpu_verify::analyse(&f).expect("analyse");
    assert!(r.errors().is_empty(), "{:?}", r.errors().iter().map(|e| e.msg.clone()).collect::<Vec<_>>());
}
