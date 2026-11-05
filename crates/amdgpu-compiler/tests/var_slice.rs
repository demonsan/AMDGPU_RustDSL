//! `var_set_lane` + `var_slice8`: values written straight into the dwords of a Frag16 loop variable and its two halves used as Frag8 MFMA
//! operands without copies (the vendor keeps P in the registers of the dead S tile); `verify` rejects a slice used after the variable is written.
use amdgpu_dsl::tile::ev;
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

fn kernel(bad: bool) -> Function {
    kernel2(bad, false)
}

/// `twice`: every word is also written into the accumulator (two uses: both writes must copy).
fn kernel2(bad: bool, twice: bool) -> Function {
    let mut b = FnBuilder::new("probe_var_slice", 64);
    let tid = b.let_(Op::WorkitemId);
    let z = b.frag_zero(DType::Frag16);
    let acc0 = b.frag_zero(DType::Frag16);
    let vars = b.begin_loop(&[(DType::Frag16, Opnd::Val(z)), (DType::Frag16, Opnd::Val(acc0))]);
    let (s, acc) = (vars[0], vars[1]);
    for k in 0..16u8 {
        let w = b.bind_i32(ev(tid) * (k as i32 + 3) + 0x3838);
        b.var_set_lane(s, k, w);
        if twice {
            b.var_set_lane(acc, k, w);
        }
    }
    let lo = b.var_slice8(s, 0);
    let hi = b.var_slice8(s, 1);
    if bad {
        b.var_set_lane(s, 3, tid);
    }
    b.var_mfma_fp8_32x32x64(acc, lo, hi);
    let one = b.c_i32(1);
    let zero = b.c_i32(0);
    let stop = b.cmp_lt_i32(zero, one);
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    b.finish()
}

#[test]
fn lanes_are_written_in_place_and_the_halves_feed_the_mfma() {
    let f = kernel(false);
    assert!(amdgpu_dsl::verify(&f).is_empty(), "{:?}", amdgpu_dsl::verify(&f));
    // copies into variable dwords (v_mov with an alias destination): the loop-entry initialisation only, unless a word has two uses
    let movs = |f: &Function| {
        let m = amdgpu_compiler::lower::lower(f).expect("lower");
        m.blocks.iter().flat_map(|b| b.insts.iter()).filter(|i| i.op == amdgpu_compiler::mir::MOp::VMovB32 && i.dst.map_or(false, |d| m.alias.contains_key(&d))).count()
    };
    assert_eq!(movs(&kernel2(false, true)), movs(&f) + 32, "single-use words are written in place (no copy); two-use words are copied twice");
    let s = amdgpu_compiler::compile(&f).expect("compile").asm;
    let mf = s.lines().find(|l| l.trim_start().starts_with("v_mfma_f32_32x32x64")).unwrap_or_else(|| panic!("{s}"));
    // A and B are the two adjacent halves of one 16-register range
    let regs: Vec<u32> = mf.split(|c: char| c == '[' || c == ':' || c == ']').filter_map(|t| t.parse().ok()).collect();
    assert!(regs.len() >= 6 && regs[4] == regs[2] + 8, "{mf}");
}

#[test]
fn a_slice_used_after_the_variable_is_written_is_rejected() {
    let errs = amdgpu_dsl::verify(&kernel(true));
    assert!(errs.iter().any(|e| e.contains("slice of var")), "{errs:?}");
}
