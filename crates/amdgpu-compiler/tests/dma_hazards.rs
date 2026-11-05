//! Hazard: a VALU instruction writing a VGPR followed by `v_readfirstlane` of that VGPR needs wait states.
//! Found on hardware with a DMA kernel (the DMA path reads the LDS base with v_readfirstlane into M0):
//! `v_add d, ..; v_readfirstlane s, d; s_mov m0, s; global_load_lds` returned a stale lane value. Same hazard class as
//! "VALU write VGPR -> MFMA operand / permlane" (2 wait states; 4 after carry-out adds).

use amdgpu_compiler::compile;
use amdgpu_dsl::{DType, FnBuilder, Op};

#[test]
fn readfirstlane_after_valu_write_is_padded() {
    let mut b = FnBuilder::new("rfl_hazard", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.lds(4096);
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c = b.c_i32(256);
    let a = b.add_i32(tid, c); // VALU-produced LDS base, consumed immediately by the DMA setup
    b.load_global_lds(p, a, 0, 4);
    let f = b.finish();
    let asm = compile(&f).expect("compile").asm;
    let lines: Vec<&str> = asm.lines().map(|l| l.split(';').next().unwrap().trim()).filter(|l| !l.is_empty()).collect();
    let i = lines.iter().position(|l| l.starts_with("v_readfirstlane_b32")).expect("readfirstlane in the asm");
    let src = lines[i].split(',').nth(1).unwrap().trim().to_string();
    // wait states between the VALU writer of `src` and the readfirstlane: instructions in between, s_nop N counts N+1
    let mut states = 0usize;
    for j in (0..i).rev() {
        let l = lines[j];
        if l.starts_with("s_nop") {
            states += l.trim_start_matches("s_nop").trim().parse::<usize>().unwrap_or(0) + 1;
            continue;
        }
        let dst = l.split_whitespace().nth(1).map(|d| d.trim_end_matches(',').to_string()).unwrap_or_default();
        if l.starts_with("v_") && dst == src {
            assert!(states >= 2, "v_readfirstlane reads {src} written by `{l}` with only {states} wait states in between:\n{asm}");
            return;
        }
        states += 1;
        if states >= 8 {
            return;
        }
    }
}