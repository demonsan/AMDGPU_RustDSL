//! The implicit SCC / VCC effects of every emitted instruction expansion must be declared in the instruction table (`op_attrs`): the
//! scheduler, the hazard and wait passes and the threading pass reason from the table, not from the text. A pseudo whose expansion
//! writes SCC without declaring it (the buffer-descriptor pseudos' `s_and_b32`) let the scheduler move it between an `s_cmp` and the
//! `s_cselect` reading it (wrong MLA decode results under the default scheduler). This test emits the final MIR of every reachable kernel
//! (shipped order and, where the scheduler is not the kernel's own, the default scheduler order) and checks each expansion's text against
//! its table row.

use amdgpu_codegen::mir::{MFunction, MInst};
use amdgpu_codegen::regalloc::Alloc;

/// SALU mnemonics that write SCC (ISA: the arithmetic / logic / shift / compare / bit-field forms).
fn writes_scc(mn: &str) -> bool {
    const P: &[&str] = &[
        "s_add_", "s_addc_", "s_sub_", "s_subb_", "s_and_", "s_or_", "s_xor_", "s_andn2_", "s_orn2_", "s_nand_", "s_nor_", "s_xnor_", "s_lshl_", "s_lshr_", "s_ashr_", "s_bfe_",
        "s_min_", "s_max_", "s_cmp_", "s_cmpk_", "s_abs_", "s_absdiff_", "s_bitcmp", "s_not_", "s_addk_", "s_lshl1_add", "s_lshl2_add", "s_lshl3_add", "s_lshl4_add", "s_bcnt", "s_ff1", "s_flbit",
        "s_quadmask", "s_wqm",
    ];
    P.iter().any(|p| mn.starts_with(p))
}

fn reads_scc(mn: &str) -> bool {
    mn.starts_with("s_cselect_") || mn.starts_with("s_addc_") || mn.starts_with("s_subb_") || mn.starts_with("s_cmov")
}

/// (writes vcc, reads vcc) of one line.
fn vcc_effects(line: &str) -> (bool, bool) {
    let mut it = line.split_whitespace();
    let mn = it.next().unwrap_or("");
    let ops: Vec<&str> = line[mn.len()..].split(',').map(|s| s.trim()).collect();
    let is_cmp_e32 = mn.starts_with("v_cmp") && (mn.ends_with("_e32") || ops.first() == Some(&"vcc"));
    let carry_out = (mn.starts_with("v_add_co") || mn.starts_with("v_addc_co") || mn.starts_with("v_sub_co") || mn.starts_with("v_subb_co") || mn.starts_with("v_mad_u64") || mn.starts_with("v_mad_i64"))
        && ops.get(1) == Some(&"vcc");
    let w = is_cmp_e32 || carry_out;
    let r = ops.iter().skip(if w { 1 } else { 0 }).skip(if carry_out { 1 } else { 0 }).any(|o| *o == "vcc") || mn.starts_with("v_cndmask") && mn.ends_with("_e32");
    (w, r)
}

fn check(name: &str, m: &MFunction, a: &Alloc, errs: &mut Vec<String>) {
    for b in &m.blocks {
        for i in &b.insts {
            let at = amdgpu_codegen::op_attrs::attrs(i.op);
            let lines: Vec<String> = amdgpu_codegen::emit::inst_lines(a, i);
            let (mut sw, mut sr, mut vw, mut vr) = (false, false, false, false);
            for l in &lines {
                let mn = l.split_whitespace().next().unwrap_or("");
                sw |= writes_scc(mn);
                sr |= reads_scc(mn);
                let (w, r) = vcc_effects(l);
                vw |= w;
                vr |= r;
            }
            let mut miss = Vec::new();
            if sw && !at.scc_write {
                miss.push("scc write");
            }
            if sr && !at.scc_read {
                miss.push("scc read");
            }
            if vw && !(at.vcc_write || at.defines_vcc) {
                miss.push("vcc write");
            }
            if vr && !(at.vcc_read || at.consumes_vcc) {
                miss.push("vcc read");
            }
            if !miss.is_empty() {
                let key = format!("{:?}: {} not declared (emits {:?})", i.op, miss.join(", "), lines);
                if !errs.iter().any(|e: &String| e.starts_with(&format!("{:?}:", i.op))) {
                    errs.push(format!("{key} [first seen in {name}]"));
                }
            }
        }
    }
}

#[test]
fn every_emitted_expansion_declares_its_implicit_scc_and_vcc_effects() {
    let mut errs = Vec::new();
    let mut n = 0;
    for k in amdgpu_kernels::reachable::manifest_names() {
        let Some(f) = amdgpu_kernels::kernel_by_name(&k) else { continue };
        if let Ok((m, a)) = amdgpu_codegen::compile_mir_alloc(&f) {
            check(&k, &m, &a, &mut errs);
            n += 1;
        }
    }
    assert!(n > 200, "only {n} kernels compiled");
    assert!(errs.is_empty(), "{} instruction kind(s) emit implicit effects their table row does not declare:\n{}", errs.len(), errs.join("\n"));
}

#[test]
fn the_vcc_and_scc_classifiers() {
    assert!(writes_scc("s_and_b32") && writes_scc("s_cmp_lt_i32") && !writes_scc("s_mov_b32") && !writes_scc("s_cselect_b32"));
    assert!(reads_scc("s_cselect_b32") && !reads_scc("s_cmp_eq_u32"));
    assert_eq!(vcc_effects("v_cmp_lt_i32_e32 vcc, v1, v2"), (true, false));
    assert_eq!(vcc_effects("v_add_co_u32 v1, vcc, v2, v3"), (true, false));
    assert_eq!(vcc_effects("v_addc_co_u32 v1, vcc, v2, v3, vcc"), (true, true));
    assert_eq!(vcc_effects("v_cndmask_b32_e32 v1, v2, v3, vcc"), (false, true));
}
