//! The VALU-writes-SGPR hazards (hazards::pad_valu_sgpr; LLVM GCNHazardRecognizer checkVMEMHazards / hasVDecCoExecHazard), pinned without
//! hardware: in the asm of probe_valu_sgpr every SGPR written by `v_readfirstlane` is read by a VMEM instruction only after >= 5 wait states
//! and by a VALU instruction only after >= 2 (the hardware read a stale buffer soffset without them).

use amdgpu_compiler::backend::OwnTail;
use amdgpu_compiler::compile_with;
use amdgpu_kernels::kernel_by_name;

/// wait states of an asm line: `s_nop k` = k + 1, any other instruction 1, labels / directives / blank 0
fn states(l: &str) -> usize {
    let t = l.trim();
    if t.is_empty() || t.ends_with(':') || t.starts_with('.') || t.starts_with(';') || t.starts_with("//") {
        return 0;
    }
    match t.strip_prefix("s_nop ") {
        Some(k) => k.trim().parse::<usize>().unwrap_or(0) + 1,
        None => 1,
    }
}

fn reads_sgpr(l: &str, s: &str) -> bool {
    // the SGPR as an operand (not the destination): any comma-separated operand after the first
    let t = l.trim();
    let Some((_, ops)) = t.split_once(' ') else { return false };
    ops.split(',').skip(1).any(|o| o.trim().split_whitespace().next() == Some(s))
}

#[test]
fn readfirstlane_results_wait_before_vmem_and_valu_readers() {
    let f = kernel_by_name("probe_valu_sgpr").expect("registered");
    let asm = compile_with(&f, &OwnTail).expect("compiles").asm;
    let lines: Vec<&str> = asm.lines().collect();
    let (mut vmem_sites, mut valu_sites) = (0, 0);
    for (i, l) in lines.iter().enumerate() {
        let t = l.trim();
        let Some(rest) = t.strip_prefix("v_readfirstlane_b32 ") else { continue };
        let s = rest.split(',').next().unwrap().trim().to_string();
        let mut gap = 0usize;
        for m in &lines[i + 1..] {
            let mt = m.trim();
            // the register is redefined: stop
            if mt.split(' ').nth(1).map_or(false, |d| d.trim_end_matches(',') == s) && !mt.starts_with("buffer_") {
                break;
            }
            if reads_sgpr(mt, &s) {
                if mt.starts_with("buffer_") || mt.starts_with("global_") {
                    assert!(gap >= 5, "VMEM read of {s} after {gap} wait states (need 5):\n{l}\n...\n{mt}");
                    vmem_sites += 1;
                } else if mt.starts_with("v_") {
                    assert!(gap >= 2, "VALU read of {s} after {gap} wait states (need 2):\n{l}\n...\n{mt}");
                    valu_sites += 1;
                }
            }
            gap += states(mt);
            if gap > 8 {
                break;
            }
        }
    }
    assert!(vmem_sites >= 32, "the probe's readfirstlane -> soffset sites: {vmem_sites}");
    let _ = valu_sites;
}
