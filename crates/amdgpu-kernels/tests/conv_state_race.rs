//! The causal_conv1d_fwd state slot race (fixed): the old protocol let every chunk of a sequence load the state slot and the sequence's LAST
//! chunk (another workgroup) write it, so the write could land before chunk 0 read the initial state (intermittent wrong outputs on
//! hardware, mostly with `_sched`). The shipped kernel lets chunk 0 alone read and write the slot.
//!
//! Mutation test of gpu_verify's cross-workgroup read / write check: the pre-fix protocol must be flagged (GV-GLB-RW, global-rw-untracked),
//! the shipped one proven (global-rw-disjoint, its state stores global-write-disjoint).
use amdgpu_kernels::conv::{conv1d_fwd, conv1d_fwd_racy_for_tests};

fn rw(f: &amdgpu_dsl::Function) -> (bool, usize, usize, usize) {
    let r = amdgpu_codegen::gpu_verify::analyse(f).expect("analyse");
    assert!(r.errors().is_empty(), "{}: {:?}", f.name, r.errors());
    let flagged = r.findings.iter().any(|x| x.code == "GV-GLB-RW" && x.msg.contains("conv_states"));
    (
        flagged,
        r.unchecked.get("global-rw-untracked").copied().unwrap_or(0),
        r.proven.get("global-rw-disjoint").copied().unwrap_or(0),
        r.unchecked.get("global-write-untracked").copied().unwrap_or(0),
    )
}

#[test]
fn the_pre_fix_state_protocol_is_flagged_and_the_shipped_one_is_proven() {
    for sched in [false, true] {
        let name = if sched { "causal_conv1d_fwd_bf16_w4_silu_sched" } else { "causal_conv1d_fwd_bf16_w4_silu" };
        let (flag_old, bad_old, _, _) = rw(&conv1d_fwd_racy_for_tests(name, 4, false, true, sched));
        assert!(flag_old && bad_old > 0, "{name}: the racy protocol (last chunk writes the slot chunk 0 reads) must be flagged");
        let (flag_new, bad_new, ok_new, wu_new) = rw(&conv1d_fwd(name, 4, false, true, sched));
        assert!(!flag_new && bad_new == 0 && ok_new > 0, "{name}: the shipped protocol must be proven (flagged {flag_new}, untracked {bad_new}, proven {ok_new})");
        assert_eq!(wu_new, 0, "{name}: every store proven workgroup-disjoint");
    }
}
