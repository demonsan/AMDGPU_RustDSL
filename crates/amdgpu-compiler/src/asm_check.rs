//! Static checks on emitted assembly for the fused-combine (last-arriver) memory contract (docs/last-arriver-memory-model.md).
//!
//! `check_lite_contract`: the sc1-only discipline that lets the hot path skip `buffer_wbl2` / `buffer_inv`:
//!  * every global load/store (not the LDS-DMA forms) carries `sc1`;
//!  * every `global_atomic_*` has `sc0` (return) and NOT `sc1` (device scope);
//!  * no `buffer_wbl2` / `buffer_inv`;
//!  * a global store must be drained (`s_waitcnt vmcnt(0)`) before the next `s_barrier` and before the next atomic (s_barrier does not drain VM_CNT, so
//!    each wave drains its own stores before the block barrier / the ticket).
//! The drain rule is checked on the linear instruction order (control flow is not followed: a drain on any path before the use counts), which is
//! exact for the straight-line release sequence these kernels use.

pub fn check_lite_contract(asm: &str) -> Result<(), Vec<String>> {
    check_lite_contract_ex(asm, false)
}

/// Same, with an escape for kernels whose FINAL output stores (read only after the kernel ends, never by another workgroup during it) are plain:
/// `allow_plain_stores = true` accepts global stores without `sc1` (loads and atomics stay strict). A plain store still counts for the drain rule.
pub fn check_lite_contract_ex(asm: &str, allow_plain_stores: bool) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();
    let mut undrained: Option<usize> = None;
    for (n, raw) in asm.lines().enumerate() {
        let l = raw.trim();
        let mnemonic = l.split_whitespace().next().unwrap_or("");
        let ln = n + 1;
        if mnemonic.starts_with("buffer_wbl2") || mnemonic.starts_with("buffer_inv") {
            errs.push(format!("line {ln}: `{l}`: cache maintenance in the lite contract (use sc1 accesses instead)"));
        }
        let is_lds_dma = mnemonic.contains("_lds");
        if (mnemonic.starts_with("global_load") || mnemonic.starts_with("global_store")) && !is_lds_dma {
            if !l.split_whitespace().any(|t| t == "sc1") && !(allow_plain_stores && mnemonic.starts_with("global_store")) {
                errs.push(format!("line {ln}: `{l}`: shared-memory access without sc1"));
            }
            if mnemonic.starts_with("global_store") {
                undrained = Some(ln);
            }
        }
        if mnemonic.starts_with("global_atomic") {
            let toks: Vec<&str> = l.split_whitespace().collect();
            if !toks.contains(&"sc0") || toks.contains(&"sc1") {
                errs.push(format!("line {ln}: `{l}`: the ticket must be `sc0` (return) device scope, without sc1"));
            }
            if let Some(s) = undrained {
                errs.push(format!("line {ln}: atomic ticket while the store of line {s} is not drained (s_waitcnt vmcnt(0))"));
            }
        }
        if mnemonic == "s_waitcnt" && l.contains("vmcnt(0)") {
            undrained = None;
        }
        if mnemonic == "s_barrier" {
            if let Some(s) = undrained {
                errs.push(format!("line {ln}: s_barrier while the store of line {s} is not drained: s_barrier does not drain VM_CNT, each wave must `s_waitcnt vmcnt(0)` first"));
                undrained = None;
            }
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}