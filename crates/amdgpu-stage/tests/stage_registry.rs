//! The stage registry: keys are unique and every kernel-side instance key is registered.

#[test]
fn registry_keys_unique_and_cover_instances() {
    let r = amdgpu_stage::stage::registry();
    let mut keys: Vec<&str> = r.iter().map(|d| d.key).collect();
    keys.sort();
    let n = keys.len();
    keys.dedup();
    assert_eq!(keys.len(), n, "duplicate stage keys");
    for k in ["move.kv_stage", "online.softmax", "mma.pv", "mma.qk", "mma.rows_x_lds", "move.ab_stage", "mma.blockscale_tile", "move.row_load", "reduce.row_sumsq", "quant.fp8_row", "move.row_store", "merged"] {
        assert!(keys.contains(&k), "{k} not registered");
    }
}