//! Executable host-truth check for the GPU-semantics contracts.
//!
//! Every `arg_tensor` / `arg_tensor_slack` extent, `grid_is`, `ka_is` and `ka_range` in the shipped
//! FMHA kernels is evaluated for concrete shapes and compared with the allocations/formulas in
//! `tests/gpu/ref_host.cpp` (the only host launcher used for these kernels). This turns the
//! documented contract into a checked one: a declaration that overstates the real allocation fails
//! here before it can produce a false "proven safe" in gpu_verify.

use amdgpu_dsl::{Ext, Function};
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Shape {
    b: i64,
    s: i64,
    h: i64,
    d: i64,
    sk: i64,
    hk: i64,
    splits: i64,
    rows_total: i64,
}

/// The 656-byte ABI / combine ABI values written by `tests/gpu/ref_host.cpp`.
fn ka(offset: u32, sh: &Shape) -> i64 {
    let hk = sh.hk;
    let d = sh.d;
    match offset {
        0x060 => sh.s,
        0x070 => sh.h * d,
        0x090 => d,
        0x0A0 => sh.s * sh.h * d,
        0x0B0 => sh.h / hk,
        0x0C0 => hk * d,
        0x0D0 => d,
        0x0E0 => sh.sk * hk * d,
        0x110 => sh.sk,
        0x140 => sh.h,
        0x150 => hk * d,
        0x160 => d,
        0x170 => sh.sk * hk * d,
        0x180 => sh.h * d * 2,
        0x190 => d * 2,
        0x1A0 => sh.s * sh.h * d * 2,
        0x230 => sh.s * sh.h * 4,
        0x240 => 4,
        0x250 => sh.sk * hk * 4,
        0x260 => 4,
        // combine ABI
        0x18 => sh.rows_total,
        0x1C => d,
        0x20 => sh.splits,
        0x280 => sh.splits,
        0x284 => sh.rows_total,
        other => panic!("ka({other:#x}) not in the host ABI"),
    }
}

fn ev(e: &Ext, sh: &Shape, grid: &[i64; 3], syms: &HashMap<&str, i64>) -> i64 {
    match e {
        Ext::Const(c) => *c,
        Ext::Ka(o) => ka(*o, sh),
        Ext::Grid(a) => grid[*a as usize],
        Ext::Sym(n) => *syms.get(n.as_str()).unwrap_or_else(|| panic!("unknown symbol {n}")),
        Ext::Add(a, b) => ev(a, sh, grid, syms) + ev(b, sh, grid, syms),
        Ext::Sub(a, b) => ev(a, sh, grid, syms) - ev(b, sh, grid, syms),
        Ext::Mul(a, b) => ev(a, sh, grid, syms) * ev(b, sh, grid, syms),
        Ext::CeilDiv(a, k) => (ev(a, sh, grid, syms) + k - 1).div_euclid(*k),
        Ext::FloorDiv(a, k) => ev(a, sh, grid, syms).div_euclid(*k),
    }
}

fn check_extents(f: &Function, sh: &Shape, grid: [i64; 3], syms: &HashMap<&str, i64>, host: &[(&str, i64, Option<i64>)]) -> Result<(), String> {
    let mut errors = Vec::new();
    for (name, want_alloc, want_logical) in host {
        let Some(ext) = f.contract.extents.iter().find(|e| e.arg == *name) else {
            errors.push(format!("{}: no contract extent for `{name}` (coverage gap)", f.name));
            continue;
        };
        let alloc = ev(&ext.alloc, sh, &grid, syms);
        if alloc > *want_alloc {
            errors.push(format!("{}: `{name}` declares alloc {alloc} > host {want_alloc}", f.name));
        }
        if let (Some(logical), Some(want_log)) = (&ext.logical, want_logical) {
            let l = ev(logical, sh, &grid, syms);
            if l > *want_log {
                errors.push(format!("{}: `{name}` declares logical {l} > host logical {want_log}", f.name));
            }
            if l > alloc {
                errors.push(format!("{}: `{name}` logical {l} > declared alloc {alloc}", f.name));
            }
        }
    }
    for (axis, e) in &f.contract.grid_defs {
        let got = ev(e, sh, &grid, syms);
        if got != grid[*axis as usize] {
            errors.push(format!("{}: grid_is({axis}) = {got} != host launch {}", f.name, grid[*axis as usize]));
        }
    }
    for (off, e) in &f.contract.ka_defs {
        let got = ev(e, sh, &grid, syms);
        let want = ka(*off, sh);
        if got != want {
            errors.push(format!("{}: ka_is({off:#x}) = {got} != host {want}", f.name));
        }
    }
    for (off, lo, hi) in &f.contract.ka_ranges {
        let v = ka(*off, sh);
        if v < *lo || (*hi != i64::MAX && v > *hi) {
            errors.push(format!("{}: ka({off:#x})={v} outside contract [{lo},{hi}]", f.name));
        }
    }
    for (name, lo, hi) in &f.contract.sym_ranges {
        let v = *syms.get(name.as_str()).unwrap_or(&0);
        if v < *lo || (*hi != i64::MAX && v > *hi) {
            errors.push(format!("{}: symbol {name}={v} outside contract [{lo},{hi}]", f.name));
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors.join("\n")) }
}

#[test]
fn fmha_contracts_match_the_host_allocations() {
    let shapes = [
        Shape { b: 1, s: 2048, h: 8, d: 128, sk: 2048, hk: 8, splits: 4, rows_total: 1 * 2048 * 8 },
        Shape { b: 4, s: 2048, h: 8, d: 128, sk: 2048, hk: 8, splits: 4, rows_total: 4 * 2048 * 8 },
        Shape { b: 1, s: 200, h: 2, d: 128, sk: 1337, hk: 1, splits: 4, rows_total: 1 * 200 * 2 },
    ];
    let mut all_errors = Vec::new();
    for sh in &shapes {
        let h = |b: i64| b;
        // non-split kernels: Q/K/V/O, Q/K scales (logical + zeroed slack), V scale
        let nonsplit = [
            ("ptr_q", h(sh.b) * sh.s * sh.h * sh.d, None),
            ("ptr_k", h(sh.b) * sh.sk * sh.hk * sh.d, None),
            ("ptr_v", h(sh.b) * sh.sk * sh.hk * sh.d, None),
            ("ptr_o", h(sh.b) * sh.s * sh.h * sh.d * 2, None),
            ("ptr_q_descale", h(sh.b) * sh.s * sh.h * 4 + 512 * sh.h * 4, Some(h(sh.b) * sh.s * sh.h * 4)),
            ("ptr_k_descale", h(sh.b) * sh.sk * sh.hk * 4 + 640 * sh.h * 4, Some(h(sh.b) * sh.sk * sh.hk * 4)),
            ("ptr_v_descale", 4096, None),
        ];
        for k in ["fmha_fwd_hd128_mxfp8_vs128", "fmha_fwd_hd128_mxfp8_t128", "fmha_fwd_hd128_mxfp8_t", "fmha_fwd_hd128_mxfp8_dsl_vs128", "fmha_fwd_hd128_mxfp8_dsl_t128", "fmha_fwd_hd128_mxfp8_v3_vs128", "fmha_fwd_hd128_mxfp8_v3_t128"] {
            let f = amdgpu_compiler::kernel_by_name(k).expect(k);
            // ref_host: rows_pb = 128 for the _t128 family, 256 otherwise (the contract's grid_is(0)
            // uses the same rows_per_block the builder was called with)
            let rows_pb = if k.contains("t128") { 128 } else { 256 };
            let grid = [((sh.s + rows_pb - 1) / rows_pb), sh.h, sh.b];
            let syms = HashMap::from([("B", sh.b), ("HK", sh.hk)]);
            if let Err(e) = check_extents(&f, sh, grid, &syms, &nonsplit) { all_errors.push(e); }
        }
        // split kernel: grid z = B * splits; Ohat/lse extents use splits * rows_total
        let f = amdgpu_compiler::kernel_by_name("fmha_fwd_hd128_mxfp8_kvsplit").expect("kvsplit");
        let grid = [((sh.s + 255) / 256), sh.h, sh.b * sh.splits];
        let syms = HashMap::from([("B", sh.b), ("HK", sh.hk)]);
        let split = [
            ("ptr_q", sh.b * sh.s * sh.h * sh.d, None),
            ("ptr_k", sh.b * sh.sk * sh.hk * sh.d, None),
            ("ptr_v", sh.b * sh.sk * sh.hk * sh.d, None),
            ("ptr_o", sh.b * sh.s * sh.h * sh.d * 2, None),
            ("ptr_q_descale", sh.b * sh.s * sh.h * 4 + 512 * sh.h * 4, Some(sh.b * sh.s * sh.h * 4)),
            ("ptr_k_descale", sh.b * sh.sk * sh.hk * 4 + 640 * sh.h * 4, Some(sh.b * sh.sk * sh.hk * 4)),
            ("ptr_v_descale", 4096, None),
            ("ptr_ohat", sh.splits * sh.rows_total * 128 * 4, None),
            ("ptr_lse", sh.splits * sh.rows_total * 4, None),
        ];
        if let Err(e) = check_extents(&f, sh, grid, &syms, &split) { all_errors.push(e); }
        // combine kernel: rows_total * 128 * 4 Ohat, rows_total * 4 lse, rows_total * 128 * 2 out
        let f = amdgpu_compiler::kernel_by_name("fmha_fwd_hd128_mxfp8_kvsplit_combine8h").expect("combine8h");
        let combine = [
            ("ptr_ohat", sh.splits * sh.rows_total * 128 * 4, None),
            ("ptr_lse", sh.splits * sh.rows_total * 4, None),
            ("ptr_out", sh.rows_total * 128 * 2, None),
        ];
        if let Err(e) = check_extents(&f, sh, [1, 1, 1], &HashMap::new(), &combine) { all_errors.push(e); }
    }
    assert!(all_errors.is_empty(), "host-contract mismatches:\n{}", all_errors.join("\n"));
}

#[test]
fn fmha_contracts_are_present() {
    // a contract-free shipped kernel must not pass silently: this test is itself the coverage guard
    for k in ["fmha_fwd_hd128_mxfp8_vs128", "fmha_fwd_hd128_mxfp8_t128", "fmha_fwd_hd128_mxfp8_kvsplit", "fmha_fwd_hd128_mxfp8_kvsplit_combine8h", "fmha_fwd_hd128_mxfp8_dsl_vs128", "fmha_fwd_hd128_mxfp8_dsl_t128", "fmha_fwd_hd128_mxfp8_dsl_kvsplit", "fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h", "fmha_fwd_hd128_mxfp8_v3_vs128", "fmha_fwd_hd128_mxfp8_v3_t128", "fmha_fwd_hd128_mxfp8_v3_kvsplit", "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"] {
        let f = amdgpu_compiler::kernel_by_name(k).expect(k);
        assert!(!f.contract.extents.is_empty(), "{k}: no contract extents (annotations missing)");
    }
}

#[test]
fn generated_host_abi_matches_the_hand_offsets_and_sizes() {
    // (kernel, [(field, offset, size)]): the hand-written 656-byte / 720-byte ABI facts that
    // ref_host.cpp used before gen-host. The generated Args/static_asserts must reproduce them.
    let hand: &[(&str, &[(&str, u32, u32)])] = &[
        ("fmha_fwd_hd128_mxfp8_v3_vs128", &[
            ("ptr_o", 0x000, 8), ("ptr_q", 0x010, 8), ("ptr_k", 0x020, 8), ("ptr_v", 0x030, 8),
            ("s_seq_len", 0x060, 4), ("s_Seqs", 0x070, 4), ("s_Hs", 0x090, 4), ("s_Bs", 0x0A0, 4),
            ("s_gqa", 0x0B0, 4), ("s_k_Seqs", 0x0C0, 4), ("s_k_Hs", 0x0D0, 4), ("s_k_Bs", 0x0E0, 4),
            ("s_kv_seq_len", 0x110, 4), ("s_q_head_num", 0x140, 4),
            ("ptr_q_descale", 0x200, 8), ("ptr_k_descale", 0x210, 8), ("ptr_v_descale", 0x220, 8),
        ]),
        ("fmha_fwd_hd128_mxfp8_v3_t128", &[
            ("ptr_o", 0x000, 8), ("ptr_q", 0x010, 8), ("ptr_k", 0x020, 8), ("ptr_v", 0x030, 8),
            ("s_seq_len", 0x060, 4), ("s_Seqs", 0x070, 4), ("ptr_q_descale", 0x200, 8),
        ]),
        ("fmha_fwd_hd128_mxfp8_v3_kvsplit", &[
            ("ptr_o", 0x000, 8), ("ptr_q", 0x010, 8), ("ptr_k", 0x020, 8), ("ptr_v", 0x030, 8),
            ("s_seq_len", 0x060, 4), ("ptr_q_descale", 0x200, 8),
            ("kvs_splits", 0x280, 4), ("kvs_rows_total", 0x284, 4),
        ]),
        ("fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h", &[
            ("ptr_ohat", 0x00, 8), ("ptr_lse", 0x08, 8), ("ptr_out", 0x10, 8),
            ("rows_total", 0x18, 4), ("d", 0x1C, 4), ("splits", 0x20, 4),
        ]),
    ];
    for (name, fields) in hand {
        let f = amdgpu_compiler::kernel_by_name(name).expect(name);
        let g = amdgpu_compiler::hostgen::generate(&f).expect("generate");
        assert!(g.header.contains(&format!("static_assert(sizeof(Args) == {}, \"kernarg size\");", f.kernarg_size)), "{name}: generated sizeof mismatch");
        for (field, off, sz) in *fields {
            let a = f.args.iter().find(|a| a.name == *field).unwrap_or_else(|| panic!("{name}: no hand arg {field}"));
            assert_eq!(a.kernarg_offset, *off, "{name}/{field}: hand offset");
            assert_eq!(a.size, *sz, "{name}/{field}: hand size");
            assert!(g.header.contains(&format!("static_assert(offsetof(Args, {field}) == 0x{off:X}")), "{name}/{field}: generated offset assert missing");
            assert!(g.json.contains(&format!("\"name\":\"{field}\",\"offset\":{off},\"size\":{sz}")), "{name}/{field}: generated JSON mismatch");
        }
        assert!(g.json.contains(&format!("\"kernel\": \"{name}\"")), "{name}: JSON kernel");
        assert!(g.json.contains(&format!("\"kernarg_size\": {}", f.kernarg_size)), "{name}: JSON kernarg_size");
    }
}

