//! Shape specialization and JIT metadata (docs/jit.md).
use amdgpu_dsl::specialize::{check, close, parse_bindings, specialize};
use amdgpu_dsl::Ext;

fn b(s: &str) -> Vec<(String, String)> {
    parse_bindings(s).unwrap()
}

#[test]
fn bindings_are_validated_against_the_arguments_and_the_contract() {
    let g = amdgpu_kernels::kernel_by_name("gemm_mxfp8_bs_v4").unwrap();
    assert!(check(&g, &b("nope=1")).unwrap_err().contains("no argument"));
    assert!(check(&g, &b("ptr_a=1")).unwrap_err().contains("pointer"));
    assert!(check(&g, &b("m=x")).unwrap_err().contains("not an integer"));
    assert!(parse_bindings("m=1,m=2").unwrap_err().contains("twice"));
    // divisibility from `m == MT * 256`, range from `MT <= 64`
    let v = check(&g, &b("m=4000")).unwrap();
    assert!(close(&g, &v).unwrap_err().contains("multiple of 256"));
    let v = check(&g, &b("m=32768")).unwrap();
    assert!(close(&g, &v).unwrap_err().contains("outside the contract range"));
    let f = amdgpu_kernels::kernel_by_name("fmha_fwd_hd128_mxfp8_v3_vs128").unwrap();
    assert!(check(&f, &b("s_gqa=32")).unwrap_err().contains("contract range"));
}

#[test]
fn bindings_close_over_the_contract_relations() {
    let g = amdgpu_kernels::kernel_by_name("gemm_mxfp8_bs_v4").unwrap();
    let v = check(&g, &b("m=4096,n=4096,k=8192")).unwrap();
    let c = close(&g, &v).unwrap();
    assert_eq!(c.syms.get("MT"), Some(&16));
    assert_eq!(c.syms.get("KT"), Some(&64));
    // stride_a (0x28) = KT * 128 is derived
    assert!(c.derived.contains(&(40, 8192)));
    let s = specialize(&g, &b("m=4096,n=4096,k=8192")).unwrap();
    assert!(s.contract.grid_defs.iter().all(|(_, e)| matches!(e, Ext::Const(16))));
    assert_eq!(s.name, g.name);
    assert_eq!(s.kernarg_size, g.kernarg_size);
}

#[test]
fn specialized_kernels_verify_strictly_and_shrink() {
    for (k, spec) in [
        ("gemm_mxfp8_bs_v4", "m=4096,n=4096,k=8192"),
        ("fmha_fwd_hd128_mxfp8_v3_vs128", "s_seq_len=2048,s_kv_seq_len=2048,s_q_head_num=8,s_gqa=1"),
        ("fmha_fwd_hd128_mxfp8_v3_t128", "s_seq_len=2048,s_kv_seq_len=2048,s_q_head_num=8,s_gqa=1"),
    ] {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let s = specialize(&f, &b(spec)).unwrap();
        let r = amdgpu_codegen::gpu_verify::analyse(&s).unwrap();
        assert!(r.errors().is_empty(), "{k}: {:?}", r.errors());
        assert!(s.stmts().len() <= f.stmts().len(), "{k}");
    }
}

#[test]
fn no_bindings_is_the_identity() {
    let f = amdgpu_kernels::kernel_by_name("fmha_fwd_hd128_mxfp8_v3_t128").unwrap();
    let s = specialize(&f, &[]).unwrap();
    assert!(amdgpu_dsl::print::dsl_structural_eq(&f, &s));
}

#[test]
fn jit_info_json_shape() {
    let f = amdgpu_kernels::kernel_by_name("gemm_mxfp8_bs_v4").unwrap();
    let binds = b("m=4096");
    let vals = check(&f, &binds).unwrap();
    let derived = close(&f, &vals).unwrap().derived;
    let s = specialize(&f, &binds).unwrap();
    let res = amdc::jit::resources(".amdhsa_group_segment_fixed_size 1024\n.amdhsa_next_free_vgpr 424\n.amdhsa_accum_offset 200\n.amdhsa_next_free_sgpr 48\n");
    assert_eq!((res.vgpr, res.agpr, res.lds_bytes, res.sgpr), (200, 224, 1024, 48));
    let j = amdc::jit::info_json(&s, &vals, &derived, "own", res);
    for key in ["\"schema\":1", "\"specialize\":{\"m\":4096}", "\"kernarg_size\":96", "\"name\":\"m\",\"offset\":24,\"size\":4,\"kind\":\"i32\",\"specializable\":true,\"specialized\":4096", "\"kind\":\"ptr\",\"specializable\":false", "\"grid\":[", "\"syms\":["] {
        assert!(j.contains(key), "missing {key} in {j}");
    }
    // every kernarg byte is covered exactly once (args + pads)
    let mut covered = 0u32;
    for part in j.split("\"offset\":").skip(1) {
        if let Some(rest) = part.split_once(",\"size\":") {
            if let (Ok(off), Some(sz)) = (rest.0.parse::<u32>(), rest.1.split(|c: char| !c.is_ascii_digit()).next().and_then(|x| x.parse::<u32>().ok())) {
                assert_eq!(off, covered, "{j}");
                covered += sz;
            }
        }
    }
    assert_eq!(covered, f.kernarg_size);
}
