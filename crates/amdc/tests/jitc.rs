//! The JIT compile request (`amdc::jitc`): the content-addressed certificate cache, `amdc serve`'s line protocol, and that a cached run produces exactly the outputs of an uncached one.

use amdc::jitc::{cert_key, certify_ir_cached, compile_request, load, parse_line, response_line, Req, Resp};

fn tmp(tag: &str) -> String {
    let d = std::env::temp_dir().join(format!("amdc-jitc-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().to_string()
}

fn spec(s: &str) -> Vec<(String, String)> {
    amdgpu_dsl::specialize::parse_bindings(s).unwrap()
}

#[test]
fn the_content_address_is_deterministic_and_follows_every_field() {
    // two independent builds of the same specialized kernel: the same key (no hash-order noise in the debug text); another shape or another kernel: another key
    for (k, s) in [("silu_mul_bf16_d4096", "m=64"), ("rmsnorm_quant_fp8_n4096", "m=64"), ("gemm_a8w8_rowcol_fp8_64", "m=64"), ("pa_decode16_bf16_hd128_p16_f", "H=64"), ("causal_conv1d_update_bf16_w4_s3_idx_silu", "slots=4")] {
        let a = cert_key(&load(k, &spec(s), None).unwrap().0);
        let b = cert_key(&load(k, &spec(s), None).unwrap().0);
        assert_eq!(a, b, "{k} {s}");
    }
    let m64 = cert_key(&load("silu_mul_bf16_d4096", &spec("m=64"), None).unwrap().0);
    let m128 = cert_key(&load("silu_mul_bf16_d4096", &spec("m=128"), None).unwrap().0);
    let other = cert_key(&load("rmsnorm_quant_fp8_n4096", &spec("m=64"), None).unwrap().0);
    let generic = cert_key(&load("silu_mul_bf16_d4096", &[], None).unwrap().0);
    assert!(m64 != m128 && m64 != other && m64 != generic);
}

#[test]
fn a_cached_certificate_is_reused_only_for_the_identical_ir() {
    let d = tmp("cache");
    let f64_ = load("silu_mul_bf16_d4096", &spec("m=64"), None).unwrap().0;
    let f128 = load("silu_mul_bf16_d4096", &spec("m=128"), None).unwrap().0;
    assert_eq!(certify_ir_cached(&f64_, &d), Ok(true), "first: computed and stored");
    assert_eq!(certify_ir_cached(&f64_, &d), Ok(false), "second: from the cache");
    assert_eq!(certify_ir_cached(&f128, &d), Ok(true), "another shape is another IR: not served from the cache");
    // a damaged entry is not trusted: it is recomputed (and rewritten)
    let entry = std::path::Path::new(&d).join(format!("{}.cert", cert_key(&f64_)));
    std::fs::write(&entry, "garbage").unwrap();
    assert_eq!(certify_ir_cached(&f64_, &d), Ok(true));
    std::fs::write(&entry, "FAILED 1 2 3 4 5\n").unwrap();
    assert_eq!(certify_ir_cached(&f64_, &d), Ok(true), "only `ok` entries count");
    let _ = std::fs::remove_dir_all(&d);
}

fn run(dir: &str, cache: Option<&str>, tag: &str) -> (Resp, Vec<u8>, Vec<u8>, String) {
    let (s, co, info) = (format!("{dir}/{tag}.s"), format!("{dir}/{tag}.co"), format!("{dir}/{tag}.json"));
    let r = compile_request(&Req { kernel: "rmsnorm_quant_fp8_n4096".into(), spec: spec("m=64"), backend: "own".into(), out_s: Some(s.clone()), co: Some(co.clone()), info: Some(info.clone()), cert_cache: cache.map(|c| c.to_string()), ..Default::default() });
    (r, std::fs::read(&co).unwrap_or_default(), std::fs::read(&s).unwrap_or_default(), std::fs::read_to_string(&info).unwrap_or_default())
}

#[test]
fn a_cached_run_produces_exactly_the_outputs_of_an_uncached_one() {
    let d = tmp("run");
    let cache = format!("{d}/certs");
    let (r0, co0, s0, i0) = run(&d, None, "nocache");
    let (r1, co1, s1, i1) = run(&d, Some(&cache), "first");
    let (r2, co2, s2, i2) = run(&d, Some(&cache), "second");
    assert_eq!((r0.code, r1.code, r2.code), (0, 0, 0), "{} {} {}", r0.msg, r1.msg, r2.msg);
    assert!(!r1.msg.contains("from the cache") && r2.msg.contains("from the cache"), "{} / {}", r1.msg, r2.msg);
    assert!(r0.msg.contains("0 unproven") && r2.msg.contains("0 unproven"));
    assert!(!co0.is_empty() && co0 == co1 && co1 == co2 && s0 == s1 && s1 == s2 && i0 == i1 && i1 == i2);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_server_protocol_round_trips_and_reports_errors_per_request() {
    let line = "7\tsilu_mul_bf16_d4096\tm=64\town\t/tmp/a.s\t-\t-\t-\t-\t-";
    let (id, req) = parse_line(line).unwrap();
    assert_eq!((id.as_str(), req.kernel.as_str(), req.spec.len(), req.out_s.as_deref(), req.co), ("7", "silu_mul_bf16_d4096", 1, Some("/tmp/a.s"), None));
    assert!(parse_line("1\tonly\tthree").is_err());
    let bad = compile_request(&Req { kernel: "no_such_kernel".into(), ..Default::default() });
    assert_eq!(bad.code, 2);
    let l = response_line("9", &Resp { code: 0, msg: "a\nb\tc\\d".into() });
    assert_eq!(l, "9\t0\ta\\nb\\tc\\\\d\n");
}
