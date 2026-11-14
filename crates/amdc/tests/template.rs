//! Shape templates (`amdc::template`): the oracle of the fast path. For every value of the declared domain a template lookup either declines or returns assembly byte-identical to the full compile; values outside
//! the verified sets (and outside the domain) are never served; a request through a template yields the same `.s`, `.co` and info as the full path.

use amdc::jitc::{compile_request, load, Req};
use amdc::template::{build, lookup, write};
use amdgpu_codegen::backend::OwnTail;
use amdgpu_codegen::{compile_with_options, CompileOptions};

fn tmp(tag: &str) -> String {
    let d = std::env::temp_dir().join(format!("amdc-tpl-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().to_string()
}

fn full(kernel: &str, p: &str, v: i64) -> Option<String> {
    let (f, _) = load(kernel, &[(p.to_string(), v.to_string())], None).ok()?;
    compile_with_options(&f, &OwnTail, &CompileOptions { verify: true }).ok().map(|c| c.asm)
}

#[test]
fn a_template_serves_exactly_the_full_compile_on_every_value_of_its_domain() {
    let dir = tmp("exh");
    // (kernel, parameter, lo, hi, step, minimum share of the domain that must be served)
    for (k, p, lo, hi, st, min_share) in [("rmsnorm_quant_fp8_n4096", "m", 4, 256, 4, 0.9), ("silu_mul_bf16_d4096", "m", 1, 200, 1, 0.9), ("gemm_a8w8_rowcol_fp8_64", "m", 64, 4096, 64, 0.9), ("pa_decode16_bf16_hd128_p16_f", "H", 1, 130, 1, 0.9)] {
        let (t, n, _, _, _) = build(k, &[], p, lo, hi, st, 8);
        assert!(!t.classes.is_empty(), "{k}: no class");
        write(&dir, &t).unwrap();
        let mut served = 0;
        for i in 0..n as i64 {
            let v = lo + i * st;
            let got = lookup(&dir, k, &[(p.to_string(), v.to_string())], "own");
            if let Some(asm) = got {
                served += 1;
                assert_eq!(Some(asm), full(k, p, v), "{k} {p}={v}: the template differs from the full compile");
            }
        }
        assert!(served as f64 >= min_share * n as f64, "{k}: only {served} of {n} values are served");
        // outside the domain, with another backend or other bindings: never served
        assert!(lookup(&dir, k, &[(p.to_string(), (hi + st * 1000).to_string())], "own").is_none());
        assert!(lookup(&dir, k, &[(p.to_string(), lo.to_string())], "llvm-tail").is_none());
        assert!(lookup(&dir, k, &[(p.to_string(), lo.to_string()), ("zzz".into(), "1".into())], "own").is_none());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_request_through_a_template_produces_the_outputs_of_the_full_path() {
    let dir = tmp("req");
    let (t, ..) = build("rmsnorm_quant_fp8_n4096", &[], "m", 4, 128, 4, 4);
    write(&format!("{dir}/t"), &t).unwrap();
    let run = |tag: &str, tpl: bool, m: i64| {
        let (s, co, info) = (format!("{dir}/{tag}.s"), format!("{dir}/{tag}.co"), format!("{dir}/{tag}.json"));
        let r = compile_request(&Req {
            kernel: "rmsnorm_quant_fp8_n4096".into(),
            spec: vec![("m".into(), m.to_string())],
            backend: "own".into(),
            out_s: Some(s.clone()),
            co: Some(co.clone()),
            info: Some(info.clone()),
            template_dir: tpl.then(|| format!("{dir}/t")),
            ..Default::default()
        });
        (r.code, r.msg, std::fs::read(&s).unwrap(), std::fs::read(&co).unwrap(), std::fs::read(&info).unwrap())
    };
    for m in [4, 40, 100, 128] {
        let (a, b) = (run("a", true, m), run("b", false, m));
        assert!(a.1.contains("verified shape template") && !b.1.contains("verified shape template"), "m={m}: {} / {}", a.1, b.1);
        assert_eq!((a.0, a.2, a.3, a.4), (b.0, b.2, b.3, b.4), "m={m}");
    }
    // a value outside the domain takes the full path
    let o = run("c", true, 4000);
    assert!(o.0 == 0 && !o.1.contains("verified shape template"));
    let _ = std::fs::remove_dir_all(&dir);
}
