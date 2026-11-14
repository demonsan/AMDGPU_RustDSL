//! The IR-level certificate (`verify_ir`) of every manifest kernel and of a set of specialized shapes against the golden `tests/golden/cert_md5_manifest.tsv`: findings, trust ledger and proven counts
//! must not change under a performance change of the analysis (regenerate with `amdc md5-manifest --cert --out ...` for an intended change).

#[test]
fn the_certificates_match_the_golden_cert_manifest() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/cert_md5_manifest.tsv");
    let golden = amdc::manifest::parse(&std::fs::read_to_string(&path).expect("tests/golden/cert_md5_manifest.tsv"));
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let now = amdc::manifest::compute_cert(jobs).expect("scrubbed environment");
    let d = amdc::manifest::diff(&golden, &now);
    assert!(d.is_empty(), "{} of {} certificates differ from the golden cert manifest:\n{}", d.len(), now.len(), d.join("\n"));
}
