//! The golden asm manifest (tests/golden/md5_manifest.tsv) must match the current compiler: the regression oracle of the four-layer refactor.
//! An intended code change regenerates it (see tests/golden/README.md) in the same commit that names its gate.

#[test]
fn the_asm_of_every_reachable_kernel_matches_the_golden_manifest() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/md5_manifest.tsv");
    let golden = amdc::manifest::parse(&std::fs::read_to_string(&path).expect("tests/golden/md5_manifest.tsv"));
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let now = amdc::manifest::compute(jobs).expect("scrubbed environment");
    let d = amdc::manifest::diff(&golden, &now);
    assert!(
        d.is_empty(),
        "{} of {} kernels differ from the golden manifest (regenerate only for an intended, gated change: tests/golden/README.md):\n{}",
        d.len(),
        now.len(),
        d.join("\n")
    );
}
