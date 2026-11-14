//! The back-end-added instruction table (tests/golden/added_table.tsv) must match the current compiler: every change of the default path's
//! copies / waits / s_nops / constant materializations shows here per kernel. Regenerate with the md5 manifest (tests/golden/README.md).

#[test]
fn the_back_end_added_instructions_match_the_golden_table() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/added_table.tsv");
    let golden = amdc::added::parse(&std::fs::read_to_string(&path).expect("tests/golden/added_table.tsv"));
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let now = amdc::added::compute(jobs).expect("scrubbed environment");
    let (d, tot) = amdc::added::diff(&golden, &now);
    assert!(d.is_empty(), "{} of {} kernels differ from the golden added-instruction table (totals: {tot}):\n{}", d.len(), now.len(), d.join("\n"));
}
