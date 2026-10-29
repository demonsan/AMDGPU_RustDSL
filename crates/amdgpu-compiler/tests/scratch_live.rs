use amdgpu_compiler::kernel_by_name;
#[test]
fn dump_vals() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs").unwrap();
    let d = f.dump();
    let lines: Vec<&str> = d.lines().collect();
    for (i, l) in lines.iter().enumerate() {
        if l.contains("Val(246)") || l.contains(" v246:") || l.contains("Val(235)") || l.contains(" v235:") || l.contains(" v192:") {
            println!("DUMP {i}: {}", l.trim());
        }
    }
}