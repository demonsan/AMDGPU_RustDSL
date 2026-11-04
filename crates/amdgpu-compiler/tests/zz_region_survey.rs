use amdgpu_dsl::Stmt;

#[test]
fn survey() {
    let mut names: Vec<String> = amdgpu_compiler::kernel_names().iter().map(|s| s.to_string()).collect();
    for k in ["fmha_fwd_hd128_mxfp8_v3_vs128", "fmha_fwd_hd128_mxfp8_v3_t128", "fmha_fwd_hd128_mxfp8_v3_kvsplit", "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h", "fmha_fwd_hd128_mxfp8_dsl_vs128", "fmha_fwd_hd128_mxfp8_dsl_t128", "fmha_fwd_hd128_mxfp8_dsl_kvsplit", "fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h", "gemm_mxfp8_bs_v3", "gemm_mxfp8_bs_v4"] {
        names.push(k.to_string());
    }
    let mut closers: std::collections::BTreeMap<String, usize> = Default::default();
    let mut crossing = 0;
    let mut unbalanced = 0;
    for n in &names {
        let Some(f) = amdgpu_compiler::kernel_by_name(n) else { continue };
        let mut open: Option<usize> = None; // depth at which the marker was opened
        let mut depth = 0usize;
        for st in &f.stmts {
            let close = |why: &str, open: &mut Option<usize>, closers: &mut std::collections::BTreeMap<String, usize>| {
                if open.is_some() {
                    *closers.entry(why.to_string()).or_insert(0) += 1;
                    *open = None;
                }
            };
            match st {
                Stmt::TileProduce { .. } | Stmt::TileConsume { .. } => {
                    close("next-marker", &mut open, &mut closers);
                    open = Some(depth);
                }
                Stmt::TileEnd => {
                    if open.is_none() {
                        *closers.entry("stray-TileEnd".into()).or_insert(0) += 1;
                    }
                    close("TileEnd", &mut open, &mut closers)
                }
                Stmt::Barrier => close("Barrier", &mut open, &mut closers),
                Stmt::BeginLoop { .. } | Stmt::BeginIf { .. } => {
                    if open.is_some() {
                        crossing += 1;
                        if crossing < 5 {
                            println!("CROSS {n}: marker open across a {st:?}");
                        }
                    }
                    depth += 1;
                }
                Stmt::Else => close("Else", &mut open, &mut closers),
                Stmt::EndLoop | Stmt::EndIf => {
                    if let Some(d) = open {
                        if d < depth {
                            // opened outside this construct (crossing)
                        }
                    }
                    close(if matches!(st, Stmt::EndLoop) { "EndLoop" } else { "EndIf" }, &mut open, &mut closers);
                    if depth == 0 {
                        unbalanced += 1;
                    } else {
                        depth -= 1;
                    }
                }
                _ => {}
            }
        }
        if open.is_some() {
            *closers.entry("function-end".into()).or_insert(0) += 1;
        }
    }
    println!("kernels {} closers {closers:?} crossing {crossing} unbalanced {unbalanced}", names.len());
}
