//! `scalarize2`: wave-uniform address chains with clamps (compare + select), minimum and v_readfirstlane results stay in SGPRs - the GEMM's per-tile
//! DMA soffsets (`key0 = (it + 1 < nkv ? it + 1 : nkv - 1) << 7`, `min`, a `read_first_lane` feeding further adds) become SALU, no VALU + readfirstlane.
use amdgpu_dsl::ev;
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

fn demo() -> Function {
    let mut b = FnBuilder::new("probe_scalarize2", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg("nkv", DType::I32, 8, 4);
    b.arg("kk", DType::I32, 12, 4);
    b.arg_tensor("p", amdgpu_dsl::Ext::c(1 << 20));
    b.ka_range(8, 1, 64);
    b.ka_range(12, 0, 1024);
    b.lds(4096);
    b.lds_tile("T", 0, 4096);
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let nkv = b.let_(Op::KernargI32 { offset: 8 });
    let kk = b.let_(Op::KernargI32 { offset: 12 });
    let rsrc = b.buffer_rsrc_sized(p, 1 << 20);
    let tid = b.let_(Op::WorkitemId);
    let voff = b.bind_i32((ev(tid) & 63) << 4);
    let c0 = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0))]);
    let it = vars[0];
    // key0 = clamp(it + 1, nkv - 1) << 7 ; org = min(key0 * kk, 65536) ; soff = read_first_lane(org) + 256
    let nx = b.i32var(it) + 1;
    let last = ev(nkv) - 1;
    let key0 = b.bind_i32(nx.lt(ev(nkv)).select_i32(nx.clone(), last) << 7);
    let org = b.bind_i32((ev(key0) * ev(kk)).min(65536));
    let rf = b.read_first_lane(org);
    let soff_v = b.bind_i32(ev(rf) + 256);
    let soff = b.read_first_lane(soff_v); // the MUBUF soffset must be an SGPR: free when the chain is already scalar
    let m0 = b.c_i32(0);
    b.tile_produce("T", 0, Opnd::Var(it));
    b.buffer_load_lds_b128(rsrc, voff, soff, m0, 0);
    b.tile_end();
    b.wait_vm(0);
    let n1 = b.bind_i32(nx);
    b.set_var(it, n1);
    let more = b.cmp_lt_i32(n1, nkv);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(stop);
    b.end_loop();
    b.finish()
}

fn asm(opts: &[&str]) -> String {
    amdgpu_compiler::lower::with_opts(opts, || amdgpu_compiler::compile(&demo())).unwrap_or_else(|e| panic!("compile with {opts:?}: {e}\n{}", demo().dump())).asm
}

fn count(s: &str, pat: &str) -> usize {
    s.lines().filter(|l| l.trim_start().starts_with(pat)).count()
}

#[test]
fn uniform_clamps_selects_and_readfirstlane_stay_scalar() {
    let base = asm(&["dce", "cmpfold", "scalarize"]);
    let ext = asm(&["dce", "cmpfold", "scalarize", "scalarize2"]);
    assert!(count(&base, "v_readfirstlane") >= 1, "without scalarize2 the clamp chain is VALU + readfirstlane:\n{base}");
    assert_eq!(count(&ext, "v_readfirstlane"), 0, "scalarize2: no readfirstlane:\n{ext}");
    assert!(count(&ext, "s_cselect_b32") >= 1 && count(&ext, "s_min_i32") >= 1, "{ext}");
    assert!(count(&ext, "v_cndmask") == 0 && count(&ext, "v_min_i32") == 0, "{ext}");
    // every s_cselect directly follows its s_cmp (nothing that writes SCC in between)
    let lines: Vec<&str> = ext.lines().map(|l| l.trim()).filter(|l| !l.is_empty() && !l.starts_with(';') && !l.starts_with("//")).collect();
    for (i, l) in lines.iter().enumerate() {
        if l.starts_with("s_cselect_b32") {
            let prev = lines[..i].iter().rev().find(|p| p.starts_with("s_")).expect("previous scalar op");
            assert!(prev.starts_with("s_cmp_") || prev.starts_with("s_mov_b32") || prev.starts_with("s_mul_i32"), "s_cselect after {prev}");
        }
    }
}

#[test]
fn uniform_loop_exit_branches_on_scc() {
    // the demo's loop exit `!(it + 1 < nkv)` compares a scalar loop variable with a kernarg: s_cmp + s_cbranch_scc*, no VALU compare / vcc
    let ext = asm(&["dce", "cmpfold", "scalarize", "scalarize2"]);
    assert!(ext.lines().any(|l| l.trim_start().starts_with("s_cbranch_scc")), "{ext}");
    assert_eq!(count(&ext, "v_cmp_"), 0, "no VALU compare left:\n{ext}");
    // without scalarize2 the old vcc form stays
    let base = asm(&["dce", "cmpfold", "scalarize"]);
    assert!(base.lines().any(|l| l.trim_start().starts_with("s_cbranch_vcc")), "{base}");
}
