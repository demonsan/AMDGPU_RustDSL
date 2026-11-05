//! Region IR: the kernel body is a tree (`Node::Loop` / `If` / `Region`); the flat marker view is derived from it.
use amdgpu_dsl::tile::ev;
use amdgpu_dsl::{DType, FnBuilder, Function, Node, Opnd, Stmt};

fn shipped() -> Vec<String> {
    let mut v: Vec<String> = amdgpu_compiler::kernel_names().iter().map(|s| s.to_string()).collect();
    for k in [
        "fmha_fwd_hd128_mxfp8_v3_vs128",
        "fmha_fwd_hd128_mxfp8_v3_t128",
        "fmha_fwd_hd128_mxfp8_v3_kvsplit",
        "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h",
        "gemm_mxfp8_bs_v4",
    ] {
        v.push(k.to_string());
    }
    v
}

fn count(nodes: &[Node], c: &mut [usize; 3]) {
    for n in nodes {
        match n {
            Node::S(_) => {}
            Node::Loop { body, .. } => {
                c[0] += 1;
                count(body, c);
            }
            Node::If { then, els, .. } => {
                c[1] += 1;
                count(then, c);
                if let Some(e) = els {
                    count(e, c);
                }
            }
            Node::Region { body, .. } => {
                c[2] += 1;
                count(body, c);
            }
            Node::Stage { body, .. } => count(body, c),
        }
    }
}

#[test]
fn every_kernel_is_a_tree_and_its_flat_view_round_trips() {
    let mut total = [0usize; 3];
    for n in shipped() {
        let f = amdgpu_compiler::kernel_by_name(&n).unwrap_or_else(|| panic!("{n}"));
        // the flat view never contains a marker inside a leaf, and parsing it gives back the same tree
        assert!(f.body().iter().all(|x| !matches!(x, Node::S(Stmt::BeginLoop { .. } | Stmt::EndLoop | Stmt::BeginIf { .. } | Stmt::Else | Stmt::EndIf | Stmt::TileProduce { .. } | Stmt::TileConsume { .. } | Stmt::TileEnd))));
        let tree = Function::parse_flat(f.stmts().to_vec()).unwrap_or_else(|e| panic!("{n}: {e}"));
        let mut g = f.clone();
        g.set_body(tree);
        assert_eq!(format!("{:?}", g.stmts()), format!("{:?}", f.stmts()), "{n}: round trip");
        let mut c = [0usize; 3];
        count(f.body(), &mut c);
        for k in 0..3 {
            total[k] += c[k];
        }
    }
    println!("loops {} ifs {} regions {}", total[0], total[1], total[2]);
    assert!(total.iter().all(|x| *x > 0));
}

fn demo(closures: bool) -> Function {
    let mut b = FnBuilder::new("region_demo", 64);
    let tid = b.let_(amdgpu_dsl::Op::WorkitemId);
    let z = b.c_i32(0);
    let lim = b.c_i32(4);
    if closures {
        b.loop_(&[(DType::I32, Opnd::Val(z))], |b, vars| {
            let i = vars[0];
            let c = b.cmp_lt_i32(tid, lim);
            b.if_else(
                c,
                |b| {
                    b.produce_region("T", 0, 0, |b| {
                        let a = b.bind_i32(ev(tid) * 4);
                        b.store_lds_b32(a, z)
                    })
                },
                |b| b.barrier(),
            );
            let n = b.add_i32(Opnd::Var(i), Opnd::I32(1));
            b.set_var(i, n);
            let d = b.let_(amdgpu_dsl::Op::CmpGeI32(Opnd::Val(n), Opnd::Val(lim)));
            b.break_if(d);
        });
    } else {
        let vars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let i = vars[0];
        let c = b.cmp_lt_i32(tid, lim);
        b.begin_if(c);
        b.tile_produce("T", 0, 0);
        let a = b.bind_i32(ev(tid) * 4);
        b.store_lds_b32(a, z);
        b.tile_end();
        b.else_();
        b.barrier();
        b.end_if();
        let n = b.add_i32(Opnd::Var(i), Opnd::I32(1));
        b.set_var(i, n);
        let d = b.let_(amdgpu_dsl::Op::CmpGeI32(Opnd::Val(n), Opnd::Val(lim)));
        b.break_if(d);
        b.end_loop();
    }
    b.finish()
}

#[test]
fn closure_forms_build_the_same_tree_as_begin_end() {
    let (a, c) = (demo(false), demo(true));
    assert_eq!(format!("{:?}", a.stmts()), format!("{:?}", c.stmts()));
    assert!(matches!(c.body().iter().find(|n| matches!(n, Node::Loop { .. })), Some(Node::Loop { .. })));
}

#[test]
#[should_panic(expected = "end_loop without its opening statement")]
fn an_end_without_its_begin_panics_at_build_time() {
    let mut b = FnBuilder::new("bad", 64);
    let c = b.c_i32(0);
    b.begin_if(c);
    b.end_loop();
}

#[test]
#[should_panic(expected = "open regions at the end of the kernel")]
fn an_unclosed_region_panics_at_finish() {
    let mut b = FnBuilder::new("bad2", 64);
    let z = b.c_i32(0);
    b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    b.finish();
}

#[test]
fn an_unbalanced_flat_edit_is_rejected() {
    let f = demo(false);
    let mut flat = f.stmts().to_vec();
    let i = flat.iter().position(|s| matches!(s, Stmt::EndIf)).unwrap();
    flat.remove(i);
    assert!(Function::parse_flat(flat).is_err());
}
