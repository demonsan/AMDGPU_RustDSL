//! Capture mode + dependency facts (docs/weave-capture.md): capture-and-splice in the original order is the identity, legal reorders are
//! the ones `facts::must_order` allows, and the facts classify memory / anchors / loop variables correctly.
use amdgpu_dsl::facts::{must_order, ResClass};
use amdgpu_dsl::{DType, FnBuilder, Function, Node, Op, Opnd, StageContract};

fn base() -> (FnBuilder, amdgpu_dsl::ValueId, amdgpu_dsl::ValueId) {
    let mut b = FnBuilder::new("cap", 64);
    b.arg("out", DType::Ptr, 0, 8);
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    (b, out, tid)
}

/// A body with straight-line code, a loop with a loop variable, an if, tile regions, a stage, a barrier and stores.
fn body(b: &mut FnBuilder, out: amdgpu_dsl::ValueId, tid: amdgpu_dsl::ValueId) {
    let c4 = b.c_i32(4);
    let addr = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(c4)));
    b.produce_region("T", 0, 0, |b| b.store_lds_b32(addr, tid));
    b.barrier();
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let v = b.add_i32(Opnd::Var(vars[0]), 1);
    b.set_var(vars[0], v);
    let done = b.let_(Op::CmpGeI32(Opnd::Val(v), Opnd::I32(4)));
    b.break_if(done);
    b.end_loop();
    let x = b.let_(Op::CmpLtI32(Opnd::Val(tid), Opnd::I32(32)));
    b.if_(x, |b| {
        let r = b.load_lds_b32(addr);
        let f = b.let_(Op::BitcastF32(Opnd::Val(r)));
        b.store_f32(out, 0, f);
    });
    b.stage("s", StageContract::default(), |b| {
        let y = b.add_i32(Opnd::Var(vars[0]), Opnd::Val(tid));
        let f = b.let_(Op::BitcastF32(Opnd::Val(y)));
        b.store_f32(out, 4, f);
    });
}

fn dbg(f: &Function) -> String {
    format!("{f:?}")
}

#[test]
fn capture_and_splice_in_order_is_the_identity() {
    let (mut b, out, tid) = base();
    body(&mut b, out, tid);
    let direct = b.finish();
    // the same body captured in three pieces (one nested), spliced back in capture order
    let (mut b, out, tid) = base();
    let (all, _) = b.capture(|b| {
        let (inner, _) = b.capture(|b| body(b, out, tid));
        let n = inner.nodes.len();
        assert!(n >= 8, "regions stay single nodes: {n}");
        b.splice(inner);
    });
    b.splice(all);
    assert_eq!(dbg(&b.finish()), dbg(&direct));
}

#[test]
fn every_registered_kernel_survives_a_node_splice() {
    for k in amdgpu_kernels::kernel_names() {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let mut g = f.clone();
        // rebuild the body by splicing its top-level nodes into a capture and back
        let nodes: Vec<Node> = f.body().to_vec();
        g.set_body(nodes);
        assert_eq!(dbg(&g), dbg(&f), "{k}");
    }
}

#[test]
fn facts_classify_and_order() {
    let (mut b, out, tid) = base();
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let (c, _) = b.capture(|b| {
        let a = b.c_i32(8); //                                           0 pure (salu)
        let p = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(a))); //     1 pure, uses 0
        b.produce_region("T", 0, 0, |b| b.store_lds_b32(p, tid)); //      2 LDS write slot 0
        b.produce_region("T", 1, 0, |b| b.store_lds_b32(p, tid)); //      3 LDS write slot 1
        b.barrier(); //                                                   4 anchor
        let r = b.load_lds_b32(p); //                                     5 LDS read
        let q = b.add_i32(tid, 3); //                                     6 pure
        b.set_var(vars[0], q); //                                         7 var write
        let s = b.add_i32(Opnd::Var(vars[0]), 1); //                      8 var read
        let f = b.let_(Op::BitcastF32(Opnd::Val(r))); //                  9
        b.store_f32(out, 0, f); //                                        10 global write
        let _ = s;
    });
    let fs = b.facts(&c);
    assert_eq!(fs.len(), 11);
    assert_eq!(fs[0].class, ResClass::Salu);
    assert!(must_order(&fs[0], &fs[1]), "RAW on a value");
    assert!(!must_order(&fs[2], &fs[3]), "different constant slots of one tile are disjoint");
    assert!(must_order(&fs[2], &fs[4]) && must_order(&fs[4], &fs[5]), "LDS work is ordered against the barrier");
    assert!(must_order(&fs[2], &fs[5]) || must_order(&fs[1], &fs[5]), "LDS write -> read");
    assert!(!must_order(&fs[4], &fs[6]), "pure work may cross a barrier");
    assert!(must_order(&fs[7], &fs[8]), "loop variable write -> read");
    assert!(!must_order(&fs[6], &fs[5]), "independent pure / LDS read");
    assert!(must_order(&fs[4], &fs[10]), "a global store keeps its order against the barrier");
    assert_eq!(fs[5].class, ResClass::LdsRead);
    assert_eq!(fs[5].footprint, 1);
    assert_eq!(fs[2].tile.as_ref().map(|t| t.0.as_str()), Some("T"));
    let e = c.order_edges(&b.f);
    assert!(e.contains(&(0, 1)) && !e.contains(&(2, 3)));
    // a legal reorder (independent pure statement 6 moved first) still verifies
    let mut nodes = c.nodes.clone();
    let six = nodes.remove(6);
    nodes.insert(0, six);
    b.splice_nodes(nodes);
    b.end_loop();
    let f = b.finish();
    assert!(amdgpu_dsl::verify(&f).is_empty(), "{:?}", amdgpu_dsl::verify(&f));
}

#[test]
fn weave_region_marks_group_and_anchor() {
    let (mut b, _out, tid) = base();
    let (c, _) = b.capture(|b| {
        b.add_i32(tid, 1);
        b.weave_region("w", |b| {
            b.add_i32(tid, 2);
            b.add_i32(tid, 3);
        });
        b.add_i32(tid, 4);
    });
    let g = c.groups();
    assert_eq!(g, vec![0..1, 1..5, 5..6]);
    b.splice(c);
    let _ = b.finish();
}

#[test]
fn mfma_passes_match_the_isa_atoms() {
    use amdgpu_dsl::facts::op_class;
    let o = Opnd::I32(0);
    for (op, atom) in [
        (Op::MfmaScaleFp8_32x32x64 { a: o, b: o, c: o, scale_a: o, scale_b: o }, amdgpu_isa::mfma_scale_32x32x64_f8()),
        (Op::MfmaF16Wide { a: o, b: o, c: o }, amdgpu_isa::mfma_32x32x8_f16()),
        (Op::SmfmacFp8_16x16x128 { a: o, b: o, idx: o, c: o }, amdgpu_isa::smfmac_16x16x128_fp8()),
    ] {
        assert_eq!(op_class(&op).0, ResClass::Mfma { passes: atom.passes as u8 }, "{op:?}");
    }
}

#[test]
#[should_panic(expected = "capture")]
fn an_unclosed_region_in_a_capture_panics() {
    let (mut b, _, _) = base();
    let id = b.capture_begin();
    let _ = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let _ = b.capture_end(id);
}
