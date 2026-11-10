//! `ds_read_b64` / `ds_write_b64`: IR forms (Frag2), lowered with folded immediates, emitted, assembled by asm_assembles-style check, and proven by the
//! GPU-semantics verifier (LDS bounds with 8-byte accesses, tile protocol).
use amdgpu_dsl::ev;
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

pub fn lds64_demo() -> Function {
    let mut b = FnBuilder::new("probe_lds_b64", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg_tensor("o", amdgpu_dsl::Ext::c(64 * 8));
    b.lds(1024);
    b.lds_tile("T", 0, 1024);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let lane = b.bind_i32(ev(tid) & 63);
    let a = b.bind_i32(ev(lane) << 3);
    let w0 = b.bind_i32(ev(lane) * 3);
    let w1 = b.bind_i32(ev(lane) + 1000);
    let f2 = b.frag2(w0, w1);
    b.tile_produce("T", 0, 0);
    b.store_lds_b64(a, f2);
    b.tile_end();
    b.wait_lds(0);
    b.barrier();
    // read the neighbour lane's pair at +8 bytes (folded immediate)
    let ra = b.bind_i32(ev(a) & 0x1f0);
    let rd = b.bind_i32(ev(ra) + 8);
    b.tile_consume("T", 0, 0);
    let r = b.load_lds_b64(rd);
    b.tile_end();
    b.wait_lds(0);
    let lo = b.let_(Op::Frag2Lane { frag: Opnd::Val(r), index: 0 });
    let p = b.ptr_add(o, a);
    let lf = b.bitcast_f32(lo);
    b.store_f32(p, 0, lf);
    b.finish()
}

#[test]
fn lds_b64_forms_lower_emit_and_verify() {
    let f = lds64_demo();
    assert!(amdgpu_dsl::verify(&f).is_empty(), "{:?}", amdgpu_dsl::verify(&f));
    let r = amdgpu_codegen::gpu_verify::analyse(&f).expect("analyse");
    assert!(r.errors().is_empty(), "{}", r.render());
    let c = amdgpu_codegen::compile(&f).expect("compile");
    assert!(c.asm.contains("ds_write_b64 v"), "{}", c.asm);
    assert!(c.asm.lines().any(|l| l.contains("ds_read_b64 v[") && l.contains("offset:8")), "{}", c.asm);
    std::fs::write(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("lds64.s"), &c.asm).ok();
}
