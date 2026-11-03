//! gpu_verify possible-bits rules for XOR / AND: bounded swizzled addresses `row ^ chunk` and `x & -16` masks are proven inside the LDS segment, and
//! the attacks (possibly negative operands, carries of overlapping sums) are still rejected.
use amdgpu_dsl::tile::ev;
use amdgpu_dsl::{FnBuilder, Function, Op, I32};

/// One `ds_read_b128` at the address built by `addr` from the workitem id, in an `lds`-byte segment.
fn kernel(lds: u32, addr: impl Fn(I32) -> I32) -> Function {
    let mut b = FnBuilder::new("gv_bits", 64);
    b.lds(lds);
    let tid = b.let_(Op::WorkitemId);
    let a = b.bind_i32(addr(ev(tid)));
    let _ = b.load_lds_b128(a);
    b.finish()
}

fn lds_oob(f: &Function) -> usize {
    let r = amdgpu_compiler::gpu_verify::analyse(f).expect("analyse");
    r.errors().iter().filter(|e| e.code == "GV-LDS-OOB").count()
}

#[test]
fn swizzled_row_xor_chunk_is_inside_the_block() {
    // (row << 7) ^ (chunk << 4) with row < 16, chunk < 8: at most 2032 = 2048 - 16 (the interval xor rule alone gives 2047)
    let f = kernel(2048, |t| ((t.clone() & 15) << 7) ^ (((t.lshr(4)) & 7) << 4));
    assert_eq!(lds_oob(&f), 0, "row ^ chunk is proven inside the 2 KB block");
    // and the same base plus a 16 KB slot offset inside an 18 KB segment
    let f = kernel(16384 + 2048, |t| (((t.clone() & 15) << 7) ^ (((t.lshr(4)) & 7) << 4)) + 16384);
    assert_eq!(lds_oob(&f), 0);
}

#[test]
fn negative_masks_of_non_negative_values() {
    let f = kernel(1024, |t| (t * 16) & -16);
    assert_eq!(lds_oob(&f), 0, "x & -16 of x in [0, 1008]");
}

#[test]
fn attacks_are_rejected() {
    // possibly negative operand: (tid - 32) ^ 16 can be negative
    assert!(lds_oob(&kernel(4096, |t| (t - 32) ^ 16)) > 0, "xor with a possibly negative operand");
    // & -16 of a possibly negative value
    assert!(lds_oob(&kernel(4096, |t| ((t - 8) * 16) & -16)) > 0, "negative value under a negative mask");
    // carries: x + x with x = (tid & 15) << 4 reaches 480 (a carry-free bit rule would say 240)
    assert!(lds_oob(&kernel(256, |t| { let x = (t & 15) << 4; x.clone() + x })) > 0, "overlapping sum carries");
    // xor of bounded parts that really exceed the segment
    assert!(lds_oob(&kernel(2032, |t| ((t.clone() & 15) << 7) ^ (((t.lshr(4)) & 7) << 4))) > 0, "2032 + 16 > 2032");
}
