//! gfx1250 MLA decode pieces: the `ds_load_tr8_b64` lane-mapping probe (the ISA gives the mapping only as diagrams).

use amdgpu_dsl::{DType, Ext, FnBuilder, Function, MemTy, Op, Opnd};

/// `ds_load_tr8_b64` mapping probe (one wave): LDS bytes 0..512 hold their own address (byte i = i mod 256 + 256 * (i >= 256) is not
/// needed: the probe reads within the first 256 bytes), lane l reads at `(l % 16) * 16 + (l / 16) * 8` (8 contiguous bytes of memory row l % 16,
/// half l / 16 of a 16 x 16 byte tile) and stores its two result dwords to out[2 l], out[2 l + 1]. The host prints which memory byte landed
/// in which (lane, byte) of the result.
pub fn probe_g1250_tr8() -> Function {
    let mut b = FnBuilder::new("probe_g1250_tr8", 32);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::c(64 * 4));
    b.lds(256);
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    // fill: lane l writes dwords 2 l and 2 l + 1 (bytes 8 l .. 8 l + 7), byte value = its address
    for h in 0..2 {
        let k = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(1)));
        let k = b.add_i32(k, Opnd::I32(h));
        let mul = b.mul_i32(k, Opnd::I32(0x0404_0404));
        let v = b.add_i32(mul, Opnd::I32(0x0302_0100));
        let a = b.let_(Op::ShlI32(Opnd::Val(k), Opnd::I32(2)));
        b.store_lds(a, v, MemTy::B32, false);
    }
    b.barrier();
    let row = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(15)));
    let half = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(4)));
    let ra = b.let_(Op::ShlI32(Opnd::Val(row), Opnd::I32(4)));
    let ha = b.let_(Op::ShlI32(Opnd::Val(half), Opnd::I32(3)));
    let addr = b.add_i32(ra, ha);
    let x = b.load_lds_tr_b8(addr);
    let off = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(3)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
    let x0 = b.frag_slice(Opnd::Val(x), 0, 1);
    let x1 = b.frag_slice(Opnd::Val(x), 1, 1);
    b.store_global(q, 0, x0, MemTy::B32, false);
    b.store_global(q, 4, x1, MemTy::B32, false);
    b.finish()
}
