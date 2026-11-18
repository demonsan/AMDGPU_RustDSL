//! The tensor-bounds rule (docs/gfx1250-hang-rules.md, "tensor bounds"): the global base of every tensor descriptor must come from a kernarg pointer with a declared extent (the descriptor's
//! address words the two halves of ONE pointer), and the tile (base + offset .. + (rows - 1) x row stride + row bytes) must be proven inside that extent. A null, unrelated or out-of-range base
//! is a compile error (a descriptor built from zeroed registers hangs the device).

use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, TdmCpol, TensorTile};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Case {
    Legal,
    /// the address words are constants (a null base)
    ConstBase,
    /// the address words are two unrelated 32-bit registers
    UnrelatedWords,
    /// low word from one kernarg pointer, high word from another
    MixedHalves,
    /// no declared extent for the pointer
    NoExtent,
    /// the tile (8 rows x 64 bytes = 512) is larger than the declared extent (256)
    PastExtent,
    /// the byte offset is an unconstrained kernarg
    UnboundedOffset,
    /// the offset is a kernarg with a declared range that keeps the tile inside the extent
    BoundedOffset,
}

fn kernel(c: Case) -> Function {
    let mut b = FnBuilder::new("tb", 64);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("other", DType::Ptr, 8, 8);
    b.arg("off", DType::I32, 16, 4);
    b.arg("w", DType::I32, 20, 4);
    b.lds(4096);
    match c {
        Case::NoExtent => {}
        Case::PastExtent => b.arg_tensor("src", Ext::c(256)),
        _ => b.arg_tensor("src", Ext::c(1024)),
    }
    b.arg_tensor("other", Ext::c(1024));
    if c == Case::BoundedOffset {
        b.ka_range(16, 0, 256);
    }
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let pq = b.let_(Op::KernargPtr { offset: 8 });
    let off = match c {
        Case::UnboundedOffset | Case::BoundedOffset => Opnd::Val(b.let_(Op::KernargI32 { offset: 16 })),
        _ => Opnd::I32(0),
    };
    let tile = TensorTile { data_size: 1, tensor_dim: [Opnd::I32(64), Opnd::I32(8)], stride0: Opnd::I32(64), tile: [64, 8], pad: None, wg_mask: 0 };
    let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), off, Opnd::I32(0));
    let g0 = match c {
        Case::ConstBase => b.let_(Op::RsrcCat { words: [Opnd::I32(1), Opnd::I32(0), Opnd::I32(0), Opnd::I32(0x8000_0000u32 as i32)] }),
        Case::UnrelatedWords => {
            let (lo, hi) = (b.let_(Op::KernargI32 { offset: 20 }), b.let_(Op::KernargI32 { offset: 16 }));
            b.let_(Op::RsrcCat { words: [Opnd::I32(1), Opnd::I32(0), Opnd::Val(lo), Opnd::Val(hi)] })
        }
        Case::MixedHalves => {
            let (lo, hi) = (b.let_(Op::PtrLo(Opnd::Val(ps))), b.let_(Op::PtrHi(Opnd::Val(pq))));
            let hm = b.let_(Op::AndI32(Opnd::Val(hi), Opnd::I32(0x01ff_ffff)));
            let w3 = b.let_(Op::AddI32(Opnd::Val(hm), Opnd::I32(0x8000_0000u32 as i32)));
            b.let_(Op::RsrcCat { words: [Opnd::I32(1), Opnd::I32(0), Opnd::Val(lo), Opnd::Val(w3)] })
        }
        _ => g0,
    };
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default());
    b.wait_tensor(0);
    b.finish()
}

fn rejected(c: Case, code: &str) {
    let e = amdgpu_codegen::compile_target(&kernel(c), "gfx1250").err().unwrap_or_else(|| panic!("the kernel compiled but must be rejected with {code}"));
    assert!(e.contains(code), "expected {code}: {e}");
}

#[test]
fn a_tile_inside_a_declared_extent_compiles() {
    amdgpu_codegen::compile_target(&kernel(Case::Legal), "gfx1250").unwrap_or_else(|e| panic!("{e}"));
    // an offset the contract bounds so that the tile stays inside
    amdgpu_codegen::compile_target(&kernel(Case::BoundedOffset), "gfx1250").unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn a_base_from_a_constant_is_rejected() {
    rejected(Case::ConstBase, "GV-TDM-PROV");
}

#[test]
fn a_base_from_unrelated_registers_is_rejected() {
    rejected(Case::UnrelatedWords, "GV-TDM-PROV");
}

#[test]
fn halves_of_different_pointers_are_rejected() {
    rejected(Case::MixedHalves, "GV-TDM-PROV");
}

#[test]
fn a_pointer_without_a_declared_extent_is_rejected() {
    rejected(Case::NoExtent, "GV-GLB-NOEXT");
}

#[test]
fn a_tile_past_the_extent_is_rejected() {
    rejected(Case::PastExtent, "GV-GLB-OOB");
}

#[test]
fn an_unbounded_offset_is_rejected() {
    rejected(Case::UnboundedOffset, "GV-GLB-OOB");
}
