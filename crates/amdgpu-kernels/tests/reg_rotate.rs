//! W3 B6b (docs/w3-b5-b6.md section 4): `reg_rotate(name, bufs, fragments)` - fragment i in buffer i mod `bufs`. The fragments of one
//! residue class share one register tuple (their lifetimes must be disjoint), the classes never overlap; a class whose members are live
//! at the same time is an error naming the two fragments.

use amdgpu_dsl::{DType, FnBuilder, Function, Opnd, RegRef};

/// 8 LDS fragments read and written back one after the other (each dead before the next is read); with `keep_first`, fragment 0 is
/// written back again at the end (alive across fragment 4, its buffer partner).
fn frags(keep_first: bool) -> Function {
    let mut b = FnBuilder::new("probe_rotate", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.lds(16384);
    let mut vs = Vec::new();
    for i in 0..8 {
        let f = b.load_lds_b128(Opnd::I32(i * 1024));
        b.store_lds_b128(Opnd::I32(8192 + i * 1024), f);
        vs.push(f);
    }
    if keep_first {
        b.store_lds_b128(Opnd::I32(15360), vs[0]);
    }
    let refs: Vec<RegRef> = vs.iter().map(|v| RegRef::Val(*v)).collect();
    b.reg_rotate("kv", 4, &refs);
    b.finish()
}

#[test]
fn a_rotation_shares_one_tuple_per_class_and_keeps_the_classes_apart() {
    let f = frags(false);
    let (m, a) = amdgpu_codegen::compile_mir_alloc(&f).unwrap_or_else(|e| panic!("{e}"));
    let (_, bufs, fr) = m.reg_hints.rotations.first().expect("rotation recorded").clone();
    assert_eq!(bufs, 4);
    let ph = |r: u32| a.phys[m.canon(r) as usize];
    for i in 0..8 {
        for j in 0..8 {
            if i % 4 == j % 4 {
                assert_eq!(ph(fr[i]), ph(fr[j]), "fragments {i} and {j} share buffer {}", i % 4);
            } else {
                assert_ne!(ph(fr[i]), ph(fr[j]), "fragments {i} and {j} are in different buffers");
            }
        }
    }
}

#[test]
fn overlapping_members_of_one_buffer_are_an_error_naming_them() {
    let e = amdgpu_codegen::compile_mir_alloc(&frags(true)).err().expect("fragment 0 lives across fragment 4");
    assert!(e.contains("rotation `kv`") && e.contains("fragment 0") && e.contains("fragment 4"), "{e}");
}
