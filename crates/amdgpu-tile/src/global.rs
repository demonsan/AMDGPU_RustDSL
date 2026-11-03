//! Register <-> global partitions (G2R loads, R2G stores) of a TiledCopy whose ROW is supplied by the kernel (a clamped, wave-dependent row the layout cannot
//! express): the view gives the in-row byte offset of each element (row stride 0), the partition splits it into per-lane bases (bound once, added to the
//! row offset in one pointer per base) and per-instruction immediates (the 13-bit offset field, non-negative here).

use crate::copy::TiledCopy;
use crate::Ctx;
use amdgpu_dsl::{FnBuilder, ValueId, I32};
use amdgpu_isa::Space;
use amdgpu_layout::{layout_to_offsets, proofs, ComposedLayout, ImmRange, IssuePlan, Ix, Layout, Swizzle};

/// Per-lane bases and per-operation (base, immediate) of a register <-> global copy.
#[derive(Clone, Debug)]
pub struct GlobalPart {
    pub tc: TiledCopy,
    pub bases: Vec<I32>,
    /// per placement: the instructions (base index, immediate)
    pub ops: Vec<Vec<(usize, i64)>>,
}

impl TiledCopy {
    /// `view`: tile coordinates -> byte offset inside the kernel-supplied row (static); `placements`: static byte offsets of each operation.
    pub fn partition_global(&self, b: &mut FnBuilder, ctx: &Ctx, lane: &Ix, view: &Layout, placements: &[i64]) -> GlobalPart {
        let a = self.atom;
        assert!(a.src == Space::Global || a.dst == Space::Global, "partition_global: {} is not a global access", a.name);
        assert!(a.lds_lane_stride.is_none(), "partition_global: {} is an LDS-direct DMA (use partition_dma)", a.name);
        let vec = self.vec();
        if let Err(e) = proofs::check_contiguous(&self.tv, &ComposedLayout::plain(view.clone()), vec, (self.elem_bits / 8) as i64, a.align) {
            panic!("partition_global({}): {e}", a.name);
        }
        let plan = IssuePlan::new(self.tv.clone(), vec);
        let n = plan.issues() as usize;
        let mut exprs: Vec<Ix> = Vec::new();
        for &x in placements {
            exprs.extend(plan.offsets(view, lane, &Ix::C(x), &ctx.dyns));
        }
        // the builder's global forms take a non-negative 12-bit-safe immediate
        let op = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange { lo: 0, hi: 4095, align: 1 }, &ctx.dyns);
        let bases = op.bases.iter().map(|x| ctx.bind(b, x)).collect();
        let ops = op.issues.chunks(n).map(|c| c.to_vec()).collect();
        GlobalPart { tc: self.clone(), bases, ops }
    }
}

impl GlobalPart {
    /// One pointer per base: `ptr + row_off + base` (bound once).
    pub fn ptrs(&self, b: &mut FnBuilder, ptr: ValueId, row_off: &I32) -> Vec<ValueId> {
        self.bases
            .iter()
            .map(|x| {
                let o = if x.as_const() == Some(0) { b.bind_i32(row_off) } else { b.bind_i32(row_off + x) };
                b.ptr_add(ptr, o)
            })
            .collect()
    }
    /// Loads of placement `o`: one value per instruction (`Frag4` for dwordx4, the dword for dword).
    pub fn load(&self, b: &mut FnBuilder, ptrs: &[ValueId], o: usize) -> Vec<ValueId> {
        self.ops[o]
            .iter()
            .map(|&(bi, imm)| match self.tc.atom.bytes {
                16 => b.load_b128(ptrs[bi], imm as u32),
                4 => b.load_f32(ptrs[bi], imm as u32),
                n => panic!("GlobalPart::load: no {n}-byte global load form"),
            })
            .collect()
    }
    /// Stores of placement `o`: one value per instruction (`Frag2` for dwordx2, `Frag4` for dwordx4, f32 for dword).
    pub fn store(&self, b: &mut FnBuilder, ptrs: &[ValueId], o: usize, vals: &[ValueId]) {
        assert_eq!(vals.len(), self.ops[o].len(), "GlobalPart::store: one value per instruction");
        for (&(bi, imm), v) in self.ops[o].iter().zip(vals) {
            match self.tc.atom.bytes {
                8 => b.store_f32x2(ptrs[bi], imm as u32, *v),
                16 => b.store_f32x4(ptrs[bi], imm as u32, *v),
                4 => b.store_f32(ptrs[bi], imm as u32, *v),
                n => panic!("GlobalPart::store: no {n}-byte global store form"),
            }
        }
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}
