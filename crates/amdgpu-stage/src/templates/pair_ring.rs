//! L2 template T1 (docs/l2-templates.md §3.1): the barrier-per-tile 2-slot ring over tile pairs, ported from the hand-written loops of the
//! backward, MLA-hp and bf16 v1 attention kernels.

use amdgpu_dsl::{ev, varx};
use amdgpu_dsl::{FnBuilder, Opnd, ValueId, VarId};

/// T1, the barrier-per-tile 2-slot ring unrolled over tile pairs (docs/l2-templates.md §3.1): the cadence the backward kernels
/// (`fmha_bwd_dq*`, `fmha_bwd_dkdv*`), the MLA high-parallelism decode (`mla_decode_bf16_hp_p*`) and the bf16 v1 forward (`fmha_fwd_hd*_bf16*`)
/// wrote by hand. One loop pass covers tiles 2·it and 2·it + 1 with static slots 0 / 1:
/// ```text
/// prologue (kernel):   stage(first, slot 0, gen 0)
/// pass it, slot s:     tile = first + 2·it + s | wait_vm(0) wait_lds(0) barrier | stage(min(tile + 1, last), slot 1 - s, gen) | consume(s, tile)
/// after the pass:      it += 1; break when it >= npair
/// ```
/// The barrier both publishes `tile` (its DMA is complete) and frees slot 1 - s (every wave is past tile - 1). The pass past the end stages a clamped
/// tile again (`last` = the pair's last tile), so there is no branch around the DMA. The generation of the staged slot is `it` for slot 0 -> 1 and
/// `it + 1` for slot 1 -> 0, matching the (slot, it) generation the consumer reads. Emission order is the hand-written one (md5-identical port).
pub struct PairRing {
    /// tile pairs to run (the loop bound)
    pub npair: ValueId,
    /// the first tile index (causal / split offsets); None = 0
    pub first: Option<ValueId>,
    /// a weave region around the next tile's stage (the `kv_stage`-style policy hook); None = no region
    pub stage_region: Option<&'static str>,
    /// emission order of the next-tile index: false = `last`, then `min(tile + 1, last)` (backward, MLA-hp); true = `tile + 1`, `last`,
    /// then the clamp (the bf16 v1 forward). Same values; the flag only keeps each port's instruction order and therefore its asm md5
    pub next_first: bool,
}

impl PairRing {
    /// Emits the two half-passes of one loop iteration and the loop exit (`it += 1`, break at `npair`); the caller opened the loop with `it` as
    /// its tile-pair counter and closes it (`end_loop`) after this call. `stage(b, tile, slot, gen)` issues the DMAs of `tile` into `slot`;
    /// `consume(b, slot, tile)` is the work on the published tile.
    pub fn run<S, C>(&self, b: &mut FnBuilder, it: VarId, mut stage: S, mut consume: C)
    where
        S: FnMut(&mut FnBuilder, ValueId, i32, Opnd),
        C: FnMut(&mut FnBuilder, i32, ValueId),
    {
        let itx = varx(it);
        for slot in 0..2i32 {
            let tile = match self.first {
                Some(f) => b.bind_i32((itx.clone() << 1) + slot + ev(f)),
                None => b.bind_i32((itx.clone() << 1) + slot),
            };
            b.wait_vm(0);
            b.wait_lds(0);
            b.barrier();
            if let Some(r) = self.stage_region {
                b.weave_region_begin(r);
            }
            let nxt1 = self.next_first.then(|| b.bind_i32(ev(tile) + 1));
            let last = match self.first {
                Some(f) => b.bind_i32((ev(self.npair) << 1) - 1 + ev(f)),
                None => b.bind_i32((ev(self.npair) << 1) - 1),
            };
            let nxt = match nxt1 {
                Some(n) => b.bind_i32(ev(n).min(ev(last))),
                None => b.bind_i32((ev(tile) + 1).min(ev(last))),
            };
            let gen_n = if slot == 0 { Opnd::Var(it) } else { Opnd::Val(b.bind_i32(itx.clone() + 1)) };
            stage(b, nxt, 1 - slot, gen_n);
            if self.stage_region.is_some() {
                b.weave_region_end();
            }
            consume(b, slot, tile);
        }
        let it1 = b.bind_i32(itx + 1);
        b.set_var(it, it1);
        let done = b.bind_cond(ev(it1).ge(ev(self.npair)));
        b.break_if(done);
    }
}