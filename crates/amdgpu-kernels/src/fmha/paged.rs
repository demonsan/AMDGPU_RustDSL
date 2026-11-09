//! Reusable pieces for paged-KV decode kernels (derived from `decode16.rs`; see `docs/pa-decode16-profile.md`).
//!
//! * [`PageGroups`]: the block-table lookups of a tile. A tile of `kt` keys starting at a multiple of `kt` spans `kt / min(page, kt)` PAGE GROUPS of consecutive keys that share one page (page 16, kt 32: 2 pages).
//!   One wave-uniform table load per group replaces one per-lane load per DMA (8 per tile in `decode16`): about 24 % fewer VMEM instructions and 11 % fewer VALU instructions per dispatch, 4 % less time at B16 (`_la` forms: 6-12 %).
//!   The page of a CLAMPED key (`min(key, k_hi - 1)`) is the page of the group's first key clamped the same way, so the tails stay exact.
//! * [`v_swizzle`]: the conflict-free layout of the V tile for the transposed reads (`ds_read_b64_tr_b16`): tile rows are 256 bytes (a whole bank cycle), so 16 consecutive rows read at the same granule hit the
//!   same 8 banks (16-way conflict, measured: 78 % of the LDS-active cycles at B64). Store the 32-byte granule `G` of tile row `r` at granule `G ^ (r % 8)`; the DMA lands lane-linear, so the swizzle is applied to
//!   the SOURCE chunk (`l16 ^ 2 (r % 8)`), the reads use `(db ^ (r0 % 8)) * 32` with the per-granule bases hoisted. Measured: bank conflicts -85 % (B64) / -100 % (B16), `SQ_WAIT_INST_LDS` -65 % / -92 %,
//!   time within noise on its own (the per-tile latency chain, not the LDS, is the limiter there).
use amdgpu_dsl::{ev, FnBuilder, I32, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageGroups {
    pub page: i32,
    pub kt: i32,
    /// keys per group
    pub gsz: i32,
    pub ngrp: i32,
}

impl PageGroups {
    /// `kt` keys per tile (a power of two; every tile starts at a multiple of `kt`), `page` keys per page (a power of two, >= 4).
    pub fn new(page: i32, kt: i32) -> PageGroups {
        assert!(page >= 4 && page.count_ones() == 1 && kt.count_ones() == 1);
        let gsz = page.min(kt);
        PageGroups { page, kt, gsz, ngrp: kt / gsz }
    }

    /// The group of the DMA `i` whose 64 lanes carry the 4 keys `4 i .. 4 i + 3` of the tile (16 lanes per key).
    pub fn group_of_dma(&self, i: i32) -> usize {
        ((4 * i) / self.gsz) as usize
    }

    /// The wave-uniform table load of group `g` of the tile starting at `key_t` (a uniform value): the raw f32-typed page id of the clamped first key of the group. `r_bt` is the block-table descriptor,
    /// `bt_soff` the sequence's row offset in bytes, `k_hi` / `maxk` the (uniform) clamp bounds.
    pub fn load(&self, b: &mut FnBuilder, r_bt: ValueId, bt_soff: ValueId, key_t: I32, k_hi: ValueId, maxk: ValueId, g: i32) -> ValueId {
        let key = b.bind_i32((key_t + g * self.gsz).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
        let bvo = b.bind_i32(ev(key).lshr(self.page.trailing_zeros() as i32) << 2);
        b.buffer_load_f32(r_bt, bvo, bt_soff, 0)
    }
}

/// The V tile swizzle (see the module docs): the granule (32 bytes = 16 d) stored at position `g` of tile row `r`.
pub fn v_swizzle(row: i32, granule: i32) -> i32 {
    granule ^ (row & 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_groups_cover_a_tile() {
        let g = PageGroups::new(16, 32);
        assert_eq!((g.gsz, g.ngrp), (16, 2));
        assert_eq!((0..8).map(|i| g.group_of_dma(i)).collect::<Vec<_>>(), [0, 0, 0, 0, 1, 1, 1, 1]);
        let g = PageGroups::new(64, 32);
        assert_eq!((g.gsz, g.ngrp), (32, 1));
        assert!((0..8).all(|i| g.group_of_dma(i) == 0));
        let g = PageGroups::new(8, 32);
        assert_eq!(g.ngrp, 4);
        assert_eq!(g.group_of_dma(7), 3);
    }

    /// Bank model of one `ds_read_b64_tr_b16` over a 16-row x 32-byte pattern (lane s of a 16-lane group at tile row 4 g + s / 4, 8 bytes at (s % 4) 8 inside the granule `db`): the conflict degree
    /// is the largest number of distinct dwords mapped to one of the 64 banks (4 bytes each) by all 64 lanes of the instruction.
    fn degree(swizzled: bool) -> usize {
        let mut worst = 0;
        for db in 0..8 {
            let mut banks = vec![std::collections::BTreeSet::new(); 64];
            for lane in 0..64 {
                let (grp, l16) = (lane / 16, lane % 16);
                let row = 4 * grp + l16 / 4;
                let gran = if swizzled { v_swizzle(row, db) } else { db };
                let byte = row * 256 + gran * 32 + (l16 % 4) * 8;
                for dw in 0..2 {
                    let addr = byte + 4 * dw;
                    banks[((addr / 4) % 64) as usize].insert(addr / 4);
                }
            }
            worst = worst.max(banks.iter().map(|s| s.len()).max().unwrap());
        }
        worst
    }

    #[test]
    fn the_swizzle_removes_the_bank_conflict() {
        assert_eq!(degree(false), 16);
        assert!(degree(true) <= 2, "degree {}", degree(true));
    }

    #[test]
    fn the_source_chunk_swizzle_inverts_the_read_swizzle() {
        // LDS granule position p of row r holds global granule p ^ (r % 8); the DMA lane l16 (chunk position) therefore loads the global chunk l16 ^ 2 (r % 8)
        for r in 0..32 {
            for l16 in 0..16 {
                let src_chunk = l16 ^ ((r & 7) << 1);
                let (gran, half) = (l16 / 2, l16 % 2);
                assert_eq!(src_chunk, 2 * v_swizzle(r, gran) + half);
            }
        }
    }
}