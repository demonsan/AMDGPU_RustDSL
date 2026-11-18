//! LDS layout planner for TDM tiles (gfx1250): the tensor data mover writes a tile row-major into LDS and can insert `amount` padding
//! dwords after every `interval` dwords (`TensorTile::pad`, CDNA5 ISA section 10.11: `if bytesStored >> 3 >= 1 << pad_interval: Laddr +=
//! pad_amount`). The planner picks the padding that removes the bank conflicts of the tile's declared reads.
//!
//! Model (CDNA5 ISA section 11: 64 LDS banks of 4 bytes): an LDS read of `bytes` per lane is served in phases of `256 / bytes` lanes
//! (64 banks x 4 bytes); a phase takes as many cycles as the most distinct dwords any bank holds (the same dword read by several lanes is a
//! broadcast). The cost of a pattern is the sum over its phases. Read patterns are given in LOGICAL tile offsets (the unpadded row-major
//! byte offset); `padded` maps them to the LDS layout the TDM writes, so a template computes its read addresses with the same function.

/// One read instruction of a wave: per lane, the logical byte offset in the tile (`None`: the lane does not read).
#[derive(Clone, Debug)]
pub struct LaneRead {
    /// bytes per lane: 4, 8 or 16
    pub bytes: u32,
    pub lanes: Vec<Option<u32>>,
}

/// The chosen padding: `pad` (interval, amount) in dwords, the conflict cost with and without it, the extra LDS bytes for `tile_bytes`.
#[derive(Clone, Debug, PartialEq)]
pub struct PadPlan {
    pub pad: Option<(u16, u16)>,
    pub cost: u32,
    pub cost_unpadded: u32,
    pub ideal: u32,
    pub extra_bytes: u32,
}

/// The LDS byte offset of logical tile offset `off` under `pad` (interval, amount in dwords): `amount` dwords after every full `interval`.
pub fn padded(off: u32, pad: Option<(u16, u16)>) -> u32 {
    match pad {
        None => off,
        Some((iv, am)) => off + (off / (4 * iv as u32)) * 4 * am as u32,
    }
}

/// LDS bytes a tile of `bytes` occupies under `pad` (the TDM adds the padding after every full interval, the last one included).
pub fn footprint(bytes: u32, pad: Option<(u16, u16)>) -> u32 {
    match pad {
        None => bytes,
        Some((iv, am)) => bytes + bytes.div_ceil(4 * iv as u32) * 4 * am as u32,
    }
}

/// Bank-conflict cost of `reads` under `pad` (sum over phases of the busiest bank's distinct dwords).
pub fn cost(reads: &[LaneRead], pad: Option<(u16, u16)>) -> u32 {
    let mut total = 0;
    for r in reads {
        let per_phase = (256 / r.bytes.max(4)) as usize;
        for phase in r.lanes.chunks(per_phase.max(1)) {
            let mut banks: Vec<Vec<u32>> = vec![Vec::new(); 64];
            for off in phase.iter().flatten() {
                let base = padded(*off, pad);
                for k in 0..r.bytes.div_ceil(4) {
                    let dw = base / 4 + k;
                    let b = &mut banks[(dw % 64) as usize];
                    if !b.contains(&dw) {
                        b.push(dw);
                    }
                }
            }
            total += banks.iter().map(|b| b.len() as u32).max().unwrap_or(0).max(1);
        }
    }
    total
}

/// The conflict-free (or least conflicting) padding for a tile of `tile_bytes` read by `reads`: intervals 2..256 dwords (powers of two,
/// the TDM's encoding), amounts 1..32 dwords; among the least-cost choices the one with the fewest extra bytes, then the largest interval.
/// No padding when it does not lower the cost.
pub fn plan(tile_bytes: u32, reads: &[LaneRead]) -> PadPlan {
    let ideal: u32 = reads.iter().map(|r| r.lanes.chunks((256 / r.bytes.max(4)) as usize).count() as u32).sum();
    let c0 = cost(reads, None);
    let mut best = PadPlan { pad: None, cost: c0, cost_unpadded: c0, ideal, extra_bytes: 0 };
    if c0 == ideal {
        return best;
    }
    for e in 1..=8u32 {
        let iv = 1u16 << e;
        for am in 1..=32u16 {
            let p = Some((iv, am));
            let c = cost(reads, p);
            let extra = footprint(tile_bytes, p) - tile_bytes;
            let better = c < best.cost || (c == best.cost && best.pad.is_some() && (extra < best.extra_bytes || (extra == best.extra_bytes && iv > best.pad.unwrap().0)));
            if better {
                best = PadPlan { pad: p, cost: c, cost_unpadded: c0, ideal, extra_bytes: extra };
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// kernel-opt's MLA K reads: lane (m = lane & 15, g = lane >> 4) reads 16 bytes at row (16 w + m) * 512 + 64 c + 32 g (+16).
    fn k_reads(row: u32) -> Vec<LaneRead> {
        (0..8)
            .flat_map(|c| {
                (0..2).map(move |h| LaneRead { bytes: 16, lanes: (0..32u32).map(|l| Some((l & 15) * row + 64 * c + 32 * (l >> 4) + 16 * h)).collect() })
            })
            .collect()
    }

    #[test]
    fn the_mla_k_tile_gets_a_bank_shift_per_row() {
        let r = k_reads(512);
        let p = plan(64 * 512, &r);
        assert!(p.cost_unpadded > p.ideal * 4, "{p:?}");
        assert_eq!(p.cost, p.ideal, "conflict-free: {p:?}");
        // 4 dwords after every 128-dword row (the hand-found fix, ROW = 528) is optimal
        assert_eq!(cost(&r, Some((128, 4))), p.ideal);
        assert!(p.extra_bytes <= 64 * 16, "{p:?}");
    }

    #[test]
    fn padded_addresses_and_footprint_agree() {
        let pad = Some((128, 4));
        assert_eq!(padded(512, pad), 528);
        assert_eq!(padded(511, pad), 511);
        assert_eq!(footprint(64 * 512, pad), 64 * 528);
        assert_eq!(footprint(100, None), 100);
    }

    #[test]
    fn a_conflict_free_pattern_needs_no_padding() {
        let r = vec![LaneRead { bytes: 16, lanes: (0..32u32).map(|l| Some(16 * l)).collect() }];
        assert_eq!(plan(4096, &r).pad, None);
    }
}