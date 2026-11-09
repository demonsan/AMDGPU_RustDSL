//! L6c: small FMHA launch components whose safety facts the verifier cannot derive (variable shifts): each emits its code AND its trusted assumptions with
//! fixed reasons (counted in the certificate), so a kernel author never writes an `assume_trusted` for them.

use amdgpu_dsl::ev;
use amdgpu_dsl::{Ext, FnBuilder, ValueId, I32};

/// KV head of a query head under GQA (`H = HK * gqa`, gqa a power of two <= 16): `head >> log2(gqa)`, log2 from 4 compares. `hk` = the contract symbol of HK.
pub fn gqa_kv_head(b: &mut FnBuilder, head: ValueId, gqa: ValueId, hk: &str) -> ValueId {
    let g = ev(gqa);
    let s1 = g.gt(1).select_i32(1, 0);
    let s2 = g.gt(2).select_i32(2, s1);
    let s3 = g.gt(4).select_i32(3, s2);
    let sh = b.let_i32(g.gt(8).select_i32(4, s3));
    let kvh = b.bind_i32(ev(head).lshr(sh));
    b.assume_trusted(kvh, Ext::c(0), Ext::sym(hk) - Ext::c(1), "kv head = head >> log2(gqa) with head < H = HK * gqa (gqa a power of two <= 16): variable shift, no interval rule");
    kvh
}

/// The KV-split launch: grid z = B * splits (splits a power of two in {2, 4, 8}) decoded into (batch, split, log2 splits).
pub fn kv_split_decode(b: &mut FnBuilder, z: ValueId, splits: ValueId, splits_ka: u32, batch_sym: &str) -> (ValueId, ValueId, ValueId) {
    let lg = b.bind_i32(ev(splits).lshr(1) - ev(splits).lshr(3));
    let bi = b.bind_i32(ev(z).lshr(ev(lg)));
    let sp = b.bind_i32(ev(z) & ((I32::c(1) << ev(lg)) - 1));
    b.assume_trusted(bi, Ext::c(0), Ext::sym(batch_sym) - Ext::c(1), "batch = grid_z >> log2(splits) with grid_z = B * splits (grid_is): variable shift, no interval rule");
    b.assume_trusted(sp, Ext::c(0), Ext::ka(splits_ka) - Ext::c(1), "split index = grid_z & (splits - 1), splits a power of two: and with a variable mask");
    // `z` is the workgroup z id at every caller: (batch, split) is a bijection of it
    b.wg_bijection(&[bi, sp], &[2], "KV split: grid z = batch * splits + split with split < splits (a power of two)");
    // z is the workgroup z id at every caller: (batch, split) is a bijection of it
    (bi, sp, lg)
}

/// `kv_split_range` for a key range that varies per block (causal: the block's last visible key + 1 = `kv_len`): a split past the block's
/// tiles is EMPTY (returned flag, 0 / 1); it is clamped to the last tile (a valid range, so the pipeline runs) and the caller discards its
/// result (log-sum-exp = -inf). Returns `(key_lo, kv_eff, empty)`.
pub fn kv_split_range_var(b: &mut FnBuilder, kv_len: ValueId, split: ValueId, lg: ValueId, skv_ka: u32) -> (ValueId, ValueId, ValueId) {
    let ntiles = b.bind_i32((ev(kv_len) + 127).lshr(7));
    let tps = b.bind_i32((ev(ntiles) + ((I32::c(1) << ev(lg)) - 1)).lshr(ev(lg)));
    b.assume_trusted(tps, Ext::c(1), Ext::ka(skv_ka).ceil_div(128), "tiles per split = ceil(ntiles / splits) in [1, ntiles] (1 <= kv_len <= SK); variable shift by log2(splits)");
    let raw = b.bind_i32(ev(split) * ev(tps));
    let empty = b.bind_i32(ev(raw).ge(ev(ntiles)).select_i32(1, 0));
    let tile_lo = b.bind_i32(ev(raw).min(ev(ntiles) - 1));
    b.assume_trusted(tile_lo, Ext::c(0), Ext::ka(skv_ka).ceil_div(128) - Ext::c(1), "first tile of this split = min(split * tps, ntiles - 1) <= ntiles - 1 with ntiles <= ceil(SK / 128) (correlation through the min)");
    let hi_r = b.let_i32(ev(tile_lo) + ev(tps));
    let tile_hi = hi_r.lt(ev(ntiles)).select_i32(&hi_r, ev(ntiles));
    let key_lo = b.bind_i32(ev(tile_lo) << 7);
    b.sym_range("KL", 0, i64::MAX);
    b.assume_trusted(key_lo, Ext::sym("KL"), Ext::sym("KL"), "KL is only a NAME for key_lo = tile_lo << 7 (free symbol >= 0) so the next assume can state the correlation with the split length");
    let key_hi_t = b.let_i32(tile_hi << 7);
    let key_hi = key_hi_t.lt(ev(kv_len)).select_i32(&key_hi_t, ev(kv_len));
    let kv_eff = b.bind_i32(key_hi - ev(key_lo));
    b.assume_trusted(kv_eff, Ext::c(1), Ext::ka(skv_ka) - Ext::sym("KL"), "split length kv_eff = min(tile_hi << 7, kv_len) - key_lo with tile_lo <= ntiles - 1 < tile_hi (>= 1) and kv_len <= SK (correlation lost by the interval engine)");
    (key_lo, kv_eff, empty)
}

/// This split's key range: tiles `[split * tps, min(ntiles, (split + 1) * tps))` with `tps = ceil(ntiles / splits)`; returns `(key_lo, kv_eff)` (first key,
/// number of keys of the split). `skv_ka` = kernarg offset of the full sequence length. Emits the four trusted facts of the host guard
/// "every split is non-empty" (splits <= ceil(SK / 128)) and names `key_lo` with the symbol `KL` so `kv_eff <= SK - KL` can be stated.
pub fn kv_split_range(b: &mut FnBuilder, kv_len: ValueId, split: ValueId, lg: ValueId, skv_ka: u32) -> (ValueId, ValueId) {
    let ntiles = b.let_i32((ev(kv_len) + 127).lshr(7));
    let tps = b.bind_i32((&ntiles + ((I32::c(1) << ev(lg)) - 1)).lshr(ev(lg)));
    b.assume_trusted(tps, Ext::c(1), Ext::ka(skv_ka).ceil_div(128), "tiles per split = ceil(ntiles / splits) in [1, ntiles] (host guard: splits <= ntiles); variable shift by log2(splits)");
    let tile_lo = b.bind_i32(ev(split) * ev(tps));
    b.assume_trusted(tile_lo, Ext::c(0), Ext::ka(skv_ka).ceil_div(128) - Ext::c(1), "first tile of this split = split * tps <= ntiles - 1 (host guard: every split is non-empty, (splits - 1) * tps < ntiles)");
    let hi_r = b.let_i32(ev(tile_lo) + ev(tps));
    let tile_hi = ntiles.lt(&hi_r).select_i32(&ntiles, &hi_r);
    let key_lo = b.bind_i32(ev(tile_lo) << 7);
    b.sym_range("KL", 0, i64::MAX);
    b.assume_trusted(key_lo, Ext::sym("KL"), Ext::sym("KL"), "KL is only a NAME for key_lo = tile_lo << 7 (free symbol >= 0) so the next assume can state the correlation with the split length");
    let key_hi_t = b.let_i32(tile_hi << 7);
    let key_hi = key_hi_t.lt(ev(kv_len)).select_i32(&key_hi_t, ev(kv_len));
    let kv_eff = b.bind_i32(key_hi - ev(key_lo));
    b.assume_trusted(kv_eff, Ext::c(1), Ext::ka(skv_ka) - Ext::sym("KL"), "split length kv_eff = key_hi - key_lo with key_hi = min(tile_hi << 7, SK) <= SK (correlation lost by the interval engine) and >= 1 by the host guard");
    (key_lo, kv_eff)
}
