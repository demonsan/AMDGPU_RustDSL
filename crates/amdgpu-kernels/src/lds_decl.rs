//! Byte-exact LDS tile declarations for kernels whose LDS layout is fixed by hand (a historical asm that must not move): each helper declares the buffer to the verifier (`lds_tile` / `lds_ring` /
//! `lds_private`, no code) and returns the [`Shared`] handle whose `at(..).write / read` brackets every access with the protocol markers. `LdsPlan` places buffers itself (16-byte aligned,
//! sequential); these take the exact base, so a layout can be declared without changing a single address. Call `b.lds(total)` first: tiles must lie inside the segment.
use amdgpu_dsl::FnBuilder;
use amdgpu_tile::{Layout, Shared};

/// One tile `[base, base + bytes)`.
pub fn tile_at(b: &mut FnBuilder, name: &str, base: i64, bytes: i64) -> Shared {
    b.lds_tile(name, base, bytes);
    Shared { name: name.to_string(), base, slot_stride: bytes, bytes, slots: 1, layout: Layout::Linear, private: false }
}

/// A ring of `slots` slots of `slot_bytes` bytes.
pub fn ring_at(b: &mut FnBuilder, name: &str, base: i64, slot_bytes: i64, slots: u32) -> Shared {
    b.lds_ring(name, base, slot_bytes, slots);
    Shared { name: name.to_string(), base, slot_stride: slot_bytes, bytes: slot_bytes, slots, layout: Layout::Linear, private: false }
}

/// A per-wave private region: `waves` slices of `per_wave` bytes (`.mine(wave)` is the wave's slice).
pub fn private_at(b: &mut FnBuilder, name: &str, base: i64, per_wave: i64, waves: u32) -> Shared {
    b.lds_private(name, base, per_wave, waves);
    Shared { name: name.to_string(), base, slot_stride: per_wave, bytes: per_wave, slots: waves, layout: Layout::Linear, private: true }
}

/// A thread-OWNED tile (`lds_owned`): thread `t` owns the bytes `base + t * col_stride + r * row_stride + [0, col_stride)`. A scope whose every access is provably at such an address is an owner event: a thread's own
/// write -> read (the next version, in place) needs no barrier, other threads' columns stay ordinary shared accesses.
pub fn owned_at(b: &mut FnBuilder, name: &str, base: i64, bytes: i64, col_stride: i64, row_stride: i64) -> Shared {
    b.lds_owned(name, base, bytes, col_stride, row_stride);
    Shared { name: name.to_string(), base, slot_stride: bytes, bytes, slots: 1, layout: Layout::Linear, private: false }
}
