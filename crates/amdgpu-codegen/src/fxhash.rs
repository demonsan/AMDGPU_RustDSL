//! A small deterministic hasher (FxHash) for the compiler's integer-keyed maps: the standard SipHash costs more than the map operations it guards in the register allocator's analyses.
//! Iteration order of a `HashMap` is unspecified either way (std's `RandomState` randomises it per process), so no code may depend on it; swapping the hasher changes no output.

use std::hash::Hasher;

#[derive(Default, Clone, Copy)]
pub struct Fx(u64);

const K: u64 = 0x517c_c1b7_2722_0a95;

impl Fx {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(K);
    }
}

impl Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            self.add(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}