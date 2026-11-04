//! Bit-support analyses shared by the layout algebra (`Ix::bits`, `Ix::bitform`) and the GPU-semantics verifier (possible bits of i32 values).
//!
//! * `upto(h)`: the possible bits of any value in `[0, h]` (all bits up to the highest bit of `h`);
//! * `add32` / `mul32`: possible bits of a 32-bit sum / product of two operands with known possible bits (disjoint sums cannot carry;
//!   otherwise everything up to the bound of the arithmetic result; overflow of 32 bits = unknown);
//! * `itv_bits32`: the possible bits implied by an interval (a provably non-negative value bounded by a constant);
//! * `BitForm`: the GF(2)-linear normal form of one-symbol index expressions (see below).

use crate::{Ctx, Itv};

/// Possible bits of any value in `[0, h]` (`h <= 0`: none).
pub fn upto(h: u64) -> u64 {
    if h == 0 {
        0
    } else {
        let n = 64 - h.leading_zeros();
        if n >= 64 {
            u64::MAX
        } else {
            (1u64 << n) - 1
        }
    }
}

/// Possible bits of a 32-bit result bounded by `s` (`u32::MAX` when `s` does not fit).
pub fn upto32(s: u64) -> u32 {
    if s > u32::MAX as u64 {
        u32::MAX
    } else {
        (upto(s).min(u32::MAX as u64)) as u32
    }
}

/// Possible bits of `a + b` (32-bit): disjoint operands cannot carry.
pub fn add32(ma: u32, mb: u32) -> u32 {
    if ma & mb == 0 {
        ma | mb
    } else {
        upto32(ma as u64 + mb as u64)
    }
}

/// Possible bits of `a * b` (32-bit).
pub fn mul32(ma: u32, mb: u32) -> u32 {
    upto32((ma as u64).saturating_mul(mb as u64))
}

/// Possible bits implied by an interval: a provably non-negative value whose tightest constant upper bound is `h` has no bit above `h`'s.
pub fn itv_bits32(i: &Itv, cx: &Ctx) -> u32 {
    if !i.nonneg(cx) {
        return u32::MAX;
    }
    match i.hi.iter().filter_map(|p| p.floor_const()).min() {
        Some(h) if (0..=i32::MAX as i128).contains(&h) => upto(h as u64) as u32,
        _ => u32::MAX,
    }
}

// Bit-field normal form of index expressions in ONE non-negative symbol.
//
// Lane / thread index math built from power-of-two strides, divisions, remainders, masks and XOR swizzles is a GF(2)-linear map of the symbol's bits
// (plus constant bits): output bit j = c_j XOR (XOR of symbol bits in `bits[j]`). Sums are accepted only when the summands' possible bits are disjoint
// (no carries), so the form is exact. The form is lowered as XOR over "shift groups": every (symbol bit i -> output bit j) pair with the same
// distance j - i forms one `(x & M) << d` (or `>> -d`) term - typically 1-3 terms for a whole DMA offset, instead of a chain of shift / and / xor / add
// per field.


/// `value = konst XOR (XOR over j: bit j set iff the parity of (x & bits[j]) is odd)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitForm {
    /// the symbol (None: a pure constant)
    pub sym: Option<u32>,
    /// width (bits) of the symbol
    pub width: u32,
    /// per output bit j (0..32): the symbol bits XORed into it
    pub bits: [u32; 32],
    pub konst: u32,
}

/// One shift group: `(x & mask)` shifted left by `shift` (negative: logical right).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub mask: u32,
    pub shift: i32,
}

impl BitForm {
    pub fn konst(k: u32) -> BitForm {
        BitForm { sym: None, width: 0, bits: [0; 32], konst: k }
    }
    /// Output bits that can be 1.
    pub fn support(&self) -> u32 {
        let mut s = self.konst;
        for (j, m) in self.bits.iter().enumerate() {
            if *m != 0 {
                s |= 1 << j;
            }
        }
        s
    }
    pub fn join_sym(a: Option<u32>, b: Option<u32>) -> Option<Option<u32>> {
        match (a, b) {
            (None, x) | (x, None) => Some(x),
            (Some(x), Some(y)) if x == y => Some(Some(x)),
            _ => None,
        }
    }
    pub fn xor(&self, o: &BitForm) -> Option<BitForm> {
        let sym = Self::join_sym(self.sym, o.sym)?;
        let mut r = BitForm { sym, width: self.width.max(o.width), bits: [0; 32], konst: self.konst ^ o.konst };
        for j in 0..32 {
            r.bits[j] = self.bits[j] ^ o.bits[j];
        }
        Some(r)
    }
    pub fn shl(&self, s: u32) -> Option<BitForm> {
        let mut r = BitForm { bits: [0; 32], konst: self.konst.checked_shl(s).unwrap_or(0), ..self.clone() };
        for j in 0..32usize {
            if self.bits[j] != 0 {
                let t = j + s as usize;
                if t >= 31 {
                    return None; // would reach the sign bit / wrap: not representable exactly
                }
                r.bits[t] = self.bits[j];
            }
        }
        if (self.konst as u64) << s > i32::MAX as u64 {
            return None;
        }
        Some(r)
    }
    pub fn shr(&self, s: u32) -> BitForm {
        let mut r = BitForm { bits: [0; 32], konst: self.konst >> s.min(31), ..self.clone() };
        for j in (s as usize)..32 {
            r.bits[j - s as usize] = self.bits[j];
        }
        r
    }
    pub fn and(&self, m: u32) -> BitForm {
        let mut r = BitForm { bits: [0; 32], konst: self.konst & m, ..self.clone() };
        for j in 0..32 {
            if m >> j & 1 == 1 {
                r.bits[j] = self.bits[j];
            }
        }
        r
    }

    /// The shift groups (deterministic order: by shift).
    pub fn groups(&self) -> Vec<Group> {
        let mut g: Vec<Group> = Vec::new();
        for (j, m) in self.bits.iter().enumerate() {
            for i in 0..32 {
                if m >> i & 1 == 1 {
                    let d = j as i32 - i as i32;
                    match g.iter_mut().find(|x| x.shift == d) {
                        Some(x) => x.mask |= 1 << i,
                        None => g.push(Group { mask: 1 << i, shift: d }),
                    }
                }
            }
        }
        g.sort_by_key(|x| x.shift);
        g
    }

    /// IR operations of the grouped lowering: per group an `and` (unless the mask covers the whole symbol) and a shift (unless 0); combining
    /// the groups; the constant.
    pub fn cost(&self) -> usize {
        let full = if self.width >= 32 { u32::MAX } else { (1u32 << self.width) - 1 };
        let g = self.groups();
        let mut c = 0;
        for x in &g {
            if x.mask != full {
                c += 1;
            }
            if x.shift != 0 {
                c += 1;
            }
        }
        c += g.len().saturating_sub(1);
        if self.konst != 0 && !g.is_empty() {
            c += 1;
        }
        c
    }

    /// Evaluates the form at a symbol value (for tests).
    pub fn eval(&self, x: u32) -> u32 {
        let mut v = self.konst;
        for (j, m) in self.bits.iter().enumerate() {
            if (x & m).count_ones() % 2 == 1 {
                v ^= 1 << j;
            }
        }
        v
    }
}

