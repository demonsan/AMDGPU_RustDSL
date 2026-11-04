//! Bit-field normal form of index expressions in ONE non-negative symbol.
//!
//! Lane / thread index math built from power-of-two strides, divisions, remainders, masks and XOR swizzles is a GF(2)-linear map of the symbol's bits
//! (plus constant bits): output bit j = c_j XOR (XOR of symbol bits in `bits[j]`). Sums are accepted only when the summands' possible bits are disjoint
//! (no carries), so the form is exact. The form is lowered as XOR over "shift groups": every (symbol bit i -> output bit j) pair with the same
//! distance j - i forms one `(x & M) << d` (or `>> -d`) term - typically 1-3 terms for a whole DMA offset, instead of a chain of shift / and / xor / add
//! per field.

use crate::ix::{Dyns, Ix};

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
    fn konst(k: u32) -> BitForm {
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
    fn join_sym(a: Option<u32>, b: Option<u32>) -> Option<Option<u32>> {
        match (a, b) {
            (None, x) | (x, None) => Some(x),
            (Some(x), Some(y)) if x == y => Some(Some(x)),
            _ => None,
        }
    }
    fn xor(&self, o: &BitForm) -> Option<BitForm> {
        let sym = Self::join_sym(self.sym, o.sym)?;
        let mut r = BitForm { sym, width: self.width.max(o.width), bits: [0; 32], konst: self.konst ^ o.konst };
        for j in 0..32 {
            r.bits[j] = self.bits[j] ^ o.bits[j];
        }
        Some(r)
    }
    fn shl(&self, s: u32) -> Option<BitForm> {
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
    fn shr(&self, s: u32) -> BitForm {
        let mut r = BitForm { bits: [0; 32], konst: self.konst >> s.min(31), ..self.clone() };
        for j in (s as usize)..32 {
            r.bits[j - s as usize] = self.bits[j];
        }
        r
    }
    fn and(&self, m: u32) -> BitForm {
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

impl Ix {
    /// The bit-field normal form, when the expression is exactly a GF(2)-linear map of ONE non-negative symbol (see the module doc).
    pub fn bitform(&self, d: &Dyns) -> Option<BitForm> {
        let pow2 = |k: i64| -> Option<u32> { (k > 0 && (k as u64).is_power_of_two()).then(|| (k as u64).trailing_zeros()) };
        match self {
            Ix::C(k) => (0..=i32::MAX as i64).contains(k).then(|| BitForm::konst(*k as u32)),
            Ix::Sym(s) => {
                let (lo, hi) = self.range(d);
                if lo < 0 || hi > i32::MAX as i64 {
                    return None;
                }
                let width = 64 - (hi as u64).leading_zeros();
                let mut bits = [0u32; 32];
                for (j, b) in bits.iter_mut().enumerate().take(width as usize) {
                    *b = 1 << j;
                }
                Some(BitForm { sym: Some(*s), width, bits, konst: 0 })
            }
            Ix::Sum(t, k) => {
                let mut acc = BitForm::konst(0);
                let mut parts: Vec<BitForm> = Vec::new();
                for (y, m) in t {
                    parts.push(y.bitform(d)?.shl(pow2(*m)?)?);
                }
                if *k != 0 {
                    parts.push(Ix::C(*k).bitform(d)?);
                }
                let mut seen = 0u32;
                for p in parts {
                    let s = p.support();
                    if s & seen != 0 {
                        return None; // a carry is possible
                    }
                    seen |= s;
                    acc = acc.xor(&p)?;
                }
                Some(acc)
            }
            Ix::Mul(a, b) => match (a.as_ref(), b.as_ref()) {
                (Ix::C(k), y) | (y, Ix::C(k)) => y.bitform(d)?.shl(pow2(*k)?),
                _ => None,
            },
            Ix::Div(a, k) => Some(a.bitform(d)?.shr(pow2(*k)?)),
            Ix::Mod(a, k) => {
                let p = pow2(*k)?;
                Some(a.bitform(d)?.and(((1u64 << p) - 1) as u32))
            }
            Ix::Xor(a, b) => a.bitform(d)?.xor(&b.bitform(d)?),
            Ix::And(a, m) => (*m >= 0).then(|| ()).and_then(|_| Some(a.bitform(d)?.and(*m as u32))),
            Ix::Shl(a, s) => a.bitform(d)?.shl(*s),
            Ix::Shr(a, s) => Some(a.bitform(d)?.shr(*s)),
        }
    }

    /// IR operations of the structural lowering (shift / and / mul / add / xor per node; leaves free).
    pub fn op_cost(&self) -> usize {
        match self {
            Ix::C(_) | Ix::Sym(_) => 0,
            Ix::Sum(t, k) => {
                let mut c = t.len().saturating_sub(1) + usize::from(*k != 0);
                for (y, m) in t {
                    c += y.op_cost() + usize::from(*m != 1);
                }
                c
            }
            Ix::Mul(a, b) | Ix::Xor(a, b) => 1 + a.op_cost() + b.op_cost(),
            Ix::Div(a, _) | Ix::Mod(a, _) | Ix::And(a, _) | Ix::Shl(a, _) | Ix::Shr(a, _) => 1 + a.op_cost(),
        }
    }
}
