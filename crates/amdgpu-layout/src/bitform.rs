//! The layout side of the bit-field normal form: `Ix::bitform` builds `amdgpu_sym::bits::BitForm` from an index expression (the form, its
//! shift groups and cost live in the shared `amdgpu-sym` crate).

pub use amdgpu_sym::bits::{BitForm, Group};

use crate::ix::{Dyns, Ix};

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
