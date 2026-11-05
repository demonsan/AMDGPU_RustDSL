//! Layouts `Shape : Stride` and the CuTe / FlyDSL layout algebra.
//!
//! Static vs dynamic: every leaf is an [`Ix`]. The ALGEBRA (composition of the right operand, complement, inverses, products, divides by layout tilers)
//! requires static SHAPES on the modes it splits; strides of the left operand of a composition may be dynamic (they are only multiplied), and dynamic
//! outer modes are supported by the by-mode divides (`logical_divide_bymode` / `zipped_divide_bymode` / `local_tile`) - the CuTe practice "tile-internal modes
//! static, the rest modes may be dynamic". Violations panic with the offending layout.

use crate::ix::{Dyns, Ix};
use crate::tuple::IntTuple;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    pub shape: IntTuple,
    pub stride: IntTuple,
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}

/// `shape : stride` (congruent trees).
pub fn make_layout(shape: impl Into<IntTuple>, stride: impl Into<IntTuple>) -> Layout {
    let (shape, stride) = (shape.into(), stride.into());
    assert!(shape.congruent(&stride), "make_layout: shape {shape} and stride {stride} are not congruent");
    Layout { shape, stride }
}

/// Compact column-major (colexicographic) layout of a shape: stride of mode i = product of the earlier modes.
pub fn make_layout_shape(shape: impl Into<IntTuple>) -> Layout {
    let shape = shape.into();
    let mut acc = Ix::C(1);
    let mut strides: Vec<Ix> = Vec::new();
    for s in shape.flatten() {
        strides.push(acc.clone());
        acc = acc.mul(&s);
    }
    let stride = shape.with_leaves(&mut strides.into_iter());
    Layout { shape, stride }
}

/// Compact layout whose modes are ordered by `order` (order[i] = rank of mode i in the stride order: `(1, 0)` = row-major for rank 2).
pub fn make_ordered_layout(shape: impl Into<IntTuple>, order: &[usize]) -> Layout {
    let shape = shape.into();
    let m = shape.modes();
    assert_eq!(m.len(), order.len(), "make_ordered_layout: one order entry per mode");
    let mut idx: Vec<usize> = (0..m.len()).collect();
    idx.sort_by_key(|&i| order[i]);
    let mut strides: Vec<IntTuple> = vec![IntTuple::L(Ix::C(0)); m.len()];
    let mut acc = Ix::C(1);
    for &i in &idx {
        let sub = make_layout_shape(m[i].clone());
        let st = sub.stride.zip_with(&IntTuple::L(acc.clone()), &|a, b| a.mul(b));
        strides[i] = st;
        acc = acc.mul(&m[i].product());
    }
    Layout { shape: IntTuple::T(m), stride: IntTuple::T(strides) }
}

/// `idx2crd(idx, shape)`: colexicographic coordinate (every mode but the last is taken modulo its size; static shapes there).
pub fn idx2crd(idx: &Ix, shape: &IntTuple, d: &Dyns) -> IntTuple {
    match shape {
        IntTuple::L(_) => IntTuple::L(idx.clone()),
        IntTuple::T(v) => {
            let mut out = Vec::with_capacity(v.len());
            let mut div = Ix::C(1);
            for (i, m) in v.iter().enumerate() {
                let q = match div.c() {
                    Some(k) => idx.div_c(k, d),
                    None => panic!("idx2crd: dynamic inner mode size {div} in {shape}"),
                };
                let sz = m.product();
                let c = if i + 1 == v.len() {
                    q
                } else {
                    q.mod_c(sz.c().unwrap_or_else(|| panic!("idx2crd: dynamic non-last mode {m} in {shape}")), d)
                };
                out.push(idx2crd(&c, m, d));
                div = div.mul(&sz);
            }
            IntTuple::T(out)
        }
    }
}

/// `crd2idx(crd, shape, stride)`: a coordinate (congruent, or a 1-D index for a hierarchical mode) to an index.
pub fn crd2idx(crd: &IntTuple, shape: &IntTuple, stride: &IntTuple, d: &Dyns) -> Ix {
    match (crd, shape, stride) {
        (IntTuple::L(c), IntTuple::L(_), IntTuple::L(s)) => c.mul(s),
        (IntTuple::L(c), IntTuple::T(_), IntTuple::T(_)) => crd2idx(&idx2crd(c, shape, d), shape, stride, d),
        (IntTuple::T(cs), IntTuple::T(ss), IntTuple::T(ds)) => {
            assert_eq!(cs.len(), ss.len(), "crd2idx: coordinate {crd} does not match shape {shape}");
            cs.iter().zip(ss).zip(ds).fold(Ix::C(0), |acc, ((c, s), t)| acc.add(&crd2idx(c, s, t, d)))
        }
        _ => panic!("crd2idx: coordinate {crd} does not match shape {shape}"),
    }
}

impl Layout {
    pub fn size(&self) -> Ix {
        self.shape.product()
    }
    /// 1 + the largest index (for non-negative strides).
    pub fn cosize(&self) -> Ix {
        let mut acc = Ix::C(1);
        for (s, t) in self.flat_modes() {
            acc = acc.add(&s.add_c(-1).mul(&t));
        }
        acc
    }
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }
    pub fn depth(&self) -> usize {
        self.shape.depth()
    }
    /// Top-level mode `i` as a layout.
    pub fn get(&self, i: usize) -> Layout {
        Layout { shape: self.shape.get(i), stride: self.stride.get(i) }
    }
    pub fn modes(&self) -> Vec<Layout> {
        (0..self.rank()).map(|i| self.get(i)).collect()
    }
    /// The (shape, stride) leaves, depth first.
    pub fn flat_modes(&self) -> Vec<(Ix, Ix)> {
        self.shape.flatten().into_iter().zip(self.stride.flatten()).collect()
    }
    /// A layout whose top-level modes are `ms`.
    pub fn tuple(ms: Vec<Layout>) -> Layout {
        Layout { shape: IntTuple::T(ms.iter().map(|m| m.shape.clone()).collect()), stride: IntTuple::T(ms.into_iter().map(|m| m.stride).collect()) }
    }
    fn from_flat(ms: Vec<(Ix, Ix)>) -> Layout {
        match ms.len() {
            0 => make_layout(1, 0),
            1 => {
                let (s, t) = ms.into_iter().next().expect("one");
                Layout { shape: IntTuple::L(s), stride: IntTuple::L(t) }
            }
            _ => Layout { shape: IntTuple::T(ms.iter().map(|m| IntTuple::L(m.0.clone())).collect()), stride: IntTuple::T(ms.into_iter().map(|m| IntTuple::L(m.1)).collect()) },
        }
    }
    /// `L(crd)`.
    pub fn crd2idx(&self, crd: &IntTuple, d: &Dyns) -> Ix {
        crd2idx(crd, &self.shape, &self.stride, d)
    }
    /// `L(i)` for a 1-D index.
    pub fn call(&self, i: &Ix, d: &Dyns) -> Ix {
        crd2idx(&IntTuple::L(i.clone()), &self.shape, &self.stride, d)
    }
    /// `L(i)` of a fully static layout.
    pub fn at(&self, i: i64) -> i64 {
        self.call(&Ix::C(i), &Dyns::new()).st()
    }
    pub fn is_static(&self) -> bool {
        self.shape.is_static() && self.stride.is_static()
    }
    pub fn append(&self, m: &Layout) -> Layout {
        let mut ms = self.modes_or_self();
        ms.push(m.clone());
        Layout::tuple(ms)
    }
    pub fn prepend(&self, m: &Layout) -> Layout {
        let mut ms = vec![m.clone()];
        ms.extend(self.modes_or_self());
        Layout::tuple(ms)
    }
    fn modes_or_self(&self) -> Vec<Layout> {
        if self.shape.is_leaf() {
            vec![self.clone()]
        } else {
            self.modes()
        }
    }
    pub fn select(&self, idx: &[usize]) -> Layout {
        Layout { shape: self.shape.select(idx), stride: self.stride.select(idx) }
    }
    pub fn group(&self, b: usize, e: usize) -> Layout {
        Layout { shape: self.shape.group(b, e), stride: self.stride.group(b, e) }
    }
    /// Pads to rank `r` with `1:0` modes.
    pub fn append_to_rank(&self, r: usize) -> Layout {
        let mut ms = self.modes_or_self();
        while ms.len() < r {
            ms.push(make_layout(1, 0));
        }
        Layout::tuple(ms)
    }
    /// Transposed rank-2 layout (modes swapped).
    pub fn transpose(&self) -> Layout {
        assert_eq!(self.rank(), 2, "transpose of a rank-{} layout", self.rank());
        self.select(&[1, 0])
    }
}

/// Merges adjacent modes `s0:d0, s1:d1` with `s0 * d0 == d1` and drops size-1 modes (the function is unchanged).
pub fn coalesce(l: &Layout) -> Layout {
    let mut out: Vec<(Ix, Ix)> = Vec::new();
    for (s, t) in l.flat_modes() {
        if s == Ix::C(1) {
            continue;
        }
        if let Some(last) = out.last_mut() {
            if last.0.mul(&last.1) == t {
                last.0 = last.0.mul(&s);
                continue;
            }
        }
        out.push((s, t));
    }
    Layout::from_flat(out)
}

/// Coalesces every top-level mode separately (the rank is kept).
pub fn coalesce_bymode(l: &Layout) -> Layout {
    if l.shape.is_leaf() {
        return coalesce(l);
    }
    Layout::tuple(l.modes().iter().map(coalesce).collect())
}

/// Removes stride-0 and size-1 modes.
pub fn filter(l: &Layout) -> Layout {
    let ms: Vec<(Ix, Ix)> = l.flat_modes().into_iter().filter(|(s, t)| *s != Ix::C(1) && *t != Ix::C(0)).collect();
    coalesce(&Layout::from_flat(ms))
}

/// CuTe `shape_div`: a / b if b divides a, else 1 when a divides b (sign kept); other pairs are not divisible.
fn shape_div(a: i64, b: i64) -> i64 {
    assert!(b != 0, "shape_div by 0");
    if a % b == 0 {
        a / b
    } else if b % a == 0 {
        a.signum() * b.signum()
    } else {
        panic!("layout algebra: {a} and {b} are not divisible either way (the composition is not representable)")
    }
}

/// `composition(A, B)`: `(A o B)(x) = A(B(x))`, with B's hierarchy. B static; A's shapes static on the modes B reaches, A's strides may be dynamic.
pub fn composition(a: &Layout, b: &Layout) -> Layout {
    if !b.shape.is_leaf() {
        return Layout::tuple(b.modes().iter().map(|m| composition(a, m)).collect());
    }
    let s = b.shape.ix().st();
    let dstride = b.stride.ix().st();
    if dstride == 0 {
        return make_layout(s, 0);
    }
    let am = coalesce(a).flat_modes();
    let mut res: Vec<(Ix, Ix)> = Vec::new();
    let (mut rest_shape, mut rest_stride) = (s, dstride);
    for (cs, cd) in am.iter().take(am.len() - 1) {
        let cs = cs.c().unwrap_or_else(|| panic!("composition: dynamic inner mode {cs} of {a}"));
        let next_shape = shape_div(cs, rest_stride.abs());
        let next_stride = shape_div(rest_stride.abs(), cs);
        if next_shape == 1 || rest_shape == 1 {
            rest_stride = next_stride;
            continue;
        }
        let new_shape = next_shape.min(rest_shape);
        assert!(rest_shape % new_shape == 0, "composition: {rest_shape} is not divisible by {new_shape} ({a} o {b})");
        res.push((Ix::C(new_shape), cd.mul_c(rest_stride)));
        rest_shape /= new_shape;
        rest_stride = next_stride;
    }
    let last_d = &am[am.len() - 1].1;
    res.push((Ix::C(rest_shape), last_d.mul_c(rest_stride)));
    coalesce(&Layout::from_flat(res))
}

/// `complement(A, m)`: the layout of the positions in `[0, m)` that A does not reach (A injective, static).
pub fn complement(a: &Layout, m: i64) -> Layout {
    let mut ms: Vec<(i64, i64)> = filter(a).flat_modes().into_iter().filter(|(s, t)| *s != Ix::C(1) && *t != Ix::C(0)).map(|(s, t)| (s.st(), t.st())).collect();
    ms.sort_by_key(|x| x.1);
    let mut res: Vec<(Ix, Ix)> = Vec::new();
    let mut cur = 1i64;
    for (s, t) in ms {
        assert!(t % cur == 0, "complement: stride {t} is not a multiple of {cur} in {a} (not a complementable layout)");
        res.push((Ix::C(t / cur), Ix::C(cur)));
        cur = s * t;
    }
    res.push((Ix::C((m + cur - 1) / cur), Ix::C(cur)));
    coalesce(&Layout::from_flat(res))
}

/// `right_inverse(L)`: `L(R(i)) = i` for `i < size(R)` (static).
pub fn right_inverse(l: &Layout) -> Layout {
    let c = coalesce(l);
    let fm: Vec<(i64, i64)> = c.flat_modes().iter().map(|(s, t)| (s.st(), t.st())).collect();
    let mut dom = Vec::with_capacity(fm.len());
    let mut acc = 1i64;
    for (s, _) in &fm {
        dom.push(acc);
        acc *= s;
    }
    let mut idx: Vec<usize> = (0..fm.len()).filter(|&k| fm[k].1 != 0).collect();
    idx.sort_by_key(|&k| fm[k].1);
    let mut res: Vec<(Ix, Ix)> = Vec::new();
    let mut cur = 1i64;
    for k in idx {
        if fm[k].1 != cur {
            break;
        }
        res.push((Ix::C(fm[k].0), Ix::C(dom[k])));
        cur = fm[k].0 * fm[k].1;
    }
    coalesce(&Layout::from_flat(res))
}

/// `left_inverse(L)`: `R(L(i)) = i` (static, injective L).
pub fn left_inverse(l: &Layout) -> Layout {
    let co = l.cosize().st();
    right_inverse(&Layout::tuple(vec![l.clone(), complement(l, co)]))
}

/// `logical_divide(A, B) = A o (B, complement(B, size(A)))`: mode 0 = the tile B, mode 1 = the tiles' positions.
pub fn logical_divide(a: &Layout, b: &Layout) -> Layout {
    let n = a.size().st();
    composition(a, &Layout::tuple(vec![b.clone(), complement(b, n)]))
}

/// By-mode divide: mode i of A divided by `tiles[i]` (a static layout); a leaf mode of A divided by a contiguous tile `t:1` may have a DYNAMIC size and
/// stride: the result is `(t : d, ceil(s / t) : t*d)`.
pub fn logical_divide_bymode(a: &Layout, tiles: &[Layout], d: &Dyns) -> Layout {
    assert_eq!(a.rank(), tiles.len(), "logical_divide_bymode: one tiler per mode of {a}");
    let am = if a.shape.is_leaf() { vec![a.clone()] } else { a.modes() };
    Layout::tuple(
        am.iter()
            .zip(tiles)
            .map(|(m, t)| {
                if t.shape.is_leaf() && t.stride == IntTuple::L(Ix::C(1)) && m.shape.is_leaf() && (!m.is_static()) {
                    let ts = t.shape.ix().st();
                    let (s, st) = (m.shape.ix(), m.stride.ix());
                    let rest = s.add_c(ts - 1).div_c(ts, d);
                    Layout::tuple(vec![make_layout(ts, st.clone()), make_layout(rest, st.mul_c(ts))])
                } else {
                    logical_divide(m, t)
                }
            })
            .collect(),
    )
}

/// `((tile modes...), (rest modes...))`.
pub fn zipped_divide_bymode(a: &Layout, tiles: &[Layout], d: &Dyns) -> Layout {
    let ld = logical_divide_bymode(a, tiles, d);
    let ms = ld.modes();
    Layout::tuple(vec![Layout::tuple(ms.iter().map(|m| m.get(0)).collect()), Layout::tuple(ms.iter().map(|m| m.get(1)).collect())])
}

/// Contiguous tiler of a tile shape: `t_i : 1` per mode.
pub fn tiler(shape: &[i64]) -> Vec<Layout> {
    shape.iter().map(|&t| make_layout(t, 1)).collect()
}

/// `local_tile(A, tile, crd)`: the tile at tile-coordinate `crd` (one entry per mode of A): (tile layout, offset of its first element).
pub fn local_tile(a: &Layout, tile: &[i64], crd: &[Ix], d: &Dyns) -> (Layout, Ix) {
    let z = zipped_divide_bymode(a, &tiler(tile), d);
    let rest = z.get(1);
    let off = rest.crd2idx(&IntTuple::T(crd.iter().map(|c| IntTuple::L(c.clone())).collect()), d);
    (z.get(0), off)
}

/// `slice(L, crd)`: `crd[i] = Some(c)` fixes top-level mode i at `c` (it disappears), `None` keeps it (CuTe `_`). Returns (layout of the kept modes, offset).
pub fn slice(l: &Layout, crd: &[Option<IntTuple>], d: &Dyns) -> (Layout, Ix) {
    assert_eq!(crd.len(), l.rank(), "slice: one coordinate (or None) per mode of {l}");
    let mut kept = Vec::new();
    let mut off = Ix::C(0);
    for (i, c) in crd.iter().enumerate() {
        let m = l.get(i);
        match c {
            None => kept.push(m),
            Some(c) => off = off.add(&m.crd2idx(c, d)),
        }
    }
    let lay = if kept.len() == 1 { kept.pop().expect("one") } else { Layout::tuple(kept) };
    (lay, off)
}

/// `dice(L, keep)`: the layout restricted to the kept top-level modes (no offset).
pub fn dice(l: &Layout, keep: &[usize]) -> Layout {
    l.select(keep)
}

/// `logical_product(A, B) = (A, complement(A, size(A) * cosize(B)) o B)`.
pub fn logical_product(a: &Layout, b: &Layout) -> Layout {
    let n = a.size().st() * b.cosize().st();
    Layout::tuple(vec![a.clone(), composition(&complement(a, n), b)])
}
/// Rank-wise zipped product halves: (A', B'') with A' = A padded to rank R and B'' = complement o B' (rank R).
fn product_halves(a: &Layout, b: &Layout) -> (Vec<Layout>, Vec<Layout>) {
    let r = a.rank().max(b.rank());
    let (ap, bp) = (a.append_to_rank(r), b.append_to_rank(r));
    let p = logical_product(&ap, &bp);
    (p.get(0).modes(), p.get(1).modes())
}
/// `blocked_product`: mode i = (A_i, B_i'): each B element is a contiguous block of A.
pub fn blocked_product(a: &Layout, b: &Layout) -> Layout {
    let (x, y) = product_halves(a, b);
    coalesce_bymode(&Layout::tuple(x.into_iter().zip(y).map(|(p, q)| Layout::tuple(vec![p, q])).collect()))
}
/// `raked_product`: mode i = (B_i', A_i): A's elements are interleaved ("raked") across the B tiles.
pub fn raked_product(a: &Layout, b: &Layout) -> Layout {
    let (x, y) = product_halves(a, b);
    coalesce_bymode(&Layout::tuple(x.into_iter().zip(y).map(|(p, q)| Layout::tuple(vec![q, p])).collect()))
}

/// `recast_layout(L, old_bits, new_bits)`: the same memory viewed with elements of `new_bits` (strides in elements of the new type).
pub fn recast_layout(l: &Layout, old_bits: u32, new_bits: u32) -> Layout {
    if old_bits == new_bits {
        return l.clone();
    }
    let ms: Vec<(Ix, Ix)> = l
        .flat_modes()
        .into_iter()
        .map(|(s, t)| {
            if new_bits > old_bits {
                assert_eq!(new_bits % old_bits, 0, "recast: {new_bits} bits is not a multiple of {old_bits}");
                let r = (new_bits / old_bits) as i64;
                match t.c() {
                    Some(0) => (s, t),
                    Some(1) => {
                        let sv = s.st();
                        assert_eq!(sv % r, 0, "recast: contiguous mode of size {sv} cannot be packed by {r}");
                        (Ix::C(sv / r), Ix::C(1))
                    }
                    Some(k) => {
                        assert_eq!(k % r, 0, "recast: stride {k} is not a multiple of {r}");
                        (s, Ix::C(k / r))
                    }
                    None => panic!("recast of a dynamic stride {t}"),
                }
            } else {
                assert_eq!(old_bits % new_bits, 0, "recast: {old_bits} bits is not a multiple of {new_bits}");
                let r = (old_bits / new_bits) as i64;
                match t.c() {
                    Some(0) => (s, t),
                    Some(1) => (s.mul_c(r), Ix::C(1)),
                    _ => (s, t.mul_c(r)),
                }
            }
        })
        .collect();
    coalesce(&Layout::from_flat(ms))
}

/// `make_layout_tv(thr, val)`: `(tiler_mn, layout_tv)` with `layout_tv(t, v)` = the linear (column-major) index in the `tiler_mn` tile of value v of thread t;
/// threads are raked over the tile (`raked_product(thr, val)`), as CuTe / FlyDSL.
pub fn make_layout_tv(thr: &Layout, val: &Layout) -> (IntTuple, Layout) {
    let mn = raked_product(thr, val);
    let tv = composition(&right_inverse(&mn), &make_layout_shape(crate::tup![thr.size(), val.size()]));
    (mn.shape.product_each(), tv)
}
