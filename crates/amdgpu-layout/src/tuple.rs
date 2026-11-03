//! Nested integer tuples (CuTe `IntTuple`): a leaf is an [`Ix`] (static constant or dynamic expression), a node is a tuple of IntTuples.

use crate::ix::Ix;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IntTuple {
    L(Ix),
    T(Vec<IntTuple>),
}

/// Leaf constructor helpers: `t(8)` is a static leaf; `tup![..]` builds nested tuples.
pub fn leaf(x: impl Into<Ix>) -> IntTuple {
    IntTuple::L(x.into())
}

impl From<i64> for IntTuple {
    fn from(v: i64) -> IntTuple {
        IntTuple::L(Ix::C(v))
    }
}
impl From<i32> for IntTuple {
    fn from(v: i32) -> IntTuple {
        IntTuple::L(Ix::C(v as i64))
    }
}
impl From<Ix> for IntTuple {
    fn from(v: Ix) -> IntTuple {
        IntTuple::L(v)
    }
}
impl From<Vec<IntTuple>> for IntTuple {
    fn from(v: Vec<IntTuple>) -> IntTuple {
        IntTuple::T(v)
    }
}

/// `tup![a, b, ...]`: an IntTuple from elements convertible into IntTuple (`tup![8, tup![4, 2]]` = `(8, (4, 2))`).
#[macro_export]
macro_rules! tup {
    ($($e:expr),* $(,)?) => { $crate::IntTuple::T(vec![$($crate::IntTuple::from($e)),*]) };
}

impl IntTuple {
    pub fn is_leaf(&self) -> bool {
        matches!(self, IntTuple::L(_))
    }
    /// Number of top-level modes (a leaf has rank 1).
    pub fn rank(&self) -> usize {
        match self {
            IntTuple::L(_) => 1,
            IntTuple::T(v) => v.len(),
        }
    }
    /// Nesting depth (a leaf has depth 0).
    pub fn depth(&self) -> usize {
        match self {
            IntTuple::L(_) => 0,
            IntTuple::T(v) => 1 + v.iter().map(|x| x.depth()).max().unwrap_or(0),
        }
    }
    /// The i-th top-level mode (a leaf is its own mode 0).
    pub fn get(&self, i: usize) -> IntTuple {
        match self {
            IntTuple::L(_) => {
                assert_eq!(i, 0, "mode {i} of a leaf");
                self.clone()
            }
            IntTuple::T(v) => v[i].clone(),
        }
    }
    pub fn modes(&self) -> Vec<IntTuple> {
        match self {
            IntTuple::L(_) => vec![self.clone()],
            IntTuple::T(v) => v.clone(),
        }
    }
    /// All leaves, depth first.
    pub fn flatten(&self) -> Vec<Ix> {
        match self {
            IntTuple::L(x) => vec![x.clone()],
            IntTuple::T(v) => v.iter().flat_map(|x| x.flatten()).collect(),
        }
    }
    /// Product of all leaves.
    pub fn product(&self) -> Ix {
        self.flatten().iter().fold(Ix::C(1), |a, x| a.mul(x))
    }
    /// Per-top-level-mode products.
    pub fn product_each(&self) -> IntTuple {
        IntTuple::T(self.modes().iter().map(|m| IntTuple::L(m.product())).collect())
    }
    /// Is every leaf static?
    pub fn is_static(&self) -> bool {
        self.flatten().iter().all(|x| x.is_static())
    }
    /// Same tree with the leaves replaced in depth-first order.
    pub fn with_leaves(&self, it: &mut dyn Iterator<Item = Ix>) -> IntTuple {
        match self {
            IntTuple::L(_) => IntTuple::L(it.next().expect("leaf count")),
            IntTuple::T(v) => IntTuple::T(v.iter().map(|x| x.with_leaves(it)).collect()),
        }
    }
    /// Do two tuples have the same tree shape?
    pub fn congruent(&self, o: &IntTuple) -> bool {
        match (self, o) {
            (IntTuple::L(_), IntTuple::L(_)) => true,
            (IntTuple::T(a), IntTuple::T(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.congruent(y)),
            _ => false,
        }
    }
    pub fn append(&self, x: IntTuple) -> IntTuple {
        let mut v = self.modes();
        v.push(x);
        IntTuple::T(v)
    }
    pub fn prepend(&self, x: IntTuple) -> IntTuple {
        let mut v = vec![x];
        v.extend(self.modes());
        IntTuple::T(v)
    }
    pub fn select(&self, idx: &[usize]) -> IntTuple {
        let m = self.modes();
        IntTuple::T(idx.iter().map(|&i| m[i].clone()).collect())
    }
    /// Groups the modes `[b, e)` into one nested mode.
    pub fn group(&self, b: usize, e: usize) -> IntTuple {
        let m = self.modes();
        let mut v: Vec<IntTuple> = m[..b].to_vec();
        v.push(IntTuple::T(m[b..e].to_vec()));
        v.extend_from_slice(&m[e..]);
        IntTuple::T(v)
    }
    /// Elementwise op on congruent tuples (or a leaf against a tuple: broadcast).
    pub fn zip_with(&self, o: &IntTuple, f: &dyn Fn(&Ix, &Ix) -> Ix) -> IntTuple {
        match (self, o) {
            (IntTuple::L(a), IntTuple::L(b)) => IntTuple::L(f(a, b)),
            (IntTuple::T(a), IntTuple::T(b)) => {
                assert_eq!(a.len(), b.len(), "zip of tuples of different rank");
                IntTuple::T(a.iter().zip(b).map(|(x, y)| x.zip_with(y, f)).collect())
            }
            (IntTuple::T(a), IntTuple::L(_)) => IntTuple::T(a.iter().map(|x| x.zip_with(o, f)).collect()),
            (IntTuple::L(_), IntTuple::T(b)) => IntTuple::T(b.iter().map(|y| self.zip_with(y, f)).collect()),
        }
    }
    /// The leaf value; panics for a tuple.
    pub fn ix(&self) -> &Ix {
        match self {
            IntTuple::L(x) => x,
            IntTuple::T(_) => panic!("expected a leaf, found the tuple {self}"),
        }
    }
}

impl fmt::Display for IntTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntTuple::L(x) => write!(f, "{x}"),
            IntTuple::T(v) => {
                write!(f, "(")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{x}")?;
                }
                write!(f, ")")
            }
        }
    }
}
