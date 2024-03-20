pub trait Projection<Idx, T> {
    /// Computes the probability projection of `self`.
    fn projection(&self) -> T;
}

pub trait MaxUncertainty<Idx, V, T> {
    type Output;

    /// Returns the uncertainty maximized opinion of `self`.
    fn max_uncertainty(&self, a: &T) -> V;

    /// Returns the uncertainty maximized opinion of `self`.
    fn uncertainty_maximized(&self, a: &T) -> Self::Output;
}

pub trait Discount<T, V> {
    type Output;
    /// Computes trust discounting of `self` with a referral trust `t`.
    fn discount(&self, t: V) -> Self::Output;
}

#[derive(Clone, Copy)]
pub enum FuseOp {
    /// Aleatory cumulative fusion operator
    ACm,
    /// Epistemic cumulative fusion operator
    ECm,
    /// Averaging belief fusion of operator
    Avg,
    /// Weighted belief fusion of operator
    Wgh,
}

pub trait Fuse<L, R, Idx> {
    type Output;
    fn fuse(&self, lhs: L, rhs: R) -> Self::Output;
}

pub trait FuseAssign<L, R, Idx> {
    fn fuse_assign(&self, lhs: &mut L, rhs: R);
}

/// The deduction operator.
pub trait Deduction<X, Y, Cond, U>: Sized {
    type Output;

    /// Computes the conditionally deduced opinion of `self` with
    /// a the marginal base rate (MBR) given by `conds` representing a collection of conditional opinions.
    /// If all conditional opinions are vacuous, i.e. \\(\forall x. u_{Y|x} = 1\\),
    /// then MBR cannot be determined so return None (vacuous opinion is deduced).
    fn deduce(self, conds: Cond) -> Option<Self::Output>;

    /// Computes the conditionally deduced opinion of `self` with a base rate `ay`.
    fn deduce_with(self, conds: Cond, ay: U) -> Self::Output;
}

/// The abduction operator.
pub trait Abduction<Cond, X, Y, T, U> {
    type Output;

    /// Computes the conditionally abduced opinion of `self` with a base rate vector `ax`
    /// by `conds` representing a collection of conditional opinions.
    /// If a marginal base rate cannot be computed from `conds`, return `None`.
    fn abduce(self, conds: Cond, ax: T) -> Option<Self::Output>;
    fn abduce_with(self, conds: Cond, ax: T, ay: &U) -> Self::Output;
}

pub trait Product2<T0, T1> {
    fn product2(t0: T0, t1: T1) -> Self;
}

pub trait Product3<T0, T1, T2> {
    fn product3(t0: T0, t1: T1, t2: T2) -> Self;
}

use std::{
    array,
    ops::{Index, IndexMut},
};

pub trait Keys<I> {
    type Iter: Iterator<Item = I> + Clone;
    fn keys() -> Self::Iter;
}

pub trait IndexedContainer<K>: Index<K> + IndexMut<K> {
    type Map<U>: Index<K, Output = U>;

    fn keys() -> impl Iterator<Item = K>;
    fn map<U, F: FnMut(K) -> U>(f: F) -> Self::Map<U>;
    fn from_fn<F: FnMut(K) -> <Self as Index<K>>::Output>(f: F) -> Self;
}

impl<T, const N: usize> IndexedContainer<usize> for [T; N] {
    type Map<U> = [U; N];

    fn keys() -> impl Iterator<Item = usize> {
        0..N
    }

    fn map<U, F: FnMut(usize) -> U>(mut f: F) -> Self::Map<U> {
        array::from_fn(|i| f(i))
    }

    fn from_fn<F: FnMut(usize) -> T>(mut f: F) -> Self {
        array::from_fn(|i| f(i))
    }
}
