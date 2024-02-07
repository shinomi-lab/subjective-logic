pub mod op;
pub mod prod;

use approx::{ulps_eq, UlpsEq};
use num_traits::Float;
use std::{
    array,
    fmt::Display,
    iter::Sum,
    ops::{AddAssign, DivAssign, Index, IndexMut},
};

use crate::errors::{check_is_one, InvalidValueError};
use crate::{approx_ext, errors::check_unit_interval};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct SimplexBase<T, V> {
    pub belief: T,
    pub uncertainty: V,
}

impl<T: std::fmt::Debug, V: Display> Display for SimplexBase<T, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?},{}", self.belief, self.uncertainty)
    }
}

impl<T, V> SimplexBase<T, V> {
    pub fn new_unchecked(b: T, u: V) -> Self {
        Self {
            belief: b,
            uncertainty: u,
        }
    }

    #[inline]
    pub fn b(&self) -> &T {
        &self.belief
    }

    #[inline]
    pub fn u(&self) -> &V {
        &self.uncertainty
    }
}

/// The generlized structure of a multinomial opinion.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct OpinionBase<S, T> {
    pub simplex: S,
    pub base_rate: T,
}

impl<S, T> OpinionBase<S, T> {
    pub fn from_simplex_unchecked(s: S, a: T) -> Self {
        Self {
            simplex: s,
            base_rate: a,
        }
    }

    pub fn as_ref(&self) -> OpinionBase<&S, &T> {
        OpinionBase {
            simplex: &self.simplex,
            base_rate: &self.base_rate,
        }
    }
}

impl<S: Display, T: Display> Display for OpinionBase<S, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.simplex, self.base_rate)
    }
}

pub type Opinion<T, U> = OpinionBase<SimplexBase<T, U>, T>;
pub type OpinionRef<'a, T, U> = OpinionBase<&'a SimplexBase<T, U>, &'a T>;

impl<T, U> Opinion<T, U> {
    fn new_unchecked(b: T, u: U, a: T) -> Self {
        Self {
            simplex: SimplexBase::new_unchecked(b, u),
            base_rate: a,
        }
    }

    #[inline]
    pub fn b(&self) -> &T {
        &self.simplex.belief
    }

    #[inline]
    pub fn u(&self) -> U
    where
        U: Copy,
    {
        self.simplex.uncertainty
    }
}

impl<T, U> OpinionRef<'_, T, U> {
    #[inline]
    pub fn b(&self) -> &T {
        &self.simplex.belief
    }

    #[inline]
    pub fn u(&self) -> U
    where
        U: Copy,
    {
        self.simplex.uncertainty
    }

    #[inline]
    pub fn into_opinion(&self) -> Opinion<T, U>
    where
        T: Clone,
        U: Copy,
    {
        Opinion::new_unchecked(
            (self.simplex.belief).clone(),
            self.simplex.uncertainty,
            (*self.base_rate).clone(),
        )
    }
}

impl<'a, 'b: 'a, T, U> From<(&'a SimplexBase<T, U>, &'b T)> for OpinionRef<'a, T, U> {
    fn from(value: (&'a SimplexBase<T, U>, &'b T)) -> Self {
        OpinionBase {
            simplex: &value.0,
            base_rate: &value.1,
        }
    }
}

impl<'a, T, U> From<&'a Opinion<T, U>> for OpinionRef<'a, T, U> {
    fn from(value: &'a Opinion<T, U>) -> Self {
        value.as_ref()
    }
}

impl<T, V: Float + UlpsEq> SimplexBase<T, V> {
    #[inline]
    pub fn vacuous<Idx>() -> Self
    where
        T: IndexedContainer<Idx, Output = V>,
    {
        Self {
            belief: T::from_fn(|_| V::zero()),
            uncertainty: V::one(),
        }
    }

    #[inline]
    pub fn is_vacuous(&self) -> bool {
        ulps_eq!(self.uncertainty, V::one())
    }

    #[inline]
    pub fn is_dogmatic(&self) -> bool {
        ulps_eq!(self.uncertainty, V::zero())
    }
}

impl<T, V: Float + UlpsEq> Opinion<T, V> {
    #[inline]
    pub fn is_vacuous(&self) -> bool {
        ulps_eq!(self.u(), V::one())
    }

    #[inline]
    pub fn is_dogmatic(&self) -> bool {
        ulps_eq!(self.u(), V::zero())
    }

    #[inline]
    pub fn vacuous_with<Idx>(base_rate: T) -> Self
    where
        T: IndexedContainer<Idx, Output = V>,
    {
        OpinionBase {
            simplex: SimplexBase::<T, V>::vacuous(),
            base_rate,
        }
    }
}

impl<'a, T, V: Float + UlpsEq> OpinionRef<'a, T, V> {
    #[inline]
    pub fn is_vacuous(&self) -> bool {
        self.simplex.is_vacuous()
    }

    #[inline]
    pub fn is_dogmatic(&self) -> bool {
        self.simplex.is_dogmatic()
    }
}

pub trait Projection<Idx, T> {
    /// Computes the probability projection of `self`.
    fn projection(&self) -> T;
}

impl<'a, Idx, T, V: Float + AddAssign + DivAssign> Projection<Idx, T> for OpinionRef<'a, T, V>
where
    T: IndexedContainer<Idx, Output = V>,
    Idx: Copy,
{
    fn projection(&self) -> T {
        let mut s = V::zero();
        let mut a = T::from_fn(|idx| {
            let p = self.b()[idx] + self.base_rate[idx] * self.u();
            s += p;
            p
        });
        for idx in T::keys() {
            a[idx] /= s;
        }
        a
    }
}

impl<Idx, T, V> Projection<Idx, T> for Opinion<T, V>
where
    T: IndexedContainer<Idx, Output = V>,
    Idx: Copy,
    V: Float + AddAssign + DivAssign,
{
    fn projection(&self) -> T {
        self.as_ref().projection()
    }
}

impl<T, V> SimplexBase<T, V>
where
    V: Float + AddAssign + DivAssign,
{
    pub fn projection<'a, Idx: Copy>(&'a self, a: &'a T) -> T
    where
        T: IndexedContainer<Idx, Output = V>,
    {
        OpinionRef::from((self, a)).projection()
    }
}

pub trait MaxUncertainty<Idx, V> {
    type Output;

    /// Returns the uncertainty maximized opinion of `self`.
    fn max_uncertainty(&self) -> V;

    /// Returns the uncertainty maximized opinion of `self`.
    fn uncertainty_maximized(&self) -> Self::Output;
}

impl<'a, T, V, Idx> MaxUncertainty<Idx, V> for OpinionRef<'a, T, V>
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = Opinion<T, V>;

    fn max_uncertainty(&self) -> V {
        let p = self.projection();
        T::keys()
            .map(|i| p[i] / self.base_rate[i])
            .reduce(<V>::min)
            .unwrap()
    }

    fn uncertainty_maximized(&self) -> Self::Output {
        let p = self.projection();
        let u_max = self.max_uncertainty();
        let b_max = T::from_fn(|i| p[i] - self.base_rate[i] * u_max);
        Opinion::new_unchecked(b_max, u_max, self.base_rate.clone())
    }
}

impl<T, V, Idx> MaxUncertainty<Idx, V> for Opinion<T, V>
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = Opinion<T, V>;

    fn max_uncertainty(&self) -> V {
        self.as_ref().max_uncertainty()
    }

    fn uncertainty_maximized(&self) -> Self::Output {
        self.as_ref().uncertainty_maximized()
    }
}

pub trait Discount<Idx, T, V> {
    type Output;
    /// Computes trust discounting of `self` with a referral trust `t`.
    fn discount(&self, t: V) -> Self::Output;
}

impl<'a, Idx, T, V> Discount<Idx, T, V> for SimplexBase<T, V>
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq,
{
    type Output = SimplexBase<T, V>;
    fn discount(&self, t: V) -> Self::Output {
        if self.is_vacuous() {
            Self::Output::vacuous()
        } else {
            SimplexBase {
                belief: T::from_fn(|i| self.b()[i] * t),
                uncertainty: V::one() - t * (V::one() - *self.u()),
            }
        }
    }
}

impl<'a, Idx, T, V> Discount<Idx, T, V> for OpinionRef<'a, T, V>
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq,
{
    type Output = Opinion<T, V>;
    fn discount(&self, t: V) -> Self::Output {
        Opinion::from_simplex_unchecked(self.simplex.discount(t), (*self.base_rate).clone())
    }
}

impl<Idx, T, V> Discount<Idx, T, V> for Opinion<T, V>
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq,
{
    type Output = Opinion<T, V>;
    fn discount(&self, t: V) -> Self::Output {
        self.as_ref().discount(t)
    }
}

fn check_simplex<V>(b: &[V], u: V) -> Result<(), InvalidValueError>
where
    V: UlpsEq + Float + AddAssign,
{
    let mut sum_b = V::zero();
    for (i, &bi) in b.iter().enumerate() {
        check_unit_interval(bi, format!("b[{i}]"))?;
        sum_b += bi;
    }

    check_unit_interval(u, "u")?;
    check_is_one(sum_b + u, "sum(b) + u")?;

    Ok(())
}

fn check_base_rate<V>(a: &[V]) -> Result<(), InvalidValueError>
where
    V: UlpsEq + Float + AddAssign,
{
    let mut sum_a = V::zero();
    for (i, &ai) in a.iter().enumerate() {
        check_unit_interval(ai, format!("a[{i}]"))?;
        sum_a += ai;
    }

    check_is_one(sum_a, "sum(a)")?;

    Ok(())
}

/// The reference of a simplex of a multinomial opinion, from which a base rate is excluded.
pub type Simplex<T, const N: usize> = SimplexBase<[T; N], T>;

/// A multinomial opinion with 1-dimensional vectors.
pub type Opinion1d<T, const N: usize> = Opinion<[T; N], T>;

/// The reference type of a multinomial opinion with 1-dimensional vectors.
pub type Opinion1dRef<'a, T, const N: usize> = OpinionRef<'a, [T; N], T>;

impl<V, const N: usize> Simplex<V, N>
where
    V: Float + AddAssign + UlpsEq,
{
    /// Creates a new simplex of a multinomial opinion (i.e. excluding a base rate) from parameters,
    /// which must satisfy the following conditions:
    /// $$
    /// \begin{aligned}
    /// \sum_{i=0}^{\mathsf N-1}\mathsf b\[i\] + \mathsf u \&= 1\quad\text{where }
    /// \mathsf b \in [0, 1]^\mathsf N, \mathsf u \in [0, 1],\\\\
    /// \mathsf u \in [0, 1].
    /// \end{aligned}
    /// $$
    ///
    /// # Errors
    /// If even pameter does not satisfy the conditions, an error is returned.
    pub fn try_new(b: [V; N], u: V) -> Result<Self, InvalidValueError> {
        check_simplex(&b, u)?;
        Ok(Self::new_unchecked(b, u))
    }

    /// Creates a new simplex of a multinomial opinion (i.e. excluding a base rate) from parameters,
    /// which reqiure the same conditions as `try_new`.
    ///
    /// # Panics
    /// Panics if even pameter does not satisfy the conditions.
    pub fn new(b: [V; N], u: V) -> Self {
        Self::try_new(b, u).unwrap()
    }
}

impl<V, const N: usize> TryFrom<([V; N], V)> for Simplex<V, N>
where
    V: Float + AddAssign + UlpsEq,
{
    type Error = InvalidValueError;

    fn try_from(value: ([V; N], V)) -> Result<Self, Self::Error> {
        Self::try_new(value.0, value.1)
    }
}

impl<V, const N: usize> Opinion1d<V, N>
where
    V: Float + AddAssign + UlpsEq,
{
    /// Creates a new multinomial opinion from parameters, which must satisfy the following conditions:
    /// $$
    /// \begin{aligned}
    /// \sum_{i=0}^{\mathsf N-1}\mathsf b\[i\] + \mathsf u \&= 1\quad\text{where }
    /// \mathsf b \in [0, 1]^\mathsf N, \mathsf u \in [0, 1],\\\\
    /// \sum_{i=0}^{\mathsf N-1}\mathsf a\[i\] \&= 1\quad\text{where }
    /// \mathsf u \in [0, 1].
    /// \end{aligned}
    /// $$
    ///
    /// # Errors
    /// If even pameter does not satisfy the conditions, an error is returned.
    pub fn try_new(b: [V; N], u: V, a: [V; N]) -> Result<Self, InvalidValueError> {
        check_simplex(&b, u)?;
        check_base_rate(&a)?;
        Ok(Self::new_unchecked(b, u, a))
    }

    /// Creates a new multinomial opinion from parameters which reqiure the same conditions as `try_new`.
    ///
    /// # Panics
    /// Panics if even pameter does not satisfy the conditions.
    pub fn new(b: [V; N], u: V, a: [V; N]) -> Self {
        Self::try_new(b, u, a).unwrap()
    }

    pub fn try_from_simplex(s: Simplex<V, N>, a: [V; N]) -> Result<Self, InvalidValueError> {
        check_base_rate(&a)?;
        Ok(Self {
            simplex: s,
            base_rate: a,
        })
    }
}

pub trait IndexedContainer<K>: Index<K> + IndexMut<K> {
    const SIZE: usize;
    type Map<U>: Index<K, Output = U>;

    fn keys() -> impl Iterator<Item = K>;
    fn map<U, F: FnMut(K) -> U>(f: F) -> Self::Map<U>;
    fn from_fn<F: FnMut(K) -> <Self as Index<K>>::Output>(f: F) -> Self;
}

impl<T, const N: usize> IndexedContainer<usize> for [T; N] {
    const SIZE: usize = N;
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

trait MBR<X, Y, T, Cond, U, V> {
    fn marginal_base_rate(ax: &T, conds: Cond) -> Option<U>;
}

impl<'a, X, Y, T, Cond, U, V> MBR<X, Y, T, &'a Cond, U, V> for U
where
    T: Index<X, Output = V>,
    Cond: IndexedContainer<X>,
    for<'b> &'b Cond::Output: Into<&'b SimplexBase<U, V>>,
    U: IndexedContainer<Y, Output = V>,
    X: Copy,
    Y: Copy,
    V: Float + Sum + UlpsEq + AddAssign + DivAssign,
{
    fn marginal_base_rate(ax: &T, conds: &'a Cond) -> Option<U> {
        let mut c = Cond::SIZE;
        for x in Cond::keys() {
            if approx_ext::is_one(conds[x].into().uncertainty) {
                c -= 1;
            }
        }
        if c == 0 {
            return None;
        }
        let mut sum_a = V::zero();
        let mut ay = U::from_fn(|y| {
            let a = Cond::keys()
                .map(|x| ax[x] * conds[x].into().belief[y])
                .sum::<V>();
            sum_a += a;
            a
        });
        for y in U::keys() {
            ay[y] /= sum_a;
        }
        Some(ay)
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use approx::{assert_ulps_eq, ulps_eq};
    use num_traits::Float;

    use crate::mul::check_base_rate;

    use super::{Discount, Opinion1d, Projection, Simplex, MBR};

    #[test]
    fn test_discount() {
        macro_rules! def {
            ($ft: ty) => {
                let w = Opinion1d::<$ft, 2>::new([0.2, 0.2], 0.6, [0.5, 0.5]);
                let w2 = w.discount(0.5);
                assert!(ulps_eq!(w2.b()[0], 0.1));
                assert!(ulps_eq!(w2.b()[1], 0.1));
                assert!(ulps_eq!(w2.u(), 1.0 - 0.2));
                assert!(ulps_eq!(w2.b()[0] + w2.b()[1] + w2.u(), 1.0));
                assert_eq!(w2.base_rate, w.base_rate);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_projection() {
        macro_rules! def {
            ($ft: ty) => {
                let w = Opinion1d::<f32, 2>::new([0.2, 0.2], 0.6, [1.0 / 3.0, 2.0 / 3.0]);
                let q: [f32; 2] = array::from_fn(|i| w.b()[i] + w.u() * w.base_rate[i]);
                let p = w.projection();
                println!("{:?}", p);
                println!("{:?}", q);
                assert_ulps_eq!(p[0] + p[1], 1.0);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_default() {
        fn f<V: Float>(w: &mut Opinion1d<V, 2>, b: &[V; 2]) {
            w.simplex.belief = *b;
            w.base_rate = *b;
        }
        macro_rules! def {
            ($ft:ty) => {
                let mut w = Opinion1d::<f32, 2>::default();
                assert_eq!(w.simplex.belief, [0.0; 2]);
                assert_eq!(w.simplex.uncertainty, 0.0);
                assert_eq!(w.base_rate, [0.0; 2]);
                let b = [1.0, 0.0];
                f(&mut w, &b);
                assert_eq!(w.b(), &b);
                assert_eq!(&w.base_rate, &b);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_mbr() {
        macro_rules! def {
            ($ft: ty) => {
                let cond1 = [Simplex::new([0.0, 0.0], 1.0), Simplex::new([0.0, 0.0], 1.0)];
                let cond2 = [
                    Simplex::new([0.0, 0.01], 0.99),
                    Simplex::new([0.0, 0.0], 1.0),
                ];
                let ax = [0.99, 0.01];

                let ay1 = <[$ft; 2]>::marginal_base_rate(&ax, &cond1);
                assert!(ay1.is_none());

                let ay2 = <[$ft; 2]>::marginal_base_rate(&ax, &cond2).unwrap();
                assert!(check_base_rate(&ay2).is_ok())
            };
        }
        def!(f32);
        def!(f64);
    }
}
