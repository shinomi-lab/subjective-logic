pub mod op;
pub mod prod;

use approx::{ulps_eq, ulps_ne};
use std::{
    array,
    fmt::Display,
    ops::{Index, Range},
};

use crate::approx_ext::ApproxRange;
use crate::errors::InvalidValueError;

#[derive(Debug, Clone, PartialEq)]
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

impl<T, V> From<(T, V)> for SimplexBase<T, V> {
    fn from(value: (T, V)) -> Self {
        SimplexBase::new_unchecked(value.0, value.1)
    }
}

/// The generlized structure of a multinomial opinion.
#[derive(Debug, Clone, PartialEq)]
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

macro_rules! impl_with_float {
    ($ft: ty) => {
        impl<T> SimplexBase<T, $ft> {
            #[inline]
            pub fn vacuous<Idx>() -> Self
            where
                T: IndexedContainer<Idx, Output = $ft>,
            {
                Self {
                    belief: T::from_fn(|_| 0.0),
                    uncertainty: 1.0,
                }
            }

            #[inline]
            pub fn is_vacuous(&self) -> bool {
                ulps_eq!(self.uncertainty, 1.0)
            }

            #[inline]
            pub fn is_dogmatic(&self) -> bool {
                ulps_eq!(self.uncertainty, 0.0)
            }
        }

        impl<T> Opinion<T, $ft> {
            #[inline]
            pub fn is_vacuous(&self) -> bool {
                ulps_eq!(self.u(), 1.0)
            }

            #[inline]
            pub fn is_dogmatic(&self) -> bool {
                ulps_eq!(self.u(), 0.0)
            }
        }

        impl<'a, T> OpinionRef<'a, T, $ft> {
            #[inline]
            pub fn is_vacuous(&self) -> bool {
                ulps_eq!(self.u(), 1.0)
            }

            #[inline]
            pub fn is_dogmatic(&self) -> bool {
                ulps_eq!(self.u(), 0.0)
            }
        }
    };
}

impl_with_float!(f32);
impl_with_float!(f64);

pub trait Projection<Idx, T> {
    /// Computes the probability projection of `self`.
    fn projection(&self) -> T;
}

macro_rules! impl_projection {
    ($ft: ty) => {
        impl<Idx, T> Projection<Idx, T> for Opinion<T, $ft>
        where
            T: IndexedContainer<Idx, Output = $ft>,
            Idx: Copy,
        {
            fn projection(&self) -> T {
                self.as_ref().projection()
            }
        }

        impl<'a, Idx, T> Projection<Idx, T> for OpinionRef<'a, T, $ft>
        where
            T: IndexedContainer<Idx, Output = $ft>,
            Idx: Copy,
        {
            fn projection(&self) -> T {
                T::from_fn(|idx| self.b()[idx] + self.base_rate[idx] * self.u())
            }
        }

        impl<T> SimplexBase<T, $ft> {
            pub fn projection<'a, Idx: Copy>(&'a self, a: &'a T) -> T
            where
                T: IndexedContainer<Idx, Output = $ft>,
            {
                OpinionRef::from((self, a)).projection()
            }
        }
    };
}

impl_projection!(f32);
impl_projection!(f64);

pub trait MaxUncertainty<Idx, V> {
    type Output;

    /// Returns the uncertainty maximized opinion of `self`.
    fn max_uncertainty(&self) -> V;

    /// Returns the uncertainty maximized opinion of `self`.
    fn uncertainty_maximized(&self) -> Self::Output;
}

macro_rules! impl_max_uncertainty {
    ($ft: ty) => {
        impl<T, Idx> MaxUncertainty<Idx, $ft> for Opinion<T, $ft>
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            type Output = Opinion<T, $ft>;

            fn max_uncertainty(&self) -> $ft {
                self.as_ref().max_uncertainty()
            }

            fn uncertainty_maximized(&self) -> Self::Output {
                self.as_ref().uncertainty_maximized()
            }
        }

        impl<'a, T, Idx> MaxUncertainty<Idx, $ft> for OpinionRef<'a, T, $ft>
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            type Output = Opinion<T, $ft>;

            fn max_uncertainty(&self) -> $ft {
                let p = self.projection();
                T::keys()
                    .map(|i| p[i] / self.base_rate[i])
                    .reduce(<$ft>::min)
                    .unwrap()
            }

            fn uncertainty_maximized(&self) -> Self::Output {
                let p = self.projection();
                let u_max = self.max_uncertainty();
                let b_max = T::from_fn(|i| p[i] - self.base_rate[i] * u_max);
                Opinion::new_unchecked(b_max, u_max, self.base_rate.clone())
            }
        }
    };
}

impl_max_uncertainty!(f32);
impl_max_uncertainty!(f64);

pub trait Discount<Idx, T, V> {
    type Output;
    /// Computes trust discounting of `self` with a referral trust `t`.
    fn discount(&self, t: V) -> Self::Output;
}

macro_rules! impl_discount {
    ($ft: ty) => {
        impl<'a, Idx, T> Discount<Idx, T, $ft> for SimplexBase<T, $ft>
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
        {
            type Output = SimplexBase<T, $ft>;
            fn discount(&self, t: $ft) -> Self::Output {
                if self.is_vacuous() {
                    Self::Output::vacuous()
                } else {
                    SimplexBase {
                        belief: T::from_fn(|i| self.b()[i] * t),
                        uncertainty: 1.0 - t * (1.0 - self.u()),
                    }
                }
            }
        }

        impl<'a, Idx, T> Discount<Idx, T, $ft> for OpinionRef<'a, T, $ft>
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
        {
            type Output = Opinion<T, $ft>;
            fn discount(&self, t: $ft) -> Self::Output {
                Opinion::from_simplex_unchecked(self.simplex.discount(t), (*self.base_rate).clone())
            }
        }

        impl<Idx, T> Discount<Idx, T, $ft> for Opinion<T, $ft>
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
        {
            type Output = Opinion<T, $ft>;
            fn discount(&self, t: $ft) -> Self::Output {
                self.as_ref().discount(t)
            }
        }
    };
}

impl_discount!(f32);
impl_discount!(f64);

/// The reference of a simplex of a multinomial opinion, from which a base rate is excluded.
pub type Simplex<T, const N: usize> = SimplexBase<[T; N], T>;

/// A multinomial opinion with 1-dimensional vectors.
pub type Opinion1d<T, const N: usize> = Opinion<[T; N], T>;

/// The reference type of a multinomial opinion with 1-dimensional vectors.
type Opinion1dRef<'a, T, const N: usize> = OpinionRef<'a, [T; N], T>;

macro_rules! impl_simplex {
    ($ft: ty) => {
        impl<const N: usize> Simplex<$ft, N> {
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
            pub fn try_new(b: [$ft; N], u: $ft) -> Result<Self, InvalidValueError> {
                if ulps_ne!(b.iter().sum::<$ft>() + u, 1.0) {
                    return Err(InvalidValueError(
                        "sum(b) + u = 1 is not satisfied".to_string(),
                    ));
                }
                if u.out_of_range(0.0, 1.0) {
                    return Err(InvalidValueError("u ∈ [0,1] is not satisfied".to_string()));
                }

                for i in 0..N {
                    if b[i].out_of_range(0.0, 1.0) {
                        return Err(InvalidValueError(format!(
                            "b[{}] ∈ [0,1] is not satisfied",
                            i
                        )));
                    }
                }
                Ok(Self::new_unchecked(b, u))
            }

            /// Creates a new simplex of a multinomial opinion (i.e. excluding a base rate) from parameters,
            /// which reqiure the same conditions as `try_new`.
            ///
            /// # Panics
            /// Panics if even pameter does not satisfy the conditions.
            pub fn new(b: [$ft; N], u: $ft) -> Self {
                Self::try_new(b, u).unwrap()
            }
        }
    };
}

impl_simplex!(f32);
impl_simplex!(f64);

macro_rules! impl_opinion {
    ($ft: ty) => {
        impl<const N: usize> Opinion1d<$ft, N> {
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
            pub fn try_new(b: [$ft; N], u: $ft, a: [$ft; N]) -> Result<Self, InvalidValueError> {
                if ulps_ne!(b.iter().sum::<$ft>() + u, 1.0) {
                    return Err(InvalidValueError(
                        "sum(b) + u = 1 is not satisfied".to_string(),
                    ));
                }
                if ulps_ne!(a.iter().sum::<$ft>(), 1.0) {
                    return Err(InvalidValueError("sum(a) = 1 is not satisfied".to_string()));
                }

                if u.out_of_range(0.0, 1.0) {
                    return Err(InvalidValueError(format!("u ∈ [0,1] is not satisfied")));
                }

                for i in 0..N {
                    if b[i].out_of_range(0.0, 1.0) {
                        return Err(InvalidValueError(format!(
                            "b[{i}] ∈ [0,1] is not satisfied"
                        )));
                    }
                    if a[i].out_of_range(0.0, 1.0) {
                        return Err(InvalidValueError(format!(
                            "a[{i}] ∈ [0,1] is not satisfied"
                        )));
                    }
                }
                Ok(Self::new_unchecked(b, u, a))
            }

            /// Creates a new multinomial opinion from parameters which reqiure the same conditions as `try_new`.
            ///
            /// # Panics
            /// Panics if even pameter does not satisfy the conditions.
            pub fn new(b: [$ft; N], u: $ft, a: [$ft; N]) -> Self {
                Self::try_new(b, u, a).unwrap()
            }

            pub fn try_from_simplex(
                s: Simplex<$ft, N>,
                a: [$ft; N],
            ) -> Result<Self, InvalidValueError> {
                if ulps_ne!(a.iter().sum::<$ft>(), 1.0) {
                    return Err(InvalidValueError("sum(a) = 1 is not satisfied".to_string()));
                }
                for i in 0..N {
                    if a[i].out_of_range(0.0, 1.0) {
                        return Err(InvalidValueError(format!(
                            "a[{i}] ∈ [0,1] is not satisfied"
                        )));
                    }
                }
                Ok(Self {
                    simplex: s,
                    base_rate: a,
                })
            }
        }
    };
}

impl_opinion!(f32);
impl_opinion!(f64);

pub trait IndexedContainer<K>: Index<K> {
    const SIZE: usize;
    type Map<U>: Index<K, Output = U>;
    type Keys: Iterator<Item = K>;
    fn keys() -> Self::Keys;
    fn map<U, F: Fn(K) -> U>(f: F) -> Self::Map<U>;
    fn from_fn<F: Fn(K) -> <Self as Index<K>>::Output>(f: F) -> Self;
}

impl<T, const N: usize> IndexedContainer<usize> for [T; N] {
    const SIZE: usize = N;
    type Map<U> = [U; N];
    type Keys = Range<usize>;

    fn keys() -> Self::Keys {
        0..N
    }

    fn map<U, F: Fn(usize) -> U>(f: F) -> [U; N] {
        array::from_fn(|i| f(i))
    }

    fn from_fn<F: Fn(usize) -> T>(f: F) -> Self {
        array::from_fn(|i| f(i))
    }
}

trait MBR<X, Y, T, Cond, U, V> {
    fn marginal_base_rate(ax: &T, conds: Cond) -> Option<U>;
}

macro_rules! impl_mbr {
    ($ft: ty) => {
        impl<'a, X, Y, T, Cond, U> MBR<X, Y, T, &'a Cond, U, $ft> for U
        where
            T: Index<X, Output = $ft>,
            Cond: IndexedContainer<X>,
            for<'b> &'b Cond::Output: Into<&'b SimplexBase<U, $ft>>,
            U: IndexedContainer<Y, Output = $ft>,
            X: Copy,
            Y: Copy,
        {
            fn marginal_base_rate(ax: &T, conds: &'a Cond) -> Option<U> {
                if ulps_eq!(
                    Cond::keys()
                        .map(|x| (&conds[x]).into().uncertainty)
                        .sum::<$ft>(),
                    Cond::SIZE as $ft
                ) {
                    return None;
                }
                let ay = U::from_fn(|y| {
                    let temp = Cond::keys()
                        .map(|x| ax[x] * (&conds[x]).into().uncertainty)
                        .sum::<$ft>();
                    let temp2 = Cond::keys()
                        .map(|x| ax[x] * (&conds[x]).into().belief[y])
                        .sum::<$ft>();
                    if temp <= <$ft>::EPSILON {
                        temp2
                    } else {
                        temp2 / (1.0 - temp)
                    }
                });
                Some(ay)
            }
        }
    };
}

impl_mbr!(f32);
impl_mbr!(f64);

#[cfg(test)]
mod tests {
    use approx::ulps_eq;

    use super::{Discount, Opinion1d};

    #[test]
    fn test_discount() {
        let w = Opinion1d::<f32, 2>::new([0.2, 0.2], 0.6, [0.5, 0.5]);
        let w2 = w.discount(0.5);
        assert!(ulps_eq!(w2.b()[0], 0.1));
        assert!(ulps_eq!(w2.b()[1], 0.1));
        assert!(ulps_eq!(w2.u(), 1.0 - 0.2));
        assert!(ulps_eq!(w2.b()[0] + w2.b()[1] + w2.u(), 1.0));
        assert_eq!(w2.base_rate, w.base_rate);
    }
}
