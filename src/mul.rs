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
pub struct SimplexBase<T, U> {
    pub belief: T,
    pub uncertainty: U,
}

impl<T: Display, U: Display> Display for SimplexBase<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.belief, self.uncertainty)
    }
}

impl<T, U> SimplexBase<T, U> {
    pub fn new_unchecked(b: T, u: U) -> Self {
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
    pub fn u(&self) -> &U {
        &self.uncertainty
    }
}

impl<T, U> From<(T, U)> for SimplexBase<T, U> {
    fn from(value: (T, U)) -> Self {
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
    pub fn u(&self) -> &U {
        &self.simplex.uncertainty
    }
}

impl<T, U> OpinionRef<'_, T, U> {
    #[inline]
    pub fn b(&self) -> &T {
        &self.simplex.belief
    }

    #[inline]
    pub fn u(&self) -> &U {
        &self.simplex.uncertainty
    }
}

impl<'a, T, U> From<&'a Opinion<T, U>> for &'a SimplexBase<T, U> {
    fn from(value: &'a Opinion<T, U>) -> Self {
        &value.simplex
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

macro_rules! impl_projection {
    ($ft: ty) => {
        /// The probability projection of `self`.
        impl<T> Opinion<T, $ft> {
            pub fn projection<Idx: Copy>(&self, idx: Idx) -> $ft
            where
                T: Index<Idx, Output = $ft>,
            {
                self.b()[idx] + self.base_rate[idx] * self.u()
            }
        }

        /// The probability projection of `self`.
        impl<'a, T> OpinionRef<'a, T, $ft> {
            pub fn projection<Idx: Copy>(&self, idx: Idx) -> $ft
            where
                T: Index<Idx, Output = $ft>,
            {
                self.b()[idx] + self.base_rate[idx] * *self.u()
            }
        }

        impl<T> SimplexBase<T, $ft> {
            pub fn projection<'a, Idx: Copy>(&'a self, a: &'a T) -> T
            where
                T: IndexContainer<Idx, Output = $ft>,
            {
                T::from_fn(|i| self.belief[i] + self.uncertainty * a[i])
            }
        }
    };
}

impl_projection!(f32);
impl_projection!(f64);

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
                    return Err(InvalidValueError(format!("u ∈ [0,1] is not satisfied")));
                }

                for i in 0..N {
                    if b[i].out_of_range(0.0, 1.0) {
                        return Err(InvalidValueError(format!(
                            "b[{i}] ∈ [0,1] is not satisfied"
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

            /// Returns the uncertainty maximized opinion of `self`.
            pub fn max_uncertainty(&self) -> $ft {
                (0..N)
                    .map(|i| self.projection(i) / self.base_rate[i])
                    .reduce(<$ft>::min)
                    .unwrap()
            }

            /// Returns the uncertainty maximized opinion of `self`.
            pub fn op_u_max(&self) -> Result<Self, InvalidValueError> {
                let u_max = self.max_uncertainty();
                let b_max = array::from_fn(|i| self.projection(i) - self.base_rate[i] * u_max);
                Self::try_new(b_max, u_max, self.base_rate.clone())
            }
        }

        impl<'a, const N: usize> Opinion1dRef<'a, $ft, N> {
            /// Returns the uncertainty maximized opinion of `self`.
            pub fn max_uncertainty(&self) -> $ft {
                (0..N)
                    .map(|i| self.projection(i) / self.base_rate[i])
                    .reduce(<$ft>::min)
                    .unwrap()
            }
        }
    };
}

impl_opinion!(f32);
impl_opinion!(f64);

pub trait IndexContainer<K>: Index<K> {
    const SIZE: usize;
    type Map<U>: Index<K, Output = U>;
    type Keys: Iterator<Item = K>;
    fn keys() -> Self::Keys;
    fn map<U, F: Fn(K) -> U>(f: F) -> Self::Map<U>;
    fn from_fn<F: Fn(K) -> <Self as Index<K>>::Output>(f: F) -> Self;
}

impl<T, const N: usize> IndexContainer<usize> for [T; N] {
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

trait MBR<'a, X, Y, T, U, V>
where
    T: Index<X, Output = V>,
    U: Index<Y, Output = V>,
{
    fn marginal_base_rate(&'a self, ax: &'a T) -> Option<U>;
}

macro_rules! impl_mbr {
    ($ft: ty) => {
        impl<'a, X, Y, T, U, SimSet> MBR<'a, X, Y, T, U, $ft> for SimSet
        where
            SimSet: IndexContainer<X>,
            &'a SimSet::Output: Into<&'a SimplexBase<U, $ft>> + 'a,
            T: Index<X, Output = $ft>,
            U: IndexContainer<Y, Output = $ft> + 'a,
            X: Copy,
            Y: Copy,
        {
            fn marginal_base_rate(&'a self, ax: &'a T) -> Option<U> {
                if ulps_eq!(
                    SimSet::keys()
                        .map(|x| (&self[x]).into().uncertainty)
                        .sum::<$ft>(),
                    SimSet::SIZE as $ft
                ) {
                    return None;
                }
                let ay = U::from_fn(|y| {
                    let temp = SimSet::keys()
                        .map(|x| ax[x] * (&self[x]).into().uncertainty)
                        .sum::<$ft>();
                    let temp2 = SimSet::keys()
                        .map(|x| ax[x] * (&self[x]).into().belief[y])
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
