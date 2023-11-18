pub mod op;

use approx::{ulps_eq, ulps_ne};
use std::{array, fmt::Display, ops::Index};

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
}

impl<S: Display, T: Display> Display for OpinionBase<S, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.simplex, self.base_rate)
    }
}

pub type Opinion<T, U> = OpinionBase<SimplexBase<T, U>, T>;
type OpinionRef<'a, T, U> = OpinionBase<&'a SimplexBase<T, U>, &'a T>;

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
        OpinionBase {
            simplex: &value.simplex,
            base_rate: &value.base_rate,
        }
    }
}

/// The reference of a simplex of a multinomial opinion, from which a base rate is excluded.
pub type Simplex<T, const N: usize> = SimplexBase<[T; N], T>;

/// A multinomial opinion with 1-dimensional vectors.
pub type Opinion1d<T, const N: usize> = Opinion<[T; N], T>;

/// The reference type of a multinomial opinion with 1-dimensional vectors.
type Opinion1dRef<'a, T, const N: usize> = OpinionRef<'a, [T; N], T>;

trait MBR<T, const N: usize, const M: usize> {
    fn marginal_base_rate(&self, ax: &[T; M]) -> Option<[T; N]>;
}

macro_rules! impl_msimplex {
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

            pub fn projection(&self, a: &[$ft; N]) -> [$ft; N] {
                array::from_fn(|i| self.belief[i] + self.uncertainty * a[i])
            }
        }

        impl<const M: usize, const N: usize> MBR<$ft, N, M> for [&Simplex<$ft, N>; M] {
            fn marginal_base_rate(&self, ax: &[$ft; M]) -> Option<[$ft; N]> {
                if ulps_eq!((0..M).map(|i| self[i].uncertainty).sum::<$ft>(), M as $ft) {
                    None
                } else {
                    Some(array::from_fn(|j| {
                        let temp = (0..M).map(|i| ax[i] * self[i].uncertainty).sum::<$ft>();
                        let temp2 = (0..M).map(|i| ax[i] * self[i].belief[j]).sum::<$ft>();
                        if temp <= <$ft>::EPSILON {
                            temp2
                        } else {
                            temp2 / (1.0 - temp)
                        }
                    }))
                }
            }
        }

        impl<const M: usize, const N: usize> MBR<$ft, N, M> for [Simplex<$ft, N>; M] {
            fn marginal_base_rate(&self, ax: &[$ft; M]) -> Option<[$ft; N]> {
                if ulps_eq!((0..M).map(|i| self[i].uncertainty).sum::<$ft>(), M as $ft) {
                    None
                } else {
                    Some(array::from_fn(|j| {
                        let temp = (0..M).map(|i| ax[i] * self[i].uncertainty).sum::<$ft>();
                        let temp2 = (0..M).map(|i| ax[i] * self[i].belief[j]).sum::<$ft>();
                        if temp <= <$ft>::EPSILON {
                            temp2
                        } else {
                            temp2 / (1.0 - temp)
                        }
                    }))
                }
            }
        }
    };
}

impl_msimplex!(f32);
impl_msimplex!(f64);

macro_rules! impl_mop {
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

        /// The probability projection of `self`.
        impl<T: Index<usize, Output = $ft>> Opinion<T, $ft> {
            pub fn projection(&self, idx: usize) -> $ft {
                self.b()[idx] + self.base_rate[idx] * self.u()
            }
        }

        /// The probability projection of `self`.
        impl<'a, T: Index<usize, Output = $ft>> OpinionRef<'a, T, $ft> {
            pub fn projection(&self, idx: usize) -> $ft {
                self.b()[idx] + self.base_rate[idx] * *self.u()
            }
        }
    };
}

impl_mop!(f32);
impl_mop!(f64);
