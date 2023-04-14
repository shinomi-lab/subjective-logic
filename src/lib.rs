//!
//! An implementation of [Subjective Logic](https://en.wikipedia.org/wiki/Subjective_logic).

use std::{array, fmt::Display, ops::Index};

/// A binomial opinion.
#[derive(Debug)]
pub struct BOpinion<T> {
    belief: T,
    disbelief: T,
    uncertainty: T,
    base_rate: T,
}

impl<T: Display> Display for BOpinion<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{},{},{}",
            self.belief, self.disbelief, self.uncertainty, self.base_rate
        )
    }
}

impl<T> BOpinion<T> {
    fn new_unchecked(b: T, d: T, u: T, a: T) -> Self {
        Self {
            belief: b,
            disbelief: d,
            uncertainty: u,
            base_rate: a,
        }
    }
}

macro_rules! impl_bsl {
    ($ft: ty) => {
        impl BOpinion<$ft> {
            /// Creates a new binomial opinion from parameters, which must satisfy the following conditions:
            /// $$
            /// \begin{aligned}
            /// \mathsf b + \mathsf d + \mathsf u \&= 1\quad\text{where }
            /// \mathsf b \in [0, 1], \mathsf u \in [0, 1],\\\\
            /// \mathsf a \in [0, 1].
            /// \end{aligned}
            /// $$
            ///
            /// # Errors
            /// If even pameter does not satisfy the conditions, an error is returned.
            pub fn try_new(b: $ft, d: $ft, u: $ft, a: $ft) -> Result<Self, InvalidValueError> {
                if !(b + d + u).approx_eq(1.0) {
                    return Err(InvalidValueError(
                        "b + d + u = 1 is not satisfied".to_string(),
                    ));
                }
                if !b.is_in_range(0.0, 1.0) {
                    return Err(InvalidValueError("b ∈ [0,1] is not satisfied".to_string()));
                }
                if !d.is_in_range(0.0, 1.0) {
                    return Err(InvalidValueError("d ∈ [0,1] is not satisfied".to_string()));
                }
                if !u.is_in_range(0.0, 1.0) {
                    return Err(InvalidValueError("u ∈ [0,1] is not satisfied".to_string()));
                }
                if !a.is_in_range(0.0, 1.0) {
                    return Err(InvalidValueError("a ∈ [0,1] is not satisfied".to_string()));
                }
                Ok(Self {
                    belief: b,
                    disbelief: d,
                    uncertainty: u,
                    base_rate: a,
                })
            }

            /// Creates a new binomial opinion from parameters which reqiure the same conditions as `try_new`.
            ///
            /// # Panics
            /// Panics if even pameter does not satisfy the conditions.
            pub fn new(b: $ft, d: $ft, u: $ft, a: $ft) -> Self {
                Self::try_new(b, d, u, a).unwrap()
            }

            /// The probability projection of `self`.
            pub fn projection(&self) -> $ft {
                self.belief + self.base_rate * self.uncertainty
            }

            /// Computes the opinion on the logical conjunction of `self` and `rhs`.
            pub fn mul(&self, rhs: &Self) -> Self {
                let a = self.base_rate * rhs.base_rate;
                let b = self.belief * rhs.belief
                    + ((1.0 - self.base_rate) * rhs.base_rate * self.belief * rhs.uncertainty
                        + (1.0 - rhs.base_rate) * self.base_rate * rhs.belief * self.uncertainty)
                        / (1.0 - a);
                let d = self.disbelief + rhs.disbelief - self.disbelief * rhs.disbelief;
                let u = self.uncertainty * rhs.uncertainty
                    + ((1.0 - rhs.base_rate) * self.belief * rhs.uncertainty
                        + (1.0 - self.base_rate) * rhs.belief * self.uncertainty)
                        / (1.0 - a);
                Self::new(b, d, u, a)
            }

            /// Computes the opinion on the logical disjunction of `self` and `rhs`.
            pub fn comul(&self, rhs: &Self) -> Self {
                let a = self.base_rate + rhs.base_rate - self.base_rate * rhs.base_rate;
                let b = self.belief + rhs.belief - self.belief * rhs.belief;
                let d = self.disbelief * rhs.disbelief
                    + (self.base_rate * (1.0 - rhs.base_rate) * self.disbelief * rhs.uncertainty
                        + rhs.base_rate
                            * (1.0 - self.base_rate)
                            * rhs.disbelief
                            * self.uncertainty)
                        / a;
                let u = self.uncertainty * rhs.uncertainty
                    + (rhs.base_rate * self.belief * rhs.uncertainty
                        + self.base_rate * rhs.belief * self.uncertainty)
                        / a;
                Self::new(b, d, u, a)
            }

            /// Computes the cumulative fusion of `self` and `rhs`.
            pub fn cfuse(&self, rhs: &Self) -> Result<Self, InvalidValueError> {
                let uu = self.uncertainty * rhs.uncertainty;
                let kappa = self.uncertainty + rhs.uncertainty - uu;
                let b = (self.belief * rhs.uncertainty + rhs.belief * self.uncertainty) / kappa;
                let d =
                    (self.disbelief * rhs.uncertainty + rhs.disbelief * self.uncertainty) / kappa;
                let u = (self.uncertainty * rhs.uncertainty) / kappa;
                let a = if self.uncertainty.approx_eq(1.0) && rhs.uncertainty.approx_eq(1.0) {
                    (self.base_rate + rhs.base_rate) / 2.0
                } else {
                    (self.base_rate * rhs.uncertainty + rhs.base_rate * self.uncertainty
                        - (self.base_rate + rhs.base_rate) * uu)
                        / (kappa - uu)
                };
                Self::try_new(b, d, u, a)
            }

            /// Computes the averaging belief fusion of `self` and `rhs`.
            pub fn afuse(&self, rhs: &Self, gamma_a: $ft) -> Result<Self, InvalidValueError> {
                let b;
                let d;
                let u;
                let a;
                if self.uncertainty.approx_eq(0.0) && rhs.uncertainty.approx_eq(0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = gamma_a * self.belief + gamma_b * rhs.belief;
                    d = gamma_a * self.disbelief + gamma_b * rhs.disbelief;
                    u = 0.0;
                    a = gamma_a * self.base_rate + gamma_b * rhs.base_rate;
                } else {
                    let upu = self.uncertainty + rhs.uncertainty;
                    b = (self.belief * rhs.uncertainty + rhs.belief * self.uncertainty) / upu;
                    d = (self.disbelief * rhs.uncertainty + rhs.disbelief * self.uncertainty) / upu;
                    u = 2.0 * self.uncertainty * rhs.uncertainty / upu;
                    a = (self.base_rate + rhs.base_rate) / 2.0;
                }
                Self::try_new(b, d, u, a)
            }

            /// Computes the weighted belief fusion of `self` and `rhs`.
            pub fn wfuse(&self, rhs: &Self, gamma_a: $ft) -> Result<Self, InvalidValueError> {
                let b;
                let d;
                let u;
                let a;
                if self.uncertainty.approx_eq(0.0) && rhs.uncertainty.approx_eq(0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = gamma_a * self.belief + gamma_b * rhs.belief;
                    d = gamma_a * self.disbelief + gamma_b * rhs.disbelief;
                    u = 0.0;
                    a = gamma_a * self.base_rate + gamma_b * rhs.base_rate;
                } else if self.uncertainty.approx_eq(1.0) && rhs.uncertainty.approx_eq(1.0) {
                    b = 0.0;
                    d = 0.0;
                    u = 1.0;
                    a = (self.base_rate + rhs.base_rate) / 2.0;
                } else {
                    let denom = self.uncertainty + rhs.uncertainty
                        - 2.0 * self.uncertainty * rhs.uncertainty;
                    let ca = 1.0 - self.uncertainty;
                    let cb = 1.0 - rhs.uncertainty;
                    b = (self.belief * ca * rhs.uncertainty + rhs.belief * cb * self.uncertainty)
                        / denom;
                    d = (self.disbelief * ca * rhs.uncertainty
                        + rhs.disbelief * cb * self.uncertainty)
                        / denom;
                    u = (2.0 - self.uncertainty - rhs.uncertainty)
                        * self.uncertainty
                        * rhs.uncertainty
                        / denom;
                    a = (self.base_rate * ca + rhs.base_rate * cb)
                        / (2.0 - self.uncertainty - rhs.uncertainty);
                }
                Self::try_new(b, d, u, a)
            }

            /// Computes the conditionally deduced opinion of `self` by a two length array of conditional opinions `cond`.
            /// If `self.uncertainty` is equal to `0.0`, this function panics.
            pub fn deduce(&self, cond: BSimplexes<$ft, 2>, ay: $ft) -> Self {
                let rvax = (1.0 - self.base_rate);
                let bi = self.belief * cond[0].belief
                    + self.disbelief * cond[1].belief
                    + self.uncertainty * (cond[0].belief * self.base_rate + cond[1].belief * rvax);
                let di = self.belief * cond[0].disbelief
                    + self.disbelief * cond[1].disbelief
                    + self.uncertainty
                        * (cond[0].disbelief * self.base_rate + cond[1].disbelief * rvax);
                let ui = self.belief * cond[0].uncertainty
                    + self.disbelief * cond[1].uncertainty
                    + self.uncertainty
                        * (cond[0].uncertainty * self.base_rate + cond[1].uncertainty * rvax);
                let k = match (
                    cond[0].belief > cond[1].belief,
                    cond[0].disbelief > cond[1].disbelief,
                ) {
                    // Case I
                    (true, true) | (false, false) => 0.0,
                    (bp, _) => {
                        let pyx = cond[0].belief * self.base_rate
                            + cond[1].belief * rvax
                            + ay * (cond[0].uncertainty * self.base_rate
                                + cond[1].uncertainty * rvax);
                        let px = self.projection();
                        let r = cond[1].belief + ay * (1.0 - cond[1].belief - cond[0].disbelief);
                        dbg!(pyx, r, px);
                        match (pyx > r, px > self.base_rate) {
                            (false, false) => {
                                if bp {
                                    // Case II.A.1
                                    self.base_rate * self.uncertainty * (bi - cond[1].belief)
                                        / (px * ay)
                                } else {
                                    // Case III.A.1
                                    rvax * self.uncertainty
                                        * (di - cond[1].disbelief)
                                        * (cond[1].belief - cond[0].belief)
                                        / (px * ay * (cond[0].disbelief - cond[1].disbelief))
                                }
                            }
                            (false, true) => {
                                if bp {
                                    // Case II.A.2
                                    self.base_rate
                                        * self.uncertainty
                                        * (di - cond[0].disbelief)
                                        * (cond[0].belief - cond[1].belief)
                                        / ((1.0 - px)
                                            * ay
                                            * (cond[1].disbelief - cond[0].disbelief))
                                } else {
                                    // Case III.A.2
                                    rvax * self.uncertainty * (bi - cond[0].belief)
                                        / ((1.0 - px) * ay)
                                }
                            }
                            (true, false) => {
                                if bp {
                                    // Case II.B.1
                                    rvax * self.uncertainty
                                        * (bi - cond[1].belief)
                                        * (cond[1].disbelief - cond[0].disbelief)
                                        / (px * (1.0 - ay) * (cond[0].belief - cond[1].belief))
                                } else {
                                    // Case III.B.1
                                    self.base_rate * self.uncertainty * (di - cond[1].disbelief)
                                        / (px * (1.0 - ay))
                                }
                            }
                            (true, true) => {
                                if bp {
                                    // Case II.B.2
                                    rvax * self.uncertainty * (di - cond[0].disbelief)
                                        / ((1.0 - px) * (1.0 - ay))
                                } else {
                                    // Case III.B.2
                                    self.base_rate
                                        * self.uncertainty
                                        * (bi - cond[0].belief)
                                        * (cond[0].disbelief - cond[1].disbelief)
                                        / ((1.0 - px)
                                            * (1.0 - ay)
                                            * (cond[1].belief - cond[0].belief))
                                }
                            }
                        }
                    }
                };
                let b = bi - ay * k;
                let d = di - (1.0 - ay) * k;
                let u = ui + k;
                let a = ay;
                Self::new(b, d, u, a)
            }

            /// Computes the uncertainty favouring discounted opinion.
            pub fn trans_unc(&self, b: $ft) -> Self {
                assert!(b.is_in_range(0.0, 1.0), "b ∈ [0,1] is not satisfied.");
                Self::new(
                    b * self.belief,
                    b * self.disbelief,
                    1.0 - b + b * self.uncertainty,
                    self.base_rate,
                )
            }

            /// Computes the opposite belief favouring discounted opinion.
            pub fn trans_opp(&self, b: $ft, d: $ft) -> Self {
                let u = 1.0 - b - d;
                assert!(u.is_in_range(0.0, 1.0), "b + d ∈ [0,1] is not satisfied.");
                Self::new(
                    b * self.belief + d * self.disbelief,
                    b * self.disbelief + d * self.belief,
                    u + (b + d) * self.uncertainty,
                    self.base_rate,
                )
            }

            /// Computes base rate sensitive discounted opinion.
            pub fn trans_bsr(&self, ev: $ft) -> Self {
                assert!(ev.is_in_range(0.0, 1.0), "ev ∈ [0,1] is not satisfied.");
                Self::new(
                    ev * self.belief,
                    ev * self.disbelief,
                    1.0 - ev * (self.belief + self.disbelief),
                    self.base_rate,
                )
            }
        }
    };
}

impl_bsl!(f32);
impl_bsl!(f64);

/// The simplex of a binomial opinion, from which a base rate is excluded.
#[derive(Debug)]
pub struct BSimplex<'a, T> {
    belief: &'a T,
    disbelief: &'a T,
    uncertainty: &'a T,
}

macro_rules! impl_bsimplex {
    ($ft: ty) => {
        impl<'a> BSimplex<'a, $ft> {
            pub fn projection(&self, a: &$ft) -> $ft {
                self.belief + self.uncertainty * a
            }
        }
    };
}

impl_bsimplex!(f32);
impl_bsimplex!(f64);

impl<'a, T> From<&'a BOpinion<T>> for BSimplex<'a, T> {
    fn from(value: &'a BOpinion<T>) -> Self {
        Self {
            belief: &value.belief,
            disbelief: &value.disbelief,
            uncertainty: &value.uncertainty,
        }
    }
}

pub struct BSimplexes<'a, T, const M: usize>([BSimplex<'a, T>; M]);

impl<'a, I, T, const M: usize> Index<I> for BSimplexes<'a, T, M>
where
    [BSimplex<'a, T>]: Index<I>,
{
    type Output = <[BSimplex<'a, T>; M] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<'a, T, const M: usize> From<&'a [BOpinion<T>; M]> for BSimplexes<'a, T, M> {
    fn from(value: &'a [BOpinion<T>; M]) -> Self {
        BSimplexes(array::from_fn(|i| (&value[i]).into()))
    }
}

/// The generlized structure of a multinomial opinion.
#[derive(Debug)]
pub struct MOpinion<T, U> {
    belief: T,
    uncertainty: U,
    base_rate: T,
}

impl<T: Display, U: Display> Display for MOpinion<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{},{}", self.belief, self.uncertainty, self.base_rate)
    }
}

impl<T, U> MOpinion<T, U> {
    fn new_unchecked(b: T, u: U, a: T) -> Self {
        Self {
            belief: b,
            uncertainty: u,
            base_rate: a,
        }
    }
}

/// A multinomial opinion with 1-dimensional vectors.
pub type MOpinion1d<T, const N: usize> = MOpinion<[T; N], T>;

/// The simplex of a multinomial opinion, from which a base rate is excluded.
#[derive(Debug)]
pub struct MSimplex<'a, T, const N: usize> {
    belief: [&'a T; N],
    uncertainty: &'a T,
}

macro_rules! impl_msimplex {
    ($ft: ty) => {
        impl<'a, const N: usize> MSimplex<'a, $ft, N> {
            pub fn projection(&self, a: &[$ft; N]) -> [$ft; N] {
                array::from_fn(|i| self.belief[i] + self.uncertainty * a[i])
            }
        }
    };
}

impl_msimplex!(f32);
impl_msimplex!(f64);

impl<'a, T, const N: usize> From<&'a MOpinion1d<T, N>> for MSimplex<'a, T, N> {
    fn from(value: &'a MOpinion1d<T, N>) -> Self {
        Self {
            belief: array::from_fn(|i| &value.belief[i]),
            uncertainty: &value.uncertainty,
        }
    }
}

impl<'a, T> From<&'a BOpinion<T>> for MSimplex<'a, T, 2> {
    fn from(value: &'a BOpinion<T>) -> Self {
        Self {
            belief: [&value.belief, &value.disbelief],
            uncertainty: &value.uncertainty,
        }
    }
}

pub struct MSimplexes<'a, T, const N: usize, const M: usize>([MSimplex<'a, T, N>; M]);

impl<'a, I, T, const N: usize, const M: usize> Index<I> for MSimplexes<'a, T, N, M>
where
    [MSimplex<'a, T, N>]: Index<I>,
{
    type Output = <[MSimplex<'a, T, N>; M] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<'a, T, const N: usize, const M: usize> From<&'a [MOpinion1d<T, N>; M]>
    for MSimplexes<'a, T, N, M>
{
    fn from(value: &'a [MOpinion1d<T, N>; M]) -> Self {
        MSimplexes(array::from_fn(|i| (&value[i]).into()))
    }
}

impl<'a, T, const M: usize> From<&'a [BOpinion<T>; M]> for MSimplexes<'a, T, 2, M> {
    fn from(value: &'a [BOpinion<T>; M]) -> Self {
        MSimplexes(array::from_fn(|i| (&value[i]).into()))
    }
}

/// The deduction operator.
pub trait Deduction<Rhs, U> {
    type Output;

    /// Computes the conditionally deduced opinion of `self` with a base rate vector `ay`
    /// by `cond` representing a collection of conditional opinions.
    fn deduce(&self, cond: Rhs, ay: U) -> Self::Output;

    /// Computes the probability projection of the deduced opinion
    fn projection(&self, cond: Rhs, ay: U) -> U;
}

macro_rules! impl_msl {
    ($ft: ty) => {
        impl<const N: usize> MOpinion1d<$ft, N> {
            /// Creates a new binomial opinion from parameters, which must satisfy the following conditions:
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
                if !(b.iter().sum::<$ft>() + u).approx_eq(1.0) {
                    return Err(InvalidValueError(
                        "sum(b) + u = 1 is not satisfied".to_string(),
                    ));
                }
                if !a.iter().sum::<$ft>().approx_eq(1.0) {
                    return Err(InvalidValueError("sum(a) = 1 is not satisfied".to_string()));
                }

                if !u.is_in_range(0.0, 1.0) {
                    return Err(InvalidValueError(format!("u ∈ [0,1] is not satisfied")));
                }

                for i in 0..N {
                    if !b[i].is_in_range(0.0, 1.0) {
                        return Err(InvalidValueError(format!(
                            "b[{i}] ∈ [0,1] is not satisfied"
                        )));
                    }
                    if !a[i].is_in_range(0.0, 1.0) {
                        return Err(InvalidValueError(format!(
                            "a[{i}] ∈ [0,1] is not satisfied"
                        )));
                    }
                }
                Ok(Self {
                    belief: b,
                    uncertainty: u,
                    base_rate: a,
                })
            }

            /// Creates a new binomial opinion from parameters which reqiure the same conditions as `try_new`.
            ///
            /// # Panics
            /// Panics if even pameter does not satisfy the conditions.
            pub fn new(b: [$ft; N], u: $ft, a: [$ft; N]) -> Self {
                Self::try_new(b, u, a).unwrap()
            }

            /// Computes the averaging belief fusion of `self` and `rhs`.
            pub fn afuse(&self, rhs: &Self, gamma_a: $ft) -> Result<Self, InvalidValueError> {
                let b;
                let u;
                let a;
                if self.uncertainty.approx_eq(0.0) && rhs.uncertainty.approx_eq(0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = array::from_fn(|i| gamma_a * self.belief[i] + gamma_b * rhs.belief[i]);
                    u = 0.0;
                    a = array::from_fn(|i| {
                        gamma_a * self.base_rate[i] + gamma_b * rhs.base_rate[i]
                    });
                } else {
                    let upu = self.uncertainty + rhs.uncertainty;
                    b = array::from_fn(|i| {
                        (self.belief[i] * rhs.uncertainty + rhs.belief[i] * self.uncertainty) / upu
                    });
                    u = 2.0 * self.uncertainty * rhs.uncertainty / upu;
                    a = array::from_fn(|i| (self.base_rate[i] + rhs.base_rate[i]) / 2.0);
                }
                Self::try_new(b, u, a)
            }

            /// Computes the weighted belief fusion of `self` and `rhs`.
            pub fn wfuse(&self, rhs: &Self, gamma_a: $ft) -> Result<Self, InvalidValueError> {
                let b;
                let u;
                let a;
                if self.uncertainty.approx_eq(0.0) && rhs.uncertainty.approx_eq(0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = array::from_fn(|i| gamma_a * self.belief[i] + gamma_b * rhs.belief[i]);
                    u = 0.0;
                    a = array::from_fn(|i| {
                        gamma_a * self.base_rate[i] + gamma_b * rhs.base_rate[i]
                    });
                } else if self.uncertainty.approx_eq(1.0) && rhs.uncertainty.approx_eq(1.0) {
                    b = [0.0; N];
                    u = 1.0;
                    a = array::from_fn(|i| (self.base_rate[i] + rhs.base_rate[i]) / 2.0);
                } else {
                    let denom = self.uncertainty + rhs.uncertainty
                        - 2.0 * self.uncertainty * rhs.uncertainty;
                    let ca = 1.0 - self.uncertainty;
                    let cb = 1.0 - rhs.uncertainty;
                    b = array::from_fn(|i| {
                        (self.belief[i] * ca * rhs.uncertainty
                            + rhs.belief[i] * cb * self.uncertainty)
                            / denom
                    });
                    u = (2.0 - self.uncertainty - rhs.uncertainty)
                        * self.uncertainty
                        * rhs.uncertainty
                        / denom;
                    a = array::from_fn(|i| {
                        (self.base_rate[i] * ca + rhs.base_rate[i] * cb)
                            / (2.0 - self.uncertainty - rhs.uncertainty)
                    });
                }
                Self::try_new(b, u, a)
            }
        }

        /// The probability projection of `self`.
        impl<T: Index<usize, Output = $ft>> MOpinion<T, $ft> {
            pub fn projection(&self, idx: usize) -> $ft {
                self.belief[idx] + self.base_rate[idx] * self.uncertainty
            }
        }

        impl<'a, const N: usize, const M: usize> Deduction<MSimplexes<'a, $ft, M, N>, [$ft; M]>
            for MOpinion1d<$ft, N>
        {
            type Output = MOpinion1d<$ft, M>;

            fn deduce(&self, cond: MSimplexes<'a, $ft, M, N>, ay: [$ft; M]) -> Self::Output {
                assert!(N > 0 && M > 1, "N > 0 and M > 1 must hold.");

                let ay: [$ft; M] = if (0..N)
                    .map(|i| cond[i].uncertainty)
                    .sum::<$ft>()
                    .approx_eq(N as $ft)
                {
                    ay
                } else {
                    array::from_fn(|j| {
                        let x = (0..N)
                            .map(|i| self.base_rate[i] * cond[i].belief[j])
                            .sum::<$ft>();
                        let y = (0..N)
                            .map(|i| self.base_rate[i] * cond[i].uncertainty)
                            .sum::<$ft>() as f64;
                        let z = 1.0f64 - y;
                        let w = x as f64 / z;
                        w as $ft
                    })
                };

                let cond_p: [[$ft; M]; N] = array::from_fn(|i| cond[i].projection(&ay));

                let pyhx: [$ft; M] =
                    array::from_fn(|j| (0..N).map(|i| self.base_rate[i] * cond_p[i][j]).sum());

                let uyhx = (0..M)
                    .map(|j| {
                        (pyhx[j]
                            - (0..N)
                                .map(|i| *cond[i].belief[j])
                                .reduce(<$ft>::min)
                                .unwrap())
                            / ay[j]
                    })
                    .reduce(<$ft>::min)
                    .unwrap();

                let u = uyhx
                    - (0..N)
                        .map(|i| (uyhx - cond[i].uncertainty) * self.belief[i])
                        .sum::<$ft>();
                let b: [$ft; M] = array::from_fn(|j| {
                    (0..N)
                        .map(|i| self.projection(i) * cond_p[i][j])
                        .sum::<$ft>()
                        - ay[j] * u
                });
                MOpinion1d::<$ft, M>::new_unchecked(b, u, ay)
            }

            fn projection(&self, cond: MSimplexes<$ft, M, N>, ay: [$ft; M]) -> [$ft; M] {
                let ay: [$ft; M] = if (0..N)
                    .map(|i| cond[i].uncertainty)
                    .sum::<$ft>()
                    .is_in_range(0.0, N as $ft)
                {
                    array::from_fn(|j| {
                        (0..N)
                            .map(|i| self.base_rate[i] * cond[i].belief[j])
                            .sum::<$ft>()
                            / (1.0
                                - (0..N)
                                    .map(|i| self.base_rate[i] * cond[i].uncertainty)
                                    .sum::<$ft>())
                    })
                } else {
                    ay
                };
                let cond_p: [[$ft; M]; N] = array::from_fn(|i| cond[i].projection(&ay));

                array::from_fn(|j| {
                    (0..N)
                        .map(|i| self.projection(i) * cond_p[i][j])
                        .sum::<$ft>()
                })
            }
        }
    };
}

impl_msl!(f32);
impl_msl!(f64);

macro_rules! impl_sl_conv {
    ($ft: ty) => {
        impl From<BOpinion<$ft>> for MOpinion1d<$ft, 2> {
            fn from(value: BOpinion<$ft>) -> Self {
                MOpinion1d::new_unchecked(
                    [value.belief, value.disbelief],
                    value.uncertainty,
                    [value.base_rate, 1.0 - value.base_rate],
                )
            }
        }

        impl From<&BOpinion<$ft>> for MOpinion1d<$ft, 2> {
            fn from(value: &BOpinion<$ft>) -> Self {
                MOpinion1d::new_unchecked(
                    [value.belief, value.disbelief],
                    value.uncertainty,
                    [value.base_rate, 1.0 - value.base_rate],
                )
            }
        }

        impl From<MOpinion1d<$ft, 2>> for BOpinion<$ft> {
            fn from(value: MOpinion1d<$ft, 2>) -> Self {
                BOpinion::<$ft>::new_unchecked(
                    value.belief[0],
                    value.belief[1],
                    value.uncertainty,
                    value.base_rate[0],
                )
            }
        }

        impl From<&MOpinion1d<$ft, 2>> for BOpinion<$ft> {
            fn from(value: &MOpinion1d<$ft, 2>) -> Self {
                BOpinion::<$ft>::new_unchecked(
                    value.belief[0],
                    value.belief[1],
                    value.uncertainty,
                    value.base_rate[0],
                )
            }
        }

        impl<'a, const N: usize> Deduction<MSimplexes<'a, $ft, 2, N>, $ft> for MOpinion1d<$ft, N> {
            type Output = BOpinion<$ft>;

            fn deduce(&self, cond: MSimplexes<'a, $ft, 2, N>, ay: $ft) -> Self::Output {
                self.deduce(cond, [ay, 1.0 - ay]).into()
            }

            fn projection(&self, cond: MSimplexes<$ft, 2, N>, ay: $ft) -> $ft {
                (0..N)
                    .map(|i| self.projection(i) * cond[i].projection(&[ay, 1.0 - ay])[0])
                    .sum::<$ft>()
            }
        }
    };
}

impl_sl_conv!(f32);
impl_sl_conv!(f64);

/// An error indicating that one or more invalid values are used.
#[derive(thiserror::Error, Debug)]
#[error("At least one parameter is invalid because {0}.")]
pub struct InvalidValueError(String);

trait EpsilonRange<T = Self> {
    fn is_in_range(self, from: T, to: T) -> bool;
}

trait EpsilonEq<T = Self> {
    fn approx_eq(self, to: T) -> bool;
}

macro_rules! impl_epsilon {
    ($ft: ty) => {
        impl EpsilonRange for $ft {
            fn is_in_range(self, from: Self, to: Self) -> bool {
                self >= from - <$ft>::EPSILON && self <= to + <$ft>::EPSILON
            }
        }
        impl EpsilonRange for &$ft {
            fn is_in_range(self, from: Self, to: Self) -> bool {
                *self >= *from - <$ft>::EPSILON && *self <= *to + <$ft>::EPSILON
            }
        }
        impl EpsilonEq for $ft {
            fn approx_eq(self, to: Self) -> bool {
                self >= to - <$ft>::EPSILON && self <= to + <$ft>::EPSILON
            }
        }
        impl EpsilonEq for &$ft {
            fn approx_eq(self, to: Self) -> bool {
                *self >= *to - <$ft>::EPSILON && *self <= *to + <$ft>::EPSILON
            }
        }
    };
}

impl_epsilon!(f32);
impl_epsilon!(f64);

impl<T> EpsilonEq for &BOpinion<T>
where
    for<'a> &'a T: EpsilonEq,
{
    fn approx_eq(self, to: Self) -> bool {
        self.belief.approx_eq(&to.belief)
            && self.disbelief.approx_eq(&to.disbelief)
            && self.uncertainty.approx_eq(&to.uncertainty)
            && self.base_rate.approx_eq(&to.base_rate)
    }
}

#[cfg(test)]
mod tests {
    use crate::{BOpinion, Deduction, EpsilonEq, MOpinion1d};

    #[test]
    fn test_bsl_boundary() {
        macro_rules! def {
            ($ft: ty) => {
                assert!(BOpinion::<$ft>::try_new(1.0, 0.0, 0.0, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.0, 1.0, 0.0, 1.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.0, 0.0, 1.0, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.5, 0.5, 0.0, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.0, 0.5, 0.5, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.5, 0.0, 0.5, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(-0.1, 0.1, 1.0, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.1, -0.1, 0.0, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.0, -0.1, 0.1, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(0.0, 1.1, -0.1, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.0, 0.1, -0.1, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.0, 0.0, 0.0, -0.1)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.0, 0.0, 0.0, 1.1)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(0.5, 0.5, 0.5, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
            };
        }

        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_msl_boundary() {
        macro_rules! def {
            ($ft: ty) => {
                assert!(MOpinion1d::<$ft, 2>::try_new([0.0, 0.0], 1.0, [0.0, 1.0]).is_ok());
                assert!(MOpinion1d::<$ft, 2>::try_new([0.0, 1.0], 0.0, [0.0, 1.0]).is_ok());
                assert!(MOpinion1d::<$ft, 2>::try_new([0.1, -0.1], 1.0, [0.0, 1.0])
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(MOpinion1d::<$ft, 2>::try_new([0.0, 1.0], 0.0, [1.1, -0.1])
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(MOpinion1d::<$ft, 2>::try_new([0.0, -1.0], 2.0, [1.1, -0.1])
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(MOpinion1d::<$ft, 2>::try_new([1.0, 1.0], -1.0, [1.1, -0.1])
                    .map_err(|e| println!("{e}"))
                    .is_err());
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_cum_fuse() {
        let w0 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
        let w1 = BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5);
        let w2 = BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5);
        assert!(w0.cfuse(&w2).is_ok());
        assert!(w1.cfuse(&w2).is_ok());
    }

    #[test]
    fn test_avg_fuse() {
        let w0 = BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5);
        let w1 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
        let w2 = BOpinion::<f32>::new(0.0, 0.5, 0.5, 0.5);
        let w3 = BOpinion::<f32>::new(0.0, 0.6, 0.4, 0.5);

        println!("{}", w0.afuse(&w1, 0.5).unwrap());
        println!("{}", w1.afuse(&w2, 0.5).unwrap());
        println!("{}", w1.afuse(&w3, 0.5).unwrap());
        println!("{}", w2.afuse(&w3, 0.5).unwrap());
    }

    #[test]
    fn test_wgt_fuse() {
        let w0 = BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5);
        let w1 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
        let w2 = BOpinion::<f32>::new(0.0, 0.5, 0.5, 0.5);
        let w3 = BOpinion::<f32>::new(0.0, 0.6, 0.4, 0.5);

        assert!(w1.wfuse(&w0, 0.5).unwrap().approx_eq(&w1));
        assert!(w2.wfuse(&w0, 0.5).unwrap().approx_eq(&w2));
        assert!(w3.wfuse(&w0, 0.5).unwrap().approx_eq(&w3));

        assert!(w2.wfuse(&w2, 0.5).unwrap().approx_eq(&w2));
        assert!(w3.wfuse(&w3, 0.5).unwrap().approx_eq(&w3));
    }

    #[test]
    fn test_fusion() {
        let w1 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
        // let w2 = BOpinion::<f32>::new(0.3, 0.0, 0.7, 0.5);
        let w2 = BOpinion::<f32>::new(0.0, 0.90, 0.10, 0.5);

        println!("{}", w1.cfuse(&w2).unwrap());
        println!("{}", w1.afuse(&w2, 0.5).unwrap());
        println!("{}", w1.wfuse(&w2, 0.5).unwrap());
    }

    #[test]
    fn test_bsl_deduction() {
        let cond = [
            BOpinion::<f32>::new(0.90, 0.02, 0.08, 0.5),
            BOpinion::<f32>::new(0.40, 0.52, 0.08, 0.5),
        ];
        let w = BOpinion::<f32>::new(0.00, 0.38, 0.62, 0.5);
        println!("{}", w.deduce((&cond).into(), 0.5));

        let cond = [
            BOpinion::<f32>::new(0.72, 0.18, 0.1, 0.5),
            BOpinion::<f32>::new(0.13, 0.57, 0.3, 0.5),
        ];
        let w = BOpinion::<f32>::new(0.7, 0.0, 0.3, 0.33);
        println!("{}", w.deduce((&cond).into(), 0.5));
    }

    #[test]
    fn test_deduction() {
        let b_a = 1.0;
        let b_xa = 0.0;
        let wa = MOpinion1d::<f32, 3>::new([b_a, 0.0, 0.0], 1.0 - b_a, [0.25, 0.25, 0.5]);
        let wxa = [
            BOpinion::<f32>::new(b_xa, 0.0, 1.0 - b_xa, 0.5),
            BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5),
            BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5),
        ];
        println!("{}", wxa[0]);
        let x = wa.deduce((&wxa).into(), 0.5);
        println!("{}", x.projection());
    }

    #[test]
    fn test_deduction1() {
        let w = BOpinion::<f32>::new(0.7, 0.0, 0.3, 1.0 / 3.0);
        let cond = [
            BOpinion::<f32>::new(0.72, 0.18, 0.1, 0.5),
            BOpinion::<f32>::new(0.13, 0.57, 0.3, 0.5),
        ];
        println!("{}", w.deduce((&cond).into(), 0.5));

        let wx = MOpinion1d::<f32, 2>::new([0.7, 0.0], 0.3, [1.0 / 3.0, 2.0 / 3.0]);
        let wyx = [
            MOpinion1d::<f32, 2>::new([0.72, 0.18], 0.1, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.13, 0.57], 0.3, [0.5, 0.5]),
        ];
        println!("{:?}", wx.deduce((&wyx).into(), [0.5, 0.5]));
    }

    #[test]
    fn test_deduction2() {
        let wa = MOpinion1d::<f32, 3>::new([0.7, 0.1, 0.0], 0.2, [0.3, 0.3, 0.4]);
        let wxa = [
            BOpinion::<f32>::new(0.7, 0.0, 0.3, 0.5),
            BOpinion::<f32>::new(0.0, 0.7, 0.3, 0.5),
            BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5),
        ];
        let wx = wa.deduce((&wxa).into(), 0.5);
        println!("{}|{}", wx, wx.projection());

        let wxa = [
            MOpinion1d::<f32, 2>::new([0.7, 0.0], 0.3, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.0, 0.7], 0.3, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.5, 0.5]),
        ];
        let wx = wa.deduce((&wxa).into(), [0.5, 0.5]);
        println!("{:?}|{}", wx, wx.projection(0));

        let wa = BOpinion::<f32>::new(0.7, 0.1, 0.2, 0.5);
        let wxa = [
            BOpinion::<f32>::new(0.7, 0.0, 0.3, 0.5),
            BOpinion::<f32>::new(0.0, 0.7, 0.3, 0.5),
        ];
        println!("{}", wa.deduce((&wxa).into(), 0.5));
    }

    #[test]
    fn test_deduction3() {
        let wa = MOpinion1d::<f32, 2>::new([0.7, 0.1], 0.2, [0.5, 0.5]);
        let wxa = [
            MOpinion1d::<f32, 3>::new([0.7, 0.0, 0.0], 0.3, [0.3, 0.3, 0.4]),
            MOpinion1d::<f32, 3>::new([0.0, 0.7, 0.0], 0.3, [0.3, 0.3, 0.4]),
        ];
        let wx = wa.deduce((&wxa).into(), [0.3, 0.3, 0.4]);
        println!("{:?}|{}", wx, wx.projection(0));
    }

    #[test]
    fn test_bo_deduction() {
        let wx = MOpinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.01, 0.99]);
        let ay = [0.01, 0.99];
        let cond = [
            MOpinion1d::<f32, 2>::new([0.99, 0.0], 0.01, ay.clone()),
            MOpinion1d::<f32, 2>::new([0.01, 0.98], 0.01, ay.clone()),
        ];
        let wy: BOpinion<f32> = wx.deduce((&cond).into(), ay).into();
        let m = BOpinion::<f32>::new(0.72, 0.18, 0.1, 0.5);

        println!("{:?}, {}", wy, wy.projection());

        let wym = wy.cfuse(&m).unwrap();
        println!("{:?}, {}", wym, wym.projection());

        let mut wyd = wy;
        for _ in 0..5 {
            wyd = wyd.cfuse(&wyd).unwrap();
        }
        let wym = wyd.cfuse(&m).unwrap();
        println!("{:?}, {}", wym, wym.projection());
        // println!("{:?}, {}", wyd, wyd.projection());
    }

    #[test]
    fn test_mo_deduction() {
        let wx = MOpinion1d::<f32, 2>::new([0.9, 0.0], 0.1, [0.1, 0.9]);
        let ay = [1.0, 0.0, 0.0];
        let wyx = [
            MOpinion1d::<f32, 3>::new([0.0, 0.8, 0.1], 0.10, ay.clone()),
            MOpinion1d::<f32, 3>::new([0.7, 0.0, 0.1], 0.20, ay.clone()),
        ];
        println!("{:?}", wx.deduce((&wyx).into(), ay))
    }
}
