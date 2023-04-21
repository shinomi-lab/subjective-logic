//!
//! An implementation of [Subjective Logic](https://en.wikipedia.org/wiki/Subjective_logic).

use approx::{ulps_eq, ulps_ne, AbsDiffEq, RelativeEq, UlpsEq};
use std::{array, fmt::Display, ops::Index};

fn zip_arrays_into<T, U, V, const N: usize>(ts: [T; N], us: [U; N]) -> [V; N]
where
    V: From<(T, U)>,
{
    let mut ts = Vec::from(ts);
    let mut us = Vec::from(us);
    array::from_fn(|i| {
        let t = ts.swap_remove(N - 1 - i);
        let u = us.swap_remove(N - 1 - i);
        (t, u).into()
    })
}

fn each_into<'a, T, U, const N: usize>(arr: &'a [T; N]) -> [&U; N]
where
    &'a T: Into<&'a U>,
{
    array::from_fn(|i| (&arr[i]).into())
}

/// A binomial opinion.
#[derive(Debug, PartialEq)]
pub struct BOpinion<T> {
    pub simplex: BSimplex<T>,
    pub base_rate: T,
}

impl<T: Display> Display for BOpinion<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{},{},{}", self.b(), self.d(), self.u(), self.a())
    }
}

impl<T> BOpinion<T> {
    fn new_unchecked(b: T, d: T, u: T, a: T) -> Self {
        Self {
            simplex: BSimplex::new_unchecked(b, d, u),
            base_rate: a,
        }
    }

    pub fn b(&self) -> &T {
        &self.simplex.b()
    }

    pub fn d(&self) -> &T {
        &self.simplex.d()
    }

    pub fn u(&self) -> &T {
        self.simplex.u()
    }

    pub fn a(&self) -> &T {
        &self.base_rate
    }
}

macro_rules! impl_bop {
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
                if ulps_ne!(b + d + u, 1.0) {
                    return Err(InvalidValueError(
                        "b + d + u = 1 is not satisfied".to_string(),
                    ));
                }
                if b.out_of_range(0.0, 1.0) {
                    return Err(InvalidValueError("b ∈ [0,1] is not satisfied".to_string()));
                }
                if d.out_of_range(0.0, 1.0) {
                    return Err(InvalidValueError("d ∈ [0,1] is not satisfied".to_string()));
                }
                if u.out_of_range(0.0, 1.0) {
                    return Err(InvalidValueError("u ∈ [0,1] is not satisfied".to_string()));
                }
                if a.out_of_range(0.0, 1.0) {
                    return Err(InvalidValueError("a ∈ [0,1] is not satisfied".to_string()));
                }
                Ok(Self::new_unchecked(b, d, u, a))
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
                self.b() + self.a() * self.u()
            }

            /// Computes the opinion on the logical conjunction of `self` and `rhs`.
            pub fn mul(&self, rhs: &Self) -> Self {
                let a = self.base_rate * rhs.base_rate;
                let b = self.b() * rhs.b()
                    + ((1.0 - self.base_rate) * rhs.base_rate * self.b() * rhs.u()
                        + (1.0 - rhs.base_rate) * self.base_rate * rhs.b() * self.u())
                        / (1.0 - a);
                let d = self.d() + rhs.d() - self.d() * rhs.d();
                let u = self.u() * rhs.u()
                    + ((1.0 - rhs.base_rate) * self.b() * rhs.u()
                        + (1.0 - self.b()) * rhs.b() * self.u())
                        / (1.0 - a);
                Self::new(b, d, u, a)
            }

            /// Computes the opinion on the logical disjunction of `self` and `rhs`.
            pub fn comul(&self, rhs: &Self) -> Self {
                let a = self.base_rate + rhs.base_rate - self.base_rate * rhs.base_rate;
                let b = self.b() + rhs.b() - self.b() * rhs.b();
                let d = self.d() * rhs.d()
                    + (self.base_rate * (1.0 - rhs.base_rate) * self.d() * rhs.u()
                        + rhs.base_rate * (1.0 - self.base_rate) * rhs.d() * self.u())
                        / a;
                let u = self.u() * rhs.u()
                    + (rhs.base_rate * self.b() * rhs.u() + self.base_rate * rhs.b() * self.u())
                        / a;
                Self::new(b, d, u, a)
            }

            /// Computes the cumulative fusion of `self` and `rhs`.
            pub fn cfuse(&self, rhs: &Self) -> Result<Self, InvalidValueError> {
                let uu = self.u() * rhs.u();
                let kappa = self.u() + rhs.u() - uu;
                let b = (self.b() * rhs.u() + rhs.b() * self.u()) / kappa;
                let d = (self.d() * rhs.u() + rhs.d() * self.u()) / kappa;
                let u = (self.u() * rhs.u()) / kappa;
                let a = if ulps_eq!(*self.u(), 1.0) && ulps_eq!(*rhs.u(), 1.0) {
                    (self.base_rate + rhs.base_rate) / 2.0
                } else {
                    (self.base_rate * rhs.u() + rhs.base_rate * self.u()
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
                if ulps_eq!(*self.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = gamma_a * self.b() + gamma_b * rhs.b();
                    d = gamma_a * self.d() + gamma_b * rhs.d();
                    u = 0.0;
                    a = gamma_a * self.base_rate + gamma_b * rhs.base_rate;
                } else {
                    let upu = self.u() + rhs.u();
                    b = (self.b() * rhs.u() + rhs.b() * self.u()) / upu;
                    d = (self.d() * rhs.u() + rhs.d() * self.u()) / upu;
                    u = 2.0 * self.u() * rhs.u() / upu;
                    a = (self.base_rate + rhs.base_rate) / 2.0;
                }
                Self::try_new(b, d, u, a)
            }

            /// Computes the weighted b() fusion of `self` and `rhs`.
            pub fn wfuse(&self, rhs: &Self, gamma_a: $ft) -> Result<Self, InvalidValueError> {
                let b;
                let d;
                let u;
                let a;
                if ulps_eq!(*self.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = gamma_a * self.b() + gamma_b * rhs.b();
                    d = gamma_a * self.d() + gamma_b * rhs.d();
                    u = 0.0;
                    a = gamma_a * self.base_rate + gamma_b * rhs.base_rate;
                } else if ulps_eq!(*self.u(), 1.0) && ulps_eq!(*rhs.u(), 1.0) {
                    b = 0.0;
                    d = 0.0;
                    u = 1.0;
                    a = (self.base_rate + rhs.base_rate) / 2.0;
                } else {
                    let denom = self.u() + rhs.u() - 2.0 * self.u() * rhs.u();
                    let ca = 1.0 - self.u();
                    let cb = 1.0 - rhs.u();
                    b = (self.b() * ca * rhs.u() + rhs.b() * cb * self.u()) / denom;
                    d = (self.d() * ca * rhs.u() + rhs.d() * cb * self.u()) / denom;
                    u = (2.0 - self.u() - rhs.u()) * self.u() * rhs.u() / denom;
                    a = (self.base_rate * ca + rhs.base_rate * cb) / (2.0 - self.u() - rhs.u());
                }
                Self::try_new(b, d, u, a)
            }

            /// Computes the conditionally deduced opinion of `self` by a two length array of conditional opinions `cond`.
            /// If `self.u()` is equal to `0.0`, this function panics.
            pub fn deduce<'a, A>(&self, cond: &'a [A; 2], ay: $ft) -> Self
            where
                &'a A: Into<&'a BSimplex<$ft>>,
            {
                let cond: [&BSimplex<$ft>; 2] = [(&cond[0]).into(), (&cond[1]).into()];
                let rvax = (1.0 - self.base_rate);
                let bi = self.b() * cond[0].b()
                    + self.d() * cond[1].b()
                    + self.u() * (cond[0].b() * self.base_rate + cond[1].b() * rvax);
                let di = self.b() * cond[0].d()
                    + self.d() * cond[1].d()
                    + self.u() * (cond[0].d() * self.base_rate + cond[1].d() * rvax);
                let ui = self.b() * cond[0].u()
                    + self.d() * cond[1].u()
                    + self.u() * (cond[0].u() * self.base_rate + cond[1].u() * rvax);
                let k = match (cond[0].b() > cond[1].b(), cond[0].d() > cond[1].d()) {
                    // Case I
                    (true, true) | (false, false) => 0.0,
                    (bp, _) => {
                        let pyx = cond[0].b() * self.base_rate
                            + cond[1].b() * rvax
                            + ay * (cond[0].u() * self.base_rate + cond[1].u() * rvax);
                        let px = self.projection();
                        let r = cond[1].b() + ay * (1.0 - cond[1].b() - cond[0].d());
                        match (pyx > r, px > self.base_rate) {
                            (false, false) => {
                                if bp {
                                    // Case II.A.1
                                    self.base_rate * self.u() * (bi - cond[1].b()) / (px * ay)
                                } else {
                                    // Case III.A.1
                                    rvax * self.u()
                                        * (di - cond[1].d())
                                        * (cond[1].b() - cond[0].b())
                                        / (px * ay * (cond[0].d() - cond[1].d()))
                                }
                            }
                            (false, true) => {
                                if bp {
                                    // Case II.A.2
                                    self.base_rate
                                        * self.u()
                                        * (di - cond[0].d())
                                        * (cond[0].b() - cond[1].b())
                                        / ((1.0 - px) * ay * (cond[1].d() - cond[0].d()))
                                } else {
                                    // Case III.A.2
                                    rvax * self.u() * (bi - cond[0].b()) / ((1.0 - px) * ay)
                                }
                            }
                            (true, false) => {
                                if bp {
                                    // Case II.B.1
                                    rvax * self.u()
                                        * (bi - cond[1].b())
                                        * (cond[1].d() - cond[0].d())
                                        / (px * (1.0 - ay) * (cond[0].b() - cond[1].b()))
                                } else {
                                    // Case III.B.1
                                    self.base_rate * self.u() * (di - cond[1].d())
                                        / (px * (1.0 - ay))
                                }
                            }
                            (true, true) => {
                                if bp {
                                    // Case II.B.2
                                    rvax * self.u() * (di - cond[0].d()) / ((1.0 - px) * (1.0 - ay))
                                } else {
                                    // Case III.B.2
                                    self.base_rate
                                        * self.u()
                                        * (bi - cond[0].b())
                                        * (cond[0].d() - cond[1].d())
                                        / ((1.0 - px) * (1.0 - ay) * (cond[1].b() - cond[0].b()))
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

            /// Computes the u() favouring discounted opinion.
            pub fn trans_unc(&self, b: $ft) -> Self {
                assert!(b.is_in_range(0.0, 1.0), "b ∈ [0,1] is not satisfied.");
                Self::new(
                    b * self.b(),
                    b * self.d(),
                    1.0 - b + b * self.u(),
                    self.base_rate,
                )
            }

            /// Computes the opposite b() favouring discounted opinion.
            pub fn trans_opp(&self, b: $ft, d: $ft) -> Self {
                let u = 1.0 - b - d;
                assert!(u.is_in_range(0.0, 1.0), "b + d ∈ [0,1] is not satisfied.");
                Self::new(
                    b * self.b() + d * self.d(),
                    b * self.d() + d * self.b(),
                    u + (b + d) * self.u(),
                    self.base_rate,
                )
            }

            /// Computes base rate sensitive discounted opinion.
            pub fn trans_bsr(&self, ev: $ft) -> Self {
                assert!(ev.is_in_range(0.0, 1.0), "ev ∈ [0,1] is not satisfied.");
                Self::new(
                    ev * self.b(),
                    ev * self.d(),
                    1.0 - ev * (self.b() + self.d()),
                    self.base_rate,
                )
            }
        }

        impl AbsDiffEq for BOpinion<$ft> {
            type Epsilon = <$ft as AbsDiffEq>::Epsilon;

            fn default_epsilon() -> Self::Epsilon {
                <$ft as AbsDiffEq>::default_epsilon()
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                self.b().abs_diff_eq(other.b(), epsilon)
                    && self.d().abs_diff_eq(other.d(), epsilon)
                    && self.u().abs_diff_eq(other.u(), epsilon)
                    && self.a().abs_diff_eq(other.a(), epsilon)
            }
        }

        impl RelativeEq for BOpinion<$ft> {
            fn default_max_relative() -> Self::Epsilon {
                <$ft as RelativeEq>::default_max_relative()
            }

            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                self.b().relative_eq(other.b(), epsilon, max_relative)
                    && self.d().relative_eq(other.d(), epsilon, max_relative)
                    && self.u().relative_eq(other.u(), epsilon, max_relative)
                    && self.a().relative_eq(other.a(), epsilon, max_relative)
            }
        }

        impl UlpsEq for BOpinion<$ft> {
            fn default_max_ulps() -> u32 {
                <$ft as UlpsEq>::default_max_ulps()
            }

            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                self.b().ulps_eq(other.b(), epsilon, max_ulps)
                    && self.d().ulps_eq(other.d(), epsilon, max_ulps)
                    && self.u().ulps_eq(other.u(), epsilon, max_ulps)
                    && self.a().ulps_eq(other.a(), epsilon, max_ulps)
            }
        }
    };
}

impl_bop!(f32);
impl_bop!(f64);

/// The simplex of a binomial opinion, from which a base rate is excluded.
#[derive(Debug, PartialEq)]
pub struct BSimplex<T>(MSimplex<T, 2>);

impl<T> BSimplex<T> {
    fn new_unchecked(b: T, d: T, u: T) -> Self {
        Self(MSimplex::new_unchecked([b, d], u))
    }

    fn b(&self) -> &T {
        &self.0.belief[0]
    }

    fn d(&self) -> &T {
        &self.0.belief[1]
    }

    fn u(&self) -> &T {
        &self.0.uncertainty
    }
}

impl<'a, T> From<&'a BOpinion<T>> for &'a BSimplex<T> {
    fn from(value: &'a BOpinion<T>) -> Self {
        &value.simplex
    }
}

impl<'a, T> From<&'a BOpinion<T>> for &'a MSimplex<T, 2> {
    fn from(value: &'a BOpinion<T>) -> Self {
        &value.simplex.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MSimplexBase<T, U> {
    belief: T,
    uncertainty: U,
}

impl<T: Display, U: Display> Display for MSimplexBase<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.belief, self.uncertainty)
    }
}

impl<T, U> MSimplexBase<T, U> {
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

impl<T, U> From<(T, U)> for MSimplexBase<T, U> {
    fn from(value: (T, U)) -> Self {
        MSimplexBase::new_unchecked(value.0, value.1)
    }
}

/// The generlized structure of a multinomial opinion.
#[derive(Debug, Clone, PartialEq)]
pub struct MOpinionBase<S, T> {
    pub simplex: S,
    pub base_rate: T,
}

impl<S, T> MOpinionBase<S, T> {
    pub fn from_simplex_unchecked(s: S, a: T) -> Self {
        Self {
            simplex: s,
            base_rate: a,
        }
    }
}

impl<S: Display, T: Display> Display for MOpinionBase<S, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.simplex, self.base_rate)
    }
}

pub type MOpinion<T, U> = MOpinionBase<MSimplexBase<T, U>, T>;
type MOpinionRef<'a, T, U> = MOpinionBase<&'a MSimplexBase<T, U>, &'a T>;

impl<T, U> MOpinion<T, U> {
    fn new_unchecked(b: T, u: U, a: T) -> Self {
        Self {
            simplex: MSimplexBase::new_unchecked(b, u),
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

impl<T, U> MOpinionRef<'_, T, U> {
    #[inline]
    pub fn b(&self) -> &T {
        &self.simplex.belief
    }

    #[inline]
    pub fn u(&self) -> &U {
        &self.simplex.uncertainty
    }
}

impl<'a, T, U> From<&'a MOpinion<T, U>> for &'a MSimplexBase<T, U> {
    fn from(value: &'a MOpinion<T, U>) -> Self {
        &value.simplex
    }
}

impl<'a, 'b: 'a, T, U> From<(&'a MSimplexBase<T, U>, &'b T)> for MOpinionRef<'a, T, U> {
    fn from(value: (&'a MSimplexBase<T, U>, &'b T)) -> Self {
        MOpinionBase {
            simplex: &value.0,
            base_rate: &value.1,
        }
    }
}

impl<'a, T, U> From<&'a MOpinion<T, U>> for MOpinionRef<'a, T, U> {
    fn from(value: &'a MOpinion<T, U>) -> Self {
        MOpinionBase {
            simplex: &value.simplex,
            base_rate: &value.base_rate,
        }
    }
}

/// The reference of a simplex of a multinomial opinion, from which a base rate is excluded.
pub type MSimplex<T, const N: usize> = MSimplexBase<[T; N], T>;

/// A multinomial opinion with 1-dimensional vectors.
pub type MOpinion1d<T, const N: usize> = MOpinion<[T; N], T>;

/// The reference type of a multinomial opinion with 1-dimensional vectors.
type MOpinion1dRef<'a, T, const N: usize> = MOpinionRef<'a, [T; N], T>;

macro_rules! impl_msimplex {
    ($ft: ty) => {
        impl<const N: usize> MSimplex<$ft, N> {
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

        impl<const M: usize, const N: usize> MBR<$ft, N, M> for [&MSimplex<$ft, N>; M] {
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

        impl<const M: usize, const N: usize> MBR<$ft, N, M> for [MSimplex<$ft, N>; M] {
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

trait MBR<T, const N: usize, const M: usize> {
    fn marginal_base_rate(&self, ax: &[T; M]) -> Option<[T; N]>;
}

impl_msimplex!(f32);
impl_msimplex!(f64);

pub enum FusionOp<U> {
    CumulativeA(U),
    CumulativeE(U),
    Averaging(U),
    Weighted(U),
}

pub trait FusionAssign<Rhs, U> {
    type Output;
    fn fusion_assign(&mut self, rhs: Rhs, op: &FusionOp<U>) -> Self::Output;
}

pub trait Fusion<Rhs, U> {
    type Output;

    /// Computes the aleatory cumulative fusion of `self` and `rhs`.
    fn cfuse_al(&self, rhs: Rhs, gamma_a: U) -> Self::Output;

    /// Computes the epistemic cumulative fusion of `self` and `rhs`.
    fn cfuse_ep(&self, rhs: Rhs, gamma_a: U) -> Self::Output;

    /// Computes the averaging belief fusion of `self` and `rhs`.
    fn afuse(&self, rhs: Rhs, gamma_a: U) -> Self::Output;

    /// Computes the weighted belief fusion of `self` and `rhs`.
    fn wfuse(&self, rhs: Rhs, gamma_a: U) -> Self::Output;
}

/// The deduction operator.
pub trait Deduction<Rhs, U> {
    type Output;

    /// Computes the conditionally deduced opinion of `self` with a base rate vector `ay`
    /// by `conds` representing a collection of conditional opinions.
    fn deduce(self, conds: Rhs, ay: U) -> Self::Output;
}

/// The abduction operator.
pub trait Abduction<Rhs, U, V>: Sized {
    type Output;

    /// Computes the conditionally abduced opinion of `self` with a base rate vector `ax`
    /// by `conds` representing a collection of conditional opinions.
    /// If a marginal base rate cannot be computed from `conds`, ay is used instead.
    fn abduce(self, conds: Rhs, ax: U, ay: Option<V>) -> Option<(Self::Output, V)>;
}

macro_rules! impl_mop {
    ($ft: ty) => {
        impl<const N: usize> MOpinion1d<$ft, N> {
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
                s: MSimplex<$ft, N>,
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

        impl<'a, const N: usize> MOpinion1dRef<'a, $ft, N> {
            /// Returns the uncertainty maximized opinion of `self`.
            pub fn max_uncertainty(&self) -> $ft {
                (0..N)
                    .map(|i| self.projection(i) / self.base_rate[i])
                    .reduce(<$ft>::min)
                    .unwrap()
            }
        }

        /// The probability projection of `self`.
        impl<T: Index<usize, Output = $ft>> MOpinion<T, $ft> {
            pub fn projection(&self, idx: usize) -> $ft {
                self.b()[idx] + self.base_rate[idx] * self.u()
            }
        }

        /// The probability projection of `self`.
        impl<'a, T: Index<usize, Output = $ft>> MOpinionRef<'a, T, $ft> {
            pub fn projection(&self, idx: usize) -> $ft {
                self.b()[idx] + self.base_rate[idx] * *self.u()
            }
        }
    };
}

impl_mop!(f32);
impl_mop!(f64);

macro_rules! impl_fusion {
    ($ft: ty) => {
        impl<const N: usize> Fusion<&MSimplex<$ft, N>, $ft> for MOpinion1d<$ft, N> {
            type Output = Result<MSimplex<$ft, N>, InvalidValueError>;
            fn cfuse_al(&self, rhs: &MSimplex<$ft, N>, gamma_a: $ft) -> Self::Output {
                let b;
                let u;
                if ulps_eq!(*self.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = array::from_fn(|i| gamma_a * self.b()[i] + gamma_b * rhs.b()[i]);
                    u = 0.0;
                } else {
                    let rhs_u = *rhs.u();
                    let temp = self.u() + rhs_u - self.u() * rhs_u;
                    b = array::from_fn(|i| (self.b()[i] * rhs_u + rhs.b()[i] * self.u()) / temp);
                    u = self.u() * rhs_u / temp;
                }
                MSimplex::<$ft, N>::try_new(b, u)
            }

            fn cfuse_ep(&self, rhs: &MSimplex<$ft, N>, gamma_a: $ft) -> Self::Output {
                let s = self.cfuse_al(rhs, gamma_a)?;
                let w = MOpinion1dRef::<$ft, N>::from((&s, &self.base_rate));
                let u_max = w.max_uncertainty();
                let b_max = array::from_fn(|i| w.projection(i) - w.base_rate[i] * u_max);
                MSimplex::<$ft, N>::try_new(b_max, u_max)
            }

            fn afuse(&self, rhs: &MSimplex<$ft, N>, gamma_a: $ft) -> Self::Output {
                let b;
                let u;
                if ulps_eq!(*self.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = array::from_fn(|i| gamma_a * self.b()[i] + gamma_b * rhs.b()[i]);
                    u = 0.0;
                } else {
                    let rhs_u = *rhs.u();
                    let upu = self.u() + rhs_u;
                    b = array::from_fn(|i| (self.b()[i] * rhs_u + rhs.b()[i] * self.u()) / upu);
                    u = 2.0 * self.u() * rhs_u / upu;
                }
                MSimplex::<$ft, N>::try_new(b, u)
            }

            fn wfuse(&self, rhs: &MSimplex<$ft, N>, gamma_a: $ft) -> Self::Output {
                let b;
                let u;
                if ulps_eq!(*self.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = array::from_fn(|i| gamma_a * self.b()[i] + gamma_b * rhs.b()[i]);
                    u = 0.0;
                } else if ulps_eq!(*self.u(), 1.0) && ulps_eq!(*rhs.u(), 1.0) {
                    b = [0.0; N];
                    u = 1.0;
                } else {
                    let rhs_u = *rhs.u();
                    let denom = self.u() + rhs_u - 2.0 * self.u() * rhs_u;
                    let ca = 1.0 - self.u();
                    let cb = 1.0 - rhs_u;
                    b = array::from_fn(|i| {
                        (self.b()[i] * ca * rhs_u + rhs.b()[i] * cb * self.u()) / denom
                    });
                    u = (2.0 - self.u() - rhs_u) * self.u() * rhs_u / denom;
                }
                MSimplex::<$ft, N>::try_new(b, u)
            }
        }

        impl<'a, A, const N: usize> Fusion<A, $ft> for MOpinion1d<$ft, N>
        where
            A: Into<MOpinion1dRef<'a, $ft, N>>,
        {
            type Output = Result<Self, InvalidValueError>;

            fn cfuse_al(&self, rhs: A, gamma_a: $ft) -> Self::Output {
                let rhs = rhs.into();
                let sr = rhs.clone().simplex;
                let s = self.cfuse_al(sr, gamma_a)?;
                let a = if ulps_eq!(*self.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    array::from_fn(|i| {
                        gamma_a * self.base_rate[i] + (1.0 - gamma_a) * rhs.base_rate[i]
                    })
                } else if ulps_eq!(*self.u(), 1.0) && ulps_eq!(*rhs.u(), 1.0) {
                    array::from_fn(|i| (self.base_rate[i] + rhs.base_rate[i]) / 2.0)
                } else {
                    let rhs_u = *rhs.u();
                    array::from_fn(|i| {
                        (self.base_rate[i] * rhs_u + rhs.base_rate[i] * self.u()
                            - (self.base_rate[i] + rhs.base_rate[i]) * self.u() * rhs_u)
                            / (self.u() + rhs_u - self.u() * rhs_u * 2.0)
                    })
                };
                Self::try_from_simplex(s, a)
            }

            fn cfuse_ep(&self, rhs: A, gamma_a: $ft) -> Self::Output {
                let w = self.cfuse_al(rhs, gamma_a)?;
                w.op_u_max()
            }

            fn afuse(&self, rhs: A, gamma_a: $ft) -> Self::Output {
                let rhs = rhs.into();
                let sr = rhs.clone().simplex;
                let s = self.afuse(sr, gamma_a)?;
                let a = if ulps_eq!(*self.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    array::from_fn(|i| {
                        gamma_a * self.base_rate[i] + (1.0 - gamma_a) * rhs.base_rate[i]
                    })
                } else {
                    array::from_fn(|i| (self.base_rate[i] + rhs.base_rate[i]) / 2.0)
                };
                Self::try_from_simplex(s, a)
            }

            fn wfuse(&self, rhs: A, gamma_a: $ft) -> Self::Output {
                let rhs = rhs.into();
                let sr = rhs.clone().simplex;
                let s = self.wfuse(sr, gamma_a)?;
                let a = if ulps_eq!(*self.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    array::from_fn(|i| {
                        gamma_a * self.base_rate[i] + (1.0 - gamma_a) * rhs.base_rate[i]
                    })
                } else if ulps_eq!(*self.u(), 1.0) && ulps_eq!(*rhs.u(), 1.0) {
                    array::from_fn(|i| (self.base_rate[i] + rhs.base_rate[i]) / 2.0)
                } else {
                    let rhs_u = *rhs.u();
                    let ca = 1.0 - self.u();
                    let cb = 1.0 - rhs_u;
                    array::from_fn(|i| {
                        (self.base_rate[i] * ca + rhs.base_rate[i] * cb) / (2.0 - self.u() - rhs_u)
                    })
                };
                Self::try_from_simplex(s, a)
            }
        }

        impl<const N: usize> FusionAssign<&MSimplex<$ft, N>, $ft> for MOpinion1d<$ft, N> {
            type Output = Result<(), InvalidValueError>;

            fn fusion_assign(
                &mut self,
                rhs: &MSimplex<$ft, N>,
                op: &FusionOp<$ft>,
            ) -> Self::Output {
                self.simplex = match *op {
                    FusionOp::CumulativeA(gamma_a) => self.cfuse_al(rhs, gamma_a),
                    FusionOp::CumulativeE(gamma_a) => self.cfuse_ep(rhs, gamma_a),
                    FusionOp::Averaging(gamma_a) => self.afuse(rhs, gamma_a),
                    FusionOp::Weighted(gamma_a) => self.wfuse(rhs, gamma_a),
                }?;
                Ok(())
            }
        }

        impl<const N: usize> FusionAssign<&MOpinion1d<$ft, N>, $ft> for MOpinion1d<$ft, N> {
            type Output = Result<(), InvalidValueError>;

            fn fusion_assign(
                &mut self,
                rhs: &MOpinion1d<$ft, N>,
                op: &FusionOp<$ft>,
            ) -> Self::Output {
                *self = match *op {
                    FusionOp::CumulativeA(gamma_a) => self.cfuse_al(rhs, gamma_a),
                    FusionOp::CumulativeE(gamma_a) => self.cfuse_ep(rhs, gamma_a),
                    FusionOp::Averaging(gamma_a) => self.afuse(rhs, gamma_a),
                    FusionOp::Weighted(gamma_a) => self.wfuse(rhs, gamma_a),
                }?;
                Ok(())
            }
        }
    };
}

impl_fusion!(f32);
impl_fusion!(f64);

macro_rules! impl_deduction {
    ($ft: ty) => {
        impl<const M: usize, const N: usize> Deduction<[MSimplex<$ft, N>; M], [$ft; N]>
            for &MOpinion1d<$ft, M>
        {
            type Output = MOpinion1d<$ft, N>;
            fn deduce(self, conds: [MSimplex<$ft, N>; M], ay: [$ft; N]) -> Self::Output {
                self.deduce(&conds, ay)
            }
        }

        impl<'a, A, const M: usize, const N: usize> Deduction<&'a [A; M], [$ft; N]>
            for &'a MOpinion1d<$ft, M>
        where
            &'a A: Into<&'a MSimplex<$ft, N>>,
        {
            type Output = MOpinion1d<$ft, N>;

            fn deduce(self, conds: &'a [A; M], ay: [$ft; N]) -> Self::Output {
                MOpinion1dRef::from(self).deduce(conds, ay)
            }
        }

        impl<'a, A, const M: usize, const N: usize> Deduction<&'a [A; M], [$ft; N]>
            for MOpinion1dRef<'a, $ft, M>
        where
            &'a A: Into<&'a MSimplex<$ft, N>>,
        {
            type Output = MOpinion1d<$ft, N>;

            fn deduce(self, conds: &'a [A; M], ay: [$ft; N]) -> Self::Output {
                assert!(M > 0 && N > 1, "N > 0 and M > 1 must hold.");
                let conds: [&MSimplex<$ft, N>; M] = each_into(conds);
                let ay = conds.marginal_base_rate(&self.base_rate).unwrap_or(ay);

                let cond_p: [[$ft; N]; M] = array::from_fn(|i| conds[i].projection(&ay));
                let pyhx: [$ft; N] =
                    array::from_fn(|j| (0..M).map(|i| self.base_rate[i] * cond_p[i][j]).sum());
                let uyhx = (0..N)
                    .map(|j| {
                        (pyhx[j]
                            - (0..M)
                                .map(|i| conds[i].belief[j])
                                .reduce(<$ft>::min)
                                .unwrap())
                            / ay[j]
                    })
                    .reduce(<$ft>::min)
                    .unwrap();

                let u = uyhx
                    - (0..M)
                        .map(|i| (uyhx - conds[i].uncertainty) * self.b()[i])
                        .sum::<$ft>();
                let b: [$ft; N] = array::from_fn(|j| {
                    (0..M)
                        .map(|i| self.projection(i) * cond_p[i][j])
                        .sum::<$ft>()
                        - ay[j] * u
                });
                MOpinion1d::<$ft, N>::new_unchecked(b, u, ay)
            }
        }

        impl<'a, A, const N: usize> Deduction<&'a [A; N], $ft> for &'a MOpinion1d<$ft, N>
        where
            &'a A: Into<&'a MSimplex<$ft, 2>>,
        {
            type Output = BOpinion<$ft>;

            fn deduce(self, cond: &'a [A; N], ay: $ft) -> Self::Output {
                self.deduce(cond, [ay, 1.0 - ay]).into()
            }
        }
    };
}

impl_deduction!(f32);
impl_deduction!(f64);

macro_rules! impl_abduction {
    ($ft: ty) => {
        impl<const N: usize, const M: usize> Abduction<&[MSimplex<$ft, N>; M], [$ft; M], [$ft; N]>
            for &MSimplex<$ft, N>
        {
            type Output = MOpinion1d<$ft, M>;

            fn abduce(
                self,
                conds_yx: &[MSimplex<$ft, N>; M],
                ax: [$ft; M],
                ay: Option<[$ft; N]>,
            ) -> Option<(Self::Output, [$ft; N])> {
                fn inverse<'a, const N: usize, const M: usize>(
                    conds_yx: &[MSimplex<$ft, N>; M],
                    ax: &[$ft; M],
                    ay: &[$ft; N],
                ) -> [MSimplex<$ft, M>; N] {
                    let p_yx: [[$ft; N]; M] = array::from_fn(|i| conds_yx[i].projection(ay));
                    let p_xy: [[$ft; M]; N] = array::from_fn(|j| {
                        array::from_fn(|i| {
                            ax[i] * p_yx[i][j] / (0..M).map(|k| ax[k] * p_yx[k][j]).sum::<$ft>()
                        })
                    });
                    let u_yx_sum = conds_yx.iter().map(|cond| cond.uncertainty).sum::<$ft>();
                    let irrelevance_yx: [$ft; N] = array::from_fn(|j| {
                        1.0 - (0..M).map(|i| p_yx[i][j]).reduce(<$ft>::max).unwrap()
                            + (0..M).map(|i| p_yx[i][j]).reduce(<$ft>::min).unwrap()
                    });
                    let weights_yx = if u_yx_sum == 0.0 {
                        [0.0; M]
                    } else {
                        array::from_fn(|i| conds_yx[i].uncertainty / u_yx_sum)
                    };
                    let u_yx_marginal: [$ft; M] = array::from_fn(|i| {
                        (0..N)
                            .map(|j| p_yx[i][j] / ay[j])
                            .reduce(<$ft>::min)
                            .unwrap()
                    });
                    let u_yx_weight: [$ft; M] = array::from_fn(|i| {
                        let tmp = u_yx_marginal[i];
                        if tmp == 0.0 {
                            0.0
                        } else {
                            weights_yx[i] * conds_yx[i].uncertainty / tmp
                        }
                    });
                    let u_yx_exp: $ft = u_yx_weight.into_iter().sum();
                    let u_xy_marginal: [$ft; N] = array::from_fn(|j| {
                        (0..M)
                            .map(|i| p_yx[i][j] / (0..M).map(|k| ax[k] * p_yx[k][j]).sum::<$ft>())
                            .reduce(<$ft>::min)
                            .unwrap()
                    });
                    let u_xy_inv: [$ft; N] = array::from_fn(|j| {
                        u_xy_marginal[j] * (u_yx_exp + (1.0 - u_yx_exp) * irrelevance_yx[j])
                    });
                    let bs: [[$ft; M]; N] =
                        array::from_fn(|j| array::from_fn(|i| p_xy[j][i] - u_xy_inv[j] * ax[i]));
                    zip_arrays_into(bs, u_xy_inv)
                }

                let ay = conds_yx.marginal_base_rate(&ax).or(ay)?;
                let inv_conds = inverse(conds_yx, &ax, &ay);
                let w = MOpinion1dRef::<$ft, N>::from((self, &ay));
                Some((w.deduce(&inv_conds, ax), ay))
            }
        }
    };
}

impl_abduction!(f32);
impl_abduction!(f64);

macro_rules! impl_convert {
    ($ft: ty) => {
        impl From<BOpinion<$ft>> for MOpinion1d<$ft, 2> {
            fn from(value: BOpinion<$ft>) -> Self {
                MOpinion1d {
                    simplex: value.simplex.0,
                    base_rate: [value.base_rate, 1.0 - value.base_rate],
                }
            }
        }

        impl From<MOpinion1d<$ft, 2>> for BOpinion<$ft> {
            fn from(value: MOpinion1d<$ft, 2>) -> Self {
                BOpinion::<$ft>::new_unchecked(
                    value.b()[0],
                    value.b()[1],
                    *value.u(),
                    value.base_rate[0],
                )
            }
        }

        impl From<&MOpinion1d<$ft, 2>> for BOpinion<$ft> {
            fn from(value: &MOpinion1d<$ft, 2>) -> Self {
                BOpinion::<$ft>::new_unchecked(
                    value.b()[0],
                    value.b()[1],
                    *value.u(),
                    value.base_rate[0],
                )
            }
        }
    };
}

impl_convert!(f32);
impl_convert!(f64);

/// An error indicating that one or more invalid values are used.
#[derive(thiserror::Error, Debug)]
#[error("At least one parameter is invalid because {0}.")]
pub struct InvalidValueError(String);

trait ApproxRange<T = Self>: Sized {
    fn is_in_range(self, from: T, to: T) -> bool;
    fn out_of_range(self, from: T, to: T) -> bool {
        !self.is_in_range(from, to)
    }
}

macro_rules! impl_approx {
    ($ft: ty) => {
        impl ApproxRange for $ft {
            fn is_in_range(self, from: Self, to: Self) -> bool {
                (self >= from && self <= to) || ulps_eq!(self, from) || ulps_eq!(self, to)
            }
        }
        impl ApproxRange for &$ft {
            fn is_in_range(self, from: Self, to: Self) -> bool {
                (self >= from && self <= to) || ulps_eq!(self, from) || ulps_eq!(self, to)
            }
        }
    };
}

impl_approx!(f32);
impl_approx!(f64);

#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq, assert_ulps_eq};

    use crate::{
        Abduction, BOpinion, Deduction, Fusion, FusionAssign, FusionOp, MOpinion1d, MSimplex,
    };

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
    fn test_cfuse_al_mul() {
        let w1 = MOpinion1d::<f32, 2>::new([0.0, 0.3], 0.7, [0.7, 0.3]);
        let w2 = MOpinion1d::<f32, 2>::new([0.7, 0.0], 0.3, [0.3, 0.7]);
        println!("{:?}", w1.cfuse_al(&w2, 0.0).unwrap());
    }

    #[test]
    fn test_cfuse_ep_mul() {
        let w1 =
            MOpinion1d::<f32, 3>::new([0.98, 0.01, 0.0], 0.01, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let w2 =
            MOpinion1d::<f32, 3>::new([0.0, 0.01, 0.90], 0.09, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        println!("{:?}", w1.cfuse_ep(&w2, 0.0).unwrap());
    }

    #[test]
    fn test_cum_fusion_bo() {
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

        assert_relative_eq!(w1.wfuse(&w0, 0.5).unwrap(), w1);
        assert_ulps_eq!(w1.wfuse(&w0, 0.5).unwrap(), w1);
        assert_relative_eq!(w2.wfuse(&w0, 0.5).unwrap(), w2);
        assert_ulps_eq!(w2.wfuse(&w0, 0.5).unwrap(), w2);
        assert_relative_eq!(w3.wfuse(&w0, 0.5).unwrap(), w3);
        assert_ulps_eq!(w3.wfuse(&w0, 0.5).unwrap(), w3);

        assert_relative_eq!(w2.wfuse(&w2, 0.5).unwrap(), w2);
        assert_ulps_eq!(w2.wfuse(&w2, 0.5).unwrap(), w2);
        assert_relative_eq!(w3.wfuse(&w3, 0.5).unwrap(), w3);
        assert_ulps_eq!(w3.wfuse(&w3, 0.5).unwrap(), w3);
    }

    #[test]
    fn test_fusion_bop() {
        let w1 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
        let w2 = BOpinion::<f32>::new(0.0, 0.90, 0.10, 0.5);
        assert!(w1.cfuse(&w2).is_ok());
        assert!(w1.afuse(&w2, 0.5).is_ok());
        assert!(w1.wfuse(&w2, 0.5).is_ok());
    }

    #[test]
    fn test_fusion_mop() {
        let w1 = MOpinion1d::<f32, 2>::new([0.5, 0.0], 0.5, [0.25, 0.75]);
        let a = [0.5, 0.5];
        let s = MSimplex::<f32, 2>::new([0.0, 0.9], 0.1);
        let w2 = MOpinion1d::<f32, 2>::from_simplex_unchecked(s.clone(), a.clone());
        assert_eq!(
            w1.cfuse_al(&w2, 0.5).unwrap(),
            w1.cfuse_al((&s, &a), 0.5).unwrap()
        );
        assert_eq!(
            w1.cfuse_ep(&w2, 0.5).unwrap(),
            w1.cfuse_ep((&s, &a), 0.5).unwrap()
        );
        assert_eq!(
            w1.afuse(&w2, 0.5).unwrap(),
            w1.afuse((&s, &a), 0.5).unwrap()
        );
        assert_eq!(
            w1.wfuse(&w2, 0.5).unwrap(),
            w1.wfuse((&s, &a), 0.5).unwrap()
        );
    }

    #[test]
    fn test_deduction_bop() {
        let cond = [
            BOpinion::<f32>::new(0.90, 0.02, 0.08, 0.5),
            BOpinion::<f32>::new(0.40, 0.52, 0.08, 0.5),
        ];
        let w = BOpinion::<f32>::new(0.00, 0.38, 0.62, 0.5);
        println!("{}", w.deduce(&cond, 0.5));

        let cond = [
            BOpinion::<f32>::new(0.72, 0.18, 0.1, 0.5),
            BOpinion::<f32>::new(0.13, 0.57, 0.3, 0.5),
        ];
        let w = BOpinion::<f32>::new(0.7, 0.0, 0.3, 0.33);
        println!("{}", w.deduce(&cond, 0.5));
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
        let x = wa.deduce(&wxa, 0.5);
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
        println!("{:?}", wx.deduce(&wyx, [0.5, 0.5]));
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
        let wx = wa.deduce(&wxa, [0.5, 0.5]);
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
        let w = MOpinion1d::<f32, 2>::new([0.1, 0.1], 0.8, [0.5, 0.5]);
        let conds = [
            MOpinion1d::<f32, 3>::new([0.7, 0.0, 0.0], 0.3, [0.5, 0.2, 0.3]),
            MOpinion1d::<f32, 3>::new([0.0, 0.7, 0.0], 0.3, [0.5, 0.2, 0.3]),
        ];
        let wy = w.deduce(&conds, [0.5, 0.25, 0.25]);
        println!("{:?}", wy);
    }

    #[test]
    fn test_abduction() {
        let conds = [
            MSimplex::<f32, 3>::new_unchecked([0.25, 0.04, 0.00], 0.71),
            MSimplex::<f32, 3>::new_unchecked([0.00, 0.50, 0.50], 0.00),
            MSimplex::<f32, 3>::new_unchecked([0.00, 0.25, 0.75], 0.00),
        ];
        let ax = [0.70, 0.20, 0.10];
        let wy = MSimplex::<f32, 3>::new_unchecked([0.00, 0.43, 0.00], 0.57);
        let (wx, ay) = wy.abduce(&conds, ax, None).unwrap();
        println!("{:?}, {:?}", wx, ay);
    }

    #[test]
    fn test_abduction2() {
        let ax = [0.01, 0.495, 0.495];
        let conds_ox = [
            MSimplex::<f32, 2>::new_unchecked([0.5, 0.0], 0.5),
            MSimplex::<f32, 2>::new_unchecked([0.5, 0.0], 0.5),
            MSimplex::<f32, 2>::new_unchecked([0.01, 0.01], 0.98),
        ];
        let mw_o = MSimplex::<f32, 2>::new_unchecked([0.0, 0.0], 1.0);
        let (mw_x, _) = mw_o.abduce(&conds_ox, ax, None).unwrap();
        println!("{:?}", mw_x);
    }

    #[test]
    fn test_fusion_assign() {
        let mut w = MOpinion1d::<f32, 2>::new([0.5, 0.25], 0.25, [0.5, 0.5]);
        let u = MOpinion1d::<f32, 2>::new([0.125, 0.75], 0.125, [0.75, 0.25]);
        let w2 = w.cfuse_al(&u, 0.5).unwrap();
        w.fusion_assign(&u, &FusionOp::CumulativeA(0.5)).unwrap();
        assert!(w == w2);
        let w2 = w.cfuse_ep(&u, 0.5).unwrap();
        w.fusion_assign(&u, &FusionOp::CumulativeE(0.5)).unwrap();
        assert!(w == w2);
        let w2 = w.afuse(&u, 0.5).unwrap();
        w.fusion_assign(&u, &FusionOp::Averaging(0.5)).unwrap();
        assert!(w == w2);
        let w2 = w.wfuse(&u, 0.5).unwrap();
        w.fusion_assign(&u, &FusionOp::Weighted(0.5)).unwrap();
        assert!(w == w2);
    }
}
