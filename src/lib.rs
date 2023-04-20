//!
//! An implementation of [Subjective Logic](https://en.wikipedia.org/wiki/Subjective_logic).

use std::{array, fmt::Display, ops::Index};

/// A binomial opinion.
#[derive(Debug)]
pub struct BOpinion<T> {
    pub simplex: MSimplex<T, 2>,
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
            simplex: MSimplex::new_unchecked([b, d], u),
            base_rate: a,
        }
    }

    pub fn b(&self) -> &T {
        &self.simplex.belief[0]
    }

    pub fn d(&self) -> &T {
        &self.simplex.belief[1]
    }

    pub fn u(&self) -> &T {
        &self.simplex.uncertainty
    }

    pub fn a(&self) -> &T {
        &self.base_rate
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
                let a = if self.u().approx_eq(&1.0) && rhs.u().approx_eq(&1.0) {
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
                if self.u().approx_eq(&0.0) && rhs.u().approx_eq(&0.0) {
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
                if self.u().approx_eq(&0.0) && rhs.u().approx_eq(&0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = gamma_a * self.b() + gamma_b * rhs.b();
                    d = gamma_a * self.d() + gamma_b * rhs.d();
                    u = 0.0;
                    a = gamma_a * self.base_rate + gamma_b * rhs.base_rate;
                } else if self.u().approx_eq(&1.0) && rhs.u().approx_eq(&1.0) {
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
            pub fn deduce(&self, cond: BSimplexRefs<$ft, 2>, ay: $ft) -> Self {
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
    };
}

impl_bsl!(f32);
impl_bsl!(f64);

/// The simplex of a binomial opinion, from which a base rate is excluded.
#[derive(Debug)]
pub struct BSimplexRef<'a, T>(MSimplexRef<'a, T, 2>);

impl<'a, T> BSimplexRef<'a, T> {
    fn b(&self) -> &T {
        &self.0.belief[0]
    }

    fn d(&self) -> &T {
        &self.0.belief[1]
    }

    fn u(&self) -> &T {
        self.0.uncertainty
    }
}

impl<'a, T> From<&'a BOpinion<T>> for BSimplexRef<'a, T> {
    fn from(value: &'a BOpinion<T>) -> Self {
        Self(value.simplex.borrow())
    }
}

pub struct BSimplexRefs<'a, T, const M: usize>([BSimplexRef<'a, T>; M]);

impl<'a, I, T, const M: usize> Index<I> for BSimplexRefs<'a, T, M>
where
    [BSimplexRef<'a, T>]: Index<I>,
{
    type Output = <[BSimplexRef<'a, T>; M] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<'a, T, const M: usize> From<&'a [BOpinion<T>; M]> for BSimplexRefs<'a, T, M> {
    fn from(value: &'a [BOpinion<T>; M]) -> Self {
        BSimplexRefs(array::from_fn(|i| (&value[i]).into()))
    }
}

#[derive(Debug, Clone)]
pub struct MSimplexBase<T, U> {
    belief: T,
    uncertainty: U,
}

impl<T: Display, U: Display> Display for MSimplexBase<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.belief, self.uncertainty)
    }
}

pub type MSimplexBaseRef<'a, T, U> = MSimplexBase<&'a T, &'a U>;

impl<T, U> MSimplexBase<T, U> {
    pub fn new_unchecked(b: T, u: U) -> Self {
        Self {
            belief: b,
            uncertainty: u,
        }
    }

    pub fn borrow(&self) -> MSimplexBaseRef<'_, T, U> {
        MSimplexBaseRef {
            belief: &self.belief,
            uncertainty: &self.uncertainty,
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

/// The reference of a simplex of a multinomial opinion, from which a base rate is excluded.
pub type MSimplex<T, const N: usize> = MSimplexBase<[T; N], T>;

/// The reference type of a simplex of a multinomial opinion, from which a base rate is excluded.
pub type MSimplexRef<'a, T, const N: usize> = MSimplexBaseRef<'a, [T; N], T>;

#[derive(Debug)]
pub struct MSimplexRefs<'a, T, const M: usize, const N: usize>([MSimplexRef<'a, T, M>; N]);

impl<'a, I, T, const M: usize, const N: usize> Index<I> for MSimplexRefs<'a, T, M, N>
where
    [MSimplexRef<'a, T, M>]: Index<I>,
{
    type Output = <[MSimplexRef<'a, T, M>; N] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<'a, T, const N: usize, const M: usize> From<&'a [MSimplex<T, N>; M]>
    for MSimplexRefs<'a, T, N, M>
{
    fn from(value: &'a [MSimplex<T, N>; M]) -> Self {
        MSimplexRefs(array::from_fn(|i| (&value[i]).borrow()))
    }
}

/// The generlized structure of a multinomial opinion.
#[derive(Debug, Clone)]
pub struct MOpinion<T, U> {
    pub simplex: MSimplexBase<T, U>,
    pub base_rate: T,
}

impl<T: Display, U: Display> Display for MOpinion<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.simplex, self.base_rate)
    }
}

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

/// The reference type of a multinomial opinion, from which a base rate is excluded.
pub type MOpinionRef<'a, T, U> = MOpinion<&'a T, &'a U>;

impl<'a, 'b: 'a, T, U> From<(MSimplexBaseRef<'a, T, U>, &'b T)> for MOpinionRef<'a, T, U> {
    fn from(value: (MSimplexBaseRef<'a, T, U>, &'b T)) -> Self {
        let simplex = value.0;
        let base_rate = value.1;
        MOpinion {
            simplex: MSimplexBase {
                belief: &simplex.belief,
                uncertainty: &simplex.uncertainty,
            },
            base_rate,
        }
    }
}

impl<'a, T, U> From<&'a MOpinion<T, U>> for MOpinionRef<'a, T, U> {
    fn from(value: &'a MOpinion<T, U>) -> Self {
        MOpinion {
            simplex: MSimplexBase {
                belief: &value.simplex.belief,
                uncertainty: &value.simplex.uncertainty,
            },
            base_rate: &value.base_rate,
        }
    }
}

impl<'a, T, U> From<MOpinionRef<'a, T, U>> for MSimplexBaseRef<'a, T, U> {
    fn from(value: MOpinionRef<'a, T, U>) -> Self {
        MSimplexBase {
            belief: value.simplex.belief,
            uncertainty: value.simplex.uncertainty,
        }
    }
}

/// A multinomial opinion with 1-dimensional vectors.
pub type MOpinion1d<T, const N: usize> = MOpinion<[T; N], T>;

/// The reference type of a multinomial opinion with 1-dimensional vectors.
pub type MOpinion1dRef<'a, T, const N: usize> = MOpinionRef<'a, [T; N], T>;

impl<'a, T, const N: usize> From<&'a [BOpinion<T>; N]> for MSimplexRefs<'a, T, 2, N> {
    fn from(value: &'a [BOpinion<T>; N]) -> Self {
        MSimplexRefs(array::from_fn(|i| value[i].simplex.borrow()))
    }
}

impl<'a, T, const N: usize, const M: usize> From<&'a [MOpinion1d<T, N>; M]>
    for MSimplexRefs<'a, T, N, M>
{
    fn from(value: &'a [MOpinion1d<T, N>; M]) -> Self {
        MSimplexRefs(array::from_fn(|i| (&value[i].simplex).borrow()))
    }
}

#[derive(Debug)]
pub struct MSimplexes<T, const M: usize, const N: usize>([MSimplex<T, M>; N]);

impl<'a, T, const M: usize, const N: usize> From<&'a MSimplexes<T, M, N>>
    for MSimplexRefs<'a, T, M, N>
{
    fn from(value: &'a MSimplexes<T, M, N>) -> Self {
        MSimplexRefs(array::from_fn(|i| (&value.0[i]).borrow()))
    }
}

impl<'a, T, const M: usize, const N: usize> From<([[T; M]; N], [T; N])> for MSimplexes<T, M, N> {
    fn from(value: ([[T; M]; N], [T; N])) -> Self {
        let (bs, us) = value;
        let mut bs = Vec::from(bs);
        let mut us = Vec::from(us);
        MSimplexes(array::from_fn(|i| {
            let b = bs.swap_remove(N - 1 - i);
            let u = us.swap_remove(N - 1 - i);
            MSimplex::new_unchecked(b, u)
        }))
    }
}

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
                if !(b.iter().sum::<$ft>() + u).approx_eq(1.0) {
                    return Err(InvalidValueError(
                        "sum(b) + u = 1 is not satisfied".to_string(),
                    ));
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

        impl<'a, const N: usize> MSimplexRef<'a, $ft, N> {
            pub fn projection(&self, a: &[$ft; N]) -> [$ft; N] {
                array::from_fn(|i| self.belief[i] + self.uncertainty * a[i])
            }
        }

        impl<'a, const M: usize, const N: usize> MSimplexRefs<'a, $ft, N, M> {
            pub fn marginal_base_rate(&self, ax: &[$ft; M]) -> Option<[$ft; N]> {
                if (0..M)
                    .map(|i| self[i].uncertainty)
                    .sum::<$ft>()
                    .approx_eq(M as $ft)
                {
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

pub trait Fusion<Rhs, U> {
    type Output;

    /// Computes the aleatory cummulative fusion of `self` and `rhs`.
    fn cfuse_al(&self, rhs: Rhs, gamma_a: U) -> Self::Output;

    /// Computes the epistemic cummulative fusion of `self` and `rhs`.
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

macro_rules! impl_msl {
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
                if !a.iter().sum::<$ft>().approx_eq(1.0) {
                    return Err(InvalidValueError("sum(a) = 1 is not satisfied".to_string()));
                }
                for i in 0..N {
                    if !a[i].is_in_range(0.0, 1.0) {
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

        impl<'a, const N: usize> Fusion<MSimplexRef<'a, $ft, N>, $ft> for MOpinion1d<$ft, N> {
            type Output = Result<MSimplex<$ft, N>, InvalidValueError>;
            fn cfuse_al(&self, rhs: MSimplexRef<'a, $ft, N>, gamma_a: $ft) -> Self::Output {
                let b;
                let u;
                if self.u().approx_eq(&0.0) && rhs.u().approx_eq(&0.0) {
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

            fn cfuse_ep(&self, rhs: MSimplexRef<'a, $ft, N>, gamma_a: $ft) -> Self::Output {
                let s = self.cfuse_al(rhs, gamma_a)?;
                let w = MOpinion1dRef::from((s.borrow(), &self.base_rate));
                let u_max = w.max_uncertainty();
                let b_max = array::from_fn(|i| w.projection(i) - w.base_rate[i] * u_max);
                MSimplex::<$ft, N>::try_new(b_max, u_max)
            }

            fn afuse(&self, rhs: MSimplexRef<'a, $ft, N>, gamma_a: $ft) -> Self::Output {
                let b;
                let u;
                if self.u().approx_eq(&0.0) && rhs.u().approx_eq(&0.0) {
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

            fn wfuse(&self, rhs: MSimplexRef<'a, $ft, N>, gamma_a: $ft) -> Self::Output {
                let b;
                let u;
                if self.u().approx_eq(&0.0) && rhs.u().approx_eq(&0.0) {
                    let gamma_b = 1.0 - gamma_a;
                    b = array::from_fn(|i| gamma_a * self.b()[i] + gamma_b * rhs.b()[i]);
                    u = 0.0;
                } else if self.u().approx_eq(&1.0) && rhs.u().approx_eq(&1.0) {
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
                let sr = MSimplexRef::from(rhs.clone());
                let s = self.cfuse_al(sr, gamma_a)?;
                let a = if self.u().approx_eq(&0.0) && rhs.u().approx_eq(&0.0) {
                    array::from_fn(|i| {
                        gamma_a * self.base_rate[i] + (1.0 - gamma_a) * rhs.base_rate[i]
                    })
                } else if self.u().approx_eq(&1.0) && rhs.u().approx_eq(&1.0) {
                    array::from_fn(|i| (self.base_rate[i] + rhs.base_rate[i]) / 2.0)
                } else {
                    let rhs_u = *rhs.u();
                    array::from_fn(|i| {
                        (self.base_rate[i] * rhs_u + rhs.base_rate[i] * self.u()
                            - (self.base_rate[i] + rhs.base_rate[i]) * self.u() * rhs_u)
                            / (self.u() + rhs_u - self.u() * rhs_u * 2.0)
                    })
                };
                dbg!(a);
                Self::try_from_simplex(s, a)
            }

            fn cfuse_ep(&self, rhs: A, gamma_a: $ft) -> Self::Output {
                let w = self.cfuse_al(rhs, gamma_a)?;
                w.op_u_max()
            }

            fn afuse(&self, rhs: A, gamma_a: $ft) -> Self::Output {
                let rhs = rhs.into();
                let sr = MSimplexRef::from(rhs.clone());
                let s = self.afuse(sr, gamma_a)?;
                let a = if self.u().approx_eq(&0.0) && rhs.u().approx_eq(&0.0) {
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
                let sr = MSimplexRef::from(rhs.clone());
                let s = self.wfuse(sr, gamma_a)?;
                let a = if self.u().approx_eq(&0.0) && rhs.u().approx_eq(&0.0) {
                    array::from_fn(|i| {
                        gamma_a * self.base_rate[i] + (1.0 - gamma_a) * rhs.base_rate[i]
                    })
                } else if self.u().approx_eq(&1.0) && rhs.u().approx_eq(&1.0) {
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

        /// The probability projection of `self`.
        impl<T: Index<usize, Output = $ft>> MOpinion<T, $ft> {
            pub fn projection(&self, idx: usize) -> $ft {
                self.b()[idx] + self.base_rate[idx] * self.u()
            }
        }

        /// The probability projection of `self`.
        impl<'a, T: Index<usize, Output = $ft>> MOpinion<&'a T, &'a $ft> {
            pub fn projection(&self, idx: usize) -> $ft {
                self.b()[idx] + self.base_rate[idx] * *self.u()
            }
        }

        impl<'a, A, const M: usize, const N: usize> Deduction<A, [$ft; N]> for &MOpinion1d<$ft, M>
        where
            A: Into<MSimplexRefs<'a, $ft, N, M>> + 'a,
        {
            type Output = MOpinion1d<$ft, N>;

            fn deduce(self, conds: A, ay: [$ft; N]) -> Self::Output {
                MOpinionRef::from(self).deduce(conds.into(), ay)
            }
        }

        impl<'a, const M: usize, const N: usize> Deduction<MSimplexes<$ft, N, M>, [$ft; N]>
            for MOpinion1dRef<'a, $ft, M>
        {
            type Output = MOpinion1d<$ft, N>;
            fn deduce(self, conds: MSimplexes<$ft, N, M>, ay: [$ft; N]) -> Self::Output {
                self.deduce(MSimplexRefs::from(&conds), ay)
            }
        }

        impl<'a, A, const M: usize, const N: usize> Deduction<A, [$ft; N]>
            for MOpinion1dRef<'a, $ft, M>
        where
            A: Into<MSimplexRefs<'a, $ft, N, M>> + 'a,
        {
            type Output = MOpinion1d<$ft, N>;

            fn deduce(self, conds: A, ay: [$ft; N]) -> Self::Output {
                assert!(M > 0 && N > 1, "N > 0 and M > 1 must hold.");
                let conds = conds.into();
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

        impl<'a, A, const N: usize, const M: usize> Abduction<A, [$ft; M], [$ft; N]>
            for &'a MSimplex<$ft, N>
        where
            A: Into<MSimplexRefs<'a, $ft, N, M>> + 'a,
        {
            type Output = MOpinion1d<$ft, M>;
            fn abduce(
                self,
                conds_yx: A,
                ax: [$ft; M],
                ay: Option<[$ft; N]>,
            ) -> Option<(Self::Output, [$ft; N])> {
                self.borrow().abduce(conds_yx, ax, ay)
            }
        }

        impl<'a, A, const N: usize, const M: usize> Abduction<A, [$ft; M], [$ft; N]>
            for MSimplexRef<'a, $ft, N>
        where
            A: Into<MSimplexRefs<'a, $ft, N, M>> + 'a,
        {
            type Output = MOpinion1d<$ft, M>;

            fn abduce(
                self,
                conds_yx: A,
                ax: [$ft; M],
                ay: Option<[$ft; N]>,
            ) -> Option<(Self::Output, [$ft; N])> {
                fn inverse<'a, const N: usize, const M: usize>(
                    conds_yx: MSimplexRefs<'a, $ft, N, M>,
                    ax: &[$ft; M],
                    ay: &[$ft; N],
                ) -> MSimplexes<$ft, M, N> {
                    let p_yx: [[$ft; N]; M] = array::from_fn(|i| conds_yx[i].projection(ay));
                    let p_xy: [[$ft; M]; N] = array::from_fn(|j| {
                        array::from_fn(|i| {
                            ax[i] * p_yx[i][j] / (0..M).map(|k| ax[k] * p_yx[k][j]).sum::<$ft>()
                        })
                    });
                    let u_yx_sum = conds_yx.0.iter().map(|cond| cond.uncertainty).sum::<$ft>();
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
                    (bs, u_xy_inv).into()
                }

                let conds_yx = conds_yx.into();
                let ay = conds_yx.marginal_base_rate(&ax).or(ay)?;
                let inv_conds = inverse(conds_yx, &ax, &ay);
                let w = MOpinion1dRef::from((self, &ay));
                Some((w.deduce(inv_conds, ax), ay))
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
                MOpinion1d {
                    simplex: value.simplex,
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

        impl<'a, const N: usize> Deduction<MSimplexRefs<'a, $ft, 2, N>, $ft>
            for &MOpinion1d<$ft, N>
        {
            type Output = BOpinion<$ft>;

            fn deduce(self, cond: MSimplexRefs<'a, $ft, 2, N>, ay: $ft) -> Self::Output {
                self.deduce(cond, [ay, 1.0 - ay]).into()
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
        self.b().approx_eq(to.b())
            && self.d().approx_eq(to.d())
            && self.u().approx_eq(to.u())
            && self.base_rate.approx_eq(&to.base_rate)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Abduction, BOpinion, Deduction, EpsilonEq, Fusion, MOpinion1d, MSimplex};

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
}
