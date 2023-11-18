use approx::{ulps_eq, ulps_ne, AbsDiffEq, RelativeEq, UlpsEq};
use std::fmt::Display;

use crate::approx_ext::ApproxRange;
use crate::errors::InvalidValueError;
use crate::mul::Simplex;

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
    pub fn new_unchecked(b: T, d: T, u: T, a: T) -> Self {
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
pub struct BSimplex<T>(pub Simplex<T, 2>);

impl<T> BSimplex<T> {
    fn new_unchecked(b: T, d: T, u: T) -> Self {
        Self(Simplex::new_unchecked([b, d], u))
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

impl<'a, T> From<&'a BOpinion<T>> for &'a Simplex<T, 2> {
    fn from(value: &'a BOpinion<T>) -> Self {
        &value.simplex.0
    }
}
