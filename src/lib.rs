//!
//! 本クレートは Subjective Logicの実装です。
//!
use std::{array, fmt::Display, ops::Index};

#[cfg_attr(doc, katexit::katexit)]
/// Binomial opinion
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

pub trait BOperator<T> {
    /// deduction
    fn ded(&self, ytx: &Self, yfx: &Self, ay: T) -> Self;
}

macro_rules! impl_bsl {
    ($ft: ty) => {
        impl BOpinion<$ft> {
            pub fn new_unchecked(b: $ft, d: $ft, u: $ft, a: $ft) -> Self {
                Self {
                    belief: b,
                    disbelief: d,
                    uncertainty: u,
                    base_rate: a,
                }
            }

            pub fn new(b: $ft, d: $ft, u: $ft, a: $ft) -> Self {
                Self::try_new(b, d, u, a).unwrap()
            }

            pub fn try_new(b: $ft, d: $ft, u: $ft, a: $ft) -> Result<Self, InvalidRangeError> {
                if !(b + d + u).approx_eq(1.0) {
                    return Err(InvalidRangeError(
                        "b + d + u = 1 is not satisfied".to_string(),
                    ));
                }
                if !b.is_in_range(0.0, 1.0) {
                    return Err(InvalidRangeError("b ∈ [0,1] is not satisfied".to_string()));
                }
                if !d.is_in_range(0.0, 1.0) {
                    return Err(InvalidRangeError("d ∈ [0,1] is not satisfied".to_string()));
                }
                if !u.is_in_range(0.0, 1.0) {
                    return Err(InvalidRangeError("u ∈ [0,1] is not satisfied".to_string()));
                }
                if !a.is_in_range(0.0, 1.0) {
                    return Err(InvalidRangeError("a ∈ [0,1] is not satisfied".to_string()));
                }
                Ok(Self {
                    belief: b,
                    disbelief: d,
                    uncertainty: u,
                    base_rate: a,
                })
            }

            pub fn projection(&self) -> $ft {
                self.belief + self.base_rate * self.uncertainty
            }

            pub fn op_projection(&self) -> $ft {
                self.belief + self.base_rate * (1.0 - self.uncertainty)
            }
        }

        impl Operator for BOpinion<$ft> {
            fn mul(&self, rhs: &Self) -> Self {
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

            fn comul(&self, rhs: &Self) -> Self {
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

            fn cfus(&self, rhs: &Self) -> Option<Self> {
                let uu = self.uncertainty * rhs.uncertainty;
                let kappa = self.uncertainty + rhs.uncertainty - uu;
                let b = (self.belief * rhs.uncertainty + rhs.belief * self.uncertainty) / kappa;
                let d =
                    (self.disbelief * rhs.uncertainty + rhs.disbelief * self.uncertainty) / kappa;
                let u = (self.uncertainty * rhs.uncertainty) / kappa;
                let a = (self.base_rate * rhs.uncertainty + rhs.base_rate * self.uncertainty
                    - (self.base_rate + rhs.base_rate) * uu)
                    / (kappa - uu);
                Self::try_new(b, d, u, a).ok()
            }
        }

        impl BOperator<$ft> for BOpinion<$ft> {
            fn ded(&self, ytx: &Self, yfx: &Self, ay: $ft) -> Self {
                let rvax = (1.0 - self.base_rate);
                let bi = self.belief * ytx.belief
                    + self.disbelief * yfx.belief
                    + self.uncertainty * (ytx.belief * self.base_rate + yfx.belief * rvax);
                let di = self.belief * ytx.disbelief
                    + self.disbelief * yfx.disbelief
                    + self.uncertainty * (ytx.disbelief * self.base_rate + yfx.disbelief * rvax);
                let ui = self.belief * ytx.uncertainty
                    + self.disbelief * yfx.uncertainty
                    + self.uncertainty
                        * (ytx.uncertainty * self.base_rate + yfx.uncertainty * rvax);
                let k = match (ytx.belief > yfx.belief, ytx.disbelief > yfx.disbelief) {
                    (true, true) | (false, false) => 0.0,
                    (bp, dp) => {
                        let evyx = ytx.belief * self.base_rate
                            + yfx.belief * rvax
                            + ay * (ytx.uncertainty * self.base_rate + yfx.uncertainty * rvax);
                        let evx = self.projection();
                        match (bp, dp) {
                            (true, false) => {
                                let r = (1.0 - ay) * yfx.belief + ay * (1.0 - ytx.disbelief);
                                match (evyx > r, evx > self.base_rate) {
                                    (false, false) => {
                                        self.base_rate * self.uncertainty * (bi - yfx.belief)
                                            / (evx * ay)
                                    }
                                    (false, true) => {
                                        self.base_rate
                                            * self.uncertainty
                                            * (di - ytx.disbelief)
                                            * (ytx.belief - yfx.belief)
                                            / ((1.0 - evx) * ay * (yfx.disbelief - ytx.disbelief))
                                    }
                                    (true, false) => {
                                        rvax * self.uncertainty
                                            * (bi - yfx.belief)
                                            * (yfx.disbelief - ytx.disbelief)
                                            / (evx * (1.0 - ay) * (ytx.belief - yfx.belief))
                                    }
                                    (true, true) => {
                                        rvax * self.uncertainty * (di - ytx.disbelief)
                                            / ((1.0 - evx) * (1.0 - ay))
                                    }
                                }
                            }
                            (false, true) => {
                                let r = ytx.belief + ay * (1.0 - ytx.belief - yfx.disbelief);
                                match (evyx > r, evx > self.base_rate) {
                                    (false, false) => {
                                        rvax * self.uncertainty
                                            * (di - yfx.disbelief)
                                            * (yfx.belief - ytx.belief)
                                            / (evx * ay * (ytx.disbelief - yfx.disbelief))
                                    }
                                    (false, true) => {
                                        rvax * self.uncertainty * (bi - ytx.belief)
                                            / ((1.0 - evx) * ay)
                                    }
                                    (true, false) => {
                                        self.base_rate * self.uncertainty * (di - yfx.disbelief)
                                            / (evx * (1.0 - ay))
                                    }
                                    (true, true) => {
                                        self.base_rate
                                            * self.uncertainty
                                            * (bi - ytx.belief)
                                            * (ytx.disbelief - yfx.disbelief)
                                            / ((1.0 - evx) * (1.0 - ay) * (yfx.belief - ytx.belief))
                                    }
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                };
                let b = bi - ay * k;
                let d = di - (1.0 - ay) * k;
                let u = ui + k;
                let a = ay;
                Self::new(b, d, u, a)
            }
        }

        impl Transitivity<$ft> for BOpinion<$ft> {
            fn unc(&self, b: $ft) -> Self {
                assert!(b.is_in_range(0.0, 1.0), "b ∈ [0,1] is not satisfied.");
                Self::new(
                    b * self.belief,
                    b * self.disbelief,
                    1.0 - b + b * self.uncertainty,
                    self.base_rate,
                )
            }
            fn opp(&self, b: $ft, d: $ft) -> Self {
                let u = 1.0 - b - d;
                assert!(u.is_in_range(0.0, 1.0), "b + d ∈ [0,1] is not satisfied.");
                Self::new(
                    b * self.belief + d * self.disbelief,
                    b * self.disbelief + d * self.belief,
                    u + (b + d) * self.uncertainty,
                    self.base_rate,
                )
            }
            fn bsr(&self, ev: $ft) -> Self {
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

pub trait Operator {
    /// multiplication
    fn mul(&self, rhs: &Self) -> Self;
    /// comultiplication
    fn comul(&self, rhs: &Self) -> Self;
    /// cumulative fusion
    fn cfus(&self, rhs: &Self) -> Option<Self>
    where
        Self: Sized;
}

pub trait Transitivity<T> {
    /// uncertainty favouring
    fn unc(&self, b: T) -> Self;
    /// opposite belief favouring
    fn opp(&self, b: T, d: T) -> Self;
    /// base rate sensitive
    fn bsr(&self, ev: T) -> Self;
}

#[derive(thiserror::Error, Debug)]
#[error("At least one parameter is invalid because {0}.")]
pub struct InvalidRangeError(String);

trait EpsilonComp {
    fn is_in_range(self, from: Self, to: Self) -> bool;
    fn approx_eq(self, to: Self) -> bool;
}

macro_rules! impl_epsilon_comp {
    ($ft: ty) => {
        impl EpsilonComp for $ft {
            fn is_in_range(self, from: Self, to: Self) -> bool {
                self >= from - <$ft>::EPSILON && self <= to + <$ft>::EPSILON
            }

            fn approx_eq(self, to: Self) -> bool {
                self >= to - <$ft>::EPSILON && self <= to + <$ft>::EPSILON
            }
        }
    };
}

impl_epsilon_comp!(f32);
impl_epsilon_comp!(f64);

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
    pub fn new_unchecked(b: T, u: U, a: T) -> Self {
        Self {
            belief: b,
            uncertainty: u,
            base_rate: a,
        }
    }
}

pub trait MOperator<Rhs, U> {
    type Output;

    /// deduction
    fn ded(&self, wyx: &Rhs, ay: U) -> Self::Output;
}

pub type MOpinion1d<T, const N: usize> = MOpinion<[T; N], T>;

macro_rules! impl_msl {
    ($ft: ty) => {
        impl<const N: usize> MOpinion1d<$ft, N> {
            pub fn try_new(b: [$ft; N], u: $ft, a: [$ft; N]) -> Result<Self, InvalidRangeError> {
                if !(b.iter().sum::<$ft>() + u).approx_eq(1.0) {
                    return Err(InvalidRangeError(
                        "sum(b) + u = 1 is not satisfied".to_string(),
                    ));
                }
                if !a.iter().sum::<$ft>().approx_eq(1.0) {
                    return Err(InvalidRangeError("sum(a) = 1 is not satisfied".to_string()));
                }

                if !u.is_in_range(0.0, 1.0) {
                    return Err(InvalidRangeError(format!("u ∈ [0,1] is not satisfied")));
                }

                for i in 0..N {
                    if !b[i].is_in_range(0.0, 1.0) {
                        return Err(InvalidRangeError(format!(
                            "b[{i}] ∈ [0,1] is not satisfied"
                        )));
                    }
                    if !a[i].is_in_range(0.0, 1.0) {
                        return Err(InvalidRangeError(format!(
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

            pub fn new(b: [$ft; N], u: $ft, a: [$ft; N]) -> Self {
                Self::try_new(b, u, a).unwrap()
            }
        }

        impl<T: Index<usize, Output = $ft>> MOpinion<T, $ft> {
            pub fn projection(&self, idx: usize) -> $ft {
                self.belief[idx] + self.base_rate[idx] * self.uncertainty
            }
        }

        impl<const N: usize, const M: usize> MOperator<[MOpinion1d<$ft, M>; N], [$ft; M]>
            for MOpinion1d<$ft, N>
        {
            type Output = MOpinion1d<$ft, M>;

            fn ded(&self, wyx: &[MOpinion1d<$ft, M>; N], ay: [$ft; M]) -> Self::Output {
                assert!(N > 0 && M > 1, "N > 0 and M > 1 must hold.");
                let eyhx: [$ft; M] = array::from_fn(|t| {
                    (0..N)
                        .map(|i| self.base_rate[i] * wyx[i].projection(t))
                        .sum()
                });

                let intu = (0..M)
                    .map(|t| {
                        let e = eyhx[t];
                        let (r, s) = {
                            let mut s = 0;
                            let mut r = 0;
                            let mut v = 1.0;
                            for r1 in 0..N {
                                for s1 in 0..N {
                                    let w = 1.0 - wyx[r1].belief[t] - wyx[r1].uncertainty
                                        + wyx[s1].belief[t];
                                    if v > w {
                                        v = w;
                                        r = r1;
                                        s = s1;
                                    }
                                }
                            }
                            (r, s)
                        };
                        let eyhxx = (1.0 - ay[t]) * wyx[s].belief[t]
                            + ay[t] * (wyx[r].belief[t] + wyx[r].uncertainty);
                        if e <= eyhxx {
                            (e - wyx[s].belief[t]) / ay[t]
                        } else {
                            (wyx[r].belief[t] + wyx[r].uncertainty - e) / (1.0 - ay[t])
                        }
                    })
                    .reduce(<$ft>::max)
                    .unwrap();

                let apex = (0..M)
                    .map(|t| {
                        if eyhx[t] < ay[t] * intu {
                            eyhx[t] / ay[t]
                        } else {
                            intu
                        }
                    })
                    .reduce(<$ft>::min)
                    .unwrap();

                let u = apex
                    - (0..N)
                        .map(|i| (apex - wyx[i].uncertainty) * self.belief[i])
                        .sum::<$ft>();
                let b: [$ft; M] = array::from_fn(|j| {
                    (0..N)
                        .map(|i| self.projection(i) * wyx[i].projection(j))
                        .sum::<$ft>()
                        - ay[j] * u
                });
                MOpinion1d::<$ft, M>::new(b, u, ay)
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

        impl<const N: usize> MOperator<[BOpinion<$ft>; N], $ft> for MOpinion1d<$ft, N> {
            type Output = BOpinion<$ft>;

            fn ded(&self, wyx: &[BOpinion<$ft>; N], ay: $ft) -> Self::Output {
                let wyx: [MOpinion1d<$ft, 2>; N] = array::from_fn(|i| (&wyx[i]).into());
                self.ded(&wyx, [ay, 1.0 - ay]).into()
            }
        }
    };
}

impl_sl_conv!(f32);
impl_sl_conv!(f64);

#[cfg(test)]
mod tests {
    use crate::{BOperator, BOpinion, MOperator, MOpinion1d};

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
    fn test_bsl_deduction() {
        let wytx = BOpinion::<f32>::new(0.90, 0.02, 0.08, 0.5);
        let wyfx = BOpinion::<f32>::new(0.40, 0.52, 0.08, 0.5);
        let wx = BOpinion::<f32>::new(0.00, 0.38, 0.62, 0.5);
        println!("{}", wx.ded(&wytx, &wyfx, 0.5));

        let wytx = BOpinion::<f32>::new(0.72, 0.18, 0.1, 0.5);
        let wyfx = BOpinion::<f32>::new(0.13, 0.57, 0.3, 0.5);
        let wx = BOpinion::<f32>::new(0.7, 0.0, 0.3, 0.33);
        println!("{}", wx.ded(&wytx, &wyfx, 0.5));
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
        let x = wa.ded(&wxa, 0.5);
        println!("{}", x.projection());
    }

    #[test]
    fn test_deduction1() {
        let wx = BOpinion::<f32>::new(0.7, 0.0, 0.3, 1.0 / 3.0);
        let wytx = BOpinion::<f32>::new(0.72, 0.18, 0.1, 0.5);
        let wyfx = BOpinion::<f32>::new(0.13, 0.57, 0.3, 0.5);
        println!("{}", wx.ded(&wytx, &wyfx, 0.5));

        let wx = MOpinion1d::<f32, 2>::new([0.7, 0.0], 0.3, [1.0 / 3.0, 2.0 / 3.0]);
        let wyx = [
            MOpinion1d::<f32, 2>::new([0.72, 0.18], 0.1, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.13, 0.57], 0.3, [0.5, 0.5]),
        ];
        println!("{:?}", wx.ded(&wyx, [0.5, 0.5]));
    }

    #[test]
    fn test_deduction2() {
        let wa = MOpinion1d::<f32, 3>::new([0.7, 0.1, 0.0], 0.2, [0.3, 0.3, 0.4]);
        let wxa = [
            BOpinion::<f32>::new(0.7, 0.0, 0.3, 0.5),
            BOpinion::<f32>::new(0.0, 0.7, 0.3, 0.5),
            BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5),
        ];
        let wx = wa.ded(&wxa, 0.5);
        println!("{}|{}", wx, wx.projection());

        let wxa = [
            MOpinion1d::<f32, 2>::new([0.7, 0.0], 0.3, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.0, 0.7], 0.3, [0.5, 0.5]),
            MOpinion1d::<f32, 2>::new([0.0, 0.0], 1.0, [0.5, 0.5]),
        ];
        let wx = wa.ded(&wxa, [0.5, 0.5]);
        println!("{:?}|{}", wx, wx.projection(0));

        let wa = BOpinion::<f32>::new(0.7, 0.1, 0.2, 0.5);
        let wxta = BOpinion::<f32>::new(0.7, 0.0, 0.3, 0.5);
        let wxfa = BOpinion::<f32>::new(0.0, 0.7, 0.3, 0.5);
        println!("{}", wa.ded(&wxta, &wxfa, 0.5));
    }

    #[test]
    fn test_deduction3() {
        let wa = MOpinion1d::<f32, 2>::new([0.7, 0.1], 0.2, [0.5, 0.5]);
        let wxa = [
            MOpinion1d::<f32, 3>::new([0.7, 0.0, 0.0], 0.3, [0.3, 0.3, 0.4]),
            MOpinion1d::<f32, 3>::new([0.0, 0.7, 0.0], 0.3, [0.3, 0.3, 0.4]),
        ];
        let wx = wa.ded(&wxa, [0.3, 0.3, 0.4]);
        println!("{:?}|{}", wx, wx.projection(0));
    }

    #[test]
    fn test_msl_deduction() {
        let a = [0.7, 0.2, 0.1];
        let wx = MOpinion1d::<f32, 3>::new([0.0, 0.5, 0.5], 0.0, a.clone());
        let wyx = [
            MOpinion1d::<f32, 3>::new([1.0, 0.00, 0.00], 0.00, a.clone()),
            MOpinion1d::<f32, 3>::new([0.0, 0.17, 0.00], 0.83, a.clone()),
            MOpinion1d::<f32, 3>::new([0.0, 0.14, 0.14], 0.72, a.clone()),
        ];
        println!("{:?}", wx.ded(&wyx, a))
    }
}
