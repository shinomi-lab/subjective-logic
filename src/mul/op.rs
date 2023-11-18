use approx::ulps_eq;
use std::array;

use super::{Opinion1d, Opinion1dRef, Simplex, MBR};
use crate::errors::InvalidValueError;

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

macro_rules! impl_fusion {
    ($ft: ty) => {
        impl<const N: usize> Fusion<&Simplex<$ft, N>, $ft> for Opinion1d<$ft, N> {
            type Output = Result<Simplex<$ft, N>, InvalidValueError>;
            fn cfuse_al(&self, rhs: &Simplex<$ft, N>, gamma_a: $ft) -> Self::Output {
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
                Simplex::<$ft, N>::try_new(b, u)
            }

            fn cfuse_ep(&self, rhs: &Simplex<$ft, N>, gamma_a: $ft) -> Self::Output {
                let s = self.cfuse_al(rhs, gamma_a)?;
                let w = Opinion1dRef::<$ft, N>::from((&s, &self.base_rate));
                let u_max = w.max_uncertainty();
                let b_max = array::from_fn(|i| w.projection(i) - w.base_rate[i] * u_max);
                Simplex::<$ft, N>::try_new(b_max, u_max)
            }

            fn afuse(&self, rhs: &Simplex<$ft, N>, gamma_a: $ft) -> Self::Output {
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
                Simplex::<$ft, N>::try_new(b, u)
            }

            fn wfuse(&self, rhs: &Simplex<$ft, N>, gamma_a: $ft) -> Self::Output {
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
                Simplex::<$ft, N>::try_new(b, u)
            }
        }

        impl<'a, A, const N: usize> Fusion<A, $ft> for Opinion1d<$ft, N>
        where
            A: Into<Opinion1dRef<'a, $ft, N>>,
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

        impl<const N: usize> FusionAssign<&Simplex<$ft, N>, $ft> for Opinion1d<$ft, N> {
            type Output = Result<(), InvalidValueError>;

            fn fusion_assign(&mut self, rhs: &Simplex<$ft, N>, op: &FusionOp<$ft>) -> Self::Output {
                self.simplex = match *op {
                    FusionOp::CumulativeA(gamma_a) => self.cfuse_al(rhs, gamma_a),
                    FusionOp::CumulativeE(gamma_a) => self.cfuse_ep(rhs, gamma_a),
                    FusionOp::Averaging(gamma_a) => self.afuse(rhs, gamma_a),
                    FusionOp::Weighted(gamma_a) => self.wfuse(rhs, gamma_a),
                }?;
                Ok(())
            }
        }

        impl<const N: usize> FusionAssign<&Opinion1d<$ft, N>, $ft> for Opinion1d<$ft, N> {
            type Output = Result<(), InvalidValueError>;

            fn fusion_assign(
                &mut self,
                rhs: &Opinion1d<$ft, N>,
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
        impl<const M: usize, const N: usize> Deduction<[Simplex<$ft, N>; M], [$ft; N]>
            for &Opinion1d<$ft, M>
        {
            type Output = Opinion1d<$ft, N>;
            fn deduce(self, conds: [Simplex<$ft, N>; M], ay: [$ft; N]) -> Self::Output {
                self.deduce(&conds, ay)
            }
        }

        // impl<'a, A, const M: usize, const N: usize> Deduction<&'a [A; M], [$ft; N]>
        //     for MOpinion1dRef<'a, $ft, M>
        // where
        //     &'a A: Into<&'a MSimplex<$ft, N>>,
        // {
        //     type Output = MOpinion1d<$ft, N>;

        //     fn deduce(self, conds: &'a [A; M], ay: [$ft; N]) -> Self::Output {
        //         MOpinion1dRef::from(self).deduce(conds, ay)
        //     }
        // }

        impl<'a, A, B, const M: usize, const N: usize> Deduction<&'a [A; M], [$ft; N]> for B
        where
            &'a A: Into<&'a Simplex<$ft, N>>,
            B: Into<Opinion1dRef<'a, $ft, M>>,
        {
            type Output = Opinion1d<$ft, N>;

            fn deduce(self, conds: &'a [A; M], ay: [$ft; N]) -> Self::Output {
                let w = self.into();
                assert!(M > 0 && N > 1, "N > 0 and M > 1 must hold.");
                let conds: [&Simplex<$ft, N>; M] = each_into(conds);
                let ay = conds.marginal_base_rate(&w.base_rate).unwrap_or(ay);

                let cond_p: [[$ft; N]; M] = array::from_fn(|i| conds[i].projection(&ay));
                let pyhx: [$ft; N] =
                    array::from_fn(|j| (0..M).map(|i| w.base_rate[i] * cond_p[i][j]).sum());
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
                        .map(|i| (uyhx - conds[i].uncertainty) * w.b()[i])
                        .sum::<$ft>();
                let b: [$ft; N] = array::from_fn(|j| {
                    (0..M).map(|i| w.projection(i) * cond_p[i][j]).sum::<$ft>() - ay[j] * u
                });
                Opinion1d::<$ft, N>::new_unchecked(b, u, ay)
            }
        }
    };
}

impl_deduction!(f32);
impl_deduction!(f64);

macro_rules! impl_abduction {
    ($ft: ty) => {
        impl<const N: usize, const M: usize> Abduction<&[Simplex<$ft, N>; M], [$ft; M], [$ft; N]>
            for &Simplex<$ft, N>
        {
            type Output = Opinion1d<$ft, M>;

            fn abduce(
                self,
                conds_yx: &[Simplex<$ft, N>; M],
                ax: [$ft; M],
                ay: Option<[$ft; N]>,
            ) -> Option<(Self::Output, [$ft; N])> {
                fn inverse<'a, const N: usize, const M: usize>(
                    conds_yx: &[Simplex<$ft, N>; M],
                    ax: &[$ft; M],
                    ay: &[$ft; N],
                ) -> [Simplex<$ft, M>; N] {
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
                let w = Opinion1dRef::<$ft, N>::from((self, &ay));
                Some((w.deduce(&inv_conds, ax), ay))
            }
        }
    };
}

impl_abduction!(f32);
impl_abduction!(f64);
