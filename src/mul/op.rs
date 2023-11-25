use approx::ulps_eq;
use std::{array, ops::Index};

use super::{
    IndexedContainer, Opinion, Opinion1d, Opinion1dRef, OpinionRef, Simplex, SimplexBase, MBR,
};
use crate::errors::InvalidValueError;

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
                let wp = w.projection();
                let u_max = w.max_uncertainty();
                let b_max = array::from_fn(|i| wp[i] - w.base_rate[i] * u_max);
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

/// The deduction operator.
pub trait Deduction<X, Y, Cond, U>: Sized {
    type Output;

    /// Computes the conditionally deduced opinion of `self` with a the marginal base rate (MBR)
    /// by `conds` representing a collection of conditional opinions.
    /// If all conditional opinions are vacuous, i.e. \\(\forall x. u_{Y|x} = 1\\),
    /// then MBR cannot be determined and a vacuous opinion is deduced.
    fn deduce(self, conds: Cond) -> Option<Self::Output>;
    fn deduce_with(self, conds: Cond, ay: U) -> Self::Output;
}

macro_rules! impl_deduction {
    ($ft: ty) => {
        impl<'b, X, Y, Cond, U, T> Deduction<X, Y, &'b Cond, U> for &'b Opinion<T, $ft>
        where
            Cond: IndexedContainer<X>,
            for<'a> &'a Cond::Output: Into<&'a SimplexBase<U, $ft>>,
            T: IndexedContainer<X, Output = $ft>,
            U: IndexedContainer<Y, Output = $ft>,
            X: Copy,
            Y: Copy,
        {
            type Output = Opinion<U, $ft>;
            fn deduce(self, conds: &'b Cond) -> Option<Self::Output> {
                self.as_ref().deduce(conds)
            }

            fn deduce_with(self, conds: &'b Cond, ay: U) -> Self::Output {
                self.as_ref().deduce_with(conds, ay)
            }
        }

        impl<'b, X, Y, Cond, U, T> Deduction<X, Y, &'b Cond, U> for OpinionRef<'b, T, $ft>
        where
            T: IndexedContainer<X, Output = $ft>,
            U: IndexedContainer<Y, Output = $ft> + MBR<X, Y, T, &'b Cond, U, $ft>,
            Cond: IndexedContainer<X>,
            for<'a> &'a Cond::Output: Into<&'a SimplexBase<U, $ft>>,
            X: Copy,
            Y: Copy,
        {
            type Output = Opinion<U, $ft>;

            fn deduce(self, conds: &'b Cond) -> Option<Self::Output> {
                let ay = U::marginal_base_rate(&self.base_rate, conds)?;
                Some(self.deduce_with(conds, ay))
            }

            fn deduce_with(self, conds: &'b Cond, ay: U) -> Self::Output {
                assert!(
                    T::SIZE > 0 && U::SIZE > 1,
                    "conds.len() > 0 and ay.len() > 1 must hold."
                );
                let cond_p: Cond::Map<U> = Cond::map(|x| conds[x].into().projection(&ay));
                let pyhx: U =
                    U::from_fn(|y| Cond::keys().map(|x| self.base_rate[x] * cond_p[x][y]).sum());
                let uyhx = U::keys()
                    .map(|y| {
                        (pyhx[y]
                            - Cond::keys()
                                .map(|x| conds[x].into().belief[y])
                                .reduce(<$ft>::min)
                                .unwrap())
                            / ay[y]
                    })
                    .reduce(<$ft>::min)
                    .unwrap();
                let u = uyhx
                    - Cond::keys()
                        .map(|x| (uyhx - conds[x].into().uncertainty) * self.b()[x])
                        .sum::<$ft>();
                let p = self.projection();
                let b =
                    U::from_fn(|y| T::keys().map(|x| p[x] * cond_p[x][y]).sum::<$ft>() - ay[y] * u);
                Opinion::<U, $ft>::new_unchecked(b, u, ay)
            }
        }
    };
}

impl_deduction!(f32);
impl_deduction!(f64);

trait InverseCondition<X, Y, T, U, V>
where
    Self: Index<X, Output = SimplexBase<U, V>>,
    T: Index<X, Output = V>,
    U: Index<Y, Output = V>,
{
    type InvCond;
    fn inverse(&self, ax: &T, ay: &U) -> Self::InvCond;
}

macro_rules! impl_inverse_condition {
    ($ft: ty) => {
        impl<Cond, T, U, X, Y> InverseCondition<X, Y, T, U, $ft> for Cond
        where
            T: IndexedContainer<X, Output = $ft>,
            U: IndexedContainer<Y, Output = $ft>,
            Cond: IndexedContainer<X, Output = SimplexBase<U, $ft>>,
            X: Copy,
            Y: Copy,
        {
            type InvCond = U::Map<SimplexBase<T, $ft>>;
            fn inverse(&self, ax: &T, ay: &U) -> Self::InvCond {
                let p_yx: Cond::Map<U> = Cond::map(|x| self[x].projection(ay));
                let p_xy: U::Map<T::Map<$ft>> = U::map(|y| {
                    T::map(|x| {
                        ax[x] * p_yx[x][y] / T::keys().map(|xd| ax[xd] * p_yx[xd][y]).sum::<$ft>()
                    })
                });
                let u_yx_sum = Cond::keys().map(|x| self[x].uncertainty).sum::<$ft>();
                let irrelevance_yx = U::from_fn(|y| {
                    1.0 - T::keys().map(|x| p_yx[x][y]).reduce(<$ft>::max).unwrap()
                        + T::keys().map(|x| p_yx[x][y]).reduce(<$ft>::min).unwrap()
                });
                let weights_yx = if u_yx_sum == 0.0 {
                    T::from_fn(|_| 0.0)
                } else {
                    T::from_fn(|x| self[x].uncertainty / u_yx_sum)
                };
                let u_yx_marginal = T::from_fn(|x| {
                    U::keys()
                        .map(|y| p_yx[x][y] / ay[y])
                        .reduce(<$ft>::min)
                        .unwrap()
                });
                let u_yx_weight = T::from_fn(|x| {
                    let tmp = u_yx_marginal[x];
                    if tmp == 0.0 {
                        0.0
                    } else {
                        weights_yx[x] * self[x].uncertainty / tmp
                    }
                });
                let u_yx_exp: $ft = T::keys().map(|x| u_yx_weight[x]).sum();
                let u_xy_marginal: U = U::from_fn(|y| {
                    T::keys()
                        .map(|x| p_yx[x][y] / T::keys().map(|k| ax[k] * p_yx[k][y]).sum::<$ft>())
                        .reduce(<$ft>::min)
                        .unwrap()
                });
                U::map(|y| {
                    let u = u_xy_marginal[y] * (u_yx_exp + (1.0 - u_yx_exp) * irrelevance_yx[y]);
                    let b = T::from_fn(|x| p_xy[y][x] - u * ax[x]);
                    SimplexBase::new_unchecked(b, u)
                })
            }
        }
    };
}

impl_inverse_condition!(f32);
impl_inverse_condition!(f64);

/// The abduction operator.
pub trait Abduction<Cond, X, Y, T, U>
where
    T: Index<X>,
    U: Index<Y>,
{
    type Output;

    /// Computes the conditionally abduced opinion of `self` with a base rate vector `ax`
    /// by `conds` representing a collection of conditional opinions.
    /// If a marginal base rate cannot be computed from `conds`, ay is used instead.
    fn abduce(self, conds: Cond, ax: T) -> Option<(Self::Output, U)>;
    fn abduce_with(self, conds: Cond, ax: T, ay: &U) -> Self::Output;
}

macro_rules! impl_abduction {
    ($ft: ty) => {
        impl<'a, Cond, X, Y, T, U> Abduction<&'a Cond, X, Y, T, U> for &'a SimplexBase<U, $ft>
        where
            Cond: InverseCondition<X, Y, T, U, $ft>
                + IndexedContainer<X, Output = SimplexBase<U, $ft>>,
            Cond::InvCond: IndexedContainer<Y, Output = SimplexBase<T, $ft>> + 'a,
            T: IndexedContainer<X, Output = $ft> + 'a,
            U: IndexedContainer<Y, Output = $ft> + MBR<X, Y, T, &'a Cond, U, $ft>,
            X: Copy,
            Y: Copy,
        {
            type Output = Opinion<T, $ft>;

            fn abduce(self, conds: &'a Cond, ax: T) -> Option<(Self::Output, U)> {
                let ay = U::marginal_base_rate(&ax, conds)?;
                Some((self.abduce_with(conds, ax, &ay), ay))
            }

            fn abduce_with(self, conds: &'a Cond, ax: T, ay: &U) -> Self::Output {
                let inv_conds = InverseCondition::inverse(conds, &ax, ay);
                OpinionRef::<U, $ft>::from((self, ay)).deduce_with(&inv_conds, ax)
            }
        }
    };
}

impl_abduction!(f32);
impl_abduction!(f64);

#[cfg(test)]
mod tests {
    use crate::mul::{op::Deduction, Opinion1d, Simplex};

    #[test]
    fn test_deduction() {
        let wx = Opinion1d::<f32, 2>::new([0.9, 0.0], 0.1, [0.1, 0.9]);
        let wxy = [
            Simplex::<f32, 3>::new([0.0, 0.8, 0.1], 0.1),
            Simplex::<f32, 3>::new([0.7, 0.0, 0.1], 0.2),
        ];
        let wy = wx.as_ref().deduce(&wxy).unwrap();
        // base rate
        assert_eq!(
            wy.base_rate.map(|a| (a * 10f32.powi(3)).round()),
            [778.0, 99.0, 123.0]
        );
        // projection
        let p = wy.projection();
        assert_eq!(
            p.map(|p| (p * 10f32.powi(3)).round()),
            [148.0, 739.0, 113.0]
        );
        // belief
        assert_eq!(
            wy.b().map(|p| (p * 10f32.powi(3)).round()),
            [63.0, 728.0, 100.0]
        );
        // uncertainty
        assert_eq!((wy.u() * 10f32.powi(3)).round(), 109.0)
    }
}
