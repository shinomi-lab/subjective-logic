use approx::ulps_eq;
use std::{array, ops::Index};

use super::{IndexedContainer, Opinion, Opinion1d, Opinion1dRef, OpinionRef, SimplexBase, MBR};
use crate::errors::InvalidValueError;

#[derive(Clone, Copy)]
pub enum FuseOp {
    /// Aleatory cumulative fusion operator
    ACm,
    /// Epistemic cumulative fusion operator
    ECm,
    /// Averaging belief fusion of operator
    Avg,
    /// Weighted belief fusion of operator
    Wgh,
}

pub trait FuseAssign<T> {
    type Err;
    fn fuse_assign(&self, lhs: &mut T, rhs: &T) -> Result<(), Self::Err>;
}

pub trait Fuse<T> {
    type Output;
    fn fuse(&self, lhs: T, rhs: T) -> Self::Output;
}

macro_rules! impl_fusion {
    ($ft: ty) => {
        impl<const N: usize> Fuse<&Opinion1d<$ft, N>> for FuseOp {
            type Output = Result<Opinion1d<$ft, N>, InvalidValueError>;

            fn fuse(&self, lhs: &Opinion1d<$ft, N>, rhs: &Opinion1d<$ft, N>) -> Self::Output {
                self.fuse(lhs.as_ref(), rhs.as_ref())
            }
        }

        impl<'a, const N: usize> Fuse<Opinion1dRef<'a, $ft, N>> for FuseOp {
            type Output = Result<Opinion1d<$ft, N>, InvalidValueError>;

            fn fuse(
                &self,
                lhs: Opinion1dRef<'a, $ft, N>,
                rhs: Opinion1dRef<'a, $ft, N>,
            ) -> Self::Output {
                if ulps_eq!(*lhs.u(), 0.0) && ulps_eq!(*rhs.u(), 0.0) {
                    let gamma_a = 0.5;
                    let gamma_b = 1.0 - gamma_a;
                    let b = array::from_fn(|i| gamma_a * lhs.b()[i] + gamma_b * rhs.b()[i]);
                    let u = 0.0;
                    let a =
                        array::from_fn(|i| gamma_a * lhs.base_rate[i] + gamma_b * rhs.base_rate[i]);
                    let w = Opinion1d::<$ft, N>::new_unchecked(b, u, a);
                    if matches!(self, FuseOp::ECm) {
                        return w.op_u_max();
                    } else {
                        return Ok(w);
                    }
                }
                let lhs_u = *lhs.u();
                let rhs_u = *rhs.u();
                match self {
                    FuseOp::ACm | FuseOp::ECm => {
                        let temp = lhs_u + rhs_u - lhs_u * rhs_u;
                        let b =
                            array::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                        let u = lhs_u * rhs_u / temp;
                        let a = if ulps_eq!(lhs_u, 1.0) && ulps_eq!(rhs_u, 1.0) {
                            array::from_fn(|i| (lhs.base_rate[i] + rhs.base_rate[i]) / 2.0)
                        } else {
                            let temp2 = temp - lhs_u * rhs_u;
                            let lhs_sum_b = 1.0 - lhs_u;
                            let rhs_sum_b = 1.0 - rhs_u;
                            array::from_fn(|i| {
                                if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                                    lhs.base_rate[i]
                                } else {
                                    (lhs.base_rate[i] * rhs_u * lhs_sum_b
                                        + rhs.base_rate[i] * lhs_u * rhs_sum_b)
                                        / temp2
                                }
                            })
                        };
                        let w = Opinion1d::<$ft, N>::try_new(b, u, a);
                        if matches!(self, FuseOp::ECm) {
                            w?.op_u_max()
                        } else {
                            w
                        }
                    }
                    FuseOp::Avg => {
                        let temp = lhs_u + rhs_u;
                        let b =
                            array::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                        let u = 2.0 * lhs_u * rhs_u / temp;
                        let a = array::from_fn(|i| (lhs.base_rate[i] + rhs.base_rate[i]) / 2.0);
                        Opinion1d::<$ft, N>::try_new(b, u, a)
                    }
                    FuseOp::Wgh => {
                        let b;
                        let u;
                        let a;
                        if ulps_eq!(lhs_u, 1.0) && ulps_eq!(rhs_u, 1.0) {
                            b = [0.0; N];
                            u = 1.0;
                            a = array::from_fn(|i| (lhs.base_rate[i] + rhs.base_rate[i]) / 2.0);
                        } else {
                            let lhs_sum_b = 1.0 - lhs_u;
                            let rhs_sum_b = 1.0 - rhs_u;
                            let temp = lhs_u + rhs_u - 2.0 * lhs_u * rhs_u;
                            let temp2 = lhs_sum_b + rhs_sum_b;
                            b = array::from_fn(|i| {
                                (lhs.b()[i] * lhs_sum_b * rhs_u + rhs.b()[i] * rhs_sum_b * lhs_u)
                                    / temp
                            });
                            u = temp2 * lhs_u * rhs_u / temp;
                            a = array::from_fn(|i| {
                                (lhs.base_rate[i] * lhs_sum_b + rhs.base_rate[i] * rhs_sum_b)
                                    / temp2
                            });
                        }
                        Opinion1d::<$ft, N>::try_new(b, u, a)
                    }
                }
            }
        }

        impl<const N: usize> FuseAssign<Opinion1d<$ft, N>> for FuseOp {
            type Err = InvalidValueError;

            fn fuse_assign(
                &self,
                lhs: &mut Opinion1d<$ft, N>,
                rhs: &Opinion1d<$ft, N>,
            ) -> Result<(), Self::Err> {
                *lhs = self.fuse(&(*lhs), &rhs)?;
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

    /// Computes the conditionally deduced opinion of `self` with
    /// a the marginal base rate (MBR) given by `conds` representing a collection of conditional opinions.
    /// If all conditional opinions are vacuous, i.e. \\(\forall x. u_{Y|x} = 1\\),
    /// then MBR cannot be determined so return None (vacuous opinion is deduced).
    fn deduce(self, conds: Cond) -> Option<Self::Output>;

    /// Computes the conditionally deduced opinion of `self` with a base rate `ay`.
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
    use crate::mul::{
        op::{Deduction, Fuse, FuseAssign, FuseOp},
        Opinion1d, Simplex,
    };

    fn nround<const N: i32>(v: f32) -> f32 {
        (v * 10f32.powi(N)).round()
    }

    fn nfract<const N: i32>(v: f32) -> f32 {
        (v * 10f32.powi(N)).fract()
    }

    #[test]
    fn test_fusion_ref() {
        let w1 = Opinion1d::<f32, 2>::new([0.5, 0.0], 0.5, [0.25, 0.75]);
        let a = [0.5, 0.5];
        let s = Simplex::<f32, 2>::new([0.0, 0.9], 0.1);
        let w2 = Opinion1d::<f32, 2>::from_simplex_unchecked(s.clone(), a.clone());
        assert_eq!(
            FuseOp::ACm.fuse(&w1, &w2).unwrap(),
            FuseOp::ACm.fuse(w1.as_ref(), (&s, &a).into()).unwrap()
        );
        assert_eq!(
            FuseOp::ECm.fuse(&w1, &w2).unwrap(),
            FuseOp::ECm.fuse(w1.as_ref(), (&s, &a).into()).unwrap()
        );
        assert_eq!(
            FuseOp::Avg.fuse(&w1, &w2).unwrap(),
            FuseOp::Avg.fuse(w1.as_ref(), (&s, &a).into()).unwrap()
        );
        assert_eq!(
            FuseOp::Wgh.fuse(&w1, &w2).unwrap(),
            FuseOp::Wgh.fuse(w1.as_ref(), (&s, &a).into()).unwrap()
        );
    }

    #[test]
    fn test_fusion_dogma() {
        let w1 =
            Opinion1d::<f32, 3>::new([0.99, 0.01, 0.0], 0.0, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let w2 =
            Opinion1d::<f32, 3>::new([0.0, 0.01, 0.99], 0.0, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);

        // A-CBF
        let w = FuseOp::ACm.fuse(&w1, &w2).unwrap();
        assert_eq!(w.b().map(nround::<3>), [495.0, 10.0, 495.0]);
        assert_eq!(nround::<3>(*w.u()), 0.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // E-CBF
        let w = FuseOp::ECm.fuse(&w1, &w2).unwrap();
        assert_eq!(w.b().map(nround::<3>), [485.0, 0.0, 485.0]);
        assert_eq!(nround::<3>(*w.u()), 30.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // ABF
        let w = FuseOp::Avg.fuse(&w1, &w2).unwrap();
        assert_eq!(w.b().map(nround::<3>), [495.0, 10.0, 495.0]);
        assert_eq!(nround::<3>(*w.u()), 0.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // WBF
        let w = FuseOp::Wgh.fuse(&w1, &w2).unwrap();
        assert_eq!(w.b().map(nround::<3>), [495.0, 10.0, 495.0]);
        assert_eq!(nround::<3>(*w.u()), 0.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
    }

    #[test]
    fn test_fusion() {
        let w1 =
            Opinion1d::<f32, 3>::new([0.98, 0.01, 0.0], 0.01, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let w2 =
            Opinion1d::<f32, 3>::new([0.0, 0.01, 0.90], 0.09, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);

        // A-CBF
        let w = FuseOp::ACm.fuse(&w1, &w2).unwrap();
        assert_eq!(w.b().map(nround::<3>), [890.0, 10.0, 91.0]);
        assert_eq!(nround::<3>(*w.u()), 9.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // E-CBF
        let w = FuseOp::ECm.fuse(&w1, &w2).unwrap();
        assert_eq!(w.b().map(nround::<3>), [880.0, 0.0, 81.0]);
        assert_eq!(nround::<3>(*w.u()), 39.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // ABF
        let w = FuseOp::Avg.fuse(&w1, &w2).unwrap();
        assert_eq!(w.b().map(nround::<3>), [882.0, 10.0, 90.0]);
        assert_eq!(nround::<3>(*w.u()), 18.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // WBF
        let w = FuseOp::Wgh.fuse(&w1, &w2).unwrap();
        assert_eq!(w.b().map(nround::<3>), [889.0, 10.0, 83.0]);
        assert_eq!(
            nround::<3>(*w.u()),
            18.0 - w.b().map(nfract::<3>).iter().sum::<f32>().round()
        );
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
    }

    #[test]
    fn test_fusion_assign() {
        let mut w = Opinion1d::<f32, 2>::new([0.5, 0.25], 0.25, [0.5, 0.5]);
        let u = Opinion1d::<f32, 2>::new([0.125, 0.75], 0.125, [0.75, 0.25]);
        let ops = [FuseOp::ACm, FuseOp::ECm, FuseOp::Avg, FuseOp::Wgh];
        for op in ops {
            let w2 = op.fuse(&w, &u).unwrap();
            op.fuse_assign(&mut w, &u).unwrap();
            assert!(w == w2);
        }
    }

    #[test]
    fn test_deduction() {
        let wx = Opinion1d::<f32, 2>::new([0.9, 0.0], 0.1, [0.1, 0.9]);
        let wxy = [
            Simplex::<f32, 3>::new([0.0, 0.8, 0.1], 0.1),
            Simplex::<f32, 3>::new([0.7, 0.0, 0.1], 0.2),
        ];
        let wy = wx.as_ref().deduce(&wxy).unwrap();
        // base rate
        assert_eq!(wy.base_rate.map(nround::<3>), [778.0, 99.0, 123.0]);
        // projection
        let p = wy.projection();
        assert_eq!(p.map(nround::<3>), [148.0, 739.0, 113.0]);
        // belief
        assert_eq!(wy.b().map(nround::<3>), [63.0, 728.0, 100.0]);
        // uncertainty
        assert_eq!(nround::<3>(*wy.u()), 109.0);
    }
}
