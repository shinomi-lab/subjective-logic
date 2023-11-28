use approx::ulps_eq;
use std::ops::Index;

use super::{IndexedContainer, MaxUncertainty, Opinion, OpinionRef, Projection, SimplexBase, MBR};

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

pub trait Fuse<L, R, Idx> {
    type Output;
    fn fuse(&self, lhs: L, rhs: R) -> Self::Output;
}

pub trait FuseAssign<L, R, Idx> {
    fn fuse_assign(&self, lhs: &mut L, rhs: R);
}

trait InnerFuse<T, Idx, V>
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    Idx: Copy,
{
    type Output;
    fn compute_simlex(&self, lhs: &SimplexBase<T, V>, rhs: &SimplexBase<T, V>) -> Self::Output;
    fn compute_base_rate(&self, lhs: OpinionRef<'_, T, V>, rhs: OpinionRef<'_, T, V>) -> T;
}

macro_rules! impl_fusion {
    ($ft: ty) => {
        impl<T, Idx> InnerFuse<T, Idx, $ft> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            type Output = SimplexBase<T, $ft>;
            fn compute_simlex(
                &self,
                lhs: &SimplexBase<T, $ft>,
                rhs: &SimplexBase<T, $ft>,
            ) -> Self::Output {
                if lhs.is_dogmatic() && rhs.is_dogmatic() {
                    Self::Output::new_unchecked(
                        T::from_fn(|i| (lhs.b()[i] + rhs.b()[i]) / 2.0),
                        0.0,
                    )
                } else {
                    match self {
                        FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() && rhs.is_vacuous() => {
                            Self::Output::vacuous()
                        }
                        FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() || rhs.is_dogmatic() => {
                            rhs.clone()
                        }
                        FuseOp::ACm | FuseOp::ECm if rhs.is_vacuous() || lhs.is_dogmatic() => {
                            lhs.clone()
                        }
                        FuseOp::ACm | FuseOp::ECm => {
                            let lhs_u = lhs.u();
                            let rhs_u = rhs.u();
                            let temp = lhs_u + rhs_u - lhs_u * rhs_u;
                            let b =
                                T::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                            let u = lhs_u * rhs_u / temp;
                            Self::Output::new_unchecked(b, u)
                        }
                        FuseOp::Avg if lhs.is_dogmatic() => lhs.clone(),
                        FuseOp::Avg if rhs.is_dogmatic() => rhs.clone(),
                        FuseOp::Avg => {
                            let lhs_u = lhs.u();
                            let rhs_u = rhs.u();
                            let temp = lhs_u + rhs_u;
                            let b =
                                T::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                            let u = 2.0 * lhs_u * rhs_u / temp;
                            Self::Output::new_unchecked(b, u)
                        }
                        FuseOp::Wgh if lhs.is_vacuous() && rhs.is_vacuous() => {
                            Self::Output::vacuous()
                        }
                        FuseOp::Wgh if lhs.is_vacuous() || rhs.is_dogmatic() => rhs.clone(),
                        FuseOp::Wgh if rhs.is_vacuous() || lhs.is_dogmatic() => lhs.clone(),
                        FuseOp::Wgh => {
                            let lhs_u = lhs.u();
                            let rhs_u = rhs.u();
                            let lhs_sum_b = 1.0 - lhs_u;
                            let rhs_sum_b = 1.0 - rhs_u;
                            let temp = lhs_u + rhs_u - 2.0 * lhs_u * rhs_u;
                            let b = T::from_fn(|i| {
                                (lhs.b()[i] * lhs_sum_b * rhs_u + rhs.b()[i] * rhs_sum_b * lhs_u)
                                    / temp
                            });
                            let u = (lhs_sum_b + rhs_sum_b) * lhs_u * rhs_u / temp;
                            Self::Output::new_unchecked(b, u)
                        }
                    }
                }
            }

            fn compute_base_rate(
                &self,
                lhs: OpinionRef<'_, T, $ft>,
                rhs: OpinionRef<'_, T, $ft>,
            ) -> T {
                if std::ptr::eq(lhs.base_rate, rhs.base_rate) {
                    lhs.base_rate.clone()
                } else if lhs.is_dogmatic() && rhs.is_dogmatic() {
                    T::from_fn(|i| (lhs.base_rate[i] + rhs.base_rate[i]) / 2.0)
                } else {
                    match self {
                        FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() && rhs.is_vacuous() => {
                            T::from_fn(|i| {
                                if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                                    lhs.base_rate[i]
                                } else {
                                    (lhs.base_rate[i] + rhs.base_rate[i]) / 2.0
                                }
                            })
                        }
                        FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() || rhs.is_dogmatic() => {
                            rhs.base_rate.clone()
                        }
                        FuseOp::ACm | FuseOp::ECm if rhs.is_vacuous() || lhs.is_dogmatic() => {
                            lhs.base_rate.clone()
                        }
                        FuseOp::ACm | FuseOp::ECm => {
                            let lhs_u = lhs.u();
                            let rhs_u = rhs.u();
                            let temp = lhs_u + rhs_u - lhs_u * rhs_u * 2.0;
                            let lhs_sum_b = 1.0 - lhs_u;
                            let rhs_sum_b = 1.0 - rhs_u;
                            T::from_fn(|i| {
                                if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                                    lhs.base_rate[i]
                                } else {
                                    (lhs.base_rate[i] * rhs_u * lhs_sum_b
                                        + rhs.base_rate[i] * lhs_u * rhs_sum_b)
                                        / temp
                                }
                            })
                        }
                        FuseOp::Avg => T::from_fn(|i| {
                            if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                                lhs.base_rate[i]
                            } else {
                                (lhs.base_rate[i] + rhs.base_rate[i]) / 2.0
                            }
                        }),
                        FuseOp::Wgh if lhs.is_vacuous() && rhs.is_vacuous() => T::from_fn(|i| {
                            if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                                lhs.base_rate[i]
                            } else {
                                (lhs.base_rate[i] + rhs.base_rate[i]) / 2.0
                            }
                        }),
                        FuseOp::Wgh if lhs.is_vacuous() => rhs.base_rate.clone(),
                        FuseOp::Wgh if rhs.is_vacuous() => lhs.base_rate.clone(),
                        FuseOp::Wgh => {
                            let lhs_u = lhs.u();
                            let rhs_u = rhs.u();
                            let lhs_sum_b = 1.0 - lhs_u;
                            let rhs_sum_b = 1.0 - rhs_u;
                            let temp = lhs_sum_b + rhs_sum_b;
                            T::from_fn(|i| {
                                if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                                    lhs.base_rate[i]
                                } else {
                                    (lhs.base_rate[i] * lhs_sum_b + rhs.base_rate[i] * rhs_sum_b)
                                        / temp
                                }
                            })
                        }
                    }
                }
            }
        }

        impl<'a, T, Idx> Fuse<OpinionRef<'a, T, $ft>, OpinionRef<'a, T, $ft>, Idx> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            type Output = Opinion<T, $ft>;

            fn fuse(
                &self,
                lhs: OpinionRef<'a, T, $ft>,
                rhs: OpinionRef<'a, T, $ft>,
            ) -> Self::Output {
                let s = self.compute_simlex(lhs.simplex, rhs.simplex);
                let a = self.compute_base_rate(lhs, rhs);
                let w = Self::Output::from_simplex_unchecked(s, a);
                if matches!(self, FuseOp::ECm) {
                    w.uncertainty_maximized()
                } else {
                    w
                }
            }
        }

        impl<T, Idx> Fuse<&Opinion<T, $ft>, &SimplexBase<T, $ft>, Idx> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            type Output = Opinion<T, $ft>;

            fn fuse(&self, lhs: &Opinion<T, $ft>, rhs: &SimplexBase<T, $ft>) -> Self::Output {
                self.fuse(lhs.as_ref(), rhs)
            }
        }

        impl<T, Idx> Fuse<&Opinion<T, $ft>, &Opinion<T, $ft>, Idx> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            type Output = Opinion<T, $ft>;

            fn fuse(&self, lhs: &Opinion<T, $ft>, rhs: &Opinion<T, $ft>) -> Self::Output {
                self.fuse(lhs.as_ref(), rhs.as_ref())
            }
        }

        impl<'a, T, Idx> Fuse<OpinionRef<'a, T, $ft>, &'a SimplexBase<T, $ft>, Idx> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            type Output = Opinion<T, $ft>;

            fn fuse(
                &self,
                lhs: OpinionRef<'a, T, $ft>,
                rhs: &'a SimplexBase<T, $ft>,
            ) -> Self::Output {
                self.fuse(lhs.clone(), OpinionRef::from((rhs, lhs.base_rate)))
            }
        }

        impl<T, Idx> Fuse<&SimplexBase<T, $ft>, &SimplexBase<T, $ft>, Idx> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            type Output = SimplexBase<T, $ft>;

            fn fuse(&self, lhs: &SimplexBase<T, $ft>, rhs: &SimplexBase<T, $ft>) -> Self::Output {
                if matches!(self, FuseOp::ECm) {
                    panic!("Epistemic fusion cannot be used with simplexes because there is no base rate.");
                }
                self.compute_simlex(lhs, rhs)
            }
        }

        impl<T, Idx> FuseAssign<Opinion<T, $ft>, &Opinion<T, $ft>, Idx> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            fn fuse_assign(&self, lhs: &mut Opinion<T, $ft>, rhs: &Opinion<T, $ft>) {
                self.fuse_assign(lhs, rhs.as_ref())
            }
        }

        impl<'a, T, Idx> FuseAssign<Opinion<T, $ft>, OpinionRef<'a, T, $ft>, Idx> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            fn fuse_assign(&self, lhs: &mut Opinion<T, $ft>, rhs: OpinionRef<'a, T, $ft>) {
                *lhs = self.fuse(lhs.as_ref(), rhs);
            }
        }

        impl<T, Idx> FuseAssign<Opinion<T, $ft>, &SimplexBase<T, $ft>, Idx> for FuseOp
        where
            T: IndexedContainer<Idx, Output = $ft> + Clone,
            Idx: Copy,
        {
            fn fuse_assign(&self, lhs: &mut Opinion<T, $ft>, rhs: &SimplexBase<T, $ft>) {
                *lhs = self.fuse(lhs.as_ref(), rhs);
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
        Opinion1d, OpinionRef, Projection, Simplex,
    };

    fn nround<const N: i32>(v: f32) -> f32 {
        (v * 10f32.powi(N)).round()
    }

    fn nfract<const N: i32>(v: f32) -> f32 {
        (v * 10f32.powi(N)).fract()
    }

    #[test]
    fn test_fusion_dogma() {
        let w1 =
            Opinion1d::<f32, 3>::new([0.99, 0.01, 0.0], 0.0, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let w2 = Simplex::<f32, 3>::new([0.0, 0.01, 0.99], 0.0);

        // A-CBF
        let w = FuseOp::ACm.fuse(&w1, &w2);
        assert_eq!(w.b().map(nround::<3>), [495.0, 10.0, 495.0]);
        assert_eq!(nround::<3>(w.u()), 0.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // E-CBF
        let w = FuseOp::ECm.fuse(&w1, &w2);
        assert_eq!(w.b().map(nround::<3>), [485.0, 0.0, 485.0]);
        assert_eq!(nround::<3>(w.u()), 30.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // ABF
        let w = FuseOp::Avg.fuse(&w1, &w2);
        assert_eq!(w.b().map(nround::<3>), [495.0, 10.0, 495.0]);
        assert_eq!(nround::<3>(w.u()), 0.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // WBF
        let w = FuseOp::Wgh.fuse(&w1, &w2);
        assert_eq!(w.b().map(nround::<3>), [495.0, 10.0, 495.0]);
        assert_eq!(nround::<3>(w.u()), 0.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
    }

    #[test]
    fn test_fusion() {
        let w1 =
            Opinion1d::<f32, 3>::new([0.98, 0.01, 0.0], 0.01, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let w2 =
            Opinion1d::<f32, 3>::new([0.0, 0.01, 0.90], 0.09, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);

        // A-CBF
        let w = FuseOp::ACm.fuse(&w1, &w2);
        assert_eq!(w.b().map(nround::<3>), [890.0, 10.0, 91.0]);
        assert_eq!(nround::<3>(w.u()), 9.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // E-CBF
        let w = FuseOp::ECm.fuse(&w1, &w2);
        assert_eq!(w.b().map(nround::<3>), [880.0, 0.0, 81.0]);
        assert_eq!(nround::<3>(w.u()), 39.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // ABF
        let w = FuseOp::Avg.fuse(&w1, &w2);
        assert_eq!(w.b().map(nround::<3>), [882.0, 10.0, 90.0]);
        assert_eq!(nround::<3>(w.u()), 18.0);
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
        // WBF
        let w = FuseOp::Wgh.fuse(&w1, &w2);
        assert_eq!(w.b().map(nround::<3>), [889.0, 10.0, 83.0]);
        assert_eq!(
            nround::<3>(w.u()),
            18.0 - w.b().map(nfract::<3>).iter().sum::<f32>().round()
        );
        assert_eq!(w.base_rate.map(nround::<3>), [333.0, 333.0, 333.0]);
    }

    #[test]
    fn test_fusion_assign() {
        let mut w = Opinion1d::<f32, 2>::new([0.5, 0.25], 0.25, [0.5, 0.5]);
        let u = Simplex::<f32, 2>::new([0.125, 0.75], 0.125);
        let ops = [FuseOp::ACm, FuseOp::ECm, FuseOp::Avg, FuseOp::Wgh];
        for op in ops {
            let w2 = op.fuse(w.as_ref(), OpinionRef::from((&u, &w.base_rate)));
            op.fuse_assign(&mut w, &u);
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
        assert_eq!(nround::<3>(wy.u()), 109.0);
    }
}
