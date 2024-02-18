use approx::{ulps_eq, UlpsEq};
use num_traits::Float;
use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, Index},
};

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

fn compute_simlex<T, V, Idx>(
    op: &FuseOp,
    lhs: &SimplexBase<T, V>,
    rhs: &SimplexBase<T, V>,
) -> SimplexBase<T, V>
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq,
    Idx: Copy,
{
    if lhs.is_dogmatic() && rhs.is_dogmatic() {
        SimplexBase::new_unchecked(
            T::from_fn(|i| (lhs.b()[i] + rhs.b()[i]) / (V::one() + V::one())),
            V::zero(),
        )
    } else {
        match op {
            FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() && rhs.is_vacuous() => {
                SimplexBase::vacuous()
            }
            FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() || rhs.is_dogmatic() => rhs.clone(),
            FuseOp::ACm | FuseOp::ECm if rhs.is_vacuous() || lhs.is_dogmatic() => lhs.clone(),
            FuseOp::ACm | FuseOp::ECm => {
                let lhs_u = *lhs.u();
                let rhs_u = *rhs.u();
                let temp = lhs_u + rhs_u - lhs_u * rhs_u;
                let b = T::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                let u = lhs_u * rhs_u / temp;
                SimplexBase::new_unchecked(b, u)
            }
            FuseOp::Avg if lhs.is_dogmatic() => lhs.clone(),
            FuseOp::Avg if rhs.is_dogmatic() => rhs.clone(),
            FuseOp::Avg => {
                let lhs_u = *lhs.u();
                let rhs_u = *rhs.u();
                let temp = lhs_u + rhs_u;
                let b = T::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                let u = (V::one() + V::one()) * lhs_u * rhs_u / temp;
                SimplexBase::new_unchecked(b, u)
            }
            FuseOp::Wgh if lhs.is_vacuous() && rhs.is_vacuous() => SimplexBase::vacuous(),
            FuseOp::Wgh if lhs.is_vacuous() || rhs.is_dogmatic() => rhs.clone(),
            FuseOp::Wgh if rhs.is_vacuous() || lhs.is_dogmatic() => lhs.clone(),
            FuseOp::Wgh => {
                let lhs_u = *lhs.u();
                let rhs_u = *rhs.u();
                let lhs_sum_b = V::one() - lhs_u;
                let rhs_sum_b = V::one() - rhs_u;
                let temp = lhs_u + rhs_u - (V::one() + V::one()) * lhs_u * rhs_u;
                let b = T::from_fn(|i| {
                    (lhs.b()[i] * lhs_sum_b * rhs_u + rhs.b()[i] * rhs_sum_b * lhs_u) / temp
                });
                let u = (lhs_sum_b + rhs_sum_b) * lhs_u * rhs_u / temp;
                SimplexBase::new_unchecked(b, u)
            }
        }
    }
}

fn compute_base_rate<T, V, Idx>(
    op: &FuseOp,
    lhs: OpinionRef<'_, T, V>,
    rhs: OpinionRef<'_, T, V>,
) -> T
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    Idx: Copy,
    V: Float + UlpsEq,
{
    if std::ptr::eq(lhs.base_rate, rhs.base_rate) {
        lhs.base_rate.clone()
    } else if lhs.is_dogmatic() && rhs.is_dogmatic() {
        T::from_fn(|i| (lhs.base_rate[i] + rhs.base_rate[i]) / (V::one() + V::one()))
    } else {
        match op {
            FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() && rhs.is_vacuous() => T::from_fn(|i| {
                if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                    lhs.base_rate[i]
                } else {
                    (lhs.base_rate[i] + rhs.base_rate[i]) / (V::one() + V::one())
                }
            }),
            FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() || rhs.is_dogmatic() => {
                rhs.base_rate.clone()
            }
            FuseOp::ACm | FuseOp::ECm if rhs.is_vacuous() || lhs.is_dogmatic() => {
                lhs.base_rate.clone()
            }
            FuseOp::ACm | FuseOp::ECm => {
                let lhs_u = lhs.u();
                let rhs_u = rhs.u();
                let temp = lhs_u + rhs_u - lhs_u * rhs_u * (V::one() + V::one());
                let lhs_sum_b = V::one() - lhs_u;
                let rhs_sum_b = V::one() - rhs_u;
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
                    (lhs.base_rate[i] + rhs.base_rate[i]) / (V::one() + V::one())
                }
            }),
            FuseOp::Wgh if lhs.is_vacuous() && rhs.is_vacuous() => T::from_fn(|i| {
                if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                    lhs.base_rate[i]
                } else {
                    (lhs.base_rate[i] + rhs.base_rate[i]) / (V::one() + V::one())
                }
            }),
            FuseOp::Wgh if lhs.is_vacuous() => rhs.base_rate.clone(),
            FuseOp::Wgh if rhs.is_vacuous() => lhs.base_rate.clone(),
            FuseOp::Wgh => {
                let lhs_u = lhs.u();
                let rhs_u = rhs.u();
                let lhs_sum_b = V::one() - lhs_u;
                let rhs_sum_b = V::one() - rhs_u;
                let temp = lhs_sum_b + rhs_sum_b;
                T::from_fn(|i| {
                    if ulps_eq!(lhs.base_rate[i], rhs.base_rate[i]) {
                        lhs.base_rate[i]
                    } else {
                        (lhs.base_rate[i] * lhs_sum_b + rhs.base_rate[i] * rhs_sum_b) / temp
                    }
                })
            }
        }
    }
}

impl<'a, T, V, Idx> Fuse<OpinionRef<'a, T, V>, OpinionRef<'a, T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: OpinionRef<'a, T, V>, rhs: OpinionRef<'a, T, V>) -> Self::Output {
        let s = compute_simlex(self, lhs.simplex, rhs.simplex);
        let a = compute_base_rate(self, lhs, rhs);
        let s = if matches!(self, FuseOp::ECm) {
            s.uncertainty_maximized(&a)
        } else {
            s
        };
        (s, a).into()
    }
}

impl<T, V, Idx> Fuse<&Opinion<T, V>, &SimplexBase<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: &Opinion<T, V>, rhs: &SimplexBase<T, V>) -> Self::Output {
        self.fuse(lhs.as_ref(), rhs)
    }
}

impl<T, V, Idx> Fuse<&Opinion<T, V>, &Opinion<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: &Opinion<T, V>, rhs: &Opinion<T, V>) -> Self::Output {
        self.fuse(lhs.as_ref(), rhs.as_ref())
    }
}

impl<'a, T, V, Idx> Fuse<OpinionRef<'a, T, V>, &'a SimplexBase<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: OpinionRef<'a, T, V>, rhs: &'a SimplexBase<T, V>) -> Self::Output {
        self.fuse(lhs.clone(), OpinionRef::from((rhs, lhs.base_rate)))
    }
}

impl<T, V, Idx> Fuse<&SimplexBase<T, V>, &SimplexBase<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = SimplexBase<T, V>;

    fn fuse(&self, lhs: &SimplexBase<T, V>, rhs: &SimplexBase<T, V>) -> Self::Output {
        if matches!(self, FuseOp::ECm) {
            panic!("Epistemic fusion cannot be used with simplexes because there is no base rate.");
        }
        compute_simlex(self, lhs, rhs)
    }
}

impl<T, V, Idx> FuseAssign<Opinion<T, V>, &Opinion<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: &Opinion<T, V>) {
        self.fuse_assign(lhs, rhs.as_ref())
    }
}

impl<'a, T, V, Idx> FuseAssign<Opinion<T, V>, OpinionRef<'a, T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: OpinionRef<'a, T, V>) {
        *lhs = self.fuse(lhs.as_ref(), rhs);
    }
}

impl<T, V, Idx> FuseAssign<Opinion<T, V>, &SimplexBase<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: &SimplexBase<T, V>) {
        *lhs = self.fuse(lhs.as_ref(), rhs);
    }
}

impl<T, V, Idx> FuseAssign<SimplexBase<T, V>, &SimplexBase<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone,
    V: Float + UlpsEq + AddAssign + DivAssign,
    Idx: Copy,
{
    fn fuse_assign(&self, lhs: &mut SimplexBase<T, V>, rhs: &SimplexBase<T, V>) {
        *lhs = self.fuse(&(*lhs), rhs);
    }
}

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

impl<'b, X, Y, Cond, U, T, V> Deduction<X, Y, &'b Cond, U> for OpinionRef<'b, T, V>
where
    T: IndexedContainer<X, Output = V>,
    U: IndexedContainer<Y, Output = V> + MBR<X, Y, T, &'b Cond, U, V>,
    Cond: IndexedContainer<X>,
    for<'a> &'a Cond::Output: Into<&'a SimplexBase<U, V>>,
    X: Copy,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum,
{
    type Output = Opinion<U, V>;

    fn deduce(self, conds: &'b Cond) -> Option<Self::Output> {
        let ay = U::marginal_base_rate(&self.base_rate, conds)?;
        Some(self.deduce_with(conds, ay))
    }

    fn deduce_with(self, conds: &'b Cond, ay: U) -> Self::Output {
        assert!(
            T::SIZE > 0 && U::SIZE > 1,
            "conds.len() > 0 and ay.len() > 1 must hold."
        );
        let cond_p = Cond::map(|x| conds[x].into().projection(&ay));
        let pyhx: U = U::from_fn(|y| Cond::keys().map(|x| self.base_rate[x] * cond_p[x][y]).sum());
        let uyhx = U::keys()
            .map(|y| {
                (pyhx[y]
                    - Cond::keys()
                        .map(|x| conds[x].into().belief[y])
                        .reduce(<V>::min)
                        .unwrap())
                    / ay[y]
            })
            .reduce(<V>::min)
            .unwrap();
        let u = uyhx
            - Cond::keys()
                .map(|x| (uyhx - conds[x].into().uncertainty) * self.b()[x])
                .sum::<V>();
        let p = self.projection();
        let b = U::from_fn(|y| T::keys().map(|x| p[x] * cond_p[x][y]).sum::<V>() - ay[y] * u);
        Opinion::<U, V>::new_unchecked(b, u, ay)
    }
}

impl<'b, X, Y, Cond, U, T, V> Deduction<X, Y, &'b Cond, U> for &'b Opinion<T, V>
where
    Cond: IndexedContainer<X>,
    for<'a> &'a Cond::Output: Into<&'a SimplexBase<U, V>>,
    T: IndexedContainer<X, Output = V>,
    U: IndexedContainer<Y, Output = V>,
    X: Copy,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq,
{
    type Output = Opinion<U, V>;
    fn deduce(self, conds: &'b Cond) -> Option<Self::Output> {
        self.as_ref().deduce(conds)
    }

    fn deduce_with(self, conds: &'b Cond, ay: U) -> Self::Output {
        self.as_ref().deduce_with(conds, ay)
    }
}

trait InverseCondition<X, Y, T, U, V>
where
    Self: Index<X, Output = SimplexBase<U, V>>,
    T: Index<X, Output = V>,
    U: Index<Y, Output = V>,
{
    type InvCond;
    fn inverse(&self, ax: &T, ay: &U) -> Self::InvCond;
}

impl<Cond, T, U, X, Y, V> InverseCondition<X, Y, T, U, V> for Cond
where
    T: IndexedContainer<X, Output = V>,
    U: IndexedContainer<Y, Output = V>,
    Cond: IndexedContainer<X, Output = SimplexBase<U, V>>,
    X: Copy,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq,
{
    type InvCond = U::Map<SimplexBase<T, V>>;
    fn inverse(&self, ax: &T, ay: &U) -> Self::InvCond {
        let p_yx: Cond::Map<U> = Cond::map(|x| self[x].projection(ay));
        let u_yx = T::from_fn(|x| self[x].max_uncertainty(ay));
        let p_xy: U::Map<T::Map<V>> = U::map(|y| {
            T::map(|x| ax[x] * p_yx[x][y] / T::keys().map(|xd| ax[xd] * p_yx[xd][y]).sum::<V>())
        });
        let irrelevance_yx = U::from_fn(|y| {
            V::one() - T::keys().map(|x| p_yx[x][y]).reduce(<V>::max).unwrap()
                + T::keys().map(|x| p_yx[x][y]).reduce(<V>::min).unwrap()
        });
        let max_u_xy: U = U::from_fn(|y| {
            T::keys()
                .map(|x| p_yx[x][y] / T::keys().map(|k| ax[k] * p_yx[k][y]).sum::<V>())
                .reduce(<V>::min)
                .unwrap()
        });
        let u_yx_sum = Cond::keys().map(|x| u_yx[x]).sum::<V>();
        let weights_yx = if u_yx_sum == V::zero() {
            T::from_fn(|_| V::zero())
        } else {
            T::from_fn(|x| u_yx[x] / u_yx_sum)
        };
        let max_u_yx = T::from_fn(|x| {
            U::keys()
                .map(|y| p_yx[x][y] / ay[y])
                .reduce(<V>::min)
                .unwrap()
        });
        let weighted_u_yx = T::from_fn(|x| {
            let u = max_u_yx[x];
            if ulps_eq!(u, V::zero()) {
                V::zero()
            } else {
                weights_yx[x] * u_yx[x] / u
            }
        });
        let wprop_u_yx: V = T::keys().map(|x| weighted_u_yx[x]).sum();
        U::map(|y| {
            let u = max_u_xy[y] * (wprop_u_yx + irrelevance_yx[y] - wprop_u_yx * irrelevance_yx[y]);
            let b = T::from_fn(|x| p_xy[y][x] - u * ax[x]);
            SimplexBase::new_unchecked(b, u)
        })
    }
}

/// The abduction operator.
pub trait Abduction<Cond, X, Y, T, U>
where
    T: Index<X>,
    U: Index<Y>,
{
    type Output;

    /// Computes the conditionally abduced opinion of `self` with a base rate vector `ax`
    /// by `conds` representing a collection of conditional opinions.
    /// If a marginal base rate cannot be computed from `conds`, return `None`.
    fn abduce(self, conds: Cond, ax: T) -> Option<Self::Output>;
    fn abduce_with(self, conds: Cond, ax: T, ay: &U) -> Self::Output;
}

impl<'a, Cond, X, Y, T, U, V> Abduction<&'a Cond, X, Y, T, U> for &'a SimplexBase<U, V>
where
    Cond: InverseCondition<X, Y, T, U, V> + IndexedContainer<X, Output = SimplexBase<U, V>>,
    Cond::InvCond: IndexedContainer<Y, Output = SimplexBase<T, V>> + 'a,
    T: IndexedContainer<X, Output = V> + 'a,
    U: IndexedContainer<Y, Output = V> + MBR<X, Y, T, &'a Cond, U, V>,
    X: Copy,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq,
{
    type Output = Opinion<T, V>;

    fn abduce(self, conds: &'a Cond, ax: T) -> Option<Self::Output> {
        let ay = U::marginal_base_rate(&ax, conds)?;
        Some(self.abduce_with(conds, ax, &ay))
    }

    fn abduce_with(self, conds: &'a Cond, ax: T, ay: &U) -> Self::Output {
        let inv_conds = InverseCondition::inverse(conds, &ax, ay);
        OpinionRef::<U, V>::from((self, ay)).deduce_with(&inv_conds, ax)
    }
}

impl<'a, Cond, X, Y, T, U, V> Abduction<&'a Cond, X, Y, T, U> for OpinionRef<'a, U, V>
where
    Cond: InverseCondition<X, Y, T, U, V> + IndexedContainer<X, Output = SimplexBase<U, V>>,
    Cond::InvCond: IndexedContainer<Y, Output = SimplexBase<T, V>> + 'a,
    T: IndexedContainer<X, Output = V> + 'a,
    U: IndexedContainer<Y, Output = V> + MBR<X, Y, T, &'a Cond, U, V>,
    X: Copy,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq,
{
    type Output = Opinion<T, V>;

    fn abduce(self, conds: &'a Cond, ax: T) -> Option<Self::Output> {
        self.simplex.abduce(conds, ax)
    }

    fn abduce_with(self, conds: &'a Cond, ax: T, ay: &U) -> Self::Output {
        self.simplex.abduce_with(conds, ax, ay)
    }
}

impl<'a, Cond, X, Y, T, U, V> Abduction<&'a Cond, X, Y, T, U> for &'a Opinion<U, V>
where
    Cond: InverseCondition<X, Y, T, U, V> + IndexedContainer<X, Output = SimplexBase<U, V>>,
    Cond::InvCond: IndexedContainer<Y, Output = SimplexBase<T, V>> + 'a,
    T: IndexedContainer<X, Output = V> + 'a,
    U: IndexedContainer<Y, Output = V> + MBR<X, Y, T, &'a Cond, U, V>,
    X: Copy,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq,
{
    type Output = Opinion<T, V>;

    fn abduce(self, conds: &'a Cond, ax: T) -> Option<Self::Output> {
        self.as_ref().abduce(conds, ax)
    }

    fn abduce_with(self, conds: &'a Cond, ax: T, ay: &U) -> Self::Output {
        self.as_ref().abduce_with(conds, ax, ay)
    }
}

#[cfg(test)]
mod tests {
    use crate::mul::{
        op::{Abduction, Deduction, Fuse, FuseAssign, FuseOp},
        Opinion1d, OpinionRef, Projection, Simplex, MBR,
    };

    macro_rules! nround {
        [$ft:ty, $n:expr] => {
            |v: $ft| (v * <$ft>::powi(10.0, $n)).round()
        };
    }

    macro_rules! nfract {
        [$ft:ty, $n:expr] => {
            |v: $ft| (v * <$ft>::powi(10.0, $n)).fract()
        };
    }

    #[test]
    fn test_fusion_dogma() {
        macro_rules! def {
            ($ft: ty) => {
                let w1 = Opinion1d::<$ft, 3>::new(
                    [0.99, 0.01, 0.0],
                    0.0,
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                );
                let w2 = Simplex::<$ft, 3>::new([0.0, 0.01, 0.99], 0.0);

                // A-CBF
                let w = FuseOp::ACm.fuse(&w1, &w2);
                assert_eq!(w.b().map(nround![$ft, 3]), [495.0, 10.0, 495.0]);
                assert_eq!(nround![$ft, 3](w.u()), 0.0);
                assert_eq!(w.base_rate.map(nround![$ft, 3]), [333.0, 333.0, 333.0]);
                // E-CBF
                let w = FuseOp::ECm.fuse(&w1, &w2);
                assert_eq!(w.b().map(nround![$ft, 3]), [485.0, 0.0, 485.0]);
                assert_eq!(nround![$ft, 3](w.u()), 30.0);
                assert_eq!(w.base_rate.map(nround![$ft, 3]), [333.0, 333.0, 333.0]);
                // ABF
                let w = FuseOp::Avg.fuse(&w1, &w2);
                assert_eq!(w.b().map(nround![$ft, 3]), [495.0, 10.0, 495.0]);
                assert_eq!(nround![$ft, 3](w.u()), 0.0);
                assert_eq!(w.base_rate.map(nround![$ft, 3]), [333.0, 333.0, 333.0]);
                // WBF
                let w = FuseOp::Wgh.fuse(&w1, &w2);
                assert_eq!(w.b().map(nround![$ft, 3]), [495.0, 10.0, 495.0]);
                assert_eq!(nround![$ft, 3](w.u()), 0.0);
                assert_eq!(w.base_rate.map(nround![$ft, 3]), [333.0, 333.0, 333.0]);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_fusion() {
        macro_rules! def {
            ($ft: ty) => {
                let w1 = Opinion1d::<$ft, 3>::new(
                    [0.98, 0.01, 0.0],
                    0.01,
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                );
                let w2 = Opinion1d::<$ft, 3>::new(
                    [0.0, 0.01, 0.90],
                    0.09,
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                );

                // A-CBF
                let w = FuseOp::ACm.fuse(&w1, &w2);
                assert_eq!(w.b().map(nround![$ft, 3]), [890.0, 10.0, 91.0]);
                assert_eq!(nround![$ft, 3](w.u()), 9.0);
                assert_eq!(w.base_rate.map(nround![$ft, 3]), [333.0, 333.0, 333.0]);
                // E-CBF
                let w = FuseOp::ECm.fuse(&w1, &w2);
                assert_eq!(w.b().map(nround![$ft, 3]), [880.0, 0.0, 81.0]);
                assert_eq!(nround![$ft, 3](w.u()), 39.0);
                assert_eq!(w.base_rate.map(nround![$ft, 3]), [333.0, 333.0, 333.0]);
                // ABF
                let w = FuseOp::Avg.fuse(&w1, &w2);
                assert_eq!(w.b().map(nround![$ft, 3]), [882.0, 10.0, 90.0]);
                assert_eq!(nround![$ft, 3](w.u()), 18.0);
                assert_eq!(w.base_rate.map(nround![$ft, 3]), [333.0, 333.0, 333.0]);
                // WBF
                let w = FuseOp::Wgh.fuse(&w1, &w2);
                assert_eq!(w.b().map(nround![$ft, 3]), [889.0, 10.0, 83.0]);
                assert_eq!(
                    nround![$ft, 3](w.u()),
                    18.0 - w.b().map(nfract![$ft, 3]).iter().sum::<$ft>().round()
                );
                assert_eq!(w.base_rate.map(nround![$ft, 3]), [333.0, 333.0, 333.0]);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_fusion_assign() {
        macro_rules! def {
            ($ft: ty) => {
                let mut w = Opinion1d::<$ft, 2>::new([0.5, 0.25], 0.25, [0.5, 0.5]);
                let u = Simplex::<$ft, 2>::new([0.125, 0.75], 0.125);
                let ops = [FuseOp::ACm, FuseOp::ECm, FuseOp::Avg, FuseOp::Wgh];
                for op in ops {
                    let w2 = op.fuse(w.as_ref(), OpinionRef::from((&u, &w.base_rate)));
                    op.fuse_assign(&mut w, &u);
                    assert!(w == w2);
                }
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_fusion_assign_simplex() {
        macro_rules! def {
            ($ft: ty) => {
                let mut w = Simplex::<$ft, 2>::new([0.5, 0.25], 0.25);
                let u = Simplex::<$ft, 2>::new([0.125, 0.75], 0.125);
                let ops = [FuseOp::ACm, FuseOp::Avg, FuseOp::Wgh];
                for op in ops {
                    let w2 = op.fuse(&w, &u);
                    op.fuse_assign(&mut w, &u);
                    assert!(w == w2);
                }
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_deduction() {
        macro_rules! def {
            ($ft: ty) => {
                let wx = Opinion1d::<$ft, 2>::new([0.9, 0.0], 0.1, [0.1, 0.9]);
                let wxy = [
                    Simplex::<$ft, 3>::new([0.0, 0.8, 0.1], 0.1),
                    Simplex::<$ft, 3>::new([0.7, 0.0, 0.1], 0.2),
                ];
                let wy = wx.as_ref().deduce(&wxy).unwrap();
                // base rate
                assert_eq!(wy.base_rate.map(nround![$ft, 3]), [778.0, 99.0, 123.0]);
                // projection
                let p = wy.projection();
                assert_eq!(p.map(nround![$ft, 3]), [148.0, 739.0, 113.0]);
                // belief
                assert_eq!(wy.b().map(nround![$ft, 3]), [63.0, 728.0, 100.0]);
                // uncertainty
                assert_eq!(nround![$ft, 3](wy.u()), 109.0);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_abduction() {
        let dcond = [
            Simplex::new([0.50, 0.25, 0.25], 0.0),
            Simplex::new([0.00, 0.50, 0.50], 0.0),
            Simplex::new([0.00, 0.25, 0.75], 0.0),
        ];
        let ax = [0.70, 0.20, 0.10];
        let wy = Opinion1d::new([0.00, 0.43, 0.00], 0.57, [0.0, 0.0, 1.0]);
        let m_ay = <[f32; 3]>::marginal_base_rate(&ax, &dcond).unwrap();
        println!("{m_ay:?}");

        let wx = wy.abduce(&dcond, ax).unwrap();
        assert_eq!(wx.b().map(nround![f32, 2]), [0.0, 7.0, 0.0]);
        assert_eq!(nround![f32, 2](wx.u()), 93.0);
        assert_eq!(wx.projection().map(nround![f32, 2]), [65.0, 26.0, 9.0]);
    }
}
