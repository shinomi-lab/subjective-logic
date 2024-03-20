pub mod labeled;
pub mod non_labeled;

use approx::{ulps_eq, UlpsEq};
use core::fmt;
use num_traits::Float;
use std::{
    fmt::Debug,
    iter::Sum,
    ops::{AddAssign, DivAssign, Index},
};

use crate::{
    approx_ext::{is_one, is_zero},
    errors::{check_is_one, check_unit_interval, InvalidValueError},
    ops::{
        Abduction, Deduction, Discount, Fuse, FuseAssign, FuseOp, IndexedContainer, MaxUncertainty,
        Product2, Product3, Projection,
    },
};

#[derive(Default, Clone, PartialEq)]
pub struct Simplex<T, V> {
    pub belief: T,
    pub uncertainty: V,
}

impl<T: Debug, V: Debug> Debug for Simplex<T, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b={:?}, u={:?}", self.belief, self.uncertainty)
    }
}

impl<T, V> Simplex<T, V> {
    #[inline]
    pub fn vacuous() -> Self
    where
        T: Default, // IndexedContainer<Idx, Output = V>,
        V: Float + Default,
    {
        Self {
            belief: T::default(), // T::from_fn(|_| V::zero()),
            uncertainty: V::one(),
        }
    }

    #[inline]
    pub fn is_vacuous(&self) -> bool
    where
        V: Float + UlpsEq,
    {
        is_one(self.uncertainty)
    }

    #[inline]
    pub fn is_dogmatic(&self) -> bool
    where
        V: Float + UlpsEq,
    {
        is_zero(self.uncertainty)
    }

    pub fn new_unchecked(b: T, u: V) -> Self {
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
    pub fn u(&self) -> &V {
        &self.uncertainty
    }

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
    pub fn try_new<Idx>(b: T, u: V) -> Result<Self, InvalidValueError>
    where
        V: UlpsEq + Float + AddAssign,
        Idx: fmt::Debug + Clone,
        T: IndexedContainer<Idx, Output = V>,
    {
        check_simplex(&b, u)?;
        Ok(Self::new_unchecked(b, u))
    }

    /// Creates a new simplex of a multinomial opinion (i.e. excluding a base rate) from parameters,
    /// which reqiure the same conditions as `try_new`.
    ///
    /// # Panics
    /// Panics if even pameter does not satisfy the conditions.
    pub fn new<Idx>(b: T, u: V) -> Self
    where
        V: UlpsEq + Float + AddAssign,
        Idx: fmt::Debug + Clone,
        T: IndexedContainer<Idx, Output = V>,
    {
        Self::try_new(b, u).unwrap()
    }

    pub fn projection<'a, Idx: Copy>(&'a self, a: &'a T) -> T
    where
        T: IndexedContainer<Idx, Output = V>,
        V: Float + AddAssign + DivAssign,
    {
        OpinionRef::from((self, a)).projection()
    }
}

/// The generlized structure of a multinomial opinion.
#[derive(Default, Clone, PartialEq)]
pub struct OpinionBase<S, T> {
    pub simplex: S,
    pub base_rate: T,
}

impl<S, T> OpinionBase<S, T> {
    pub fn as_ref(&self) -> OpinionBase<&S, &T> {
        OpinionBase {
            simplex: &self.simplex,
            base_rate: &self.base_rate,
        }
    }
}

impl<S: Debug, T: Debug> Debug for OpinionBase<S, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}, a={:?}", self.simplex, self.base_rate)
    }
}

pub type Opinion<T, U> = OpinionBase<Simplex<T, U>, T>;
pub type OpinionRef<'a, T, U> = OpinionBase<&'a Simplex<T, U>, &'a T>;

impl<T, V> Opinion<T, V> {
    #[inline]
    pub fn is_vacuous(&self) -> bool
    where
        V: Float + UlpsEq,
    {
        self.simplex.is_vacuous()
    }

    #[inline]
    pub fn is_dogmatic(&self) -> bool
    where
        V: Float + UlpsEq,
    {
        self.simplex.is_dogmatic()
    }

    #[inline]
    pub fn vacuous_with(base_rate: T) -> Self
    where
        T: Default, // IndexedContainer<Idx, Output = V>,
        V: Float + Default,
    {
        OpinionBase {
            simplex: Simplex::<T, V>::vacuous(),
            base_rate,
        }
    }

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
    pub fn try_new<Idx>(b: T, u: V, a: T) -> Result<Self, InvalidValueError>
    where
        V: UlpsEq + Float + AddAssign,
        Idx: fmt::Debug + Clone,
        T: IndexedContainer<Idx, Output = V>,
    {
        check_simplex(&b, u)?;
        check_base_rate(&a)?;
        Ok(Self::new_unchecked(b, u, a))
    }

    /// Creates a new multinomial opinion from parameters which reqiure the same conditions as `try_new`.
    ///
    /// # Panics
    /// Panics if even pameter does not satisfy the conditions.
    pub fn new<Idx>(b: T, u: V, a: T) -> Self
    where
        V: UlpsEq + Float + AddAssign,
        Idx: fmt::Debug + Clone,
        T: IndexedContainer<Idx, Output = V>,
    {
        Self::try_new(b, u, a).unwrap()
    }

    fn new_unchecked(b: T, u: V, a: T) -> Self {
        Self {
            simplex: Simplex::new_unchecked(b, u),
            base_rate: a,
        }
    }

    #[inline]
    pub fn b(&self) -> &T {
        &self.simplex.belief
    }

    #[inline]
    pub fn u(&self) -> V
    where
        V: Copy,
    {
        self.simplex.uncertainty
    }
}

impl<T, V> OpinionRef<'_, T, V> {
    #[inline]
    pub fn b(&self) -> &T {
        &self.simplex.belief
    }

    #[inline]
    pub fn u(&self) -> V
    where
        V: Copy,
    {
        self.simplex.uncertainty
    }

    #[inline]
    pub fn into_opinion(&self) -> Opinion<T, V>
    where
        T: Clone,
        V: Copy,
    {
        Opinion::new_unchecked(
            (self.simplex.belief).clone(),
            self.simplex.uncertainty,
            (*self.base_rate).clone(),
        )
    }

    #[inline]
    pub fn is_vacuous(&self) -> bool
    where
        V: Float + UlpsEq,
    {
        self.simplex.is_vacuous()
    }

    #[inline]
    pub fn is_dogmatic(&self) -> bool
    where
        V: Float + UlpsEq,
    {
        self.simplex.is_dogmatic()
    }
}

impl<S, T> From<(S, T)> for OpinionBase<S, T> {
    fn from((simplex, base_rate): (S, T)) -> Self {
        Self { simplex, base_rate }
    }
}

impl<'a, T, U> From<&'a Opinion<T, U>> for OpinionRef<'a, T, U> {
    fn from(value: &'a Opinion<T, U>) -> Self {
        value.as_ref()
    }
}

impl<'a, Idx, T, V: Float + AddAssign + DivAssign> Projection<Idx, T> for OpinionRef<'a, T, V>
where
    T: IndexedContainer<Idx, Output = V>,
    Idx: Copy,
{
    fn projection(&self) -> T {
        let mut s = V::zero();
        let mut a = T::from_fn(|idx| {
            let p = self.b()[idx] + self.base_rate[idx] * self.u();
            s += p;
            p
        });
        for idx in T::keys() {
            a[idx] /= s;
        }
        a
    }
}

impl<Idx, T, V> Projection<Idx, T> for Opinion<T, V>
where
    T: IndexedContainer<Idx, Output = V>,
    Idx: Copy,
    V: Float + AddAssign + DivAssign,
{
    fn projection(&self) -> T {
        self.as_ref().projection()
    }
}

impl<'a, T, V, Idx> MaxUncertainty<Idx, V, T> for Simplex<T, V>
where
    T: IndexedContainer<Idx, Output = V>,
    V: Float + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = Simplex<T, V>;

    fn max_uncertainty(&self, a: &T) -> V {
        let p = self.projection(a);
        T::keys().map(|i| p[i] / a[i]).reduce(<V>::min).unwrap()
    }

    fn uncertainty_maximized(&self, a: &T) -> Self::Output {
        let p = self.projection(a);
        let u_max = self.max_uncertainty(a);
        let b_max = T::from_fn(|i| p[i] - a[i] * u_max);
        Simplex::new_unchecked(b_max, u_max)
    }
}

impl<T, V> Discount<T, V> for Simplex<T, V>
where
    T: FromIterator<V> + Default,
    for<'a> &'a T: IntoIterator<Item = &'a V>,
    V: Float + UlpsEq + Default,
{
    type Output = Simplex<T, V>;
    fn discount(&self, t: V) -> Self::Output {
        if self.is_vacuous() {
            Self::Output::vacuous()
        } else {
            Simplex {
                belief: T::from_iter(self.b().into_iter().map(|&b| b * t)),
                uncertainty: V::one() - t * (V::one() - *self.u()),
            }
        }
    }
}

impl<'b, T, V> Discount<T, V> for OpinionRef<'b, T, V>
where
    T: FromIterator<V> + Default + Clone,
    for<'a> &'a T: IntoIterator<Item = &'a V>,
    V: Float + UlpsEq + Default,
{
    type Output = Opinion<T, V>;
    fn discount(&self, t: V) -> Self::Output {
        (self.simplex.discount(t), (*self.base_rate).clone()).into()
    }
}

impl<T, V> Discount<T, V> for Opinion<T, V>
where
    T: FromIterator<V> + Default + Clone,
    for<'a> &'a T: IntoIterator<Item = &'a V>,
    V: Float + UlpsEq + Default,
{
    type Output = Opinion<T, V>;
    fn discount(&self, t: V) -> Self::Output {
        self.as_ref().discount(t)
    }
}

fn check_simplex<Idx, T, V>(b: &T, u: V) -> Result<(), InvalidValueError>
where
    V: UlpsEq + Float + AddAssign,
    Idx: fmt::Debug + Clone,
    T: IndexedContainer<Idx, Output = V>,
{
    let mut sum_b = V::zero();
    for i in T::keys() {
        let bi = b[i.clone()];
        check_unit_interval(bi, format!("b[{i:?}]"))?;
        sum_b += bi;
    }

    check_unit_interval(u, "u")?;
    check_is_one(sum_b + u, "sum(b) + u")?;

    Ok(())
}

fn check_base_rate<Idx, T, V>(a: &T) -> Result<(), InvalidValueError>
where
    V: UlpsEq + Float + AddAssign,
    T: IndexedContainer<Idx, Output = V>,
    Idx: fmt::Debug + Clone,
{
    let mut sum_a = V::zero();
    for i in T::keys() {
        let ai = a[i.clone()];
        check_unit_interval(ai, format!("a[{i:?}]"))?;
        sum_a += ai;
    }

    check_is_one(sum_a, "sum(a)")?;

    Ok(())
}

fn compute_simlex<T, V, Idx>(op: &FuseOp, lhs: &Simplex<T, V>, rhs: &Simplex<T, V>) -> Simplex<T, V>
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + fmt::Debug + AddAssign + Default,
    Idx: Copy + fmt::Debug,
{
    if lhs.is_dogmatic() && rhs.is_dogmatic() {
        Simplex::new(
            T::from_fn(|i| (lhs.b()[i] + rhs.b()[i]) / (V::one() + V::one())),
            V::zero(),
        )
    } else {
        match op {
            FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() && rhs.is_vacuous() => Simplex::vacuous(),
            FuseOp::ACm | FuseOp::ECm if lhs.is_vacuous() || rhs.is_dogmatic() => rhs.clone(),
            FuseOp::ACm | FuseOp::ECm if rhs.is_vacuous() || lhs.is_dogmatic() => lhs.clone(),
            FuseOp::ACm | FuseOp::ECm => {
                let lhs_u = *lhs.u();
                let rhs_u = *rhs.u();
                let temp = lhs_u + rhs_u - lhs_u * rhs_u;
                let b = T::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                let u = lhs_u * rhs_u / temp;
                Simplex::new(b, u)
            }
            FuseOp::Avg if lhs.is_dogmatic() => lhs.clone(),
            FuseOp::Avg if rhs.is_dogmatic() => rhs.clone(),
            FuseOp::Avg => {
                let lhs_u = *lhs.u();
                let rhs_u = *rhs.u();
                let temp = lhs_u + rhs_u;
                let b = T::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                let u = (V::one() + V::one()) * lhs_u * rhs_u / temp;
                Simplex::new(b, u)
            }
            FuseOp::Wgh if lhs.is_vacuous() && rhs.is_vacuous() => Simplex::vacuous(),
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
                Simplex::new(b, u)
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
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
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

impl<T, V, Idx> Fuse<&Opinion<T, V>, &Simplex<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: &Opinion<T, V>, rhs: &Simplex<T, V>) -> Self::Output {
        self.fuse(lhs.as_ref(), rhs)
    }
}

impl<T, V, Idx> Fuse<&Opinion<T, V>, &Opinion<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: &Opinion<T, V>, rhs: &Opinion<T, V>) -> Self::Output {
        self.fuse(lhs.as_ref(), rhs.as_ref())
    }
}

impl<'a, T, V, Idx> Fuse<OpinionRef<'a, T, V>, &'a Simplex<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: OpinionRef<'a, T, V>, rhs: &'a Simplex<T, V>) -> Self::Output {
        self.fuse(lhs.clone(), OpinionRef::from((rhs, lhs.base_rate)))
    }
}

impl<T, V, Idx> Fuse<&Simplex<T, V>, &Simplex<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
{
    type Output = Simplex<T, V>;

    fn fuse(&self, lhs: &Simplex<T, V>, rhs: &Simplex<T, V>) -> Self::Output {
        if matches!(self, FuseOp::ECm) {
            panic!("Epistemic fusion cannot be used with simplexes because there is no base rate.");
        }
        compute_simlex(self, lhs, rhs)
    }
}

impl<T, V, Idx> FuseAssign<Opinion<T, V>, &Opinion<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: &Opinion<T, V>) {
        self.fuse_assign(lhs, rhs.as_ref())
    }
}

impl<'a, T, V, Idx> FuseAssign<Opinion<T, V>, OpinionRef<'a, T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: OpinionRef<'a, T, V>) {
        *lhs = self.fuse(lhs.as_ref(), rhs);
    }
}

impl<T, V, Idx> FuseAssign<Opinion<T, V>, &Simplex<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: &Simplex<T, V>) {
        *lhs = self.fuse(lhs.as_ref(), rhs);
    }
}

impl<T, V, Idx> FuseAssign<Simplex<T, V>, &Simplex<T, V>, Idx> for FuseOp
where
    T: IndexedContainer<Idx, Output = V> + Clone + Default,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Default,
    Idx: Copy + fmt::Debug,
{
    fn fuse_assign(&self, lhs: &mut Simplex<T, V>, rhs: &Simplex<T, V>) {
        *lhs = self.fuse(&(*lhs), rhs);
    }
}

// trait MBR<X, Y, T, Cond, U, V> {
//     fn marginal_base_rate(ax: &T, conds: Cond) -> Option<U>;
// }

// impl<'a, X, Y, T, Cond, U, V> MBR<X, Y, T, &'a Cond, U, V> for U
// where
//     T: Index<X, Output = V>,
//     Cond: IndexedContainer<X>,
//     for<'b> &'b Cond::Output: Into<&'b Simplex<U, V>>,
//     U: IndexedContainer<Y, Output = V>,
//     X: Copy,
//     Y: Copy,
//     V: Float + Sum + UlpsEq + AddAssign + DivAssign,
// {
//     fn marginal_base_rate(ax: &T, conds: &'a Cond) -> Option<U> {
//         if Cond::keys().all(|x| conds[x].into().is_vacuous()) {
//             return None;
//         }
//         let mut sum_a = V::zero();
//         let mut ay = U::from_fn(|y| {
//             let a = Cond::keys()
//                 .map(|x| ax[x] * conds[x].into().belief[y])
//                 .sum::<V>();
//             sum_a += a;
//             a
//         });
//         for y in U::keys() {
//             ay[y] /= sum_a;
//         }
//         Some(ay)
//     }
// }

fn mbr<'a, X, Y, T, Cond, U, V>(ax: &T, conds: &'a Cond) -> Option<U>
where
    T: Index<X, Output = V>,
    Cond: IndexedContainer<X>,
    for<'b> &'b Cond::Output: Into<&'b Simplex<U, V>>,
    U: IndexedContainer<Y, Output = V>,
    X: Copy,
    Y: Copy,
    V: Float + Sum + UlpsEq + AddAssign + DivAssign,
{
    if Cond::keys().all(|x| conds[x].into().is_vacuous()) {
        return None;
    }
    let mut sum_a = V::zero();
    let mut ay = U::from_fn(|y| {
        let a = Cond::keys()
            .map(|x| ax[x] * conds[x].into().belief[y])
            .sum::<V>();
        sum_a += a;
        a
    });
    for y in U::keys() {
        ay[y] /= sum_a;
    }
    Some(ay)
}

impl<'b, X, Y, Cond, U, T, V> Deduction<X, Y, &'b Cond, U> for OpinionRef<'b, T, V>
where
    T: IndexedContainer<X, Output = V>,
    U: IndexedContainer<Y, Output = V>, // + MBR<X, Y, T, &'b Cond, U, V>,
    Cond: IndexedContainer<X>,
    for<'a> &'a Cond::Output: Into<&'a Simplex<U, V>>,
    X: Copy,
    Y: Copy + fmt::Debug,
    V: Float + AddAssign + DivAssign + Sum + fmt::Debug + UlpsEq,
{
    type Output = Opinion<U, V>;

    fn deduce(self, conds: &'b Cond) -> Option<Self::Output> {
        let ay = mbr::<X, Y, T, Cond, U, V>(&self.base_rate, conds)?;
        Some(self.deduce_with(conds, ay))
    }

    fn deduce_with(self, conds: &'b Cond, ay: U) -> Self::Output {
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
        Opinion::<U, V>::new(b, u, ay)
    }
}

impl<'b, X, Y, Cond, U, T, V> Deduction<X, Y, &'b Cond, U> for &'b Opinion<T, V>
where
    Cond: IndexedContainer<X>,
    for<'a> &'a Cond::Output: Into<&'a Simplex<U, V>>,
    T: IndexedContainer<X, Output = V>,
    U: IndexedContainer<Y, Output = V>,
    X: Copy,
    Y: Copy + fmt::Debug,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq + fmt::Debug,
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
    Self: Index<X, Output = Simplex<U, V>>,
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
    Cond: IndexedContainer<X, Output = Simplex<U, V>>,
    X: Copy + fmt::Debug,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq + fmt::Debug,
{
    type InvCond = U::Map<Simplex<T, V>>;
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
            Simplex::new(b, u)
        })
    }
}

impl<'a, Cond, X, Y, T, U, V> Abduction<&'a Cond, X, Y, T, U> for &'a Simplex<U, V>
where
    Cond: InverseCondition<X, Y, T, U, V> + IndexedContainer<X, Output = Simplex<U, V>>,
    Cond::InvCond: IndexedContainer<Y, Output = Simplex<T, V>> + 'a,
    T: IndexedContainer<X, Output = V> + 'a,
    U: IndexedContainer<Y, Output = V>, // + MBR<X, Y, T, &'a Cond, U, V>,
    X: Copy + fmt::Debug,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq + fmt::Debug,
{
    type Output = Opinion<T, V>;

    fn abduce(self, conds: &'a Cond, ax: T) -> Option<Self::Output> {
        let ay = mbr::<X, Y, T, Cond, U, V>(&ax, conds)?;
        Some(self.abduce_with(conds, ax, &ay))
    }

    fn abduce_with(self, conds: &'a Cond, ax: T, ay: &U) -> Self::Output {
        let inv_conds = InverseCondition::inverse(conds, &ax, ay);
        OpinionRef::<U, V>::from((self, ay)).deduce_with(&inv_conds, ax)
    }
}

impl<'a, Cond, X, Y, T, U, V> Abduction<&'a Cond, X, Y, T, U> for OpinionRef<'a, U, V>
where
    Cond: InverseCondition<X, Y, T, U, V> + IndexedContainer<X, Output = Simplex<U, V>>,
    Cond::InvCond: IndexedContainer<Y, Output = Simplex<T, V>> + 'a,
    T: IndexedContainer<X, Output = V> + 'a,
    U: IndexedContainer<Y, Output = V>, // + MBR<X, Y, T, &'a Cond, U, V>,
    X: Copy + fmt::Debug,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq + fmt::Debug,
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
    Cond: InverseCondition<X, Y, T, U, V> + IndexedContainer<X, Output = Simplex<U, V>>,
    Cond::InvCond: IndexedContainer<Y, Output = Simplex<T, V>> + 'a,
    T: IndexedContainer<X, Output = V> + 'a,
    U: IndexedContainer<Y, Output = V>, // + MBR<X, Y, T, &'a Cond, U, V>,
    X: Copy + fmt::Debug,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq + fmt::Debug,
{
    type Output = Opinion<T, V>;

    fn abduce(self, conds: &'a Cond, ax: T) -> Option<Self::Output> {
        self.as_ref().abduce(conds, ax)
    }

    fn abduce_with(self, conds: &'a Cond, ax: T, ay: &U) -> Self::Output {
        self.as_ref().abduce_with(conds, ax, ay)
    }
}

impl<'a, X, Y, XY, V> Product2<&'a Opinion<X, V>, &'a Opinion<Y, V>> for Opinion<XY, V>
where
    Opinion<XY, V>: Product2<OpinionRef<'a, X, V>, OpinionRef<'a, Y, V>>,
{
    fn product2(w0: &'a Opinion<X, V>, w1: &'a Opinion<Y, V>) -> Self {
        Product2::product2(w0.as_ref(), w1.as_ref())
    }
}

impl<'a, X, Y, Z, XYZ, V> Product3<&'a Opinion<X, V>, &'a Opinion<Y, V>, &'a Opinion<Z, V>>
    for Opinion<XYZ, V>
where
    Opinion<XYZ, V>: Product3<OpinionRef<'a, X, V>, OpinionRef<'a, Y, V>, OpinionRef<'a, Z, V>>,
{
    fn product3(w0: &'a Opinion<X, V>, w1: &'a Opinion<Y, V>, w2: &'a Opinion<Z, V>) -> Self {
        Product3::product3(w0.as_ref(), w1.as_ref(), w2.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use approx::{assert_ulps_eq, ulps_eq};

    use super::{check_base_rate, mbr, Opinion, OpinionRef, Projection, Simplex};
    use crate::{
        marr1, marr2,
        ops::{Abduction, Deduction, Discount, Fuse, FuseAssign, FuseOp, MaxUncertainty, Product2},
    };

    #[test]
    fn test_discount() {
        macro_rules! def {
            ($ft: ty) => {
                let w = Opinion::<_, $ft>::new(marr1![0.2, 0.2], 0.6, marr1![0.5, 0.5]);
                let w2 = w.discount(0.5);
                assert!(ulps_eq!(w2.b()[[0]], 0.1));
                assert!(ulps_eq!(w2.b()[[1]], 0.1));
                assert!(ulps_eq!(w2.u(), 1.0 - 0.2));
                assert!(ulps_eq!(w2.b()[[0]] + w2.b()[[1]] + w2.u(), 1.0));
                assert_eq!(w2.base_rate, w.base_rate);
            };
        }

        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_projection() {
        macro_rules! def {
            ($ft: ty) => {
                let w = Opinion::new([0.2, 0.2], 0.6, [1.0 / 3.0, 2.0 / 3.0]);
                let q: [$ft; 2] = array::from_fn(|i| w.b()[i] + w.u() * w.base_rate[i]);
                let p = w.projection();
                println!("{:?}", p);
                println!("{:?}", q);
                assert_ulps_eq!(p[0] + p[1], 1.0);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_default() {
        macro_rules! def {
            ($ft: ty) => {
                let mut w = Opinion::<_, $ft>::default();
                assert_eq!(w.simplex.belief, [0.0; 2]);
                assert_eq!(w.simplex.uncertainty, 0.0);
                assert_eq!(w.base_rate, [0.0; 2]);
                let b = [1.0, 0.0];
                w.simplex.belief = b;
                w.base_rate = b;
                assert_eq!(w.b(), &b);
                assert_eq!(&w.base_rate, &b);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_max_u() {
        let w = Simplex::new([0.0, 0.0], 1.0);
        assert_eq!(w.max_uncertainty(&[0.0, 1.0]), 1.0);
    }

    #[test]
    fn test_float_size() {
        //  [0.744088550149486, 0.2483314796475328] u=0.007579970202981336
        //  [0.578534230384464, 0.413885799412554y] u=0.007579970202981336
        //  [0.578534230384464, 0.413885799412554y] u=0.007579970202981336
        let conds = marr2![
            [
                Simplex::try_from(([0.95f64, 0.00], 0.05)).unwrap(),
                Simplex::try_from(([0.7440885, 0.24833153], 0.00757997)).unwrap(),
            ],
            [
                // Simplex::new_unchecked([0.5785341, 0.41388586], 0.00757997),
                // Simplex::new_unchecked([0.5785342, 0.41388586], 0.00757997),
                Simplex::try_from(([0.5785342, 0.41388586], 0.00757994)).unwrap(),
                Simplex::try_from(([0.5785342, 0.41388586], 0.00757994)).unwrap(),
            ]
        ];
        let w0 = Opinion::vacuous_with([0.4995, 0.5005]);
        let w1 = Opinion::vacuous_with([0.000001, 0.999999]);
        let w1d = Opinion::new([0.0, 0.95], 0.05, [0.000001, 0.999999]);
        let w = Opinion::product2(&w0, &w1).deduce(&conds);
        let wd = Opinion::product2(&w0, &w1d).deduce(&conds);
        println!("{w:?}");
        println!("{wd:?}");
    }

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
                let w1 = Opinion::new([0.99, 0.01, 0.0], 0.0, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
                let w2 = Simplex::new([0.0, 0.01, 0.99], 0.0);

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
                let w1 = Opinion::<_, $ft>::new(
                    [0.98, 0.01, 0.0],
                    0.01,
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                );
                let w2 = Opinion::<_, $ft>::new(
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
                let mut w = Opinion::<_, $ft>::new([0.5, 0.25], 0.25, [0.5, 0.5]);
                let u = Simplex::<_, $ft>::new([0.125, 0.75], 0.125);
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
                let mut w = Simplex::<_, $ft>::new([0.5, 0.25], 0.25);
                let u = Simplex::<_, $ft>::new([0.125, 0.75], 0.125);
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
    fn test_mbr() {
        macro_rules! def {
            ($ft: ty) => {
                let cond1 = [Simplex::new([0.0, 0.0], 1.0), Simplex::new([0.0, 0.0], 1.0)];
                let cond2 = [
                    Simplex::new([0.0, 0.01], 0.99),
                    Simplex::new([0.0, 0.0], 1.0),
                ];
                let ax = [0.99, 0.01];

                let ay1 = mbr(&ax, &cond1);
                assert!(ay1.is_none());

                let ay2 = mbr(&ax, &cond2).unwrap();
                assert!(check_base_rate(&ay2).is_ok())
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_deduction() {
        macro_rules! def {
            ($ft: ty) => {
                let wx = Opinion::<_, $ft>::new([0.9, 0.0], 0.1, [0.1, 0.9]);
                let wxy = [
                    Simplex::<_, $ft>::new([0.0, 0.8, 0.1], 0.1),
                    Simplex::<_, $ft>::new([0.7, 0.0, 0.1], 0.2),
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
        let conds = [
            Simplex::new_unchecked([0.25, 0.04, 0.00], 0.71),
            Simplex::new_unchecked([0.00, 0.50, 0.50], 0.00),
            Simplex::new_unchecked([0.00, 0.25, 0.75], 0.00),
        ];
        let ax = [0.70, 0.20, 0.10];
        let wy = Opinion::new([0.00, 0.43, 0.00], 0.57, [0.5, 0.5, 0.0]);
        let wx = wy.abduce(&conds, ax).unwrap();
        let m_ay = mbr(&ax, &conds).unwrap();
        println!("{:?}", wx);
        println!("{:?}", m_ay);
    }

    #[test]
    fn test_abduction2() {
        let ax = [0.01, 0.495, 0.495];
        let conds_ox = [
            Simplex::new_unchecked([0.5, 0.0], 0.5),
            Simplex::new_unchecked([0.5, 0.0], 0.5),
            Simplex::new_unchecked([0.01, 0.01], 0.98),
        ];
        let mw_o = Opinion::new([0.0, 0.0], 1.0, [0.5, 0.5]);
        let mw_x = mw_o.abduce(&conds_ox, ax).unwrap();
        println!("{:?}", mw_x);
    }

    #[test]
    fn test_abduction3() {
        let dcond = [
            Simplex::new([0.50, 0.25, 0.25], 0.0),
            Simplex::new([0.00, 0.50, 0.50], 0.0),
            Simplex::new([0.00, 0.25, 0.75], 0.0),
        ];
        let ax = [0.70, 0.20, 0.10];
        let wy = Opinion::new([0.00, 0.43, 0.00], 0.57, [0.0, 0.0, 1.0]);
        let m_ay = mbr(&ax, &dcond).unwrap();
        println!("{m_ay:?}");

        let wx = wy.abduce(&dcond, ax).unwrap();
        assert_eq!(wx.b().map(nround![f32, 2]), [0.0, 7.0, 0.0]);
        assert_eq!(nround![f32, 2](wx.u()), 93.0);
        assert_eq!(wx.projection().map(nround![f32, 2]), [65.0, 26.0, 9.0]);
    }
}
