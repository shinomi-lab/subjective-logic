pub mod labeled;
pub mod non_labeled;

use approx::{ulps_eq, UlpsEq};
use core::fmt;
use num_traits::Float;
use std::{
    fmt::Debug,
    iter::Sum,
    ops::{AddAssign, DivAssign, Index, IndexMut},
};

use crate::{
    approx_ext::{is_one, is_zero},
    errors::{check_is_one, check_unit_interval, InvalidValueError},
    iter::{Container, ContainerMap, FromFn},
    ops::{
        Abduction, Deduction, Discount, Fuse, FuseAssign, FuseOp, Indexes, MaxUncertainty,
        Product2, Product3, Projection, Zeros,
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
        T: Zeros,
        V: Float,
    {
        Self {
            belief: T::zeros(),
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
        T: Container<Idx, Output = V>,
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
        T: Container<Idx, Output = V>,
    {
        Self::try_new(b, u).unwrap()
    }

    pub fn projection<'a, Idx>(&'a self, a: &'a T) -> T
    where
        T: Container<Idx, Output = V> + FromFn<Idx, V> + IndexMut<Idx>,
        V: Float + AddAssign + DivAssign,
        Idx: Copy,
    {
        OpinionRef::from((self, a)).projection()
    }

    fn normalized<Idx>(mut b: T, mut u: V) -> Self
    where
        T: Container<Idx, Output = V> + IndexMut<Idx>,
        V: Float + Sum + DivAssign + AddAssign + UlpsEq,
        Idx: Clone + fmt::Debug,
    {
        let s = T::indexes().map(|i| b[i]).sum::<V>() + u;
        for i in T::indexes() {
            b[i] /= s;
        }
        u /= s;
        Self::new_unchecked(b, u)
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
        T: Zeros,
        V: Float,
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
        T: Container<Idx, Output = V>,
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
        T: Container<Idx, Output = V>,
    {
        Self::try_new(b, u, a).unwrap()
    }

    pub fn normalized<Idx>(b: T, u: V, mut a: T) -> Self
    where
        V: UlpsEq + Float + AddAssign + DivAssign,
        Idx: fmt::Debug + Clone + Copy,
        T: Container<Idx, Output = V> + IndexMut<Idx, Output = V>,
    {
        normalize_prob_dist(&mut a);
        Self {
            simplex: Simplex::new_unchecked(b, u),
            base_rate: a,
        }
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

fn normalize_prob_dist<'a, Idx, T, V>(p: &'a mut T)
where
    Idx: Copy,
    T: Indexes<Idx> + IndexMut<Idx, Output = V>,
    V: Float + AddAssign + DivAssign,
{
    let mut s = V::zero();
    for i in T::indexes() {
        s += p[i];
    }
    for i in T::indexes() {
        p[i] /= s;
    }
}

impl<'a, Idx, T, V> Projection<Idx, T> for OpinionRef<'a, T, V>
where
    T: FromFn<Idx, V> + Indexes<Idx> + Index<Idx, Output = V> + IndexMut<Idx>,
    Idx: Copy,
    V: Float + AddAssign + DivAssign,
{
    fn projection(&self) -> T {
        let mut a = T::from_fn(|idx| {
            let p = self.b()[idx.clone()] + self.base_rate[idx] * self.u();
            p
        });
        normalize_prob_dist(&mut a);
        a
    }
}

impl<Idx, T, V> Projection<Idx, T> for Opinion<T, V>
where
    T: Container<Idx, Output = V> + FromFn<Idx, V> + IndexMut<Idx>,
    Idx: Copy,
    V: Float + AddAssign + DivAssign,
{
    fn projection(&self) -> T {
        self.as_ref().projection()
    }
}

impl<'a, T, V, Idx> MaxUncertainty<Idx, V, T> for Simplex<T, V>
where
    T: Container<Idx, Output = V> + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + AddAssign + DivAssign,
    Idx: Copy,
{
    type Output = Simplex<T, V>;

    fn max_uncertainty(&self, a: &T) -> V {
        let p = self.projection(a);
        T::indexes().map(|i| p[i] / a[i]).reduce(<V>::min).unwrap()
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
    T: FromIterator<V> + Zeros,
    for<'a> &'a T: IntoIterator<Item = &'a V>,
    V: Float + UlpsEq,
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
    T: FromIterator<V> + Zeros + Clone,
    for<'a> &'a T: IntoIterator<Item = &'a V>,
    V: Float + UlpsEq,
{
    type Output = Opinion<T, V>;
    fn discount(&self, t: V) -> Self::Output {
        (self.simplex.discount(t), (*self.base_rate).clone()).into()
    }
}

impl<T, V> Discount<T, V> for Opinion<T, V>
where
    T: FromIterator<V> + Zeros + Clone,
    for<'a> &'a T: IntoIterator<Item = &'a V>,
    V: Float + UlpsEq,
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
    T: Container<Idx, Output = V>,
{
    let mut sum_b = V::zero();
    for (i, &bi) in b.iter_with() {
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
    T: Container<Idx, Output = V>,
    Idx: fmt::Debug + Clone,
{
    let mut sum_a = V::zero();
    for (i, &ai) in a.iter_with() {
        check_unit_interval(ai, format!("a[{i:?}]"))?;
        sum_a += ai;
    }

    check_is_one(sum_a, "sum(a)")?;

    Ok(())
}

fn compute_simlex<T, V, Idx>(op: &FuseOp, lhs: &Simplex<T, V>, rhs: &Simplex<T, V>) -> Simplex<T, V>
where
    T: Container<Idx, Output = V> + FromFn<Idx, V> + Zeros + Clone + IndexMut<Idx>,
    V: Float + UlpsEq + fmt::Debug + AddAssign + DivAssign + Sum,
    Idx: Copy + fmt::Debug,
{
    if lhs.is_dogmatic() && rhs.is_dogmatic() {
        Simplex::normalized(
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
                Simplex::normalized(b, u)
            }
            FuseOp::Avg if lhs.is_dogmatic() => lhs.clone(),
            FuseOp::Avg if rhs.is_dogmatic() => rhs.clone(),
            FuseOp::Avg => {
                let lhs_u = *lhs.u();
                let rhs_u = *rhs.u();
                let temp = lhs_u + rhs_u;
                let b = T::from_fn(|i| (lhs.b()[i] * rhs_u + rhs.b()[i] * lhs_u) / temp);
                let u = (V::one() + V::one()) * lhs_u * rhs_u / temp;
                Simplex::normalized(b, u)
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
                Simplex::normalized(b, u)
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
    T: Index<Idx, Output = V> + Clone + FromFn<Idx, V>,
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
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
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
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
    Idx: Copy + fmt::Debug,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: &Opinion<T, V>, rhs: &Simplex<T, V>) -> Self::Output {
        self.fuse(lhs.as_ref(), rhs)
    }
}

impl<T, V, Idx> Fuse<&Opinion<T, V>, &Opinion<T, V>, Idx> for FuseOp
where
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
    Idx: Copy + fmt::Debug,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: &Opinion<T, V>, rhs: &Opinion<T, V>) -> Self::Output {
        self.fuse(lhs.as_ref(), rhs.as_ref())
    }
}

impl<'a, T, V, Idx> Fuse<OpinionRef<'a, T, V>, &'a Simplex<T, V>, Idx> for FuseOp
where
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
    Idx: Copy + fmt::Debug,
{
    type Output = Opinion<T, V>;

    fn fuse(&self, lhs: OpinionRef<'a, T, V>, rhs: &'a Simplex<T, V>) -> Self::Output {
        self.fuse(lhs.clone(), OpinionRef::from((rhs, lhs.base_rate)))
    }
}

impl<T, V, Idx> Fuse<&Simplex<T, V>, &Simplex<T, V>, Idx> for FuseOp
where
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
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
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
    Idx: Copy + fmt::Debug,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: &Opinion<T, V>) {
        self.fuse_assign(lhs, rhs.as_ref())
    }
}

impl<'a, T, V, Idx> FuseAssign<Opinion<T, V>, OpinionRef<'a, T, V>, Idx> for FuseOp
where
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
    Idx: Copy + fmt::Debug,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: OpinionRef<'a, T, V>) {
        *lhs = self.fuse(lhs.as_ref(), rhs);
    }
}

impl<T, V, Idx> FuseAssign<Opinion<T, V>, &Simplex<T, V>, Idx> for FuseOp
where
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
    Idx: Copy + fmt::Debug,
{
    fn fuse_assign(&self, lhs: &mut Opinion<T, V>, rhs: &Simplex<T, V>) {
        *lhs = self.fuse(lhs.as_ref(), rhs);
    }
}

impl<T, V, Idx> FuseAssign<Simplex<T, V>, &Simplex<T, V>, Idx> for FuseOp
where
    T: Container<Idx, Output = V> + Clone + Zeros + FromFn<Idx, V> + IndexMut<Idx>,
    V: Float + UlpsEq + AddAssign + DivAssign + fmt::Debug + Sum,
    Idx: Copy + fmt::Debug,
{
    fn fuse_assign(&self, lhs: &mut Simplex<T, V>, rhs: &Simplex<T, V>) {
        *lhs = self.fuse(&(*lhs), rhs);
    }
}

pub fn mbr<'a, X, Y, T, Cond, U, V>(ax: &T, conds: &'a Cond) -> Option<U>
where
    T: Index<X, Output = V>,
    Cond: Container<X>,
    for<'b> &'b Cond::Output: Into<&'b Simplex<U, V>>,
    U: Container<Y, Output = V> + FromFn<Y, V> + IndexMut<Y>,
    X: Copy,
    Y: Copy,
    V: Float + Sum + UlpsEq + AddAssign + DivAssign,
{
    if Cond::indexes().all(|x| conds[x].into().is_vacuous()) {
        return None;
    }
    let mut sum_a = V::zero();
    let mut ay = U::from_fn(|y| {
        let a = Cond::indexes()
            .map(|x| ax[x] * conds[x].into().belief[y.clone()])
            .sum::<V>();
        sum_a += a;
        a
    });
    for y in U::indexes() {
        ay[y] /= sum_a;
    }
    Some(ay)
}

fn projections<'a, X, Y, Cond, U, V>(conds: &'a Cond, ay: &U) -> Cond::Map<U>
where
    Y: Copy,
    Cond: Index<X> + ContainerMap<X>,
    Cond::Output: 'a,
    &'a Cond::Output: Into<&'a Simplex<U, V>>,
    U: Container<Y, Output = V> + FromFn<Y, V> + IndexMut<Y> + 'a,
    V: Float + AddAssign + DivAssign + 'a,
{
    Cond::map(|x| conds[x].into().projection(ay))
}

impl<'a, X, Y, Cond, U, T, V> Deduction<X, Y, &'a Cond, U> for OpinionRef<'a, T, V>
where
    T: Container<X, Output = V> + FromFn<X, V> + IndexMut<X>,
    Cond: Container<X> + ContainerMap<X>,
    for<'b> &'b Cond::Output: Into<&'b Simplex<U, V>>,
    U: Container<Y, Output = V> + FromFn<Y, V> + IndexMut<Y>,
    X: Copy,
    Y: Copy + fmt::Debug,
    V: Float + Sum + UlpsEq + AddAssign + DivAssign + fmt::Debug,
{
    type Output = Opinion<U, V>;

    fn deduce(self, conds: &'a Cond) -> Option<Self::Output> {
        let ay = mbr::<X, Y, T, Cond, U, V>(&self.base_rate, conds)?;
        Some(self.deduce_with(conds, ay))
    }

    fn deduce_with(self, conds: &'a Cond, ay: U) -> Self::Output {
        let cond_p = projections(conds, &ay);
        let pyhx: U = U::from_fn(|y| {
            self.base_rate
                .iter_with()
                .map(|(x, &a)| a * cond_p[x][y])
                .sum()
        });
        let uyhx = U::indexes()
            .map(|y| {
                (pyhx[y]
                    - Cond::indexes()
                        .map(|x| conds[x].into().belief[y])
                        .reduce(<V>::min)
                        .unwrap())
                    / ay[y]
            })
            .reduce(<V>::min)
            .unwrap();
        let u = uyhx
            - Cond::indexes()
                .map(|x| (uyhx - conds[x].into().uncertainty) * self.b()[x])
                .sum::<V>();
        let p = self.projection();
        let b = U::from_fn(|y| T::indexes().map(|x| p[x] * cond_p[x][y]).sum::<V>() - ay[y] * u);
        Opinion::<U, V>::from((Simplex::normalized(b, u), ay))
    }
}

impl<'a, X, Y, Cond, U, T, V> Deduction<X, Y, &'a Cond, U> for &'a Opinion<T, V>
where
    T: Container<X, Output = V> + FromFn<X, V> + IndexMut<X>,
    Cond: Container<X> + ContainerMap<X>,
    for<'b> &'b Cond::Output: Into<&'b Simplex<U, V>>,
    U: Container<Y, Output = V> + FromFn<Y, V> + IndexMut<Y>,
    X: Copy,
    Y: Copy + fmt::Debug,
    V: Float + Sum + UlpsEq + AddAssign + DivAssign + fmt::Debug,
{
    type Output = Opinion<U, V>;
    fn deduce(self, conds: &'a Cond) -> Option<Self::Output> {
        self.as_ref().deduce(conds)
    }

    fn deduce_with(self, conds: &'a Cond, ay: U) -> Self::Output {
        self.as_ref().deduce_with(conds, ay)
    }
}

pub trait InverseCondition<X, Y, T, U, V>
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
    T: Container<X, Output = V> + ContainerMap<X> + FromFn<X, V> + IndexMut<X>,
    U: Container<Y, Output = V> + ContainerMap<Y> + FromFn<Y, V> + IndexMut<Y>,
    Cond: Container<X, Output = Simplex<U, V>> + ContainerMap<X>,
    X: Copy + fmt::Debug,
    Y: Copy,
    V: Float + AddAssign + DivAssign + Sum + UlpsEq + fmt::Debug,
{
    type InvCond = U::Map<Simplex<T, V>>;

    fn inverse(&self, ax: &T, ay: &U) -> Self::InvCond {
        let p_yx: Cond::Map<U> = Cond::map(|x| self[x].projection(ay));
        let u_yx = T::from_fn(|x| self[x].max_uncertainty(ay));
        let p_xy: U::Map<T::Map<V>> = U::map(|y| {
            T::map(|x| ax[x] * p_yx[x][y] / T::indexes().map(|xd| ax[xd] * p_yx[xd][y]).sum::<V>())
        });
        let irrelevance_yx = U::from_fn(|y| {
            V::one() - T::indexes().map(|x| p_yx[x][y]).reduce(<V>::max).unwrap()
                + T::indexes().map(|x| p_yx[x][y]).reduce(<V>::min).unwrap()
        });
        let max_u_xy: U = U::from_fn(|y| {
            T::indexes()
                .map(|x| p_yx[x][y] / T::indexes().map(|k| ax[k] * p_yx[k][y]).sum::<V>())
                .reduce(<V>::min)
                .unwrap()
        });
        let u_yx_sum = T::indexes().map(|x| u_yx[x]).sum::<V>();
        let weights_yx = if u_yx_sum == V::zero() {
            T::from_fn(|_| V::zero())
        } else {
            T::from_fn(|x| u_yx[x] / u_yx_sum)
        };
        let max_u_yx = T::from_fn(|x| {
            U::indexes()
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
        let wprop_u_yx = T::indexes().map(|x| weighted_u_yx[x]).sum::<V>();
        U::map(|y| {
            let u = max_u_xy[y] * (wprop_u_yx + irrelevance_yx[y] - wprop_u_yx * irrelevance_yx[y]);
            let b = T::from_fn(|x| p_xy[y][x] - u * ax[x]);
            Simplex::normalized(b, u)
        })
    }
}

impl<'a, Cond, X, Y, T, U, V> Abduction<&'a Cond, X, Y, T, U> for &'a Simplex<U, V>
where
    Cond: InverseCondition<X, Y, T, U, V> + Container<X, Output = Simplex<U, V>>,
    Cond::InvCond: Container<Y, Output = Simplex<T, V>> + ContainerMap<Y> + 'a,
    T: Container<X, Output = V> + FromFn<X, V> + IndexMut<X> + 'a,
    U: Container<Y, Output = V> + FromFn<Y, V> + IndexMut<Y>,
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
    Cond: InverseCondition<X, Y, T, U, V> + Container<X, Output = Simplex<U, V>>,
    Cond::InvCond: Container<Y, Output = Simplex<T, V>> + ContainerMap<Y> + 'a,
    T: Container<X, Output = V> + FromFn<X, V> + IndexMut<X> + 'a,
    U: Container<Y, Output = V> + FromFn<Y, V> + IndexMut<Y>,
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
    Cond: InverseCondition<X, Y, T, U, V> + Container<X, Output = Simplex<U, V>>,
    Cond::InvCond: Container<Y, Output = Simplex<T, V>> + ContainerMap<Y> + 'a,
    T: Container<X, Output = V> + FromFn<X, V> + IndexMut<X> + 'a,
    U: Container<Y, Output = V> + FromFn<Y, V> + IndexMut<Y>,
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
