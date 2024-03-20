use core::fmt;
use std::ops::{AddAssign, DivAssign};

use approx::UlpsEq;
use itertools::izip;
use num_traits::Float;

use crate::{
    errors::InvalidValueError,
    multi_array::labeled::{product2_iter, product3_iter, Domain, MArrD1, MArrD2, MArrD3},
    ops::{Product2, Product3, Projection},
};

use super::{Opinion, OpinionRef, Simplex};

/// A simplex with a 1-dimensional domain `D0`.
pub type SimplexD1<D0, V> = Simplex<MArrD1<D0, V>, V>;

/// A multinomial opinion with a 1-dimensional domain `D0`.
pub type OpinionD1<D0, V> = Opinion<MArrD1<D0, V>, V>;

/// A reference of a multinomial opinion with a 1-dimensional domain `D0`.
pub type OpinionRefD1<'a, D0, V> = OpinionRef<'a, MArrD1<D0, V>, V>;

impl<D0, V> TryFrom<(Vec<V>, V)> for SimplexD1<D0, V>
where
    D0: Domain<Idx = usize>,
    V: Float + AddAssign + UlpsEq,
{
    type Error = InvalidValueError;

    fn try_from(value: (Vec<V>, V)) -> Result<Self, Self::Error> {
        Self::try_new(MArrD1::from_iter(value.0), value.1)
    }
}

impl<'a, D0, D1, V> Product2<OpinionRefD1<'a, D0, V>, OpinionRefD1<'a, D1, V>>
    for Opinion<MArrD2<D0, D1, V>, V>
where
    D0: Domain<Idx = usize>,
    D1: Domain<Idx = usize>,
    V: Float + AddAssign + DivAssign + fmt::Debug + UlpsEq + Default,
{
    fn product2(w0: OpinionRefD1<D0, V>, w1: OpinionRefD1<D1, V>) -> Self {
        let p0 = w0.projection();
        let p1 = w1.projection();
        let p_iter = product2_iter(&p0, &p1);
        let b_iter = product2_iter(&w0.simplex.belief, &w1.simplex.belief);
        let a = MArrD2::product2(&w0.base_rate, &w1.base_rate);
        let u = izip!(p_iter.clone(), b_iter, a.values())
            .map(|(p, b, &a)| (p - b) / a)
            .reduce(V::min)
            .unwrap();
        let b = MArrD2::<D0, D1, V>::from_iter(p_iter.zip(a.values()).map(|(p, &a)| p - a * u));
        Opinion::new(b, u, a)
    }
}

impl<'a, D0, D1, D2, V>
    Product3<OpinionRefD1<'a, D0, V>, OpinionRefD1<'a, D1, V>, OpinionRefD1<'a, D2, V>>
    for Opinion<MArrD3<D0, D1, D2, V>, V>
where
    D0: Domain<Idx = usize>,
    D1: Domain<Idx = usize>,
    D2: Domain<Idx = usize>,
    V: Float + AddAssign + DivAssign + fmt::Debug + UlpsEq + Default,
{
    fn product3(w0: OpinionRefD1<D0, V>, w1: OpinionRefD1<D1, V>, w2: OpinionRefD1<D2, V>) -> Self {
        let p0 = w0.projection();
        let p1 = w1.projection();
        let p2 = w2.projection();
        let p_iter = product3_iter(&p0, &p1, &p2);
        let b_iter = product3_iter(&w0.simplex.belief, &w1.simplex.belief, &w2.simplex.belief);
        let a = MArrD3::product3(&w0.base_rate, &w1.base_rate, &w2.base_rate);
        let u = izip!(p_iter.clone(), b_iter, a.values())
            .map(|(p, b, &a)| (p - b) / a)
            .reduce(V::min)
            .unwrap();
        let b = MArrD3::<D0, D1, D2, _>::from_iter(p_iter.zip(a.values()).map(|(p, &a)| p - a * u));
        Opinion::new(b, u, a)
    }
}
