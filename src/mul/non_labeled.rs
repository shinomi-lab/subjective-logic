use std::{
    fmt,
    ops::{AddAssign, DivAssign},
    usize,
};

use approx::UlpsEq;
use num_traits::Float;

use super::{check_base_rate, IndexedContainer, Opinion, OpinionRef, Projection, Simplex};
use crate::ops::{Product2, Product3};
use crate::{
    errors::InvalidValueError,
    multi_array::non_labeled::{MArr2, MArr3},
};

/// A simplex of a multinomial opinion, from which a base rate is excluded.
pub type Simplex1d<V, const N: usize> = Simplex<[V; N], V>;

/// A multinomial opinion with 1-dimensional vectors.
pub type Opinion1d<V, const N: usize> = Opinion<[V; N], V>;

/// A reference of a multinomial opinion with 1-dimensional vectors.
pub type Opinion1dRef<'a, V, const N: usize> = OpinionRef<'a, [V; N], V>;

impl<V, const N: usize> TryFrom<([V; N], V)> for Simplex1d<V, N>
where
    V: Float + AddAssign + UlpsEq,
{
    type Error = InvalidValueError;

    fn try_from(value: ([V; N], V)) -> Result<Self, Self::Error> {
        Self::try_new(value.0, value.1)
    }
}

impl<V, const N: usize> Simplex1d<V, N>
where
    V: Float + AddAssign + UlpsEq,
{
    pub fn into_opinion(self, a: [V; N]) -> Result<Opinion1d<V, N>, InvalidValueError> {
        check_base_rate(&a)?;
        Ok(Opinion1d {
            simplex: self,
            base_rate: a,
        })
    }
}

impl<'a, V, const D0: usize, const D1: usize>
    Product2<Opinion1dRef<'a, V, D0>, Opinion1dRef<'a, V, D1>> for Opinion<MArr2<V, D0, D1>, V>
where
    V: Float + AddAssign + DivAssign + fmt::Debug + UlpsEq,
{
    fn product2(w0: Opinion1dRef<V, D0>, w1: Opinion1dRef<V, D1>) -> Self {
        let p = MArr2::product2(&w0.projection(), &w1.projection());
        let a = MArr2::from_fn(|d| w0.base_rate[d[0]] * w1.base_rate[d[1]]);
        let u = MArr2::<V, D0, D1>::keys()
            .map(|d| (p[d] - w0.b()[d[0]] * w1.b()[d[1]]) / a[d])
            .reduce(<V>::min)
            .unwrap();
        let b = MArr2::from_fn(|d| p[d] - a[d] * u);
        Opinion::new(b, u, a)
    }
}

impl<'a, V, const D0: usize, const D1: usize, const D2: usize>
    Product3<Opinion1dRef<'a, V, D0>, Opinion1dRef<'a, V, D1>, Opinion1dRef<'a, V, D2>>
    for Opinion<MArr3<V, D0, D1, D2>, V>
where
    V: Float + AddAssign + DivAssign + fmt::Debug + UlpsEq,
{
    fn product3(w0: Opinion1dRef<V, D0>, w1: Opinion1dRef<V, D1>, w2: Opinion1dRef<V, D2>) -> Self {
        let p = MArr3::product3(&w0.projection(), &w1.projection(), &w2.projection());
        let a = MArr3::from_fn(|d| w0.base_rate[d[0]] * w1.base_rate[d[1]] * w2.base_rate[d[2]]);
        let u = MArr3::<V, D0, D1, D2>::keys()
            .map(|d| (p[d] - w0.b()[d[0]] * w1.b()[d[1]] * w2.b()[d[2]]) / a[d])
            .reduce(<V>::min)
            .unwrap();
        let b = MArr3::from_fn(|d| p[d] - a[d] * u);
        Opinion::new(b, u, a)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::{Product2, Projection};
    use crate::{
        marr2,
        mul::{
            non_labeled::{Opinion1d, Simplex1d},
            IndexedContainer, Opinion,
        },
        multi_array::non_labeled::MArr2,
        ops::Deduction,
    };

    #[test]
    fn test_prod2() {
        macro_rules! def {
            ($ft: ty) => {
                let w0 = Opinion1d::<$ft, 2>::new([0.1, 0.2], 0.7, [0.75, 0.25]);
                let w1 = Opinion1d::<$ft, 3>::new([0.1, 0.2, 0.3], 0.4, [0.5, 0.49, 0.01]);
                let w01 = Opinion::product2(w0.as_ref(), w1.as_ref());
                let p = w01.projection();
                assert_ulps_eq!(p.into_iter().sum::<$ft>(), 1.0);
                let p01 = MArr2::<_, 2, 3>::product2(&w0.projection(), &w1.projection());
                println!("{:?}", w01);
                println!("{:?}, {}", p, p.into_iter().sum::<$ft>());
                println!("{:?}, {}", p01, p01.into_iter().sum::<$ft>());
                for d in MArr2::<$ft, 2, 3>::keys() {
                    println!("{}", (p[d] - w01.b()[d]) / w01.base_rate[d]);
                }
            };
        }
        def!(f32);
        def!(f64);
    }

    macro_rules! nround {
        [$ft:ty, $n:expr] => {
            |v: $ft| (v * <$ft>::powi(10.0, $n)).round()
        };
    }

    #[test]
    fn test_deduction() {
        macro_rules! def {
            ($ft: ty) => {
                let wx = Opinion1d::<$ft, 2>::new([0.9, 0.0], 0.1, [0.1, 0.9]);
                let wy = Opinion1d::<$ft, 2>::new([0.5, 0.5], 0.0, [0.5, 0.5]);
                let wxy = Opinion::product2(&wx, &wy);
                let conds = marr2![
                    [
                        Simplex1d::<$ft, 3>::new([0.0, 0.8, 0.1], 0.1),
                        Simplex1d::<$ft, 3>::new([0.0, 0.8, 0.1], 0.1),
                    ],
                    [
                        Simplex1d::<$ft, 3>::new([0.7, 0.0, 0.1], 0.2),
                        Simplex1d::<$ft, 3>::new([0.7, 0.0, 0.1], 0.2),
                    ]
                ];
                let wy = wxy.as_ref().deduce(&conds).unwrap();
                // base rate
                assert_eq!(wy.base_rate.map(nround![$ft, 3]), [778.0, 99.0, 123.0]);
                // projection
                let p = wy.projection();
                assert_eq!(p.map(nround![$ft, 3]), [148.0, 739.0, 113.0]);
                // belief
                assert_eq!(wy.b().map(nround![$ft, 3]), [63.0, 728.0, 100.0]);
                // uncertainty
                assert_eq!(nround![$ft, 3](wy.u()), 109.0);

                assert_ulps_eq!(wy.base_rate.into_iter().sum::<$ft>(), 1.0);
                assert_ulps_eq!(wy.b().into_iter().sum::<$ft>() + wy.u(), 1.0);
                assert_ulps_eq!(wy.projection().into_iter().sum::<$ft>(), 1.0);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_msl_boundary() {
        macro_rules! def {
            ($ft: ty) => {
                assert!(Opinion1d::<$ft, 2>::try_new([0.0, 0.0], 1.0, [0.0, 1.0]).is_ok());
                assert!(Opinion1d::<$ft, 2>::try_new([0.0, 1.0], 0.0, [0.0, 1.0]).is_ok());
                assert!(Opinion1d::<$ft, 2>::try_new([0.1, -0.1], 1.0, [0.0, 1.0])
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(Opinion1d::<$ft, 2>::try_new([0.0, 1.0], 0.0, [1.1, -0.1])
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(Opinion1d::<$ft, 2>::try_new([0.0, -1.0], 2.0, [1.1, -0.1])
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(Opinion1d::<$ft, 2>::try_new([1.0, 1.0], -1.0, [1.1, -0.1])
                    .map_err(|e| println!("{e}"))
                    .is_err());
            };
        }
        def!(f32);
        def!(f64);
    }
}
