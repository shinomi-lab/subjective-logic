use core::fmt;
use std::ops::{AddAssign, DivAssign};

use approx::UlpsEq;
use itertools::izip;
use num_traits::Float;

use crate::{
    domain::Domain,
    domain::DomainConv,
    errors::InvalidValueError,
    multi_array::labeled::{product2_iter, product3_iter, MArrD1, MArrD2, MArrD3},
    ops::{Product2, Product3, Projection},
};

use super::{Opinion, OpinionRef, Simplex};

/// A simplex with a 1-dimensional domain `D0`.
pub type SimplexD1<D0, V> = Simplex<MArrD1<D0, V>, V>;

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

impl<D, E, V> DomainConv<SimplexD1<E, V>> for SimplexD1<D, V>
where
    D: Domain,
    E: Domain + From<D>,
{
    fn conv(self) -> SimplexD1<E, V> {
        Simplex {
            belief: self.belief.conv(),
            uncertainty: self.uncertainty,
        }
    }
}

/// A multinomial opinion with a 1-dimensional domain `D0`.
pub type OpinionD1<D0, V> = Opinion<MArrD1<D0, V>, V>;

/// A multinomial opinion with a 1-dimensional domain `D0`.
pub type OpinionD2<D0, D1, V> = Opinion<MArrD2<D0, D1, V>, V>;

/// A multinomial opinion with a 1-dimensional domain `D0`.
pub type OpinionD3<D0, D1, D2, V> = Opinion<MArrD3<D0, D1, D2, V>, V>;

/// A reference of a multinomial opinion with a 1-dimensional domain `D0`.
pub type OpinionRefD1<'a, D0, V> = OpinionRef<'a, MArrD1<D0, V>, V>;

impl<'a, D0, D1, V> Product2<OpinionRefD1<'a, D0, V>, OpinionRefD1<'a, D1, V>>
    for OpinionD2<D0, D1, V>
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
        let u = izip!(p_iter.clone(), b_iter, &a)
            .map(|(p, b, &a)| (p - b) / a)
            .reduce(V::min)
            .unwrap();
        let b = MArrD2::<D0, D1, V>::from_iter(p_iter.zip(&a).map(|(p, &a)| p - a * u));
        Opinion::normalized(b, u, a)
    }
}

impl<'a, D0, D1, D2, V>
    Product3<OpinionRefD1<'a, D0, V>, OpinionRefD1<'a, D1, V>, OpinionRefD1<'a, D2, V>>
    for OpinionD3<D0, D1, D2, V>
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
        let u = izip!(p_iter.clone(), b_iter, &a)
            .map(|(p, b, &a)| (p - b) / a)
            .reduce(V::min)
            .unwrap();
        let b = MArrD3::<D0, D1, D2, _>::from_iter(p_iter.zip(&a).map(|(p, &a)| p - a * u));
        Opinion::normalized(b, u, a)
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use approx::{assert_ulps_eq, ulps_eq};

    use super::{OpinionD1, OpinionD2, OpinionRefD1, SimplexD1};
    use crate::{
        domain::{Domain, DomainConv},
        impl_domain,
        iter::Container,
        marr_d1, marr_d2,
        mul::{check_base_rate, mbr},
        multi_array::labeled::{MArrD1, MArrD2},
        ops::{
            Abduction, Deduction, Discount, Fuse, FuseAssign, FuseOp, MaxUncertainty, Product2,
            Projection,
        },
    };

    macro_rules! nround {
        [$ft:ty, $n:expr] => {
            |v: $ft| (v * <$ft>::powi(10.0, $n)).round()
        };
    }
    macro_rules! nround_arr {
        ($ft:ty, $n:expr, $iter:expr) => {
            array::from_fn(|i| ($iter[i] * <$ft>::powi(10.0, $n)).round())
        };
    }

    macro_rules! nfract {
        [$ft:ty, $n:expr] => {
            |v: $ft| (v * <$ft>::powi(10.0, $n)).fract()
        };
    }

    struct X;
    impl_domain!(X = 2);

    struct Y;
    impl_domain!(Y = 3);

    struct Z;
    impl_domain!(Z = 2);

    #[test]
    fn test_prod2() {
        macro_rules! def {
            ($ft: ty) => {
                let w0 = OpinionD1::<X, $ft>::new(marr_d1!(_; [0.1, 0.2]), 0.7, marr_d1!(_; [0.75, 0.25]));
                let w1 = OpinionD1::<Y, $ft>::new(
                    marr_d1!(_; [0.1, 0.2, 0.3]),
                    0.4,
                    marr_d1!(_; [0.5, 0.49, 0.01]),
                );
                let w01 = OpinionD2::product2(w0.as_ref(), w1.as_ref());
                let p = w01.projection();
                assert_ulps_eq!(p.into_iter().sum::<$ft>(), 1.0);
                let p01 = MArrD2::product2(&w0.projection(), &w1.projection());
                println!("{:?}", w01);
                println!("{:?}, {}", p, p.into_iter().sum::<$ft>());
                println!("{:?}, {}", p01, p01.into_iter().sum::<$ft>());
                for (d, &pd) in p.iter_with() {
                    println!("{}", (pd - w01.b()[d]) / w01.base_rate[d]);
                }
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_msl_boundary() {
        macro_rules! def {
            ($ft: ty) => {
                assert!(
                    OpinionD1::<X, $ft>::try_new(marr_d1![0.0, 0.0], 1.0, marr_d1![0.0, 1.0])
                        .is_ok()
                );
                assert!(
                    OpinionD1::<X, $ft>::try_new(marr_d1![0.0, 1.0], 0.0, marr_d1![0.0, 1.0])
                        .is_ok()
                );
                assert!(
                    OpinionD1::<X, $ft>::try_new(marr_d1![0.1, -0.1], 1.0, marr_d1![0.0, 1.0])
                        .map_err(|e| println!("{e}"))
                        .is_err()
                );
                assert!(
                    OpinionD1::<X, $ft>::try_new(marr_d1![0.0, 1.0], 0.0, marr_d1![1.1, -0.1])
                        .map_err(|e| println!("{e}"))
                        .is_err()
                );
                assert!(OpinionD1::<X, $ft>::try_new(
                    marr_d1![0.0, -1.0],
                    2.0,
                    marr_d1![1.1, -0.1]
                )
                .map_err(|e| println!("{e}"))
                .is_err());
                assert!(OpinionD1::<X, $ft>::try_new(
                    marr_d1![1.0, 1.0],
                    -1.0,
                    marr_d1![1.1, -0.1]
                )
                .map_err(|e| println!("{e}"))
                .is_err());
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_discount() {
        macro_rules! def {
            ($ft: ty) => {
                let w = OpinionD1::<X, $ft>::new(marr_d1!(_; [0.2, 0.2]), 0.6, marr_d1!(_; [0.5, 0.5]));
                let w2 = w.discount(0.5);
                assert!(ulps_eq!(w2.b()[0], 0.1));
                assert!(ulps_eq!(w2.b()[1], 0.1));
                assert!(ulps_eq!(w2.u(), 1.0 - 0.2));
                assert!(ulps_eq!(w2.b()[0] + w2.b()[1] + w2.u(), 1.0));
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
                let w =
                    OpinionD1::<X, _>::new(marr_d1![0.2, 0.2], 0.6, marr_d1![1.0 / 3.0, 2.0 / 3.0]);
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
                let mut w = OpinionD1::<X, $ft>::default();
                assert_eq!(w.simplex.belief, MArrD1::default());
                assert_eq!(w.simplex.uncertainty, 0.0);
                assert_eq!(w.base_rate, MArrD1::default());
                let b = marr_d1!(X; [1.0, 0.0]);
                w.simplex.belief = b.clone();
                w.base_rate = b.clone();
                assert_eq!(w.b(), &b);
                assert_eq!(&w.base_rate, &b);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_max_u() {
        let w = SimplexD1::new(marr_d1!(X; [0.0, 0.0]), 1.0);
        assert_eq!(w.max_uncertainty(&marr_d1![0.0, 1.0]), 1.0);
    }

    #[test]
    fn test_float_size() {
        //  [0.744088550149486, 0.2483314796475328] u=0.007579970202981336
        //  [0.578534230384464, 0.413885799412554y] u=0.007579970202981336
        //  [0.578534230384464, 0.413885799412554y] u=0.007579970202981336
        let conds = marr_d2!(X, X; [
            [
                SimplexD1::<X, _>::try_from((vec![0.95f64, 0.00], 0.05)).unwrap(),
                SimplexD1::<X, _>::try_from((vec![0.7440885, 0.24833153], 0.00757997)).unwrap(),
            ],
            [
                // Simplex::new_unchecked([0.5785341, 0.41388586], 0.00757997),
                // Simplex::new_unchecked([0.5785342, 0.41388586], 0.00757997),
                SimplexD1::<X, _>::try_from((vec![0.5785342, 0.41388586], 0.00757994)).unwrap(),
                SimplexD1::<X, _>::try_from((vec![0.5785342, 0.41388586], 0.00757994)).unwrap(),
            ]
        ]);
        let w0 = OpinionD1::<X, _>::vacuous_with(marr_d1![0.4995, 0.5005]);
        let w1 = OpinionD1::<X, _>::vacuous_with(marr_d1![0.000001, 0.999999]);
        let w1d = OpinionD1::<X, _>::new(marr_d1![0.0, 0.95], 0.05, marr_d1![0.000001, 0.999999]);
        let w = OpinionD2::product2(&w0, &w1).deduce(&conds);
        let wd = OpinionD2::product2(&w0, &w1d).deduce(&conds);
        println!("{w:?}");
        println!("{wd:?}");
    }

    #[test]
    fn test_fusion_dogma() {
        macro_rules! def {
            ($ft: ty) => {
                let w1 = OpinionD1::<Y, _>::new(
                    marr_d1![0.99, 0.01, 0.0],
                    0.0,
                    marr_d1![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                );
                let w2 = SimplexD1::<Y, _>::new(marr_d1![0.0, 0.01, 0.99], 0.0);

                // A-CBF
                let w = FuseOp::ACm.fuse(&w1, &w2);
                assert_eq!(
                    array::from_fn(|i| w.b()[i]).map(nround![$ft, 3]),
                    [495.0, 10.0, 495.0]
                );
                assert_eq!(nround![$ft, 3](w.u()), 0.0);
                assert_eq!(
                    array::from_fn(|i| w.base_rate[i]).map(nround![$ft, 3]),
                    [333.0, 333.0, 333.0]
                );
                // E-CBF
                let w = FuseOp::ECm.fuse(&w1, &w2);
                assert_eq!(
                    array::from_fn(|i| w.b()[i]).map(nround![$ft, 3]),
                    [485.0, 0.0, 485.0]
                );
                assert_eq!(nround![$ft, 3](w.u()), 30.0);
                assert_eq!(
                    array::from_fn(|i| w.base_rate[i]).map(nround![$ft, 3]),
                    [333.0, 333.0, 333.0]
                );
                // ABF
                let w = FuseOp::Avg.fuse(&w1, &w2);
                assert_eq!(
                    array::from_fn(|i| w.b()[i]).map(nround![$ft, 3]),
                    [495.0, 10.0, 495.0]
                );
                assert_eq!(nround![$ft, 3](w.u()), 0.0);
                assert_eq!(
                    array::from_fn(|i| w.base_rate[i]).map(nround![$ft, 3]),
                    [333.0, 333.0, 333.0]
                );
                // WBF
                let w = FuseOp::Wgh.fuse(&w1, &w2);
                assert_eq!(
                    array::from_fn(|i| w.b()[i]).map(nround![$ft, 3]),
                    [495.0, 10.0, 495.0]
                );
                assert_eq!(nround![$ft, 3](w.u()), 0.0);
                assert_eq!(
                    array::from_fn(|i| w.base_rate[i]).map(nround![$ft, 3]),
                    [333.0, 333.0, 333.0]
                );
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_fusion() {
        macro_rules! def {
            ($ft: ty) => {
                let w1 = OpinionD1::<Y, $ft>::new(
                    marr_d1![0.98, 0.01, 0.0],
                    0.01,
                    marr_d1![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                );
                let w2 = OpinionD1::<Y, $ft>::new(
                    marr_d1![0.0, 0.01, 0.90],
                    0.09,
                    marr_d1![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                );

                // A-CBF
                let w = FuseOp::ACm.fuse(&w1, &w2);
                assert_eq!(nround_arr!($ft, 3, w.b()), [890.0, 10.0, 91.0]);
                assert_eq!(nround![$ft, 3](w.u()), 9.0);
                assert_eq!(nround_arr!($ft, 3, w.base_rate), [333.0, 333.0, 333.0]);
                // E-CBF
                let w = FuseOp::ECm.fuse(&w1, &w2);
                assert_eq!(nround_arr!($ft, 3, w.b()), [880.0, 0.0, 81.0]);
                assert_eq!(nround![$ft, 3](w.u()), 39.0);
                assert_eq!(nround_arr!($ft, 3, w.base_rate), [333.0, 333.0, 333.0]);
                // ABF
                let w = FuseOp::Avg.fuse(&w1, &w2);
                assert_eq!(nround_arr!($ft, 3, w.b()), [882.0, 10.0, 90.0]);
                assert_eq!(nround![$ft, 3](w.u()), 18.0);
                assert_eq!(nround_arr!($ft, 3, w.base_rate), [333.0, 333.0, 333.0]);
                // WBF
                let w = FuseOp::Wgh.fuse(&w1, &w2);
                assert_eq!(nround_arr!($ft, 3, w.b()), [889.0, 10.0, 83.0]);
                assert_eq!(
                    nround![$ft, 3](w.u()),
                    18.0 - w
                        .b()
                        .iter()
                        .cloned()
                        .map(nfract![$ft, 3])
                        .sum::<$ft>()
                        .round()
                );
                assert_eq!(nround_arr!($ft, 3, w.base_rate), [333.0, 333.0, 333.0]);
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_fusion_assign() {
        macro_rules! def {
            ($ft: ty) => {
                let mut w = OpinionD1::<X, $ft>::new(marr_d1![0.5, 0.25], 0.25, marr_d1![0.5, 0.5]);
                let u = SimplexD1::<X, $ft>::new(marr_d1![0.125, 0.75], 0.125);
                let ops = [FuseOp::ACm, FuseOp::ECm, FuseOp::Avg, FuseOp::Wgh];
                for op in ops {
                    let w2 = op.fuse(w.as_ref(), OpinionRefD1::from((&u, &w.base_rate)));
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
                let mut w = SimplexD1::<X, $ft>::new(marr_d1![0.5, 0.25], 0.25);
                let u = SimplexD1::<X, $ft>::new(marr_d1![0.125, 0.75], 0.125);
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
                let cond1 = marr_d1!(Z; [SimplexD1::<X, $ft>::new(marr_d1![0.0, 0.0], 1.0), SimplexD1::new(marr_d1![0.0, 0.0], 1.0)]);
                let cond2 = marr_d1!(Z; [
                    SimplexD1::<X, _>::new(marr_d1![0.0, 0.01], 0.99),
                    SimplexD1::<X, _>::new(marr_d1![0.0, 0.0], 1.0),
                ]);
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
                let wx = OpinionD1::<X, $ft>::new(marr_d1![0.9, 0.0], 0.1, marr_d1![0.1, 0.9]);
                let wz = OpinionD1::<Z, $ft>::new(marr_d1![0.5, 0.5], 0.0, marr_d1![0.5, 0.5]);
                let wxz = OpinionD2::<X, Z, $ft>::product2(&wx, &wz);
                let conds = marr_d2!(X, Z; [
                    [
                        SimplexD1::<Y, $ft>::new(marr_d1![0.0, 0.8, 0.1], 0.1),
                        SimplexD1::<Y, $ft>::new(marr_d1![0.0, 0.8, 0.1], 0.1),
                    ],
                    [
                        SimplexD1::<Y, $ft>::new(marr_d1![0.7, 0.0, 0.1], 0.2),
                        SimplexD1::<Y, $ft>::new(marr_d1![0.7, 0.0, 0.1], 0.2),
                    ]
                ]);
                let wy = wxz.as_ref().deduce(&conds).unwrap();
                // base rate
                assert_eq!(array::from_fn(|i| wy.base_rate[i]).map(nround![$ft, 3]), [778.0, 99.0, 123.0]);
                // projection
                let p = wy.projection();
                assert_eq!(array::from_fn(|i| p[i]).map(nround![$ft, 3]), [148.0, 739.0, 113.0]);
                // belief
                assert_eq!(array::from_fn(|i| wy.b()[i]).map(nround![$ft, 3]), [63.0, 728.0, 100.0]);
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
    fn test_deduction2() {
        macro_rules! def {
            ($ft: ty) => {
                let wx = OpinionD1::<X, $ft>::new(marr_d1![0.9, 0.0], 0.1, marr_d1![0.1, 0.9]);
                let wxy = marr_d1!(X; [
                    SimplexD1::<Y, $ft>::new(marr_d1![0.0, 0.8, 0.1], 0.1),
                    SimplexD1::<Y, $ft>::new(marr_d1![0.7, 0.0, 0.1], 0.2),
                ]);
                let wy = wx.as_ref().deduce(&wxy).unwrap();
                // base rate
                assert_eq!(nround_arr!($ft, 3, wy.base_rate), [778.0, 99.0, 123.0]);
                // projection
                let p = wy.projection();
                assert_eq!(nround_arr!($ft, 3, p), [148.0, 739.0, 113.0]);
                // belief
                assert_eq!(nround_arr!($ft, 3, wy.b()), [63.0, 728.0, 100.0]);
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
            SimplexD1::new_unchecked(marr_d1!(Y;[0.25, 0.04, 0.00]), 0.71),
            SimplexD1::new_unchecked(marr_d1!(Y;[0.00, 0.50, 0.50]), 0.00),
            SimplexD1::new_unchecked(marr_d1!(Y;[0.00, 0.25, 0.75]), 0.00),
        ];
        let ax = [0.70, 0.20, 0.10];
        let wy = OpinionD1::new(marr_d1![0.00, 0.43, 0.00], 0.57, marr_d1![0.5, 0.5, 0.0]);
        let wx = wy.abduce(&conds, ax).unwrap();
        let m_ay = mbr(&ax, &conds).unwrap();
        println!("{:?}", wx);
        println!("{:?}", m_ay);
    }

    #[test]
    fn test_abduction2() {
        let ax = [0.01, 0.495, 0.495];
        let conds_ox = marr_d1!(Y; [
            SimplexD1::<X, _>::try_from((vec![0.5, 0.0], 0.5)).unwrap(),
            SimplexD1::<X, _>::try_from((vec![0.5, 0.0], 0.5)).unwrap(),
            SimplexD1::<X, _>::try_from((vec![0.01, 0.01], 0.98)).unwrap(),
        ]);
        let mw_o = OpinionD1::<X, _>::new(marr_d1![0.0, 0.0], 1.0, marr_d1![0.5, 0.5]);
        let mw_x = mw_o.abduce(&conds_ox, ax).unwrap();
        println!("{:?}", mw_x);
    }

    #[test]
    fn test_abduction3() {
        let dcond = marr_d1!(Y; [
            SimplexD1::<Y, _>::new(marr_d1![0.50, 0.25, 0.25], 0.0),
            SimplexD1::new(marr_d1![0.00, 0.50, 0.50], 0.0),
            SimplexD1::new(marr_d1![0.00, 0.25, 0.75], 0.0),
        ]);
        let ax = marr_d1!(Y; [0.70, 0.20, 0.10]);
        let wy = OpinionD1::new(marr_d1![0.00, 0.43, 0.00], 0.57, marr_d1![0.0, 0.0, 1.0]);
        let m_ay = mbr(&ax, &dcond).unwrap();
        println!("{m_ay:?}");

        let wx = wy.abduce(&dcond, ax).unwrap();
        assert_eq!(nround_arr!(f32, 2, wx.b()), [0.0, 7.0, 0.0]);
        assert_eq!(nround![f32, 2](wx.u()), 93.0);
        assert_eq!(nround_arr!(f32, 2, wx.projection()), [65.0, 26.0, 9.0]);
    }

    struct W;
    impl_domain!(W from X);

    #[test]
    fn test_conv() {
        let wx = SimplexD1::<X, _>::try_from((vec![0.5, 0.0], 0.5)).unwrap();
        let ww: SimplexD1<W, _> = wx.conv();
        assert_eq!(X::LEN, W::LEN);
        assert_eq!(ww.b(), &marr_d1![0.5, 0.0]);
    }

    #[test]
    fn test_fuse_error() {
        let mut w = OpinionD1::<X, f32>::new(
            marr_d1![0.25, 0.00016426429],
            0.7498357,
            marr_d1![0.9993434, 0.0006566254],
        );
        let s = SimplexD1::new(marr_d1![0.0, 0.03733333], 0.9626667);
        FuseOp::Wgh.fuse_assign(&mut w, &s);
        println!("{w:?}");
    }

    use super::super::MergeJointConditions2;

    #[test]
    fn test_merge_joint_conds() {
        // let k = 0.0001;
        let k = 0.00001;
        let cyx = marr_d1!(X; [
            SimplexD1::<Y, f64>::new(marr_d1![0.4, 0.0, 0.2], 0.4),
            SimplexD1::<Y, f64>::new(marr_d1![0.4, 0.0 + k, 0.2], 0.4 - k),
        ]);
        let cyz = marr_d1!(Z; [
            SimplexD1::<Y, f64>::new(marr_d1![0.2, 0.2, 0.0], 0.6),
            SimplexD1::<Y, f64>::new(marr_d1![0.2, 0.2, k], 0.6 - k),
        ]);
        let ax = marr_d1!(X; [0.5, 0.5]);
        let az = marr_d1!(Z; [0.5, 0.5]);
        let ay = marr_d1!(Y; [0.1, 0.1, 0.8]);

        println!("{:?}", cyz[0].projection(&mbr(&az, &cyz).unwrap()));
        println!("{:?}", cyz[1].projection(&mbr(&az, &cyz).unwrap()));
        println!("{:?}", cyz[0].projection(&ay));
        println!("{:?}", cyz[1].projection(&ay));

        let cyxz: Option<MArrD2<X, Z, SimplexD1<Y, f64>>> =
            MArrD1::<Y, _>::merge_cond2(&cyx, &cyz, &ax, &az, &ay);
        assert!(cyxz.is_some());
        println!("{cyxz:?}");

        let cyzx: Option<MArrD2<Z, X, SimplexD1<Y, f64>>> =
            MArrD1::<Y, _>::merge_cond2(&cyz, &cyx, &az, &ax, &ay);
        assert!(cyzx.is_some());
        println!("{cyxz:?}");
    }
}
