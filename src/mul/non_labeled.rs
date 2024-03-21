use std::{
    array, fmt,
    ops::{AddAssign, DivAssign, Range},
    usize,
};

use approx::UlpsEq;
use num_traits::{Float, Zero};

use super::{check_base_rate, Opinion, OpinionRef, Simplex};
use crate::ops::{Container, ContainerMap, FromFn, Indexes, Product2, Product3, Projection, Zeros};
use crate::{
    errors::InvalidValueError,
    multi_array::non_labeled::{MArr2, MArr3},
};

impl<V: Zero, const N: usize> Zeros for [V; N] {
    fn zeros() -> Self {
        array::from_fn(|_| V::zero())
    }
}

impl<T, const N: usize> Indexes<usize> for [T; N] {
    type Iter = Range<usize>;

    fn indexes() -> Self::Iter {
        0..N
    }
}

impl<T, const N: usize> Container<usize> for [T; N] {}

impl<T, const N: usize> ContainerMap<usize> for [T; N] {
    type Map<U> = [U; N];
}

impl<V, const N: usize> FromFn<usize, V> for [V; N] {
    fn from_fn<F: FnMut(usize) -> V>(f: F) -> Self {
        array::from_fn(f)
    }
}

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
        let u = MArr2::<V, D0, D1>::indexes()
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
        let u = MArr3::<V, D0, D1, D2>::indexes()
            .map(|d| (p[d] - w0.b()[d[0]] * w1.b()[d[1]] * w2.b()[d[2]]) / a[d])
            .reduce(<V>::min)
            .unwrap();
        let b = MArr3::from_fn(|d| p[d] - a[d] * u);
        Opinion::new(b, u, a)
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use approx::{assert_ulps_eq, ulps_eq};

    use super::{Opinion1d, Simplex1d};
    use crate::{
        marr1, marr2,
        mul::{check_base_rate, mbr, Indexes, Opinion, OpinionRef, Simplex},
        multi_array::non_labeled::MArr2,
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

    macro_rules! nfract {
        [$ft:ty, $n:expr] => {
            |v: $ft| (v * <$ft>::powi(10.0, $n)).fract()
        };
    }

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
                for d in MArr2::<$ft, 2, 3>::indexes() {
                    println!("{}", (p[d] - w01.b()[d]) / w01.base_rate[d]);
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
    fn test_deduction2() {
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
