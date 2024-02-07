use num_traits::Float;

use super::{IndexedContainer, Opinion, Opinion1d, Opinion1dRef, Projection};
use std::{
    ops::{AddAssign, DivAssign, Index, IndexMut},
    usize,
};

#[derive(Clone)]
pub struct HigherRange<const N: usize> {
    k: Option<[usize; N]>,
    size: [usize; N],
}

impl<const N: usize> HigherRange<N> {
    fn new(size: [usize; N]) -> Self {
        Self {
            k: if size.iter().all(|s| *s > 0) {
                Some([0; N])
            } else {
                None
            },
            size,
        }
    }
}

impl<const N: usize> Iterator for HigherRange<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        match self.k.take() {
            None => None,
            Some(mut k) => {
                let cur_k = k.clone();
                for i in (0..N).rev() {
                    if k[i] + 1 < self.size[i] {
                        k[i] += 1;
                        self.k = Some(k);
                        break;
                    } else if i > 0 {
                        k[i] = 0;
                    } else {
                        self.k = None;
                    }
                }
                Some(cur_k)
            }
        }
    }
}

pub struct MyIter<'a, S>
where
    &'a S: IntoIterator,
{
    iters: std::slice::Iter<'a, S>,
    iter: Option<<&'a S as IntoIterator>::IntoIter>,
}

impl<'a, S> MyIter<'a, S>
where
    &'a S: IntoIterator,
{
    pub fn new(mut iters: std::slice::Iter<'a, S>) -> Self {
        let iter = iters.next().map(|t| t.into_iter());
        Self { iters, iter }
    }
}

impl<'a, S> Iterator for MyIter<'a, S>
where
    &'a S: IntoIterator,
{
    type Item = <<&'a S as IntoIterator>::IntoIter as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.take() {
            None => None,
            Some(mut iter) => match iter.next() {
                None => {
                    self.iter = self.iters.next().map(|t| t.into_iter());
                    self.iter.as_mut()?.next()
                }
                Some(t) => {
                    self.iter = Some(iter);
                    Some(t)
                }
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HA1<V, const K0: usize>(pub [V; K0]);

#[derive(Clone, Debug, PartialEq)]
pub struct HA2<V, const K0: usize, const K1: usize>(pub [HA1<V, K1>; K0]);

#[derive(Clone, Debug, PartialEq)]
pub struct HA3<V, const K0: usize, const K1: usize, const K2: usize>(pub [HA2<V, K1, K2>; K0]);

macro_rules! index {
    (1) => {
        fn index(&self, [k0]: [usize; 1]) -> &Self::Output {
            &self.0[k0]
        }
    };
    ($n:tt) => {
        fn index(&self, [k0, k @ ..]: [usize; $n]) -> &Self::Output {
            &self.0[k0][k]
        }
    };
}

macro_rules! index_mut {
    (1) => {
        fn index_mut(&mut self, [k0]: [usize; 1]) -> &mut Self::Output {
            &mut self.0[k0]
        }
    };
    ($n:tt) => {
        fn index_mut(&mut self, [k0, k @ ..]: [usize; $n]) -> &mut Self::Output {
            &mut self.0[k0][k]
        }
    };
}

macro_rules! from_fn {
    ($f:ident;1) => {
        HA1(std::array::from_fn(|k0| $f([k0])))
    };
    ($f:ident;2) => {
        HA2(std::array::from_fn(|k0| {
            HA1(std::array::from_fn(|k1| $f([k0, k1])))
        }))
    };
    ($f:ident;3) => {
        HA3(std::array::from_fn(|k0| {
            HA2(std::array::from_fn(|k1| {
                HA1(std::array::from_fn(|k2| $f([k0, k1, k2])))
            }))
        }))
    };
}

macro_rules! impl_ha {
    ($n:tt, $ha:ident[$k:ident$(, $ks:ident)*]) => {
        impl<V, const $k: usize$(, const $ks: usize)*> Index<[usize; $n]> for $ha<V, $k$(, $ks)*> {
            type Output = V;
            index!($n);
        }

        impl<V, const $k: usize$(, const $ks: usize)*> IndexMut<[usize; $n]> for $ha<V, $k$(, $ks)*> {
            index_mut!($n);
        }

        impl<V, const $k: usize$(, const $ks: usize)*> IndexedContainer<[usize; $n]> for $ha<V, $k$(, $ks)*> {
            const SIZE: usize = $k$( * $ks)*;
            type Map<U> = $ha<U, $k$(, $ks)*>;

            fn keys() -> impl Iterator<Item = [usize; $n]> {
                HigherRange::new([$k$(, $ks)*])
            }

            fn map<U, F: FnMut([usize; $n]) -> U>(mut f: F) -> $ha<U, $k$(, $ks)*> {
                from_fn!(f;$n)
            }

            fn from_fn<F: FnMut([usize; $n]) -> Self::Output>(mut f: F) -> Self {
                from_fn!(f;$n)
            }
        }

        impl<V, const $k: usize$(, const $ks: usize)*> $ha<V, $k$(, $ks)*> {
            pub fn len(&self) -> usize {
                Self::SIZE
            }
        }
    };
}

impl_ha!(1, HA1[K0]);
impl_ha!(2, HA2[K0, K1]);
impl_ha!(3, HA3[K0, K1, K2]);

macro_rules! impl_into_iter {
    ($n:tt, $ha:ident[$k:ident$(, $ks:ident)+], $ha0:ident) => {
        impl<'a, V, const $k: usize$(, const $ks: usize)+> IntoIterator for &'a $ha<V, $k$(, $ks)+> {
            type Item = &'a V;
            type IntoIter = MyIter<'a, $ha0<V$(, $ks)+>>;

            fn into_iter(self) -> Self::IntoIter {
                let iters = self.0.as_ref().iter();
                MyIter::new(iters)
            }
        }
    };
}

impl<'a, V, const K0: usize> IntoIterator for &'a HA1<V, K0> {
    type Item = &'a V;
    type IntoIter = std::slice::Iter<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.as_ref().iter()
    }
}

impl_into_iter!(2, HA2[K0, K1], HA1);
impl_into_iter!(3, HA3[K0, K1, K2], HA2);

#[derive(Clone, Debug)]
pub struct HigherArr2<V, const K0: usize, const K1: usize>(pub Box<HA2<V, K0, K1>>);

#[derive(Clone, Debug)]
pub struct HigherArr3<V, const K0: usize, const K1: usize, const K2: usize>(
    pub Box<HA3<V, K0, K1, K2>>,
);

#[macro_export(local_inner_macros)]
macro_rules! ha1 {
    [$($e:expr),*$(,)?] => {
        $crate::mul::prod::HA1([$($e,)*])
    };
    (ext; [$($e:expr),*$(,)?]) => {
        ha1!($($e,)*)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! ha2 {
    [$($e:tt),*$(,)?] => {
        $crate::mul::prod::HA2([$(ha1!(ext; $e),)*])
    };
    (ext; [$($e:tt),*$(,)?]) => {
        ha2!($($e,)*)
    }
}

#[macro_export(local_inner_macros)]
macro_rules! ha3 {
    [$($e:tt),*$(,)?] => {
        $crate::mul::prod::HA3([$(ha2!(ext; $e),)*])
    };
}

#[macro_export(local_inner_macros)]
macro_rules! harr2 {
    [$e:tt$(,$es:tt)*] => {
        $crate::mul::prod::HigherArr2(Box::new(ha2!($e$(,$es)*)))
    };
}

#[macro_export(local_inner_macros)]
macro_rules! harr3 {
    [$e:tt$(,$es:tt)*] => {
        $crate::mul::prod::HigherArr3(Box::new(ha3!($e$(,$es)*)))
    };
}

macro_rules! impl_higher_arr {
    ($n:tt, $ha:ident, $higher_arr:ident [$k:ident$(, $ks:ident)*]) => {
        impl<V, const $k: usize$(, const $ks: usize)*> std::ops::Deref for $higher_arr<V, $k$(, $ks)+> {
            type Target = $ha<V, $k$(, $ks)*>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<V, const $k: usize$(, const $ks: usize)*> Index<[usize; $n]> for $higher_arr<V, $k$(, $ks)*> {
            type Output = V;
            fn index(&self, k: [usize; $n]) -> &Self::Output {
                &self.0[k]
            }
        }

        impl<V, const $k: usize$(, const $ks: usize)*> IndexMut<[usize; $n]> for $higher_arr<V, $k$(, $ks)*> {
            fn index_mut(&mut self, k: [usize; $n]) -> &mut Self::Output {
                &mut self.0[k]
            }
        }

        impl<V, const $k: usize$(, const $ks: usize)*> IndexedContainer<[usize; $n]> for $higher_arr<V, $k$(, $ks)*> {
            const SIZE: usize = $k$( * $ks)*;
            type Map<U> = $higher_arr<U, $k$(, $ks)*>;

            fn keys() -> impl Iterator<Item = [usize; $n]> {
                HigherRange::new([$k$(, $ks)*])
            }

            fn map<U, F: FnMut([usize; $n]) -> U>(mut f: F) -> $higher_arr<U, $k$(, $ks)*> {
                $higher_arr(Box::new(from_fn!(f;$n)))
            }

            fn from_fn<F: FnMut([usize; $n]) -> Self::Output>(mut f: F) -> Self {
                $higher_arr(Box::new(from_fn!(f;$n)))
            }
        }
    };
}

// impl_higher_arr!(1, HA1, HigherArr1[K0]);
impl_higher_arr!(2, HA2, HigherArr2[K0, K1]);
impl_higher_arr!(3, HA3, HigherArr3[K0, K1, K2]);

impl<T, U, const K0: usize> From<[T; K0]> for HA1<U, K0>
where
    U: From<T>,
{
    #[inline]
    fn from(value: [T; K0]) -> Self {
        HA1(value.map(U::from))
    }
}

impl<T, U, const K0: usize, const K1: usize> From<[[T; K1]; K0]> for HA2<U, K0, K1>
where
    U: From<T>,
{
    #[inline]
    fn from(value: [[T; K1]; K0]) -> Self {
        HA2(value.map(HA1::from))
    }
}

impl<T, U, const K0: usize, const K1: usize, const K2: usize> From<[[[T; K2]; K1]; K0]>
    for HA3<U, K0, K1, K2>
where
    U: From<T>,
{
    #[inline]
    fn from(value: [[[T; K2]; K1]; K0]) -> Self {
        HA3(value.map(HA2::from))
    }
}

impl<T, U, const K0: usize, const K1: usize> From<[[T; K1]; K0]> for HigherArr2<U, K0, K1>
where
    U: From<T>,
{
    #[inline]
    fn from(value: [[T; K1]; K0]) -> Self {
        HigherArr2(Box::new(value.into()))
    }
}

impl<T, U, const K0: usize, const K1: usize, const K2: usize> From<[[[T; K2]; K1]; K0]>
    for HigherArr3<U, K0, K1, K2>
where
    U: From<T>,
{
    fn from(value: [[[T; K2]; K1]; K0]) -> Self {
        HigherArr3(Box::new(value.into()))
    }
}

pub trait Product2<T0, T1> {
    fn product2(t0: T0, t1: T1) -> Self;
}

pub trait Product3<T0, T1, T2> {
    fn product3(t0: T0, t1: T1, t2: T2) -> Self;
}

impl<V: Float, const D0: usize, const D1: usize> Product2<&[V; D0], &[V; D1]>
    for HigherArr2<V, D0, D1>
{
    fn product2(w0: &[V; D0], w1: &[V; D1]) -> Self {
        Self::from_fn(|d| w0[d[0]] * w1[d[1]])
    }
}

impl<V: Float, const D0: usize, const D1: usize, const D2: usize>
    Product3<&[V; D0], &[V; D1], &[V; D2]> for HigherArr3<V, D0, D1, D2>
{
    fn product3(w0: &[V; D0], w1: &[V; D1], w2: &[V; D2]) -> Self {
        Self::from_fn(|d| w0[d[0]] * w1[d[1]] * w2[d[2]])
    }
}

impl<'a, V, const D0: usize, const D1: usize> Product2<&Opinion1d<V, D0>, &Opinion1d<V, D1>>
    for Opinion<HigherArr2<V, D0, D1>, V>
where
    V: Float + AddAssign + DivAssign,
{
    fn product2(w0: &Opinion1d<V, D0>, w1: &Opinion1d<V, D1>) -> Self {
        Product2::product2(w0.as_ref(), w1.as_ref())
    }
}

impl<'a, V, const D0: usize, const D1: usize>
    Product2<Opinion1dRef<'a, V, D0>, Opinion1dRef<'a, V, D1>> for Opinion<HigherArr2<V, D0, D1>, V>
where
    V: Float + AddAssign + DivAssign,
{
    fn product2(w0: Opinion1dRef<V, D0>, w1: Opinion1dRef<V, D1>) -> Self {
        let p = HigherArr2::product2(&w0.projection(), &w1.projection());
        let a = HigherArr2::from_fn(|d| w0.base_rate[d[0]] * w1.base_rate[d[1]]);
        let u = HigherArr2::<V, D0, D1>::keys()
            .map(|d| (p[d] - w0.b()[d[0]] * w1.b()[d[1]]) / a[d])
            .reduce(<V>::min)
            .unwrap();
        let b = HigherArr2::from_fn(|d| p[d] - a[d] * u);
        Opinion::new_unchecked(b, u, a)
    }
}

impl<'a, V, const D0: usize, const D1: usize, const D2: usize>
    Product3<Opinion1dRef<'a, V, D0>, Opinion1dRef<'a, V, D1>, Opinion1dRef<'a, V, D2>>
    for Opinion<HigherArr3<V, D0, D1, D2>, V>
where
    V: Float + AddAssign + DivAssign,
{
    fn product3(w0: Opinion1dRef<V, D0>, w1: Opinion1dRef<V, D1>, w2: Opinion1dRef<V, D2>) -> Self {
        let p = HigherArr3::product3(&w0.projection(), &w1.projection(), &w2.projection());
        let a =
            HigherArr3::from_fn(|d| w0.base_rate[d[0]] * w1.base_rate[d[1]] * w2.base_rate[d[2]]);
        let u = HigherArr3::<V, D0, D1, D2>::keys()
            .map(|d| (p[d] - w0.b()[d[0]] * w1.b()[d[1]] * w2.b()[d[2]]) / a[d])
            .reduce(<V>::min)
            .unwrap();
        let b = HigherArr3::from_fn(|d| p[d] - a[d] * u);
        Opinion::new_unchecked(b, u, a)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::{HigherArr2, HigherArr3, HigherRange, Product2, Projection};
    use crate::mul::{op::Deduction, IndexedContainer, Opinion, Opinion1d, Simplex};

    fn higher_range_checker2(size: [usize; 2]) {
        let mut r = HigherRange::new(size);
        for i0 in 0..size[0] {
            for i1 in 0..size[1] {
                assert!(r.next().unwrap() == [i0, i1]);
            }
        }
        assert!(r.next().is_none());
    }

    fn higher_range_checker3(size: [usize; 3]) {
        let mut r = HigherRange::new(size);
        for i0 in 0..size[0] {
            for i1 in 0..size[1] {
                for i2 in 0..size[2] {
                    assert!(r.next().unwrap() == [i0, i1, i2]);
                }
            }
        }
        assert!(r.next().is_none());
    }

    #[test]
    fn test_higher_range2() {
        higher_range_checker2([3, 4]);
        higher_range_checker2([4, 3]);
        higher_range_checker2([0, 4]);
        higher_range_checker2([3, 0]);
        higher_range_checker2([0, 0]);
    }

    #[test]
    fn test_higher_range3() {
        higher_range_checker3([2, 3, 4]);
        higher_range_checker3([2, 4, 3]);
        higher_range_checker3([3, 2, 4]);
        higher_range_checker3([3, 4, 2]);
        higher_range_checker3([4, 2, 3]);
        higher_range_checker3([4, 3, 2]);
        higher_range_checker3([0, 3, 4]);
        higher_range_checker3([2, 0, 4]);
        higher_range_checker3([2, 3, 0]);
        higher_range_checker3([0, 0, 4]);
        higher_range_checker3([0, 3, 0]);
        higher_range_checker3([2, 0, 0]);
        higher_range_checker3([0, 0, 0]);
    }

    #[test]
    fn ha_into() {
        let h = harr2![[0, 1, 2], [2, 3, 4]];
        let arr = [[0, 1, 2], [2, 3, 4]];
        let g: HigherArr2<i32, 2, 3> = arr.into();
        assert_eq!(h.0, g.0);

        let h = harr3![[[0], [1]], [[2], [2]], [[3], [4]]];
        let arr = [[[0], [1]], [[2], [2]], [[3], [4]]];
        let g: HigherArr3<i32, 3, 2, 1> = arr.into();
        assert_eq!(h.0, g.0);
    }

    #[test]
    fn ha_macro() {
        let h = ha1![std::vec![0], std::vec![1], std::vec![1]];
        let h2 = ha2![[std::vec![0], std::vec![1]], [std::vec![1], std::vec![1]]];
        let h3 = ha3![
            [[std::vec![0], std::vec![1]]],
            [[std::vec![1], std::vec![1]]]
        ];
        assert_eq!(h.len(), 3);
        assert_eq!(h2.len(), 4);
        assert_eq!(h3.len(), 4);
    }

    #[test]
    fn test_iter2() {
        let c: HigherArr2<_, 2, 3> = HigherArr2::from_fn(|[i, j]| (i, j));
        let mut iter = (&c).into_iter();
        for i in 0..2 {
            for j in 0..3 {
                assert!(iter.next() == Some(&(i, j)));
            }
        }
    }

    #[test]
    fn test_iter3() {
        let c: HigherArr3<_, 2, 3, 4> = HigherArr3::from_fn(|[i, j, k]| (i, j, k));
        let mut iter = (&c).into_iter();
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert!(iter.next() == Some(&(i, j, k)));
                }
            }
        }
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
                let p01 = HigherArr2::<_, 2, 3>::product2(&w0.projection(), &w1.projection());
                println!("{:?}", w01);
                println!("{:?}, {}", p, p.into_iter().sum::<$ft>());
                println!("{:?}, {}", p01, p01.into_iter().sum::<$ft>());
                for d in HigherRange::new([2, 3]) {
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
                let conds = harr2![
                    [
                        Simplex::<$ft, 3>::new([0.0, 0.8, 0.1], 0.1),
                        Simplex::<$ft, 3>::new([0.0, 0.8, 0.1], 0.1),
                    ],
                    [
                        Simplex::<$ft, 3>::new([0.7, 0.0, 0.1], 0.2),
                        Simplex::<$ft, 3>::new([0.7, 0.0, 0.1], 0.2),
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
}
