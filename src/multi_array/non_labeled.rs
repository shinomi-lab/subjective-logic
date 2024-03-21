use std::{
    array,
    ops::{Index, IndexMut, Mul},
    usize,
};

use num_traits::Zero;

use crate::ops::{Container, ContainerMap, FromFn, Indexes, Product2, Product3, Zeros};

#[derive(Clone)]
pub struct MultiRange<const N: usize> {
    k: Option<[usize; N]>,
    size: [usize; N],
}

impl<const N: usize> MultiRange<N> {
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

impl<const N: usize> Iterator for MultiRange<N> {
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

pub struct Iter<'a, S>
where
    &'a S: IntoIterator,
{
    iters: std::slice::Iter<'a, S>,
    iter: Option<<&'a S as IntoIterator>::IntoIter>,
}

impl<'a, S> Iter<'a, S>
where
    &'a S: IntoIterator,
{
    pub fn new(mut iters: std::slice::Iter<'a, S>) -> Self {
        let iter = iters.next().map(|t| t.into_iter());
        Self { iters, iter }
    }
}

impl<'a, S> Iterator for Iter<'a, S>
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

/// `Vec` is expected to have length `K0`.
#[derive(Clone, Debug, PartialEq)]
pub struct MArr1<V, const K0: usize>(Vec<V>);

impl<V, const K0: usize> MArr1<V, K0> {
    pub fn new(arr: [V; K0]) -> Self {
        MArr1(Vec::from(arr))
    }
}

impl<V, const K0: usize> Default for MArr1<V, K0>
where
    [V; K0]: Default,
{
    fn default() -> Self {
        Self::new(Default::default())
    }
}

/// `Vec` is expected to have length `K1`.
#[derive(Clone, Debug, PartialEq)]
pub struct MArr2<V, const K0: usize, const K1: usize>(pub(crate) Vec<MArr1<V, K1>>);

impl<V, const K0: usize, const K1: usize> MArr2<V, K0, K1> {
    pub fn new(arr: [MArr1<V, K1>; K0]) -> Self {
        MArr2(Vec::from(arr))
    }
}

impl<V, const K0: usize, const K1: usize> Default for MArr2<V, K0, K1>
where
    [MArr1<V, K1>; K0]: Default,
{
    fn default() -> Self {
        Self::new(Default::default())
    }
}

/// `Vec` is expected to have length `K2`.
#[derive(Clone, Debug, PartialEq)]
pub struct MArr3<V, const K0: usize, const K1: usize, const K2: usize>(
    pub(crate) Vec<MArr2<V, K1, K2>>,
);

impl<V, const K0: usize, const K1: usize, const K2: usize> MArr3<V, K0, K1, K2> {
    pub fn new(arr: [MArr2<V, K1, K2>; K0]) -> Self {
        MArr3(Vec::from(arr))
    }
}

impl<V, const K0: usize, const K1: usize, const K2: usize> Default for MArr3<V, K0, K1, K2>
where
    [MArr2<V, K1, K2>; K0]: Default,
{
    fn default() -> Self {
        Self::new(Default::default())
    }
}

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

impl<V, const K0: usize> FromIterator<V> for MArr1<V, K0> {
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl<V, const K0: usize, const K1: usize> FromIterator<V> for MArr2<V, K0, K1> {
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let mut i0 = Vec::with_capacity(K0);
        for _ in 0..K0 {
            let mut i1 = Vec::with_capacity(K1);
            for _ in 0..K1 {
                i1.push(iter.next().unwrap());
            }
            i0.push(MArr1(i1));
        }
        Self(i0)
    }
}

impl<V, const K0: usize, const K1: usize, const K2: usize> FromIterator<V>
    for MArr3<V, K0, K1, K2>
{
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let mut i0 = Vec::with_capacity(K0);
        for _ in 0..K0 {
            let mut i1 = Vec::with_capacity(K1);
            for _ in 0..K1 {
                let mut i2 = Vec::with_capacity(K2);
                for _ in 0..K2 {
                    i2.push(iter.next().unwrap());
                }
                i1.push(MArr1(i2));
            }
            i0.push(MArr2(i1));
        }
        Self(i0)
    }
}

impl<V: Zero, const K0: usize> Zeros for MArr1<V, K0> {
    fn zeros() -> Self {
        MArr1::new(array::from_fn(|_| V::zero()))
    }
}

impl<V: Zero, const K0: usize, const K1: usize> Zeros for MArr2<V, K0, K1> {
    fn zeros() -> Self {
        MArr2::new(array::from_fn(|_| MArr1::zeros()))
    }
}

impl<V: Zero, const K0: usize, const K1: usize, const K2: usize> Zeros for MArr3<V, K0, K1, K2> {
    fn zeros() -> Self {
        MArr3::new(array::from_fn(|_| MArr2::zeros()))
    }
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

        impl<V, const $k: usize$(, const $ks: usize)*> Indexes<[usize; $n]> for $ha<V, $k$(, $ks)*> {
            type Iter = MultiRange<$n>;

            fn indexes() -> Self::Iter {
                MultiRange::new([$k$(, $ks)*])
            }
        }

        impl<V, const $k: usize$(, const $ks: usize)*> FromFn<[usize; $n], V> for $ha<V, $k$(, $ks)*> {
            fn from_fn<F: FnMut([usize; $n]) -> V>(f: F) -> Self {
                Self::from_iter(Self::indexes().map(f))
            }
        }

        impl<V, const $k: usize$(, const $ks: usize)*> Container<[usize; $n]> for $ha<V, $k$(, $ks)*> {}

        impl<V, const $k: usize$(, const $ks: usize)*> ContainerMap<[usize; $n]> for $ha<V, $k$(, $ks)*> {
            type Map<U> = $ha<U, $k$(, $ks)*>;
        }

        impl<V, const $k: usize$(, const $ks: usize)*> $ha<V, $k$(, $ks)*> {
            pub fn len(&self) -> usize {
                $k$( * $ks)*
            }
        }
    };
}

impl_ha!(1, MArr1[K0]);
impl_ha!(2, MArr2[K0, K1]);
impl_ha!(3, MArr3[K0, K1, K2]);

macro_rules! impl_into_iter {
    ($n:tt, $ha:ident[$k:ident$(, $ks:ident)+], $ha0:ident) => {
        impl<'a, V, const $k: usize$(, const $ks: usize)+> IntoIterator for &'a $ha<V, $k$(, $ks)+> {
            type Item = &'a V;
            type IntoIter = Iter<'a, $ha0<V$(, $ks)+>>;

            fn into_iter(self) -> Self::IntoIter {
                let iters = self.0.iter();
                Iter::new(iters)
            }
        }
    };
}

impl<'a, V, const K0: usize> IntoIterator for &'a MArr1<V, K0> {
    type Item = &'a V;
    type IntoIter = std::slice::Iter<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl_into_iter!(2, MArr2[K0, K1], MArr1);
impl_into_iter!(3, MArr3[K0, K1, K2], MArr2);

#[macro_export(local_inner_macros)]
macro_rules! marr1 {
    [$($e:expr),*$(,)?] => {
        $crate::multi_array::non_labeled::MArr1::new([$($e,)*])
    };
    (ext; [$($e:expr),*$(,)?]) => {
        marr1!($($e,)*)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! marr2 {
    [$($e:tt),*$(,)?] => {
        $crate::multi_array::non_labeled::MArr2::new([$(marr1!(ext; $e),)*])
    };
    (ext; [$($e:tt),*$(,)?]) => {
        marr2!($($e,)*)
    }
}

#[macro_export(local_inner_macros)]
macro_rules! marr3 {
    [$($e:tt),*$(,)?] => {
        $crate::multi_array::non_labeled::MArr3::new([$(marr2!(ext; $e),)*])
    };
}

impl<T, U, const K0: usize> TryFrom<[T; K0]> for MArr1<U, K0>
where
    U: TryFrom<T>,
{
    type Error = U::Error;

    #[inline]
    fn try_from(value: [T; K0]) -> Result<Self, Self::Error> {
        Ok(MArr1(
            value
                .into_iter()
                .map(U::try_from)
                .collect::<Result<_, _>>()?,
        ))
    }
}

impl<T, U, const K0: usize, const K1: usize> TryFrom<[[T; K1]; K0]> for MArr2<U, K0, K1>
where
    MArr1<U, K1>: TryFrom<[T; K1]>,
{
    type Error = <MArr1<U, K1> as TryFrom<[T; K1]>>::Error;

    #[inline]
    fn try_from(value: [[T; K1]; K0]) -> Result<Self, Self::Error> {
        Ok(MArr2(
            value
                .into_iter()
                .map(MArr1::try_from)
                .collect::<Result<_, _>>()?,
        ))
    }
}

impl<T, U, const K0: usize, const K1: usize, const K2: usize> TryFrom<[[[T; K2]; K1]; K0]>
    for MArr3<U, K0, K1, K2>
where
    MArr2<U, K1, K2>: TryFrom<[[T; K2]; K1]>,
{
    type Error = <MArr2<U, K1, K2> as TryFrom<[[T; K2]; K1]>>::Error;

    #[inline]
    fn try_from(value: [[[T; K2]; K1]; K0]) -> Result<Self, Self::Error> {
        Ok(MArr3(
            value
                .into_iter()
                .map(MArr2::try_from)
                .collect::<Result<_, _>>()?,
        ))
    }
}

impl<V: Mul<Output = V> + Copy, const D0: usize, const D1: usize> Product2<&[V; D0], &[V; D1]>
    for MArr2<V, D0, D1>
{
    fn product2(w0: &[V; D0], w1: &[V; D1]) -> Self {
        Self::from_fn(|d| w0[d[0]] * w1[d[1]])
    }
}

impl<V: Mul<Output = V> + Copy, const D0: usize, const D1: usize, const D2: usize>
    Product3<&[V; D0], &[V; D1], &[V; D2]> for MArr3<V, D0, D1, D2>
{
    fn product3(w0: &[V; D0], w1: &[V; D1], w2: &[V; D2]) -> Self {
        Self::from_fn(|d| w0[d[0]] * w1[d[1]] * w2[d[2]])
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mul::non_labeled::Simplex1d,
        multi_array::non_labeled::{MArr2, MArr3, MultiRange},
        ops::FromFn,
    };

    use super::MArr1;

    fn multi_range_checker2(size: [usize; 2]) {
        let mut r = MultiRange::new(size);
        for i0 in 0..size[0] {
            for i1 in 0..size[1] {
                assert!(r.next().unwrap() == [i0, i1]);
            }
        }
        assert!(r.next().is_none());
    }

    fn multi_range_checker3(size: [usize; 3]) {
        let mut r = MultiRange::new(size);
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
    fn test_multi_range2() {
        multi_range_checker2([3, 4]);
        multi_range_checker2([4, 3]);
        multi_range_checker2([0, 4]);
        multi_range_checker2([3, 0]);
        multi_range_checker2([0, 0]);
    }

    #[test]
    fn test_multi_range3() {
        multi_range_checker3([2, 3, 4]);
        multi_range_checker3([2, 4, 3]);
        multi_range_checker3([3, 2, 4]);
        multi_range_checker3([3, 4, 2]);
        multi_range_checker3([4, 2, 3]);
        multi_range_checker3([4, 3, 2]);
        multi_range_checker3([0, 3, 4]);
        multi_range_checker3([2, 0, 4]);
        multi_range_checker3([2, 3, 0]);
        multi_range_checker3([0, 0, 4]);
        multi_range_checker3([0, 3, 0]);
        multi_range_checker3([2, 0, 0]);
        multi_range_checker3([0, 0, 0]);
    }

    #[test]
    fn ma_try_into() {
        let h = marr2![[0, 1, 2], [2, 3, 4]];
        let arr = [[0, 1, 2], [2, 3, 4]];
        let g: MArr2<i32, 2, 3> = arr.try_into().unwrap();
        assert_eq!(h, g);

        let h = marr3![[[0], [1]], [[2], [2]], [[3], [4]]];
        let arr = [[[0], [1]], [[2], [2]], [[3], [4]]];
        let g: MArr3<i32, 3, 2, 1> = arr.try_into().unwrap();
        assert_eq!(h, g);

        let h = MArr2::<u32, 1, 2>::try_from([[0i32, 0i32]]);
        assert!(h.is_ok());
        let g = MArr2::<u32, 1, 2>::try_from([[0i32, -1i32]]);
        println!("{:?}", g);
        assert!(g.is_err());

        let cond = MArr2::<Simplex1d<f32, 2>, 1, 1>::try_from([[([0.0, 0.0], 0.0)]]);
        println!("{:?}", cond);
        assert!(cond.is_err());
    }

    #[test]
    fn ha_default() {
        let h = MArr2::<i32, 1, 2>::default();
        assert_eq!(h, marr2![[0, 0]]);
        let h = MArr3::<i32, 1, 2, 3>::default();
        assert_eq!(h, marr3![[[0, 0, 0], [0, 0, 0]]]);
    }

    #[test]
    fn ha_macro() {
        let h = marr1![std::vec![0], std::vec![1], std::vec![1]];
        let h2 = marr2![[std::vec![0], std::vec![1]], [std::vec![1], std::vec![1]]];
        let h3 = marr3![
            [[std::vec![0], std::vec![1]]],
            [[std::vec![1], std::vec![1]]]
        ];
        assert_eq!(h.len(), 3);
        assert_eq!(h2.len(), 4);
        assert_eq!(h3.len(), 4);
    }

    #[test]
    fn test_iter2() {
        let c = MArr2::<_, 2, 3>::from_fn(|[i, j]| (i, j));
        let mut iter = (&c).into_iter();
        for i in 0..2 {
            for j in 0..3 {
                assert!(iter.next() == Some(&(i, j)));
            }
        }
    }

    #[test]
    fn test_iter3() {
        let c = MArr3::<_, 2, 3, 4>::from_fn(|[i, j, k]| (i, j, k));
        let mut iter = (&c).into_iter();
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let a = iter.next();
                    println!("{a:?}");
                    assert!(a == Some(&(i, j, k)));
                }
            }
        }
    }

    #[test]
    fn test_from_iter() {
        let a = MArr1::<_, 2>::from_iter(0..2);
        for (i, v) in a.into_iter().enumerate() {
            assert_eq!(i, *v);
        }

        let a = MArr2::<_, 2, 3>::from_iter(0..6);
        for (i, v) in a.into_iter().enumerate() {
            assert_eq!(i, *v);
        }

        let a = MArr3::<_, 2, 3, 4>::from_iter(0..24);
        for (i, v) in a.into_iter().enumerate() {
            assert_eq!(i, *v);
        }
    }
}
