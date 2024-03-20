use std::{
    cmp, fmt,
    marker::PhantomData,
    ops::{Index, IndexMut, Mul, Range},
    slice,
};

use itertools::iproduct;

use crate::ops::{IndexedContainer, Keys, Product2, Product3};

pub trait Domain {
    const LEN: Self::Idx;
    type Idx: Into<usize> + Clone + fmt::Debug;
}

impl<D: Domain<Idx = usize>> Keys<usize> for D {
    type Iter = Range<D::Idx>;

    fn keys() -> Self::Iter {
        0..D::LEN
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

pub struct IterMut<'a, S>
where
    &'a mut S: IntoIterator,
{
    iters: std::slice::IterMut<'a, S>,
    iter: Option<<&'a mut S as IntoIterator>::IntoIter>,
}

impl<'a, S> IterMut<'a, S>
where
    &'a mut S: IntoIterator,
{
    pub fn new(mut iters: std::slice::IterMut<'a, S>) -> Self {
        let iter = iters.next().map(|t| t.into_iter());
        Self { iters, iter }
    }
}

impl<'a, S> Iterator for IterMut<'a, S>
where
    &'a mut S: IntoIterator,
{
    type Item = <<&'a mut S as IntoIterator>::IntoIter as Iterator>::Item;

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

// pub struct IntoIter<S>
// where
//     S: IntoIterator,
// {
//     iters: std::vec::IntoIter<S>,
//     iter: Option<<S as IntoIterator>::IntoIter>,
// }

// impl<S> Iterator for IntoIter<S>
// where
//     S: IntoIterator,
// {
//     type Item = <<S as IntoIterator>::IntoIter as Iterator>::Item;

//     fn next(&mut self) -> Option<Self::Item> {
//         match self.iter.take() {
//             None => None,
//             Some(mut iter) => match iter.next() {
//                 None => {
//                     self.iter = self.iters.next().map(|t| t.into_iter());
//                     self.iter.as_mut()?.next()
//                 }
//                 Some(t) => {
//                     self.iter = Some(iter);
//                     Some(t)
//                 }
//             },
//         }
//     }
// }

#[derive(Clone)]
pub struct MArrD1<D0: Domain, V> {
    _marker: PhantomData<D0>,
    inner: Vec<V>,
}

impl<D0: Domain, V: fmt::Debug> fmt::Debug for MArrD1<D0, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:?}", tynm::type_name::<D0>(), self.inner)
    }
}

impl<D0, V> cmp::PartialEq for MArrD1<D0, V>
where
    D0: Domain,
    V: fmt::Debug + cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D0, V> Default for MArrD1<D0, V>
where
    D0: Domain,
    V: Default,
{
    fn default() -> Self {
        let mut inner = Vec::with_capacity(D0::LEN.into());
        inner.resize_with(D0::LEN.into(), V::default);
        Self::new(inner)
    }
}

impl<D0: Domain, V> Index<D0::Idx> for MArrD1<D0, V> {
    type Output = V;

    fn index(&self, index: D0::Idx) -> &Self::Output {
        &self.inner[index.into()]
    }
}

impl<D0: Domain, V> IndexMut<D0::Idx> for MArrD1<D0, V> {
    fn index_mut(&mut self, index: D0::Idx) -> &mut Self::Output {
        &mut self.inner[index.into()]
    }
}

impl<D0, U, V> TryFrom<Vec<U>> for MArrD1<D0, V>
where
    D0: Domain,
    V: TryFrom<U>,
{
    type Error = V::Error;
    fn try_from(value: Vec<U>) -> Result<Self, Self::Error> {
        Ok(Self::new(
            value
                .into_iter()
                .map(V::try_from)
                .collect::<Result<_, _>>()?,
        ))
    }
}

impl<'a, D0, V> IntoIterator for &'a MArrD1<D0, V>
where
    D0: Domain,
{
    type Item = &'a V;
    type IntoIter = slice::Iter<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'a, D0, V> IntoIterator for &'a mut MArrD1<D0, V>
where
    D0: Domain,
{
    type Item = &'a mut V;
    type IntoIter = slice::IterMut<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

impl<D0, V> FromIterator<V> for MArrD1<D0, V>
where
    D0: Domain,
{
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        Self::new(Vec::from_iter(iter))
    }
}

impl<D0: Domain<Idx = usize>, V> IndexedContainer<D0::Idx> for MArrD1<D0, V> {
    type Map<U> = MArrD1<D0, U>;

    fn keys() -> impl Iterator<Item = D0::Idx> {
        D0::keys()
    }

    fn map<U, F: FnMut(D0::Idx) -> U>(f: F) -> Self::Map<U> {
        Self::Map::<U>::from_iter(D0::keys().map(f))
    }

    fn from_fn<F: FnMut(D0::Idx) -> <Self as Index<D0::Idx>>::Output>(f: F) -> Self {
        Self::from_iter(D0::keys().map(f))
    }
}

impl<D0, V> MArrD1<D0, V>
where
    D0: Domain,
{
    pub fn new(inner: Vec<V>) -> Self {
        assert!(inner.len() == D0::LEN.into());
        Self {
            _marker: PhantomData,
            inner,
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (D0::Idx, &'_ V)>
    where
        D0: Keys<D0::Idx>,
    {
        self.indexes().zip(self.values())
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (D0::Idx, &'_ mut V)>
    where
        D0: Keys<D0::Idx>,
    {
        self.indexes().zip(self.values_mut())
    }

    #[inline]
    pub fn indexes(&self) -> impl Iterator<Item = D0::Idx>
    where
        D0: Keys<D0::Idx>,
    {
        D0::keys()
    }

    #[inline]
    pub fn values(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    #[inline]
    pub fn values_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

#[derive(Clone)]
pub struct MArrD2<D0: Domain, D1: Domain, V> {
    inner: MArrD1<D0, MArrD1<D1, V>>,
}

impl<D0: Domain, D1: Domain, V: fmt::Debug> fmt::Debug for MArrD2<D0, D1, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

impl<D0, D1, V> cmp::PartialEq for MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
    V: fmt::Debug + cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D0, D1, V> Default for MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
    V: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<D0: Domain, D1: Domain, V> Index<(D0::Idx, D1::Idx)> for MArrD2<D0, D1, V> {
    type Output = V;

    fn index(&self, index: (D0::Idx, D1::Idx)) -> &Self::Output {
        &self.inner[index.0][index.1]
    }
}

impl<D0: Domain, D1: Domain, V> IndexMut<(D0::Idx, D1::Idx)> for MArrD2<D0, D1, V> {
    fn index_mut(&mut self, index: (D0::Idx, D1::Idx)) -> &mut Self::Output {
        &mut self.inner[index.0][index.1]
    }
}

impl<D0, D1, U, V> TryFrom<Vec<Vec<U>>> for MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
    V: TryFrom<U>,
{
    type Error = V::Error;
    fn try_from(value: Vec<Vec<U>>) -> Result<Self, Self::Error> {
        let arr = value
            .into_iter()
            .map(MArrD1::<D1, V>::try_from)
            .collect::<Result<_, _>>()?;
        Ok(Self::new(arr))
    }
}

impl<'a, D0, D1, V> IntoIterator for &'a MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
{
    type Item = &'a V;
    type IntoIter = Iter<'a, MArrD1<D1, V>>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.inner.into_iter())
    }
}

impl<'a, D0, D1, V> IntoIterator for &'a mut MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
{
    type Item = &'a mut V;
    type IntoIter = IterMut<'a, MArrD1<D1, V>>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut::new((&mut self.inner).into_iter())
    }
}

impl<D0, D1, V> FromIterator<V> for MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
{
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        let mut inner = Vec::with_capacity(D0::LEN.into());
        let mut v = Vec::from_iter(iter);
        for _ in 0..D0::LEN.into() {
            inner.push(MArrD1::<D1, _>::from_iter(v.drain(0..D1::LEN.into())));
        }
        Self::new(inner)
    }
}

impl<D0, D1, V> IndexedContainer<(D0::Idx, D1::Idx)> for MArrD2<D0, D1, V>
where
    D0: Domain<Idx = usize>,
    D1: Domain<Idx = usize>,
{
    type Map<U> = MArrD2<D0, D1, U>;

    fn keys() -> impl Iterator<Item = (D0::Idx, D1::Idx)> {
        iproduct!(D0::keys(), D1::keys())
    }

    fn map<U, F: FnMut((D0::Idx, D1::Idx)) -> U>(f: F) -> Self::Map<U> {
        Self::Map::<U>::from_iter(Self::keys().map(f))
    }

    fn from_fn<F: FnMut((D0::Idx, D1::Idx)) -> <Self as Index<(D0::Idx, D1::Idx)>>::Output>(
        f: F,
    ) -> Self {
        Self::from_iter(Self::keys().map(f))
    }
}

impl<D0, D1, V> MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
{
    pub fn new(arr: Vec<MArrD1<D1, V>>) -> Self {
        Self {
            inner: MArrD1::new(arr),
        }
    }

    pub fn from_multi_iter<I0, I1>(iter: I0) -> Self
    where
        I0: IntoIterator<Item = I1>,
        I1: IntoIterator<Item = V>,
    {
        Self {
            inner: MArrD1::from_iter(iter.into_iter().map(|vs| MArrD1::from_iter(vs))),
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ((D0::Idx, D1::Idx), &'_ V)>
    where
        D0: Keys<D0::Idx>,
        D1: Keys<D1::Idx>,
    {
        self.indexes().zip(self.values())
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = ((D0::Idx, D1::Idx), &'_ mut V)>
    where
        D0: Keys<D0::Idx>,
        D1: Keys<D1::Idx>,
    {
        self.indexes().zip(self.values_mut())
    }

    #[inline]
    pub fn indexes(&self) -> impl Iterator<Item = (D0::Idx, D1::Idx)>
    where
        D0: Keys<D0::Idx>,
        D1: Keys<D1::Idx>,
    {
        iproduct!(D0::keys(), D1::keys())
    }

    #[inline]
    pub fn values(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    #[inline]
    pub fn values_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

#[derive(Clone)]
pub struct MArrD3<D0: Domain, D1: Domain, D2: Domain, V> {
    inner: MArrD1<D0, MArrD2<D1, D2, V>>,
}

impl<D0: Domain, D1: Domain, D2: Domain, V: fmt::Debug> fmt::Debug for MArrD3<D0, D1, D2, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

impl<D0, D1, D2, V> cmp::PartialEq for MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
    V: fmt::Debug + cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D0, D1, D2, V> Default for MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
    V: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<D0, D1, D2, V> Index<(D0::Idx, D1::Idx, D2::Idx)> for MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
{
    type Output = V;

    fn index(&self, index: (D0::Idx, D1::Idx, D2::Idx)) -> &Self::Output {
        &self.inner[index.0][(index.1, index.2)]
    }
}

impl<D0, D1, D2, V> IndexMut<(D0::Idx, D1::Idx, D2::Idx)> for MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
{
    fn index_mut(&mut self, index: (D0::Idx, D1::Idx, D2::Idx)) -> &mut Self::Output {
        &mut self.inner[index.0][(index.1, index.2)]
    }
}

impl<D0, D1, D2, U, V> TryFrom<Vec<Vec<Vec<U>>>> for MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
    V: TryFrom<U>,
{
    type Error = V::Error;
    fn try_from(value: Vec<Vec<Vec<U>>>) -> Result<Self, Self::Error> {
        let arr = value
            .into_iter()
            .map(MArrD2::<D1, D2, V>::try_from)
            .collect::<Result<_, _>>()?;
        Ok(Self::new(arr))
    }
}

impl<'a, D0, D1, D2, V> IntoIterator for &'a MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
{
    type Item = &'a V;
    type IntoIter = Iter<'a, MArrD2<D1, D2, V>>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.inner.into_iter())
    }
}

impl<'a, D0, D1, D2, V> IntoIterator for &'a mut MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
{
    type Item = &'a mut V;
    type IntoIter = IterMut<'a, MArrD2<D1, D2, V>>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut::new((&mut self.inner).into_iter())
    }
}

impl<D0, D1, D2, V> FromIterator<V> for MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
{
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        let mut arr = Vec::with_capacity(D0::LEN.into());
        let mut v = Vec::from_iter(iter);
        for _ in 0..D0::LEN.into() {
            let mut arr2 = Vec::with_capacity(D1::LEN.into());
            for _ in 0..D1::LEN.into() {
                arr2.push(MArrD1::<D2, _>::from_iter(v.drain(0..D2::LEN.into())));
            }
            arr.push(MArrD2::<D1, D2, _>::new(arr2));
        }
        Self::new(arr)
    }
}

impl<D0, D1, D2, V> IndexedContainer<(D0::Idx, D1::Idx, D2::Idx)> for MArrD3<D0, D1, D2, V>
where
    D0: Domain<Idx = usize>,
    D1: Domain<Idx = usize>,
    D2: Domain<Idx = usize>,
{
    type Map<U> = MArrD3<D0, D1, D2, U>;

    fn keys() -> impl Iterator<Item = (D0::Idx, D1::Idx, D2::Idx)> {
        iproduct!(D0::keys(), D1::keys(), D2::keys())
    }

    fn map<U, F: FnMut((D0::Idx, D1::Idx, D2::Idx)) -> U>(f: F) -> Self::Map<U> {
        Self::Map::<U>::from_iter(Self::keys().map(f))
    }

    fn from_fn<
        F: FnMut((D0::Idx, D1::Idx, D2::Idx)) -> <Self as Index<(D0::Idx, D1::Idx, D2::Idx)>>::Output,
    >(
        f: F,
    ) -> Self {
        Self::from_iter(Self::keys().map(f))
    }
}

impl<D0, D1, D2, V> MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
{
    pub fn new(arr: Vec<MArrD2<D1, D2, V>>) -> Self {
        Self {
            inner: MArrD1::new(arr),
        }
    }

    pub fn from_multi_iter<I0, I1, I2>(iter: I0) -> Self
    where
        I0: IntoIterator<Item = I1>,
        I1: IntoIterator<Item = I2>,
        I2: IntoIterator<Item = V>,
    {
        Self {
            inner: MArrD1::from_iter(iter.into_iter().map(|vs| MArrD2::from_multi_iter(vs))),
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ((D0::Idx, D1::Idx, D2::Idx), &'_ V)>
    where
        D0: Keys<D0::Idx>,
        D1: Keys<D1::Idx>,
        D2: Keys<D2::Idx>,
    {
        self.indexes().zip(self.values())
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = ((D0::Idx, D1::Idx, D2::Idx), &'_ mut V)>
    where
        D0: Keys<D0::Idx>,
        D1: Keys<D1::Idx>,
        D2: Keys<D2::Idx>,
    {
        self.indexes().zip(self.values_mut())
    }

    #[inline]
    pub fn indexes(&self) -> impl Iterator<Item = (D0::Idx, D1::Idx, D2::Idx)>
    where
        D0: Keys<D0::Idx>,
        D1: Keys<D1::Idx>,
        D2: Keys<D2::Idx>,
    {
        iproduct!(D0::keys(), D1::keys(), D2::keys())
    }

    #[inline]
    pub fn values(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    #[inline]
    pub fn values_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<D0, D1, V> Product2<&MArrD1<D0, V>, &MArrD1<D1, V>> for MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
    V: Mul<Output = V> + Copy,
{
    fn product2(w0: &MArrD1<D0, V>, w1: &MArrD1<D1, V>) -> Self {
        Self::from_iter(product2_iter(w0, w1))
    }
}

#[inline]
pub fn product2_iter<'a, D0, D1, V>(
    w0: &'a MArrD1<D0, V>,
    w1: &'a MArrD1<D1, V>,
) -> impl Iterator<Item = V> + Clone + 'a
where
    D0: Domain,
    D1: Domain,
    V: Mul<Output = V> + Copy,
{
    iproduct!(w0.values(), w1.values()).map(|(&v0, &v1)| v0 * v1)
}

impl<D0, D1, D2, V> Product3<&MArrD1<D0, V>, &MArrD1<D1, V>, &MArrD1<D2, V>>
    for MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
    V: Mul<Output = V> + Copy,
{
    fn product3(w0: &MArrD1<D0, V>, w1: &MArrD1<D1, V>, w2: &MArrD1<D2, V>) -> Self {
        Self::from_iter(product3_iter(w0, w1, w2))
    }
}

#[inline]
pub fn product3_iter<'a, D0, D1, D2, V>(
    w0: &'a MArrD1<D0, V>,
    w1: &'a MArrD1<D1, V>,
    w2: &'a MArrD1<D2, V>,
) -> impl Iterator<Item = V> + Clone + 'a
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
    V: Mul<Output = V> + Copy,
{
    iproduct!(w0.values(), w1.values(), w2.values()).map(|(&v0, &v1, &v2)| v0 * v1 * v2)
}

#[cfg(test)]
mod tests {
    use crate::multi_array::labeled::MArrD3;
    use crate::ops::{Product2, Product3};

    use super::{Keys, MArrD2};

    use super::{Domain, MArrD1};

    struct X;
    impl Domain for X {
        const LEN: usize = 2;
        type Idx = usize;
    }

    struct Y;
    impl Domain for Y {
        const LEN: usize = 3;
        type Idx = usize;
    }

    struct Z;
    impl Domain for Z {
        const LEN: usize = 2;
        type Idx = usize;
    }

    #[test]
    fn test_keys() {
        let mut keys = X::keys();
        for i in 0..X::LEN {
            assert_eq!(i, keys.next().unwrap());
        }
    }

    #[test]
    fn test_from_iter() {
        let ma = MArrD2::<X, Y, _>::from_iter(0..6);
        for (i, v) in ma.values().enumerate() {
            assert_eq!(i, *v);
        }
        let ma2 = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        assert_eq!(ma, ma2);

        let ma = MArrD3::<X, Y, Z, _>::from_iter(0..12);
        for (i, v) in ma.values().enumerate() {
            assert_eq!(i, *v);
        }
        let ma2 = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        assert_eq!(ma, ma2);
    }

    #[test]
    fn test_indexes() {
        let ma = MArrD1::<X, _>::from_iter([0, 1]);
        for i in ma.indexes() {
            assert_eq!(ma[i], i);
        }
        let ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        for (i, j) in ma.indexes() {
            assert_eq!(ma[(i, j)], i * Y::LEN + j);
        }
        let ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        for (i, j, k) in ma.indexes() {
            assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_values() {
        let ma = MArrD1::<X, _>::from_iter([0, 1]);
        for (i, v) in ma.values().enumerate() {
            assert_eq!(*v, i);
        }
        let ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        for (i, v) in ma.values().enumerate() {
            assert_eq!(*v, i);
        }
        let ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        for (i, v) in ma.values().enumerate() {
            assert_eq!(*v, i);
        }
    }

    #[test]
    fn test_iter() {
        let ma = MArrD1::<X, _>::from_iter([0, 1]);
        for (i, v) in ma.iter() {
            assert_eq!(*v, i);
        }
        let ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        for ((i, j), v) in ma.iter() {
            assert_eq!(*v, i * Y::LEN + j);
        }
        let ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        for ((i, j, k), v) in ma.iter() {
            assert_eq!(*v, i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_values_mut() {
        let mut ma = MArrD1::<X, _>::default();
        for (i, v) in ma.values_mut().enumerate() {
            *v = i;
        }
        for i in ma.indexes() {
            assert_eq!(ma[i], i);
        }

        let mut ma = MArrD2::<X, Y, _>::default();
        for (i, v) in ma.values_mut().enumerate() {
            *v = i;
        }
        for (i, j) in ma.indexes() {
            assert_eq!(ma[(i, j)], i * Y::LEN + j);
        }

        let mut ma = MArrD3::<X, Y, Z, _>::default();
        for (i, v) in ma.values_mut().enumerate() {
            *v = i;
        }
        for (i, j, k) in ma.indexes() {
            assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_iter_mut() {
        let mut ma = MArrD1::<X, _>::default();
        for (i, v) in ma.iter_mut() {
            *v = i;
        }
        for i in ma.indexes() {
            assert_eq!(ma[i], i);
        }

        let mut ma = MArrD2::<X, Y, _>::default();
        for ((i, j), v) in ma.iter_mut() {
            *v = i * Y::LEN + j;
        }
        println!("{ma:?}");
        for (i, j) in ma.indexes() {
            assert_eq!(ma[(i, j)], i * Y::LEN + j);
        }

        let mut ma = MArrD3::<X, Y, Z, _>::default();
        for ((i, j, k), v) in ma.iter_mut() {
            *v = i * Y::LEN * X::LEN + j * Z::LEN + k;
        }
        for (i, j, k) in ma.indexes() {
            assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_try_into() {
        let arr = vec![0, 1];
        let ma1 = MArrD1::<X, i32>::from_iter(arr.clone());
        let ma2: MArrD1<X, i32> = arr.try_into().unwrap();
        assert_eq!(ma1, ma2);

        let arr = vec![vec![0, 1, 2], vec![2, 3, 4]];
        let ma1 = MArrD2::<X, Y, i32>::from_multi_iter(arr.clone());
        let ma2: MArrD2<X, Y, i32> = arr.try_into().unwrap();
        assert_eq!(ma1, ma2);

        let arr = vec![
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        ];
        let ma1 = MArrD3::<X, Y, Z, i32>::from_multi_iter(arr.clone());
        let ma2: MArrD3<X, Y, Z, i32> = arr.try_into().unwrap();
        assert_eq!(ma1, ma2);

        let h = MArrD2::<X, Y, u32>::try_from(vec![vec![0i32, 0, 0], vec![0i32, 0, 0]]);
        assert!(h.is_ok());
        let g = MArrD2::<X, Y, u32>::try_from(vec![vec![0, -1, 0], vec![0, -1, 0]]);
        println!("{:?}", g);
        assert!(g.is_err());
    }

    #[test]
    fn test_format() {
        let ma = MArrD1::<X, _>::from_iter([1, 2]);
        assert_eq!(format!("{ma:?}"), "X[1, 2]");

        let ma = MArrD2::<X, Y, _>::from_multi_iter([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(format!("{ma:?}"), "X[Y[1, 2, 3], Y[4, 5, 6]]");

        let ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
        ]);
        assert_eq!(
            format!("{ma:?}"),
            "X[Y[Z[1, 2], Z[3, 4], Z[5, 6]], Y[Z[1, 2], Z[3, 4], Z[5, 6]]]"
        );
    }

    #[test]
    fn test_index() {
        let ma = MArrD1::<X, _>::from_iter([0, 1]);
        for i in X::keys() {
            assert_eq!(ma[i], i);
        }
        let ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        for i in X::keys() {
            for j in Y::keys() {
                assert_eq!(ma[(i, j)], i * Y::LEN + j);
            }
        }
        let ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        for i in X::keys() {
            for j in Y::keys() {
                for k in Z::keys() {
                    assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
                }
            }
        }
    }

    #[test]
    fn test_default() {
        let ma = MArrD1::<X, i32>::default();
        for i in X::keys() {
            assert_eq!(ma[i], 0);
        }

        let ma = MArrD2::<X, Y, i32>::default();
        for i in X::keys() {
            for j in Y::keys() {
                assert_eq!(ma[(i, j)], 0);
            }
        }

        let ma = MArrD3::<X, Y, Z, i32>::default();
        for i in X::keys() {
            for j in Y::keys() {
                for k in Z::keys() {
                    assert_eq!(ma[(i, j, k)], 0);
                }
            }
        }
    }

    #[test]
    fn test_index_mut() {
        let mut ma = MArrD1::<X, _>::default();
        for i in X::keys() {
            ma[i] = i;
        }
        for i in X::keys() {
            assert_eq!(ma[i], i);
        }

        let mut ma = MArrD2::<X, Y, _>::default();
        for i in X::keys() {
            for j in Y::keys() {
                ma[(i, j)] = i * Y::LEN + j;
            }
        }
        for i in X::keys() {
            for j in Y::keys() {
                assert_eq!(ma[(i, j)], i * Y::LEN + j);
            }
        }

        let mut ma = MArrD3::<X, Y, Z, _>::default();
        for i in X::keys() {
            for j in Y::keys() {
                for k in Z::keys() {
                    ma[(i, j, k)] = i * Y::LEN * X::LEN + j * Z::LEN + k;
                }
            }
        }
        for i in X::keys() {
            for j in Y::keys() {
                for k in Z::keys() {
                    assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
                }
            }
        }
    }

    #[test]
    fn test_product() {
        let mx = MArrD1::<X, _>::from_iter([1, 2]);
        let my = MArrD1::<Y, _>::from_iter([3, 4, 5]);
        let mz = MArrD1::<Z, _>::from_iter([6, 7]);

        let mxy = MArrD2::product2(&mx, &my);
        for ((i, j), xy) in mxy.iter() {
            assert_eq!(mx[i] * my[j], *xy);
        }

        let mxyz = MArrD3::product3(&mx, &my, &mz);
        for ((i, j, k), xyz) in mxyz.iter() {
            assert_eq!(mx[i] * my[j] * mz[k], *xyz);
        }
    }
}
