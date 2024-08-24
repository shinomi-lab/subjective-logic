use std::{
    cmp, fmt,
    marker::PhantomData,
    ops::{Index, IndexMut, Mul},
    slice,
};

use itertools::iproduct;
use num_traits::Zero;

use crate::{
    domain::{Domain, DomainConv, Keys},
    iter::{Container, ContainerMap, FromFn},
    ops::{Indexes, Product2, Product3, Zeros},
};

pub struct Iter<'a, S>
where
    &'a S: IntoIterator,
{
    iters: std::slice::Iter<'a, S>,
    iter: Option<<&'a S as IntoIterator>::IntoIter>,
}

impl<'a, S> Clone for Iter<'a, S>
where
    &'a S: IntoIterator,
    <&'a S as IntoIterator>::IntoIter: Clone,
{
    fn clone(&self) -> Self {
        Self {
            iters: self.iters.clone(),
            iter: self.iter.clone(),
        }
    }
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

pub struct MArrD1<D0: Domain, V> {
    _marker: PhantomData<D0>,
    inner: Vec<V>,
}

impl<D0: Domain, V: fmt::Debug> fmt::Debug for MArrD1<D0, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:?}", tynm::type_name::<D0>(), self.inner)
    }
}

impl<D0: Domain, V: Clone> Clone for MArrD1<D0, V> {
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
            inner: self.inner.clone(),
        }
    }
}

impl<D0, V> cmp::PartialEq for MArrD1<D0, V>
where
    D0: Domain,
    V: cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D0, V> Default for MArrD1<D0, V>
where
    D0: Domain + Keys<D0::Idx>,
    V: Default,
{
    fn default() -> Self {
        Self::from_fn(|_| V::default())
    }
}

impl<D0, V> Zeros for MArrD1<D0, V>
where
    D0: Domain + Keys<D0::Idx>,
    V: Zero,
{
    fn zeros() -> Self {
        Self::from_fn(|_| V::zero())
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

impl<D0: Domain + Keys<D0::Idx>, V> Keys<D0::Idx> for MArrD1<D0, V> {
    fn keys() -> impl Iterator<Item = D0::Idx> + Clone {
        D0::keys()
    }
}

impl<D0: Domain + Keys<D0::Idx>, V> Indexes<D0::Idx> for MArrD1<D0, V> {
    fn indexes() -> impl Iterator<Item = D0::Idx> {
        D0::keys()
    }
}

impl<D0: Domain + Keys<D0::Idx>, V> FromFn<D0::Idx, V> for MArrD1<D0, V> {
    fn from_fn<F: FnMut(D0::Idx) -> V>(f: F) -> Self {
        Self::from_iter(Self::keys().map(f))
    }
}

impl<D0: Domain + Keys<D0::Idx>, V> Container<D0::Idx> for MArrD1<D0, V> {}

impl<D0: Domain + Keys<D0::Idx>, V> ContainerMap<D0::Idx> for MArrD1<D0, V> {
    type Map<U> = MArrD1<D0, U>;
}

impl<D0, E, V> DomainConv<MArrD1<E, V>> for MArrD1<D0, V>
where
    D0: Domain,
    E: Domain + From<D0>,
{
    fn conv(self) -> MArrD1<E, V> {
        MArrD1 {
            _marker: PhantomData,
            inner: self.inner,
        }
    }
}

impl<D0, V> MArrD1<D0, V>
where
    D0: Domain,
{
    pub fn new(inner: Vec<V>) -> Self {
        assert!(inner.len() == D0::LEN);
        Self {
            _marker: PhantomData,
            inner,
        }
    }

    #[inline]
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    pub fn as_ref(&self) -> MArrD1<D0, &V> {
        MArrD1::from_iter(self.iter())
    }
}

pub struct MArrD2<D0: Domain, D1: Domain, V> {
    inner: MArrD1<D0, MArrD1<D1, V>>,
}

impl<D0: Domain, D1: Domain, V: fmt::Debug> fmt::Debug for MArrD2<D0, D1, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

impl<D0: Domain, D1: Domain, V: Clone> Clone for MArrD2<D0, D1, V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<D0, D1, V> cmp::PartialEq for MArrD2<D0, D1, V>
where
    D0: Domain,
    D1: Domain,
    V: cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D0, D1, V> Default for MArrD2<D0, D1, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    V: Default,
{
    fn default() -> Self {
        Self::from_fn(|_| V::default())
    }
}

impl<D0, D1, V> Zeros for MArrD2<D0, D1, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    V: Zero,
{
    fn zeros() -> Self {
        Self::from_fn(|_| V::zero())
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
        let mut inner = Vec::with_capacity(D0::LEN);
        let mut v = Vec::from_iter(iter);
        for _ in 0..D0::LEN {
            inner.push(MArrD1::<D1, _>::from_iter(v.drain(0..D1::LEN)));
        }
        Self::new(inner)
    }
}

impl<D0, D1, V> Keys<(D0::Idx, D1::Idx)> for MArrD2<D0, D1, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
{
    fn keys() -> impl Iterator<Item = (D0::Idx, D1::Idx)> + Clone {
        iproduct!(D0::keys(), D1::keys())
    }
}

impl<D0, D1, V> Indexes<(D0::Idx, D1::Idx)> for MArrD2<D0, D1, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
{
    fn indexes() -> impl Iterator<Item = (D0::Idx, D1::Idx)> {
        Self::keys()
    }
}

impl<D0, D1, V> FromFn<(D0::Idx, D1::Idx), V> for MArrD2<D0, D1, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
{
    fn from_fn<F: FnMut((D0::Idx, D1::Idx)) -> V>(f: F) -> Self {
        Self::from_iter(Self::keys().map(f))
    }
}

impl<D0, D1, V> Container<(D0::Idx, D1::Idx)> for MArrD2<D0, D1, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
{
}

impl<D0, D1, V> ContainerMap<(D0::Idx, D1::Idx)> for MArrD2<D0, D1, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
{
    type Map<U> = MArrD2<D0, D1, U>;
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
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    pub fn down(&self, idx: D0::Idx) -> &MArrD1<D1, V> {
        &self.inner[idx]
    }

    pub fn down_mut(&mut self, idx: D0::Idx) -> &mut MArrD1<D1, V> {
        self.inner.index_mut(idx)
    }
}

pub struct MArrD3<D0: Domain, D1: Domain, D2: Domain, V> {
    inner: MArrD1<D0, MArrD2<D1, D2, V>>,
}

impl<D0: Domain, D1: Domain, D2: Domain, V: fmt::Debug> fmt::Debug for MArrD3<D0, D1, D2, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

impl<D0: Domain, D1: Domain, D2: Domain, V: Clone> Clone for MArrD3<D0, D1, D2, V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<D0, D1, D2, V> cmp::PartialEq for MArrD3<D0, D1, D2, V>
where
    D0: Domain,
    D1: Domain,
    D2: Domain,
    V: cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<D0, D1, D2, V> Default for MArrD3<D0, D1, D2, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    D2: Domain + Keys<D2::Idx>,
    V: Default,
{
    fn default() -> Self {
        Self::from_fn(|_| V::default())
    }
}

impl<D0, D1, D2, V> Zeros for MArrD3<D0, D1, D2, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    D2: Domain + Keys<D2::Idx>,
    V: Zero,
{
    fn zeros() -> Self {
        Self::from_fn(|_| V::zero())
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
        let mut arr = Vec::with_capacity(D0::LEN);
        let mut v = Vec::from_iter(iter);
        for _ in 0..D0::LEN {
            let mut arr2 = Vec::with_capacity(D1::LEN);
            for _ in 0..D1::LEN {
                arr2.push(MArrD1::<D2, _>::from_iter(v.drain(0..D2::LEN)));
            }
            arr.push(MArrD2::<D1, D2, _>::new(arr2));
        }
        Self::new(arr)
    }
}

impl<D0, D1, D2, V> Keys<(D0::Idx, D1::Idx, D2::Idx)> for MArrD3<D0, D1, D2, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    D2: Domain + Keys<D2::Idx>,
{
    fn keys() -> impl Iterator<Item = (D0::Idx, D1::Idx, D2::Idx)> + Clone {
        iproduct!(D0::keys(), D1::keys(), D2::keys())
    }
}

impl<D0, D1, D2, V> Indexes<(D0::Idx, D1::Idx, D2::Idx)> for MArrD3<D0, D1, D2, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    D2: Domain + Keys<D2::Idx>,
{
    fn indexes() -> impl Iterator<Item = (D0::Idx, D1::Idx, D2::Idx)> {
        Self::keys()
    }
}

impl<D0, D1, D2, V> FromFn<(D0::Idx, D1::Idx, D2::Idx), V> for MArrD3<D0, D1, D2, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    D2: Domain + Keys<D2::Idx>,
{
    fn from_fn<F: FnMut((D0::Idx, D1::Idx, D2::Idx)) -> V>(f: F) -> Self {
        Self::from_iter(Self::keys().map(f))
    }
}

impl<D0, D1, D2, V> Container<(D0::Idx, D1::Idx, D2::Idx)> for MArrD3<D0, D1, D2, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    D2: Domain + Keys<D2::Idx>,
{
}

impl<D0, D1, D2, V> ContainerMap<(D0::Idx, D1::Idx, D2::Idx)> for MArrD3<D0, D1, D2, V>
where
    D0: Domain + Keys<D0::Idx>,
    D1: Domain + Keys<D1::Idx>,
    D2: Domain + Keys<D2::Idx>,
{
    type Map<U> = MArrD3<D0, D1, D2, U>;
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
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    pub fn down(&self, idx: D0::Idx) -> &MArrD2<D1, D2, V> {
        &self.inner[idx]
    }

    pub fn down_mut(&mut self, idx: D0::Idx) -> &mut MArrD2<D1, D2, V> {
        self.inner.index_mut(idx)
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
    iproduct!(w0, w1).map(|(&v0, &v1)| v0 * v1)
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
    iproduct!(w0, w1, w2).map(|(&v0, &v1, &v2)| v0 * v1 * v2)
}

#[macro_export]
macro_rules! marr_d1 {
    ($d0:tt;$e:expr) => {
        $crate::multi_array::labeled::MArrD1::<$d0, _>::from_iter($e)
    };
    [$($e:expr),*$(,)?] => {
        $crate::multi_array::labeled::MArrD1::<_, _>::from_iter([$($e,)*])
    };
}

#[macro_export]
macro_rules! marr_d2 {
    ($d0:tt,$d1:tt;$e:expr) => {
        $crate::multi_array::labeled::MArrD2::<$d0, $d1, _>::from_multi_iter($e)
    };
    [$($e:tt),*$(,)?] => {
        $crate::multi_array::labeled::MArrD2::<_, _, _>::from_multi_iter([$($e,)*])
    };
}

#[macro_export]
macro_rules! marr_d3 {
    ($d0:tt,$d1:tt,$d2:tt;$e:expr) => {
        $crate::multi_array::labeled::MArrD3::<$d0, $d1, $d2, _>::from_multi_iter($e)
    };
    [$($e:tt),*$(,)?] => {
        $crate::multi_array::labeled::MArrD3::<_, _, _, _>::from_multi_iter([$($e,)*])
    };
}

#[cfg(test)]
mod tests {
    use crate::domain::DomainConv;
    use crate::impl_domain;
    use crate::iter::Container;
    use crate::multi_array::labeled::MArrD3;
    use crate::ops::{Indexes, Product2, Product3, Zeros};

    use super::{Keys, MArrD2};

    use super::{Domain, MArrD1};

    struct X;
    impl_domain!(X = 2);
    struct Y;
    impl_domain!(Y = 3);
    struct Z;
    impl_domain!(Z = 2);

    #[test]
    fn test_clone() {
        let ma = MArrD1::<X, i32>::zeros();
        let ma2 = ma.clone();
        assert_eq!(ma, ma2);
        let ma = MArrD2::<X, Y, i32>::zeros();
        let ma2 = ma.clone();
        assert_eq!(ma, ma2);
        let ma = MArrD3::<X, Y, Z, i32>::zeros();
        let ma2 = ma.clone();
        assert_eq!(ma, ma2);
    }

    #[test]
    fn test_keys() {
        let mut keys = X::keys();
        for i in 0..X::LEN {
            assert_eq!(i, keys.next().unwrap());
        }
    }

    #[test]
    fn test_macro1() {
        let ma = marr_d1!(X; [0, 1]);
        for i in X::keys() {
            assert_eq!(ma[i], i);
        }
        let ma = marr_d2!(X, Y; [[0, 1, 2], [3, 4, 5]]);
        for (i, j) in MArrD2::<X, Y, usize>::indexes() {
            assert_eq!(ma[(i, j)], i * Y::LEN + j);
        }
        let ma = marr_d3!(X, Y, Z; [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]);
        for (i, j, k) in MArrD3::<X, Y, Z, usize>::indexes() {
            assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_macro1_2() {
        let ma: MArrD1<X, usize> = marr_d1!(_; [0, 1]);
        for i in X::keys() {
            assert_eq!(ma[i], i);
        }
        let ma: MArrD2<X, Y, usize> = marr_d2!(_, _; [[0, 1, 2], [3, 4, 5]]);
        for (i, j) in MArrD2::<X, Y, usize>::indexes() {
            assert_eq!(ma[(i, j)], i * Y::LEN + j);
        }
        let ma: MArrD3<X, Y, Z, usize> =
            marr_d3!(_, _, _; [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]);
        for (i, j, k) in MArrD3::<X, Y, Z, usize>::indexes() {
            assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_macro2() {
        let ma: MArrD1<X, usize> = marr_d1![0, 1];
        for i in X::keys() {
            assert_eq!(ma[i], i);
        }
        let ma: MArrD2<X, Y, usize> = marr_d2![[0, 1, 2], [3, 4, 5]];
        for (i, j) in MArrD2::<X, Y, usize>::indexes() {
            assert_eq!(ma[(i, j)], i * Y::LEN + j);
        }
        let ma: MArrD3<X, Y, Z, usize> =
            marr_d3![[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]];
        for (i, j, k) in MArrD3::<X, Y, Z, usize>::indexes() {
            assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_from_iter() {
        let ma = MArrD2::<X, Y, _>::from_iter(0..6);
        for (i, v) in ma.into_iter().enumerate() {
            assert_eq!(i, *v);
        }
        let ma2 = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        assert_eq!(ma, ma2);

        let ma = MArrD3::<X, Y, Z, _>::from_iter(0..12);
        for (i, v) in ma.into_iter().enumerate() {
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
        for i in X::keys() {
            assert_eq!(ma[i], i);
        }
        let ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        for (i, j) in MArrD2::<X, Y, usize>::indexes() {
            assert_eq!(ma[(i, j)], i * Y::LEN + j);
        }
        let ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        for (i, j, k) in MArrD3::<X, Y, Z, usize>::indexes() {
            assert_eq!(ma[(i, j, k)], i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_iter() {
        let ma = MArrD1::<X, _>::from_iter([0, 1]);
        for (i, v) in ma.into_iter().enumerate() {
            assert_eq!(*v, i);
        }
        let ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        for (i, v) in ma.into_iter().enumerate() {
            assert_eq!(*v, i);
        }
        let ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        for (i, v) in ma.into_iter().enumerate() {
            assert_eq!(*v, i);
        }
    }

    #[test]
    fn test_clone_iter() {
        let ma = MArrD1::<X, _>::from_iter([0, 1]);
        let iter = ma.iter();
        let ma2 = MArrD1::<X, _>::from_iter(iter.clone().cloned());
        for x in X::keys() {
            assert_eq!(ma[x], ma2[x]);
        }
        let ma2 = MArrD1::<X, _>::from_iter(iter.clone().cloned());
        for x in X::keys() {
            assert_eq!(ma[x], ma2[x]);
        }
        let ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        let iter = ma.iter();
        let ma2 = MArrD2::<X, Y, _>::from_iter(iter.clone().cloned());
        for x in X::keys() {
            for y in Y::keys() {
                assert_eq!(ma[(x, y)], ma2[(x, y)]);
            }
        }
        let ma2 = MArrD2::<X, Y, _>::from_iter(iter.clone().cloned());
        for x in X::keys() {
            for y in Y::keys() {
                assert_eq!(ma[(x, y)], ma2[(x, y)]);
            }
        }
    }

    #[test]
    fn test_iter_with() {
        let ma = MArrD1::<X, _>::from_iter([0, 1]);
        for (i, v) in ma.iter_with() {
            assert_eq!(*v, i);
        }
        let ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        for ((i, j), v) in ma.iter_with() {
            assert_eq!(*v, i * Y::LEN + j);
        }
        let ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        for ((i, j, k), v) in ma.iter_with() {
            assert_eq!(*v, i * Y::LEN * X::LEN + j * Z::LEN + k);
        }
    }

    #[test]
    fn test_iter_mut() {
        let mut ma = MArrD1::<X, _>::zeros();
        for (i, v) in ma.iter_mut().enumerate() {
            *v = i;
        }
        for i in MArrD1::<X, usize>::indexes() {
            assert_eq!(ma[i], i);
        }

        let mut ma = MArrD2::<X, Y, _>::zeros();
        for (i, v) in ma.iter_mut().enumerate() {
            *v = i;
        }
        for (i, j) in MArrD2::<X, Y, usize>::indexes() {
            assert_eq!(ma[(i, j)], i * Y::LEN + j);
        }

        let mut ma = MArrD3::<X, Y, Z, _>::zeros();
        for (i, v) in ma.iter_mut().enumerate() {
            *v = i;
        }
        for (i, j, k) in MArrD3::<X, Y, Z, usize>::indexes() {
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
        let ma = MArrD1::<X, i32>::zeros();
        for i in X::keys() {
            assert_eq!(ma[i], 0);
        }

        let ma = MArrD2::<X, Y, i32>::zeros();
        for i in X::keys() {
            for j in Y::keys() {
                assert_eq!(ma[(i, j)], 0);
            }
        }

        let ma = MArrD3::<X, Y, Z, i32>::zeros();
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
        let mut ma = MArrD1::<X, _>::zeros();
        for i in X::keys() {
            ma[i] = i;
        }
        for i in X::keys() {
            assert_eq!(ma[i], i);
        }

        let mut ma = MArrD2::<X, Y, _>::zeros();
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

        let mut ma = MArrD3::<X, Y, Z, _>::zeros();
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
        for ((i, j), xy) in mxy.iter_with() {
            assert_eq!(mx[i] * my[j], *xy);
        }

        let mxyz = MArrD3::product3(&mx, &my, &mz);
        for ((i, j, k), xyz) in mxyz.iter_with() {
            assert_eq!(mx[i] * my[j] * mz[k], *xyz);
        }
    }

    struct W;
    impl_domain!(W from X);

    #[test]
    fn test_conv() {
        let mx = MArrD1::<X, _>::from_iter([1, 2]);
        let mw: MArrD1<W, _> = mx.conv();
        assert_eq!(X::LEN, W::LEN);
        assert_eq!(mw.inner, vec![1, 2]);
    }

    #[test]
    fn test_down() {
        let mut ma = MArrD2::<X, Y, _>::from_multi_iter([[0, 1, 2], [3, 4, 5]]);
        assert_eq!(ma.down(0), &marr_d1!(Y; [0, 1, 2]));
        assert_eq!(ma.down(1), &marr_d1!(Y; [3, 4, 5]));
        let a = marr_d1!(Y; [8, 7, 6]);
        *ma.down_mut(1) = a;
        assert_eq!(ma.down(1), &marr_d1!(Y; [8, 7, 6]));

        let mut ma = MArrD3::<X, Y, Z, _>::from_multi_iter([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]);
        assert_eq!(ma.down(0), &marr_d2!(Y, Z; [[0, 1], [2, 3], [4, 5]]));
        assert_eq!(ma.down(1), &marr_d2!(Y, Z; [[6, 7], [8, 9], [10, 11]]));
        let a = marr_d2!(Y, Z; [[9, 8], [7, 6], [5, 4]]);
        *ma.down_mut(1) = a;
        assert_eq!(ma.down(1), &marr_d2!(Y, Z; [[9, 8], [7, 6], [5, 4]]));
    }

    #[test]
    fn to_ref() {
        let a = marr_d1!(X; [1, 2]);
        let b = a.as_ref();
        for x in X::keys() {
            assert_eq!(a[x], *b[x]);
        }
    }
}
