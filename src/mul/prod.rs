use super::IndexContainer;
use std::{
    array::from_fn,
    ops::{Index, IndexMut},
    usize,
};

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

pub struct HA2<V, const K1: usize, const K0: usize>([[V; K1]; K0]);

impl<'a, V, const K0: usize, const K1: usize> IntoIterator for &'a HA2<V, K1, K0> {
    type Item = &'a V;
    type IntoIter = MyIter<'a, [V; K1]>;

    fn into_iter(self) -> Self::IntoIter {
        let iters = self.0.as_ref().iter();
        MyIter::new(iters)
    }
}

impl<V, const K0: usize, const K1: usize> Index<[usize; 2]> for HA2<V, K1, K0> {
    type Output = V;

    fn index(&self, k: [usize; 2]) -> &Self::Output {
        &self.0[k[0]][k[1]]
    }
}

impl<V, const K0: usize, const K1: usize> IndexMut<[usize; 2]> for HA2<V, K1, K0> {
    fn index_mut(&mut self, k: [usize; 2]) -> &mut Self::Output {
        &mut self.0[k[0]][k[1]]
    }
}

impl<V, const K0: usize, const K1: usize> IndexContainer<[usize; 2]> for HA2<V, K1, K0> {
    const SIZE: usize = K0 * K1;
    type Keys = HigherRange<2>;
    type Map<W> = HA2<W, K1, K0>;

    fn keys() -> Self::Keys {
        HigherRange::new([K0, K1])
    }

    fn map<U, F: Fn([usize; 2]) -> U>(f: F) -> HA2<U, K1, K0> {
        HA2::from_fn(|k| f(k))
    }

    fn from_fn<F: Fn([usize; 2]) -> Self::Output>(f: F) -> Self {
        HA2(from_fn(|k0| from_fn(|k1| f([k0, k1]))))
    }
}

pub struct HA3<V, const K2: usize, const K1: usize, const K0: usize>([HA2<V, K2, K1>; K0]);

impl<'a, V, const K0: usize, const K1: usize, const K2: usize> IntoIterator
    for &'a HA3<V, K2, K1, K0>
{
    type Item = &'a V;
    type IntoIter = MyIter<'a, HA2<V, K2, K1>>;

    fn into_iter(self) -> Self::IntoIter {
        let iters = self.0.as_ref().iter();
        MyIter::new(iters)
    }
}

impl<V, const K0: usize, const K1: usize, const K2: usize> Index<[usize; 3]>
    for HA3<V, K2, K1, K0>
{
    type Output = V;

    fn index(&self, k: [usize; 3]) -> &Self::Output {
        &self.0[k[0]][[k[1], k[2]]]
    }
}

impl<V, const K0: usize, const K1: usize, const K2: usize> IndexMut<[usize; 3]>
    for HA3<V, K2, K1, K0>
{
    fn index_mut(&mut self, k: [usize; 3]) -> &mut Self::Output {
        &mut self.0[k[0]][[k[1], k[2]]]
    }
}

impl<V, const K0: usize, const K1: usize, const K2: usize> IndexContainer<[usize; 3]>
    for HA3<V, K2, K1, K0>
{
    const SIZE: usize = K0 * K1 * K2;
    type Keys = HigherRange<3>;
    type Map<W> = HA3<W, K2, K1, K0>;

    fn keys() -> Self::Keys {
        HigherRange::new([K0, K1, K2])
    }

    fn map<U, F: Fn([usize; 3]) -> U>(f: F) -> HA3<U, K2, K1, K0> {
        HA3::from_fn(|k| f(k))
    }

    fn from_fn<F: Fn([usize; 3]) -> Self::Output>(f: F) -> Self {
        HA3(from_fn(|k0| HA2::from_fn(|[k1, k2]| f([k0, k1, k2]))))
    }
}

pub struct HigherArr2<V, const K1: usize, const K0: usize>(Box<HA2<V, K1, K0>>);

impl<V, const K1: usize, const K0: usize> HigherArr2<V, K1, K0> {
    pub fn from_fn<F: Fn([usize; 2]) -> V>(f: F) -> Self {
        Self(Box::new(HA2::from_fn(|k| f(k))))
    }
}

impl<V, const K1: usize, const K0: usize> std::ops::Deref for HigherArr2<V, K1, K0> {
    type Target = HA2<V, K1, K0>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct HigherArr3<V, const K2: usize, const K1: usize, const K0: usize>(
    Box<HA3<V, K2, K1, K0>>,
);

impl<V, const K2: usize, const K1: usize, const K0: usize> HigherArr3<V, K2, K1, K0> {
    pub fn from_fn<F: Fn([usize; 3]) -> V>(f: F) -> Self {
        Self(Box::new(HA3::from_fn(|k| f(k))))
    }
}

impl<V, const K2: usize, const K1: usize, const K0: usize> std::ops::Deref
    for HigherArr3<V, K2, K1, K0>
{
    type Target = HA3<V, K2, K1, K0>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{HigherArr2, HigherArr3, HigherRange};

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
    fn test_iter2() {
        let c: HigherArr2<_, 2, 3> = HigherArr2::from_fn(|[i, j]| (i, j));
        let mut iter = (&c).into_iter();
        for i in 0..3 {
            for j in 0..2 {
                assert!(iter.next() == Some(&(i, j)));
            }
        }
    }

    #[test]
    fn test_iter3() {
        let c: HigherArr3<_, 2, 3, 4> = HigherArr3::from_fn(|[i, j, k]| (i, j, k));
        let mut iter = (&c).into_iter();
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..2 {
                    assert!(iter.next() == Some(&(i, j, k)));
                }
            }
        }
    }
}
