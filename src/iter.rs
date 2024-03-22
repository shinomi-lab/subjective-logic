use std::ops::Index;

use crate::ops::Indexes;

pub trait Container<K>: Indexes<K> + Index<K> {
    fn iter_with<'a>(&'a self) -> impl Iterator<Item = (K, &'a Self::Output)>
    where
        K: Clone,
        Self::Output: 'a,
    {
        Self::indexes().map(|i| (i.clone(), &self[i]))
    }
}

pub trait ContainerMap<K> {
    type Map<U>: FromFn<K, U> + Index<K, Output = U>;

    fn map<U, F: FnMut(K) -> U>(f: F) -> Self::Map<U> {
        Self::Map::from_fn(f)
    }
}

pub trait FromFn<K, V> {
    fn from_fn<F: FnMut(K) -> V>(f: F) -> Self;
}
