use std::ops::Range;

pub trait Domain {
    type Idx: Into<usize> + Clone;
    const LEN: usize;
}

#[macro_export]
macro_rules! impl_domain {
    ($s:ident = $l:expr) => {
        impl Domain for $s {
            const LEN: usize = $l;
            type Idx = usize;
        }
    };

    ($s:ident from $f:ident) => {
        impl_domain!($s = $f::LEN);
        impl From<$f> for $s {
            fn from(_: $f) -> $s {
                $s
            }
        }
    };
}

pub trait Keys<I> {
    type Iter: Iterator<Item = I> + Clone;
    fn keys() -> Self::Iter;
}

impl<D: Domain<Idx = usize>> Keys<usize> for D {
    type Iter = Range<D::Idx>;

    fn keys() -> Self::Iter {
        0..D::LEN
    }
}

pub trait DomainConv<T> {
    fn conv(self) -> T;
}
