pub trait Domain {
    type Idx: Into<usize> + From<usize> + Clone;
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

#[macro_export]
macro_rules! new_type_domain {
    ($v:tt $s:ident = $l:expr) => {
        #[derive(Debug, Clone, Copy)]
        $v struct $s(pub usize);
        impl Domain for $s {
            const LEN: usize = $l;
            type Idx = Self;
        }
        impl From<usize> for $s {
            fn from(value: usize) -> Self {
                Self(value)
            }
        }

        impl From<$s> for usize {
            fn from(value: $s) -> Self {
                value.0
            }
        }
    };

    ($v:tt $s:ident from $f:ident) => {
        new_type_domain!($v $s = $f::LEN);
        impl From<$f> for $s {
            fn from(value: $f) -> $s {
                $s(value.0)
            }
        }
        impl From<$s> for $f {
            fn from(value: $s) -> $f {
                $f(value.0)
            }
        }
    };
}

pub trait Keys<I> {
    fn keys() -> impl Iterator<Item = I> + Clone;
}

impl<D: Domain> Keys<D::Idx> for D {
    fn keys() -> impl Iterator<Item = D::Idx> + Clone {
        (0..D::LEN).map(|i| i.into())
    }
}

pub trait DomainConv<T> {
    fn conv(self) -> T;
}
