use approx::ulps_eq;

pub trait ApproxRange<T = Self>: Sized {
    fn is_in_range(self, from: T, to: T) -> bool;
    fn out_of_range(self, from: T, to: T) -> bool {
        !self.is_in_range(from, to)
    }
}

macro_rules! impl_approx {
    ($ft: ty) => {
        impl ApproxRange for $ft {
            fn is_in_range(self, from: Self, to: Self) -> bool {
                (self >= from && self <= to) || ulps_eq!(self, from) || ulps_eq!(self, to)
            }
        }
        impl ApproxRange for &$ft {
            fn is_in_range(self, from: Self, to: Self) -> bool {
                (self >= from && self <= to) || ulps_eq!(self, from) || ulps_eq!(self, to)
            }
        }
    };
}

impl_approx!(f32);
impl_approx!(f64);
