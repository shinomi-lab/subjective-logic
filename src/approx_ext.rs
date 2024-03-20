use approx::{ulps_eq, UlpsEq};
use num_traits::Float;

#[inline]
pub fn is_in_range<V: UlpsEq + PartialOrd>(v: V, from: V, to: V) -> bool {
    (v >= from && v <= to) || ulps_eq!(v, from) || ulps_eq!(v, to)
}

#[inline]
pub fn out_of_range<V: UlpsEq + PartialOrd>(v: V, from: V, to: V) -> bool {
    !is_in_range(v, from, to)
}

#[inline]
pub fn in_unit_interval<V: Float + UlpsEq>(v: V) -> bool {
    is_in_range(v, V::zero(), V::one())
}

#[inline]
pub fn is_one<V: Float + UlpsEq>(v: V) -> bool {
    ulps_eq!(v, V::one())
}

#[inline]
pub fn is_zero<V: Float + UlpsEq>(v: V) -> bool {
    ulps_eq!(v, V::zero())
}
