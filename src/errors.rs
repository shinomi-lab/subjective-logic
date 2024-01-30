use approx::UlpsEq;
use num_traits::Float;

use crate::approx_ext;

/// An error indicating that one or more invalid values are used.
#[derive(thiserror::Error, Debug)]
#[error("At least one parameter is invalid because {0}.")]
pub struct InvalidValueError(pub String);

#[inline]
pub fn check_unit_interval<V: Float + UlpsEq, S: Into<String>>(
    v: V,
    label: S,
) -> Result<(), InvalidValueError> {
    if approx_ext::in_unit_interval(v) {
        Ok(())
    } else {
        Err(InvalidValueError(format!(
            "{} ∈ [0,1] is not satisfied",
            label.into()
        )))
    }
}

#[inline]
pub fn check_is_one<V: Float + UlpsEq, S: Into<String>>(
    v: V,
    label: S,
) -> Result<(), InvalidValueError> {
    if approx_ext::is_one(v) {
        Ok(())
    } else {
        Err(InvalidValueError(format!(
            "{} = 1 is not satisfied",
            label.into()
        )))
    }
}
