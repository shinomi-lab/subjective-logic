use crate::bi::BOpinion;
use crate::mul::non_labeled::Opinion1d;

macro_rules! impl_convert {
    ($ft: ty) => {
        impl From<BOpinion<$ft>> for Opinion1d<$ft, 2> {
            fn from(value: BOpinion<$ft>) -> Self {
                Opinion1d {
                    simplex: value.simplex.0,
                    base_rate: [value.base_rate, 1.0 - value.base_rate],
                }
            }
        }

        impl From<Opinion1d<$ft, 2>> for BOpinion<$ft> {
            fn from(value: Opinion1d<$ft, 2>) -> Self {
                BOpinion::<$ft>::new_unchecked(
                    value.b()[0],
                    value.b()[1],
                    value.u(),
                    value.base_rate[0],
                )
            }
        }

        impl From<&Opinion1d<$ft, 2>> for BOpinion<$ft> {
            fn from(value: &Opinion1d<$ft, 2>) -> Self {
                BOpinion::<$ft>::new_unchecked(
                    value.b()[0],
                    value.b()[1],
                    value.u(),
                    value.base_rate[0],
                )
            }
        }
    };
}

impl_convert!(f32);
impl_convert!(f64);
