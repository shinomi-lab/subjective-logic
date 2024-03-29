//!
//! An implementation of [Subjective Logic](https://en.wikipedia.org/wiki/Subjective_logic).

pub mod approx_ext;
pub mod bi;
pub mod convert;
pub mod domain;
pub mod errors;
pub mod iter;
pub mod mul;
pub mod multi_array;
pub mod ops;

#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq, assert_ulps_eq};

    use crate::{
        bi::{BOpinion, BSimplex},
        marr2, marr3,
        mul::{Opinion, OpinionRef, Simplex},
        ops::{Deduction, Projection},
    };

    #[test]
    fn test_bsl_boundary() {
        macro_rules! def {
            ($ft: ty) => {
                assert!(BOpinion::<$ft>::try_new(1.0, 0.0, 0.0, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.0, 1.0, 0.0, 1.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.0, 0.0, 1.0, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.5, 0.5, 0.0, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.0, 0.5, 0.5, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(0.5, 0.0, 0.5, 0.0).is_ok());
                assert!(BOpinion::<$ft>::try_new(-0.1, 0.1, 1.0, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.1, -0.1, 0.0, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.0, -0.1, 0.1, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(0.0, 1.1, -0.1, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.0, 0.1, -0.1, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.0, 0.0, 0.0, -0.1)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(1.0, 0.0, 0.0, 1.1)
                    .map_err(|e| println!("{e}"))
                    .is_err());
                assert!(BOpinion::<$ft>::try_new(0.5, 0.5, 0.5, 0.0)
                    .map_err(|e| println!("{e}"))
                    .is_err());
            };
        }

        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_cum_fusion_bo() {
        macro_rules! def {
            ($ft: ty) => {
                let w0 = BOpinion::<$ft>::new(0.5, 0.0, 0.5, 0.5);
                let w1 = BOpinion::<$ft>::new(0.0, 0.0, 1.0, 0.5);
                let w2 = BOpinion::<$ft>::new(0.0, 0.0, 1.0, 0.5);
                assert!(w0.cfuse(&w2).is_ok());
                assert!(w1.cfuse(&w2).is_ok());
            };
        }
        def!(f32);
        def!(f64);
    }

    #[test]
    fn test_avg_fuse() {
        let w0 = BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5);
        let w1 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
        let w2 = BOpinion::<f32>::new(0.0, 0.5, 0.5, 0.5);
        let w3 = BOpinion::<f32>::new(0.0, 0.6, 0.4, 0.5);

        println!("{}", w0.afuse(&w1, 0.5).unwrap());
        println!("{}", w1.afuse(&w2, 0.5).unwrap());
        println!("{}", w1.afuse(&w3, 0.5).unwrap());
        println!("{}", w2.afuse(&w3, 0.5).unwrap());
    }

    #[test]
    fn test_wgt_fuse() {
        let w0 = BOpinion::<f32>::new(0.0, 0.0, 1.0, 0.5);
        let w1 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
        let w2 = BOpinion::<f32>::new(0.0, 0.5, 0.5, 0.5);
        let w3 = BOpinion::<f32>::new(0.0, 0.6, 0.4, 0.5);

        assert_relative_eq!(w1.wfuse(&w0, 0.5).unwrap(), w1);
        assert_ulps_eq!(w1.wfuse(&w0, 0.5).unwrap(), w1);
        assert_relative_eq!(w2.wfuse(&w0, 0.5).unwrap(), w2);
        assert_ulps_eq!(w2.wfuse(&w0, 0.5).unwrap(), w2);
        assert_relative_eq!(w3.wfuse(&w0, 0.5).unwrap(), w3);
        assert_ulps_eq!(w3.wfuse(&w0, 0.5).unwrap(), w3);

        assert_relative_eq!(w2.wfuse(&w2, 0.5).unwrap(), w2);
        assert_ulps_eq!(w2.wfuse(&w2, 0.5).unwrap(), w2);
        assert_relative_eq!(w3.wfuse(&w3, 0.5).unwrap(), w3);
        assert_ulps_eq!(w3.wfuse(&w3, 0.5).unwrap(), w3);
    }

    #[test]
    fn test_fusion_bop() {
        let w1 = BOpinion::<f32>::new(0.5, 0.0, 0.5, 0.5);
        let w2 = BOpinion::<f32>::new(0.0, 0.90, 0.10, 0.5);
        assert!(w1.cfuse(&w2).is_ok());
        assert!(w1.afuse(&w2, 0.5).is_ok());
        assert!(w1.wfuse(&w2, 0.5).is_ok());
    }

    #[test]
    fn test_deduction_bop() {
        let cond = [
            BSimplex::<f32>::new(0.90, 0.02, 0.08),
            BSimplex::<f32>::new(0.40, 0.52, 0.08),
        ];
        let w = BOpinion::<f32>::new(0.00, 0.38, 0.62, 0.5);
        println!("{}", w.deduce(&cond, 0.5));

        let cond = [
            BSimplex::<f32>::new(0.72, 0.18, 0.1),
            BSimplex::<f32>::new(0.13, 0.57, 0.3),
        ];
        let w = BOpinion::<f32>::new(0.7, 0.0, 0.3, 0.33);
        println!("{}", w.deduce(&cond, 0.5));
    }

    #[test]
    fn test_deduction_mop() {
        let a = [0.5, 0.5];
        let s = Simplex::new([0.0, 0.9], 0.1);
        let w = Opinion::from((s.clone(), a.clone()));
        let ay = [0.75, 0.25];
        let conds = [
            Simplex::new([0.5, 0.25], 0.25),
            Simplex::new([0.5, 0.25], 0.25),
        ];
        let w2 = w.deduce_with(&conds, ay);
        let w3 = OpinionRef::from((&s, &a)).deduce_with(&conds, ay);
        assert_eq!(w2, w3);
    }

    #[test]
    fn test_deduction() {
        let b_a = 1.0;
        let b_xa = 0.0;
        let wa = Opinion::new([b_a, 0.0, 0.0], 1.0 - b_a, [0.25, 0.25, 0.5]);
        let wxa = [
            BSimplex::<f32>::new(b_xa, 0.0, 1.0 - b_xa),
            BSimplex::<f32>::new(0.0, 0.0, 1.0),
            BSimplex::<f32>::new(0.0, 0.0, 1.0),
        ];
        println!("{}", wxa[0]);
        let x: BOpinion<f32> = wa.deduce_with(&wxa, [0.5, 0.5]).into();
        println!("{}", x.projection());
    }

    #[test]
    fn test_deduction1() {
        let w = BOpinion::<f32>::new(0.7, 0.0, 0.3, 1.0 / 3.0);
        let cond = [
            BSimplex::<f32>::new(0.72, 0.18, 0.1),
            BSimplex::<f32>::new(0.13, 0.57, 0.3),
        ];
        println!("{}", w.deduce(&cond, 0.5));

        let wx = Opinion::new([0.7, 0.0], 0.3, [1.0 / 3.0, 2.0 / 3.0]);
        let wyx = [
            Simplex::new([0.72, 0.18], 0.1),
            Simplex::new([0.13, 0.57], 0.3),
        ];
        println!("{:?}", wx.as_ref().deduce_with(&wyx, [0.5, 0.5]));
    }

    #[test]
    fn test_deduction2() {
        let wa = Opinion::new([0.7, 0.1, 0.0], 0.2, [0.3, 0.3, 0.4]);
        let wxa = [
            BSimplex::<f32>::new(0.7, 0.0, 0.3),
            BSimplex::<f32>::new(0.0, 0.7, 0.3),
            BSimplex::<f32>::new(0.0, 0.0, 1.0),
        ];
        let wx: BOpinion<f32> = wa.deduce_with(&wxa, [0.5, 0.5]).into();
        println!("{}|{}", wx, wx.projection());

        let wxa = [
            Simplex::new([0.7, 0.0], 0.3),
            Simplex::new([0.0, 0.7], 0.3),
            Simplex::new([0.0, 0.0], 1.0),
        ];
        let wx = wa.as_ref().deduce_with(&wxa, [0.5, 0.5]);
        println!("{:?}|{}", wx, wx.projection()[0]);

        let wa = BOpinion::<f32>::new(0.7, 0.1, 0.2, 0.5);
        let wxa = [
            BSimplex::<f32>::new(0.7, 0.0, 0.3),
            BSimplex::<f32>::new(0.0, 0.7, 0.3),
        ];
        println!("{}", wa.deduce(&wxa, 0.5));
    }

    #[test]
    fn test_macro_import() {
        // let h = harr1![std::vec![0], std::vec![1], std::vec![1]];
        let h2 = marr2![[std::vec![0], std::vec![1]], [std::vec![1], std::vec![1]]];
        let h3 = marr3![
            [[std::vec![0], std::vec![1]]],
            [[std::vec![1], std::vec![1]]]
        ];
        // assert_eq!(h.len(), 3);
        assert_eq!(h2.len(), 4);
        assert_eq!(h3.len(), 4);
    }
}
