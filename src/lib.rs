//! # burn-ttt — Test-Time Training for Burn
//!
//! | arXiv | Function | What |
//! |-------|----------|------|
//! | [2512.23675](https://arxiv.org/abs/2512.23675) | `ttt_loss` | Next-token MSE loss on context at test time |
//!
//! TTT-E2E (Tandon et al., 2025): continue learning at test time via
//! next-token prediction on the given context, compressing the context
//! into model weights. Constant latency regardless of context length.
use burn::tensor::{backend::Backend, Tensor};

/// Masked MSE loss for test-time training on a sliding window context.
///
/// `pred`: `[B, L, D]` — model predictions
/// `target`: `[B, L, D]` — ground truth (shifted by 1)
/// `mask`: `[B, L]` — 1.0 for valid positions, 0.0 for padding
///
/// Returns scalar loss suitable for a single SGD step at test time.
/// Use with Burn's autograd for the gradient computation.
pub fn ttt_loss<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let se = (pred - target).powf_scalar(2.0).mean_dim(2);
    let [b, l, _] = se.dims();
    let se = se.reshape([b, l]);
    (se * mask.clone()).sum().div(mask.sum().clamp_min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = NdArray;
    fn dev() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn ttt_loss_finite() {
        let p = Tensor::<B, 3>::random([1, 16, 64], Distribution::Default, &dev());
        let t = Tensor::<B, 3>::random([1, 16, 64], Distribution::Default, &dev());
        let m = Tensor::<B, 2>::ones([1, 16], &dev());
        let l = ttt_loss(p, t, m);
        let v: f32 = f32::from_le_bytes(l.into_data().bytes[..4].try_into().unwrap());
        assert!(v.is_finite() && v >= 0.0);
    }
    #[test]
    fn ttt_loss_zero_when_equal() {
        let p = Tensor::<B, 3>::ones([2, 8, 32], &dev());
        let m = Tensor::<B, 2>::ones([2, 8], &dev());
        let l = ttt_loss(p.clone(), p, m);
        let v: f32 = f32::from_le_bytes(l.into_data().bytes[..4].try_into().unwrap());
        assert!(v.abs() < 1e-4);
    }
}
