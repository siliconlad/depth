import pytest
from tinygrad.tensor import Tensor

from depth.loss.smoothness import smoothness_loss


class TestSmoothnessLoss:
    def test_scalar_output(self):
        disp = Tensor.rand(1, 1, 32, 32)
        img = Tensor.rand(1, 3, 32, 32)
        assert smoothness_loss(disp, img).shape == ()

    def test_constant_disparity_is_zero(self):
        disp = Tensor.ones(1, 1, 32, 32)
        img = Tensor.rand(1, 3, 32, 32)
        assert smoothness_loss(disp, img).numpy() == pytest.approx(0.0)

    def test_positive(self):
        disp = Tensor.rand(1, 1, 32, 32)
        img = Tensor.rand(1, 3, 32, 32)
        assert smoothness_loss(disp, img).numpy() > 0.0

    def test_smooth_disparity_lower_than_noisy(self):
        img = Tensor.rand(1, 3, 32, 32)
        smooth_disp = Tensor.ones(1, 1, 32, 32) + Tensor.rand(1, 1, 32, 32) * 0.01
        noisy_disp = Tensor.rand(1, 1, 32, 32)
        assert smoothness_loss(smooth_disp, img).numpy() < smoothness_loss(noisy_disp, img).numpy()

    def test_edge_aware_weighting(self):
        disp = Tensor.rand(1, 1, 32, 32)
        flat_img = Tensor.ones(1, 3, 32, 32) * 0.5
        edgy_img = Tensor.rand(1, 3, 32, 32)
        # Strong image gradients suppress the penalty, so loss should be lower with edges
        assert smoothness_loss(disp, edgy_img).numpy() < smoothness_loss(disp, flat_img).numpy()
