import pytest
from tinygrad.tensor import Tensor

from depth.loss.photometric import l1_loss, ssim_loss, photometric_loss


class TestL1Loss:
    def test_scalar_output(self):
        x = Tensor.rand(1, 3, 64, 64)
        y = Tensor.rand(1, 3, 64, 64)
        assert l1_loss(x, y).shape == ()

    def test_identical_images(self):
        x = Tensor.randn(1, 3, 32, 32)
        assert l1_loss(x, x).numpy() == pytest.approx(0.0, abs=1e-5)

    def test_positive(self):
        x = Tensor.randn(1, 3, 32, 32)
        y = Tensor.randn(1, 3, 32, 32)
        assert l1_loss(x, y).numpy() > 0.0

    def test_symmetric(self):
        x = Tensor.randn(1, 3, 32, 32)
        y = Tensor.randn(1, 3, 32, 32)
        assert l1_loss(x, y).numpy() == pytest.approx(l1_loss(y, x).numpy())


class TestSSIMLoss:
    def test_scalar_output(self):
        x = Tensor.rand(1, 3, 64, 64)
        y = Tensor.rand(1, 3, 64, 64)
        assert ssim_loss(x, y).shape == ()

    def test_identical_images(self):
        x = Tensor.rand(1, 3, 32, 32)
        assert ssim_loss(x, x).numpy() == pytest.approx(0.0, abs=1e-5)

    def test_different_images(self):
        x = Tensor.rand(1, 3, 32, 32)
        y = Tensor.rand(1, 3, 32, 32)
        assert ssim_loss(x, y).numpy() > 0.0


class TestPhotometricLoss:
    def test_scalar_output(self):
        x = Tensor.rand(1, 3, 32, 32)
        y = Tensor.rand(1, 3, 32, 32)
        assert photometric_loss(x, y).shape == ()

    def test_identical_images(self):
        x = Tensor.rand(1, 3, 32, 32)
        assert photometric_loss(x, x).numpy() == pytest.approx(0.0, abs=1e-5)

    def test_different_images(self):
        x = Tensor.rand(1, 3, 32, 32)
        y = Tensor.rand(1, 3, 32, 32)
        assert photometric_loss(x, y).numpy() > 0.0
