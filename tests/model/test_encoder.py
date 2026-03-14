from tinygrad.tensor import Tensor

from depth.model.encoder import ResidualEncoder


class TestImageEncoder:
    def setup_method(self):
        self.encoder = ResidualEncoder()

    def test_output_count(self):
        x = Tensor.randn(1, 3, 256, 256)
        out = self.encoder(x)
        assert len(out) == 5

    def test_output_channels(self):
        x = Tensor.randn(1, 3, 256, 256)
        stem, l1, l2, l3, l4 = self.encoder(x)
        assert stem.shape == (1, 64, 128, 128)
        assert l1.shape == (1, 64, 64, 64)
        assert l2.shape == (1, 128, 32, 32)
        assert l3.shape == (1, 256, 16, 16)
        assert l4.shape == (1, 512, 8, 8)

    def test_batch_preserved(self):
        x = Tensor.randn(4, 3, 128, 128)
        stem, l1, l2, l3, l4 = self.encoder(x)
        for feat in (stem, l1, l2, l3, l4):
            assert feat.shape[0] == 4
