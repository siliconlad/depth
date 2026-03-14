from tinygrad.tensor import Tensor

from depth.model.decoder import Decoder
from depth.model.encoder import ResidualEncoder


class TestDecoder:
    def setup_method(self):
        self.encoder = ResidualEncoder()
        self.decoder = Decoder()

    def test_output_count(self):
        x = Tensor.randn(1, 3, 256, 256)
        out = self.decoder(self.encoder(x))
        assert len(out) == 4

    def test_output_shapes(self):
        x = Tensor.randn(1, 3, 256, 256)
        final, stem, layer_1, layer_2 = self.decoder(self.encoder(x))
        # final: /1, stem: /2, layer_1: /4, layer_2: /8
        assert final.shape == (1, 1, 256, 256)
        assert stem.shape == (1, 1, 128, 128)
        assert layer_1.shape == (1, 1, 64, 64)
        assert layer_2.shape == (1, 1, 32, 32)

    def test_sigmoid_range(self):
        x = Tensor.randn(1, 3, 128, 128)
        for head in self.decoder(self.encoder(x)):
            vals = head.numpy()
            assert vals.min() >= 0.0
            assert vals.max() <= 1.0

    def test_batch_preserved(self):
        x = Tensor.randn(2, 3, 128, 128)
        for head in self.decoder(self.encoder(x)):
            assert head.shape[0] == 2

    def test_single_channel_output(self):
        x = Tensor.randn(1, 3, 128, 128)
        for head in self.decoder(self.encoder(x)):
            assert head.shape[1] == 1
