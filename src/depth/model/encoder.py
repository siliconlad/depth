from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, BatchNorm


class ResidualBlock:
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        self._conv2d_a = Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=stride)
        self._conv2d_b = Conv2d(c_out, c_out, kernel_size=3, padding=1)

        self._batch_norm_a = BatchNorm(c_out)
        self._batch_norm_b = BatchNorm(c_out)

        self._dim_same = c_in == c_out and stride == 1
        if not self._dim_same:
            self._skip_conv2d = Conv2d(c_in, c_out, kernel_size=1, stride=stride)
            self._skip_batch_norm = BatchNorm(c_out)

    def __call__(self, x: Tensor) -> Tensor:
        res = self._conv2d_a(x)
        res = self._batch_norm_a(res)
        res = res.relu()
        res = self._conv2d_b(res)
        res = self._batch_norm_b(res)
        res = res + (x if self._dim_same else self._skip_batch_norm(self._skip_conv2d(x)))
        return res.relu()


class ResidualStem:
    def __init__(self, c_in: int, c_out: int) -> None:
        self._stem_conv2d = Conv2d(c_in, c_out, kernel_size=7, stride=2, padding=3)
        self._stem_batch_norm = BatchNorm(c_out)

    def __call__(self, x: Tensor) -> Tensor:
        temp = self._stem_conv2d(x)
        temp = self._stem_batch_norm(temp)
        temp = temp.relu()
        temp = temp.max_pool2d(kernel_size=(3, 3), stride=2, padding=1)
        assert isinstance(temp, Tensor), "Expected tensor!"
        return temp


class ResidualLayer:
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        self._block_1 = ResidualBlock(c_in, c_out, stride=stride)
        self._block_2 = ResidualBlock(c_out, c_out)

    def __call__(self, x: Tensor) -> Tensor:
        temp = self._block_1(x)
        return self._block_2(temp)


class ResidualEncoder:
    def __init__(self):
        self._stem_conv2d = Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self._stem_batch_norm = BatchNorm(64)

        self._layer_1 = ResidualLayer(64, 64)
        self._layer_2 = ResidualLayer(64, 128, stride=2)
        self._layer_3 = ResidualLayer(128, 256, stride=2)
        self._layer_4 = ResidualLayer(256, 512, stride=2)

    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        :param images: Tensor of shape (B, C, H, W)
        """

        layer_0 = self._stem_batch_norm(self._stem_conv2d(images)).relu()
        stem = layer_0.max_pool2d(kernel_size=(3, 3), stride=2, padding=1)
        assert isinstance(stem, Tensor), "Expected tensor!"

        layer_1 = self._layer_1(stem)
        layer_2 = self._layer_2(layer_1)
        layer_3 = self._layer_3(layer_2)
        layer_4 = self._layer_4(layer_3)

        return layer_0, layer_1, layer_2, layer_3, layer_4
