import numpy as np
from tinygrad.device import Device
from tinygrad.tensor import Tensor


def test_setup():
    assert Device.DEFAULT == "NV"
    result = (Tensor.randn(2,3) @ Tensor.randn(3,2)).numpy()
    assert isinstance(result, np.ndarray)
