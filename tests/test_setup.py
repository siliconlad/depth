import pytest
import numpy as np
from tinygrad.device import Device
from tinygrad.tensor import Tensor


def test_setup():
    assert Device.DEFAULT == "NV"
    result = (Tensor.randn(2, 3) @ Tensor.randn(3, 2)).numpy()
    assert isinstance(result, np.ndarray)


@pytest.mark.camera
def test_camera():
    try:
        import pyzed.sl as sl
    except ImportError:
        pytest.fail("Could not import pyzed. Run task sync-all to install.")

    cam = sl.Camera()
    status = cam.open(sl.InitParameters())
    assert status == sl.ERROR_CODE.SUCCESS, "Faqiled to open camera"
