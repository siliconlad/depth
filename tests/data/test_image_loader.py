import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from depth.data.image_loader import (
    CameraIntrinsics,
    ImageLoader,
    StereoCalibration,
    StereoImages,
)

W = 1080
H = 1920


def make_data_dir(root, *, left_idxs=(0,), right_idxs=(0,), depth_idxs=(0,)):
    # Write rgb images to appropriate directory
    for subdir, idxs in (("left", left_idxs), ("right", right_idxs)):
        d = root / subdir
        d.mkdir(parents=True, exist_ok=True)
        for idx in idxs:
            img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            Image.fromarray(img).save(d / f"{idx:05d}.jpg")

    # Write depth image if requested
    depth_dir = root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    for idx in depth_idxs:
        np.save(depth_dir / f"{idx:05d}.npy", np.random.rand(H, W).astype(np.float32))

    # Write calibration json file
    (root / "calibration.json").write_text(
        StereoCalibration(
            image_width=W,
            image_height=H,
            baseline=0.063,
            left=CameraIntrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0),
            right=CameraIntrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0),
        ).model_dump_json()
    )


class TestImageLoader:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        make_data_dir(root)
        self.loader = ImageLoader(root)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_len(self):
        assert len(self.loader) == 1

    def test_returns_stereo_images(self):
        assert isinstance(self.loader[0], StereoImages)

    def test_image_shape_and_dtype(self):
        sample = self.loader[0]
        assert sample.left.shape == (3, H, W)
        assert sample.right.shape == (3, H, W)
        assert sample.left.dtype.name == "float"
        assert sample.right.dtype.name == "float"

    def test_image_normalized(self):
        left = self.loader[0].left.numpy()
        assert left.min() >= 0.0
        assert left.max() <= 1.0

    def test_depth_loaded(self):
        sample = self.loader[0]
        assert sample.depth is not None
        assert sample.depth.shape == (H, W)

    def test_calibration(self):
        calib = self.loader[0].calib
        assert calib.image_width == W
        assert calib.image_height == H
        assert calib.baseline == 0.063
        assert calib.left.fx == 500.0


class TestImageLoaderNoDepth:
    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        make_data_dir(root, depth_idxs=())
        self.loader = ImageLoader(root)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_depth_is_none(self):
        assert self.loader[0].depth is None


def test_non_contiguous_left():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        make_data_dir(root, left_idxs=(0, 2), right_idxs=(0, 2))
        with pytest.raises(AssertionError, match="not contiguous"):
            ImageLoader(root)


def test_non_contiguous_right():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        make_data_dir(root, left_idxs=(0, 1), right_idxs=(0, 3))
        with pytest.raises(AssertionError, match="not contiguous"):
            ImageLoader(root)


def test_mismatched_left_right_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        make_data_dir(root, left_idxs=(0, 1), right_idxs=(0,))
        with pytest.raises(AssertionError, match="Different number"):
            ImageLoader(root)
