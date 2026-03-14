import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
from tinygrad.tensor import Tensor
from PIL import Image
from pydantic import BaseModel, ConfigDict


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    fx: float
    fy: float
    cx: float
    cy: float


class StereoCalibration(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    image_width: int
    """Width of the image."""

    image_height: int
    """Height of the image."""

    baseline: float
    """Baseline between the stereo camera pair in meters."""

    left: CameraIntrinsics
    """Camera intrinsics of the left camera in the stereo pair."""

    right: CameraIntrinsics
    """Camera intrinsics of the right camera in the stereo pair."""


@dataclass
class StereoImages:
    left: Tensor
    """Image from the left camera of the stereo pair."""

    right: Tensor
    """Image from the right camera of the stereo pair."""

    depth: Tensor | None
    """Depth image from the stereo pair if available."""

    calib: StereoCalibration
    "Calibration of the stereo pair."


class ImageLoader:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._left_data_dir = self._data_dir / 'left'
        self._right_data_dir = self._data_dir / 'right'
        self._depth_data_dir = self._data_dir / 'depth'

        # Read the calibration information
        with (self._data_dir / 'calibration.json').open('r') as f:
            self._calibration = StereoCalibration.model_validate(json.load(f))

        self._num_images = self._discover_images(self._left_data_dir)
        num_right = self._discover_images(self._right_data_dir)
        assert self._num_images == num_right, 'Different number of left and right images!'

    def _discover_images(self, image_dir: Path) -> int:
        images = glob((image_dir / '*.jpg').as_posix())
        idxs = [int(Path(p).stem) for p in images]
        assert sorted(idxs) == list(range(len(idxs))), 'Images not contiguous!'
        return len(idxs)

    def _load_image(self, image_path: Path) -> Tensor:
        img = np.array(Image.open(image_path)).astype(np.float32) / 255.0
        return Tensor(img.transpose(2, 0, 1))

    def _load_depth(self, image_path: Path) -> Tensor | None:
        if image_path.exists():
            return Tensor(np.load(image_path))
        return None

    def __len__(self) -> int:
        return self._num_images

    def __getitem__(self, idx: int) -> StereoImages:
        return StereoImages(
            left=self._load_image(self._left_data_dir / f'{idx:05d}.jpg'),
            right=self._load_image(self._right_data_dir / f'{idx:05d}.jpg'),
            depth=self._load_depth(self._depth_data_dir / f'{idx:05d}.npy'),
            calib=self._calibration
        )
