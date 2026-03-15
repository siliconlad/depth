# Data Loaders

## Image Loader

The `ImageLoader` loads stereo image pairs from a directory on disk. Use
`scripts/capture_pyzed.py` (or `task snapshot`) to capture data in this format.

### Directory Structure

```
data_dir/
    left/
        00000.jpg
        00001.jpg
        ...
    right/
        00000.jpg
        00001.jpg
        ...
    depth/              # optional
        00000.npy
        00001.npy
        ...
    calibration.json
```

- **`left/` and `right/`**: Rectified stereo image pairs as JPEG files. Filenames
  must be zero-padded 5-digit indices starting from `00000`, with no gaps.
  Left and right directories must contain the same number of images.
- **`depth/`**: Optional ground-truth depth maps as NumPy `.npy` files (float32,
  values in meters). If a depth file is missing for a given index, the loader
  returns `None` for that sample's depth.
- **`calibration.json`**: Camera calibration parameters shared across all images
  in the directory.

### calibration.json Format

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "baseline": 0.063,
  "left": {
    "fx": 1509.27,
    "fy": 1509.27,
    "cx": 941.83,
    "cy": 573.47
  },
  "right": {
    "fx": 1509.27,
    "fy": 1509.27,
    "cx": 941.83,
    "cy": 573.47
  }
}
```

| Field | Description |
|---|---|
| `image_width`, `image_height` | Image dimensions in pixels |
| `baseline` | Distance between left and right cameras in meters |
| `fx`, `fy` | Focal length in pixels (horizontal and vertical) |
| `cx`, `cy` | Principal point (optical center) in pixels |

### Usage

```python
from pathlib import Path
from depth.data.image_loader import ImageLoader

loader = ImageLoader(Path("data"))
print(len(loader))       # number of stereo pairs

sample = loader[0]
sample.left              # Tensor (3, H, W), float32, [0, 1]
sample.right             # Tensor (3, H, W), float32, [0, 1]
sample.depth             # Tensor (H, W), float32, meters — or None
sample.calib.baseline    # float, meters
sample.calib.left.fx     # float, pixels
```
