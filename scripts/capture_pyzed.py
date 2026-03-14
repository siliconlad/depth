"""Capture a single stereo pair from a ZED camera and save to disk.

Saves rectified left/right images and camera calibration parameters.

Usage:
    uv run python scripts/capture_pyzed.py [OUTPUT_DIR] [--resolution RESOLUTION]

Output structure:
    OUTPUT_DIR/
        left/00000.jpg
        right/00000.jpg
        depth/00000.npy
        calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

try:
    import pyzed.sl as sl
except ImportError:
    print("Run `task sync-all` to install ZED python api. Ensure ZED SDK is installed.")
    raise

RESOLUTION_MAP = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}


def capture(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)

    # Create output directories
    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    depth_dir = output_dir / "depth"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Open camera
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER

    res = RESOLUTION_MAP.get(args.resolution.upper())
    if res is None:
        print(f"Unknown resolution: {args.resolution}")
        print(f"Available: {', '.join(RESOLUTION_MAP.keys())}")
        sys.exit(1)
    init_params.camera_resolution = res

    cam = sl.Camera()
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {status}")
        sys.exit(1)

    # Get calibration
    cam_info = cam.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    baseline_m = calib.get_camera_baseline()  # already in meters
    left_params = calib.left_cam
    right_params = calib.right_cam

    h = int(left_params.image_size.height)
    w = int(left_params.image_size.width)

    print(f"Resolution: {w}x{h}")
    print(f"Baseline: {baseline_m:.4f} m")
    print(f"Focal length: fx={left_params.fx:.2f} fy={left_params.fy:.2f} px")

    # Save calibration
    calibration = {
        "image_width": w,
        "image_height": h,
        "baseline_m": baseline_m,
        "left": {
            "fx": left_params.fx,
            "fy": left_params.fy,
            "cx": left_params.cx,
            "cy": left_params.cy,
        },
        "right": {
            "fx": right_params.fx,
            "fy": right_params.fy,
            "cx": right_params.cx,
            "cy": right_params.cy,
        },
    }
    calib_path = output_dir / "calibration.json"
    calib_path.write_text(json.dumps(calibration, indent=2) + "\n")
    print(f"Saved calibration to {calib_path}")

    # Grab a frame
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    err = cam.grab(runtime_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Grab error: {err}")
        cam.close()
        sys.exit(1)

    cam.retrieve_image(left_mat, sl.VIEW.LEFT)
    cam.retrieve_image(right_mat, sl.VIEW.RIGHT)
    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

    # Save images (BGRA -> RGB -> JPEG)
    left_rgb = left_mat.numpy()[:, :, :3][:, :, ::-1]
    right_rgb = right_mat.numpy()[:, :, :3][:, :, ::-1]
    depth_np = depth_mat.numpy()  # float32, meters

    filename = "00000"
    PILImage.fromarray(left_rgb).save(left_dir / f"{filename}.jpg", quality=95)
    PILImage.fromarray(right_rgb).save(right_dir / f"{filename}.jpg", quality=95)
    np.save(depth_dir / f"{filename}.npy", depth_np)

    print(f"Saved left/{filename}.jpg, right/{filename}.jpg, depth/{filename}.npy")

    cam.close()


def main():
    parser = argparse.ArgumentParser(
        description="Capture a stereo pair from a ZED camera."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--resolution",
        default="HD1080",
        choices=list(RESOLUTION_MAP.keys()),
        help="Camera resolution (default: HD1080)",
    )
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
