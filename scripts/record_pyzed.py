"""Record stereo pairs from a ZED camera and save as an MCAP file.

Records rectified left/right compressed images along with camera calibration info.

Usage:
    uv run --script scripts/record_pyzed.py output.mcap [--duration SECONDS] [--resolution RESOLUTION]
    uv run --script scripts/record_pyzed.py output.mcap --svo input.svo

The --svo flag converts an existing SVO file instead of recording live.
"""

import argparse
import io
import signal
import sys
import time
from pathlib import Path

from PIL import Image as PILImage
from pybag.mcap_writer import McapFileWriter
from pybag.ros2.humble import builtin_interfaces, sensor_msgs, std_msgs

try:
    import pyzed.sl as sl
except ImportError:
    print("Run `task sync` to install ZED python api. Ensure ZED SDK is installed.")
    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESOLUTION_MAP = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
}


def make_header(frame_id: str, timestamp_ns: int) -> std_msgs.Header:
    sec = int(timestamp_ns // 1_000_000_000)
    nanosec = int(timestamp_ns % 1_000_000_000)
    return std_msgs.Header(
        stamp=builtin_interfaces.Time(sec=sec, nanosec=nanosec),
        frame_id=frame_id,
    )


def make_camera_info(
    cam_params: sl.CameraParameters,
    frame_id: str,
    timestamp_ns: int,
    baseline_m: float,
) -> sensor_msgs.CameraInfo:
    """Build a CameraInfo message from ZED calibration parameters.

    For rectified images:
    - D (distortion) is all zeros (rectification removes distortion)
    - R (rectification matrix) is identity
    - K is the intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    - P is the projection matrix [fx, 0, cx, Tx; 0, fy, cy, 0; 0, 0, 1, 0]
      where Tx = -fx * baseline for the right camera, 0 for left
    """
    fx = cam_params.fx
    fy = cam_params.fy
    cx = cam_params.cx
    cy = cam_params.cy
    h = int(cam_params.image_size.height)
    w = int(cam_params.image_size.width)

    is_right = frame_id == "right"
    tx = -fx * baseline_m if is_right else 0.0

    return sensor_msgs.CameraInfo(
        header=make_header(frame_id, timestamp_ns),
        height=h,
        width=w,
        distortion_model="plumb_bob",
        d=[0.0, 0.0, 0.0, 0.0, 0.0],
        k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        binning_x=0,
        binning_y=0,
        roi=sensor_msgs.RegionOfInterest(
            x_offset=0,
            y_offset=0,
            height=0,
            width=0,
            do_rectify=False,
        ),
    )


def make_compressed_image(
    mat: sl.Mat,
    frame_id: str,
    timestamp_ns: int,
    quality: int = 95,
) -> sensor_msgs.CompressedImage:
    """Build a CompressedImage message from a ZED Mat (BGRA -> JPEG)."""
    arr = mat.numpy()[:, :, :3]  # BGRA -> BGR
    rgb = arr[:, :, ::-1]  # BGR -> RGB for PIL
    buf = io.BytesIO()
    PILImage.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return sensor_msgs.CompressedImage(
        header=make_header(frame_id, timestamp_ns),
        format="jpeg",
        data=buf.getvalue(),
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True


def record(args: argparse.Namespace) -> None:
    global _stop
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Open ZED camera (live or from SVO) ----
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # we only need images
    init_params.coordinate_units = sl.UNIT.METER

    if args.svo:
        print(f"Reading from SVO: {args.svo}")
        init_params.set_from_svo_file(args.svo)
        init_params.svo_real_time_mode = False
    else:
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

    # ---- Get calibration ----
    cam_info = cam.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    baseline_m = calib.get_camera_baseline()  # already in meters (coordinate_units=METER)
    left_params = calib.left_cam
    right_params = calib.right_cam

    print(f"Resolution: {int(left_params.image_size.width)}x{int(left_params.image_size.height)}")
    print(f"Baseline: {baseline_m:.4f} m")
    print(f"Focal length: fx={left_params.fx:.2f} fy={left_params.fy:.2f} px")

    # ---- Pre-register MCAP channels with schemas ----
    writer = McapFileWriter.open(str(output_path), profile="ros2", chunk_compression="lz4")

    writer.add_channel("/camera/left/image_rect/compressed", schema=sensor_msgs.CompressedImage)
    writer.add_channel("/camera/right/image_rect/compressed", schema=sensor_msgs.CompressedImage)
    writer.add_channel("/camera/left/camera_info", schema=sensor_msgs.CameraInfo)
    writer.add_channel("/camera/right/camera_info", schema=sensor_msgs.CameraInfo)

    # ---- Grab loop ----
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    frame_count = 0
    start_time = time.monotonic()
    duration = args.duration if not args.svo else float("inf")

    print("Recording... (Ctrl+C to stop)")

    try:
        while not _stop:
            err = cam.grab(runtime_params)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("End of SVO file reached.")
                break
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"Grab error: {err}")
                continue

            elapsed = time.monotonic() - start_time
            if elapsed >= duration:
                break

            # Retrieve rectified images (VIEW.LEFT / VIEW.RIGHT are rectified by default)
            cam.retrieve_image(left_mat, sl.VIEW.LEFT)
            cam.retrieve_image(right_mat, sl.VIEW.RIGHT)

            timestamp_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

            # Build messages
            left_img = make_compressed_image(left_mat, "left", timestamp_ns, args.quality)
            right_img = make_compressed_image(right_mat, "right", timestamp_ns, args.quality)
            left_info = make_camera_info(left_params, "left", timestamp_ns, baseline_m)
            right_info = make_camera_info(right_params, "right", timestamp_ns, baseline_m)

            # Write to MCAP
            writer.write_message("/camera/left/image_rect/compressed", timestamp_ns, left_img)
            writer.write_message("/camera/right/image_rect/compressed", timestamp_ns, right_img)
            writer.write_message("/camera/left/camera_info", timestamp_ns, left_info)
            writer.write_message("/camera/right/camera_info", timestamp_ns, right_info)

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.monotonic() - start_time)
                print(f"\r  Frames: {frame_count} ({fps:.1f} fps)", end="", flush=True)

    finally:
        print(f"\n\nRecorded {frame_count} frames to {output_path}")
        writer.close()
        cam.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Record stereo pairs from a ZED camera to an MCAP file."
    )
    parser.add_argument("output", help="Output MCAP file path")
    parser.add_argument(
        "--duration",
        type=float,
        default=float("inf"),
        help="Recording duration in seconds (default: until Ctrl+C)",
    )
    parser.add_argument(
        "--resolution",
        default="HD1080",
        choices=list(RESOLUTION_MAP.keys()),
        help="Camera resolution (default: HD1080)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)",
    )
    parser.add_argument(
        "--svo",
        default=None,
        help="Path to an SVO file to convert (instead of live recording)",
    )
    args = parser.parse_args()
    record(args)


if __name__ == "__main__":
    main()
