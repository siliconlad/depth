# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///

"""Install the ZED Python API (pyzed) into the project's uv environment.

Usage:
    uv run --script scripts/install_zed.py
    uv run --script scripts/install_zed.py --venv .venv

Downloads the correct pyzed wheel for the target venv's Python version
and installs it using `uv pip install`.
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

ARCH_VERSION = platform.machine()
BASE_URL = "https://download.stereolabs.com/zedsdk/"


def get_target_python_version(venv_path: Path) -> tuple[str, str]:
    """Get the Python major.minor version from the target venv."""
    if sys.platform == "win32":
        python_bin = venv_path / "Scripts" / "python.exe"
    else:
        python_bin = venv_path / "bin" / "python"

    if not python_bin.exists():
        print(f"ERROR: Python not found at {python_bin}")
        print("Make sure the venv exists. Run 'uv sync' first.")
        sys.exit(1)

    output = subprocess.check_output(
        [str(python_bin), "-c", "import platform; print(platform.python_version())"],
        text=True,
    ).strip()
    parts = output.split(".")
    return parts[0], parts[1]


def get_zed_sdk_version() -> tuple[str, str]:
    """Read ZED SDK version from the installed SDK headers."""
    if sys.platform == "win32":
        zed_path = os.getenv("ZED_SDK_ROOT_DIR")
        if zed_path is None:
            print("ERROR: You must install the ZED SDK.")
            sys.exit(1)
        include_path = zed_path + "/include"
    elif "linux" in sys.platform:
        zed_path = "/usr/local/zed"
        if not os.path.isdir(zed_path):
            print("ERROR: You must install the ZED SDK.")
            sys.exit(1)
        include_path = zed_path + "/include"
    else:
        print(f"ERROR: Unsupported platform: {sys.platform}")
        sys.exit(1)

    # Try both header locations
    for header in ["sl/Camera.hpp", "sl_zed/defines.hpp"]:
        header_path = os.path.join(include_path, header)
        if os.path.isfile(header_path):
            with open(header_path, "r", encoding="utf-8") as f:
                data = f.read()
            major_match = re.search(r"ZED_SDK_MAJOR_VERSION (.*)", data)
            minor_match = re.search(r"ZED_SDK_MINOR_VERSION (.*)", data)
            if major_match and minor_match:
                return major_match.group(1).strip(), minor_match.group(1).strip()

    print("ERROR: Could not determine ZED SDK version from headers.")
    sys.exit(1)


def check_valid_wheel(file_path: str) -> bool:
    """Check if a file is a valid .whl (ZIP archive, >150KB)."""
    try:
        file_size = os.stat(file_path).st_size / 1000.0
    except FileNotFoundError:
        return False

    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
        is_zip = header == b"PK\x03\x04"
    except Exception:
        is_zip = False

    return (file_size > 150) and is_zip


def main():
    parser = argparse.ArgumentParser(
        description="Install the ZED Python API (pyzed) into the project's uv environment."
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Path to the target virtual environment (default: .venv)",
    )
    args = parser.parse_args()

    venv_path = Path(args.venv).resolve()
    if not venv_path.exists():
        print(f"ERROR: venv not found at {venv_path}")
        print("Run 'uv sync' first to create it.")
        sys.exit(1)

    # Check architecture
    arch = platform.architecture()[0]
    if arch != "64bit":
        print(f"ERROR: 64-bit Python required, found {arch}")
        sys.exit(1)

    # Detect target Python version from the venv
    py_major, py_minor = get_target_python_version(venv_path)

    # Detect ZED SDK version
    sdk_major, sdk_minor = get_zed_sdk_version()

    # Determine platform strings
    if "linux" in sys.platform:
        if "aarch64" in ARCH_VERSION:
            os_version = "linux_aarch64"
        else:
            os_version = "linux_x86_64"
        whl_platform = "linux"
    elif sys.platform == "win32":
        os_version = f"win_{ARCH_VERSION.lower()}"
        whl_platform = "win"
    else:
        print(f"ERROR: Unsupported platform: {sys.platform}")
        sys.exit(1)

    print(f"Target venv:  {venv_path}")
    print(f"Python:       {py_major}.{py_minor}")
    print(f"ZED SDK:      {sdk_major}.{sdk_minor}")
    print(f"Platform:     {os_version}")

    # Build wheel filename and URL
    cp_tag = f"cp{py_major}{py_minor}"
    whl_name = f"pyzed-{sdk_major}.{sdk_minor}-{cp_tag}-{cp_tag}-{whl_platform}_{ARCH_VERSION.lower()}.whl"
    whl_url = f"{BASE_URL}{sdk_major}.{sdk_minor}/whl/{os_version}/{whl_name}"

    print(f"\n-> Downloading {whl_url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        whl_path = os.path.join(tmpdir, whl_name)

        try:
            r = requests.get(whl_url, allow_redirects=True)
            r.raise_for_status()
            with open(whl_path, "wb") as f:
                f.write(r.content)
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to download wheel: {e}")
            sys.exit(1)

        if not check_valid_wheel(whl_path):
            print(
                "\nERROR: Downloaded file is not a valid wheel."
                "\nNo pyzed wheel available for this configuration."
                "\nIt can be manually installed from source: https://github.com/stereolabs/zed-python-api"
            )
            sys.exit(1)

        print(f"-> Installing into {venv_path}")
        try:
            subprocess.check_call([
                "uv", "pip", "install",
                "--python", str(venv_path / "bin" / "python"),
                "--force-reinstall",
                whl_path,
            ])
        except subprocess.CalledProcessError:
            print("ERROR: uv pip install failed.")
            sys.exit(1)

    print("\nDone! pyzed is now installed in your uv environment.")


if __name__ == "__main__":
    main()
