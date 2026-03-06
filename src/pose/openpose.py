"""
src/pose/openpose.py
Wraps the OpenPose body_pose_model.pth checkpoint
to extract 18-point skeleton keypoints from a person image.
"""
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))
sys.path.insert(0, str(_ROOT / "IDM-VTON" / "preprocess"))


def extract_openpose_keypoints(image_path: str, ckpt_dir: str) -> Any:
    """
    Run OpenPose on the input image and return body keypoints.

    Args:
        image_path: Path to the person image.
        ckpt_dir:   Path to ckpt/openpose/ directory.

    Returns:
        keypoints: List/array of (x, y) joint coordinates.
                   Returns None if pose extraction fails.
    """
    # TODO: integrate IDM-VTON's openpose wrapper
    # from openpose import OpenPose
    # model = OpenPose(ckpt_dir)
    # return model.predict(image_path)
    raise NotImplementedError("OpenPose integration — implement in next sprint.")
