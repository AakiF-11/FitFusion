"""
src/pose — Pose estimation wrappers
Wraps: IDM-VTON/generate_densepose.py + openpose ckpt
"""
from .openpose import extract_openpose_keypoints
from .densepose import generate_densepose_map

__all__ = ["extract_openpose_keypoints", "generate_densepose_map"]
