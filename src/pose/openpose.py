"""
src/pose/openpose.py
====================
Wrapper for IDM-VTON's OpenPose body-pose estimation.

Upstream code:  IDM-VTON/preprocess/openpose/run_openpose.py
Model weights:  ckpt/openpose/ckpts/body_pose_model.pth

The annotator_ckpts_path global (defined in annotator/util.py and re-bound
into annotator/openpose/__init__.py) is monkey-patched at load-time so the
OpenposeDetector finds weights from the caller-supplied ckpt_dir, not from
IDM-VTON's own hardcoded path — important for environments without the
IDM-VTON/ckpt symlink (e.g. local Windows dev).
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union

log = logging.getLogger(__name__)

# ── IDM-VTON path setup ───────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parents[2]
_IDM_DIR      = _ROOT / "IDM-VTON"
_OPENPOSE_DIR = _IDM_DIR / "preprocess" / "openpose"

for _p in [str(_IDM_DIR), str(_OPENPOSE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Module-level singleton cache ──────────────────────────────────────────────
_detector_cache: Dict[str, Any] = {}   # ckpt_ckpts_dir → OpenposeDetector


def _get_detector(ckpt_ckpts_dir: str) -> Any:
    """
    Lazy-load and cache an OpenposeDetector keyed by its ckpts/ directory.

    Patching strategy
    -----------------
    OpenposeDetector.__init__ reads the module-level name
    ``annotator_ckpts_path`` from its own module dict
    (``annotator.openpose.__init__``) at *call time*, not at import time
    (Python resolves globals dynamically).  Therefore patching that module
    attribute before instantiating the detector reliably redirects where it
    looks for ``body_pose_model.pth``.
    """
    if ckpt_ckpts_dir in _detector_cache:
        return _detector_cache[ckpt_ckpts_dir]

    # Import util first so its module object exists in sys.modules before
    # annotator.openpose imports it via ``from annotator.util import ...``.
    import annotator.util as _ann_util
    import annotator.openpose as _ann_op

    # Patch the module-level name in BOTH modules.
    _ann_util.annotator_ckpts_path = ckpt_ckpts_dir
    _ann_op.annotator_ckpts_path   = ckpt_ckpts_dir

    detector = _ann_op.OpenposeDetector()
    _detector_cache[ckpt_ckpts_dir] = detector
    log.info("OpenPose detector loaded from %s", ckpt_ckpts_dir)
    return detector


# ── Public API ────────────────────────────────────────────────────────────────
def extract_openpose_keypoints(
    image_path: Union[str, np.ndarray, Any],
    ckpt_dir: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run OpenPose on a person image and return 18-joint body keypoints.

    run_pipeline.py passes ``ckpt_dir = ./ckpt/openpose``.
    The body_pose_model.pth must be at::

        <ckpt_dir>/ckpts/body_pose_model.pth

    Args:
        image_path: Person image — file path (str), PIL Image, or HWC uint8
                    numpy array.  Full-body shot, ideally 768 × 1024.
        ckpt_dir:   Path to ``ckpt/openpose/`` directory (the ``ckpts/``
                    sub-directory with the .pth file is resolved automatically).
        output_dir: If provided, the pose-skeleton visualisation PNG is saved
                    here as ``openpose_vis.png``.

    Returns:
        dict with two keys:

        ``pose_keypoints_2d``
            List of 18 ``[x_px, y_px]`` floats.  Joints follow the COCO-18
            ordering::

                0 Nose   1 Neck    2 RShoulder  3 RElbow  4 RWrist
                5 LShoulder  6 LElbow  7 LWrist  8 RHip   9 RKnee
                10 RAnkle  11 LHip  12 LKnee  13 LAnkle  14 REye
                15 LEye  16 REar  17 LEar

            Undetected joints are ``[0.0, 0.0]``.

        ``pose_image``
            PIL RGB Image (768 × 1024) of the skeleton overlay, ready for
            diffusion conditioning in Stage 5.
    """
    import cv2
    import torch
    from PIL import Image as PILImage
    from annotator.util import resize_image, HWC3

    # ckpt_dir = "./ckpt/openpose"  →  weights at ./ckpt/openpose/ckpts/body_pose_model.pth
    ckpt_ckpts_dir = str(Path(ckpt_dir) / "ckpts")
    detector = _get_detector(ckpt_ckpts_dir)

    # ── Normalise input to HWC uint8 numpy ────────────────────────────────────
    if isinstance(image_path, str):
        img = np.asarray(PILImage.open(image_path).convert("RGB"))
    elif isinstance(image_path, PILImage.Image):
        img = np.asarray(image_path.convert("RGB"))
    else:
        img = np.asarray(image_path)
    img = img.astype(np.uint8)

    with torch.no_grad():
        input_img           = HWC3(img)
        input_img           = resize_image(input_img, 384)   # shortest side → 384 px
        H, W, _             = input_img.shape                # expected: 512 × 384
        pose, detected_map  = detector(input_img, hand_and_face=False)

    # ── Guard: no person detected ──────────────────────────────────────────────
    if not pose["bodies"]["subset"]:
        log.warning("OpenPose: no person detected — returning zero keypoints.")
        zero_kpts = [[0.0, 0.0]] * 18
        blank_img = PILImage.new("RGB", (768, 1024), (0, 0, 0))
        return {"pose_keypoints_2d": zero_kpts, "pose_image": blank_img}

    # ── Build ordered 18-joint candidate list (IDM-VTON convention) ───────────
    candidate = list(pose["bodies"]["candidate"])
    subset    = list(pose["bodies"]["subset"][0][:18])

    for i in range(18):
        if subset[i] == -1:
            # joint not detected — insert a placeholder at position i
            candidate.insert(i, [0, 0])
            for j in range(i, 18):
                if subset[j] != -1:
                    subset[j] += 1
        elif subset[i] != i:
            candidate.pop(i)
            for j in range(i, 18):
                if subset[j] != -1:
                    subset[j] -= 1

    candidate = candidate[:18]

    # Scale normalised [0–1] → absolute pixel co-ordinates at (W × H)
    for i in range(18):
        candidate[i] = [
            float(candidate[i][0]) * W,
            float(candidate[i][1]) * H,
        ]

    # ── Build 768 × 1024 pose visualisation image ──────────────────────────────
    pose_vis_bgr = cv2.resize(
        detected_map.astype(np.uint8),
        (768, 1024),
        interpolation=cv2.INTER_LANCZOS4,
    )
    pose_pil = PILImage.fromarray(
        cv2.cvtColor(pose_vis_bgr, cv2.COLOR_BGR2RGB)
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "openpose_vis.png")
        pose_pil.save(save_path)
        log.info("Pose visualisation saved → %s", save_path)

    return {
        "pose_keypoints_2d": candidate,   # list[18] of [x_px, y_px]
        "pose_image":        pose_pil,    # PIL RGB 768 × 1024
    }
