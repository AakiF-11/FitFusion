"""
src/masking/human_parsing.py
============================
Wrapper for IDM-VTON's SCHP dual-ONNX human-parsing pipeline.

Upstream code:  IDM-VTON/preprocess/humanparsing/
Models:
    ckpt/humanparsing/parsing_atr.onnx  — ATR 18-class body segmentation
    ckpt/humanparsing/parsing_lip.onnx  — LIP model for neck-region refinement

SCHP class-label map (uint8, 0–18)
-----------------------------------
 0  Background   1  Hat          2  Hair         3  Glove
 4  Sunglasses   5  Upper-cloth  6  Dress        7  Coat
 8  Socks        9  Pants       10  Jumpsuits    11  Scarf
12  Skirt       13  Face        14  Left-arm     15  Right-arm
16  Left-leg    17  Right-leg   18  Neck  ← added by LIP refinement

Note on SimpleFolderDataset
---------------------------
``onnx_inference`` internally creates a ``SimpleFolderDataset``.  That
class's ``__init__`` explicitly handles ``isinstance(root, Image.Image)``,
so a PIL Image can be passed directly as the ``input_dir`` argument —
avoiding any temp-directory I/O.
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Union

log = logging.getLogger(__name__)

# ── IDM-VTON path setup ───────────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parents[2]
_HUMANPARSING  = _ROOT / "IDM-VTON" / "preprocess" / "humanparsing"

for _p in [
    str(_ROOT / "IDM-VTON"),
    str(_HUMANPARSING),
    str(_HUMANPARSING / "datasets"),
    str(_HUMANPARSING / "utils"),
    str(_HUMANPARSING / "networks"),
    str(_HUMANPARSING / "modules"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Module-level ONNX session cache ───────────────────────────────────────────
_session_cache: Dict[str, tuple] = {}   # ckpt_dir → (atr_session, lip_session)


def _get_sessions(ckpt_dir: str) -> tuple:
    """
    Lazy-create and cache the two ONNX Runtime sessions.

    We create the sessions directly from explicit model paths rather than
    using the ``Parsing`` class from ``run_parsing.py``, which hardcodes
    the ckpt path relative to its own file location.  Owning the session
    creation here gives the caller full control over the weight directory.
    """
    if ckpt_dir in _session_cache:
        return _session_cache[ckpt_dir]

    import onnxruntime as ort

    atr_path = os.path.join(ckpt_dir, "parsing_atr.onnx")
    lip_path = os.path.join(ckpt_dir, "parsing_lip.onnx")

    if not os.path.exists(atr_path):
        raise FileNotFoundError(
            f"parsing_atr.onnx not found at '{atr_path}'.  "
            f"Ensure ckpt/humanparsing/ contains both ONNX models."
        )
    if not os.path.exists(lip_path):
        raise FileNotFoundError(
            f"parsing_lip.onnx not found at '{lip_path}'.  "
            f"Ensure ckpt/humanparsing/ contains both ONNX models."
        )

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

    # ATR model — primary 18-class body segmentation
    atr_session = ort.InferenceSession(
        atr_path,
        sess_options=sess_opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    # LIP model — neck-mask refinement (run on CPU to avoid VRAM contention)
    lip_session = ort.InferenceSession(
        lip_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    log.info("Human-parsing ONNX sessions loaded from '%s'", ckpt_dir)
    _session_cache[ckpt_dir] = (atr_session, lip_session)
    return atr_session, lip_session


# ── Public API ────────────────────────────────────────────────────────────────
def run_human_parsing(
    image: Union[str, np.ndarray, Any],
    ckpt_dir: str,
) -> np.ndarray:
    """
    Run SCHP dual-ONNX human parsing on the person image.

    run_pipeline.py passes ``ckpt_dir = ./ckpt/humanparsing``.
    Both ONNX models are required::

        <ckpt_dir>/parsing_atr.onnx
        <ckpt_dir>/parsing_lip.onnx

    Args:
        image:    Person image — file path (str), PIL Image, or HWC uint8
                  numpy array.
        ckpt_dir: Path to ``ckpt/humanparsing/`` directory.

    Returns:
        schp_mask: (H, W) uint8 numpy array of SCHP integer class labels
                   in the range 0–18.  The spatial resolution matches the
                   input image (after the ATR model's internal 512 × 512
                   affine warp and inverse transform).
    """
    from PIL import Image as PILImage
    from parsing_api import onnx_inference

    atr_session, lip_session = _get_sessions(ckpt_dir)

    # ── Normalise input to PIL RGB ─────────────────────────────────────────────
    if isinstance(image, str):
        pil_img = PILImage.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = PILImage.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    # Pass PIL Image directly — SimpleFolderDataset detects isinstance(root, Image.Image)
    # and handles it without any filesystem I/O.
    output_img, _face_mask = onnx_inference(atr_session, lip_session, pil_img)

    # output_img is a paletted 'P'-mode PIL Image where pixel values ARE the
    # SCHP class indices (0–18).  np.asarray on a 'P'-mode image returns the
    # raw 2-D array of those index values.
    return np.asarray(output_img)
