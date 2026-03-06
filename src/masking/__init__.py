"""
src/masking — Segmentation, confidence scoring, adaptive masks
Wraps: fitfusion/masking/compositor.py
       fitfusion/masking/confidence_scorer.py
       IDM-VTON/size_adaptive_mask.py
"""
from .compositor import restore_original_skin
from .confidence import score_mask_validity
from .adaptive_mask import generate_adaptive_mask
from .human_parsing import run_human_parsing

__all__ = [
    "restore_original_skin",
    "score_mask_validity",
    "generate_adaptive_mask",
    "run_human_parsing",
]
