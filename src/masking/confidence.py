"""
src/masking/confidence.py
Thin re-export of fitfusion/masking/confidence_scorer.py.
Validates mask geometry against OpenPose skeleton.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fitfusion.masking.confidence_scorer import score_mask_validity  # noqa: F401
