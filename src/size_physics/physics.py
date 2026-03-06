"""
src/size_physics/physics.py
Thin re-export of IDM-VTON/size_aware_vton.py.
Contains SizeAwareVTON — the 7-layer physics engine.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))

from size_aware_vton import SizeAwareVTON, FitType, FitProfile, classify_fit  # noqa: F401
