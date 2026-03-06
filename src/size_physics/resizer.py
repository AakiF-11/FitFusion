"""
src/size_physics/resizer.py
Thin re-export of IDM-VTON/size_aware_pipeline.py GarmentResizer.
Physically resizes flat garment images based on size ratios.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))

from size_aware_pipeline import GarmentResizer  # noqa: F401
