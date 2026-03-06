"""
src/preprocessing — Input standardization
Wraps: fitfusion/utils/preprocessing.py (background, neckline)
       + image dimension validation
"""
from .background import standardize_background
from .validator import validate_image

__all__ = ["standardize_background", "validate_image"]
