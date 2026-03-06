"""
src/catalog/brand.py
Thin re-export of IDM-VTON/brand_catalog.py.
Handles B2B garment catalog ingestion and size-to-model matching.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))

from brand_catalog import BrandCatalog  # noqa: F401
