"""
src/size_physics/charts.py
Thin re-export of IDM-VTON/size_charts.py.
Provides size measurement database and ratio computation.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "IDM-VTON"))

from size_charts import compute_size_ratio, get_garment_dimensions  # noqa: F401
