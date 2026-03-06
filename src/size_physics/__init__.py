"""
src/size_physics — Size chart math + garment physics
Wraps: IDM-VTON/size_charts.py
       IDM-VTON/size_aware_vton.py
       IDM-VTON/size_aware_pipeline.py (GarmentResizer)
       IDM-VTON/tps_warper.py
"""
from .charts import compute_size_ratio, get_garment_dimensions
from .physics import SizeAwareVTON
from .resizer import GarmentResizer

__all__ = ["compute_size_ratio", "get_garment_dimensions", "SizeAwareVTON", "GarmentResizer"]
