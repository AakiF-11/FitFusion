"""
src/inference — IDM-VTON model loading + inference execution
Wraps: IDM-VTON/src/* (UNets, TryonPipeline)
       IDM-VTON/run_tryon.py
       IDM-VTON/tryon_api.py
"""
from .model_loader import load_idm_vton_pipeline
from .tryon import run_tryon

__all__ = ["load_idm_vton_pipeline", "run_tryon"]
