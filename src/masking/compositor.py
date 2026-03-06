"""
src/masking/compositor.py
Thin re-export of fitfusion/masking/compositor.py.
Restores original skin pixels over the generated try-on output.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fitfusion.masking.compositor import restore_original_skin  # noqa: F401
