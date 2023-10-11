"""colormap palettes for your palate"""

from .models import TastyColorMap, TastyColorBar
from .core import cook_tcmap, pair_tcbar

__version__ = "0.0.0"

__all__ = ["cook_tcmap", "pair_tcbar", "TastyColorMap", "TastyColorBar"]
