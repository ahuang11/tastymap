"""Color palettes for your palate"""

from .core import cook_tmap, pair_tbar
from .models import TastyBar, TastyMap

try:
    from .ui import TastyKitchen
except ImportError:
    pass

__version__ = "0.4.1"

__all__ = ["cook_tmap", "pair_tbar", "TastyMap", "TastyBar", "TastyKitchen"]
