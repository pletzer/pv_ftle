
from .palm_ftle import cli  # optional: export CLI helper
# from ._ftlecpp import ...  # re-export selected C++ symbols if desired

# Optional version binding:
try:
    from .__version__ import __version__
except Exception:
    __version__ = "0.0.0"

