"""
Lightweight stub to avoid importing the real matplotlib.

On some macOS environments, importing matplotlib can crash the Python process during
font cache initialization. This project does not require plotting to run experiments,
so we provide a minimal stub to satisfy accidental imports without side effects.
"""

__version__ = "0.0-stub"

class MatplotlibStubError(RuntimeError):
    pass

