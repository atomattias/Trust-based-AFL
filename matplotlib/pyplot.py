"""
Minimal matplotlib.pyplot stub.

If any plotting code is executed, raise a clear error instead of crashing the process.
"""

from . import MatplotlibStubError

def __getattr__(name):
    raise MatplotlibStubError(
        "Plotting is disabled in this environment (matplotlib stub in use)."
    )

