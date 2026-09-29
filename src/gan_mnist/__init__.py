"""A small unconditional GAN on MNIST.

The package holds the logic; ``scripts/`` holds everything that is not
library code, and the report is built from files the runs wrote rather
than from anything imported here.
"""

from ._config import RunConfig, resolve_device

__all__ = ["RunConfig", "resolve_device"]
__version__ = "1.0.0"
