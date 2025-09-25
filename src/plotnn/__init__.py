"""plotnn: Python API for LaTeX/TikZ neural network diagrams.

Public API:
- Diagram: Main builder class.
- Element subclasses: Input, Conv, Pool, etc.
- Blocks: TwoConvPoolBlock, UnconvBlock, etc.
"""

from .blocks import *  # noqa: F403
from .layers import generate_pdf, generate_png, generate_svg  # noqa: F401

__version__ = "0.1.1"
