"""plotnn: Python API for LaTeX/TikZ neural network diagrams.

Public API:
- Diagram: Main builder class.
- Element subclasses: Input, Conv, Pool, etc.
- Blocks: TwoConvPoolBlock, UnconvBlock, etc.
- Renderers: DiagramRenderer for advanced usage.
- Templates: LaTeXTemplate for customization.
- Compilers: LaTeXCompiler, FormatConverter for low-level control.
"""

from .blocks import *  # noqa: F403
from .compiler import FormatConverter, LaTeXCompiler  # noqa: F401
from .layers import generate_pdf, generate_png, generate_svg  # noqa: F401
from .renderer import DiagramRenderer  # noqa: F401
from .templates import LaTeXTemplate  # noqa: F401
from pathlib import Path
from .blocks import Diagram


__version__ = "0.2.0"
