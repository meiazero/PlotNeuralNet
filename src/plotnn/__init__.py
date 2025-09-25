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
from .compiler import *  # noqa: F401
from .layers import *  # noqa: F401
from .renderer import *  # noqa: F401
from .templates import *  # noqa: F401

__version__ = "0.2.0"
