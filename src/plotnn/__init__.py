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
from .blocks import (
    Activation,
    Concat,
    DepthwiseConv,
    Flatten,
    GenericBox,
    Normalization,
    RNNCell,
    SeparableConv,
    Split,
    SqueezeExcitation,
    TransformerBlock,
    TransposeConv,
)
from .compiler import *  # noqa: F401
from .layers import *  # noqa: F401
from .renderer import *  # noqa: F401
from .templates import *  # noqa: F401

__all__ = [
    "Diagram",
    "Input",
    "Conv",
    "ConvConvRelu",
    "ConvRes",
    "ConvSoftMax",
    "Pool",
    "UnPool",
    "Connection",
    "Skip",
    "Dense",
    "SoftMax",
    "Sum",
    "TokenEmbedding",
    "PositionalEncoding",
    "MultiHeadAttention",
    "FeedForward",
    "LayerNorm",
    "Add",
    "OutputProjection",
    "Dropout",
    "Activation",
    "Normalization",
    "RNNCell",
    "GenericBox",
    "Concat",
    "Split",
    "DepthwiseConv",
    "SeparableConv",
    "TransposeConv",
    "Flatten",
    "SqueezeExcitation",
    "TransformerBlock",
]
