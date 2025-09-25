from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from .layers import (
    to_activation,
    to_add,
    to_concat,
    to_connection,
    to_conv,
    to_conv_conv_relu,
    to_conv_res,
    to_conv_softmax,
    to_depthwise_conv,
    to_embedding,
    to_feed_forward,
    to_flatten,
    to_generic_box,
    to_input,
    to_layer_norm,
    to_multihead_attention,
    to_normalization,
    to_output_projection,
    to_pool,
    to_positional_encoding,
    to_rnn_cell,
    to_separable_conv,
    to_skip,
    to_softmax,
    to_split,
    to_squeeze_excitation,
    to_sum,
    to_transformer_block,
    to_transpose_conv,
    to_unpool,
)
from .renderer import DiagramRenderer

logger = logging.getLogger(__name__)


class Element(ABC):
    """Base class for diagram elements that generate LaTeX snippets."""

    @abstractmethod
    def build(self) -> list[str]:
        """Generate LaTeX snippets for this element."""
        raise NotImplementedError


@dataclass
class Leaf(Element):
    """Simple element with fixed LaTeX."""

    tex: str

    def build(self) -> list[str]:
        return [self.tex]


# Common Layers as Dataclasses
@dataclass
class Input(Element):
    """Input layer with image."""

    pathfile: str | Path
    to: str = "(-3,0,0)"
    width: int = 8
    height: int = 8
    name: str = "input"
    anchor_scale: float = 0.01

    def __post_init__(self):
        assert self.width > 0, "Width must be positive"
        assert self.height > 0, "Height must be positive"
        if isinstance(self.pathfile, Path):
            self.pathfile = self.pathfile.as_posix()

    def build(self) -> list[str]:
        return [
            to_input(
                str(self.pathfile),
                to=self.to,
                width=self.width,
                height=self.height,
                name=self.name,
                anchor_scale=self.anchor_scale,
            )
        ]


@dataclass
class Conv(Element):
    """Convolutional layer."""

    name: str
    s_filer: int = 256
    n_filer: int = 64
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 40
    depth: int = 40
    caption: str = " "

    def __post_init__(self):
        assert self.width > 0, "Width must be positive"

    def build(self) -> list[str]:
        return [
            to_conv(
                name=self.name,
                s_filer=self.s_filer,
                n_filer=self.n_filer,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class ConvConvRelu(Element):
    """Double Conv + ReLU."""

    name: str
    s_filer: int = 256
    n_filer: tuple[int, int] = (64, 64)
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: tuple[int, int] = (2, 2)
    height: int = 40
    depth: int = 40
    caption: str = " "

    def build(self) -> list[str]:
        return [
            to_conv_conv_relu(
                name=self.name,
                s_filer=self.s_filer,
                n_filer=self.n_filer,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class Pool(Element):
    """Pooling layer."""

    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 32
    depth: int = 32
    opacity: float = 0.5
    caption: str = " "

    def build(self) -> list[str]:
        return [
            to_pool(
                name=self.name,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


@dataclass
class UnPool(Element):
    """Unpooling layer."""

    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 32
    depth: int = 32
    opacity: float = 0.5
    caption: str = " "

    def build(self) -> list[str]:
        return [
            to_unpool(
                name=self.name,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


@dataclass
class ConvRes(Element):
    """Residual Conv."""

    name: str
    s_filer: int = 256
    n_filer: int = 64
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 6
    height: int = 40
    depth: int = 40
    opacity: float = 0.2
    caption: str = " "

    def build(self) -> list[str]:
        return [
            to_conv_res(
                name=self.name,
                s_filer=self.s_filer,
                n_filer=self.n_filer,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


@dataclass
class ConvSoftMax(Element):
    """Conv + SoftMax."""

    name: str
    s_filer: int = 40
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 40
    depth: int = 40
    caption: str = " "

    def build(self) -> list[str]:
        return [
            to_conv_softmax(
                name=self.name,
                s_filer=self.s_filer,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class SoftMax(Element):
    """SoftMax layer."""

    name: str
    s_filer: int = 10
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 2
    height: int = 3
    depth: int = 25
    opacity: float = 0.8
    caption: str = " "

    def build(self) -> list[str]:
        return [
            to_softmax(
                name=self.name,
                s_filer=self.s_filer,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


@dataclass
class Sum(Element):
    """Sum operation."""

    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    radius: float = 2.5
    opacity: float = 0.6

    def build(self) -> list[str]:
        return [
            to_sum(
                self.name, offset=self.offset, to=self.to, radius=self.radius, opacity=self.opacity
            )
        ]


@dataclass
class Connection(Element):
    """Connection between layers."""

    of: str
    to: str

    def build(self) -> list[str]:
        return [to_connection(self.of, self.to)]


@dataclass
class Skip(Element):
    """Skip connection."""

    of: str
    to: str
    pos: float = 1.25

    def build(self) -> list[str]:
        return [to_skip(self.of, self.to, pos=self.pos)]


@dataclass
class Dense(Element):
    """Dense (Fully Connected) layer."""

    name: str
    units: int = 128
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 1
    depth: int = 20
    caption: str = "Dense"

    def build(self) -> list[str]:
        return [
            to_conv(
                name=self.name,
                s_filer=self.units,
                n_filer=1,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


# ---------------- Transformer specific elements -----------------


@dataclass
class TokenEmbedding(Element):
    name: str
    vocab_size: int = 30522
    model_dim: int = 768
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 30
    depth: int = 30
    caption: str = "Embed"

    def build(self) -> list[str]:
        return [
            to_embedding(
                name=self.name,
                vocab_size=self.vocab_size,
                model_dim=self.model_dim,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class PositionalEncoding(Element):
    name: str
    seq_len: int = 512
    model_dim: int = 768
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 30
    depth: int = 30
    caption: str = "PosEnc"

    def build(self) -> list[str]:
        return [
            to_positional_encoding(
                name=self.name,
                seq_len=self.seq_len,
                model_dim=self.model_dim,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class MultiHeadAttention(Element):
    name: str
    heads: int = 8
    model_dim: int = 768
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 2
    height: int = 28
    depth: int = 28
    caption: str = "MHA"

    def build(self) -> list[str]:
        return [
            to_multihead_attention(
                name=self.name,
                heads=self.heads,
                model_dim=self.model_dim,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class FeedForward(Element):
    name: str
    model_dim: int = 768
    hidden_dim: int = 3072
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 2
    height: int = 26
    depth: int = 26
    caption: str = "FFN"

    def build(self) -> list[str]:
        return [
            to_feed_forward(
                name=self.name,
                model_dim=self.model_dim,
                hidden_dim=self.hidden_dim,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class LayerNorm(Element):
    name: str
    model_dim: int = 768
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 20
    depth: int = 20
    caption: str = "LN"

    def build(self) -> list[str]:
        return [
            to_layer_norm(
                name=self.name,
                model_dim=self.model_dim,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class Add(Element):
    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    radius: float = 2.5
    caption: str = "+"

    def build(self) -> list[str]:
        return [
            to_add(
                name=self.name,
                offset=self.offset,
                to=self.to,
                radius=self.radius,
                caption=self.caption,
            )
        ]


@dataclass
class OutputProjection(Element):
    name: str
    vocab_size: int = 30522
    model_dim: int = 768
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 28
    depth: int = 28
    caption: str = "Proj"

    def build(self) -> list[str]:
        return [
            to_output_projection(
                name=self.name,
                vocab_size=self.vocab_size,
                model_dim=self.model_dim,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class Dropout(Element):
    """Dropout layer."""

    name: str
    rate: float = 0.5
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 32
    depth: int = 32
    opacity: float = 0.3
    caption: str = f"Dropout {rate}"

    def build(self) -> list[str]:
        return [
            to_pool(
                name=self.name,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


# ---------------- Extended generic elements -----------------


@dataclass
class Activation(Element):
    name: str
    act: str = "ReLU"
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 18
    depth: int = 18
    caption: str | None = None

    def build(self) -> list[str]:
        return [
            to_activation(
                name=self.name,
                act=self.act,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class Normalization(Element):
    name: str
    kind: str = "BN"
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 18
    depth: int = 18
    caption: str | None = None

    def build(self) -> list[str]:
        return [
            to_normalization(
                name=self.name,
                kind=self.kind,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class RNNCell(Element):
    name: str
    cell: str = "LSTM"
    hidden_size: int = 512
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 2
    height: int = 26
    depth: int = 26
    caption: str | None = None

    def build(self) -> list[str]:
        return [
            to_rnn_cell(
                name=self.name,
                cell=self.cell,
                hidden_size=self.hidden_size,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class GenericBox(Element):
    name: str
    label_left: str = " "
    label_right: str = " "
    zlabel: str | int = " "
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int | tuple[int, int] = 1
    height: int = 20
    depth: int = 20
    caption: str = " "
    fill: str = "\\GenericColor"
    opacity: float = 0.35

    def build(self) -> list[str]:
        return [
            to_generic_box(
                name=self.name,
                label_left=self.label_left,
                label_right=self.label_right,
                zlabel=self.zlabel,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
                fill=self.fill,
                opacity=self.opacity,
            )
        ]


# --- New extended layer dataclasses ---
@dataclass
class DepthwiseConv(Element):
    name: str
    channels: int
    kernel: str = "3x3"
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 30
    depth: int = 30
    caption: str = "DW"

    def build(self) -> list[str]:  # noqa: D401
        return [
            to_depthwise_conv(
                name=self.name,
                channels=self.channels,
                kernel=self.kernel,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class SeparableConv(Element):
    name: str
    in_channels: int
    out_channels: int
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: tuple[int, int] = (1, 1)
    height: int = 30
    depth: int = 30
    caption: str = "SepConv"

    def build(self) -> list[str]:
        return [
            to_separable_conv(
                name=self.name,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class TransposeConv(Element):
    name: str
    s_filer: int = 256
    n_filer: int = 64
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 2
    height: int = 30
    depth: int = 30
    caption: str = "DeConv"

    def build(self) -> list[str]:
        return [
            to_transpose_conv(
                name=self.name,
                s_filer=self.s_filer,
                n_filer=self.n_filer,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class Flatten(Element):
    name: str
    features: int
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 1
    height: int = 12
    depth: int = 12
    caption: str = "Flatten"

    def build(self) -> list[str]:
        return [
            to_flatten(
                name=self.name,
                features=self.features,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class SqueezeExcitation(Element):
    name: str
    channels: int
    se_ratio: float = 0.25
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: int = 2
    height: int = 18
    depth: int = 18
    caption: str = "SE"

    def build(self) -> list[str]:
        return [
            to_squeeze_excitation(
                name=self.name,
                channels=self.channels,
                se_ratio=self.se_ratio,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class TransformerBlock(Element):
    name: str
    model_dim: int = 768
    heads: int = 8
    mlp_dim: int = 3072
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: tuple[int, int] = (2, 2)
    height: int = 34
    depth: int = 34
    caption: str = "Block"

    def build(self) -> list[str]:
        return [
            to_transformer_block(
                name=self.name,
                model_dim=self.model_dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=self.height,
                depth=self.depth,
                caption=self.caption,
            )
        ]


@dataclass
class Concat(Element):
    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    radius: float = 2.2

    def build(self) -> list[str]:
        return [to_concat(name=self.name, offset=self.offset, to=self.to, radius=self.radius)]


@dataclass
class Split(Element):
    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    radius: float = 2.2

    def build(self) -> list[str]:
        return [to_split(name=self.name, offset=self.offset, to=self.to, radius=self.radius)]


class Block(Element):
    """Composable block of elements."""

    def __init__(self, children: Sequence[Element] | None = None):
        self.children: list[Element] = list(children or [])

    def add(self, *els: Element) -> Self:
        self.children.extend(els)
        return self

    def build(self) -> list[str]:
        out: list[str] = []
        for el in self.children:
            out.extend(el.build())
        return out


class TwoConvPoolBlock(Block):
    """Two Conv + Pool block."""

    def __init__(
        self,
        name: str,
        bottom: str,
        top: str,
        s_filer: int = 256,
        n_filer: int = 64,
        offset: str = "(1,0,0)",
        size: tuple[int, int, int] = (32, 32, 4),
        opacity: float = 0.5,
    ):
        ccr = ConvConvRelu(
            name=f"ccr_{name}",
            s_filer=s_filer,
            n_filer=(n_filer, n_filer),
            offset=offset,
            to=f"({bottom}-east)",
            width=(size[2] // 2, size[2] // 2),
            height=size[0],
            depth=size[1],
        )
        pool = Pool(
            name=top,
            offset="(0,0,0)",
            to=f"(ccr_{name}-east)",
            width=1,
            height=size[0] - (size[0] // 4),
            depth=size[1] - (size[0] // 4),
            opacity=opacity,
        )
        conn = Connection(of=bottom, to=f"ccr_{name}")
        super().__init__([ccr, pool, conn])


class UnconvBlock(Block):
    """Unconv block with residuals."""

    def __init__(
        self,
        name: str,
        bottom: str,
        top: str,
        s_filer: int = 256,
        n_filer: int = 64,
        offset: str = "(1,0,0)",
        size: tuple[int, int, int] = (32, 32, 4),
        opacity: float = 0.5,
    ):
        seq: list[Element] = [
            UnPool(
                name=f"unpool_{name}",
                offset=offset,
                to=f"({bottom}-east)",
                width=1,
                height=size[0],
                depth=size[1],
                opacity=opacity,
            ),
            ConvRes(
                name=f"ccr_res_{name}",
                offset="(0,0,0)",
                to=f"(unpool_{name}-east)",
                s_filer=s_filer,
                n_filer=n_filer,
                width=size[2],
                height=size[0],
                depth=size[1],
                opacity=opacity,
            ),
            Conv(
                name=f"ccr_{name}",
                offset="(0,0,0)",
                to=f"(ccr_res_{name}-east)",
                s_filer=s_filer,
                n_filer=n_filer,
                width=size[2],
                height=size[0],
                depth=size[1],
            ),
            ConvRes(
                name=f"ccr_res_c_{name}",
                offset="(0,0,0)",
                to=f"(ccr_{name}-east)",
                s_filer=s_filer,
                n_filer=n_filer,
                width=size[2],
                height=size[0],
                depth=size[1],
                opacity=opacity,
            ),
            Conv(
                name=top,
                offset="(0,0,0)",
                to=f"(ccr_res_c_{name}-east)",
                s_filer=s_filer,
                n_filer=n_filer,
                width=size[2],
                height=size[0],
                depth=size[1],
            ),
            Connection(of=bottom, to=f"unpool_{name}"),
        ]
        super().__init__(seq)


# Diagram Builder
class Diagram:
    """Main class for building and rendering diagrams."""

    def __init__(self) -> None:
        self.elements: list[Element] = []

    def add(self, *els: Element) -> Self:
        self.elements.extend(els)
        return self

    def extend(self, els: Iterable[Element]) -> Self:
        self.elements.extend(els)
        return self

    def build(self) -> list[str]:
        """Generate LaTeX snippets."""
        out: list[str] = []
        for el in self.elements:
            out.extend(el.build())
        return out

    def to_tex(self, inline_styles: bool = True, include_colors: bool = True) -> str:
        """Generate full LaTeX document."""
        from .templates import LaTeXTemplate

        latex_parts = []
        for element in self.elements:
            latex_parts.extend(element.build())

        return LaTeXTemplate.full_document(
            latex_parts, inline_styles=inline_styles, include_colors=include_colors
        )

    def save_tex(
        self, path: str | Path, inline_styles: bool = True, include_colors: bool = True
    ) -> Path:
        """Save LaTeX to file."""

        renderer = DiagramRenderer()
        return renderer.render_to_tex(
            self.elements, path, inline_styles=inline_styles, include_colors=include_colors
        )

    def render_pdf(
        self,
        out_pdf: str | Path,
        inline_styles: bool = True,
        include_colors: bool = True,
        keep_tex: bool | str | Path = True,
    ) -> Path:
        """Render to PDF."""

        renderer = DiagramRenderer()
        return renderer.render_to_pdf(
            self.elements,
            out_pdf,
            inline_styles=inline_styles,
            include_colors=include_colors,
            keep_tex=keep_tex,
        )
