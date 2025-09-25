from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from .layers import (
    to_connection,
    to_conv,
    to_conv_conv_relu,
    to_conv_res,
    to_conv_softmax,
    to_input,
    to_pool,
    to_skip,
    to_softmax,
    to_sum,
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
