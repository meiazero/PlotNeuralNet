from __future__ import annotations

# from typing import Iterable, Sequence
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import layers as L

__all__ = [
    "Element",
    "Leaf",
    "Block",
    "Input",
    "Conv",
    "ConvConvRelu",
    "Pool",
    "UnPool",
    "ConvRes",
    "ConvSoftMax",
    "SoftMax",
    "Sum",
    "Connection",
    "Skip",
    "TwoConvPoolBlock",
    "UnconvBlock",
    "ResBlock",
    "Diagram",
    # Legacy wrappers
    "block_2ConvPool",
    "block_Unconv",
    "block_Res",
]


class Element:
    """Peça básica que sabe se transformar em fragmentos LaTeX."""

    def build(self) -> list[str]:  # pragma: no cover - simples delegação
        raise NotImplementedError


@dataclass
class Leaf(Element):
    tex: str

    def build(self) -> list[str]:
        return [self.tex]


# Folhas utilitárias (camadas)
@dataclass
class Input(Element):
    pathfile: str
    to: str = "(-3,0,0)"
    width: float = 8
    height: float = 8
    name: str = "temp"

    def build(self) -> list[str]:
        return [L.to_input(self.pathfile, to=self.to, width=int(self.width), height=int(self.height), name=self.name)]


@dataclass
class Conv(Element):
    name: str
    s_filer: int = 256
    n_filer: int = 64
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: float = 1
    height: float = 40
    depth: float = 40
    caption: str = " "

    def build(self) -> list[str]:
        return [
            L.to_Conv(
                name=self.name,
                s_filer=self.s_filer,
                n_filer=self.n_filer,
                offset=self.offset,
                to=self.to,
                width=int(self.width),
                height=int(self.height),
                depth=int(self.depth),
                caption=self.caption,
            )
        ]


@dataclass
class ConvConvRelu(Element):
    name: str
    s_filer: int = 256
    n_filer: tuple[int, int] = (64, 64)
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: tuple[float, float] = (2, 2)
    height: float = 40
    depth: float = 40
    caption: str = " "

    def build(self) -> list[str]:
        return [
            L.to_ConvConvRelu(
                name=self.name,
                s_filer=self.s_filer,
                n_filer=self.n_filer,
                offset=self.offset,
                to=self.to,
                width=(int(self.width[0]), int(self.width[1])),
                height=int(self.height),
                depth=int(self.depth),
                caption=self.caption,
            )
        ]


@dataclass
class Pool(Element):
    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: float = 1
    height: float = 32
    depth: float = 32
    opacity: float = 0.5
    caption: str = " "

    def build(self) -> list[str]:
        return [
            L.to_Pool(
                name=self.name,
                offset=self.offset,
                to=self.to,
                width=int(self.width),
                height=int(self.height),
                depth=int(self.depth),
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


@dataclass
class UnPool(Element):
    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: float = 1
    height: float = 32
    depth: float = 32
    opacity: float = 0.5
    caption: str = " "

    def build(self) -> list[str]:
        return [
            L.to_UnPool(
                name=self.name,
                offset=self.offset,
                to=self.to,
                width=int(self.width),
                height=int(self.height),
                depth=int(self.depth),
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


@dataclass
class ConvRes(Element):
    name: str
    s_filer: int = 256
    n_filer: int = 64
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: float = 6
    height: float = 40
    depth: float = 40
    opacity: float = 0.2
    caption: str = " "

    def build(self) -> list[str]:
        return [
            L.to_ConvRes(
                name=self.name,
                s_filer=self.s_filer,
                n_filer=self.n_filer,
                offset=self.offset,
                to=self.to,
                width=int(self.width),
                height=int(self.height),
                depth=int(self.depth),
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


@dataclass
class ConvSoftMax(Element):
    name: str
    s_filer: int = 40
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: float = 1
    height: float = 40
    depth: float = 40
    caption: str = " "

    def build(self) -> list[str]:
        return [
            L.to_ConvSoftMax(
                name=self.name,
                s_filer=self.s_filer,
                offset=self.offset,
                to=self.to,
                width=int(self.width),
                height=int(self.height),
                depth=int(self.depth),
                caption=self.caption,
            )
        ]


@dataclass
class SoftMax(Element):
    name: str
    s_filer: int = 10
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    width: float = 1.5
    height: float = 3
    depth: float = 25
    opacity: float = 0.8
    caption: str = " "

    def build(self) -> list[str]:
        return [
            L.to_SoftMax(
                name=self.name,
                s_filer=self.s_filer,
                offset=self.offset,
                to=self.to,
                width=self.width,
                height=int(self.height),
                depth=int(self.depth),
                opacity=self.opacity,
                caption=self.caption,
            )
        ]


@dataclass
class Sum(Element):
    name: str
    offset: str = "(0,0,0)"
    to: str = "(0,0,0)"
    radius: float = 2.5
    opacity: float = 0.6

    def build(self) -> list[str]:
        return [L.to_Sum(self.name, offset=self.offset, to=self.to, radius=self.radius, opacity=self.opacity)]


@dataclass
class Connection(Element):
    of: str
    to: str

    def build(self) -> list[str]:
        return [L.to_connection(self.of, self.to)]


@dataclass
class Skip(Element):
    of: str
    to: str
    pos: float = 1.25

    def build(self) -> list[str]:
        return [L.to_skip(self.of, self.to, pos=self.pos)]


class Block(Element):
    def __init__(self, children: Sequence[Element] | None = None):
        self.children: list[Element] = list(children or [])

    def add(self, *els: Element) -> Block:
        self.children.extend(els)
        return self

    def build(self) -> list[str]:
        out: list[str] = []
        for el in self.children:
            out.extend(el.build())
        return out


class TwoConvPoolBlock(Block):
    def __init__(
        self,
        name: str,
        botton: str,
        top: str,
        s_filer: int = 256,
        n_filer: int = 64,
        offset: str = "(1,0,0)",
        size: tuple[int, int, float] = (32, 32, 3.5),
        opacity: float = 0.5,
    ):
        ccr = ConvConvRelu(
            name=f"ccr_{name}",
            s_filer=s_filer,
            n_filer=(n_filer, n_filer),
            offset=offset,
            to=f"({botton}-east)",
            width=(int(size[2]), int(size[2])),
            height=size[0],
            depth=size[1],
        )
        pool = Pool(
            name=f"{top}",
            offset="(0,0,0)",
            to=f"(ccr_{name}-east)",
            width=1,
            height=size[0] - int(size[0] / 4),
            depth=size[1] - int(size[0] / 4),
            opacity=opacity,
        )
        conn = Connection(of=f"{botton}", to=f"ccr_{name}")
        super().__init__([ccr, pool, conn])


class UnconvBlock(Block):
    def __init__(
        self,
        name: str,
        botton: str,
        top: str,
        s_filer: int = 256,
        n_filer: int = 64,
        offset: str = "(1,0,0)",
        size: tuple[int, int, float] = (32, 32, 3.5),
        opacity: float = 0.5,
    ):
        seq: list[Element] = [
            UnPool(name=f"unpool_{name}", offset=offset, to=f"({botton}-east)", width=1, height=size[0], depth=size[1], opacity=opacity),
            ConvRes(name=f"ccr_res_{name}", offset="(0,0,0)", to=f"(unpool_{name}-east)", s_filer=s_filer, n_filer=n_filer, width=int(size[2]), height=size[0], depth=size[1], opacity=opacity),
            Conv(name=f"ccr_{name}", offset="(0,0,0)", to=f"(ccr_res_{name}-east)", s_filer=s_filer, n_filer=n_filer, width=int(size[2]), height=size[0], depth=size[1]),
            ConvRes(name=f"ccr_res_c_{name}", offset="(0,0,0)", to=f"(ccr_{name}-east)", s_filer=s_filer, n_filer=n_filer, width=int(size[2]), height=size[0], depth=size[1], opacity=opacity),
            Conv(name=f"{top}", offset="(0,0,0)", to=f"(ccr_res_c_{name}-east)", s_filer=s_filer, n_filer=n_filer, width=int(size[2]), height=size[0], depth=size[1]),
            Connection(of=f"{botton}", to=f"unpool_{name}"),
        ]
        super().__init__(seq)


class ResBlock(Block):
    def __init__(
        self,
        num: int,
        name: str,
        botton: str,
        top: str,
        s_filer: int = 256,
        n_filer: int = 64,
        offset: str = "(0,0,0)",
        size: tuple[int, int, float] = (32, 32, 3.5),
        opacity: float = 0.5,
    ):
        layers_names = [*(f"{name}_{i}" for i in range(num - 1)), top]
        seq: list[Element] = []
        current = botton
        for nm in layers_names:
            seq.append(
                Conv(
                    name=f"{nm}",
                    offset=offset,
                    to=f"({current}-east)",
                    s_filer=s_filer,
                    n_filer=n_filer,
                    width=int(size[2]),
                    height=size[0],
                    depth=size[1],
                )
            )
            seq.append(Connection(of=f"{current}", to=f"{nm}"))
            current = nm
        # skip connection entre 2º e penúltimo
        if len(layers_names) >= 3:
            seq.append(Skip(of=layers_names[1], to=layers_names[-2], pos=1.25))
        super().__init__(seq)


class Diagram:
    """Construtor de diagramas de maneira tipada e extensível."""

    def __init__(self) -> None:
        self.elements: list[Element] = []

    def add(self, *els: Element) -> Diagram:
        self.elements.extend(els)
        return self

    def extend(self, els: Iterable[Element]) -> Diagram:
        self.elements.extend(list(els))
        return self

    def build(self) -> list[str]:
        out: list[str] = []
        for el in self.elements:
            out.extend(el.build())
        return out

    # Atalhos para renderização
    def to_tex(self, inline_styles: bool = True, include_colors: bool = True) -> str:
        return L.to_document(self.build(), inline_styles=inline_styles, include_colors=include_colors)

    def save_tex(self, path: str, inline_styles: bool = True, include_colors: bool = True) -> str:
        L.to_generate(self.build(), pathname=path, inline_styles=inline_styles, include_colors=include_colors)
        return path

    def render_pdf(self, out_pdf: str, inline_styles: bool = True, include_colors: bool = True) -> Path:
        return L.generate_pdf(self.build(), out_pdf, inline_styles=inline_styles, include_colors=include_colors)

    def render_png(self, out_png: str, dpi: int = 300, inline_styles: bool = True, include_colors: bool = True) -> Path:
        return L.generate_png(self.build(), out_png, dpi=dpi, inline_styles=inline_styles, include_colors=include_colors)


# ======================
# Wrappers legados
# ======================
def block_2ConvPool(name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return TwoConvPoolBlock(name=name, botton=botton, top=top, s_filer=s_filer, n_filer=n_filer, offset=offset, size=size, opacity=opacity).build()


def block_Unconv(name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return UnconvBlock(name=name, botton=botton, top=top, s_filer=s_filer, n_filer=n_filer, offset=offset, size=size, opacity=opacity).build()


def block_Res(num, name, botton, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32, 32, 3.5), opacity=0.5):
    return ResBlock(num=num, name=name, botton=botton, top=top, s_filer=s_filer, n_filer=n_filer, offset=offset, size=size, opacity=opacity).build()
