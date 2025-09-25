"""Layer generation"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .compiler import FormatConverter, LaTeXCompiler
from .renderer import DiagramRenderer
from .templates import LaTeXTemplate, _layers_dir_path  # noqa: F401

logger = logging.getLogger(__name__)


def to_head_pkg() -> str:
    return LaTeXTemplate.document_header_external()


def to_head_inline() -> str:
    return LaTeXTemplate.document_header_inline()


def to_colors() -> str:
    return LaTeXTemplate.color_definitions()


def to_begin() -> str:
    return LaTeXTemplate.document_begin()


def to_end() -> str:
    return LaTeXTemplate.document_end()


def to_document(arch: list[str], inline_styles: bool = True, include_colors: bool = True) -> str:
    return LaTeXTemplate.full_document(
        arch, inline_styles=inline_styles, include_colors=include_colors
    )


def to_generate(
    arch: list[str],
    pathname: str = "file.tex",
    inline_styles: bool = True,
    include_colors: bool = True,
) -> None:
    DiagramRenderer().render_to_tex(
        arch, pathname, inline_styles=inline_styles, include_colors=include_colors
    )


def compile_tex_to_pdf(
    tex_content: str, out_pdf: str | Path, keep_tex: bool | str | Path = True
) -> Path:
    compiler = LaTeXCompiler()
    return compiler.compile_to_pdf(tex_content, out_pdf, keep_tex=keep_tex)


def pdf_to_format(
    pdf_path: Path, out_path: Path, format: str, dpi: int = 300, page: int = 1
) -> Path:
    converter = FormatConverter()
    return converter.pdf_to_format(pdf_path, out_path, format, dpi=dpi, page=page)


def to_input(
    pathfile: str,
    to: str = "(-3,0,0)",
    width: int = 8,
    height: int = 8,
    name: str = "temp",
) -> str:
    half_w = width / 2
    half_h = height / 2
    return (
        f"\\node[canvas is zy plane at x=0] ({name}) at {to} "
        f"{{\\includegraphics[width={width}cm,height={height}cm]{{{pathfile}}}}};"
        f"\\coordinate ({name}-east) at ($({name}.center)+({half_w}cm,0,0)$);"
        f"\\coordinate ({name}-west) at ($({name}.center)-({half_w}cm,0,0)$);"
        f"\\coordinate ({name}-north) at ($({name}.center)+(0,{half_h}cm,0)$);"
        f"\\coordinate ({name}-south) at ($({name}.center)-(0,{half_h}cm,0)$);"
    )


def to_connection(of: str, to: str) -> str:
    return f"\\draw [connection]  ({of}-east)    -- node {{\\midarrow}} ({to}-west);"


def to_conv(
    name: str,
    s_filer: int = 256,
    n_filer: int = 64,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 40,
    depth: int = 40,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{ {n_filer}, }}}},
        zlabel={s_filer},
        fill=\\ConvColor,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_conv_conv_relu(
    name: str,
    s_filer: int = 256,
    n_filer: tuple[int, int] = (64, 64),
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: tuple[int, int] = (2, 2),
    height: int = 40,
    depth: int = 40,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{ {offset} }}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{ {{ {n_filer[0]} }}, {{ {n_filer[1]} }} }},
        zlabel={s_filer},
        fill=\\ConvColor,
        bandfill=\\ConvReluColor,
        height={height},
        width={{ {width[0]} , {width[1]} }},
        depth={depth}
        }}
    }};"""


def to_pool(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 32,
    depth: int = 32,
    opacity: float = 0.5,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{ {offset} }}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        fill=\\PoolColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_unpool(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 32,
    depth: int = 32,
    opacity: float = 0.5,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{ {offset} }}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        fill=\\UnpoolColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_conv_res(
    name: str,
    s_filer: int = 256,
    n_filer: int = 64,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 6,
    height: int = 40,
    depth: int = 40,
    opacity: float = 0.2,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{ {offset} }}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{ {{ {n_filer} }}, }},
        zlabel={s_filer},
        fill={{rgb:white,1;black,3}},
        bandfill={{rgb:white,1;black,2}},
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_conv_softmax(
    name: str,
    s_filer: int = 40,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 40,
    depth: int = 40,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        zlabel={s_filer},
        fill=\\SoftmaxColor,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_softmax(
    name: str,
    s_filer: int = 10,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 2,
    height: int = 3,
    depth: int = 25,
    opacity: float = 0.8,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{ " " ,"dummy" }},
        zlabel={s_filer},
        fill=\\SoftmaxColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_sum(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    radius: float = 2.5,
    opacity: float = 0.6,
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Ball={{
        name={name},
        fill=\\SumColor,
        opacity={opacity},
        radius={radius},
        logo=$+$
        }}
    }};"""


def to_skip(of: str, to: str, pos: float = 1.25) -> str:
    return f"""\\path ({of}-southeast) -- ({of}-northeast) coordinate[pos={pos}] ({of}-top) ;
\\path ({to}-south)  -- ({to}-north)  coordinate[pos={pos}] ({to}-top) ;
\\draw [copyconnection]  ({of}-northeast)
-- node {{\\copymidarrow}}({of}-top)
-- node {{\\copymidarrow}}({to}-top)
-- node {{\\copymidarrow}} ({to}-north);"""


def generate_pdf(
    arch: list[str],
    out_pdf: str | Path,
    inline_styles: bool = True,
    include_colors: bool = True,
    keep_tex: bool | str | Path = True,
) -> Path:
    doc = to_document(arch, inline_styles=inline_styles, include_colors=include_colors)
    return compile_tex_to_pdf(doc, out_pdf, keep_tex=keep_tex)


def generate_png(
    arch: list[str],
    out_png: str | Path,
    dpi: int = 300,
    inline_styles: bool = True,
    include_colors: bool = True,
    keep_tex: bool | str | Path = True,
) -> Path:
    out_png_path = Path(out_png).resolve()
    if keep_tex is False:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_tmp = Path(tmpdir) / "temp.pdf"
            generate_pdf(
                arch,
                pdf_tmp,
                inline_styles=inline_styles,
                include_colors=include_colors,
                keep_tex=False,
            )
            return pdf_to_format(pdf_tmp, out_png_path, "png", dpi=dpi)
    pdf_path = out_png_path.with_suffix(".pdf")
    generate_pdf(
        arch,
        pdf_path,
        inline_styles=inline_styles,
        include_colors=include_colors,
        keep_tex=keep_tex,
    )
    return pdf_to_format(pdf_path, out_png_path, "png", dpi=dpi)


def generate_svg(
    arch: list[str],
    out_svg: str | Path,
    inline_styles: bool = True,
    include_colors: bool = True,
    keep_tex: bool | str | Path = True,
) -> Path:
    out_svg_path = Path(out_svg).resolve()
    if keep_tex is False:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_tmp = Path(tmpdir) / "temp.pdf"
            generate_pdf(
                arch,
                pdf_tmp,
                inline_styles=inline_styles,
                include_colors=include_colors,
                keep_tex=False,
            )
            return pdf_to_format(pdf_tmp, out_svg_path, "svg")
    pdf_path = out_svg_path.with_suffix(".pdf")
    generate_pdf(
        arch,
        pdf_path,
        inline_styles=inline_styles,
        include_colors=include_colors,
        keep_tex=keep_tex,
    )
    return pdf_to_format(pdf_path, out_svg_path, "svg")
