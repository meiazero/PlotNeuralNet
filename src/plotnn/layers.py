"""Layer generation functions and legacy compatibility interface."""

from __future__ import annotations

import logging
from pathlib import Path

# Import the new modular components
from .compiler import LaTeXCompiler, FormatConverter
from .renderer import DiagramRenderer
from .templates import LaTeXTemplate, _layers_dir_path

logger = logging.getLogger(__name__)


# Legacy compatibility functions - now use the new modular approach
def _layers_dir_path() -> str:
    """Get path to layers directory."""
    from .templates import _layers_dir_path
    return _layers_dir_path()


def to_head_pkg() -> str:
    """Generate LaTeX header with external packages."""
    return LaTeXTemplate.document_header_external()


def to_head_inline() -> str:
    """Generate LaTeX header with inline styles."""
    return LaTeXTemplate.document_header_inline()


def to_colors():
    """Generate color definitions."""
    return LaTeXTemplate.color_definitions()


def to_begin():
    """Generate document begin."""
    return LaTeXTemplate.document_begin()


def to_end() -> str:
    """Generate document end."""
    return LaTeXTemplate.document_end()


def to_document(arch: list[str], inline_styles: bool = True, include_colors: bool = True) -> str:
    """Generate complete LaTeX document."""
    return LaTeXTemplate.full_document(arch, inline_styles=inline_styles, include_colors=include_colors)


def to_generate(
    arch: list[str],
    pathname: str = "file.tex",
    inline_styles: bool = True,
    include_colors: bool = True,
) -> None:
    """Generate LaTeX file (legacy compatibility)."""
    renderer = DiagramRenderer()
    renderer.render_to_tex(arch, pathname, inline_styles=inline_styles, include_colors=include_colors)


def compile_tex_to_pdf(tex_content: str, out_pdf: str | Path) -> Path:
    """Compile LaTeX to PDF (legacy compatibility)."""
    compiler = LaTeXCompiler()
    return compiler.compile_to_pdf(tex_content, out_pdf)


def pdf_to_format(
    pdf_path: Path, out_path: Path, format: str, dpi: int = 300, page: int = 1
) -> Path:
    """Convert PDF to other format (legacy compatibility)."""
    converter = FormatConverter()
    return converter.pdf_to_format(pdf_path, out_path, format, dpi=dpi, page=page)


# Layer generation functions
def to_input(
    pathfile: str, to: str = "(-3,0,0)", width: int = 8, height: int = 8, name: str = "temp"
) -> str:
    """Generate input layer LaTeX."""
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
    """Generate connection LaTeX."""
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


def to_end() -> str:
    return """
\\end{tikzpicture}
\\end{document}
"""


def to_document(arch: list[str], inline_styles: bool = True, include_colors: bool = True) -> str:
    head = to_head_inline() if inline_styles else to_head_pkg()
    parts = [head]
    if include_colors:
        parts.append(to_colors())
    parts.append(to_begin())
    parts.extend(arch)
    parts.append(to_end())
    return "".join(parts)


def to_generate(
    arch: list[str],
    pathname: str = "file.tex",
    inline_styles: bool = True,
    include_colors: bool = True,
) -> None:
    doc = to_document(arch, inline_styles=inline_styles, include_colors=include_colors)
    with open(pathname, "w", encoding="utf-8") as f:
        f.write(doc)


def compile_tex_to_pdf(tex_content: str, out_pdf: str | Path) -> Path:
    out_pdf_path = Path(out_pdf).resolve()
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tex_file = tmp / "diagram.tex"
        tex_file.write_text(tex_content, encoding="utf-8")

        if shutil.which("latexmk"):
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", tex_file.name]
        else:
            cmd = ["pdflatex", "-interaction=nonstopmode", tex_file.name]
            subprocess.run(cmd, cwd=tmp, check=False)  # Second pass
            subprocess.run(cmd, cwd=tmp, check=True)

        produced = tmp / "diagram.pdf"
        if not produced.exists():
            raise RuntimeError("LaTeX compilation failed to produce PDF. Check logs.")
        shutil.copyfile(produced, out_pdf_path)
    logger.info(f"PDF generated at {out_pdf_path}")
    return out_pdf_path


def pdf_to_format(
    pdf_path: Path, out_path: Path, format: str, dpi: int = 300, page: int = 1
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if format not in ("png", "svg"):
        raise ValueError("Format must be 'png' or 'svg'")

    tool = shutil.which("pdftocairo")
    if tool:
        args = ["-r", str(dpi), f"-f{page}", f"-l{page}", str(pdf_path), "-singlefile"]
        if format == "png":
            cmd = [tool, "-png"] + args + [str(out_path.with_suffix("").as_posix())]
        else:
            cmd = [tool, "-svg"] + args + [str(out_path)]
        subprocess.run(cmd, check=True)
        return out_path

    if format == "svg":
        raise RuntimeError("SVG requires pdftocairo.")

    im = shutil.which("magick") or shutil.which("convert")
    if im:
        cmd = [im, "-density", str(dpi), f"{pdf_path}[{page-1}]", "-quality", "100", str(out_path)]
        subprocess.run(cmd, check=True)
        return out_path

    gs = shutil.which("gs")
    if gs:
        cmd = [
            gs,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pngalpha",
            f"-r{dpi}",
            f"-dFirstPage={page}",
            f"-dLastPage={page}",
            f"-sOutputFile={out_path}",
            str(pdf_path),
        ]
        subprocess.run(cmd, check=True)
        return out_path

    raise RuntimeError(f"No tool found for {format} conversion (pdftocairo/ImageMagick/gs).")


def generate_pdf(
    arch: list[str], out_pdf: str | Path, inline_styles: bool = True, include_colors: bool = True
) -> Path:
    doc = to_document(arch, inline_styles=inline_styles, include_colors=include_colors)
    return compile_tex_to_pdf(doc, out_pdf)


def generate_png(
    arch: list[str],
    out_png: str | Path,
    dpi: int = 300,
    inline_styles: bool = True,
    include_colors: bool = True,
) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "temp.pdf"
        generate_pdf(arch, pdf_path, inline_styles=inline_styles, include_colors=include_colors)
        return pdf_to_format(pdf_path, Path(out_png), "png", dpi=dpi)


def generate_svg(
    arch: list[str], out_svg: str | Path, inline_styles: bool = True, include_colors: bool = True
) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "temp.pdf"
        generate_pdf(arch, pdf_path, inline_styles=inline_styles, include_colors=include_colors)
        return pdf_to_format(pdf_path, Path(out_svg), "svg")
