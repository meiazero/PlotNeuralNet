from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _layers_dir_path() -> str:
    path = Path(__file__).parent / "layers"
    return str(path.resolve()).replace("\\", "/")


def to_head_pkg() -> str:
    pathlayers = _layers_dir_path()
    if not pathlayers.endswith("/"):
        pathlayers += "/"
    return f"""
\\documentclass[border=8pt, multi, tikz]{{standalone}}
\\usepackage{{import}}
\\subimport{{{pathlayers}}}{{init}}
\\usetikzlibrary{{positioning}}
\\usetikzlibrary{{3d}}
"""


def _read_pkg_text(*rel_parts: str) -> str:
    p = Path(__file__).parent.joinpath(*rel_parts)
    return p.read_text(encoding="utf-8")


def _inline_layers_tex() -> str:
    parts: list[str] = []
    init = _read_pkg_text("layers", "init.tex")
    parts.append(init)
    for sty in ("Ball.sty", "Box.sty", "RightBandedBox.sty"):
        txt = _read_pkg_text("layers", sty)
        lines = [ln for ln in txt.splitlines() if not ln.lstrip().startswith("\\ProvidesPackage")]
        parts.append("\n".join(lines))
    return "\n".join(parts) + "\n"


def to_head_inline() -> str:
    return (
        """
\\documentclass[border=8pt, multi, tikz]{standalone}
\\usetikzlibrary{positioning}
\\usetikzlibrary{3d}
"""
        + _inline_layers_tex()
    )


def to_colors():
    return """
\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}
\\def\\ConvReluColor{rgb:yellow,5;red,5;white,5}
\\def\\PoolColor{rgb:red,1;black,0.3}
\\def\\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\\def\\FcColor{rgb:blue,5;red,2.5;white,5}
\\def\\FcReluColor{rgb:blue,5;red,5;white,4}
\\def\\SoftmaxColor{rgb:magenta,5;black,7}
\\def\\SumColor{rgb:blue,5;green,15}
"""


def to_begin():
    return """
\\newcommand{\\copymidarrow}{\\tikz \\draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\\begin{document}
\\begin{tikzpicture}
\\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\\edgecolor,opacity=0.7]
\\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""


def to_input(
    pathfile: str, to: str = "(-3,0,0)", width: int = 8, height: int = 8, name: str = "temp"
) -> str:
    return f"""\\node[canvas is zy plane at x=0] ({name}) at {to} {{\\includegraphics[width={width}cm,height={height}cm]{{{pathfile}}}}};"""


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
