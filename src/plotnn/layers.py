from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _layers_dir_path() -> str:
    """Resolve o caminho local (no filesystem) para os recursos LaTeX do pacote.

    Retorna um caminho POSIX (slashes) que aponta para o diretório 'latex/layers' do pacote.
    """
    path = Path(__file__).parent / "latex" / "layers"
    return str(path.resolve()).replace("\\", "/")


def to_head(projectpath: str) -> str:
    """Mantém compatibilidade: Recebe um diretório que contém 'layers/'."""
    pathlayers = os.path.join(projectpath, "layers/").replace("\\", "/")
    return r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{"""+ pathlayers + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image
"""


def to_head_pkg() -> str:
    """Cabeçalho usando os recursos LaTeX embarcados no pacote.

    Não requer parâmetro; resolve o diretório correto automaticamente.
    """
    pathlayers = _layers_dir_path()
    # Garante barra final
    if not pathlayers.endswith("/"):
        pathlayers += "/"
    return r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{"""+ pathlayers + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image
"""


def _read_pkg_text(*rel_parts: str) -> str:
    """Lê texto de um recurso dentro de plotnn/latex/layers (via filesystem)."""
    p = Path(__file__).parent.joinpath(*rel_parts)
    return p.read_text(encoding="utf-8")


def _inline_layers_tex() -> str:
    r"""Concatena conteúdo de init.tex e .sty necessários diretamente no preâmbulo.

    Remove linhas \ProvidesPackage para evitar ruído.
    """
    parts: list[str] = []
    try:
        init = _read_pkg_text("latex", "layers", "init.tex")
        parts.append(init)
    except Exception:
        pass
    for sty in ("Ball.sty", "Box.sty", "RightBandedBox.sty"):
        try:
            txt = _read_pkg_text("latex", "layers", sty)
            lines = [ln for ln in txt.splitlines() if not ln.lstrip().startswith("\\ProvidesPackage")]
            parts.append("\n".join(lines))
        except Exception:
            continue
    return "\n".join(parts) + "\n"


def to_head_inline() -> str:
    """Cabeçalho completo com definições necessárias sem dependência de .tex externos."""
    return (
        r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usetikzlibrary{positioning}
\usetikzlibrary{3d}
"""
        + _inline_layers_tex()
    )


def to_cor():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}
\def\SumColor{rgb:blue,5;green,15}
"""


def to_begin():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""


def to_input(pathfile, to='(-3,0,0)', width=8, height=8, name="temp"):
    parts = [
        "\\node[canvas is zy plane at x=0] (",
        name,
        ") at ",
        to,
        " {\\includegraphics[width=",
        str(width),
        "cm,height=",
        str(height),
        "cm]{",
        pathfile,
        "}};\n",
    ]
    return "".join(parts)


def to_Conv(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    parts = [
        "\\pic[shift={",
        offset,
        "}] at ",
        to,
        "\n    {Box={\n        name=",
        name,
        ",\n        caption=",
        caption,
        ",\n        xlabel={{",
        str(n_filer),
        ", }},\n        zlabel=",
        str(s_filer),
        ",\n        fill=\\ConvColor,\n        height=",
        str(height),
        "\n        width=",
        str(width),
        "\n        depth=",
        str(depth),
        "\n        }\n    };\n",
    ]
    return "".join(parts)


def to_ConvConvRelu(name, s_filer=256, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption=" "):
    parts = [
        "\\pic[shift={ ", offset, " }] at ", to, "\n    {RightBandedBox={\n        name=", name, ",\n        caption=", caption, "\n        xlabel={{ ", str(n_filer[0]), ", ", str(n_filer[1]), " }},\n        zlabel=", str(s_filer), "\n        fill=\\ConvColor,\n        bandfill=\\ConvReluColor,\n        height=", str(height), "\n        width={ ", str(width[0]), " , ", str(width[1]), " },\n        depth=", str(depth), "\n        }\n    };\n",
    ]
    return "".join(parts)


def to_Pool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    parts = [
        "\\pic[shift={ ", offset, " }] at ", to, "\n    {Box={\n        name=",
        name,
        ",\n        caption=",
        caption,
        "\n        fill=\\PoolColor,\n        opacity=",
        str(opacity),
        "\n        height=",
        str(height),
        "\n        width=",
        str(width),
        "\n        depth=",
        str(depth),
        "\n        }\n    };\n",
    ]
    return "".join(parts)


def to_UnPool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    parts = [
        "\\pic[shift={ ", offset, " }] at ", to, "\n    {Box={\n        name=",
        name,
        "\n        caption=",
        caption,
        "\n        fill=\\UnpoolColor,\n        opacity=",
        str(opacity),
        "\n        height=",
        str(height),
        "\n        width=",
        str(width),
        "\n        depth=",
        str(depth),
        "\n        }\n    };\n",
    ]
    return "".join(parts)


def to_ConvRes(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=6, height=40, depth=40, opacity=0.2, caption=" "):
    parts = [
        "\\pic[shift={ ", offset, " }] at ", to, "\n    {RightBandedBox={\n        name=",
        name,
        "\n        caption=",
        caption,
        "\n        xlabel={{ ",
        str(n_filer),
        ", }},\n        zlabel=",
        str(s_filer),
        "\n        fill={rgb:white,1;black,3},\n        bandfill={rgb:white,1;black,2},\n        opacity=",
        str(opacity),
        "\n        height=",
        str(height),
        "\n        width=",
        str(width),
        "\n        depth=",
        str(depth),
        "\n        }\n    };\n",
    ]
    return "".join(parts)


def to_ConvSoftMax(name, s_filer=40, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    parts = [
        "\\pic[shift={",
        offset,
        "}] at ",
        to,
        "\n    {Box={\n        name=",
        name,
        ",\n        caption=",
        caption,
        "\n        zlabel=",
        str(s_filer),
        "\n        fill=\\SoftmaxColor,\n        height=",
        str(height),
        "\n        width=",
        str(width),
        "\n        depth=",
        str(depth),
        "\n        }\n    };\n",
    ]
    return "".join(parts)


def to_SoftMax(name, s_filer=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" "):
    parts = [
        "\\pic[shift={",
        offset,
        "}] at ",
        to,
        "\n    {Box={\n        name=",
        name,
        ",\n        caption=",
        caption,
        "\n        xlabel={{ \" \" ,\"dummy\" }},\n        zlabel=",
        str(s_filer),
        "\n        fill=\\SoftmaxColor,\n        opacity=",
        str(opacity),
        "\n        height=",
        str(height),
        "\n        width=",
        str(width),
        "\n        depth=",
        str(depth),
        "\n        }\n    };\n",
    ]
    return "".join(parts)


def to_Sum(name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6):
    parts = [
        "\\pic[shift={",
        offset,
        "}] at ",
        to,
        "\n    {Ball={\n        name=",
        name,
        "\n        fill=\\SumColor,\n        opacity=",
        str(opacity),
        "\n        radius=",
        str(radius),
        "\n        logo=$+$\n        }\n    };\n",
    ]
    return "".join(parts)


def to_connection(of, to):
    parts = [
        "\\draw [connection]  (",
        of,
        "-east)    -- node {\\midarrow} (",
        to,
        "-west);\n",
    ]
    return "".join(parts)


def to_skip(of, to, pos=1.25):
    parts = [
        "\\path (",
        of,
        "-southeast) -- (",
        of,
        "-northeast) coordinate[pos=",
        str(pos),
        "] (",
        of,
        "-top) ;\n",
        "\\path (",
        to,
        "-south)  -- (",
        to,
        "-north)  coordinate[pos=",
        str(pos),
        "] (",
        to,
        "-top) ;\n",
        "\\draw [copyconnection]  (",
        of,
        "-northeast)\n-- node {\\copymidarrow}(",
        of,
        "-top)\n-- node {\\copymidarrow}(",
        to,
        "-top)\n-- node {\\copymidarrow} (",
        to,
        "-north);\n",
    ]
    return "".join(parts)


def to_end():
    return r"""
\end{tikzpicture}
\end{document}
"""


def to_document(arch, inline_styles: bool = True, include_colors: bool = True) -> str:
    """Monta um documento LaTeX completo a partir dos fragmentos de arquitetura.

    - inline_styles=True insere todo o preâmbulo inline, sem depender de arquivos externos.
    - include_colors=True inclui as definições de cores padrão.
    """
    head = to_head_inline() if inline_styles else to_head_pkg()
    parts = [head]
    if include_colors:
        parts.append(to_cor())
    parts.append(to_begin())
    parts.extend(arch)
    parts.append(to_end())
    return "".join(parts)


def to_generate(arch, pathname: str = "file.tex", inline_styles: bool = True, include_colors: bool = True) -> None:
    """Gera um arquivo .tex (por padrão, com preâmbulo inline para não depender de .sty externos)."""
    doc = to_document(arch, inline_styles=inline_styles, include_colors=include_colors)
    with open(pathname, "w", encoding="utf-8") as f:
        f.write(doc)


# ===== Renderização: PDF/PNG direto do Python =====
def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def compile_tex_to_pdf(tex_content: str, out_pdf: os.PathLike[str] | str) -> Path:
    """Compila um documento LaTeX (string) para PDF.

    Usa latexmk se disponível; caso contrário, tenta pdflatex (duas passagens).
    Retorna o caminho do PDF gerado (igual a out_pdf).
    """
    out_pdf_path = Path(out_pdf).resolve()
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tex_file = tmp / "diagram.tex"
        tex_file.write_text(tex_content, encoding="utf-8")

        if _which("latexmk"):
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", tex_file.name]
            subprocess.run(cmd, cwd=tmp, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            if not _which("pdflatex"):
                raise RuntimeError("Nenhum motor LaTeX encontrado (latexmk/pdflatex). Instale um TeX Live.")
            for _ in range(2):
                cmd = ["pdflatex", "-interaction=nonstopmode", tex_file.name]
                subprocess.run(cmd, cwd=tmp, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        produced = tmp / "diagram.pdf"
        if not produced.exists():
            pdfs = list(tmp.glob("*.pdf"))
            if not pdfs:
                raise RuntimeError("Compilação LaTeX não produziu PDF.")
            produced = pdfs[0]
        shutil.copyfile(produced, out_pdf_path)
    return out_pdf_path


def pdf_to_png(pdf_path: os.PathLike[str] | str, out_png: os.PathLike[str] | str, dpi: int = 300, page: int = 1) -> Path:
    """Converte um PDF para PNG. Tenta pdftocairo, depois ImageMagick, depois Ghostscript.

    page é 1-based.
    """
    pdf = Path(pdf_path).resolve()
    png = Path(out_png).resolve()
    png.parent.mkdir(parents=True, exist_ok=True)

    if _which("pdftocairo"):
        cmd = [
            "pdftocairo",
            "-png",
            f"-r{dpi}",
            "-singlefile",
            f"-f{page}",
            f"-l{page}",
            str(pdf),
            str(png.with_suffix("").as_posix()),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return png

    im = _which("magick") or _which("convert")
    if im:
        cmd = [im, "-density", str(dpi), f"{pdf}[{page-1}]", "-quality", "100", str(png)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return png

    if _which("gs"):
        cmd = [
            "gs",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pngalpha",
            f"-r{dpi}",
            f"-dFirstPage={page}",
            f"-dLastPage={page}",
            f"-sOutputFile={png}",
            str(pdf),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return png

    raise RuntimeError("Nenhum conversor PDF->PNG encontrado (pdftocairo/ImageMagick/gs).")


def generate_pdf(arch, out_pdf: os.PathLike[str] | str, inline_styles: bool = True, include_colors: bool = True) -> Path:
    """Gera PDF diretamente a partir de uma arquitetura (fragmentos)."""
    doc = to_document(arch, inline_styles=inline_styles, include_colors=include_colors)
    return compile_tex_to_pdf(doc, out_pdf)


def generate_png(arch, out_png: os.PathLike[str] | str, dpi: int = 300, inline_styles: bool = True, include_colors: bool = True) -> Path:
    """Gera PNG diretamente a partir de uma arquitetura (fragmentos)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "out.pdf"
        generate_pdf(arch, pdf_path, inline_styles=inline_styles, include_colors=include_colors)
        return pdf_to_png(pdf_path, out_png, dpi=dpi)
