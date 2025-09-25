"""LaTeX template generation for neural network diagrams."""

from __future__ import annotations

from pathlib import Path


def _layers_dir_path() -> str:
    """Get the path to the layers directory."""
    path = Path(__file__).parent / "layers"
    return str(path.resolve()).replace("\\", "/")


def _read_pkg_text(*rel_parts: str) -> str:
    """Read text from a package file."""
    p = Path(__file__).parent.joinpath(*rel_parts)
    return p.read_text(encoding="utf-8")


def _inline_layers_tex() -> str:
    """Read and combine all layer style files inline."""
    parts: list[str] = []

    # Add init.tex content but remove \usepackage lines
    init = _read_pkg_text("layers", "init.tex")
    init_lines = [ln for ln in init.splitlines() if not ln.lstrip().startswith("\\usepackage")]
    parts.append("\n".join(init_lines))

    # Add style files without the \ProvidesPackage line
    for sty in ("Ball.sty", "Box.sty", "RightBandedBox.sty"):
        txt = _read_pkg_text("layers", sty)
        lines = [ln for ln in txt.splitlines() if not ln.lstrip().startswith("\\ProvidesPackage")]
        parts.append("\n".join(lines))

    return "\n".join(parts) + "\n"


class LaTeXTemplate:
    """Generates LaTeX document templates for neural network diagrams."""

    @staticmethod
    def document_header_inline() -> str:
        """Generate document header with inline styles."""
        return (
            """\\documentclass[border=8pt, multi, tikz]{standalone}
\\usetikzlibrary{positioning}
\\usetikzlibrary{3d}
\\usetikzlibrary{calc}
\\usetikzlibrary{fit}
\\usetikzlibrary{arrows.meta}
\\usetikzlibrary{shapes.geometric}
"""
            + _inline_layers_tex()
        )

    @staticmethod
    def document_header_external() -> str:
        """Generate document header with external style files."""
        pathlayers = _layers_dir_path()
        if not pathlayers.endswith("/"):
            pathlayers += "/"
        return f"""\\documentclass[border=8pt, multi, tikz]{{standalone}}
\\usepackage{{import}}
\\subimport{{{pathlayers}}}{{init}}
\\usetikzlibrary{{positioning}}
\\usetikzlibrary{{3d}}
\\usetikzlibrary{{calc}}
\\usetikzlibrary{{fit}}
\\usetikzlibrary{{arrows.meta}}
\\usetikzlibrary{{shapes.geometric}}
"""

    @staticmethod
    def color_definitions() -> str:
        """Generate color definitions."""
        return """
\\def\\ConvColor{rgb:yellow,5;red,2.5;white,5}
\\def\\ConvReluColor{rgb:yellow,5;red,5;white,5}
\\def\\PoolColor{rgb:red,1;black,0.3}
\\def\\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\\def\\FcColor{rgb:blue,5;red,2.5;white,5}
\\def\\FcReluColor{rgb:blue,5;red,5;white,4}
\\def\\SoftmaxColor{rgb:magenta,5;black,7}
\\def\\SumColor{rgb:blue,5;green,15}
\\def\\ActivationColor{rgb:yellow,2;green,5}
\\def\\NormColor{rgb:blue,2;white,5}
\\def\\RNNColor{rgb:green,4;blue,2;white,5}
\\def\\OpColor{rgb:magenta,3;black,3}
\\def\\GenericColor{rgb:black,2;white,8}
\\def\\DepthwiseColor{rgb:yellow,3;green,2;white,6}
\\def\\SepConvColor{rgb:yellow,5;green,4;white,4}
\\def\\TransposeConvColor{rgb:blue,4;green,3;white,5}
\\def\\FlattenColor{rgb:black,3;white,9}
\\def\\SEColor{rgb:blue,6;green,2;white,2}
\\def\\TransformerBlockColor{rgb:magenta,4;blue,4;white,5}
"""

    @staticmethod
    def document_begin() -> str:
        """Generate document begin section."""
        return """
\\newcommand{\\copymidarrow}{\\tikz \\draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\\begin{document}
\\begin{tikzpicture}
\\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\\edgecolor,opacity=0.7]
\\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""

    @staticmethod
    def document_end() -> str:
        """Generate document end section."""
        return """
\\end{tikzpicture}
\\end{document}
"""

    @classmethod
    def full_document(
        cls, content: list[str], inline_styles: bool = True, include_colors: bool = True
    ) -> str:
        """Generate complete LaTeX document."""
        parts = []

        if inline_styles:
            parts.append(cls.document_header_inline())
        else:
            parts.append(cls.document_header_external())

        if include_colors:
            parts.append(cls.color_definitions())

        parts.append(cls.document_begin())

        parts.extend(content)

        parts.append(cls.document_end())

        return "".join(parts)
