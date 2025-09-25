"""High-level rendering interface for neural network diagrams."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .compiler import FormatConverter, LaTeXCompiler
from .templates import LaTeXTemplate

if TYPE_CHECKING:
    from .blocks import Element


class DiagramRenderer:
    """High-level interface for rendering neural network diagrams."""

    def __init__(self):
        self.latex_compiler = LaTeXCompiler()
        self.format_converter = FormatConverter()

    def _elements_to_latex(self, elements: list[Element] | list[str]) -> list[str]:
        """Convert elements to LaTeX strings."""
        if not elements:
            return []

        # Check if first element is a string or has build method
        if isinstance(elements[0], str):
            return elements  # type: ignore

        # Elements have build method
        latex_parts = []
        for element in elements:
            latex_parts.extend(element.build())  # type: ignore
        return latex_parts

    def render_to_tex(
        self,
        elements: list[Element] | list[str],
        output_path: str | Path,
        inline_styles: bool = True,
        include_colors: bool = True
    ) -> Path:
        """Render diagram elements to LaTeX file."""
        latex_parts = self._elements_to_latex(elements)

        # Generate full document
        document = LaTeXTemplate.full_document(
            latex_parts,
            inline_styles=inline_styles,
            include_colors=include_colors
        )

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")

        return output_path

    def render_to_pdf(
        self,
        elements: list[Element] | list[str],
        output_path: str | Path,
        inline_styles: bool = True,
        include_colors: bool = True
    ) -> Path:
        """Render diagram elements to PDF file."""
        latex_parts = self._elements_to_latex(elements)

        # Generate LaTeX document
        document = LaTeXTemplate.full_document(
            latex_parts,
            inline_styles=inline_styles,
            include_colors=include_colors
        )

        # Compile to PDF
        return self.latex_compiler.compile_to_pdf(document, output_path)

    def render_to_png(
        self,
        elements: list[Element] | list[str],
        output_path: str | Path,
        dpi: int = 300,
        inline_styles: bool = True,
        include_colors: bool = True
    ) -> Path:
        """Render diagram elements to PNG file."""
        # First generate PDF in temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "temp.pdf"
            self.render_to_pdf(elements, pdf_path, inline_styles, include_colors)

            # Convert PDF to PNG
            return self.format_converter.pdf_to_format(
                pdf_path, Path(output_path), "png", dpi=dpi
            )

    def render_to_svg(
        self,
        elements: list[Element] | list[str],
        output_path: str | Path,
        inline_styles: bool = True,
        include_colors: bool = True
    ) -> Path:
        """Render diagram elements to SVG file."""
        # First generate PDF in temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "temp.pdf"
            self.render_to_pdf(elements, pdf_path, inline_styles, include_colors)

            # Convert PDF to SVG
            return self.format_converter.pdf_to_format(
                pdf_path, Path(output_path), "svg"
            )
