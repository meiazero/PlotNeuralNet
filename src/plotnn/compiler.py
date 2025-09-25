"""LaTeX compilation and format conversion utilities."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class LaTeXCompiler:
    """Handles LaTeX compilation to PDF."""

    def __init__(self):
        self.available_tools = self._check_available_tools()

    def _check_available_tools(self) -> dict[str, bool]:
        """Check which LaTeX tools are available."""
        return {
            'latexmk': shutil.which("latexmk") is not None,
            'pdflatex': shutil.which("pdflatex") is not None,
        }

    def compile_to_pdf(self, tex_content: str, out_pdf: str | Path) -> Path:
        """Compile LaTeX content to PDF."""
        out_pdf_path = Path(out_pdf).resolve()
        out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tex_file = tmp / "diagram.tex"
            tex_file.write_text(tex_content, encoding="utf-8")

            if self.available_tools['latexmk']:
                cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-silent", tex_file.name]
                subprocess.run(cmd, cwd=tmp, check=True)
            elif self.available_tools['pdflatex']:
                cmd = ["pdflatex", "-interaction=nonstopmode", "no-shell-escape", tex_file.name]
                # Run twice for references
                subprocess.run(cmd, cwd=tmp, check=False)
                subprocess.run(cmd, cwd=tmp, check=True)
            else:
                raise RuntimeError("No LaTeX compiler found. Install latexmk or pdflatex.")

            produced = tmp / "diagram.pdf"
            if not produced.exists():
                raise RuntimeError("LaTeX compilation failed to produce PDF. Check logs.")

            shutil.copyfile(produced, out_pdf_path)

        logger.info(f"PDF generated at {out_pdf_path}")
        return out_pdf_path


class FormatConverter:
    """Handles conversion from PDF to other formats."""

    def __init__(self):
        self.available_tools = self._check_available_tools()

    def _check_available_tools(self) -> dict[str, bool]:
        """Check which conversion tools are available."""
        return {
            'pdftocairo': shutil.which("pdftocairo") is not None,
            'magick': shutil.which("magick") is not None,
            'convert': shutil.which("convert") is not None,
            'gs': shutil.which("gs") is not None,
        }

    def pdf_to_format(
        self,
        pdf_path: Path,
        out_path: Path,
        format: str,
        dpi: int = 300,
        page: int = 1
    ) -> Path:
        """Convert PDF to specified format."""
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if format not in ("png", "svg"):
            raise ValueError("Format must be 'png' or 'svg'")

        # Try pdftocairo first (best quality)
        if self.available_tools['pdftocairo']:
            return self._convert_with_pdftocairo(pdf_path, out_path, format, dpi, page)

        # For SVG, pdftocairo is required
        if format == "svg":
            raise RuntimeError("SVG conversion requires pdftocairo. Please install poppler-utils.")

        # Try ImageMagick
        if self.available_tools['magick'] or self.available_tools['convert']:
            return self._convert_with_imagemagick(pdf_path, out_path, dpi, page)

        # Try Ghostscript
        if self.available_tools['gs']:
            return self._convert_with_ghostscript(pdf_path, out_path, dpi, page)

        raise RuntimeError(
            f"No tool found for {format} conversion. "
            "Install poppler-utils, ImageMagick, or Ghostscript."
        )

    def _convert_with_pdftocairo(
        self, pdf_path: Path, out_path: Path, format: str, dpi: int, page: int
    ) -> Path:
        """Convert using pdftocairo."""
        tool = shutil.which("pdftocairo")
        args = ["-r", str(dpi), f"-f={page}", f"-l={page}", str(pdf_path), "-singlefile"]

        if format == "png":
            # For PNG, pdftocairo adds the extension automatically, so we need to provide the path without extension
            base_path = out_path.with_suffix("")
            cmd = [tool, "-png"] + args + [str(base_path)]
        else:  # svg
            cmd = [tool, "-svg"] + args + [str(out_path)]

        subprocess.run(cmd, check=True)

        return out_path

    def _convert_with_imagemagick(
        self, pdf_path: Path, out_path: Path, dpi: int, page: int
    ) -> Path:
        """Convert using ImageMagick."""
        tool = shutil.which("magick") or shutil.which("convert")
        cmd = [
            tool, "-density", str(dpi), f"{pdf_path}[{page-1}]",
            "-quality", "100", str(out_path)
        ]
        subprocess.run(cmd, check=True)
        return out_path

    def _convert_with_ghostscript(
        self, pdf_path: Path, out_path: Path, dpi: int, page: int
    ) -> Path:
        """Convert using Ghostscript."""
        tool = shutil.which("gs")
        cmd = [
            tool, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pngalpha",
            f"-r{dpi}", f"-dFirstPage={page}", f"-dLastPage={page}",
            f"-sOutputFile={out_path}", str(pdf_path),
        ]
        subprocess.run(cmd, check=True)
        return out_path
