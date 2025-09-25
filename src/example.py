from __future__ import annotations

from pathlib import Path

from plotnn import (
    Connection,
    Conv,
    Diagram,
    Pool,
    Skip,
)


def build_diagram() -> Diagram:

    d = (
        Diagram()
        .add(Conv(name="conv1", n_filer=64, width=2, height=32, depth=32))
        .add(Connection(of="conv1", to="pool1"))
        .add(Pool(name="pool1", width=1, height=16, depth=16))
        .add(Connection(of="conv1", to="pool1"))
        .add(Skip(of="conv1", to="pool1", pos=1.25))
    )

    return d


def main() -> None:
    d = build_diagram()
    out_dir = Path(__file__).resolve().parents[1] / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_path = out_dir / "diagram.tex"
    pdf_path = out_dir / "diagram.pdf"
    png_path = out_dir / "diagram.png"
    svg_path = out_dir / "diagram.svg"

    d.save_tex(path=tex_path.as_posix(), inline_styles=True)
    d.render_pdf(pdf_path, inline_styles=True)
    d.render_png(png_path, dpi=300, inline_styles=True)
    d.render_svg(svg_path, inline_styles=True)


if __name__ == "__main__":
    main()
