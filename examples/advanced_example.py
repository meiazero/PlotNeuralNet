"""Advanced example demonstrating the improved PlotNN API."""

from __future__ import annotations

from pathlib import Path

from plotnn import (
    Connection,
    Conv,
    Diagram,
    Pool,
    Skip,
)


def create_complex_network() -> Diagram:
    """Create a more complex neural network diagram."""
    d = Diagram()

    # Input layer
    d.add(Conv(name="input", n_filer=3, width=1, height=64, depth=64, caption="Input"))

    # First conv block
    d.add(
        Conv(
            name="conv1",
            n_filer=32,
            width=2,
            height=64,
            depth=64,
            to="(input-east)",
            caption="Conv1",
        )
    )
    d.add(Pool(name="pool1", width=1, height=32, depth=32, to="(conv1-east)", caption="Pool1"))
    d.add(Connection(of="input", to="conv1"))
    d.add(Connection(of="conv1", to="pool1"))

    # Second conv block with skip connection
    d.add(
        Conv(
            name="conv2",
            n_filer=64,
            width=3,
            height=32,
            depth=32,
            to="(pool1-east)",
            caption="Conv2",
        )
    )
    d.add(Pool(name="pool2", width=1, height=16, depth=16, to="(conv2-east)", caption="Pool2"))
    d.add(Connection(of="pool1", to="conv2"))
    d.add(Connection(of="conv2", to="pool2"))
    d.add(Skip(of="pool1", to="pool2", pos=1.5))

    # Third conv block
    d.add(
        Conv(
            name="conv3",
            n_filer=128,
            width=4,
            height=16,
            depth=16,
            to="(pool2-east)",
            caption="Conv3",
        )
    )
    d.add(Pool(name="pool3", width=1, height=8, depth=8, to="(conv3-east)", caption="Pool3"))
    d.add(Connection(of="pool2", to="conv3"))
    d.add(Connection(of="conv3", to="pool3"))

    return d


def main() -> None:
    d = create_complex_network()

    out_dir = Path(__file__).resolve().parents[1] / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    tex_path = out_dir / "complex_diagram.tex"
    d.save_tex(tex_path, inline_styles=True)

    pdf_path = out_dir / "complex_diagram.pdf"
    d.render_pdf(pdf_path, inline_styles=True)


if __name__ == "__main__":
    main()
