"""Advanced example demonstrating the improved PlotNN API."""

from __future__ import annotations

from pathlib import Path

from plotnn import (
    Connection,
    Conv,
    Diagram,
    DiagramRenderer,
    Pool,
    Skip,
)


def create_complex_network() -> Diagram:
    """Create a more complex neural network diagram."""
    d = Diagram()

    # Input layer
    d.add(Conv(name="input", n_filer=3, width=1, height=64, depth=64, caption="Input"))

    # First conv block
    d.add(Conv(name="conv1", n_filer=32, width=2, height=64, depth=64, to="(input-east)", caption="Conv1"))
    d.add(Pool(name="pool1", width=1, height=32, depth=32, to="(conv1-east)", caption="Pool1"))
    d.add(Connection(of="input", to="conv1"))
    d.add(Connection(of="conv1", to="pool1"))

    # Second conv block with skip connection
    d.add(Conv(name="conv2", n_filer=64, width=3, height=32, depth=32, to="(pool1-east)", caption="Conv2"))
    d.add(Pool(name="pool2", width=1, height=16, depth=16, to="(conv2-east)", caption="Pool2"))
    d.add(Connection(of="pool1", to="conv2"))
    d.add(Connection(of="conv2", to="pool2"))
    d.add(Skip(of="pool1", to="pool2", pos=1.5))

    # Third conv block
    d.add(Conv(name="conv3", n_filer=128, width=4, height=16, depth=16, to="(pool2-east)", caption="Conv3"))
    d.add(Pool(name="pool3", width=1, height=8, depth=8, to="(conv3-east)", caption="Pool3"))
    d.add(Connection(of="pool2", to="conv3"))
    d.add(Connection(of="conv3", to="pool3"))

    return d


def main() -> None:
    """Demonstrate the improved out-of-the-box functionality."""
    print("🚀 Creating complex neural network diagram...")

    # Create diagram
    d = create_complex_network()

    # Setup output directory
    out_dir = Path(__file__).resolve().parents[1] / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs using the simple API (out-of-the-box functionality)
    print("📝 Generating LaTeX...")
    tex_path = out_dir / "complex_diagram.tex"
    d.save_tex(tex_path, inline_styles=True)
    print(f"   ✓ LaTeX saved to {tex_path}")

    print("📄 Generating PDF...")
    pdf_path = out_dir / "complex_diagram.pdf"
    d.render_pdf(pdf_path, inline_styles=True)
    print(f"   ✓ PDF saved to {pdf_path}")

    print("🖼️  Generating PNG...")
    png_path = out_dir / "complex_diagram.png"
    d.render_png(png_path, dpi=200, inline_styles=True)
    print(f"   ✓ PNG saved to {png_path}")

    # Demonstrate advanced usage with DiagramRenderer
    print("🔧 Using advanced DiagramRenderer...")
    renderer = DiagramRenderer()

    svg_path = out_dir / "complex_diagram.svg"
    renderer.render_to_svg(d.elements, svg_path, inline_styles=True)
    print(f"   ✓ SVG saved to {svg_path}")

    # Show file sizes
    print("\n📊 Generated files:")
    for file_path in [tex_path, pdf_path, png_path, svg_path]:
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   {file_path.name}: {size:,} bytes")

    print(f"\n✨ All files generated successfully in {out_dir}")
    print("🎉 PlotNN now works completely out-of-the-box!")


if __name__ == "__main__":
    main()
