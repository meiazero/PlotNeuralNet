# plotnn: Python API for Generating Neural Network Diagrams in LaTeX/TikZ

`plotnn` allows you to build neural network architectures programmatically in Python and generate LaTeX/TikZ code, PDF, or PNG outputs. It's lightweight, with no core dependencies, and focuses on ease of use.

## Installation

```bash
pip install plotnn
```

## Recommended Tools

- [Poetry](https://python-poetry.org)
- [Mise](https://mise.jdx.dev/)

## Getting Started

### Usage

Build diagrams fluently:

```python
from plotnn import Diagram, Input, Conv, Connection, Skip, Pool

d = (
    Diagram()
    .add(Conv(name="conv1", n_filer=64, width=2, height=32, depth=32))
    .add(Connection(of="conv1", to="pool1"))
    .add(Pool(name="pool1", width=1, height=16, depth=16))
    .add(Connection(of="conv1", to="pool1"))
    .add(Skip(of="conv1", to="pool1", pos=1.25))
)

# Generate outputs
d.save_tex("diagram.tex")  # LaTeX file
d.render_pdf("diagram.pdf")  # Requires pdflatex/latexmk
d.render_png("diagram.png", dpi=300)  # Requires pdftocairo/ImageMagick/gs
d.render_svg("diagram.svg")  # Optional, if pdftocairo available
```

For complex architectures like U-Net, use pre-built blocks:

```python
from plotnn import TwoConvPoolBlock, UnconvBlock

# ... add to Diagram as above
```

## Requirements for Rendering

- PDF: `pdflatex` or `latexmk`
- PNG/SVG: `pdftocairo` (preferred), or ImageMagick (`convert`), or Ghostscript (`gs`)

## Development

- Lint: make lint
- Format: make format
- Test: make test
- Build: make build

## Acknowledgements

**PlotNeuralNet** [artifact](https://doi.org/10.5281/zenodo.2526396)
