from __future__ import annotations

"""Exemplo simples de uma CNN sequencial para demonstrar a API plotnn.

Estrutura:
Input -> Conv(32) -> Pool -> Conv(64) -> Pool -> Conv(128) -> Softmax

Gera arquivos em build/basic_cnn.(tex|pdf)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plotnn import Diagram, Input, ConvConvRelu, Pool, ConvSoftMax, Connection  # noqa: E402


def main() -> Diagram:
    d = Diagram()
    out_dir = PROJECT_ROOT / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imagem de entrada reutilizada (se não existir, ignora)
    img_path = PROJECT_ROOT / "models-examples" / "fcn8s" / "cats.jpg"
    if not img_path.exists():
        img_path = Path("cats.jpg")

    d.add(Input(pathfile=img_path, name="input", width=5, height=5))

    # Primeira conv
    d.add(
        ConvConvRelu(
            name="conv1",
            s_filer=64,
            n_filer=(32, 32),
            offset="(1.2,0,0)",
            to="(input-east)",
            width=(2, 2),
            height=32,
            depth=32,
        )
    )
    d.add(Connection("input", "conv1"))
    d.add(Pool(name="pool1", to="(conv1-east)", height=24, depth=24, width=1, opacity=0.5))

    # Segunda conv
    d.add(
        ConvConvRelu(
            name="conv2",
            s_filer=128,
            n_filer=(64, 64),
            offset="(1.4,0,0)",
            to="(pool1-east)",
            width=(3, 3),
            height=24,
            depth=24,
        )
    )
    d.add(Connection("pool1", "conv2"))
    d.add(Pool(name="pool2", to="(conv2-east)", height=16, depth=16, width=1, opacity=0.5))

    # Terceira conv (redução adicional)
    d.add(
        ConvConvRelu(
            name="conv3",
            s_filer=256,
            n_filer=(128, 128),
            offset="(1.6,0,0)",
            to="(pool2-east)",
            width=(4, 4),
            height=16,
            depth=16,
        )
    )
    d.add(Connection("pool2", "conv3"))

    # Saída softmax (ex: 10 classes)
    d.add(
        ConvSoftMax(
            name="output",
            s_filer=10,
            offset="(1.6,0,0)",
            to="(conv3-east)",
            width=1,
            height=14,
            depth=14,
            caption="Softmax",
        )
    )
    d.add(Connection("conv3", "output"))

    # Render
    d.save_tex(out_dir / "basic_cnn.tex")
    d.render_pdf(out_dir / "basic_cnn.pdf", keep_tex=True)

    return d


if __name__ == "__main__":  # pragma: no cover
    main()
