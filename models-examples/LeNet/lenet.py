"""Exemplo LeNet usando a API Python moderna (`plotnn`).

Gera o diagrama equivalente ao `lenet.tex` legado.

Execução:
    python lenet.py  (gera lenet_modern.tex e, se possível, lenet_modern.pdf)
"""

from __future__ import annotations

from pathlib import Path

try:
    from plotnn import Diagram, Conv, Pool, SoftMax, Connection
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "A biblioteca 'plotnn' não está instalada. Instale com 'pip install plotnn' ou 'pip install -e .' dentro do repositório."
    ) from e


def build_lenet() -> Diagram:
    d = Diagram()

    # input (representado como um "conv" raso só para visualização de volume)
    d.add(
        Conv(name="conv0", n_filer=1, s_filer=32, width=1, height=32, depth=32)
    )

    # conv1 6 filtros 28x28
    d.add(
        Conv(
            name="conv1",
            n_filer=6,
            s_filer=28,
            to="(conv0-east)",
            width=6,
            height=28,
            depth=28,
        ),
        Connection(of="conv0", to="conv1"),
    )

    # pool1 14x14
    d.add(
        Pool(
            name="pool1",
            to="(conv1-east)",
            width=6,
            height=14,
            depth=14,
        ),
        Connection(of="conv1", to="pool1"),
    )

    # conv2 16 filtros 10x10
    d.add(
        Conv(
            name="conv2",
            n_filer=16,
            s_filer=10,
            to="(pool1-east)",
            width=16,
            height=10,
            depth=10,
        ),
        Connection(of="pool1", to="conv2"),
    )

    # pool2 5x5
    d.add(
        Pool(
            name="pool2",
            to="(conv2-east)",
            width=16,
            height=5,
            depth=5,
        ),
        Connection(of="conv2", to="pool2"),
    )

    # conv3 (camada totalmente conectada achatada) 120
    d.add(
        Conv(
            name="conv3",
            n_filer=1,
            s_filer=120,
            to="(pool2-east)",
            width=1,
            height=1,
            depth=120,
        ),
        Connection(of="pool2", to="conv3"),
    )

    # conv4 (FC) 84
    d.add(
        Conv(
            name="conv4",
            n_filer=1,
            s_filer=84,
            to="(conv3-east)",
            width=1,
            height=1,
            depth=84,
        ),
        Connection(of="conv3", to="conv4"),
    )

    # softmax 10 classes
    d.add(
        SoftMax(name="soft1", s_filer=10, to="(conv4-east)", width=2, height=3, depth=25, caption="SOFT"),
        Connection(of="conv4", to="soft1"),
    )

    return d


def main() -> None:
    diagram = build_lenet()
    out_dir = Path(__file__).parent

    tex_path = out_dir / "lenet_modern.tex"
    pdf_path = out_dir / "lenet_modern.pdf"

    diagram.save_tex(tex_path, inline_styles=True)
    try:
        diagram.render_pdf(pdf_path, inline_styles=True)
    except Exception as e:  # pragma: no cover - ambiente sem LaTeX
        print(f"[WARN] Falha ao gerar PDF: {e}")


if __name__ == "__main__":
    main()
