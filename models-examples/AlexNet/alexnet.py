"""Exemplo AlexNet usando API Python.

"""

from __future__ import annotations

from pathlib import Path

try:  # Garantir que o exemplo só funciona se a lib estiver instalada
    from plotnn import Diagram, Conv, Pool, SoftMax, Connection
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "A biblioteca 'plotnn' não está instalada. Instale com 'pip install plotnn' ou, no repositório, 'pip install -e .'"
    ) from e


def build_alexnet() -> Diagram:
    d = Diagram()

    d.add(Conv(name="conv0", n_filer=3, s_filer=224, width=3, height=45, depth=45))

    d.add(
        Conv(
            name="conv1",
            n_filer=96,
            s_filer=55,
            to="(conv0-east)",
            width=5,
            height=11,
            depth=11,
        ),
        Connection(of="conv0", to="conv1"),
    )

    d.add(
        Pool(
            name="pool1",
            to="(conv1-east)",
            width=1,
            height=5,
            depth=5,
        ),
        Connection(of="conv1", to="pool1"),
    )

    d.add(
        Conv(
            name="conv2",
            n_filer=256,
            s_filer=27,
            to="(pool1-east)",
            width=13,
            height=5,
            depth=5,
        ),
        Connection(of="pool1", to="conv2"),
    )

    d.add(
        Pool(
            name="pool2",
            to="(conv2-east)",
            width=1,
            height=3,
            depth=3,
        ),
        Connection(of="conv2", to="pool2"),
    )

    # conv3/4/5
    d.add(
        Conv(
            name="conv3",
            n_filer=384,
            s_filer=13,
            to="(pool2-east)",
            width=19,
            height=3,
            depth=3,
        ),
        Connection(of="pool2", to="conv3"),
    )
    d.add(
        Conv(
            name="conv4",
            n_filer=384,
            s_filer=13,
            to="(conv3-east)",
            width=19,
            height=3,
            depth=3,
        ),
        Connection(of="conv3", to="conv4"),
    )
    d.add(
        Conv(
            name="conv5",
            n_filer=256,
            s_filer=13,
            to="(conv4-east)",
            width=13,
            height=3,
            depth=3,
        ),
        Connection(of="conv4", to="conv5"),
    )

    d.add(
        Pool(
            name="pool3",
            to="(conv5-east)",
            width=1,
            height=1,
            depth=1,
        ),
        Connection(of="conv5", to="pool3"),
    )

    # FC camadas (modeladas como conv estreitas)
    d.add(
        Conv(
            name="fc1",
            n_filer=1,
            s_filer=4096,
            to="(pool3-east)",
            width=1,
            height=1,
            depth=41,
        ),
        Connection(of="pool3", to="fc1"),
    )
    d.add(
        Conv(
            name="fc2",
            n_filer=1,
            s_filer=4096,
            to="(fc1-east)",
            width=1,
            height=1,
            depth=41,
        ),
        Connection(of="fc1", to="fc2"),
    )

    d.add(
        SoftMax(
            name="soft1",
            s_filer=1000,
            to="(fc2-east)",
            width=2,
            height=3,
            depth=25,
            caption="SOFT",
        ),
        Connection(of="fc2", to="soft1"),
    )
    return d


def main() -> None:
    diagram = build_alexnet()
    out_dir = Path(__file__).parent
    tex_path = out_dir / "alexnet_modern.tex"
    pdf_path = out_dir / "alexnet_modern.pdf"
    diagram.save_tex(tex_path, inline_styles=True)
    try:
        diagram.render_pdf(pdf_path, inline_styles=True)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Falha ao gerar PDF: {e}")


if __name__ == "__main__":
    main()
