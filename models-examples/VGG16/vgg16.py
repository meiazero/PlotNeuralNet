"""Exemplo VGG16 com API Python moderna.

Usa blocos ConvConvRelu e Pool para aproximar o layout de `vgg16.tex`.
"""

from __future__ import annotations

from pathlib import Path

try:
    from plotnn import Diagram, ConvConvRelu, Pool, SoftMax, Connection
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "A biblioteca 'plotnn' não está instalada. Instale com 'pip install plotnn' ou 'pip install -e .' dentro do repositório."
    ) from e


def build_vgg16() -> Diagram:
    d = Diagram()

    # Bloco conv1 (64,64)
    d.add(
        ConvConvRelu(
            name="cr1",
            n_filer=(64, 64),
            s_filer=224,
            width=(2, 2),
            height=40,
            depth=40,
        )
    )
    d.add(
        Pool(
            name="p1",
            to="(cr1-east)",
            height=35,
            depth=35,
        )
    )
    d.add(Connection(of="cr1", to="p1"))

    # Bloco conv2 (128,128)
    d.add(
        ConvConvRelu(
            name="cr2",
            n_filer=(128, 128),
            s_filer=112,
            to="(p1-east)",
            width=(3, 3),
            height=35,
            depth=35,
        ),
        Connection(of="p1", to="cr2"),
    )
    d.add(
        Pool(
            name="p2",
            to="(cr2-east)",
            height=30,
            depth=30,
        ),
        Connection(of="cr2", to="p2"),
    )

    # Bloco conv3 (256,256,256) - representado aqui com duas bandas (terceira pode ser adicionada se desejar)
    d.add(
        ConvConvRelu(
            name="cr3",
            n_filer=(256, 256),
            s_filer=56,
            to="(p2-east)",
            width=(4, 4),
            height=30,
            depth=30,
        ),
        Connection(of="p2", to="cr3"),
    )
    d.add(
        Pool(
            name="p3",
            to="(cr3-east)",
            height=23,
            depth=23,
        ),
        Connection(of="cr3", to="p3"),
    )

    # Bloco conv4 (512,512,512)
    d.add(
        ConvConvRelu(
            name="cr4",
            n_filer=(512, 512),
            s_filer=28,
            to="(p3-east)",
            width=(7, 7),
            height=23,
            depth=23,
        ),
        Connection(of="p3", to="cr4"),
    )
    d.add(
        Pool(
            name="p4",
            to="(cr4-east)",
            height=15,
            depth=15,
        ),
        Connection(of="cr4", to="p4"),
    )

    # Bloco conv5 (512,512,512)
    d.add(
        ConvConvRelu(
            name="cr5",
            n_filer=(512, 512),
            s_filer=14,
            to="(p4-east)",
            width=(7, 7),
            height=15,
            depth=15,
        ),
        Connection(of="p4", to="cr5"),
    )
    d.add(
        Pool(
            name="p5",
            to="(cr5-east)",
            height=10,
            depth=10,
        ),
        Connection(of="cr5", to="p5"),
    )

    # Camadas fully connected simplificadas
    d.add(
        ConvConvRelu(
            name="fc6",
            n_filer=(1, 1),
            s_filer=4096,
            to="(p5-east)",
            width=(3, 3),
            height=3,
            depth=100,
        ),
        Connection(of="p5", to="fc6"),
    )
    d.add(
        ConvConvRelu(
            name="fc7",
            n_filer=(1, 1),
            s_filer=4096,
            to="(fc6-east)",
            width=(3, 3),
            height=3,
            depth=100,
        ),
        Connection(of="fc6", to="fc7"),
    )

    d.add(
        SoftMax(
            name="softmax",
            s_filer=1000,  # K classes
            to="(fc7-east)",
            width=2,
            height=3,
            depth=25,
        ),
        Connection(of="fc7", to="softmax"),
    )
    return d


def main() -> None:
    diagram = build_vgg16()
    out_dir = Path(__file__).parent
    tex_path = out_dir / "vgg16_modern.tex"
    pdf_path = out_dir / "vgg16_modern.pdf"
    diagram.save_tex(tex_path, inline_styles=True)
    try:
        diagram.render_pdf(pdf_path, inline_styles=True)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Falha ao gerar PDF: {e}")


if __name__ == "__main__":
    main()
