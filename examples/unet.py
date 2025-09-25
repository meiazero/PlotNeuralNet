from __future__ import annotations

from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plotnn import (
    Connection,
    ConvConvRelu,
    ConvSoftMax,
    Diagram,
    Input,
    Pool,
    Skip,
    UnPool,
)


def _add_encoder_stage(
    d: Diagram,
    name: str,
    prev: str,
    n_filt: int,
    size_hw: int,
    depth: int,
    width_pair: tuple[int, int],
    pool_opacity: float = 0.5,
):
    """Adiciona um bloco encoder (ConvConvRelu + Pool + Connection)."""
    ccr_name = f"enc_{name}"
    pool_name = f"pool_{name}"
    d.add(
        ConvConvRelu(
            name=ccr_name,
            s_filer=n_filt * 2,  # apenas para variar o label z
            n_filer=(n_filt, n_filt),
            offset="(1.2,0,0)",
            to=f"({prev}-east)",
            width=width_pair,
            height=size_hw,
            depth=size_hw,
        )
    )
    d.add(
        Pool(
            name=pool_name,
            offset="(0,0,0)",
            to=f"({ccr_name}-east)",
            width=1,
            height=max(4, size_hw - size_hw // 4),
            depth=max(4, size_hw - size_hw // 4),
            opacity=pool_opacity,
        )
    )
    d.add(Connection(prev, ccr_name))
    return ccr_name, pool_name


def _add_decoder_stage(
    d: Diagram,
    name: str,
    prev: str,
    skip_from: str,
    n_filt: int,
    size_hw: int,
    width_pair: tuple[int, int],
):
    """Adiciona um bloco decoder (UnPool + ConvConvRelu + Skip)."""
    un_name = f"unpool_{name}"
    dec_name = f"dec_{name}"
    d.add(
        UnPool(
            name=un_name,
            offset="(1.4,0,0)",
            to=f"({prev}-east)",
            width=1,
            height=size_hw,
            depth=size_hw,
            opacity=0.5,
        )
    )
    d.add(
        ConvConvRelu(
            name=dec_name,
            s_filer=n_filt * 2,
            n_filer=(n_filt, n_filt),
            offset="(0,0,0)",
            to=f"({un_name}-east)",
            width=width_pair,
            height=size_hw,
            depth=size_hw,
        )
    )
    d.add(Connection(prev, un_name))
    d.add(Skip(of=skip_from, to=dec_name, pos=1.25))
    return dec_name


def main() -> Diagram:
    """Gera o diagrama UNet e renderiza em PDF."""

    d = Diagram()
    out_dir = Path(__file__).resolve().parents[1] / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Caminho da imagem de exemplo (já existente no repo)
    img_path = Path(__file__).resolve().parents[1] / "models-examples" / "fcn8s" / "cats.jpg"
    if not img_path.exists():  # fallback para evitar falha
        img_path = Path("cats.jpg")

    d.add(Input(pathfile=img_path, name="input", width=6, height=6))

    # Encoder -----------------------------------------------------------------
    enc_specs = [
        ("1", 64, 40, 40, (2, 2)),
        ("2", 128, 32, 32, (3, 3)),
        ("3", 256, 24, 24, (4, 4)),
        ("4", 512, 16, 16, (5, 5)),
    ]

    last_name = "input"
    encoder_blocks: list[str] = []  # nomes dos ccr encoder para skip
    pool_names: list[str] = []
    for name, nf, h, dsize, width_pair in enc_specs:
        ccr_name, pool_name = _add_encoder_stage(
            d,
            name=name,
            prev=last_name,
            n_filt=nf,
            size_hw=h,
            depth=dsize,
            width_pair=width_pair,
        )
        encoder_blocks.append(ccr_name)
        pool_names.append(pool_name)
        last_name = pool_name

    # Bottleneck --------------------------------------------------------------
    bottleneck_name = "bottleneck"
    d.add(
        ConvConvRelu(
            name=bottleneck_name,
            s_filer=1024,
            n_filer=(1024, 1024),
            offset="(1.6,0,0)",
            to=f"({last_name}-east)",
            width=(6, 6),
            height=12,
            depth=12,
            caption="Bottleneck",
        )
    )
    d.add(Connection(last_name, bottleneck_name))

    # Decoder -----------------------------------------------------------------
    # Percorre encoder invertido para criar estágios decoder
    decoder_specs = [
        ("4", 512, 16, (5, 5)),
        ("3", 256, 24, (4, 4)),
        ("2", 128, 32, (3, 3)),
        ("1", 64, 40, (2, 2)),
    ]

    prev_name = bottleneck_name
    for name, nf, h, width_pair in decoder_specs:
        skip_from = f"enc_{name}"  # nome do encoder correspondente
        prev_name = _add_decoder_stage(
            d,
            name=name,
            prev=prev_name,
            skip_from=skip_from,
            n_filt=nf,
            size_hw=h,
            width_pair=width_pair,
        )

    # Camada de saída (ex: mapa de classes)
    d.add(
        ConvSoftMax(
            name="output",
            s_filer=2,  # supondo segmentação binária
            offset="(1.4,0,0)",
            to=f"({prev_name}-east)",
            width=1,
            height=40,
            depth=40,
            caption="Softmax",
        )
    )
    d.add(Connection(prev_name, "output"))

    pdf_path = out_dir / "unet.pdf"
    d.save_tex(path=out_dir / "unet.tex")
    d.render_pdf(pdf_path, keep_tex=True)

    return d


if __name__ == "__main__":
    main()
