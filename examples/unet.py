from __future__ import annotations

import sys
from pathlib import Path

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


def main() -> Diagram:
    """Gera o diagrama UNet de forma sequencial (cada camada explícita).

    Estrutura: 4 níveis encoder -> bottleneck -> 4 níveis decoder com skips.
    """

    d = Diagram()
    out_dir = Path(__file__).resolve().parents[1] / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imagem de entrada
    img_path = Path(__file__).resolve().parents[1] / "models-examples" / "fcn8s" / "cats.jpg"
    d.add(Input(pathfile=img_path, name="input", width=8, height=8, anchor_scale=0.01))

    # ---------------- Encoder Nível 1 ----------------
    d.add(
        ConvConvRelu(
            name="enc_1",
            s_filer=128,
            n_filer=(64, 64),
            offset="(1.2,0,0)",
            to="(input-east)",
            width=(2, 2),
            height=40,
            depth=40,
        )
    )
    d.add(Connection("input", "enc_1"))
    d.add(
        Pool(
            name="pool_1",
            to="(enc_1-east)",
            height=30,
            depth=30,
            width=1,
            opacity=0.5,
        )
    )

    # ---------------- Encoder Nível 2 ----------------
    d.add(
        ConvConvRelu(
            name="enc_2",
            s_filer=256,
            n_filer=(128, 128),
            offset="(1.2,0,0)",
            to="(pool_1-east)",
            width=(3, 3),
            height=32,
            depth=32,
        )
    )
    d.add(Connection("pool_1", "enc_2"))
    d.add(
        Pool(
            name="pool_2",
            to="(enc_2-east)",
            height=24,
            depth=24,
            width=1,
            opacity=0.5,
        )
    )

    # ---------------- Encoder Nível 3 ----------------
    d.add(
        ConvConvRelu(
            name="enc_3",
            s_filer=512,
            n_filer=(256, 256),
            offset="(1.2,0,0)",
            to="(pool_2-east)",
            width=(4, 4),
            height=24,
            depth=24,
        )
    )
    d.add(Connection("pool_2", "enc_3"))
    d.add(
        Pool(
            name="pool_3",
            to="(enc_3-east)",
            height=18,
            depth=18,
            width=1,
            opacity=0.5,
        )
    )

    # ---------------- Encoder Nível 4 ----------------
    d.add(
        ConvConvRelu(
            name="enc_4",
            s_filer=1024,
            n_filer=(512, 512),
            offset="(1.4,0,0)",
            to="(pool_3-east)",
            width=(5, 5),
            height=16,
            depth=16,
        )
    )
    d.add(Connection("pool_3", "enc_4"))
    d.add(
        Pool(
            name="pool_4",
            to="(enc_4-east)",
            height=12,
            depth=12,
            width=1,
            opacity=0.5,
        )
    )

    # ---------------- Bottleneck ----------------
    d.add(
        ConvConvRelu(
            name="bottleneck",
            s_filer=2048,
            n_filer=(1024, 1024),
            offset="(1.6,0,0)",
            to="(pool_4-east)",
            width=(6, 6),
            height=12,
            depth=12,
            caption="Bottleneck",
        )
    )
    d.add(Connection("pool_4", "bottleneck"))

    # ---------------- Decoder Nível 4 ----------------
    d.add(
        UnPool(
            name="unpool_4",
            offset="(1.4,0,0)",
            to="(bottleneck-east)",
            width=1,
            height=16,
            depth=16,
            opacity=0.5,
        )
    )
    d.add(Connection("bottleneck", "unpool_4"))
    d.add(
        ConvConvRelu(
            name="dec_4",
            s_filer=1024,
            n_filer=(512, 512),
            to="(unpool_4-east)",
            width=(5, 5),
            height=16,
            depth=16,
        )
    )
    d.add(Skip(of="enc_4", to="dec_4", pos=1.25))

    # ---------------- Decoder Nível 3 ----------------
    d.add(
        UnPool(
            name="unpool_3",
            offset="(1.4,0,0)",
            to="(dec_4-east)",
            width=1,
            height=24,
            depth=24,
            opacity=0.5,
        )
    )
    d.add(Connection("dec_4", "unpool_3"))
    d.add(
        ConvConvRelu(
            name="dec_3",
            s_filer=512,
            n_filer=(256, 256),
            to="(unpool_3-east)",
            width=(4, 4),
            height=24,
            depth=24,
        )
    )
    d.add(Skip(of="enc_3", to="dec_3", pos=1.25))

    # ---------------- Decoder Nível 2 ----------------
    d.add(
        UnPool(
            name="unpool_2",
            offset="(1.4,0,0)",
            to="(dec_3-east)",
            width=1,
            height=32,
            depth=32,
            opacity=0.5,
        )
    )
    d.add(Connection("dec_3", "unpool_2"))
    d.add(
        ConvConvRelu(
            name="dec_2",
            s_filer=256,
            n_filer=(128, 128),
            to="(unpool_2-east)",
            width=(3, 3),
            height=32,
            depth=32,
        )
    )
    d.add(Skip(of="enc_2", to="dec_2", pos=1.25))

    # ---------------- Decoder Nível 1 ----------------
    d.add(
        UnPool(
            name="unpool_1",
            offset="(1.4,0,0)",
            to="(dec_2-east)",
            width=1,
            height=40,
            depth=40,
            opacity=0.5,
        )
    )
    d.add(Connection("dec_2", "unpool_1"))
    d.add(
        ConvConvRelu(
            name="dec_1",
            s_filer=128,
            n_filer=(64, 64),
            to="(unpool_1-east)",
            width=(2, 2),
            height=40,
            depth=40,
        )
    )
    d.add(Skip(of="enc_1", to="dec_1", pos=1.25))

    # ---------------- Saída ----------------
    d.add(
        ConvSoftMax(
            name="output",
            s_filer=2,
            to="(dec_1-east)",
            offset="(1.4,0,0)",
            width=1,
            height=40,
            depth=40,
            caption="Softmax",
        )
    )
    d.add(Connection("dec_1", "output"))

    # Renderização
    d.save_tex(out_dir / "unet.tex")
    d.render_pdf(out_dir / "unet.pdf", keep_tex=True)
    return d


if __name__ == "__main__":
    main()
