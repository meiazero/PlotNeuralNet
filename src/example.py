from __future__ import annotations

from pathlib import Path

from plotnn import (
    Connection,
    Conv,
    ConvSoftMax,
    Diagram,
    Input,
    Skip,
    TwoConvPoolBlock,
    UnconvBlock,
)


def build_unet_diagram() -> Diagram:
    repo_root = Path(__file__).resolve().parents[1]
    img_path = repo_root / "models-examples" / "fcn8s" / "cats.jpg"

    d = Diagram()

    d.add(Input(pathfile=img_path, name="in", width=6, height=6))

    d.add(
        TwoConvPoolBlock(
            name="down1",
            bottom="in",
            top="p1",
            n_filer=64,
            s_filer=256,
            offset="(1,0,0)",
            size=(32, 32, 4),
        )
    )
    d.add(
        TwoConvPoolBlock(
            name="down2",
            bottom="p1",
            top="p2",
            n_filer=128,
            s_filer=256,
            offset="(1,0,0)",
            size=(28, 28, 4),
        )
    )
    d.add(
        TwoConvPoolBlock(
            name="down3",
            bottom="p2",
            top="p3",
            n_filer=256,
            s_filer=256,
            offset="(1,0,0)",
            size=(24, 24, 4),
        )
    )
    d.add(
        TwoConvPoolBlock(
            name="down4",
            bottom="p3",
            top="p4",
            n_filer=512,
            s_filer=256,
            offset="(1,0,0)",
            size=(20, 20, 4),
        )
    )

    d.add(
        Conv(
            name="bottleneck",
            to="(p4-east)",
            offset="(1,0,0)",
            n_filer=1024,
            s_filer=256,
            width=4,
            height=18,
            depth=18,
        )
    )
    d.add(Connection("p4", "bottleneck"))

    d.add(
        UnconvBlock(
            name="up4",
            bottom="bottleneck",
            top="u4",
            n_filer=512,
            s_filer=256,
            offset="(1,0,0)",
            size=(20, 20, 4),
        )
    )
    d.add(Skip(of="ccr_down4", to="ccr_up4", pos=1.25))

    d.add(
        UnconvBlock(
            name="up3",
            bottom="u4",
            top="u3",
            n_filer=256,
            s_filer=256,
            offset="(1,0,0)",
            size=(24, 24, 4),
        )
    )
    d.add(Skip(of="ccr_down3", to="ccr_up3", pos=1.25))

    d.add(
        UnconvBlock(
            name="up2",
            bottom="u3",
            top="u2",
            n_filer=128,
            s_filer=256,
            offset="(1,0,0)",
            size=(28, 28, 4),
        )
    )
    d.add(Skip(of="ccr_down2", to="ccr_up2", pos=1.25))

    d.add(
        UnconvBlock(
            name="up1",
            bottom="u2",
            top="u1",
            n_filer=64,
            s_filer=256,
            offset="(1,0,0)",
            size=(32, 32, 4),
        )
    )
    d.add(Skip(of="ccr_down1", to="ccr_up1", pos=1.25))

    d.add(
        ConvSoftMax(
            name="softmax",
            to="(u1-east)",
            offset="(1,0,0)",
            s_filer=2,
            width=1,
            height=8,
            depth=8,
        )
    )
    d.add(Connection("u1", "softmax"))

    return d


def main() -> None:
    d = build_unet_diagram()
    out_dir = Path(__file__).resolve().parents[1] / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pdf_path = out_dir / "unet.pdf"
    # png_path = out_dir / "unet.png"
    # svg_path = out_dir / "unet.svg"

    # d.render_pdf(pdf_path)
    # try:
    #     d.render_png(png_path, dpi=300)
    #     d.render_svg(svg_path)
    # except RuntimeError as e:
    #     print(f"Warning: {e}")

    # print(f"PDF: {pdf_path}")
    # print(f"PNG: {png_path}")
    # print(f"SVG: {svg_path}")
    d.save_tex(path=f"{out_dir}/unet.tex", inline_styles=True, include_colors=True)


if __name__ == "__main__":
    main()
