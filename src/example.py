from __future__ import annotations

from pathlib import Path

from plotnn.blocks import (
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
	"""Monta um diagrama UNet simples usando a nova API OO.

	Notas:
		- Usa um caminho absoluto para a imagem de entrada, garantindo compilação do LaTeX
        mesmo em diretório temporário.
		- Os nomes internos seguem o padrão dos blocos para facilitar conexões e skips.
	"""
	repo_root = Path(__file__).resolve().parents[1]
	img_path = (repo_root / "models-examples" / "fcn8s" / "cats.jpg").as_posix()

	d = Diagram()

	# Entrada (imagem)
	d.add(Input(pathfile=img_path, name="in", to="(-3,0,0)", width=6, height=6))

	# Caminho de descida (encoder)
	d.add(TwoConvPoolBlock(name="down1", botton="in", top="p1", n_filer=64, s_filer=256, offset="(1,0,0)", size=(32, 32, 3.5)))
	d.add(TwoConvPoolBlock(name="down2", botton="p1", top="p2", n_filer=128, s_filer=256, offset="(1,0,0)", size=(28, 28, 3.5)))
	d.add(TwoConvPoolBlock(name="down3", botton="p2", top="p3", n_filer=256, s_filer=256, offset="(1,0,0)", size=(24, 24, 3.5)))
	d.add(TwoConvPoolBlock(name="down4", botton="p3", top="p4", n_filer=512, s_filer=256, offset="(1,0,0)", size=(20, 20, 3.5)))

	# Gargalo (bottleneck)
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

	# Caminho de subida (decoder) com skips
	d.add(UnconvBlock(name="up4", botton="bottleneck", top="u4", n_filer=512, s_filer=256, offset="(1,0,0)", size=(20, 20, 3.5)))
	d.add(Skip(of="ccr_down4", to="ccr_up4", pos=1.25))

	d.add(UnconvBlock(name="up3", botton="u4", top="u3", n_filer=256, s_filer=256, offset="(1,0,0)", size=(24, 24, 3.5)))
	d.add(Skip(of="ccr_down3", to="ccr_up3", pos=1.25))

	d.add(UnconvBlock(name="up2", botton="u3", top="u2", n_filer=128, s_filer=256, offset="(1,0,0)", size=(28, 28, 3.5)))
	d.add(Skip(of="ccr_down2", to="ccr_up2", pos=1.25))

	d.add(UnconvBlock(name="up1", botton="u2", top="u1", n_filer=64, s_filer=256, offset="(1,0,0)", size=(32, 32, 3.5)))
	d.add(Skip(of="ccr_down1", to="ccr_up1", pos=1.25))

	# Saída (softmax)
	d.add(
		ConvSoftMax(
			name="softmax",
			to="(u1-east)",
			offset="(1,0,0)",
			s_filer=2,  # ex.: 2 classes
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

	pdf_path = (out_dir / "unet.pdf").as_posix()
	png_path = (out_dir / "unet.png").as_posix()

	# Gera PDF e PNG diretamente do diagrama
	# d.render_pdf(pdf_path)

	try:
		d.render_png(png_path, dpi=300)
	except Exception as e:
		# PNG depende de ferramentas externas (pdftocairo/ImageMagick/gs);
		# se não disponíveis, mantemos só o PDF.
		print(f"Aviso: não foi possível gerar PNG: {e}")

	print(f"PDF: {pdf_path}")
	print(f"PNG: {png_path}")


if __name__ == "__main__":
	main()
