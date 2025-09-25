"""Layer generation"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .compiler import FormatConverter, LaTeXCompiler
from .renderer import DiagramRenderer
from .templates import LaTeXTemplate, _layers_dir_path  # noqa: F401

logger = logging.getLogger(__name__)


def to_head_pkg() -> str:
    return LaTeXTemplate.document_header_external()


def to_head_inline() -> str:
    return LaTeXTemplate.document_header_inline()


def to_colors() -> str:
    return LaTeXTemplate.color_definitions()


def to_begin() -> str:
    return LaTeXTemplate.document_begin()


def to_end() -> str:
    return LaTeXTemplate.document_end()


def to_document(arch: list[str], inline_styles: bool = True, include_colors: bool = True) -> str:
    return LaTeXTemplate.full_document(
        arch, inline_styles=inline_styles, include_colors=include_colors
    )


def to_generate(
    arch: list[str],
    pathname: str = "file.tex",
    inline_styles: bool = True,
    include_colors: bool = True,
) -> None:
    DiagramRenderer().render_to_tex(
        arch, pathname, inline_styles=inline_styles, include_colors=include_colors
    )


def compile_tex_to_pdf(
    tex_content: str, out_pdf: str | Path, keep_tex: bool | str | Path = True
) -> Path:
    compiler = LaTeXCompiler()
    return compiler.compile_to_pdf(tex_content, out_pdf, keep_tex=keep_tex)


def pdf_to_format(
    pdf_path: Path, out_path: Path, format: str, dpi: int = 300, page: int = 1
) -> Path:
    converter = FormatConverter()
    return converter.pdf_to_format(pdf_path, out_path, format, dpi=dpi, page=page)


def to_input(
    pathfile: str,
    to: str = "(-3,0,0)",
    width: int = 8,
    height: int = 8,
    name: str = "temp",
    anchor_scale: float = 0.01,
) -> str:
    half_w = width * anchor_scale
    half_h = height / 2
    return (
        f"\\node[canvas is zy plane at x=0] ({name}) at {to} "
        f"{{\\includegraphics[width={width}cm,height={height}cm]{{{pathfile}}}}};"
        f"\\coordinate ({name}-east) at ($({name}.center)+({half_w}cm,0,0)$);"
        f"\\coordinate ({name}-west) at ($({name}.center)-({half_w}cm,0,0)$);"
        f"\\coordinate ({name}-north) at ($({name}.center)+(0,{half_h}cm,0)$);"
        f"\\coordinate ({name}-south) at ($({name}.center)-(0,{half_h}cm,0)$);"
    )


def to_connection(of: str, to: str) -> str:
    return f"\\draw [connection]  ({of}-east)    -- node {{\\midarrow}} ({to}-west);"


def to_conv(
    name: str,
    s_filer: int = 256,
    n_filer: int = 64,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 40,
    depth: int = 40,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{ {n_filer}, }}}},
        zlabel={s_filer},
        fill=\\ConvColor,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_conv_conv_relu(
    name: str,
    s_filer: int = 256,
    n_filer: tuple[int, int] = (64, 64),
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: tuple[int, int] = (2, 2),
    height: int = 40,
    depth: int = 40,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{ {offset} }}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{ {{ {n_filer[0]} }}, {{ {n_filer[1]} }} }},
        zlabel={s_filer},
        fill=\\ConvColor,
        bandfill=\\ConvReluColor,
        height={height},
        width={{ {width[0]} , {width[1]} }},
        depth={depth}
        }}
    }};"""


def to_pool(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 32,
    depth: int = 32,
    opacity: float = 0.5,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{ {offset} }}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        fill=\\PoolColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_unpool(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 32,
    depth: int = 32,
    opacity: float = 0.5,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{ {offset} }}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        fill=\\UnpoolColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_conv_res(
    name: str,
    s_filer: int = 256,
    n_filer: int = 64,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 6,
    height: int = 40,
    depth: int = 40,
    opacity: float = 0.2,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{ {offset} }}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{ {{ {n_filer} }}, }},
        zlabel={s_filer},
        fill={{rgb:white,1;black,3}},
        bandfill={{rgb:white,1;black,2}},
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_conv_softmax(
    name: str,
    s_filer: int = 40,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 40,
    depth: int = 40,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        zlabel={s_filer},
        fill=\\SoftmaxColor,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


# ---------------- Transformer specific helpers -----------------
def to_embedding(
    name: str,
    vocab_size: int = 30522,
    model_dim: int = 768,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 30,
    depth: int = 30,
    caption: str = "Embed",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{ {model_dim}, }}}},
        zlabel={vocab_size},
        fill=\\FcColor,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_positional_encoding(
    name: str,
    seq_len: int = 512,
    model_dim: int = 768,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 30,
    depth: int = 30,
    caption: str = "PosEnc",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{{{ {model_dim} }}, {{ PE }} }},
        zlabel={seq_len},
        fill=\\FcColor,
        bandfill=\\FcReluColor,
        height={height},
        width={{{width} , {width}}},
        depth={depth}
        }}
    }};"""


def to_multihead_attention(
    name: str,
    heads: int = 8,
    model_dim: int = 768,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 2,
    height: int = 28,
    depth: int = 28,
    caption: str = "MHA",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{{{ {model_dim} }}, {{ h={heads} }} }},
        zlabel={model_dim},
        fill=\\ConvColor,
        bandfill=\\ConvReluColor,
        height={height},
        width={{{width} , {width}}},
        depth={depth}
        }}
    }};"""


def to_feed_forward(
    name: str,
    model_dim: int = 768,
    hidden_dim: int = 3072,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 2,
    height: int = 26,
    depth: int = 26,
    caption: str = "FFN",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{{{ {model_dim} }}, {{ {hidden_dim} }} }},
        zlabel={model_dim},
        fill=\\FcColor,
        bandfill=\\FcReluColor,
        height={height},
        width={{{width} , {width}}},
        depth={depth}
        }}
    }};"""


def to_layer_norm(
    name: str,
    model_dim: int = 768,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 20,
    depth: int = 20,
    caption: str = "LN",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{ {model_dim}, }}}},
        zlabel={model_dim},
        fill=\\PoolColor,
        opacity=0.4,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_add(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    radius: float = 2.5,
    caption: str = "+",
) -> str:
    # Reaproveita estilo Ball, mas podemos alterar no futuro
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Ball={{
        name={name},
        fill=\\SumColor,
        opacity=0.6,
        radius={radius},
        logo=$+$
        }}
    }};"""


def to_output_projection(
    name: str,
    vocab_size: int = 30522,
    model_dim: int = 768,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 28,
    depth: int = 28,
    caption: str = "Proj",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{ {vocab_size}, }}}},
        zlabel={model_dim},
        fill=\\SoftmaxColor,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_softmax(
    name: str,
    s_filer: int = 10,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 2,
    height: int = 3,
    depth: int = 25,
    opacity: float = 0.8,
    caption: str = " ",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{ " " ,"dummy" }},
        zlabel={s_filer},
        fill=\\SoftmaxColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_sum(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    radius: float = 2.5,
    opacity: float = 0.6,
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Ball={{
        name={name},
        fill=\\SumColor,
        opacity={opacity},
        radius={radius},
        logo=$+$
        }}
    }};"""


# --------- Generic / Extended operations for broader model coverage ---------
def to_activation(
    name: str,
    act: str = "ReLU",
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 18,
    depth: int = 18,
    caption: str | None = None,
) -> str:
    caption = caption or act
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        fill=\\ActivationColor,
        opacity=0.6,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_normalization(
    name: str,
    kind: str = "BN",
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 18,
    depth: int = 18,
    caption: str | None = None,
) -> str:
    caption = caption or kind
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        fill=\\NormColor,
        opacity=0.45,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_rnn_cell(
    name: str,
    cell: str = "LSTM",
    hidden_size: int = 512,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 2,
    height: int = 26,
    depth: int = 26,
    caption: str | None = None,
) -> str:
    caption = caption or f"{cell}"
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{{{ {hidden_size} }}, {{ {cell} }} }},
        zlabel={hidden_size},
        fill=\\RNNColor,
        bandfill=\\FcReluColor,
        opacity=0.85,
        height={height},
        width={{ {width} , {width} }},
        depth={depth}
        }}
    }};"""


def to_generic_box(
    name: str,
    label_left: str = " ",
    label_right: str = " ",
    zlabel: str | int = " ",
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int | tuple[int, int] = 1,
    height: int = 20,
    depth: int = 20,
    caption: str = " ",
    fill: str = "\\GenericColor",
    opacity: float = 0.35,
) -> str:
    if isinstance(width, tuple):
        width_tex = f"{{ {width[0]} , {width[1]} }}"
        pic_type = "RightBandedBox"
        extra = "bandfill=\\FcReluColor,"
    else:
        width_tex = str(width)
        pic_type = "Box"
        extra = ""
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{{pic_type}={{
        name={name},
        caption={caption},
        xlabel={{{{ {label_left} }}, {{ {label_right} }} }},
        zlabel={zlabel},
        fill={fill},
        {extra}
        opacity={opacity},
        height={height},
        width={width_tex},
        depth={depth}
        }}
    }};"""


# -------- New extended layer primitives ---------
def to_depthwise_conv(
    name: str,
    channels: int,
    kernel: str = "3x3",
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 30,
    depth: int = 30,
    caption: str = "DW",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{ {channels}, }} }},
        zlabel={kernel},
        fill=\\DepthwiseColor,
        opacity=0.85,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_separable_conv(
    name: str,
    in_channels: int,
    out_channels: int,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: tuple[int, int] = (1, 1),
    height: int = 30,
    depth: int = 30,
    caption: str = "SepConv",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{{{ {in_channels} }}, {{ {out_channels} }} }},
        zlabel=DW+PW,
        fill=\\SepConvColor,
        bandfill=\\DepthwiseColor,
        opacity=0.9,
        height={height},
        width={{{width[0]} , {width[1]}}},
        depth={depth}
        }}
    }};"""


def to_transpose_conv(
    name: str,
    s_filer: int = 256,
    n_filer: int = 64,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 2,
    height: int = 30,
    depth: int = 30,
    caption: str = "DeConv",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{ {n_filer}, }}}},
        zlabel={s_filer},
        fill=\\TransposeConvColor,
        opacity=0.85,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_flatten(
    name: str,
    features: int,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 1,
    height: int = 12,
    depth: int = 12,
    caption: str = "Flatten",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{ {features}, }}}},
        zlabel=feat,
        fill=\\FlattenColor,
        opacity=0.7,
        height={height},
        width={width},
        depth={depth}
        }}
    }};"""


def to_squeeze_excitation(
    name: str,
    channels: int,
    se_ratio: float = 0.25,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: int = 2,
    height: int = 18,
    depth: int = 18,
    caption: str = "SE",
) -> str:
    hidden = int(channels * se_ratio)
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{{{ {channels} }}, {{ {hidden} }} }},
        zlabel=SE,
        fill=\\SEColor,
        bandfill=\\ActivationColor,
        opacity=0.8,
        height={height},
        width={{{width} , {width}}},
        depth={depth}
        }}
    }};"""


def to_transformer_block(
    name: str,
    model_dim: int = 768,
    heads: int = 8,
    mlp_dim: int = 3072,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: tuple[int, int] = (2, 2),
    height: int = 34,
    depth: int = 34,
    caption: str = "Block",
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{{{ h={heads} }}, {{ {mlp_dim} }} }},
        zlabel={model_dim},
        fill=\\TransformerBlockColor,
        bandfill=\\FcReluColor,
        opacity=0.85,
        height={height},
        width={{{width[0]} , {width[1]}}},
        depth={depth}
        }}
    }};"""


def to_concat(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    radius: float = 2.2,
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Ball={{
        name={name},
        fill=\\OpColor,
        opacity=0.55,
        radius={radius},
        logo=$\\parallel$
        }}
    }};"""


def to_split(
    name: str,
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    radius: float = 2.2,
) -> str:
    return f"""\\pic[shift={{{offset}}}] at {to}
    {{Ball={{
        name={name},
        fill=\\OpColor,
        opacity=0.55,
        radius={radius},
        logo=$\\bowtie$
        }}
    }};"""


def to_skip(of: str, to: str, pos: float = 1.25) -> str:
    return f"""\\path ({of}-southeast) -- ({of}-northeast) coordinate[pos={pos}] ({of}-top) ;
\\path ({to}-south)  -- ({to}-north)  coordinate[pos={pos}] ({to}-top) ;
\\draw [copyconnection]  ({of}-northeast)
-- node {{\\copymidarrow}}({of}-top)
-- node {{\\copymidarrow}}({to}-top)
-- node {{\\copymidarrow}} ({to}-north);"""


def generate_pdf(
    arch: list[str],
    out_pdf: str | Path,
    inline_styles: bool = True,
    include_colors: bool = True,
    keep_tex: bool | str | Path = True,
) -> Path:
    doc = to_document(arch, inline_styles=inline_styles, include_colors=include_colors)
    return compile_tex_to_pdf(doc, out_pdf, keep_tex=keep_tex)


def generate_png(
    arch: list[str],
    out_png: str | Path,
    dpi: int = 300,
    inline_styles: bool = True,
    include_colors: bool = True,
    keep_tex: bool | str | Path = True,
) -> Path:
    out_png_path = Path(out_png).resolve()
    if keep_tex is False:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_tmp = Path(tmpdir) / "temp.pdf"
            generate_pdf(
                arch,
                pdf_tmp,
                inline_styles=inline_styles,
                include_colors=include_colors,
                keep_tex=False,
            )
            return pdf_to_format(pdf_tmp, out_png_path, "png", dpi=dpi)
    pdf_path = out_png_path.with_suffix(".pdf")
    generate_pdf(
        arch,
        pdf_path,
        inline_styles=inline_styles,
        include_colors=include_colors,
        keep_tex=keep_tex,
    )
    return pdf_to_format(pdf_path, out_png_path, "png", dpi=dpi)


def generate_svg(
    arch: list[str],
    out_svg: str | Path,
    inline_styles: bool = True,
    include_colors: bool = True,
    keep_tex: bool | str | Path = True,
) -> Path:
    out_svg_path = Path(out_svg).resolve()
    if keep_tex is False:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_tmp = Path(tmpdir) / "temp.pdf"
            generate_pdf(
                arch,
                pdf_tmp,
                inline_styles=inline_styles,
                include_colors=include_colors,
                keep_tex=False,
            )
            return pdf_to_format(pdf_tmp, out_svg_path, "svg")
    pdf_path = out_svg_path.with_suffix(".pdf")
    generate_pdf(
        arch,
        pdf_path,
        inline_styles=inline_styles,
        include_colors=include_colors,
        keep_tex=keep_tex,
    )
    return pdf_to_format(pdf_path, out_svg_path, "svg")
