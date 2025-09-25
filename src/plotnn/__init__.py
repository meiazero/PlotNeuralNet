"""plotnn: utilitários para gerar código LaTeX/TikZ de arquiteturas de redes neurais.

API pública principal:
- from plotnn.layers import *
- from plotnn.blocks import *
"""

# Reexporta símbolos principais para ergonomia
try:
    from .blocks import *  # type: ignore # noqa: F401,F403
    from .layers import *  # type: ignore # noqa: F401,F403
except Exception:
    # Em ambientes de análise estática ou durante instalação, os módulos podem não estar resolvidos ainda.
    pass

# Não definimos __all__ explicitamente para manter compatibilidade com import *
