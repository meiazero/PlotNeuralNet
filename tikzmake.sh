#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <arquivo_sem_ext> [--no-open]" >&2
    exit 1
fi

TARGET="$1"
OPEN_PDF=true
if [[ "${2:-}" == "--no-open" ]]; then
    OPEN_PDF=false
fi

python3 "$TARGET.py"
pdflatex -interaction=nonstopmode -halt-on-error "$TARGET.tex"

rm -f "$TARGET.aux" "$TARGET.log" "$TARGET.vscodeLog" || true

if $OPEN_PDF; then
    if [[ "${OSTYPE:-}" == "darwin"* ]]; then
            open "$TARGET.pdf" || true
    else
            xdg-open "$TARGET.pdf" >/dev/null 2>&1 || true
    fi
fi
