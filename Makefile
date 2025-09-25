PY=poetry run python
RUFF=poetry run ruff
BLACK=poetry run black
OUTDIR=dist

.PHONY: lint format build

lint:
	$(RUFF) check . --fix

format:
	$(BLACK) .
	$(RUFF) check --fix .

build:
	rm -rf dist && poetry build -n --output=$(OUTDIR)

# example-simple:
# 	cd examples && $(PY) test_simple.py && pdflatex -interaction=nonstopmode -halt-on-error test_simple.tex

# example-unet:
# 	cd examples && $(PY) unet.py && pdflatex -interaction=nonstopmode -halt-on-error unet.tex
