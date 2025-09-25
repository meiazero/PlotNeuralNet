PY=poetry run python
RUFF=poetry run ruff
BLACK=poetry run black

.PHONY: lint format build example-simple example-unet

lint:
	$(RUFF) check .

format:
	$(BLACK) .
	$(RUFF) check --fix .

build:
	poetry build -n

# example-simple:
# 	cd examples && $(PY) test_simple.py && pdflatex -interaction=nonstopmode -halt-on-error test_simple.tex

# example-unet:
# 	cd examples && $(PY) unet.py && pdflatex -interaction=nonstopmode -halt-on-error unet.tex
