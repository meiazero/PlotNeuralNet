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

example:
	$(PY) src/example.py
