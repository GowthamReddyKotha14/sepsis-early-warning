# Each step writes its output; run them in order the first time.
.PHONY: install data audit baseline test

install:          ## Python packages (use a virtual environment)
	pip install -e ".[dev]"

data:             ## download PhysioNet 2019 (about 41 MB) -> data/hourly.parquet
	python -m sepsis data

audit:            ## cohort summary -> results/audit.json
	python -m sepsis audit

baseline:         ## SIRS / qSOFA scores -> results/baseline.json
	python -m sepsis baseline

test:
	pytest -q
