# Each step writes its output; run them in order the first time.
.PHONY: install data audit baseline train evaluate test

install:          ## Python packages (use a virtual environment)
	pip install -e ".[dev]"

data:             ## download PhysioNet 2019 (about 41 MB) -> data/hourly.parquet
	python -m sepsis data

audit:            ## cohort summary -> results/audit.json
	python -m sepsis audit

baseline:         ## SIRS / qSOFA scores -> results/baseline.json
	python -m sepsis baseline

train:            ## features + model on hospital A -> models/sepsis_hgb.joblib (a few minutes)
	python -m sepsis train

evaluate:         ## choose alarm level on validation, score hospital B once -> results/model.json
	python -m sepsis evaluate

test:
	pytest -q
