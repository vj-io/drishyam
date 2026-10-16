PY ?= .venv/bin/python
ARM ?= local

.PHONY: test eval bench data serve lint

test:
	$(PY) -m pytest -q

eval:  ## 200-image evaluation -> public/results (about 10-15 minutes)
	$(PY) -m evals.harness --data data/synth200 --n 200 --out public/results

bench:  ## micro + macro + parity for one arm: make bench ARM=local
	bash bench/run_all.sh $(ARM)

data:  ## CODEBRIM download (8.3 GB); asks for confirmation first
	bash scripts/get_data.sh

serve:
	DRISHYAM_SERVE_STATIC=1 .venv/bin/uvicorn drishyam.web:app --port 8000

lint:
	.venv/bin/ruff check .
