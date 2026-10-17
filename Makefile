PY ?= .venv/bin/python
ARM ?= local
DATA ?= data/synth200

.PHONY: test e2e eval bench data serve lint

test:  ## unit, API, MCP, CLI and eval tests (same as CI)
	$(PY) -m pytest -q --ignore=tests/e2e

e2e:  ## browser tests under the production CSP; needs Playwright (see README)
	$(PY) tests/e2e/test_ui_mock.py && $(PY) tests/e2e/test_ui_live.py

eval:  ## 200-image evaluation -> public/results (a few minutes); real data: make eval DATA=data/codebrim
	$(PY) -m evals.harness --data $(DATA) --n 200 --out public/results

bench:  ## micro + macro + parity for one arm: make bench ARM=local
	bash bench/run_all.sh $(ARM)

data:  ## CODEBRIM download (8.3 GB); asks for confirmation first
	bash scripts/get_data.sh

serve:
	DRISHYAM_SERVE_STATIC=1 .venv/bin/uvicorn drishyam.web:app --port 8000

lint:
	.venv/bin/ruff check .
