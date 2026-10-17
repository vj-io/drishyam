PY ?= .venv/bin/python
ARM ?= local
DATA ?= data/synth200
N ?= 200
OUT ?= public/results
RAW ?= results
CODEBRIM ?= data/codebrim
EVAL = $(PY) -m evals.harness --cost-model fixed

.PHONY: test e2e eval eval-real eval-large eval-dev eval-live eval-model bench data serve lint

test:  ## unit, API, MCP, CLI and eval tests (same as CI)
	$(PY) -m pytest -q --ignore=tests/e2e

e2e:  ## browser tests under the production CSP; needs Playwright (see README)
	$(PY) tests/e2e/test_ui_mock.py && $(PY) tests/e2e/test_ui_live.py

eval:  ## 200-image synthetic evaluation (seed 7) -> public/results, with per_image.json.gz (a few minutes)
	$(EVAL) --data $(DATA) --n $(N) --seed 7 --out $(OUT) --results-dir $(RAW)

eval-real:  ## CODEBRIM eval (run `make data` first); its licence bars sharing the images: only summary.json and failures_text.json are published
	$(EVAL) --data $(CODEBRIM) --n 200 --out results/codebrim --results-dir results/codebrim
	mkdir -p public/results/codebrim
	cp results/codebrim/summary.json results/codebrim/failures_text.json public/results/codebrim/

eval-large:  ## 100 synthetic images at CODEBRIM's 4608x3456 (seed 7): does zooming beat one downscale as images grow?
	$(EVAL) --data data/synth_large --n 100 --seed 7 --size 4608x3456 --out runs/eval-large --results-dir runs/eval-large
	mkdir -p public/results/large && cp runs/eval-large/summary.json public/results/large/summary.json

eval-dev:  ## the 60-image dev split (seed 4242) used while tuning; never published
	$(EVAL) --data data/synth_dev --n 60 --seed 4242 --out runs/eval-dev --results-dir runs/eval-dev

eval-live:  ## the synthetic eval on measured wall time instead (load changes the agent's choices) -> public/results/live
	$(PY) -m evals.harness --cost-model live --data $(DATA) --n $(N) --seed 7 --out runs/eval-live --results-dir runs/eval-live
	mkdir -p public/results/live && cp runs/eval-live/summary.json public/results/live/summary.json

eval-model:  ## after scripts/install_model.sh: leakage check, then the ONNX on CODEBRIM and synth200 vs the classical runs
	$(PY) -m evals.model_eval --onnx $(or $(ONNX),models/drishyam_dacl10k.onnx) --codebrim $(CODEBRIM) $(if $(DACL10K),--dacl10k $(DACL10K))

bench:  ## micro + macro + parity for one arm: make bench ARM=local
	bash bench/run_all.sh $(ARM)

data:  ## CODEBRIM download (8.3 GB); asks for confirmation first
	bash scripts/get_data.sh

serve:
	DRISHYAM_SERVE_STATIC=1 .venv/bin/uvicorn drishyam.web:app --port 8000

lint:
	.venv/bin/ruff check .
