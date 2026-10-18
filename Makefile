PY ?= .venv/bin/python
ARM ?= local
DATA ?= data/synth200
N ?= 200
OUT ?= public/results
RAW ?= results
CODEBRIM ?= data/codebrim
# MAX_LOAD=4: evals and benchmarks exit at start if the 1-min load average is at or above it (checked in the process)
MAX_LOAD ?=
EVAL = $(PY) -m evals.harness --cost-model fixed $(if $(MAX_LOAD),--max-load $(MAX_LOAD))

.PHONY: test e2e eval eval-real eval-large eval-dev eval-live eval-model eval-ablations eval-ablations-real eval-clean bench data serve lint

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

eval-live:  ## deployment behaviour: measured wall time and the governor on (load changes the agent's choices) -> public/results/live
	$(PY) -m evals.harness --cost-model live --governor $(if $(MAX_LOAD),--max-load $(MAX_LOAD)) --data $(DATA) --n $(N) --seed 7 --out runs/eval-live --results-dir runs/eval-live
	mkdir -p public/results/live && cp runs/eval-live/summary.json public/results/live/summary.json

eval-model:  ## after scripts/install_model.sh: leakage check, then the ONNX on CODEBRIM and synth200 vs the classical runs
	$(PY) -m evals.model_eval --publish --onnx $(or $(ONNX),models/drishyam_dacl10k.onnx) --codebrim $(CODEBRIM) $(if $(DACL10K),--dacl10k $(DACL10K))

eval-ablations:  ## FR-21 ablations (7 variants vs the full agent, caps 0.3 and 1.0, fixed costs) on synth200 and the 4608 set -> public/results/ablations.json
	$(EVAL) --ablation --data $(DATA) --n $(N) --seed 7 --out $(OUT) --results-dir runs/ablations/synth
	$(EVAL) --ablation --data data/synth_large --n 100 --seed 7 --size 4608x3456 --out $(OUT) --results-dir runs/ablations/large

eval-ablations-real:  ## the same ablations on CODEBRIM at the 0.3 cap only (aggregates only: no image is written)
	$(EVAL) --ablation --ablation-caps 0.3 --data $(CODEBRIM) --n 200 --out $(OUT) --results-dir runs/ablations/codebrim

eval-clean:  ## false alarms: 50 synthetic images with distractors and no damage, plus the demo's bridge_08 -> public/results/clean
	test -f data/synth_clean/annotations.json || $(PY) -m evals.synth dataset --clean --out data/synth_clean --n 50 --seed 11
	$(EVAL) --data data/synth_clean --n 51 --seed 11 --chaos 0,0,0 --out runs/eval-clean --results-dir runs/eval-clean
	mkdir -p public/results/clean && cp runs/eval-clean/summary.json public/results/clean/summary.json

bench:  ## micro + macro + parity for one arm: make bench ARM=local [MAX_LOAD=4]
	MAX_LOAD=$(MAX_LOAD) bash bench/run_all.sh $(ARM)

data:  ## CODEBRIM download (8.3 GB); asks for confirmation first
	bash scripts/get_data.sh

serve:
	DRISHYAM_SERVE_STATIC=1 .venv/bin/uvicorn drishyam.web:app --port 8000

lint:
	.venv/bin/ruff check .
