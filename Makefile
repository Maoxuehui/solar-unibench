# Solar-UniBench — reproducibility entrypoints.
#
# One-command Phase-8 reproduction. Conda env `solar_uni` is assumed
# (`D:\anaconda_data\envs\solar_uni` on the reference machine); override PY to
# point at its python if your layout differs. Heavy outputs go to D: (C: is
# space-constrained on the reference machine).
#
# Full-config runs are GPU + multi-hour; the dev (fast) path is the default for
# `benchmark` so `make benchmark` is quick, and `make benchmark-full` produces
# the publication set.
PY ?= python

.PHONY: help materialize-folsom materialize-skippd benchmark benchmark-full figures test lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-22s %s\n", $$1, $$2}'

materialize-folsom:  ## Rebuild the Folsom 2014 sky-image zarr (250k frames, ~15-30 min)
	$(PY) scripts/rebuild_sky_zarr.py --dataset folsom --only-year 2014 --limit 0

materialize-skippd:  ## Rebuild the full SKIPPD sky-image zarr (~363k frames)
	$(PY) scripts/rebuild_sky_zarr.py --dataset skippd --limit 0

benchmark:  ## Dev-config experiments (fast, bounded) -> figures/
	$(PY) scripts/run_experiments.py --out figures

benchmark-full:  ## Phase-8 full-config publication set -> figures_full/ (GPU, multi-hour)
# Tip: run E8/E9 first then the rest; RUN_SLOW=1 enables E9's real-data raw arm.
	$(PY) scripts/run_experiments.py --full --out figures_full
	RUN_SLOW=1 $(PY) scripts/run_experiments.py --full --only E9 --out figures_full

figures:  ## Package the ~7 paper figures (PDF + 300dpi PNG) from figures_full/
	$(PY) scripts/select_figures.py --src figures_full --out figures_full/paper

test:  ## Fast test suite (excludes RUN_SLOW)
	$(PY) -m pytest tests/ -q -W ignore

clean:  ## Remove generated figures/results caches (keeps standardized data)
	rm -rf figures/ figures_full/ .raw_cache/
