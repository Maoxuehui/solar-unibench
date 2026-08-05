"""Dev vs full-config presets for the experiment grid (Phase 8 Stream B).

The experiments (E8 benchmark, E9 governance) ship with **TINY dev configs**
(small models, ``epochs=3``, a per-entity row cap, 2 horizons) so the suite runs
in seconds on bounded data and the test-suite stays fast. The *publishable*
Phase-8 numbers need the opposite: full-capacity models trained to convergence
on the FULL standardized data across the FULL 3-horizon grid.

Activation
----------
Set the ``SUB_FULL_CONFIG`` environment variable to any truthy value (e.g.
``SUB_FULL_CONFIG=1``) BEFORE importing the experiment modules. The orchestrator
``scripts/run_experiments.py --full`` does this for you. When unset, every
preset below resolves to the exact TINY dev values the tests expect — default
behavior is bit-identical to pre-Phase-8.

Presets are tuned for the local 2x RTX 3090 (24 GiB): deep tabular models on
``cuda`` at large width + full epochs; CrossViViT scaled toward the full ~145M
multimodal model (vs the Phase-6b small runnable variant). ``n_jobs=-1`` for
LightGBM uses all cores. Adjust via env vars if a different compute budget is
needed.
"""
from __future__ import annotations

import os

#: Full-config master switch. Truthy -> publishable presets; falsy -> TINY dev.
FULL: bool = bool(os.environ.get("SUB_FULL_CONFIG"))

#: Per-entity row cap (temporal dev slice). Dev caps to 2000 so deep models fit
#: in seconds; FULL uses 50,000 — a 25x-larger, tractable slice that bounds each
#: grid cell to minutes (vs uncapped Folsom 1.5M rows = 10h+ for deep models).
#: 0 means truly uncapped; raise toward 0 if full-data rigor is later required.
#: NOTE for Folsom: rows are temporal, so the first 50k are early-2014 — the
#: image-bearing window, so CrossViViT trains on real sky images, not zeros.
MAX_ROWS_PER_ENTITY: int = 50_000 if FULL else 2000

#: Forecast-horizon grid (hours). Dev = 2-horizon intra-day; full adds day-ahead.
HORIZONS_HOURS: tuple[int, ...] = (1, 6, 24) if FULL else (1, 6)

#: E9 representative-dataset grid. Dev = GEFCom only (fast, PV, tabular);
#: full = the full image+tabular set so the governance thesis is shown across
#: datasets (not just GEFCom).
E9_DATASETS: tuple[str, ...] = ("gefcom2014",) if not FULL else (
    "gefcom2014", "skippd", "folsom", "sunlake",
)

# ---------------------------------------------------------------------------
# Model parameter presets
# ---------------------------------------------------------------------------

if FULL:  # publishable Phase-8 configs (GPU, full capacity, to-convergence)
    LGBM_PARAMS: dict = {"n_estimators": 500, "n_jobs": -1, "random_state": 0}
    MLP_PARAMS: dict = {
        "hidden_layer_sizes": (256, 128), "max_iter": 500, "random_state": 0,
    }
    LSTM_PARAMS: dict = {
        "hidden_size": 128, "num_layers": 2, "epochs": 50, "batch_size": 256,
        "lr": 1e-3, "device": "cuda",
    }
    PATCHTST_PARAMS: dict = {
        "d_model": 128, "n_heads": 8, "patch_len": 4, "stride": 2, "depth": 3,
        "epochs": 50, "batch_size": 256, "lr": 1e-4, "device": "cuda",
    }
    # PatchTST channel-DEPENDENT variant (cross-channel mixing in the patch
    # projection) — the CD counterpart run alongside the default CI PatchTST to
    # separate temporal-pattern from variate-interaction modelling (patch_len 4
    # at lookback 12/24 captures high-frequency cloud motion per the ESSD ask).
    PATCHTST_CD_PARAMS: dict = {
        "d_model": 128, "n_heads": 8, "patch_len": 4, "stride": 2, "depth": 3,
        "epochs": 50, "batch_size": 256, "lr": 1e-4, "device": "cuda",
        "variant": "patchtst_cd",
    }
    DLINEAR_PARAMS: dict = {
        "kernel_size": 25, "epochs": 50, "batch_size": 256, "lr": 1e-3, "device": "cuda",
    }
    iTRANSFORMER_PARAMS: dict = {
        "d_model": 128, "n_heads": 8, "depth": 3, "epochs": 50, "batch_size": 256,
        "lr": 1e-3, "device": "cuda",
    }
    # CrossViViT scaled toward the full ~145M multimodal model (Phase-6b ran a
    # small d_model=64/depth=2 runnable variant). lr=1e-4 (the value used for
    # the reported Phase-8 results; lower than the model default 1e-3, which we
    # found overfits/diverges on the smaller per-entity slices here).
    CROSSVIVIT_PARAMS: dict = {
        "d_model": 384, "n_heads": 8, "depth": 8, "patch_len": 4,
        "img_size": 64, "img_patch": 8, "epochs": 50, "batch_size": 64,
        "lr": 1e-4, "dropout": 0.1, "device": "cuda",
    }
else:  # TINY dev configs (mirror the pre-Phase-8 constants; fast tests)
    LGBM_PARAMS = {"n_estimators": 50, "n_jobs": 1, "random_state": 0}
    MLP_PARAMS = {"hidden_layer_sizes": (32,), "max_iter": 80, "random_state": 0}
    LSTM_PARAMS = {
        "hidden_size": 16, "num_layers": 1, "epochs": 3, "batch_size": 64,
        "device": "cpu",
    }
    PATCHTST_PARAMS = {
        "d_model": 32, "n_heads": 4, "patch_len": 4, "stride": 2, "depth": 1,
        "epochs": 3, "batch_size": 64, "device": "cpu",
    }
    PATCHTST_CD_PARAMS = {
        "d_model": 32, "n_heads": 4, "patch_len": 4, "stride": 2, "depth": 1,
        "epochs": 3, "batch_size": 64, "device": "cpu", "variant": "patchtst_cd",
    }
    DLINEAR_PARAMS = {"kernel_size": 25, "epochs": 3, "batch_size": 64, "device": "cpu"}
    iTRANSFORMER_PARAMS = {
        "d_model": 32, "n_heads": 4, "depth": 1, "epochs": 3, "batch_size": 64,
        "device": "cpu",
    }
    CROSSVIVIT_PARAMS = {
        "d_model": 32, "n_heads": 4, "depth": 1, "patch_len": 4, "img_size": 32,
        "img_patch": 4, "epochs": 3, "batch_size": 32, "device": "cpu",
    }


def describe() -> dict:
    """Return the active config as a serializable dict (for result provenance)."""
    return {
        "full": FULL,
        "max_rows_per_entity": MAX_ROWS_PER_ENTITY,
        "horizons_hours": list(HORIZONS_HOURS),
        "e9_datasets": list(E9_DATASETS),
        "models": {
            "LightGBM": LGBM_PARAMS,
            "MLP": MLP_PARAMS,
            "LSTM": LSTM_PARAMS,
            "PatchTST": PATCHTST_PARAMS,
            "PatchTST-CD": PATCHTST_CD_PARAMS,
            "DLinear": DLINEAR_PARAMS,
            "iTransformer": iTRANSFORMER_PARAMS,
            "CrossViViT": CROSSVIVIT_PARAMS,
        },
    }
