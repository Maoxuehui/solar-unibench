"""Forecast baseline models.

* :class:`SmartPersistenceModel` — clear-sky-scaled physical skill reference
  (spec §10). Target-agnostic: works for PV power and GHI alike.
* :class:`LightGBMModel` — per-horizon LightGBM tabular baseline
  (direct multi-step regression on flattened past + future features).
* :class:`MLPModel` — single multi-output sklearn ``MLPRegressor`` baseline
  (configurable; ``skippd_paper`` preset reproduces Nie et al. 2020).
* :class:`LSTMModel` — stacked-LSTM sequence baseline (direct multi-step head;
  consumes ``X_past[n, lb, Fp]`` as a recurrent sequence).
* :class:`PatchTSTModel` — PatchTST Transformer baseline (patches ``X_past``
  along the lookback axis; ``variant='patchtst'`` channel-independent,
  ``variant='patchtst_cd'`` channel-dependent, ``variant='vanilla'`` a plain
  TS-Transformer — Nie et al. ICLR 2023).
* :class:`DLinearModel` — trend/seasonal decomposition + two channel-shared
  linear heads (Zeng et al. AAAI 2023 — the Transformer antagonist).
* :class:`iTransformerModel` — inverted Transformer: attention across variable
  channels (Liu et al. ICLR 2024 — the complement to PatchTST's time attention).
* :class:`CrossViViTModel` — multimodal tabular ↔ sky-image cross-attention
  baseline (Boussif et al., NeurIPS 2023; small runnable variant). Patches
  ``X_past`` for tabular tokens and conv-embeds the per-window sky-image cube
  for image tokens, fusing via bidirectional cross-attention; degrades to a
  tabular self-attention path when ``images`` is ``None``.
"""
from .crossvivit_model import CrossViViTModel
from .dlinear_model import DLinearModel
from .itransformer_model import iTransformerModel
from .lightgbm_model import LightGBMModel
from .lstm_model import LSTMModel
from .mlp_model import MLPModel
from .patchtst_model import PatchTSTModel
from .smart_persistence import SmartPersistenceModel

__all__ = [
    "SmartPersistenceModel",
    "LightGBMModel",
    "MLPModel",
    "LSTMModel",
    "PatchTSTModel",
    "DLinearModel",
    "iTransformerModel",
    "CrossViViTModel",
]
