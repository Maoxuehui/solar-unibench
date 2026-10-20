"""Benchmark model suite — the paper's 8 models (+1 variant).

Paper-facing seams over the canonical implementations in
``solarunibench.forecast.models`` (and ``benchmark.crossvivit_image``):
  baselines.SmartPersistenceModel   Smart Persistence reference
  ml_models.LightGBMModel/MLPModel  micro models
  deep_models.DLinearModel / LSTMModel / iTransformerModel /
      PatchTSTModel / CrossViViTModel
"""
from .baselines import SmartPersistenceModel
from .ml_models import LightGBMModel, MLPModel

__all__ = ["SmartPersistenceModel", "LightGBMModel", "MLPModel"]
