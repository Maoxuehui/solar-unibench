"""Curation pipeline module — the paper's data-governance layer.

Couples each numbered equation to its implementation module:

  Eq. (1)     :mod:`alignment`          spatial & timezone inversion
  Eq. (2)-(3) :mod:`quality_audit`      physical bounds & masking
  Eq. (4)-(5) :mod:`feature_extraction` NRBR & optical-flow reliability
"""
from .alignment import recover_geolocation, recover_tsr_alignment
from .feature_extraction import nrbr_map, optical_flow_reliability
from .quality_audit import apply_physical_bounds

__all__ = [
    "recover_geolocation",
    "recover_tsr_alignment",
    "apply_physical_bounds",
    "nrbr_map",
    "optical_flow_reliability",
]
