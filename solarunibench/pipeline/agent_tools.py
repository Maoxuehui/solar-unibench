"""DEPRECATED canonical location — moved to ``solarunibench/agent/tools.py``
(physics implementations now live in ``solarunibench/curation``).
This shim keeps old imports working."""
from ..agent.tools import (  # noqa: F401
    tool_apply_physical_qc,
    tool_extract_physics_features,
    tool_recover_geolocation,
)
