"""SolarCurationAgent — agentic curation workflow over the pipeline tools.

The tool layer lives in :mod:`solarunibench.pipeline.agent_tools` (real
physics operators with agent-readable contracts). This package provides a
framework-agnostic reference agent: a deterministic keyword planner with a
ReAct-style trace. Swap in LangChain / AutoGEN by feeding the same tools
through their tool-use adapters, e.g.::

    from langchain.tools import StructuredTool
    from solarunibench.pipeline.agent_tools import (
        tool_recover_geolocation, tool_apply_physical_qc,
        tool_extract_physics_features)
    tools = [StructuredTool.from_function(t) for t in
             (tool_recover_geolocation, tool_apply_physical_qc,
              tool_extract_physics_features)]
"""
from .agent import SolarCurationAgent, export_canonical

__all__ = ["SolarCurationAgent", "export_canonical"]
