"""Curation Agent execution entry / Skill definition (reference agent).

Implements the SKILL.md execution order — (1) geolocation if metadata is
missing, (2) physical-bounds QC, (3) visual physics features, (4) canonical
export — with a deterministic keyword planner and a ReAct-style trace.
Replace the planner with an LLM (LangChain / AutoGEN) while keeping the
very same tools; see the adapter snippet in ``agent/__init__.py``.
"""
from solarunibench.agent.agent import (  # noqa: F401  (canonical home)
    STEPS,
    SolarCurationAgent,
    export_canonical,
)

__all__ = ["STEPS", "SolarCurationAgent", "export_canonical",
           "main"]


def main(argv: list[str] | None = None) -> int:
    """CLI shim mirroring ``scripts/run_agent_curation.py``."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cmd = [sys.executable,
           str(repo / "scripts" / "run_agent_curation.py")]
    if argv:
        cmd += argv
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
