"""Reference ReAct-style curation agent (framework-agnostic).

Reads SKILL.md's execution order: (1) geolocation if metadata is missing,
(2) physical-bounds QC, (3) visual physics features, (4) canonical export.
The planner is a deterministic keyword matcher over the instruction — enough
for reproducible demos and CI, and trivially replaceable by an LLM planner
driving the very same tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .tools import (
    tool_apply_physical_qc,
    tool_extract_physics_features,
    tool_recover_geolocation,
)

#: canonical step order from SKILL.md
STEPS: tuple[tuple[str, str, Callable], ...] = (
    ("geolocate", "tool_recover_geolocation", tool_recover_geolocation),
    ("physical_qc", "tool_apply_physical_qc", tool_apply_physical_qc),
    ("visual_features", "tool_extract_physics_features",
     tool_extract_physics_features),
)

_TRIGGERS = {
    "geolocate": ("geoloc", "metadata", "lat", "lon", "location", "site",
                  "where"),
    "physical_qc": ("bound", "clean", "qc", "quality", "curat", "kt",
                    "clamp", "night"),
    "visual_features": ("nrbr", "flow", "visual", "feature", "image",
                        "camera", "satellite", "sky"),
}


class SolarCurationAgent:
    """Dispatches curation tools per a natural-language instruction.

    Parameters
    ----------
    tools:
        Optional mapping ``{step_name: callable}`` overriding the default
        registry (useful for mocks in tests or framework adapters).
    verbose:
        Emit the ReAct-style trace (Thought/Action/Observation) to stdout.
    """

    def __init__(self, tools: dict[str, Callable] | None = None,
                 verbose: bool = True):
        self.tools = {name: fn for name, _doc, fn in STEPS}
        if tools:
            self.tools.update(tools)
        self.verbose = verbose

    # -- planning ---------------------------------------------------------
    def plan(self, instruction: str) -> list[str]:
        """Keyword-plan which canonical steps the instruction asks for.

        An instruction naming no tool runs the full pipeline (SKILL.md
        default); an instruction that names any tool runs ONLY those.
        """
        low = instruction.lower()
        hits = [name for name, words in _TRIGGERS.items()
                if any(w in low for w in words)]
        return hits or [name for name, _, _ in STEPS]

    # -- execution ---------------------------------------------------------
    def run(self, instruction: str, context: dict) -> dict:
        """Execute the planned steps against ``context``.

        ``context`` carries the data each tool needs (any subset):
          ``times``      UTC DatetimeIndex        -> geolocate
          ``power``      1-D target series        -> geolocate
          ``df_ts``      standardized-shaped DF   -> physical_qc
          ``image_tensor`` (T,H,W,3) RGB stack    -> visual_features

        Returns ``{"plan": [...], "observations": {step: payload}}``; the
        physical_qc observation is the governed DataFrame, so callers can
        export it to the canonical Parquet/Zarr contract (step 4).
        """
        plan = self.plan(instruction)
        report: dict = {"instruction": instruction, "plan": plan,
                        "observations": {}}
        if self.verbose:
            print(f"Thought: instruction={instruction!r}\n"
                  f"Thought: plan={plan}")
        for name in plan:
            fn = self.tools[name]
            if self.verbose:
                print(f"Action: {fn.__name__}")
            if name == "geolocate":
                if "times" not in context or "power" not in context:
                    obs = {"skipped": "context lacks times/power"}
                else:
                    obs = fn(context["times"], context["power"])
            elif name == "physical_qc":
                if "df_ts" not in context:
                    obs = {"skipped": "context lacks df_ts"}
                else:
                    obs = fn(context["df_ts"],
                             capacity=context.get("capacity", 1.0))
            else:  # visual_features
                if "image_tensor" not in context:
                    obs = {"skipped": "context lacks image_tensor"}
                else:
                    obs = fn(context["image_tensor"])
            report["observations"][name] = obs
            if self.verbose:
                summary = obs if not hasattr(obs, "shape") else \
                    f"DataFrame {obs.shape}"
                print(f"Observation: {summary}\n")
        return report


def export_canonical(df, out_path: str | Path) -> Path:
    """Step 4 of SKILL.md: write the governed frame to the Parquet contract."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path
