# Solar-UniBench Curation Agent Skill

## Description

An autonomous agentic skill for physics-guided solar data curation and
standardization. Given raw, uncurated solar telemetry and (optionally) sky
or satellite imagery, the agent harmonizes it into the analysis-ready
Solar-UniBench contract by dispatching the tools below in the canonical
order. All tools wrap the same physics operators the batch pipeline uses
(`solarunibench/pipeline/agent_tools.py`) — the agent decides WHEN, the
operators decide HOW.

## Tools

| Tool | Equation | Purpose |
|---|---|---|
| `tool_recover_geolocation` | Eq. (1) | Invert (lat, lon, tz) from the clear-sky shape of a target series. |
| `tool_apply_physical_qc` | Eq. (2)–(3) | Nocturnal zeroing, k_t ∈ [0, 1.5] clamp, quality flags, PV capacity clip. |
| `tool_extract_physics_features` | Eq. (4)–(5) | NRBR color-ratio map and optical-flow reliability from image stacks. |

## Execution Steps

1. Check if geospatial metadata (lat, lon, timezone) exists. If missing,
   call `tool_recover_geolocation`.
2. Audit physical limits: set nocturnal values to 0 and clamp clearness
   index k_t ∈ [0, 1.5] via `tool_apply_physical_qc`.
3. If a visual modality exists (ASI / satellite), compute NRBR and
   optical-flow reliability maps via `tool_extract_physics_features`.
4. Export to the canonical Parquet / Zarr contract.

## Entry Point

```bash
python scripts/run_agent_curation.py                      # sample demo
python scripts/run_agent_curation.py --instruction "recover the site coordinates only"
```

The reference agent (`solarunibench.agent.SolarCurationAgent`) is
framework-agnostic; to drive the same tools from LangChain / AutoGEN,
wrap them with the platform's tool adapter (see the snippet in
`solarunibench/agent/__init__.py`).
