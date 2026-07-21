"""E1 — Geo-recovery accuracy + TSR validation + downstream forecasting value.

Spec §11 (E1). Three sub-analyses, three figures, one results CSV — all routed
through the shared :mod:`solarunibench.viz` design system:

1. **Blind test** (known-coord dataset, Folsom): run the clear-sky geo-recovery
   *blind* (the operator ignores the stored coords and uses only the target
   signal), then compare the recovered (lat, lon) to the TRUE coords stored in
   ``metadata``. Report the great-circle error (km) via the haversine formula.
   Folsom carries measured ``ghi_actual`` — a target the clear-sky method can
   actually match (unlike cloudy PV, where it is weak). If the recovery is poor
   on the chosen target, that IS the E1 finding and it is reported honestly,
   with TSR noted as the reliable independent method.

2. **TSR validation** (GEFCom): for each zone's recovered coords (already
   TSR-recovered → Australia, stored in metadata), compare the theoretical
   top-of-atmosphere daily irradiation (pure solar geometry from the recovered
   latitude) against the dataset's ``nwp_VAR178`` (TSR) field. Report R²/RMSE
   via :func:`solarunibench.operators.tsr_validation.validate_with_tsr`.

3. **Downstream value** (GEFCom): a TINY MLP via the forecast framework, target
   ``pv_power_normalized`` — (a) raw PV lags only vs (b) + the features unlocked
   by recovered coords (``ghi_clearsky``, ``cos_zenith``, ``clearness_index``).
   Report Δ Daytime_MAE. Full-config numbers come at Phase 8.

Pure function contract: ``run(loader, out_dir) -> {figures, metrics}`` reads the
loader and writes to ``out_dir``; it never mutates standardized data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..forecast.core import EndogenousFeatureEngineer, PipelineConfig, SolarForecastPipeline
from ..forecast.models.mlp_model import MLPModel
from ..operators import solar_geometry as sg
from ..operators.geo_recover import recover_geo_clearsky_global
from ..operators.tsr_validation import validate_with_tsr
from ..viz.io import save_fig, save_metrics
from ..viz.style import DATASET_COLORS, PALETTE, set_style

__all__ = ["run"]


# ---------------------------------------------------------------------------
# Bounded dev configuration (full-config numbers come at Phase 8).
# ---------------------------------------------------------------------------

#: Dataset with KNOWN coords used for the blind test (geo_inferred=False).
BLIND_DATASET: str = "folsom"
#: Target the clear-sky method can match (measured irradiance, not cloudy PV).
BLIND_TARGET: str = "ghi_actual"
#: Bounded grid for the clear-sky global search (5-deg keeps dev tractable;
#: resolution is not the limiting factor — the hemisphere ambiguity is).
_BLIND_KW: dict = {
    "sample_size": 2400,
    "top_n": 10,
    "lat_grid": np.arange(-60.0, 60.0001, 5.0),
    "lon_grid": np.arange(-180.0, 180.0001, 5.0),
}

#: Contiguous hourly chunk fed to the clear-sky recovery. The operator's uniform
#: STRIDE subsample (``step = len // sample_size``) destroys the diurnal cycle on
#: multi-year series (Folsom is 1.55M hourly rows -> a ~54-day stride that aliases
#: away the daily signal the longitude recovery depends on). Taking a contiguous
#: chunk and disabling the subsample (``sample_size = chunk len``) preserves the
#: diurnal resolution the method needs. 90 days spans enough seasonality for the
#: latitude search while staying bounded.
_BLIND_CHUNK_HOURS: int = 90 * 24

#: Dataset whose coords were TSR-recovered and which carries VAR178 + PV.
_DOWN_DATASET: str = "gefcom2014"
_DOWN_TARGET: str = "pv_power_normalized"
_DOWN_LOOKBACK: int = 24
_DOWN_HORIZON: int = 1
_DOWN_STRIDE: int = 6
#: TINY MLP for fast iteration (Phase 8 scales this up).
_DOWN_MLP_KW: dict = {"hidden_layer_sizes": (16,), "max_iter": 30, "random_state": 0}

#: Features unlocked by recovered coords — the treatment arm of the ablation.
_GEO_FEATURES: tuple[str, ...] = ("ghi_clearsky", "cos_zenith", "clearness_index")

#: ECMWF accumulated top-of-atmosphere radiation column in GEFCom.
_TSR_COL: str = "nwp_VAR178"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def _naive_utc(series: pd.Series) -> pd.DatetimeIndex:
    """Coerce the timestamp column to a naive-UTC DatetimeIndex."""
    ts = pd.to_datetime(series)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.DatetimeIndex(ts)


# ---------------------------------------------------------------------------
# Sub-analysis 1: blind test on a known-coord dataset
# ---------------------------------------------------------------------------

def _blind_test(loader) -> dict:
    df = loader.load_tabular(BLIND_DATASET)
    df = df.dropna(subset=[BLIND_TARGET]).sort_values(
        ["entity_id", "timestamp_utc"]
    ).reset_index(drop=True)
    if not len(df):
        return {
            "dataset": BLIND_DATASET, "target": BLIND_TARGET,
            "true_lat": float("nan"), "true_lon": float("nan"),
            "coords_known": False,
            "recovered_lat": None, "recovered_lon": None,
            "great_circle_error_km": float("nan"),
            "details": {"status": "no data"},
        }

    true_lat = float(df["latitude"].iloc[0])
    true_lon = float(df["longitude"].iloc[0])
    geo_inferred = bool(df["geo_inferred"].iloc[0]) if "geo_inferred" in df.columns else True
    # Take a CONTIGUOUS bounded chunk at full hourly resolution. The operator's
    # uniform stride subsample destroys the diurnal cycle on long series (see
    # ``_BLIND_CHUNK_HOURS``); a contiguous chunk gives clear-sky its fair shot.
    df = df.iloc[:_BLIND_CHUNK_HOURS]
    times = _naive_utc(df["timestamp_utc"])
    target = df[BLIND_TARGET].to_numpy(dtype=float)
    kw = dict(_BLIND_KW)
    kw["sample_size"] = len(df)  # keep the chunk's hourly resolution (no stride)
    # BLIND recovery: the operator sees ONLY the target signal, never the coords.
    res = recover_geo_clearsky_global(times, target, **kw)
    rec_lat = res.get("latitude")
    rec_lon = res.get("longitude")
    if rec_lat is not None and rec_lon is not None:
        err = _haversine_km(true_lat, true_lon, rec_lat, rec_lon)
    else:
        err = float("nan")
    return {
        "dataset": BLIND_DATASET,
        "target": BLIND_TARGET,
        "true_lat": true_lat,
        "true_lon": true_lon,
        "coords_known": not geo_inferred,
        "recovered_lat": float(rec_lat) if rec_lat is not None else None,
        "recovered_lon": float(rec_lon) if rec_lon is not None else None,
        "great_circle_error_km": float(err),
        "details": res.get("details", {}),
    }


def _plot_blindtest(blind: dict, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6, 5))
    color = DATASET_COLORS.get(blind["dataset"], PALETTE[0])
    ax.scatter(
        [blind["true_lon"]], [blind["true_lat"]], marker="*",
        s=260, color=color, edgecolor="black", linewidth=0.6,
        label="True (known coords)", zorder=3,
    )
    rec_lat, rec_lon = blind["recovered_lat"], blind["recovered_lon"]
    if rec_lat is not None and rec_lon is not None:
        ax.scatter(
            [rec_lon], [rec_lat], marker="o", s=140,
            color=PALETTE[1], edgecolor="black", linewidth=0.6,
            label="Recovered (clear-sky, blind)", zorder=3,
        )
        ax.plot(
            [blind["true_lon"], rec_lon], [blind["true_lat"], rec_lat],
            "--", color=PALETTE[5], linewidth=1.5, zorder=2,
        )
        mid_lon = 0.5 * (blind["true_lon"] + rec_lon)
        mid_lat = 0.5 * (blind["true_lat"] + rec_lat)
        ax.annotate(
            f"{blind['great_circle_error_km']:.0f} km",
            xy=(mid_lon, mid_lat), fontsize=10, color=PALETTE[5],
            ha="center", va="bottom",
        )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    status = "clear-sky recovery" if rec_lat is not None else "clear-sky recovery: no solution"
    ax.set_title(f"Blind geo-recovery — {blind['dataset']} ({blind['target']})\n{status}")
    ax.legend()
    return save_fig(fig, "fig_geo_blindtest", out_dir)


# ---------------------------------------------------------------------------
# Sub-analysis 2: TSR validation of recovered GEFCom coords
# ---------------------------------------------------------------------------

def _tsr_validation(loader) -> tuple[list[dict], dict]:
    """Return (per-day rows for plotting, per-zone summary metrics)."""
    df = loader.load_tabular(_DOWN_DATASET)
    if _TSR_COL not in df.columns:
        return [], {}
    df = df.sort_values(["entity_id", "timestamp_utc"]).reset_index(drop=True)
    plot_rows: list[dict] = []
    per_zone: dict[str, dict] = {}
    for eid, g in df.groupby("entity_id", sort=False):
        lat = float(g["latitude"].iloc[0])
        lon = float(g["longitude"].iloc[0])
        times = _naive_utc(g["timestamp_utc"])
        tsr = g[_TSR_COL].to_numpy(dtype=float)
        val = validate_with_tsr(times, tsr, lat, lon)
        per_zone[eid] = {
            "latitude": lat,
            "longitude": lon,
            "r2_latitude": float(val["r2_latitude"]),
            "rmse_latitude": float(val["rmse_latitude"]),
            "longitude_implied": float(val["longitude_implied"]),
            "recovered_longitude": float(val["details"].get("recovered_longitude", float("nan"))),
            "n_days": int(val["details"].get("n_days", 0)),
        }
        # Daily observed TSR total vs theoretical TOA daily irradiation (for the
        # 1:1 scatter). Reuses the same definition as validate_with_tsr.
        s = pd.Series(tsr, index=times).resample("1D").sum()
        doys = s.index.dayofyear.to_numpy(dtype=float)
        theory = sg.toa_daily_irradiation(doys, lat)
        for obs, thr in zip(s.to_numpy(), theory):
            if np.isfinite(obs) and thr > 1e-3:
                plot_rows.append({"zone": eid, "observed": float(obs), "theoretical": float(thr)})
    return plot_rows, per_zone


def _plot_tsr_validation(plot_rows: list[dict], per_zone: dict, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if plot_rows:
        dfp = pd.DataFrame(plot_rows)
        zones = sorted(dfp["zone"].unique())
        for i, z in enumerate(zones):
            sub = dfp[dfp["zone"] == z]
            r2 = per_zone.get(z, {}).get("r2_latitude", float("nan"))
            label = f"{z} (R²={r2:.2f})" if np.isfinite(r2) else z
            ax.scatter(
                sub["theoretical"], sub["observed"], s=28, alpha=0.75,
                color=PALETTE[i % len(PALETTE)], label=label,
            )
        lo = min(dfp["theoretical"].min(), dfp["observed"].min())
        hi = max(dfp["theoretical"].max(), dfp["observed"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.5, label="1:1")
    ax.set_xlabel("Theoretical daily TOA irradiation (J/m²)")
    ax.set_ylabel("Observed daily VAR178 / TSR (J/m²)")
    ax.set_title("TSR validation — recovered geometry vs GEFCom VAR178")
    ax.legend()
    return save_fig(fig, "fig_geo_tsr_validation", out_dir)


# ---------------------------------------------------------------------------
# Sub-analysis 3: downstream forecasting value (GEFCom, TINY MLP)
# ---------------------------------------------------------------------------

def _finite_fraction(series: pd.Series) -> float:
    """Fraction of finite values in a column (0 for all-NaN / object-None)."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0


def _run_downstream_arm(loader, label: str, feature_cols: list[str]) -> dict:
    data_root = str(getattr(loader, "data_root", "") or "")
    config = PipelineConfig(
        dataset=_DOWN_DATASET,
        target=_DOWN_TARGET,
        lookback=_DOWN_LOOKBACK,
        horizon=_DOWN_HORIZON,
        stride=_DOWN_STRIDE,
        feature_cols=list(feature_cols),
        data_root=data_root or None,
    )
    engineer = EndogenousFeatureEngineer(_DOWN_TARGET, feature_cols=list(feature_cols))
    model = MLPModel(name=f"MLP_{label}", **_DOWN_MLP_KW)
    pipeline = SolarForecastPipeline(config, engineer, model)

    # Reuse the injected loader so the experiment reads only standardized data
    # (no hidden re-instantiation of SolarUniBenchLoader). Drop rows whose target
    # is NaN — the MLP cannot fit on missing labels (Honest NaN, never fabricated).
    def _df_provider(cfg):
        d = loader.load_tabular(cfg.dataset)
        return d.dropna(subset=[_DOWN_TARGET]).reset_index(drop=True)

    res = pipeline.run(df_provider=_df_provider)
    day = res.get("daytime", {}) or {}
    return {
        "label": label,
        "feature_cols": list(feature_cols),
        "daytime_mae": float(day.get("Daytime_MAE", float("nan"))),
        "daytime_rmse": float(day.get("Daytime_RMSE", float("nan"))),
        "n_daytime": int(day.get("n_daytime", 0)),
    }


def _downstream_value(loader) -> dict[str, dict]:
    """Run the raw-lags vs +geo ablation, using only geo features actually present.

    Per the spec's Honest-NaN principle, a feature column that is entirely NaN for
    the dataset (e.g. ``clearness_index`` on GEFCom, which is PV-only and carries
    no measured ``ghi_actual`` to form ``ghi_actual/ghi_clearsky``) is NOT
    fabricated — it is dropped and the omission is reported in the metrics. The
    recovered coords still unlock the populated anchors (``ghi_clearsky``,
    ``cos_zenith``).
    """
    df = loader.load_tabular(_DOWN_DATASET)
    requested = list(_GEO_FEATURES)
    usable = [c for c in requested if c in df.columns and _finite_fraction(df[c]) > 0.5]
    return {
        "raw_lags": _run_downstream_arm(loader, "raw_lags", []),
        "geo_features": _run_downstream_arm(loader, "geo_features", usable),
        "geo_features_requested": requested,
        "geo_features_used": usable,
    }


def _down_metrics(down: dict[str, dict]) -> dict:
    base = down["raw_lags"]["daytime_mae"]
    geo = down["geo_features"]["daytime_mae"]
    delta = float(base - geo) if (np.isfinite(base) and np.isfinite(geo)) else float("nan")
    return {
        "baseline_daytime_mae": base,
        "geo_daytime_mae": geo,
        "delta_daytime_mae": delta,  # positive = geo features reduce MAE
        "feature_cols": down.get("geo_features_requested", list(_GEO_FEATURES)),
        "geo_features_used": down.get("geo_features_used", []),
        "raw_lags": down["raw_lags"],
        "geo_features": down["geo_features"],
    }


def _plot_skill_gain(down: dict[str, dict], out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    labels = ["raw PV lags", "+geo features"]
    maes = [down["raw_lags"]["daytime_mae"], down["geo_features"]["daytime_mae"]]
    bars = ax.bar(labels, maes, color=[PALETTE[0], PALETTE[1]], width=0.55)
    ax.set_ylabel("Daytime MAE (normalized PV)")
    used = down.get("geo_features_used", [])
    used_str = ",".join(used) if used else "(none populated)"
    ax.set_title(f"Downstream value of recovered coords (GEFCom, TINY MLP)\n"
                 f"geo arm features: {used_str}")
    ymax = max([m for m in maes if np.isfinite(m)], default=1.0)
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    for bar, m in zip(bars, maes):
        if np.isfinite(m):
            ax.text(
                bar.get_x() + bar.get_width() / 2, m, f"{m:.4f}",
                ha="center", va="bottom", fontsize=10,
            )
    delta = maes[0] - maes[1] if all(np.isfinite(m) for m in maes) else float("nan")
    if np.isfinite(delta):
        ax.text(
            0.5, 0.97, f"ΔMAE = {delta:+.4f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    return save_fig(fig, "fig_geo_skill_gain", out_dir)


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

def _save_results_csv(metrics: dict, out_dir: Path) -> Path:
    rows: list[dict] = []
    bt = metrics["blind_test"]
    rows.append({
        "section": "blind_test",
        "key": bt["dataset"],
        "great_circle_error_km": bt["great_circle_error_km"],
        "true_lat": bt["true_lat"], "true_lon": bt["true_lon"],
        "recovered_lat": bt["recovered_lat"], "recovered_lon": bt["recovered_lon"],
    })
    for zone, v in metrics["tsr_validation"].items():
        rows.append({"section": "tsr_validation", "key": zone, **v})
    dn = metrics["downstream"]
    rows.append({
        "section": "downstream", "key": "raw_lags",
        "daytime_mae": dn["raw_lags"]["daytime_mae"],
        "n_daytime": dn["raw_lags"]["n_daytime"],
    })
    rows.append({
        "section": "downstream", "key": "geo_features",
        "daytime_mae": dn["geo_features"]["daytime_mae"],
        "n_daytime": dn["geo_features"]["n_daytime"],
        "delta_daytime_mae": dn["delta_daytime_mae"],
        "geo_features_used": ",".join(dn.get("geo_features_used", [])),
    })
    return save_metrics(pd.DataFrame(rows), "results_e1_geo", out_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(loader, out_dir) -> dict:
    """Run experiment E1 and return ``{figures: [paths], metrics: {...}}``.

    Reads only standardized packages via ``loader``; writes figures + a results
    CSV under ``out_dir``. Pure: does not mutate ``loader`` state or the
    standardized data.
    """
    set_style("light")
    out_dir = Path(out_dir)
    figures: list[Path] = []
    metrics: dict = {}

    # 1. Blind test (Folsom, known coords).
    blind = _blind_test(loader)
    figures += _plot_blindtest(blind, out_dir)
    metrics["blind_test"] = blind

    # 2. TSR validation (GEFCom recovered coords vs VAR178).
    plot_rows, per_zone = _tsr_validation(loader)
    figures += _plot_tsr_validation(plot_rows, per_zone, out_dir)
    metrics["tsr_validation"] = per_zone

    # 3. Downstream value (GEFCom, raw vs +geo TINY MLP).
    down = _downstream_value(loader)
    figures += _plot_skill_gain(down, out_dir)
    metrics["downstream"] = _down_metrics(down)

    _save_results_csv(metrics, out_dir)

    return {"figures": [str(p) for p in figures], "metrics": metrics}
