"""Quality audit: night-zeroing, clearness index, quality flag, physical clamp."""
import numpy as np
import pandas as pd


def apply_quality_audit(df_ts: pd.DataFrame, capacity: float = 1.0) -> pd.DataFrame:
    df = df_ts.copy()
    eps = 1e-6
    df["clearness_index"] = np.clip(
        df.get("ghi_actual", np.nan) / (df.get("ghi_clearsky", np.nan) + eps), 0.0, 1.5
    ).astype("float32")

    df["quality_flag"] = 0  # good
    night = df["cos_zenith"] <= 0
    # Zero night values only for columns that actually exist AND carry data — never
    # fabricate a missing modality (Honest NaN). For GEFCom (PV-only, no ghi_actual)
    # this must NOT create a ghi_actual column of fabricated zeros; for Folsom
    # (irradiance-only, all-NA pv_power_normalized) this must NOT fabricate night
    # zeros for a PV sensor that doesn't exist.
    night_cols = [c for c in ("ghi_actual", "pv_power_normalized")
                  if c in df.columns and df[c].notna().any()]
    if night_cols:
        df.loc[night, night_cols] = 0.0
    df.loc[night, "quality_flag"] = 3  # night
    # Night clearness_index is only meaningful when ghi_actual exists AND carries
    # data; otherwise the ratio is already NaN and must stay NaN (don't fabricate
    # 0.0 at night for a missing irradiance modality).
    if "ghi_actual" in df.columns and df["ghi_actual"].notna().any():
        df.loc[night, "clearness_index"] = np.float32(0.0)

    # curtailment / clipping: PV pinned at capacity across daytime — only when the
    # PV modality is actually present (non-NA); an all-NA column is PV-less.
    if "pv_power_normalized" in df.columns and df["pv_power_normalized"].notna().any():
        df["pv_power_normalized"] = np.clip(df["pv_power_normalized"], 0.0, capacity)
    # physical clamp on irradiance (no negatives) — only for columns carrying data.
    for c in ("ghi_actual", "dni_actual", "dhi_actual"):
        if c in df and df[c].notna().any():
            df[c] = df[c].clip(lower=0.0)
    return df
