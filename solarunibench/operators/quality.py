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
    df.loc[night, ["ghi_actual", "pv_power_normalized"]] = 0.0
    df.loc[night, "quality_flag"] = 3  # night
    # night clearness_index must be 0.0 (ghi_clearsky ~= 0 at night would
    # otherwise leave the pre-clip ratio clipped to 1.5); keep float32 dtype.
    df.loc[night, "clearness_index"] = np.float32(0.0)

    # curtailment / clipping: PV pinned at capacity across daytime
    if "pv_power_normalized" in df:
        df["pv_power_normalized"] = np.clip(df["pv_power_normalized"], 0.0, capacity)
    # physical clamp on irradiance (no negatives)
    for c in ("ghi_actual", "dni_actual", "dhi_actual"):
        if c in df:
            df[c] = df[c].clip(lower=0.0)
    return df
