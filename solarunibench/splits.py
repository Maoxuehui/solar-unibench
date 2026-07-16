"""Frozen train/val/test splits. Per-entity chronological; dataset-specific test cutoff."""
import json
from pathlib import Path
import pandas as pd


def make_splits(df_ts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return df_ts with an added `split` column in {train, val, test}, per entity.

    Temporal strategy: rows with timestamp >= test_start are `test`; the most recent
    `val_ratio` fraction of the remainder is `val`; the rest is `train` — leak-free.
    """
    strategy = cfg["strategy"]
    if strategy != "temporal":
        raise NotImplementedError(f"split strategy '{strategy}' is implemented in a later phase")
    out = df_ts.copy()
    labels = pd.Series("train", index=out.index, dtype="object")
    test_start = pd.Timestamp(cfg["test_start"], tz="UTC")
    val_ratio = float(cfg.get("val_ratio", 0.1))
    for _, idx in out.groupby("entity_id").groups.items():
        sub = out.loc[idx].sort_values("timestamp_utc")
        is_test = sub["timestamp_utc"] >= test_start
        rest = sub[~is_test]
        n_val = max(1, int(len(rest) * val_ratio))
        val_idx = rest.index[-n_val:]
        labels.loc[sub.index[is_test]] = "test"
        labels.loc[val_idx] = "val"
    out["split"] = labels.values
    return out


def build(dataset_dir, cfg: dict, splits_version: str) -> dict:
    """Stamp the `split` column into time_series.parquet (in place) + write manifest."""
    dataset_dir = Path(dataset_dir)
    ts_path = dataset_dir / "time_series.parquet"
    df_ts = pd.read_parquet(ts_path)
    splits = make_splits(df_ts, cfg)
    # write the full table back WITH the new split column (single source of truth)
    splits.to_parquet(ts_path, index=False)
    # Per-split timestamp bounds (min/max `timestamp_utc`) make the frozen split
    # fully self-documenting without re-reading configs/splits.yaml.
    split_bounds = {}
    for label, grp in splits.groupby("split"):
        ts = grp["timestamp_utc"]
        split_bounds[label] = {
            "min": pd.Timestamp(ts.min()).isoformat(),
            "max": pd.Timestamp(ts.max()).isoformat(),
        }
    manifest = {"dataset": dataset_dir.name, "splits_version": splits_version,
                "strategy": cfg.get("strategy"),
                "test_start": cfg.get("test_start"),
                "counts": {k: int(v) for k, v in splits["split"].value_counts().to_dict().items()},
                "split_bounds": split_bounds}
    (dataset_dir / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
