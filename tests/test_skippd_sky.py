import io
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from PIL import Image
from solarunibench.imaging.skippd_sky import build_skippd_sky_zarr


def _write_fixture(root: Path, n=3):
    for split in ("train", "test"):
        d = root / "default" / split
        d.mkdir(parents=True)
        rows = []
        for i in range(n):
            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
            rows.append({"image": {"bytes": buf.getvalue(), "path": f"{i}.png"},
                         "time": pd.Timestamp(f"2018-06-01 12:{i:02d}", tz="US/Pacific"), "pv": 0.5})
        pd.DataFrame(rows).to_parquet(d / "0000.parquet")


def test_build_skippd_sky_zarr(tmp_path):
    _write_fixture(tmp_path, n=3)
    out = tmp_path / "sky.zarr"
    res = build_skippd_sky_zarr(tmp_path / "default", out, limit=4)
    arr = zarr.open(str(out), mode="r")["images"]
    assert arr.shape[0] == 4  # 3 train + 1 test (capped at limit=4)
    assert arr.shape[1:] == (4, 256, 256)
    assert arr.dtype == np.float32
    assert res["n_frames"] == arr.shape[0]
