import io, tarfile, numpy as np, zarr
from pathlib import Path
from PIL import Image
from solarunibench.imaging.folsom_sky import build_folsom_sky_zarr, iter_folsom_sky_frames
import pandas as pd


def _make_tar(path, names):
    with tarfile.open(path, "w:bz2") as tf:
        for nm in names:
            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG")
            data = buf.getvalue()
            ti = tarfile.TarInfo(nm); ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def test_iter_frames_parses_utc_timestamp(tmp_path):
    tar = tmp_path / "2014.tar.bz2"
    _make_tar(tar, ["2014/06/01/20140601_120000.jpg", "2014/06/01/20140601_120100.jpg"])
    frames = list(iter_folsom_sky_frames(tar))
    assert len(frames) == 2
    ts0, f0 = frames[0]
    assert ts0 == pd.Timestamp("2014-06-01 12:00:00", tz="UTC")
    assert f0.shape == (4, 256, 256)


def test_build_folsom_sky_zarr(tmp_path):
    tar = tmp_path / "2014.tar.bz2"
    _make_tar(tar, [f"2014/06/01/20140601_1200{i:02d}.jpg" for i in range(5)])
    out = tmp_path / "sky.zarr"
    res = build_folsom_sky_zarr([tar], out, limit=3)
    arr = zarr.open(str(out), mode="r")["images"]
    assert arr.shape == (3, 4, 256, 256)
    assert res["n_frames"] == 3
