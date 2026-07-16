"""Slow real-data integration tests for the Phase 4 image builders (RUN_SLOW-gated).

Runs each imaging builder on a bounded slice of the REAL datasets and asserts the
output zarr exists with the expected shape. These do NOT go through the adapter
pipeline — they exercise the builder + operator stack (BRR, zarr I/O, optical
flow) directly on real bytes, which is what the adapters call.

Set ``RUN_SLOW=1`` to run. Defaults to the canonical benchmark data root; override
with ``SOLAR_UNIBENCH_DATA_ROOT``.

Note on SunLake: ``hub://crossvivit/SunLake`` is pulled on-demand per build (not
materialized locally). The hub ``context/data`` tensor has shape ``(M, 11, 64, 64)``
with one sample per timestep, so we verify the leading dim equals the requested
``limit_frames`` (20), NOT 11 — the old ``[0]``-indexing bug returned a single
``(11, 64, 64)`` frame.
"""
import os
import time

import pytest
import zarr

DATA = os.environ.get("SOLAR_UNIBENCH_DATA_ROOT", "D:/monica/workspace/solar_benchmark_dataset")
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")


@slow
def test_real_skippd_sky(tmp_path):
    from solarunibench.imaging.skippd_sky import build_skippd_sky_zarr

    src = f"{DATA}/SKIPPD_all/default"
    out = tmp_path / "sky.zarr"
    t0 = time.perf_counter()
    res = build_skippd_sky_zarr(src, out, limit=50)
    dt = time.perf_counter() - t0
    arr = zarr.open(str(out), mode="r")["images"]
    assert arr.shape == (res["n_frames"], 4, 256, 256)
    assert res["n_frames"] == 50
    assert arr.dtype.kind == "f"
    print(f"\n[SKIPPD sky integration] frames={res['n_frames']} "
          f"shape={arr.shape} build_seconds={dt:.1f}")


@slow
def test_real_folsom_sky(tmp_path):
    from solarunibench.imaging.folsom_sky import build_folsom_sky_zarr

    tar = f"{DATA}/Folsom_data/Folsom_sky_images_2014.tar.bz2"
    if not os.path.exists(tar):
        pytest.skip(f"{tar} not present — Folsom sky tarball missing")
    out = tmp_path / "sky.zarr"
    t0 = time.perf_counter()
    res = build_folsom_sky_zarr([tar], out, limit=50)
    dt = time.perf_counter() - t0
    arr = zarr.open(str(out), mode="r")["images"]
    assert arr.shape[0] == 50
    assert arr.shape[1:] == (4, 256, 256)
    assert res["n_frames"] == 50
    print(f"\n[Folsom sky integration] frames={res['n_frames']} "
          f"shape={arr.shape} build_seconds={dt:.1f}")


@slow
def test_real_sunlake_satellite(tmp_path):
    from solarunibench.imaging.sunlake_satellite import build_sunlake_satellite_zarr

    t0 = time.perf_counter()
    res = build_sunlake_satellite_zarr(
        "hub://crossvivit/SunLake", tmp_path, limit_frames=20, steps_k=4
    )
    dt = time.perf_counter() - t0
    z = zarr.open(str(tmp_path / "satellite_future.zarr"), mode="r")
    # Leading dim MUST be the requested frame count (20), not 11 — the old
    # `[0]`-indexing bug dropped (M,...) to a single (11, 64, 64) frame.
    assert z["context"].shape[0] == 20, (
        f"context leading dim {z['context'].shape[0]} != 20 (hub-read bug regression)"
    )
    assert z["context"].shape[1:] == (11, 64, 64)
    # future: (steps_k, M-1, C, H, W) — extrapolation runs over consecutive pairs.
    assert z["future"].shape[0] == 4
    assert z["future"].shape[1] == 19  # M-1
    print(f"\n[SunLake satellite integration] context_frames={z['context'].shape[0]} "
          f"future_shape={z['future'].shape} station={res['station']} "
          f"build_seconds={dt:.1f}")
