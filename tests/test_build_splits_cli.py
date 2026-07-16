import subprocess
import sys
import pandas as pd


def test_cli_writes_gefcom_splits(tmp_path):
    from tests.test_gefcom_adapter import _write_fixture
    from solarunibench.pipeline.adapters.gefcom import GEFCom2014Pipeline
    _write_fixture(tmp_path)
    GEFCom2014Pipeline(raw_data_dir=str(tmp_path / "GEFCom2014_data"),
                       output_dir=str(tmp_path / "out"), dataset_name="gefcom2014").run_pipeline()
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/build_splits.py", "--dataset", "gefcom2014",
           "--output-root", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ts = pd.read_parquet(out / "gefcom2014" / "time_series.parquet")
    assert "split" in ts.columns
    assert set(ts["split"].unique()) <= {"train", "val", "test"}
    assert (out / "gefcom2014" / "splits_manifest.json").exists()
