import subprocess, sys
from pathlib import Path
import pandas as pd


def test_cli_builds_gefcom_from_fixture(tmp_path):
    # reuse the same fixture writer as the adapter test
    from tests.test_gefcom_adapter import _write_fixture
    _write_fixture(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, "scripts/build_dataset.py", "--dataset", "gefcom2014",
           "--data-root", str(tmp_path), "--output-root", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "gefcom2014" / "metadata.parquet").exists()
    assert (out / "gefcom2014" / "manifest.json").exists()
