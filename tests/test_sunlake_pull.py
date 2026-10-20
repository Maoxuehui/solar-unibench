from scripts.data_tools.pull_sunlake_tabular import find_channel_indices


def test_find_channel_indices_cab():
    ch = ["GHI", "RH [%]", "T2 [°C]", "PoPoPoPo [hPa]", "DIF [W/m**2]", "DIR [W/m**2]", "ghi", "dni", "dhi"]
    idx = find_channel_indices(ch)
    assert idx["ghi_actual"] == 0   # uppercase GHI = ground truth
    assert idx["dhi_actual"] == 4   # DIF
    assert idx["dni_actual"] == 5   # DIR


def test_find_channel_indices_missing_returns_none():
    idx = find_channel_indices(["RH [%]", "T2 [°C]"])  # no GHI/DIF/DIR
    assert idx["ghi_actual"] is None
    assert idx["dhi_actual"] is None
    assert idx["dni_actual"] is None
