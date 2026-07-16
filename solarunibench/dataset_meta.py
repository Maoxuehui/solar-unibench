"""Known per-dataset station metadata (coords are public; geo_inferred=False)."""

FOLSOM = {
    "entity_id": "FOLSOM_STN",
    "dataset_source": "folsom",
    "latitude": 38.67,
    "longitude": -121.15,
    "elevation_meters": 100.0,
    "local_timezone": "America/Los_Angeles",
    "surface_albedo": 0.2,
    "capacity_kw": float("nan"),  # Folsom is irradiance-only
    "geo_inferred": False,
}

SKIPP_CAPACITY_KW = 30.1  # SKIPPD residential array rated power (pv peaks ~29.6 raw)
SKIPP = {
    "entity_id": "SKIPPD_STFORD",
    "dataset_source": "skippd",
    "latitude": 37.42,
    "longitude": -122.17,
    "elevation_meters": 30.0,
    "local_timezone": "America/Los_Angeles",
    "surface_albedo": 0.2,
    "capacity_kw": SKIPP_CAPACITY_KW,
    "geo_inferred": False,
}
