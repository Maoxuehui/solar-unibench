"""Per-dataset physics-time configuration for the E8 unified benchmark.

The benchmark is **physics-time-aligned**, not step-locked: every dataset is
evaluated at the same three physical forecast horizons (short / medium / long),
with the step counts derived from each dataset's native sampling rate. This
module is the single source of truth for that mapping.

Matrix (see ``docs/superpowers/plans/2026-08-13-e8-unified-benchmark.md``):

================= =========== =========== =============== ================== =======
 dataset           target      sampling    lookback        horizons (steps)   image
================= =========== =========== =============== ================== =======
 folsom            ghi_actual  1 min       240 (4 h)       [5, 10, 15]        yes
 skippd            pv_power_n  1 min       240 (4 h)       [5, 10, 15]        yes
 sunlake           ghi_actual  30 min      24 (12 h)       [1, 2, 3]          no
 gefcom2014        pv_power_n  60 min      24 (24 h)       [1, 2, 3]          no
================= =========== =========== =============== ================== =======

Targets: ``pv_power_normalized`` for PV datasets (gefcom2014, skippd),
``ghi_actual`` for irradiance datasets (folsom, sunlake).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DatasetConfig:
    """Physics-time config for one benchmark dataset.

    Attributes
    ----------
    target
        Forecast target column: ``ghi_actual`` (GHI datasets) or
        ``pv_power_normalized`` (PV datasets).
    sampling_mins
        Native sampling interval in minutes (1, 30, or 60).
    lookback_steps
        Lookback window length in steps (sampling-multiples). Combined with
        ``sampling_mins`` gives the physical lookback duration.
    horizons_steps
        Forecast horizons in steps. ``horizons_steps[i] * sampling_mins`` equals
        ``physical_horizons_min(dataset)[i]``.
    image_dataset
        Whether sky/satellite images are available (CrossViViT is run only on
        image datasets).
    """

    target: str
    sampling_mins: int
    lookback_steps: int
    horizons_steps: list[int]
    image_dataset: bool

    def __post_init__(self) -> None:
        if self.sampling_mins <= 0:
            raise ValueError(f"sampling_mins must be positive (got {self.sampling_mins})")
        if self.lookback_steps <= 0:
            raise ValueError(
                f"lookback_steps must be positive (got {self.lookback_steps})"
            )
        if len(self.horizons_steps) == 0:
            raise ValueError("horizons_steps must be non-empty")
        if any(h <= 0 for h in self.horizons_steps):
            raise ValueError(
                f"horizons_steps must all be positive (got {self.horizons_steps})"
            )


# Physical forecast horizons in minutes: short / medium / long. Shared across
# every dataset so the benchmark reports skill at comparable lead times.
_PHYSICAL_HORIZONS_MIN: dict[str, tuple[int, ...]] = {
    "folsom": (5, 10, 15),
    "skippd": (5, 10, 15),
    "sunlake": (30, 60, 90),
    "gefcom2014": (60, 120, 180),
}


DATASET_CONFIG: Mapping[str, DatasetConfig] = {
    "folsom": DatasetConfig(
        target="ghi_actual",
        sampling_mins=1,
        lookback_steps=240,  # 4 h
        horizons_steps=[5, 10, 15],
        image_dataset=True,
    ),
    "skippd": DatasetConfig(
        target="pv_power_normalized",
        sampling_mins=1,
        lookback_steps=240,  # 4 h
        horizons_steps=[5, 10, 15],
        image_dataset=True,
    ),
    "sunlake": DatasetConfig(
        target="ghi_actual",
        sampling_mins=30,
        lookback_steps=24,  # 12 h
        horizons_steps=[1, 2, 3],
        image_dataset=False,
    ),
    "gefcom2014": DatasetConfig(
        target="pv_power_normalized",
        sampling_mins=60,
        lookback_steps=24,  # 24 h
        horizons_steps=[1, 2, 3],
        image_dataset=False,
    ),
}


def physical_horizons_min(dataset: str) -> list[int]:
    """Return the physical forecast horizons (minutes) for ``dataset``.

    ``[5, 10, 15]`` for the 1-min datasets (folsom, skippd),
    ``[30, 60, 90]`` for sunlake, ``[60, 120, 180]`` for gefcom2014.

    Raises
    ------
    KeyError
        If ``dataset`` is not in the benchmark matrix.
    """
    if dataset not in _PHYSICAL_HORIZONS_MIN:
        raise KeyError(
            f"unknown dataset '{dataset}'; expected one of "
            f"{sorted(_PHYSICAL_HORIZONS_MIN)}"
        )
    return list(_PHYSICAL_HORIZONS_MIN[dataset])
