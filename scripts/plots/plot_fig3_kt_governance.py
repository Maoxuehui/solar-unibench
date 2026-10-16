"""Standalone Figure — clearness-index governance (fig03 panel a extracted).

Publication-ready Raw-vs-Curated kt comparison, REAL DATA ONLY:
* Raw       : publisher-csv kt (ghi / ghi_clearsky) with measured
              super-physical spikes (midday kt up to 2.09).
* Curated   : standardized clearness_index — clamped to [0, 1.5] and
              night-zeroed (cos(theta_z) <= 0 => 0) by operators/quality.py.
Layout fixes per user spec:
* wspace 0.4 — the right cell's y-label ($k_t$, curated) clears the left
  cell's right border.
* Red (#D90429) raw column / deep-blue (#0F4C81) curated column.
* 300 dpi PNG + vector PDF, bbox_inches='tight'.

Run (conda env solar_uni):
    python scripts/plot_fig3_kt_governance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the fig3 module's loaders, day picker, palette and rcParams
# (importing it applies the house rcParams block).
from scripts.plot_fig3_governance import (
    COSZ_MIDDAY, KT_SUPER, RAW_COLOR, CUR_COLOR, SPIKE_COLOR,
    INK, INK_2, INK_MUTED, load_joined, pick_day_a,
)
import matplotlib as mpl

WIDTH_MM = 200
HEIGHT_MM = 78
WSPACE = 0.4                     # user spec: right y-label clears left border
OUT_DIR = _REPO_ROOT / "Figures"
OUT_STEM = "fig03_clearness_index_governance"


def main() -> None:
    print("loading real files ...")
    j = load_joined()
    day = pick_day_a(j)

    t0 = pd.Timestamp(day) + pd.Timedelta(hours=8)      # UTC day window
    t1 = t0 + pd.Timedelta(days=1)
    w = j.loc[t0:t1]
    x = np.arange(len(w)) / 60.0
    kt_raw = w["kt_raw"].to_numpy(dtype=float).copy()
    kt_raw[w["ghi_clearsky"].to_numpy() < 10.0] = np.nan
    kt_cur = w["clearness_index"].to_numpy(dtype=float)
    cosz = w["cos_zenith"].to_numpy()
    night = cosz <= 0
    spike = (~night) & (cosz > COSZ_MIDDAY) & (kt_raw > KT_SUPER)

    fig, (ax_r, ax_c) = plt.subplots(1, 2,
                                     figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.155,
                        wspace=WSPACE)

    for ax, kt, tag in ((ax_r, kt_raw, "raw"), (ax_c, kt_cur, "curated")):
        ax.axvspan(0, float(night.argmax()) / 60.0, color="gray", alpha=0.12,
                   zorder=0)
        ax.axvspan(float(len(w) - night[::-1].argmax()) / 60.0, 24.0,
                   color="gray", alpha=0.12, zorder=0)
        ax.plot(x, kt, color=RAW_COLOR if tag == "raw" else CUR_COLOR, lw=0.7)
        ax.axhline(1.0, color=INK_MUTED, lw=0.5, ls=(0, (4, 2)), zorder=1)
        ax.axhline(1.5, color=SPIKE_COLOR, lw=0.6, ls=(0, (4, 2)), zorder=1)
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_ylim(0, 2.4)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xlabel("Hour (UTC)")

    # Night callouts centered inside the LEFT astronomical-night band
    # (band bounds are computed from cos(zenith) <= 0, not hardcoded).
    x_n0 = float(night.argmax()) / 60.0

    # -- Raw (ungoverned) ----------------------------------------------------
    ax_r.set_ylabel(r"$k_t$ (–)")
    ax_r.plot(x[spike], kt_raw[spike], linestyle="none", marker="o",
              ms=1.6, color=INK, zorder=5)
    i_peak = int(np.nanargmax(np.where(spike, kt_raw, np.nan)))
    kmax = float(kt_raw[i_peak])
    ax_r.annotate(f"super-physical: $k_t$ up to {kmax:.1f}\n"
                  r"(clear-sky limit $k_t$ = 1 dashed)",
                  xy=(x[i_peak], kt_raw[i_peak]),
                  xytext=(2.0, 2.1),
                  fontsize=7, color=INK, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.4, color=INK,
                                  connectionstyle="arc3,rad=-0.18"))
    ax_r.text(0.55, 0.10, "night:\n$k_t$ undefined (gap)",
              ha="left", va="bottom", fontsize=7, color=INK_MUTED,
              linespacing=1.35)
    ax_r.set_title("(a) Raw (ungoverned)", loc="left", pad=3)

    # -- Curated (governed) --------------------------------------------------
    ax_c.set_ylabel(r"$k_t$ (–)")
    n_clamp = int((w["clearness_index"] >= 1.5 - 1e-3).sum())
    ax_c.annotate(f"clamped at $k_t$ = 1.5\n({n_clamp} samples this day)",
                  xy=(0.03, 0.95), xycoords="axes fraction", ha="left",
                  va="top", fontsize=7, color=SPIKE_COLOR)
    ax_c.text(0.55, 0.10, "night:\n" + r"$k_t$ zeroed ($\cos\theta_z \leq 0$)",
              ha="left", va="bottom", fontsize=7, color=INK_MUTED,
              linespacing=1.35)
    ax_c.set_title("(b) Curated (governed)", loc="left", pad=3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300,
                    bbox_inches="tight")
        print(f"  {OUT_STEM}.{ext}")
    plt.close(fig)
    print(f"done -> {OUT_DIR / (OUT_STEM + '.png')}")


if __name__ == "__main__":
    main()
