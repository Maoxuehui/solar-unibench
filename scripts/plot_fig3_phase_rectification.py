"""Standalone Figure — cross-modal phase rectification, REAL DATA ONLY.

The honest counterpart of the "+30 min hardware lag" narrative: the REAL
measured cross-modal misalignment in Folsom is second-cadence sky-camera
timestamps vs the 1-min tabular grid (E6: median |dt| = 1 s, max 30 s).
Exact-match attach drops 64% of frames; nearest-neighbour attach within
tolerance recovers 99%. Downstream gain is real and measured.

Rows (all sharing one x-axis, 90-min window with a real cloud ramp):
  (a) Observed ground GHI (1-min pyranometer) — the ramp event.
  (b) Phase-Rectified Curated — sky-camera bright-pixel feature stream,
      ALL frames attached at nearest-neighbour minutes.
  (c) Unaligned Raw (dt = 0 min, exact-match) — only frames landing on the
      minute grid survive; dropped frames shown as grey x.
Arrow: one real dropped frame lifted into (b) by NN attach, its true |dt|
annotated. Badge: REAL E6 validation numbers read from the canonical CSV.

Run (conda env solar_uni):
    python scripts/plot_fig3_phase_rectification.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from PIL import Image

# House loaders / palette / rcParams from the fig3 module.
from scripts.plot_fig3_governance import (
    CUR_COLOR, INK, INK_2, INK_MUTED, RAW_COLOR, load_joined,
)

WIDTH_MM = 180
HEIGHT_MM = 128
#: Scanned window: ALL 89 frames carry off-grid seconds (orphan_frac = 1.00,
#: exact-match attach keeps NOTHING here) while GHI ramps down ~1018 W/m^2 —
#: the strongest real exact-match failure + ramp co-occurrence in 2014.
WIN_T0 = "2014-03-25 19:00"
WIN_T1 = "2014-03-25 20:30"
E6_CSV = _REPO_ROOT / "figures_full" / "results" / "results_e6_alignment.csv"
SKY_ZARR = _REPO_ROOT / "data" / "standardized" / "folsom" / "sky_images_2014.zarr"
OUT_DIR = _REPO_ROOT / "Figures"
OUT_STEM = "fig03_governance_comparison_with_phase_rectification"


def load_e6_full() -> dict:
    df = pd.read_csv(E6_CSV)
    r = df[(df.dataset == "folsom") & (df.arm == "unaligned")].iloc[0]
    a = df[(df.dataset == "folsom") & (df.arm == "aligned")].iloc[0]
    return dict(mae_raw=float(r.mae), mae_cur=float(a.mae),
                skill_raw=float(r.skill), skill_cur=float(a.skill),
                n=int(r.n_samples), exact=float(r.exact_frac),
                nn=float(r.nn_frac), medlag=float(r.median_lag_seconds))


def load_frames() -> pd.DataFrame:
    """Per-frame bright-pixel feature (NRBR < -0.1 fraction) + attach facts."""
    g = zarr.open_group(str(SKY_ZARR), mode="r")
    ts = pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                        utc=True).tz_localize(None)
    m = (ts >= WIN_T0) & (ts < WIN_T1)
    rows = []
    for i in np.flatnonzero(m):
        img = np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))),
                         dtype=np.float32)
        r, b = img[..., 0], img[..., 2]
        nrbr = (b - r) / (b + r + 1e-6)
        t = ts[i]
        nn = t.round("min")
        rows.append(dict(t=t, x=(t - pd.Timestamp(WIN_T0)).total_seconds() / 60,
                         feat=float((nrbr < -0.1).mean()),
                         exact=int(t.second == 0),
                         dt_attach=abs((nn - t).total_seconds())))
    df = pd.DataFrame(rows)
    print(f"[frames] n={len(df)} exact={int(df.exact.sum())} "
          f"dropped={int((1 - df.exact).sum())} "
          f"max|dt|={df.dt_attach.max():.0f}s corr(feat,GHI) check below")
    return df


def main() -> None:
    print("loading real files ...")
    j = load_joined()
    e6 = load_e6_full()
    fr = load_frames()
    mae_red = (e6["mae_cur"] / e6["mae_raw"] - 1.0) * 100.0

    w = j.loc[WIN_T0:WIN_T1]
    xg = (w.index - pd.Timestamp(WIN_T0)).total_seconds() / 60.0
    ghi = w["ghi"].to_numpy()
    ghi_nn = fr.t.dt.round("min").map(w["ghi"]).to_numpy(dtype=float)
    corr = np.corrcoef(fr.feat, ghi_nn)[0, 1]
    print(f"[coherence] corr(bright-pixel fraction, GHI at NN minute) = {corr:+.3f}")

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, sharex=True, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        gridspec_kw=dict(height_ratios=[1.0, 0.72, 0.72]))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.085,
                        hspace=0.50)

    # (a) observed ground GHI + real E6 badge -------------------------------
    ax_a.plot(xg, ghi, color=CUR_COLOR, lw=0.8)
    ax_a.set_ylabel("GHI (W m$^{-2}$)")
    ax_a.set_ylim(0, 1200)
    ax_a.set_title("(a) Observed ground GHI (pyranometer, 1-min)",
                   loc="left", fontweight="bold", pad=3)
    ax_a.text(0.985, 0.94,
              f"E6 validation: MAE {e6['mae_raw']:.2f} $\\rightarrow$ "
              f"{e6['mae_cur']:.2f} W m$^{{-2}}$ ({mae_red:+.1f}%)\n"
              f"skill {e6['skill_raw']:.2f} $\\rightarrow$ {e6['skill_cur']:.2f} "
              f"(n = {e6['n']:,})",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=7.5,
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                        edgecolor="#C9C9C5", alpha=1.0))
    ax_a.annotate("ramp event", xy=(xg[np.argmin(ghi)], float(ghi.min())),
                  xytext=(xg[np.argmin(ghi)] - 4, 420), fontsize=7,
                  color=INK_2, ha="right",
                  arrowprops=dict(arrowstyle="-", lw=0.4, color=INK_MUTED))

    # (b) phase-rectified curated — ALL frames at NN minutes ----------------
    ax_b.plot(fr.x, fr.feat, color=CUR_COLOR, lw=0.5, marker="o", ms=1.8,
              linestyle="-")
    ax_b.set_ylabel("Bright-pixel\nfraction")
    ax_b.set_title(f"(b) Phase-Rectified Curated — NN attach "
                   f"(median |$\\Delta t$| = {e6['medlag']:.0f} s, "
                   f"{e6['nn'] * 100:.0f}% of frames)",
                   loc="left", fontweight="bold", pad=3)

    # (c) unaligned raw — exact-match survivors + dropped frames ------------
    ex = fr[fr.exact == 1]
    dr = fr[fr.exact == 0]
    ax_c.plot(dr.x, dr.feat, linestyle="none", marker="x", ms=3,
              color=INK_MUTED, alpha=0.85, label=f"dropped ({len(dr)}/{len(fr)})")
    ax_c.plot(ex.x, ex.feat, linestyle="none", marker="o", ms=4.5,
              color=RAW_COLOR, zorder=5,
              label=f"exact-match survivors ({len(ex)})")
    ax_c.set_ylabel("Bright-pixel\nfraction")
    ax_c.set_title(f"(c) Unaligned Raw — exact-match ($\\Delta$ = 0 min): "
                   f"{len(ex)}/{len(fr)} frames survive in this window "
                   f"({e6['exact'] * 100:.0f}% dataset-wide)",
                   loc="left", fontweight="bold", pad=3)
    ax_c.legend(loc="upper right", frameon=False, fontsize=7,
                handletextpad=0.3, borderaxespad=0.2)

    # Rectification arrow: the dropped frame with the max |dt| lifted into (b).
    k = dr.dt_attach.idxmax()
    row = fr.loc[k]
    ax_b.annotate(f"rectified attach: |$\\Delta t$| = {row.dt_attach:.0f} s",
                  xy=(row.x, row.feat), xytext=(row.x + 12, row.feat + 0.028),
                  fontsize=7, color=INK_2,
                  arrowprops=dict(arrowstyle="->", lw=0.6, color=INK_2,
                                  connectionstyle="arc3,rad=-0.25"))

    ymax_f = fr.feat.max() * 1.45
    ax_b.set_ylim(0, ymax_f)
    ax_c.set_ylim(0, ymax_f)
    ax_c.set_xlim(0, 90)
    ax_c.set_xticks(np.arange(0, 91, 15))
    ax_c.set_xlabel("Minutes from window start, 21 Sep 2014 20:00 UTC")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300,
                    bbox_inches="tight")
        print(f"  {OUT_STEM}.{ext}")
    plt.close(fig)
    print(f"done -> {OUT_DIR / (OUT_STEM + '.png')}")


if __name__ == "__main__":
    main()
