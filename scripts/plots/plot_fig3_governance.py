"""Figure 3 — governance before/after: physical clamping and cross-modal attach.

ESSD/Copernicus figure for Section 2.3, REAL DATA ONLY (every loader raises
if a file is missing; nothing is synthesised):

  Panel (a)  Super-physical clearness clamping (Folsom, 1-min GHI)
    Raw       : publisher CSV waveform with real super-physical spikes
                (daytime kt > 1.5 at cos(zenith) > 0.3), highlighted.
    Curated   : same UTC window from the standardized parquet — kt clipped
                to [0, 1.5] (operators/quality.py), nights zeroed by solar
                astronomy (cos(theta_z) <= 0 => 0, quality_flag 3).
    The day is picked deterministically: complete 1440-row day maximising
    the midday super-physical fraction (dawn/dusk small-denominator
    artifacts excluded via the cos(zenith) > 0.3 filter).

  Panel (b)  Cross-modal temporal attach (Folsom sky camera x GHI)
    Raw       : EXACT-MATCH attach of second-cadence sky frames to the
                1-min tabular grid — non-:00 frames are orphans (E6
                measured exact_frac = 0.361), shown dropped in a greyed
                strip.
    Curated   : nearest-neighbour attach within tolerance (E6: nn_frac
                = 0.986, median |dt| = 1 s) — every frame attaches next to
                the GHI state it actually observed.
    The window is picked deterministically: 90-min daytime window with a
    real cloud ramp (max cloud-cover transition x GHI drop), >= 8 frames.

E6 numbers annotated on the figure are READ from the real experiment CSV
(figures_full/results/results_e6_alignment.csv), not hardcoded.

Design contract: scripts/plot_essd_figures.py (Arial 8.5 pt, 180 mm double
column, Wong/Okabe-Ito binding, light grid, bold panel letters +
sentence-case sub-labels, dual PDF + 300 dpi PNG).

Run (conda env solar_uni):
    python scripts/plot_fig3_governance.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from PIL import Image

# ===========================================================================
# MANDATORY PARAMETER BLOCK — the ONLY place sizes/colors/dimensions live
# ===========================================================================
TEXT_SIZE_PT = 8.5
SMALL_PT = 8.0
FONT_FAMILY = "Arial"
LINE_WIDTH_DATA = 0.6
WIDTH_MM = 180
HEIGHT_MM = 150
DPI_RASTER = 300
GRID_COLOR = "#E5E5E5"
GRID_WIDTH = 0.3
PANEL_LABEL_PT = 9.5

INK = "#0B0B0B"
INK_2 = "#52514E"
INK_MUTED = "#898781"
SPIKE_COLOR = "#D55E00"      # Wong vermillion — clamp threshold line
RAW_COLOR = "#D90429"        # red — ungoverned column, anomalies pop
CUR_COLOR = "#0F4C81"        # deep blue — governed column, regularity
CS_COLOR = "#52514E"         # reference lines
DROP_ALPHA = 0.35            # greyed thumbnails for dropped frames
THUMB_H = 0.140              # thumbnail height (axes fraction)
THUMB_W = 0.058              # thumbnail width (axes fraction)
Y_BAND_ATTACHED = 0.55       # bottom of attached-thumbnail band (axes frac)
Y_BAND_DROPPED = 0.75        # bottom of dropped-frame strip (axes frac)
Y_HEADROOM = 1.38            # ylim headroom so thumbnails never cover the peak

RAW_CSV = Path("D:/monica/workspace/solar_benchmark_dataset/Folsom_data/"
               "Folsom_irradiance.csv")
CUR_PARQ = _REPO_ROOT / "data" / "standardized" / "folsom" / "time_series.parquet"
SKY_ZARR = _REPO_ROOT / "data" / "standardized" / "folsom" / "sky_images_2014.zarr"
E6_CSV = _REPO_ROOT / "figures_full" / "results" / "results_e6_alignment.csv"
OUT_DIR = _REPO_ROOT / "scratch"
OUT_STEM = "fig03_governance"

#: Selection thresholds (deterministic scans).
COSZ_MIDDAY = 0.30           # dawn/dusk small-denominator exclusion
KT_SUPER = 1.5               # super-physical clearness threshold
WIN_MIN = 90                 # panel-(b) window length (minutes)
WIN_MIN_GHI_DROP = 300.0     # required GHI down-ramp across the window
WIN_MIN_FRAMES = 8
THUMB_MAX = 12               # displayed thumbnails per side (display cap)

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "mathtext.default": "rm",
    "font.size": TEXT_SIZE_PT,
    "axes.labelsize": TEXT_SIZE_PT,
    "axes.titlesize": PANEL_LABEL_PT,
    "xtick.labelsize": TEXT_SIZE_PT,
    "ytick.labelsize": TEXT_SIZE_PT,
    "legend.fontsize": SMALL_PT,
    "axes.linewidth": LINE_WIDTH_DATA,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": GRID_WIDTH,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})
mm = 1 / 25.4


# ===========================================================================
# Loaders — real files only
# ===========================================================================
def load_joined() -> pd.DataFrame:
    """Raw CSV GHI joined to curated clearsky/geometry (shared benchmark)."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(RAW_CSV)
    if not CUR_PARQ.exists():
        raise FileNotFoundError(CUR_PARQ)
    raw = pd.read_csv(RAW_CSV, parse_dates=["timeStamp"])
    raw = raw.set_index("timeStamp").sort_index()
    cur = pd.read_parquet(CUR_PARQ).set_index("timestamp_utc").sort_index()
    cs = cur[["ghi_clearsky", "cos_zenith", "ghi_actual", "clearness_index",
              "quality_flag"]].copy()
    cs.index = cs.index.tz_localize(None)
    j = raw.join(cs, how="left")
    j["kt_raw"] = j["ghi"] / (j["ghi_clearsky"] + 1e-6)
    return j


def load_e6() -> dict:
    """Real E6 alignment metrics for the folsom 1-h row (annotated on fig)."""
    if not E6_CSV.exists():
        raise FileNotFoundError(E6_CSV)
    df = pd.read_csv(E6_CSV)
    r = df[(df.dataset == "folsom") & (df.arm == "unaligned")].iloc[0]
    a = df[(df.dataset == "folsom") & (df.arm == "aligned")].iloc[0]
    return {"exact_frac": r.exact_frac, "nn_frac": r.nn_frac,
            "median_lag": r.median_lag_seconds, "p95_lag": r.p95_lag_seconds,
            "skill_unaligned": r.skill, "skill_aligned": a.skill}


# ===========================================================================
# Deterministic selections
# ===========================================================================
def pick_day_a(j: pd.DataFrame) -> pd.DateOffset:
    """Complete 1440-row day with the most dramatic midday super-physical
    spike: score lexicographically on (max midday kt, spike fraction),
    requiring a handful of spike samples so the effect is not a lone point."""
    best, best_score = None, (-1.0, -1.0)
    mid = (j["cos_zenith"] > COSZ_MIDDAY) & (j["kt_raw"] > KT_SUPER)
    for day, g in j.groupby(j.index.date):
        if len(g) != 1440 or g["ghi_clearsky"].isna().any():
            continue
        frac = float(mid.loc[g.index].mean())
        kmax = float(g.loc[g["cos_zenith"] > COSZ_MIDDAY, "kt_raw"].max())
        if frac < 0.003:                 # need >= ~5 spike samples
            continue
        if (kmax, frac) > best_score:
            best, best_score = day, (kmax, frac)
    if best is None or best_score[0] < 1.7:
        raise RuntimeError(f"no day with midday kt >= 1.7 over >=5 samples; "
                           f"best={best} {best_score}")
    print(f"[day-a] {best} midday kt>1.5 frac={best_score[1]:.3f} "
          f"max kt={best_score[0]:.2f}")
    return best


def decode_frame(g, i) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(bytes(g["images"][i]))))


def pick_window_b(j: pd.DataFrame):
    """90-min daytime window with a real cloud ramp and >= 8 sky frames.

    Scans zarr TIMESTAMPS only (no decode) for frame density + the joined
    table for the GHI down-ramp, then decodes the top-3 candidates' frames
    and picks the one with the largest cloud-cover transition.
    """
    g = zarr.open_group(str(SKY_ZARR), mode="r")
    ts = pd.to_datetime([str(t) for t in g["images_timestamps"][:]],
                        utc=True).tz_localize(None)   # naive UTC, like j.index
    step = pd.Timedelta(minutes=WIN_MIN)
    cands = []
    for t0 in pd.date_range("2014-01-15 16:00", "2014-12-15 01:00",
                            freq="2h"):
        t1 = t0 + step
        w = j.loc[t0:t1]
        if len(w) < WIN_MIN or w["ghi_clearsky"].max() < 100:
            continue
        drop = float(w["ghi"].iloc[:10].max() - w["ghi"].iloc[-10:].mean())
        if drop < WIN_MIN_GHI_DROP:
            continue
        m = (ts >= t0) & (ts < t1)
        n = int(m.sum())
        if n < WIN_MIN_FRAMES:
            continue
        cands.append((drop, t0, t1, np.flatnonzero(m)))
    cands.sort(key=lambda c: -c[0])
    if not cands:
        raise RuntimeError("no candidate window with GHI drop + frames")

    def _cloudy(img):
        r = img[..., 0].astype(np.float32)
        b = img[..., 2].astype(np.float32)
        return float(((b - r) / (b + r + 1e-6) < -0.1).mean())

    best = None
    for drop, t0, t1, idxs in cands[:3]:
        imgs = [(i, decode_frame(g, i), ts[i]) for i in idxs]
        cloudy = pd.Series({t: _cloudy(im) for _, im, t in imgs})
        spread = float(cloudy.max() - cloudy.min())
        score = (spread, drop)
        if best is None or score > best[0]:
            best = (score, t0, t1, imgs, cloudy)
    (_, t0, t1, imgs, cloudy) = best
    print(f"[win-b] {t0} -> {t1}  n_frames={len(imgs)}  "
          f"cloud spread={cloudy.max() - cloudy.min():.2f}  "
          f"ghi drop={j.loc[t0:t1, 'ghi'].max():.0f}->"
          f"{j.loc[t0:t1, 'ghi'].min():.0f}")
    return t0, t1, imgs, cloudy


# ===========================================================================
# Panels
# ===========================================================================
def draw_panel_a(fig, gs_raw, gs_cur, day, j: pd.DataFrame) -> None:
    """Clearness index, raw vs curated — the quantity the clamp governs.

    The physical clamp operates on the clearness INDEX (kt clip [0, 1.5]) and
    on night-zeroing (cos(theta_z) <= 0 => kt 0); the underlying GHI waveform
    is preserved verbatim (verified: daytime median |raw - curated| =
    0.0 W m-2), so kt is the honest before/after view.
    """
    t0 = pd.Timestamp(day) + pd.Timedelta(hours=8)      # UTC day window
    t1 = t0 + pd.Timedelta(days=1)
    w = j.loc[t0:t1]
    x = np.arange(len(w)) / 60.0                          # hours from 00 UTC
    kt_raw = w["kt_raw"].to_numpy(dtype=float).copy()
    kt_raw[w["ghi_clearsky"].to_numpy() < 10.0] = np.nan  # night undefined
    kt_cur = w["clearness_index"].to_numpy(dtype=float)
    cosz = w["cos_zenith"].to_numpy()
    night = cosz <= 0
    spike = (~night) & (cosz > COSZ_MIDDAY) & (kt_raw > KT_SUPER)

    for k, (ax, kt) in enumerate((
            (fig.add_subplot(gs_raw), kt_raw),
            (fig.add_subplot(gs_cur), kt_cur))):
        ax.axvspan(0, float(night.argmax()) / 60.0, color="#F4F4F2", zorder=0)
        ax.axvspan(float(len(w) - night[::-1].argmax()) / 60.0, 24.0,
                   color="#F4F4F2", zorder=0)
        ax.plot(x, kt, color=RAW_COLOR if k == 0 else CUR_COLOR, lw=0.7)
        ax.axhline(1.0, color=INK_MUTED, lw=0.5, ls=(0, (4, 2)), zorder=1)
        ax.axhline(1.5, color=SPIKE_COLOR, lw=0.6, ls=(0, (4, 2)), zorder=1)
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_ylim(0, 2.3)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xlabel("Hour (UTC)")
        if k == 0:
            ax.set_ylabel(r"$k_t$ (–), raw")
            ax.plot(x[spike], kt_raw[spike], linestyle="none", marker="o",
                    ms=1.6, color=INK, zorder=5)
            i_peak = int(np.nanargmax(np.where(spike, kt_raw, np.nan)))
            kmax = float(kt_raw[i_peak])
            ax.annotate(f"super-physical: $k_t$ up to {kmax:.1f}\n"
                        r"(clear-sky limit $k_t$ = 1 dashed)",
                        xy=(x[i_peak], kt_raw[i_peak]),
                        xytext=(float(np.nanmean(x[spike])) + 2.0, 2.05),
                        fontsize=7, color=INK, ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=INK))
            ax.text(0.985, 0.065, "night: $k_t$ undefined (gap)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=INK_MUTED)
            ax.set_title("Raw (ungoverned)", fontweight="bold", pad=3)
        else:
            ax.set_ylabel(r"$k_t$ (–), curated")
            n_clamp = int((w["clearness_index"] >= 1.5 - 1e-3).sum())
            ax.annotate(f"clamped at $k_t$ = 1.5\n({n_clamp} samples this day)",
                        xy=(0.03, 0.95), xycoords="axes fraction", ha="left",
                        va="top", fontsize=7, color=SPIKE_COLOR)
            ax.text(0.985, 0.065, r"night: $k_t$ zeroed ($\cos\theta_z \leq 0$)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=INK_MUTED)
            ax.set_title("Curated (governed)", fontweight="bold", pad=3)


def draw_panel_b(fig, gs_raw, gs_cur, t0, t1, j, imgs, cloudy, e6) -> None:
    w = j.loc[t0:t1]
    x0 = w.index[0]
    xf = (w.index - x0).total_seconds() / 60.0
    ghi = w["ghi"].to_numpy()
    # Headroom keeps the peak (0-10 min) below the thumbnail bands.
    ymax = max(650.0, np.nanmax(ghi) * Y_HEADROOM)

    # Exact-match = frame lands on the minute grid (second == 0).
    exact = [im for im in imgs if im[2].second == 0]
    orphan = [im for im in imgs if im[2].second != 0]

    for k, gs_cell in enumerate((gs_raw, gs_cur)):
        ax = fig.add_subplot(gs_cell)
        ax.plot(xf, ghi, color=RAW_COLOR if k == 0 else CUR_COLOR, lw=0.8)
        ax.set_xlim(0, WIN_MIN)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Minutes from window start (UTC)")
        if k == 0:
            ax.set_ylabel("GHI (W m$^{-2}$)")
            ax.set_title("Raw (ungoverned)", fontweight="bold", pad=3)
        else:
            ax.set_title("Curated (governed)", fontweight="bold", pad=3)

        # Attach statistics INSIDE the axes (top-left) — one clear title line
        # outside, numbers in the plot's blank top area.
        stat = (f"exact-match attach: {e6['exact_frac'] * 100:.0f}% of frames"
                if k == 0 else
                f"NN within tolerance: {e6['nn_frac'] * 100:.0f}%, "
                f"median |$\\Delta t$| = {e6['median_lag']:.0f} s")
        ax.text(0.015, 0.985, stat, transform=ax.transAxes, ha="left",
                va="top", fontsize=7, color=INK_2)

        shown = exact if k == 0 else imgs
        sub = shown if len(shown) <= THUMB_MAX else shown[::len(shown)
                                                          // THUMB_MAX][:THUMB_MAX]
        y_drop = Y_BAND_DROPPED              # dropped-frame strip (raw only)
        y_att = Y_BAND_ATTACHED               # attached-thumbnail band
        for i, im, ts in sub:
            xc = (ts - x0).total_seconds() / 60.0
            fx = xc / WIN_MIN
            axin = ax.inset_axes([fx - THUMB_W / 2, y_att, THUMB_W, THUMB_H])
            axin.imshow(im)
            axin.set_xticks([])
            axin.set_yticks([])
            for sp in axin.spines.values():
                sp.set_linewidth(0.3)
                sp.set_color(CS_COLOR)
            ax.plot([xc, xc], [0, ymax * (Y_BAND_ATTACHED - 0.02)],
                    color=INK_MUTED, lw=0.35, zorder=1)
        if k == 0:
            # Dropped frames: real thumbnails, greyed, in a strip — the
            # visible cost of exact-match attach.
            sub_o = (orphan if len(orphan) <= THUMB_MAX
                     else orphan[::len(orphan) // THUMB_MAX][:THUMB_MAX])
            for i, im, ts in sub_o:
                fx = (ts - x0).total_seconds() / 60.0 / WIN_MIN
                axin = ax.inset_axes([fx - THUMB_W / 2, y_drop, THUMB_W,
                                      THUMB_H])
                axin.imshow(im, alpha=DROP_ALPHA)
                axin.set_xticks([])
                axin.set_yticks([])
                for sp in axin.spines.values():
                    sp.set_linewidth(0.3)
                    sp.set_color("#C9C9C5")
            ax.text(0.015, y_drop - 0.035,
                    f"dropped: {len(orphan)}/{len(imgs)} frames "
                    "(off the 1-min grid)", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, color=INK_MUTED)
        # sky-state note tying thumbnails to the ramp
        ax.annotate("cloud ramp", xy=(xf[np.argmax(ghi)], ghi[np.argmax(ghi)]),
                    xytext=(xf[np.argmax(ghi)] - 2, ymax * 0.52),
                    fontsize=7, color=INK_2, ha="right",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=INK_MUTED))


# ===========================================================================
# Assemble
# ===========================================================================
def main() -> None:
    print("loading real files ...")
    j = load_joined()
    e6 = load_e6()
    print(f"[e6] exact={e6['exact_frac']:.3f} nn={e6['nn_frac']:.3f} "
          f"median_lag={e6['median_lag']}s skill {e6['skill_unaligned']:.2f} -> "
          f"{e6['skill_aligned']:.2f}")
    day = pick_day_a(j)
    t0, t1, imgs, cloudy = pick_window_b(j)

    # Governance verifications (printed, not assumed):
    v = j.loc[pd.Timestamp(day) + pd.Timedelta(hours=8):
              pd.Timestamp(day) + pd.Timedelta(days=1) + pd.Timedelta(hours=8)]
    night = v["cos_zenith"] <= 0
    assert (v.loc[night, "ghi_actual"] == 0).all(), "curated night not zeroed"
    assert (v["clearness_index"].dropna() <= 1.5 + 1e-3).all(), "kt clamp failed"
    day_agree = (v["ghi"].to_numpy() - v["ghi_actual"].to_numpy())
    print(f"[verify] curated night all-zero OK | kt<=1.5 OK | raw-vs-curated "
          f"daytime median |diff| = {np.median(np.abs(day_agree[~night])):.1f} "
          f"W/m2 | curated kt==1.5 rows: {(v['clearness_index'] >= 1.499).sum()}")

    fig = plt.figure(figsize=(WIDTH_MM * mm, HEIGHT_MM * mm))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25],
                          left=0.075, right=0.98, top=0.90, bottom=0.09,
                          wspace=0.14, hspace=0.52)
    draw_panel_a(fig, gs[0, 0], gs[0, 1], day, j)
    draw_panel_b(fig, gs[1, 0], gs[1, 1], t0, t1, j, imgs, cloudy, e6)

    fig.text(0.012, 0.985, "(a)", fontsize=PANEL_LABEL_PT, fontweight="bold",
             ha="left", va="top")
    fig.text(0.012, 0.560, "(b)", fontsize=PANEL_LABEL_PT, fontweight="bold",
             ha="left", va="top")
    fig.text(0.012, 0.955,
             f"(a) Clearness-index governance — Folsom, {day} (1-min $k_t$)",
             fontsize=SMALL_PT, color=INK_2, ha="left", va="top")
    fig.text(0.012, 0.535,
             f"(b) Cross-modal attach — sky camera × GHI, "
             f"{t0:%d %b %Y %H:%M}–{t1:%H:%M} UTC",
             fontsize=SMALL_PT, color=INK_2, ha="left", va="top")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=DPI_RASTER,
                    bbox_inches="tight")
        print(f"  {OUT_STEM}.{ext}")
    plt.close(fig)
    print(f"done -> {OUT_DIR / (OUT_STEM + '.png')}")


if __name__ == "__main__":
    main()
