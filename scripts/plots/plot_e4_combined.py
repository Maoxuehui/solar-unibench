"""E4 combined optical-flow figure — (a) extrapolation sequence over (b) decay.

Two-row panel merging the two core E4 diagnostics:
* **(a, top)** context → extrapolated → real future → uncertainty (lead K=1
  gallery pair, IR_016) — same content as ``fig_optflow_sequence.png``.
* **(b, bottom)** PSNR (dB) and SSIM vs lead K — same content as
  ``fig_optflow_psnr_decay.png``, side by side in one row.

Titles carry no "E4" prefix (panel letters + LaTeX caption own the explanation).
Data is recomputed from the standardized SunLake satellite cube via the E4
experiment's own evaluation path (identical numbers to results_e4_optflow.csv).

Run: python scripts/plot_e4_combined.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from solarunibench.experiments import e4_optflow as e4  # noqa: E402
from solarunibench.loader import SolarUniBenchLoader  # noqa: E402
from solarunibench.viz.style import PALETTE, set_style  # noqa: E402

OUT = _REPO_ROOT / "figures_full"
_GALLERY_CHANNEL = e4._GALLERY_CHANNEL


def _load_eval() -> tuple[dict, dict]:
    """Re-run the bounded E4 evaluation on the standardized SunLake cube."""
    loader = SolarUniBenchLoader(str(_REPO_ROOT / "data" / "standardized"))
    context = loader.get_tensor_cube(e4._DATASET, "satellite_context")
    future = loader.get_tensor_cube(e4._DATASET, "satellite_future")
    unc = loader.get_tensor_cube(e4._DATASET, "satellite_uncertainty")
    if context is None or future is None or unc is None:
        raise RuntimeError("no satellite extrapolation cube for sunlake")
    m = min(context.shape[0], e4._MAX_PAIRS_DEV + 1)
    metrics = e4._evaluate(
        np.asarray(context[:m], dtype=np.float32),
        np.asarray(future[:, : m - 1], dtype=np.float32),
        np.asarray(unc[:, : m - 1], dtype=np.float32),
        max_pairs=e4._MAX_PAIRS_DEV,
        nominal_levels=e4._NOMINAL_LEVELS,
    )
    return metrics["per_lead"], metrics["decay"]


def main() -> int:
    set_style("light")
    per_lead, decay = _load_eval()

    fig = plt.figure(figsize=(12.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.34, wspace=0.24)

    # -- (a) sequence row: 4 image panels spanning the full width ------------
    leads = [k for k in per_lead if per_lead[k]["pred"].shape[0] > 0]
    k = 1 if 1 in leads else leads[0]
    pair_idx = 0
    seed = per_lead[k]["seed"][pair_idx, _GALLERY_CHANNEL]
    pred = per_lead[k]["pred"][pair_idx, _GALLERY_CHANNEL]
    truth = per_lead[k]["truth"][pair_idx, _GALLERY_CHANNEL]
    unc = per_lead[k]["unc"][pair_idx]

    sub = gs[0].subgridspec(1, 4, wspace=0.06)
    axes_seq = [fig.add_subplot(s) for s in sub]
    titles = ("Context (seed)", f"Extrapolated (K={k})", "Real future",
              "Reliability (uncertainty)")
    imgs = (seed, pred, truth, unc)
    cmaps = ("inferno", "inferno", "inferno", "viridis")
    for ax, im, t, cm in zip(axes_seq, imgs, titles, cmaps):
        m_ = ax.imshow(im, cmap=cm)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(m_, ax=ax, fraction=0.046, pad=0.04)

    # -- (b) decay row: PSNR | SSIM ------------------------------------------
    psnr = np.asarray(decay["psnr_per_lead"], dtype=float)
    ssim = np.asarray(decay["ssim_per_lead"], dtype=float)
    kk = np.arange(1, psnr.size + 1)
    ax_l = fig.add_subplot(gs[1, 0])
    ax_l.plot(kk, psnr, color=PALETTE[0], marker="o", markersize=6, linewidth=2.0)
    ax_l.set_xlabel("Lead time K (steps)")
    ax_l.set_ylabel("PSNR (dB)")
    ax_l.set_title("Extrapolation quality vs lead time — PSNR (higher = better)",
                   fontsize=10)
    ax_l.grid(True, alpha=0.3)
    ax_r = fig.add_subplot(gs[1, 1])
    ax_r.plot(kk, ssim, color=PALETTE[1], marker="s", markersize=6, linewidth=2.0)
    ax_r.set_xlabel("Lead time K (steps)")
    ax_r.set_ylabel("SSIM")
    ax_r.set_title("Extrapolation quality vs lead time — SSIM (1.0 = identical)",
                   fontsize=10)
    ax_r.grid(True, alpha=0.3)

    fig.text(0.008, 0.975, "(a)", fontsize=13, fontweight="bold")
    fig.text(0.008, 0.455, "(b)", fontsize=13, fontweight="bold")

    out = OUT / "fig_optflow_combined"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png (+ .pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
