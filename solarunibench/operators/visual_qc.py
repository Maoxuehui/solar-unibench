"""Visual-modality quality control (Q0-Q3) for ASI sky cameras and satellite
patches — the imagery-side counterpart of ``operators/quality.py`` (tabular).

Four gates, one pipeline, two modes:

  Q0 Radiometric physical bounds
      * per-pixel bounds check against the channel's physical range
        ([0, 255] uint8 cameras, [0, 1] float satellite) — out-of-range pixels
        are counted (dry-run) or clipped + flagged (apply);
      * NIGHT GUARD on ratio features: when the solar zenith angle
        theta_z >= 90 deg the frame is astronomical night — ratio features
        such as NRBR = (B-R)/(B+R) must NOT be computed (denominator -> 0
        divergence), so ``nrbr_guard`` returns NaN for the whole frame.

  Q1 Camera glare & overexposure (ASI only)
      * geometric glare mask: the sun disc is projected into pixel space with
        an EQUIDISTANT FISHEYE assumption (image centre = zenith, r_px =
        (theta_z / 90 deg) * half-width, azimuth from north, clockwise,
        east-right image convention) and a circular mask of ``glare_radius_px``
        is drawn (cv2.circle). FALLBACK: when geometry is missing/degenerate
        (zenith >= 90, NaN azimuth) the brightest saturated-blob centroid is
        used instead;
      * overexposed dead frame: uint8-scale mean > 245 AND std < 5.0 ->
        frame-level reject.

  Q2 Extreme-zenith filtering (satellite only)
      * theta_z > 85 deg -> dawn/dusk low-sun invalid frame (frame-level
        flag; excluded from downstream optical flow).

  Q3 Variance drop-frame check (all modalities)
      * Var(frame_t - frame_{t-1}) on frames normalised to float [0, 1]
        (uint8 inputs are divided by 255 first, so the thresholds are
        format-invariant): var < 1e-4 -> frozen/stuck frame;
        var > 1000.0 -> salt-and-pepper transport noise. Both are
        frame-level rejects.

Modes (``config={'mode': ...}``):
  * ``dry_run`` (default): NOTHING is modified — masks/flags/per-frame stats
    are returned and aggregate counters accumulate for
    :meth:`VisualQCPipeline.print_diagnostics_report`.
  * ``apply``: float tensors get invalid pixels/frames set to NaN; uint8
    images get them set to 0 AND a same-shape bool ``validity_mask`` is
    returned so downstream models can select. Frame-level rejects (Q1
    overexposure, Q3 drop-frames) wipe the whole frame.

Design contract (ESSD/honest-NaN, mirroring ``quality.py``): never fabricate
values — night NRBR is NaN, not 0; apply only where a gate actually fired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

__all__ = ["VisualQCPipeline", "FrameQCResult"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class FrameQCResult:
    """Per-frame QC outcome (both modes; ``frame`` is untouched in dry-run)."""
    frame: np.ndarray                    # original (dry_run) or governed (apply)
    validity_mask: np.ndarray            # bool, True = valid pixel (uint8 path)
    glare_mask: np.ndarray               # bool, True = glare pixel (Q1)
    flags: dict = field(default_factory=dict)   # q0_bounds/q1_overexposed/...
    sun_uv: tuple | None = None          # geometric sun pixel centre (or fallback)
    sun_source: str = "none"             # 'fisheye' | 'brightest' | 'none'
    frame_nan: bool = False              # whole frame invalidated (apply)


def _as_float01(frame: np.ndarray) -> np.ndarray:
    """Normalise any input to float [0, 1] (uint8 -> /255; clip negatives)."""
    f = np.asarray(frame, dtype=np.float32)
    if f.size and f.max() > 1.5:                      # uint8-scale content
        f = f / 255.0
    return f


def _gray01(frame: np.ndarray) -> np.ndarray:
    """Grayscale in [0, 1] from (H,W), (H,W,C) uint8/float inputs.

    3-channel input uses Rec. 601 luma weights; multispectral stacks with a
    different channel count (e.g. 11-channel SEVIRI patches) fall back to the
    channel mean — a format-generic intensity for the Q3 variance gate."""
    f = _as_float01(frame)
    if f.ndim == 3:
        if f.shape[-1] == 3:
            f = f @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        else:
            f = f.mean(axis=-1)
    return np.clip(f, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class VisualQCPipeline:
    """Four-gate visual QC with a diagnostics collector.

    Parameters (all optional in ``config``; defaults per the QC spec):
      mode            'dry_run' | 'apply'                      (dry_run)
      modality        'asi' | 'satellite' | 'auto'             (auto: uint8->asi)
      glare_radius_px Q1 solar-glare disc radius                (25)
      overexp_mean    Q1 overexposure mean threshold, uint8 eq. (245)
      overexp_std     Q1 overexposure std threshold, uint8 eq.  (5.0)
      bounds          Q0 physical range as (lo, hi)             (None=auto)
      zenith_max_deg  Q2 dawn/dusk zenith threshold             (85)
      var_frozen      Q3 minimum frame-diff variance            (1e-4)
      var_noisy       Q3 maximum frame-diff variance            (1000.0)
      invalid_value_u8 apply-mode value for invalid uint8 px    (0)
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.mode = cfg.get("mode", "dry_run")
        if self.mode not in ("dry_run", "apply"):
            raise ValueError(f"mode must be 'dry_run' or 'apply', "
                             f"got {self.mode!r}")
        self.modality = cfg.get("modality", "auto")
        self.glare_radius_px = int(cfg.get("glare_radius_px", 25))
        self.overexp_mean = float(cfg.get("overexp_mean", 245.0))
        self.overexp_std = float(cfg.get("overexp_std", 5.0))
        self.bounds = cfg.get("bounds")               # None -> per-dtype auto
        self.zenith_max_deg = float(cfg.get("zenith_max_deg", 85.0))
        self.var_frozen = float(cfg.get("var_frozen", 1e-4))
        self.var_noisy = float(cfg.get("var_noisy", 1000.0))
        self.invalid_u8 = cfg.get("invalid_value_u8", 0)
        #: fixed-mount camera calibration for the geometric sun projection
        self.camera_rotation_deg = float(cfg.get("camera_rotation_deg", 0.0))
        self.camera_radius_scale = float(cfg.get("camera_radius_scale", 1.0))
        self.reset_stats()

    # -- stats collector ----------------------------------------------------
    def reset_stats(self) -> None:
        self.n_frames = 0
        self.q0_oob_pixels = 0
        self.q0_total_pixels = 0
        self.q1_glare_pixels = 0
        self.q1_total_pixels = 0
        self.q1_overexposed = 0
        self.q2_high_zenith = 0        # theta_z > 85 deg (twilight + night)
        self.q2_twilight = 0           # 85 < theta_z <= 90 (dawn/dusk band)
        self.q3_drop = 0

    def print_diagnostics_report(self) -> str:
        """Emit the six-line diagnostics report (also returned as text)."""
        pct = lambda a, b: (100.0 * a / b) if b else float("nan")
        lines = [
            f"Visual QC diagnostics - mode={self.mode}, "
            f"frames={self.n_frames}",
            f"Total Processed Frames:        {self.n_frames}",
            f"Q0 Out-of-bounds Ratio (%):    "
            f"{pct(self.q0_oob_pixels, self.q0_total_pixels):.4f}",
            f"Q1 Glare Area Ratio (%):       "
            f"{pct(self.q1_glare_pixels, self.q1_total_pixels):.4f}",
            f"Q1 Overexposed Ratio (%):      "
            f"{pct(self.q1_overexposed, self.n_frames):.4f}",
            f"Q2 High-Zenith Ratio (%):      "
            f"{pct(self.q2_high_zenith, self.n_frames):.4f}",
            f"Q2 Twilight-Band Ratio (%):    "
            f"{pct(self.q2_twilight, self.n_frames):.4f}",
            f"Q3 Drop-Frame Ratio (%):       "
            f"{pct(self.q3_drop, self.n_frames):.4f}",
        ]
        report = "\n".join(lines)
        print(report)
        return report

    # -- Q1 helpers ----------------------------------------------------------
    def sun_pixel_uv(self, shape, solar_azimuth_deg=None,
                     solar_zenith_deg=None, frame=None) -> tuple[tuple, str]:
        """Sun pixel centre: equidistant fisheye projection (primary) or the
        brightest saturated-blob centroid (fallback). Returns ((u, v), src).

        Camera calibration knobs (fixed-mount ASI, config):
          camera_rotation_deg  clockwise rotation of the azimuth->image
                              direction (mounting orientation; e.g. Folsom's
                              camera is rotated ~177 deg vs north-up-east-right)
          camera_radius_scale  r_px scale (<1 when the lens maps the hemisphere
                              inside the frame, e.g. Folsom ~0.80)
        """
        h, w = shape[:2]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r_max = min(h, w) / 2.0
        if (solar_zenith_deg is not None and np.isfinite(solar_zenith_deg)
                and 0.0 <= solar_zenith_deg < 90.0
                and solar_azimuth_deg is not None
                and np.isfinite(solar_azimuth_deg)):
            r = (solar_zenith_deg / 90.0) * r_max * self.camera_radius_scale
            az = np.deg2rad(solar_azimuth_deg + self.camera_rotation_deg)
            return (cx + r * np.sin(az), cy - r * np.cos(az)), "fisheye"
        if frame is not None:                          # fallback: brightest blob
            g = (_gray01(frame) * 255.0).astype(np.uint8)
            sat = g >= 250
            if sat.any():
                M = cv2.moments(sat.astype(np.uint8), binaryImage=True)
                if M["m00"] > 0:
                    return ((M["m10"] / M["m00"], M["m01"] / M["m00"]),
                            "brightest")
        return (cx, cy), "none"

    # -- gates ----------------------------------------------------------------
    def q0_bounds(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """(out_of_bounds_mask, oob_pixel_ratio) against the physical range."""
        f = np.asarray(frame)
        lo, hi = self.bounds if self.bounds else (
            (0.0, 255.0) if f.dtype == np.uint8 else (0.0, 1.0))
        oob = (f < lo) | (f > hi)
        return oob, float(oob.mean()) if oob.size else 0.0

    def q0_nrbr_guard(self, nrbr: np.ndarray | None,
                      solar_zenith_deg: float | None) -> np.ndarray | None:
        """Night guard for ratio features: theta_z >= 90 deg -> all-NaN."""
        if nrbr is None:
            return None
        if solar_zenith_deg is not None and np.isfinite(solar_zenith_deg) \
                and solar_zenith_deg >= 90.0:
            return np.full(nrbr.shape, np.nan, dtype=np.float32)
        return nrbr

    def q1_glare_mask(self, shape, sun_uv: tuple) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(round(sun_uv[0])), int(round(sun_uv[1]))),
                   self.glare_radius_px, 1, thickness=-1)
        return mask.astype(bool)

    def check_overexposed(self, frame: np.ndarray) -> bool:
        g = (_gray01(frame) * 255.0)
        return bool(g.mean() > self.overexp_mean and g.std() < self.overexp_std)

    def check_high_zenith(self, solar_zenith_deg: float | None) -> bool:
        return bool(solar_zenith_deg is not None
                    and np.isfinite(solar_zenith_deg)
                    and solar_zenith_deg > self.zenith_max_deg)

    def q3_frame_diff_variance(self, frame: np.ndarray,
                               prev_frame: np.ndarray) -> float | None:
        """Var(frame_t - frame_{t-1}) on [0,1]-normalised grayscale."""
        if prev_frame is None:
            return None
        d = _gray01(frame) - _gray01(prev_frame)
        return float(d.var())

    # -- main entry ------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, *,
                      prev_frame: np.ndarray | None = None,
                      solar_azimuth_deg: float | None = None,
                      solar_zenith_deg: float | None = None,
                      cos_zenith: float | None = None,
                      modality: str | None = None) -> FrameQCResult:
        """Run Q0-Q3 on one frame. ``cos_zenith`` (table column) is accepted
        as the geometry source when the zenith angle itself is unavailable."""
        if cos_zenith is not None and solar_zenith_deg is None:
            solar_zenith_deg = float(np.degrees(np.arccos(
                np.clip(float(cos_zenith), -1.0, 1.0))))
        modality = modality or self.modality
        if modality == "auto":
            modality = "asi" if np.asarray(frame).dtype == np.uint8 \
                else "satellite"

        f = np.asarray(frame)
        h, w = f.shape[:2]
        self.n_frames += 1

        # Q0 — radiometric bounds
        oob, oob_ratio = self.q0_bounds(f)
        self.q0_oob_pixels += int(oob.sum())
        self.q0_total_pixels += int(oob.size)

        # Q1 — glare mask + overexposure (ASI only)
        if modality == "asi":
            sun_uv, sun_src = self.sun_pixel_uv(
                f.shape, solar_azimuth_deg, solar_zenith_deg, frame=f)
            glare = self.q1_glare_mask(f.shape, sun_uv)
            overexposed = self.check_overexposed(f)
        else:
            sun_uv, sun_src, glare, overexposed = None, "none", \
                np.zeros((h, w), dtype=bool), False
        self.q1_glare_pixels += int(glare.sum())
        self.q1_total_pixels += int(glare.size)
        if overexposed:
            self.q1_overexposed += 1

        # Q2 — extreme zenith (satellite gate, but recorded for any input)
        high_zen = self.check_high_zenith(solar_zenith_deg) \
            if modality == "satellite" else False
        if high_zen:
            self.q2_high_zenith += 1
            # NOTE: the standardized tables CLAMP night cos_zenith at 0.0
            # (z == 90.0 exactly), so feed satellite QC an ANALYTIC zenith for
            # the twilight band; the strict < 90 guard excludes clamped nights.
            if solar_zenith_deg is not None and solar_zenith_deg < 90.0:
                self.q2_twilight += 1

        # Q3 — drop-frame variance
        var = self.q3_frame_diff_variance(f, prev_frame)
        drop = var is not None and (var < self.var_frozen
                                    or var > self.var_noisy)
        if drop:
            self.q3_drop += 1

        flags = {"q0_oob_ratio": oob_ratio, "q1_overexposed": overexposed,
                 "q2_high_zenith": high_zen, "q3_var": var, "q3_drop": drop}

        # -- apply mode -----------------------------------------------------
        out = f.copy() if self.mode == "apply" else f
        # pixel-level (H, W) validity: any-channel OOB invalidates the pixel
        valid2d = ~oob.any(axis=-1) if oob.ndim == 3 else ~oob
        frame_nan = bool(overexposed or drop)
        if self.mode == "apply":
            if frame_nan:
                if out.dtype == np.uint8:
                    out[...] = self.invalid_u8
                else:
                    out = out.astype(np.float32)
                    out[...] = np.nan
                valid = np.zeros((h, w), dtype=bool)
            else:
                if out.dtype == np.uint8:
                    if oob.any():
                        out[oob] = self.invalid_u8
                    if modality == "asi" and glare.any():
                        out[glare] = self.invalid_u8
                    valid = valid2d & ~glare if modality == "asi" else valid2d
                else:
                    out = out.astype(np.float32)
                    out[oob] = np.nan
                    valid = valid2d
        else:
            valid = valid2d & ~glare if modality == "asi" else valid2d
        return FrameQCResult(frame=out, validity_mask=valid, glare_mask=glare,
                             flags=flags, sun_uv=sun_uv, sun_source=sun_src,
                             frame_nan=frame_nan)

    def process_frames(self, frames: Iterable[np.ndarray], *,
                       solar_geometry: Iterable[tuple | None] | None = None,
                       modality: str | None = None) -> list[FrameQCResult]:
        """Batch convenience: sequential frames share Q3 differencing.
        ``solar_geometry`` yields per-frame (azimuth_deg, zenith_deg | None)."""
        results, prev = [], None
        geo = iter(solar_geometry or [])
        for fr in frames:
            try:
                az, zen = next(geo)
            except StopIteration:
                az, zen = None, None
            results.append(self.process_frame(
                fr, prev_frame=prev, solar_azimuth_deg=az,
                solar_zenith_deg=zen, modality=modality))
            prev = fr
        return results
