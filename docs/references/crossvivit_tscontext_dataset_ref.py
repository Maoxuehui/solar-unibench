"""REFERENCE ONLY (provided by user 2026-07-17) — CrossViViT TSContextDataset
+ authoritative optical-flow forward-splat extrapolation.

NOT imported by the package. Two uses:
  1. Optical-flow revision (operators/optical_flow.py): replace the simple
     cv2.remap(minus-sign) warp with `warp_image` + `_bilinear_forward_splat` —
     forward flow F_{t-1->t} on the prev-frame grid; sample I_t at q=p+F, then
     forward-splat to r=p+(steps+1)F; collisions averaged by bilinear weights;
     holes (disocclusions) filled by nearest covered pixel (distanceTransform).
     Do NOT negate the flow and do NOT pre-multiply by t — `warp_image(steps=)`
     does the Lagrangian extrapolation internally.
  2. Phase 6b CrossViViT integration: the deeplake TSContextDataset pattern
     (context/optflow/ts channels, seq_len/label_len/pred_len, per-station stats).

Source: CrossViViT repo (gitbooo/CrossViVit) data_provider, adapted by user.
"""
import json
import os
from typing import Dict, List, Tuple, Union

import deeplake
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import cv2

from data_provider.dataset_utils import calculate_possible_starts

deeplake.constants.WANDB_INTEGRATION_ENABLED = False


def build_transform(input_size):
    transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def build_transform_tensor(input_size):
    def transform_tensor(img_tensor):
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = F.interpolate(img_tensor, size=(input_size, input_size), mode='bicubic', align_corners=False)
        img_tensor = img_tensor.squeeze(0)
        return img_tensor
    return transform_tensor


def dynamic_preprocess_tensor(img_tensor, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    C, orig_height, orig_width = img_tensor.shape
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    img_tensor_4d = img_tensor.unsqueeze(0)
    resized_img = F.interpolate(img_tensor_4d, size=(target_height, target_width), mode='bicubic', align_corners=False)
    resized_img = resized_img.squeeze(0)
    processed_images = []
    num_cols = target_width // image_size
    num_rows = target_height // image_size
    for i in range(blocks):
        row = i // num_cols
        col = i % num_cols
        tile = resized_img[:, row * image_size:(row + 1) * image_size, col * image_size:(col + 1) * image_size]
        processed_images.append(tile)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail = F.interpolate(img_tensor_4d, size=(image_size, image_size), mode='bicubic', align_corners=False).squeeze(0)
        processed_images.append(thumbnail)
    return processed_images


def load_image_tensor(img_tensor, input_size=448, max_num=12):
    transform_tensor = build_transform_tensor(input_size=input_size)
    images = dynamic_preprocess_tensor(img_tensor, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform_tensor(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def _bilinear_forward_splat(values, dst_x, dst_y, valid_mask=None):
    """Bilinearly splat source-grid values onto destination coordinates.

    values: [H, W]; dst_x, dst_y: [H, W] continuous destination coords.
    Returns (output [H,W] collision-normalized, weight_sum [H,W]; 0 => hole).
    """
    values = np.asarray(values, dtype=np.float32)
    dst_x = np.asarray(dst_x, dtype=np.float32)
    dst_y = np.asarray(dst_y, dtype=np.float32)
    h, w = values.shape
    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
    finite = np.isfinite(values) & np.isfinite(dst_x) & np.isfinite(dst_y) & valid_mask
    x0 = np.floor(dst_x).astype(np.int64)
    y0 = np.floor(dst_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = dst_x - x0
    wy = dst_y - y0
    accum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    def add_corner(x_idx, y_idx, weight):
        valid = finite & (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h) & (weight > 0)
        if not np.any(valid):
            return
        flat_idx = y_idx[valid] * w + x_idx[valid]
        np.add.at(accum.ravel(), flat_idx, values[valid] * weight[valid])
        np.add.at(weight_sum.ravel(), flat_idx, weight[valid])

    add_corner(x0, y0, (1.0 - wx) * (1.0 - wy))
    add_corner(x1, y0, wx * (1.0 - wy))
    add_corner(x0, y1, (1.0 - wx) * wy)
    add_corner(x1, y1, wx * wy)
    output = np.zeros_like(accum)
    covered = weight_sum > 1e-6
    output[covered] = accum[covered] / weight_sum[covered]
    return output, weight_sum


def warp_image(image, flow_prev_to_curr, steps=1):
    """Extrapolate I_t with the saved forward flow F_{t-1->t}.

    Forward flow on the prev-frame grid: q = p + F_{t-1->t}(p). Constant
    Lagrangian velocity => the point is at r = p + (steps+1)*F at t+steps.
    Sample I_t at q (cv2.remap), then forward-splat to r. Collisions averaged
    by bilinear weights; holes (disocclusions) filled from nearest covered pixel.
    Do NOT negate the flow; do NOT pre-multiply by t.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    image = np.asarray(image, dtype=np.float32)
    flow_prev_to_curr = np.asarray(flow_prev_to_curr, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"image must have shape [H, W], got {image.shape}")
    if flow_prev_to_curr.shape != image.shape + (2,):
        raise ValueError("flow_prev_to_curr must have shape [H, W, 2], "
                         f"got {flow_prev_to_curr.shape} for image {image.shape}")
    h, w = image.shape
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    u = flow_prev_to_curr[..., 0]
    v = flow_prev_to_curr[..., 1]
    current_x = x + u
    current_y = y + v
    valid_current = (np.isfinite(u) & np.isfinite(v) & (current_x >= 0.0) & (current_x <= w - 1)
                     & (current_y >= 0.0) & (current_y <= h - 1))
    current_values = cv2.remap(image, current_x, current_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    future_x = x + (steps + 1) * u
    future_y = y + (steps + 1) * v
    predicted, weight_sum = _bilinear_forward_splat(current_values, future_x, future_y, valid_mask=valid_current)
    covered = weight_sum > 1e-6
    if np.all(covered):
        return predicted
    if not np.any(covered):
        return image.copy()
    hole_map = (~covered).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(hole_map, distanceType=cv2.DIST_L2,
                                                maskSize=cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL)
    label_values = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    label_values[labels[covered]] = predicted[covered]
    predicted[~covered] = label_values[labels[~covered]]
    return predicted.astype(np.float32, copy=False)


class TSContextDataset(Dataset):
    def __init__(self, data_dir, stats_path, context_channels, optflow_channels, ts_channels,
                 ts_target_channels, years, stations, mode="train", use_target=True,
                 image_size=None, crop=None, seq_len=24 * 2, label_len=12, pred_len=24 * 2,
                 timestamp_dir="", timestamp_range="20082022", token_len=0, tokenizer=None,
                 num_image_token=0):
        self.data_dir = data_dir
        self.stats_path = stats_path
        self.context_channels = context_channels
        self.optflow_channels = optflow_channels
        self.ts_channels = ts_channels
        self.ts_target_channels = ts_target_channels
        self.years = years[mode]
        self.stations = stations[mode]
        self.crop = crop
        self.mode = mode
        self.image_size = tuple(image_size) if image_size is not None else None
        self.use_target = use_target
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.timestamp_dir = timestamp_dir
        self.timestamp_range = timestamp_range
        self.token_len = token_len
        self.tokenizer = tokenizer
        self.input_ids = None
        self.num_image_token = num_image_token
        self.n_samples = []
        self.year_mapping = {}
        if self.stats_path is not None:
            with open(self.stats_path, "r") as fp:
                self.stats = json.load(fp)
        else:
            self.stats = None
        self.deeplake_ds = deeplake.load(self.data_dir, read_only=True)
        for year in self.years:
            for station in self.stations:
                possible_starts_context, possible_starts_station = calculate_possible_starts(
                    self.deeplake_ds[f"{year}/context/time_utc"].numpy()[0],
                    self.deeplake_ds[f"{year}/{station}/time_utc"].numpy()[0],
                    frames_total=self.seq_len + self.pred_len,
                )
                n_in_group = len(possible_starts_context)
                self.n_samples.append(n_in_group)
                for i in range(self.n_samples[-1]):
                    step_id_station = possible_starts_station[i]
                    step_id_context = possible_starts_context[i]
                    self.year_mapping[i + sum(self.n_samples[:-1])] = (str(year), station, step_id_station, step_id_context)
        if self.timestamp_dir:
            self.timestamp_emb = {}
            for station in self.stations:
                timestamps_path = os.path.join(self.timestamp_dir, f"cvv_timestamps_forecast_{station.split('_')[-1]}_{self.timestamp_range}.json")
                timestamp_embs_path = os.path.join(self.timestamp_dir, f"cvv_timestamps_forecast_{station.split('_')[-1]}_{self.timestamp_range}.pt")
                if os.path.exists(timestamps_path) and os.path.exists(timestamp_embs_path):
                    timestamps = json.load(open(timestamps_path))
                    timestamp_embs = torch.load(timestamp_embs_path)
                    timestamp_to_idx = {t: i for i, t in enumerate(timestamps)}
                    self.timestamp_emb[station] = {"timestamp_embs": timestamp_embs, "timestamp_to_idx": timestamp_to_idx}
                else:
                    self.timestamp_emb[station] = None
        else:
            self.timestamp_emb = {}
        self._mean = {}
        self._std = {}
        self._context_channel_ids = None
        self._optflow_channel_ids = None
        self._ts_channel_ids = None
        self._ts_target_channel_ids = None

    # ... (get_stats / get_channel_ids / __getitem__ elided — see user paste for full deeplake sampling + warp_image usage) ...
