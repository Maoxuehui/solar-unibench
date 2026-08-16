"""CrossViViT-inspired lightweight multimodal baseline — tabular ↔ image
cross-attention.

Inspired by Boussif et al. 2023, "Improving day-ahead Solar Irradiance Time
Series Forecasting by Leveraging Spatio-Temporal Context" (arXiv:2306.01112;
official code: https://github.com/gitbooo/CrossViVit). This is NOT the
official 145M RoCrossViViT — it is a small self-contained variant built for
the unified benchmark's shared training budget (see the comparison notes in
``docs/references/crossvivit_tscontext_dataset_ref.py`` for the official
data-side pattern that informed the windowing design). The model FUSES two
modalities under the :class:`ForecastModel` interface:

* **Tabular branch** — patch-tokenizes ``X_past[n, lb, Fp]`` along the lookback
  axis (PatchTST-style; each patch of ``patch_len`` timesteps × all channels is
  linearly projected to ``d_model``) → a sequence of tabular tokens.
* **Future branch** — the known astronomical trajectory
  ``X_future[n, horizon, Ff]`` (ghi_clearsky / cos_zenith) is linearly embedded
  per future step into ``horizon`` extra tokens appended to the tabular
  sequence (official CrossViViT feeds time coordinates into both modalities;
  this is the benchmark's equivalent — the model can attend to what the sun
  WILL do, not just what it did).
* **Image branch** — the per-window sky-image cube ``images[n, lb, 4, H, W]``
  is resized to ``img_size`` (via :func:`solarunibench.imaging.resize.resize_frames`,
  so mixed native sizes like Folsom-256 / SKIPPD-64 batch together), frame-
  subsampled to ``img_frame_stride``, conv-patch-embedded per retained frame,
  and each token receives an additive learnable frame-position embedding so
  the cloud MOTION (frame ordering) survives — no mean-pool over the lookback
  (the official model keeps the temporal axis; mean-pooling it was the main
  structural loss of the previous version).
* **Bidirectional cross-attention** — ``depth`` blocks where tabular tokens
  attend to image tokens AND image tokens attend to tabular tokens; the pooled
  fused representation feeds a linear horizon head → ``predict`` returns
  ``(n, horizon)`` in one pass.

Design notes
------------
* **Graceful tabular fallback.** When ``images`` is ``None`` the image-token
  sequence collapses to the tabular tokens, so cross-attention reduces to
  tabular self-attention and the model still produces ``(n, horizon)``. This
  keeps it runnable on purely tabular datasets (PV / GEFCom).
* **SMALL runnable variant.** The defaults (``d_model=64``, ``depth=2``,
  ``img_size=32``) are a deliberately small model so the benchmark suite can
  actually train it. The full 145M CrossViViT (large RoPE, per-station stats,
  deeplake context/optflow/ts channels) is a Phase-8 training-budget decision
  (spec §15); this is the Phase-6b runnable variant. See
  ``docs/references/crossvivit_tscontext_dataset_ref.py`` for the reference
  ``TSContextDataset`` pattern (context + optical-flow-extrapolated future
  frames) the full model consumes.
* **Scaled features in, no re-scaling inside.** The pipeline's
  :class:`PhysicalFeatureScaler` (default-on) has already normalized the
  tabular block before it reaches the model; do not re-scale.
* **Determinism + Honest NaN.** Same contract as the LSTM/PatchTST baselines:
  a fixed ``seed`` (Python/NumPy/Torch + cuDNN deterministic) makes fits
  reproducible; NaN/inf features are sanitized to 0 (images too) so a stray
  NaN cannot poison training or emit non-finite forecasts.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..core import ForecastModel
from ._torch_base import TorchForecastMixin, fit_loop, train_kwargs

# Valid tokenization variants. ``crossvivit`` is the default bidirectional
# cross-attention model; ``tabular`` disables the image branch entirely at
# construction (a convenience ablation — equivalent to never passing images).
_VARIANTS: tuple[str, ...] = ("crossvivit", "tabular")


class _FFN(nn.Module):
    """Two-layer feed-forward network (Transformer-style, GELU)."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class _CrossAttnBlock(nn.Module):
    """One bidirectional cross-attention block.

    Tabular tokens attend to image tokens (query=tab, kv=img) then image tokens
    attend back to the updated tabular tokens (query=img, kv=tab); each sub-layer
    has its own residual + LayerNorm + FFN (post-LN style).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.tab_mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.img_mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_tab1 = nn.LayerNorm(d_model)
        self.ln_tab2 = nn.LayerNorm(d_model)
        self.ln_img1 = nn.LayerNorm(d_model)
        self.ln_img2 = nn.LayerNorm(d_model)
        self.ffn_tab = _FFN(d_model, dropout)
        self.ffn_img = _FFN(d_model, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, tab: torch.Tensor, img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ta, _ = self.tab_mha(tab, img, img, need_weights=False)
        tab = self.ln_tab1(tab + self.drop(ta))
        tab = self.ln_tab2(tab + self.ffn_tab(tab))
        ia, _ = self.img_mha(img, tab, tab, need_weights=False)
        img = self.ln_img1(img + self.drop(ia))
        img = self.ln_img2(img + self.ffn_img(img))
        return tab, img


class _CrossViViTNet(nn.Module):
    """Tabular-patch + future + image-conv tokenizers, ``depth`` cross-attn
    blocks, head.

    The image conv branch and the future-token projection are ALWAYS built
    (cheap, fixed-size from ``img_size`` / ``img_patch`` / ``horizon``); they
    are used only when ``forward`` receives the corresponding tensor, so
    fit/predict are robust to the presence or absence of images / X_future
    (and to fit-without / predict-with mismatches).
    """

    def __init__(
        self,
        lookback: int,
        n_channels: int,
        horizon: int,
        patch_len: int,
        stride: int,
        img_size: int,
        img_patch: int,
        in_channels: int,
        d_model: int,
        n_heads: int,
        depth: int,
        dropout: float,
        img_frame_stride: int = 8,
        n_future_channels: int = 2,
    ):
        super().__init__()
        if lookback < patch_len:
            raise ValueError(
                f"lookback must be >= patch_len (got lookback={lookback}, "
                f"patch_len={patch_len})."
            )
        if img_size % img_patch != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by img_patch ({img_patch})."
            )
        if img_frame_stride < 1:
            raise ValueError(
                f"img_frame_stride must be >= 1 (got {img_frame_stride})."
            )
        self.patch_len = patch_len
        self.stride = stride
        # Tabular tokenizer: unfold lookback → (n_tb_tokens) patches of width
        # patch_len across ALL channels, linearly projected to d_model.
        self.n_tb_tokens = (lookback - patch_len) // stride + 1
        self.tab_proj = nn.Linear(patch_len * n_channels, d_model)
        self.tab_pos = nn.Parameter(torch.zeros(1, self.n_tb_tokens, d_model))
        nn.init.trunc_normal_(self.tab_pos, std=0.02)

        # Future tokenizer: each future step of the known astronomical
        # trajectory becomes one token (query-side "what the sun will do").
        self.fut_proj = nn.Linear(n_future_channels, d_model)
        self.fut_pos = nn.Parameter(torch.zeros(1, horizon, d_model))
        nn.init.trunc_normal_(self.fut_pos, std=0.02)

        # Image tokenizer: conv patch-embed (kernel=stride=img_patch) over the
        # 4 RGB+BRR channels, flattened to n_img_tokens per RETAINED frame
        # (frames subsampled to img_frame_stride so lb*n_img_tokens stays
        # tractable). Each token gets patch-pos + frame-pos so cloud motion
        # ordering survives into the cross-attention (no temporal mean-pool).
        self.img_patch = img_patch
        self.img_frame_stride = img_frame_stride
        self.n_img_frames = (lookback + img_frame_stride - 1) // img_frame_stride
        self.n_img_tokens = (img_size // img_patch) ** 2
        self.img_conv = nn.Conv2d(in_channels, d_model, kernel_size=img_patch, stride=img_patch)
        self.img_pos = nn.Parameter(torch.zeros(1, self.n_img_tokens, d_model))
        self.frame_pos = nn.Parameter(torch.zeros(1, self.n_img_frames, 1, d_model))
        nn.init.trunc_normal_(self.img_pos, std=0.02)
        nn.init.trunc_normal_(self.frame_pos, std=0.02)

        self.blocks = nn.ModuleList(
            [_CrossAttnBlock(d_model, n_heads, dropout) for _ in range(depth)]
        )
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.head = nn.Linear(d_model, horizon)

    def _tab_tokens(self, x: torch.Tensor, x_future: torch.Tensor | None) -> torch.Tensor:
        # x: (n, lb, Fp) → unfold lookback → (n, n_tb, Fp, patch_len).
        n = x.shape[0]
        patches = x.unfold(1, self.patch_len, self.stride)  # (n, n_tb, Fp, patch_len)
        patches = patches.contiguous().reshape(n, self.n_tb_tokens, -1)
        tokens = self.tab_proj(patches) + self.tab_pos
        if x_future is not None:
            tokens = torch.cat([tokens, self.fut_proj(x_future) + self.fut_pos], dim=1)
        return tokens

    def _img_tokens(self, images: torch.Tensor) -> torch.Tensor:
        # images: (n, lb, C, H, W) already resized to img_size. Subsample
        # frames, embed each retained frame independently, tag tokens with
        # patch-pos + frame-pos → (n, n_img_frames*n_img_tokens, d_model).
        n, lb, c, h, w = images.shape
        # Never let the stride out-run a short lookback (lb < stride would
        # keep zero frames → empty token sequence → NaN through attention).
        stride = min(self.img_frame_stride, lb)
        frames = images[:, stride - 1 :: stride]
        kept = frames.shape[1]
        flat = frames.reshape(n * kept, c, h, w)
        feats = self.img_conv(flat)  # (n*kept, d_model, gh, gw)
        gh, gw = feats.shape[-2:]
        feats = feats.reshape(n, kept, feats.shape[1], gh * gw)  # (n, kept, d, n_img_tokens)
        feats = feats.permute(0, 1, 3, 2)  # (n, kept, n_img_tokens, d_model)
        feats = feats + self.img_pos.unsqueeze(1) + self.frame_pos[:, :kept]
        return feats.reshape(n, kept * self.n_img_tokens, feats.shape[-1])

    def forward(
        self,
        x: torch.Tensor,
        images: torch.Tensor | None = None,
        x_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tab = self._tab_tokens(x, x_future)
        # Graceful fallback: no images → image tokens collapse to tabular tokens
        # (cross-attention reduces to tabular self-attention; shapes unchanged).
        img = self._img_tokens(images) if images is not None else tab
        for block in self.blocks:
            tab, img = block(tab, img)
        fused = torch.cat([tab.mean(dim=1), img.mean(dim=1)], dim=-1)  # (n, 2*d_model)
        return self.head(F.gelu(self.fuse(fused)))  # (n, horizon)


class CrossViViTModel(TorchForecastMixin, ForecastModel):
    """Cross-attention multimodal (tabular + sky-image) forecaster.

    Parameters
    ----------
    d_model:
        Transformer hidden width (must be divisible by ``n_heads``).
    n_heads:
        Number of attention heads per cross-attention sub-layer.
    depth:
        Number of bidirectional cross-attention blocks.
    patch_len:
        Patch length (lookback timesteps per tabular token).
    stride:
        Stride between tabular patches along the lookback axis; ``None`` →
        non-overlapping (``stride = patch_len``).
    img_size:
        Square target size sky images are resized to (via
        :func:`resize_frames`) so mixed native sizes batch together.
    img_patch:
        Conv patch size for the image tokenizer (``img_size`` must be divisible
        by it); yields ``(img_size // img_patch) ** 2`` image tokens per frame.
    img_frame_stride:
        Keep every ``img_frame_stride``-th lookback frame for the image-token
        sequence (bounds the sequence length; 8 → 240 frames become 30).
    in_channels:
        Number of image channels (4 = RGB + BRR for the standardized sky cube).
    dropout:
        Dropout inside the cross-attention blocks + FFNs.
    lr:
        Adam learning rate.
    epochs:
        Number of training epochs over the fit window set.
    batch_size:
        Minibatch size (full-batch when larger than ``n``).
    device:
        ``"cuda"`` / ``"cpu"`` / a :class:`torch.device`; ``None`` auto-selects
        CUDA when available else CPU.
    name:
        Display name returned by the :attr:`name` property.
    variant:
        ``"crossvivit"`` (default; full multimodal model) or ``"tabular"`` (the
        image branch is never used — a convenience ablation equivalent to
        always passing ``images=None``).
    seed:
        Seed for Python/NumPy/Torch RNG (and cuDNN determinism) applied at the
        start of :meth:`fit`; guarantees reproducible fits.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        depth: int = 2,
        patch_len: int = 4,
        stride: int | None = None,
        img_size: int = 32,
        img_patch: int = 4,
        img_frame_stride: int = 8,
        in_channels: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 16,
        weight_decay: float = 0.0,
        patience: int | None = None,
        val_fraction: float = 0.0,
        accum_steps: int = 1,
        device: str | torch.device | None = None,
        name: str = "CrossViViT",
        variant: str = "crossvivit",
        seed: int = 0,
    ):
        if variant not in _VARIANTS:
            raise ValueError(
                f"unknown CrossViViT variant '{variant}' (known: {list(_VARIANTS)})"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.img_size = img_size
        self.img_patch = img_patch
        self.img_frame_stride = img_frame_stride
        self.in_channels = in_channels
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # FULL-config training additions (defaults preserve old tiny behavior).
        # In FULL mode batch_size=64 + accum_steps=4 → micro-batch 16 fits 24GB
        # at 256x256 sky images while keeping an effective batch of 64.
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.accum_steps = accum_steps
        self.device = self._resolve_device(device)
        self._name = name
        self.variant = variant
        self.seed = seed
        self._net: _CrossViViTNet | None = None

    @property
    def name(self) -> str:
        return self._name

    # -- ForecastModel API ---------------------------------------------------

    def fit(
        self,
        X_past: np.ndarray,
        Y: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
    ) -> None:
        """Train the CrossViViT cross-attention net with Adam + MAE (L1) loss.

        ``X_future[n, horizon, Ff]`` (known astronomical trajectory) is
        embedded as extra tabular tokens. ``times`` is accepted only to honor
        the :class:`ForecastModel` signature. ``images`` is the multimodal
        input ``(n, lookback, 4, H, W)``; ``None`` trains the tabular-only
        path.
        """
        X = np.asarray(X_past, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"X_past must be 3-D (n, lookback, Fp); got shape {X.shape}"
            )
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, horizon); got shape {Y.shape}")
        n, lookback, fp = X.shape
        horizon = Y.shape[1]
        if lookback < self.patch_len:
            raise ValueError(
                f"lookback must be >= patch_len (got lookback={lookback}, "
                f"patch_len={self.patch_len})."
            )
        Xf = self._prepare_future(X_future, n, horizon)

        imgs = self._prepare_images(images, n)  # resized float32 (n,lb,4,img,img) | None

        self._seed(self.seed)
        self._net = _CrossViViTNet(
            lookback=lookback,
            n_channels=fp,
            horizon=horizon,
            patch_len=self.patch_len,
            stride=self.stride,
            img_size=self.img_size,
            img_patch=self.img_patch,
            in_channels=self.in_channels,
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout,
            img_frame_stride=self.img_frame_stride,
            n_future_channels=Xf.shape[2] if Xf is not None else 2,
        ).to(self.device)
        loss_fn = nn.L1Loss()  # MAE — robust, matches the Phase 6b spec.

        Xt = self._sanitize(torch.from_numpy(X)).to(self.device)
        Yt = torch.from_numpy(Y).to(self.device)
        Xft = self._future_tensor(Xf)  # (n,horizon,Ff) on device | None
        imgt = self._image_tensor(imgs)  # (n,lb,4,img,img) on device | None

        # fit_loop calls net(*[t[batch] for t in tensors]) positionally; wrap
        # the net so each minibatch slice is routed to forward(x, images=,
        # x_future=) by keyword — robust to which modalities are present.
        tensors = tuple(t for t in (Xt, imgt, Xft) if t is not None)
        keys = [
            name
            for name, t in (("x", Xt), ("images", imgt), ("x_future", Xft))
            if t is not None
        ]

        class _KwNet(nn.Module):
            """Route positional minibatch tensors to named forward args."""

            def __init__(self, inner: nn.Module, keys: list[str]):
                super().__init__()
                self.inner = inner
                self._keys = keys

            def forward(self, *batch: torch.Tensor) -> torch.Tensor:
                return self.inner(**dict(zip(self._keys, batch)))

        fit_loop(
            _KwNet(self._net, keys),
            tensors,
            Yt,
            loss_fn=loss_fn,
            **train_kwargs(self),
        )
        return None

    def predict(
        self,
        X_past: np.ndarray,
        times: np.ndarray | None = None,
        X_future: np.ndarray | None = None,
        *,
        images: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predictions of shape ``(n, horizon)`` as float32 numpy."""
        if self._net is None:
            raise RuntimeError(
                "CrossViViTModel.predict called before fit; "
                "call fit(X_past, Y, ...) first."
            )
        X = np.asarray(X_past, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"X_past must be 3-D (n, lookback, Fp); got shape {X.shape}"
            )
        n = X.shape[0]
        imgs = self._prepare_images(images, n)  # CPU numpy (n,lb,4,img,img) | None
        horizon = self._net.head.out_features
        Xf = self._prepare_future(X_future, n, horizon)
        # Chunked inference: stage Xt + the image cube per micro-batch (the
        # training micro-batch = batch_size // accum_steps, sized for the image
        # path's 24GB VRAM budget) so the whole (large) test image cube is never
        # staged on device at once and the attention kernel grid limit is not
        # tripped. Numerically identical to a single forward.
        eval_bs = max(1, self.batch_size // max(1, getattr(self, "accum_steps", 1)))
        has_img = imgs is not None
        has_fut = Xf is not None
        outs = []
        self._net.eval()
        with torch.no_grad():
            for start in range(0, n, eval_bs):
                sl = slice(start, start + eval_bs)
                Xt = self._sanitize(torch.from_numpy(X[sl])).to(self.device)
                imgt = self._image_tensor(imgs[sl]) if has_img else None
                Xft = self._future_tensor(Xf[sl]) if has_fut else None
                outs.append(self._net(Xt, imgt, Xft))
        pred = torch.cat(outs, dim=0).cpu().numpy().astype(np.float32)
        return pred

    # -- internals -----------------------------------------------------------

    def _prepare_future(
        self, X_future: np.ndarray | None, n: int, horizon: int
    ) -> np.ndarray | None:
        """Validate (and float32-cast) the future trajectory (or None)."""
        if X_future is None:
            return None
        Xf = np.asarray(X_future, dtype=np.float32)
        if Xf.ndim != 3 or Xf.shape[0] != n or Xf.shape[1] != horizon:
            raise ValueError(
                f"X_future must be (n={n}, horizon={horizon}, Ff); got shape {Xf.shape}"
            )
        return Xf

    def _future_tensor(self, Xf: np.ndarray | None) -> torch.Tensor | None:
        """Sanitize + move the future trajectory to the device (or None)."""
        if Xf is None:
            return None
        return self._sanitize(torch.from_numpy(Xf)).to(self.device)

    def _prepare_images(
        self, images: np.ndarray | None, n: int
    ) -> np.ndarray | None:
        """Resize the raw sky cube to ``img_size`` (or None).

        Accepts ``(n, lookback, 4, H, W)`` at any native H/W and resizes every
        frame to ``(img_size, img_size)`` via :func:`resize_frames` so mixed
        native sizes (Folsom-256, SKIPPD-64) batch together. Returns ``None``
        for the tabular-only path.
        """
        if self.variant == "tabular" or images is None:
            return None
        imgs = np.asarray(images, dtype=np.float32)
        if imgs.ndim != 5 or imgs.shape[2] != self.in_channels:
            raise ValueError(
                f"images must be (n, lookback, {self.in_channels}, H, W); "
                f"got shape {imgs.shape}"
            )
        if imgs.shape[0] != n:
            raise ValueError(
                f"images lead dim ({imgs.shape[0]}) != X_past lead dim ({n})."
            )
        n_, lb, c, h, w = imgs.shape
        flat = imgs.reshape(n_ * lb, c, h, w)
        from ...imaging.resize import resize_frames

        resized = np.asarray(
            resize_frames(flat, (self.img_size, self.img_size))
        ).reshape(n_, lb, c, self.img_size, self.img_size)
        return resized

    def _image_tensor(self, imgs: np.ndarray | None) -> torch.Tensor | None:
        """Sanitize + move the prepared image cube to the device (or None)."""
        if imgs is None:
            return None
        return self._sanitize(torch.from_numpy(imgs)).to(self.device)
