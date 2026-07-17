import numpy as np
import pytest

from solarunibench.imaging.resize import resize_frames


def test_resize_frames_downsample_shape_and_finite():
    # (lb,4,256,256) -> (lb,4,64,64): the Folsom-to-SKIPPD common-size case.
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 1, (2, 4, 256, 256)).astype(np.float32)
    out = resize_frames(frames, (64, 64))
    assert out.shape == (2, 4, 64, 64)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_resize_frames_constant_pattern_preserved():
    # INTER_AREA of a per-channel constant is exactly that constant -> content preserved.
    frame = np.stack([
        np.full((256, 256), 0.50, np.float32),  # R
        np.full((256, 256), 0.25, np.float32),  # G
        np.full((256, 256), 0.80, np.float32),  # B
        np.full((256, 256), 0.30, np.float32),  # BRR (any range; resize is range-agnostic)
    ])[None]  # (1,4,256,256)
    out = resize_frames(frame, (64, 64))
    assert out.shape == (1, 4, 64, 64)
    assert np.allclose(out[0, 0], 0.50, atol=1e-5)
    assert np.allclose(out[0, 1], 0.25, atol=1e-5)
    assert np.allclose(out[0, 2], 0.80, atol=1e-5)
    assert np.allclose(out[0, 3], 0.30, atol=1e-5)


def test_resize_frames_identity_when_already_target():
    # SKIPPD native 64x64 resized to 64x64 is a no-op (same object returned).
    rng = np.random.default_rng(1)
    frames = rng.uniform(-1, 1, (1, 4, 64, 64)).astype(np.float32)
    out = resize_frames(frames, (64, 64))
    assert out.shape == (1, 4, 64, 64)
    assert np.array_equal(out, frames)


def test_resize_frames_mixed_batch_sizes_to_common_target():
    # The multimodal batching case: Folsom 256x256 + SKIPPD 64x64 both -> 64x64.
    rng = np.random.default_rng(2)
    big = rng.uniform(0, 1, (1, 4, 256, 256)).astype(np.float32)
    small = rng.uniform(0, 1, (1, 4, 64, 64)).astype(np.float32)
    assert resize_frames(big, (64, 64)).shape == (1, 4, 64, 64)
    assert resize_frames(small, (64, 64)).shape == (1, 4, 64, 64)


def test_resize_frames_upsample_shape():
    # Upsample path (bilinear): 64x64 -> 128x128. Shape + finite; range stays valid.
    rng = np.random.default_rng(3)
    frames = rng.uniform(0, 1, (1, 4, 64, 64)).astype(np.float32)
    out = resize_frames(frames, (128, 128))
    assert out.shape == (1, 4, 128, 128)
    assert np.isfinite(out).all()


def test_resize_frames_accepts_int_target_and_torch():
    torch = pytest.importorskip("torch")
    frames = torch.from_numpy(np.random.rand(1, 4, 256, 256).astype(np.float32))
    out = resize_frames(frames, 64)  # int target -> (64,64)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 4, 64, 64)


def test_resize_frames_rejects_bad_shape():
    with pytest.raises(ValueError):
        resize_frames(np.zeros((4, 256, 256), np.float32), (64, 64))
