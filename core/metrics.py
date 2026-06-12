"""Math used across analyses: linear CKA, centered kernels, cosine centering,
axis projection, RMSNorm. No I/O — pure NumPy."""
from __future__ import annotations

import numpy as np


def kernel(X: np.ndarray) -> np.ndarray:
    """N×N centered Gram matrix: K = X_c X_cᵀ with X_c = X − mean(X). Used by linear CKA."""
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    return Xc @ Xc.T


def linear_cka(K1: np.ndarray, K2: np.ndarray) -> float:
    """⟨K1, K2⟩_F / (‖K1‖_F · ‖K2‖_F)  ∈ [0, 1] for centered Grams."""
    num = float(np.sum(K1 * K2))
    den = float(np.linalg.norm(K1) * np.linalg.norm(K2))
    return num / max(den, 1e-12)


def unit_center(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Center by `mu` (per-layer dataset mean), then L2-normalize rows."""
    Xc = X.astype(np.float32) - mu
    return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-8, None)


def axis_projection(a: np.ndarray, s4: np.ndarray, p: np.ndarray):
    """Per-row affine coordinate t = ⟨a − s4, p − s4⟩ / ‖p − s4‖²  and off-axis residual.

    Translation/rotation/scale invariant (depends only on differences). NaN rows and
    near-zero axes are dropped.
    """
    delta = p - s4
    denom = (delta * delta).sum(axis=1)
    au = a - s4
    valid = ~(np.isnan(s4).any(1) | np.isnan(p).any(1) | np.isnan(a).any(1)) & (denom > 1e-8)
    delta, au, denom = delta[valid], au[valid], denom[valid]
    t = (au * delta).sum(axis=1) / denom
    resid = np.linalg.norm(au - t[:, None] * delta, axis=1) / np.sqrt(denom)
    return t, resid


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm in NumPy: x_i ← x_i / sqrt(mean(x²) + eps) * gamma. Matches HF's LlamaRMSNorm."""
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + eps) * gamma
