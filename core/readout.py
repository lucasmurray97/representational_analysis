"""Logit-lens helpers: load the final RMSNorm gamma + unembedding W_U directly from the cached
checkpoint without instantiating the model. Tied-embedding models reuse model.embed_tokens."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


def load_unembedding(model: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (W_U [vocab, d] float32, gamma [d] float32, eps) from the cached HF snapshot."""
    cache = Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--"))
    snaps = sorted(glob.glob(str(cache / "snapshots/*")))
    if not snaps:
        raise SystemExit(f"No cached snapshot for {model} under {cache}")
    snap = Path(snaps[-1])
    eps = json.loads((snap / "config.json").read_text()).get("rms_norm_eps", 1e-5)
    from safetensors import safe_open
    files = glob.glob(str(snap / "*.safetensors"))
    wu = gamma = None
    for fp in files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if "model.norm.weight" in keys:
                gamma = f.get_tensor("model.norm.weight").float().numpy()
            for k in ("lm_head.weight", "model.embed_tokens.weight"):
                if k in keys and wu is None:
                    wu = f.get_tensor(k).float().numpy()
    if wu is None or gamma is None:
        raise SystemExit("Could not find unembedding / final-norm weights in checkpoint.")
    return wu, gamma, float(eps)
