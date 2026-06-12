"""Bracket-aware loaders. These pull the right rows/brackets for the experiment kind:

    carrier  : same prompt template (one sentence slot) filled with S4 vs P -> two ROWS;
               U comes from one filling, P_U from the S4 row's slot, I from the P row's slot.
    p5_fwd   : two-sentence prompt with sentence_1 = P_U (S4 fill), sentence_2 = I (P fill).
    p5_rev   : same template with the swap.

The contrast-quad loader returns the four sentence views (S4·O1, P·O2, P·O1, S4·O2) used by
contrast / projection / logit-lens analyses. Layer means are cached per (fwd, rev) pair.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file, save_file

from . import io


# Map a bracket-pooling name to the safetensors key suffix produced by analyses/extract.py.
# Keep in sync with the writes in extract.py (`pooled_bracket_<field>{suffix}.safetensors`).
_POOL_SUFFIX = {"mean": "", "last": "_last", "bertscore": "_bertscore"}


def _bracket_suffix(pooling: str) -> str:
    try:
        return _POOL_SUFFIX[pooling]
    except KeyError:
        raise ValueError(
            f"unknown bracket pooling {pooling!r}; expected one of {sorted(_POOL_SUFFIX)}"
        )


def carrier_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For an N0 carrier run, return (i_s4, i_p) — row indices for the S4 fill and P fill."""
    s4_pos, p_pos = {}, {}
    for i, pid in enumerate(meta["prompt_id"]):
        r, c = pid.split(":")
        (s4_pos if c == "S4" else p_pos)[int(r)] = i
    rows = sorted(set(s4_pos) & set(p_pos))
    return np.array([s4_pos[r] for r in rows]), np.array([p_pos[r] for r in rows])


def load_triplet(run: Path, layer: int, kind: str, pooling: str = "mean"):
    """Return (U, P_U, I, N) at one layer. All N×d, aligned across rows.

    kind == 'carrier'  : P_U/I differ by ROW (S4 vs P fill); brackets are answer + sentence_slot.
    kind == 'p5_fwd'   : sentence_1 = P_U, sentence_2 = I.
    kind == 'p5_rev'   : sentence_1 = I,   sentence_2 = P_U.
    """
    sfx = _bracket_suffix(pooling)
    manifest = json.loads((run / "manifest.json").read_text())
    meta = pd.read_parquet(run / "metadata.parquet")

    if kind == "carrier":
        slot = manifest.get("sentence_slot") or "sentence_1"
        i_s4, i_p = carrier_indices(meta)
        U = io.load_layer(run, layer, "bracket_answer" + sfx)[0].astype(np.float32)[i_s4]
        PU = io.load_layer(run, layer, f"bracket_{slot}" + sfx)[0].astype(np.float32)[i_s4]
        I = io.load_layer(run, layer, f"bracket_{slot}" + sfx)[0].astype(np.float32)[i_p]
        return U, PU, I, len(i_s4)

    if kind in ("p5_fwd", "p5_rev"):
        U = io.load_layer(run, layer, "bracket_answer" + sfx)[0].astype(np.float32)
        S1 = io.load_layer(run, layer, "bracket_sentence_1" + sfx)[0].astype(np.float32)
        S2 = io.load_layer(run, layer, "bracket_sentence_2" + sfx)[0].astype(np.float32)
        PU, I = (S1, S2) if kind == "p5_fwd" else (S2, S1)
        return U, PU, I, U.shape[0]

    raise ValueError(f"unknown kind {kind!r}")


def load_contrast_quad(fwd: Path, rev: Path, layer: int):
    """Return the four sentence views + Q/A + per-run final tokens at one layer.

    fwd: O1=S4 (blind), O2=P (saw S4)        rev: O1=P (blind), O2=S4 (saw P)
    Returns dict with keys: s4_blind, p_contr, p_blind, s4_contr, q, a, f_fwd, f_rev.
    Q and A are taken from fwd (causal mask: identical across runs).
    """
    return {
        "s4_blind": io.load_layer(fwd, layer, "bracket_sentence_1")[0].astype(np.float32),
        "p_contr":  io.load_layer(fwd, layer, "bracket_sentence_2")[0].astype(np.float32),
        "p_blind":  io.load_layer(rev, layer, "bracket_sentence_1")[0].astype(np.float32),
        "s4_contr": io.load_layer(rev, layer, "bracket_sentence_2")[0].astype(np.float32),
        "q":        io.load_layer(fwd, layer, "bracket_question")[0].astype(np.float32),
        "a":        io.load_layer(fwd, layer, "bracket_answer")[0].astype(np.float32),
        "f_fwd":    io.load_layer(fwd, layer, "last")[0].astype(np.float32),
        "f_rev":    io.load_layer(rev, layer, "last")[0].astype(np.float32),
    }


# Per-bracket / per-pair-of-runs layer means used to remove anisotropy before cosine.
_QUAD_KEYS = ["bracket_question", "bracket_answer", "bracket_sentence_1", "bracket_sentence_2"]


def shared_means(fwd: Path, rev: Path, layers: list[int], recompute: bool = False) -> dict[int, np.ndarray]:
    """Per-layer mean over both runs' bracket vectors (deduped). Cached in the fwd dir."""
    fwd, rev = Path(fwd), Path(rev)
    cache, meta_p = fwd / "contrast_layer_means.safetensors", fwd / "contrast_layer_means.json"
    if cache.exists() and meta_p.exists() and not recompute:
        prev = json.loads(meta_p.read_text())
        if prev.get("rev") == str(rev) and set(layers).issubset(prev.get("layers", [])):
            arrs = load_file(str(cache))
            return {l: arrs[f"layer_{l}"] for l in layers}
    means = {}
    for l in layers:
        parts = []
        for run in (fwd, rev):
            for k in _QUAD_KEYS:
                a = io.load_layer(run, l, k)[0].astype(np.float32)
                parts.append(a[~np.isnan(a).any(axis=1)])
        means[l] = np.unique(np.concatenate(parts, axis=0), axis=0).mean(axis=0)
    save_file({f"layer_{l}": means[l] for l in layers}, str(cache))
    meta_p.write_text(json.dumps({"rev": str(rev), "layers": sorted(layers)}, indent=2))
    return means
