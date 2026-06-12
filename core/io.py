"""Save / load helpers for extracted hidden-state embeddings.

Layout written under outputs/<model>/<set>/:
    manifest.json            run provenance (model, dtype, layers, pooling, versions, ts)
    metadata.parquet         one row per prompt (row, prompt_id, prompt_sha1, n_tokens, answer*)
    pooled_<pooling>.safetensors   key "layer_{l}" -> float16 [n_prompts, d]   (last / mean)
    tokens_all.safetensors         key "layer_{l}" -> float16 [sum_tokens, d]  (pooling == all)
    tokens_index.parquet     one row per kept token (row, prompt_id, token_pos, token_id, token_str)

Reads use the numpy backend of safetensors so downstream analysis does not need torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file, save_file

MEANS_FILE = "layer_means.safetensors"
MEANS_META = "layer_means.json"


def _save_layers(path: Path, layer_arrays: dict[int, np.ndarray]) -> None:
    save_file({f"layer_{l}": np.ascontiguousarray(a) for l, a in layer_arrays.items()}, str(path))


def save_run(
    out_dir: str | Path,
    *,
    pooled: dict[str, dict[int, np.ndarray]],
    metadata: pd.DataFrame,
    manifest: dict,
    tokens_all: dict[int, np.ndarray] | None = None,
    tokens_index: pd.DataFrame | None = None,
) -> Path:
    """Persist one extraction run. Returns the run directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, layer_arrays in pooled.items():
        _save_layers(out / f"pooled_{name}.safetensors", layer_arrays)
    if tokens_all:
        _save_layers(out / "tokens_all.safetensors", tokens_all)
    if tokens_index is not None:
        tokens_index.to_parquet(out / "tokens_index.parquet", index=False)

    metadata.to_parquet(out / "metadata.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def load_layer(run_dir: str | Path, layer: int, pooling: str = "whole"):
    """Load one layer's vectors plus the aligned metadata.

    pooling == "all"                       -> (array [sum_tokens, d], token-level index DataFrame).
    pooling in {"whole", "last", "bracket_<field>"} -> (array [n_prompts, d], prompt metadata DataFrame).
    """
    run = Path(run_dir)
    if pooling == "all":
        arr = load_file(str(run / "tokens_all.safetensors"))[f"layer_{layer}"]
        return arr, pd.read_parquet(run / "tokens_index.parquet")
    arr = load_file(str(run / f"pooled_{pooling}.safetensors"))[f"layer_{layer}"]
    return arr, pd.read_parquet(run / "metadata.parquet")


def layer_means(run_dir: str | Path, source_keys: list[str], layers: list[int],
                recompute: bool = False) -> dict[int, np.ndarray]:
    """Per-layer mean over the whole dataset, used to center embeddings (remove anisotropy).

    The mean at layer l pools every vector from `source_keys` (e.g. bracket_question,
    bracket_answer, bracket_sentence_1) across all rows. Cached to layer_means.safetensors;
    recomputed only if `recompute` or if the cache's source_keys/layers don't cover the request.
    """
    run = Path(run_dir)
    cache, meta_p = run / MEANS_FILE, run / MEANS_META
    spec_keys = sorted(source_keys)

    if cache.exists() and meta_p.exists() and not recompute:
        prev = json.loads(meta_p.read_text(encoding="utf-8"))
        if (prev.get("source_keys") == spec_keys and prev.get("dedup") is True
                and set(layers).issubset(prev.get("layers", []))):
            arrs = load_file(str(cache))
            return {l: arrs[f"layer_{l}"] for l in layers}

    # Dedupe identical vectors per bracket before averaging, so vectors repeated across
    # rows (e.g. question/answer, which are shared across a row's sentence variants) are
    # counted once rather than weighting the mean toward them.
    means: dict[int, np.ndarray] = {}
    n_used = {}
    for l in layers:
        parts = []
        for k in source_keys:
            a = load_layer(run, l, k)[0].astype(np.float32)
            a = a[~np.isnan(a).any(axis=1)]          # drop empty-span (NaN) rows
            a = np.unique(a, axis=0)                 # dedupe exact duplicates
            parts.append(a)
        stacked = np.concatenate(parts, axis=0)
        means[l] = stacked.mean(axis=0)
        n_used[l] = int(stacked.shape[0])
    save_file({f"layer_{l}": means[l] for l in layers}, str(cache))
    meta_p.write_text(json.dumps({"source_keys": spec_keys, "layers": sorted(layers),
                                  "dedup": True, "n_per_layer": n_used}, indent=2), encoding="utf-8")
    return means
