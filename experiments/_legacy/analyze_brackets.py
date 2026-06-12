"""Per-layer centered-cosine of the carrier sentence vs question / answer.

For a carrier run (one prompt per row x sentence column, with bracket vectors), this
centers every vector by the cached per-layer dataset mean, L2-normalizes, and measures
cosine(sentence, question) and cosine(sentence, answer) at each layer, split by which
column filled the slot (S4, P, ...). Because question/answer are identical across a row's
sentence variants, it also reports the within-row paired contrast between the first two
sentence columns (e.g. S4 - P), which cancels the row.

    python analyze_brackets.py --run outputs/Llama-3.2-1B/n0_carrier
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import io


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="A carrier extraction run directory.")
    p.add_argument("--slot", default=None, help="Sentence slot name (default: from manifest).")
    p.add_argument("--against", nargs="+", default=["question", "answer"],
                   help="Which brackets to compare the sentence against.")
    p.add_argument("--layers", nargs="+", type=int, default=None, help="Default: manifest layers.")
    p.add_argument("--recompute-means", action="store_true", help="Override the cached per-layer means.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def unit_center(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Center by mu then L2-normalize each row."""
    Xc = X.astype(np.float32) - mu
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.clip(n, 1e-8, None)


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    slot = args.slot or manifest.get("sentence_slot")
    if not slot:
        raise SystemExit("This run is not a carrier run (no sentence_slot). Pass --slot.")
    layers = args.layers or manifest["layers"]
    meta = pd.read_parquet(run / "metadata.parquet")
    if "sentence_col" not in meta.columns:
        raise SystemExit("metadata has no sentence_col; not a carrier run.")

    sent_key = f"bracket_{slot}"
    against_keys = [f"bracket_{a}" for a in args.against]
    source_keys = [sent_key] + against_keys
    means = io.layer_means(run, source_keys, layers, recompute=args.recompute_means)

    cols = list(dict.fromkeys(meta["sentence_col"].tolist()))   # e.g. ['S4','P']
    row_of = meta["prompt_id"].str.split(":").str[0].to_numpy() # stimulus row id

    # results[against][col] = mean cosine over layers; errors = 95% CI half-width.
    results = {a: {c: [] for c in cols} for a in args.against}
    errors = {a: {c: [] for c in cols} for a in args.against}
    paired = {a: [] for a in args.against}  # mean within-row (cols[0]-cols[1]) per layer
    sent_col = meta["sentence_col"].to_numpy()
    for l in layers:
        Sc = unit_center(io.load_layer(run, l, sent_key)[0], means[l])
        for a, akey in zip(args.against, against_keys):
            Ac = unit_center(io.load_layer(run, l, akey)[0], means[l])
            cos = (Sc * Ac).sum(axis=1)                          # row-wise cosine
            for c in cols:
                vals = cos[sent_col == c]
                vals = vals[~np.isnan(vals)]
                results[a][c].append(float(vals.mean()))
                sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                errors[a][c].append(1.96 * sem)                  # 95% CI of the mean
            if len(cols) >= 2:
                df = pd.DataFrame({"row": row_of, "col": meta["sentence_col"].to_numpy(), "cos": cos})
                wide = df.pivot_table(index="row", columns="col", values="cos")
                if cols[0] in wide and cols[1] in wide:
                    paired[a].append(float((wide[cols[0]] - wide[cols[1]]).mean()))
                else:
                    paired[a].append(np.nan)

    # Console summary.
    print(f"run: {run}  | slot: {slot} | layers: {layers} | sentence cols: {cols}")
    print(f"centering: per-layer dataset mean over {source_keys} "
          f"(cached{' [recomputed]' if args.recompute_means else ''})\n")
    for a in args.against:
        print(f"centered cosine(sentence, {a}):")
        head = "  layer " + "".join(f"{c:>10}" for c in cols)
        if len(cols) >= 2:
            head += f"   {cols[0]+'-'+cols[1]:>10}"
        print(head)
        for i, l in enumerate(layers):
            line = f"  {l:>5} " + "".join(f"{results[a][c][i]:>10.3f}" for c in cols)
            if len(cols) >= 2:
                line += f"   {paired[a][i]:>+10.3f}"
            print(line)
        print()

    # Plot.
    n = len(args.against)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False)
    palette = plt.get_cmap("tab10").colors
    layers_x = np.array(layers)
    for ax, a in zip(axes[0], args.against):
        for k, c in enumerate(cols):
            m, e = np.array(results[a][c]), np.array(errors[a][c])
            color = palette[k % 10]
            ax.fill_between(layers_x, m - e, m + e, color=color, alpha=0.2, linewidth=0)
            ax.plot(layers_x, m, marker="o", color=color, label=c)
        ax.set_title(f"centered cos(sentence, {a})")
        ax.set_xlabel("hidden layer"); ax.set_ylabel("cosine")
        ax.legend(title="slot col", frameon=True)
    fig.suptitle(f"{manifest.get('model_alias','model')} — carrier sentence vs {'/'.join(args.against)} "
                 f"(per-layer centered cosine)")
    out = Path(args.output) if args.output else Path("plots/experiments/_legacy") / f"brackets_{run.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
