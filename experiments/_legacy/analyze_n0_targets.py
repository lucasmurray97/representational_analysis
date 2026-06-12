"""Per-layer centered-cosine of the single-sentence carrier vs question / answer / final token.

For an N0 carrier run (one prompt per row x sentence column {S4, P}, with bracket vectors and
the `last` decision token), this centers every vector by the cached per-layer dataset mean
(over the sentence/question/answer brackets, deduped), L2-normalizes, and measures, split by
which column filled the slot:
    cos(sentence, question)   cos(sentence, answer)   cos(sentence, final token)
The "final token" is each prompt's own last position (h_t^(L)); it is the only position that
has attended to the whole prompt, so it is the decision representation. question/answer are
identical across a row's two sentence variants, so their within-row S4-P contrast cancels the row.

    python analyze_n0_targets.py --run outputs/Llama-3.2-1B/n0_current
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

TARGETS = ["question", "answer", "final"]   # "final" = each prompt's own last token


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="An N0 carrier extraction run directory.")
    p.add_argument("--slot", default=None, help="Sentence slot name (default: from manifest).")
    p.add_argument("--layers", nargs="+", type=int, default=None, help="Default: manifest layers.")
    p.add_argument("--recompute-means", action="store_true", help="Override the cached per-layer means.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def unit_center(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    Xc = X.astype(np.float32) - mu
    return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-8, None)


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
    means = io.layer_means(run, [sent_key, "bracket_question", "bracket_answer"], layers,
                           recompute=args.recompute_means)

    cols = list(dict.fromkeys(meta["sentence_col"].tolist()))   # e.g. ['S4','P']
    sent_col = meta["sentence_col"].to_numpy()
    row_of = meta["prompt_id"].str.split(":").str[0].to_numpy()

    results = {t: {c: [] for c in cols} for t in TARGETS}
    errors = {t: {c: [] for c in cols} for t in TARGETS}
    paired = {t: [] for t in TARGETS}   # within-row cols[0]-cols[1] per layer

    for l in layers:
        mu = means[l]
        S = unit_center(io.load_layer(run, l, sent_key)[0], mu)
        T = {"question": unit_center(io.load_layer(run, l, "bracket_question")[0], mu),
             "answer": unit_center(io.load_layer(run, l, "bracket_answer")[0], mu),
             "final": unit_center(io.load_layer(run, l, "last")[0], mu)}
        for t in TARGETS:
            cos = (S * T[t]).sum(axis=1)
            for c in cols:
                vals = cos[sent_col == c]
                vals = vals[~np.isnan(vals)]
                results[t][c].append(float(vals.mean()))
                sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                errors[t][c].append(1.96 * sem)
            if len(cols) >= 2:
                df = pd.DataFrame({"row": row_of, "col": sent_col, "cos": cos})
                wide = df.pivot_table(index="row", columns="col", values="cos")
                paired[t].append(float((wide[cols[0]] - wide[cols[1]]).mean())
                                 if cols[0] in wide and cols[1] in wide else np.nan)

    print(f"run: {run} | slot: {slot} | layers: {layers} | cols: {cols}\n")
    for t in TARGETS:
        print(f"centered cosine(sentence, {t}):")
        head = "  layer " + "".join(f"{c:>10}" for c in cols)
        if len(cols) >= 2:
            head += f"   {cols[0]+'-'+cols[1]:>10}"
        print(head)
        for i, l in enumerate(layers):
            line = f"  {l:>5} " + "".join(f"{results[t][c][i]:>10.3f}" for c in cols)
            if len(cols) >= 2:
                line += f"   {paired[t][i]:>+10.3f}"
            print(line)
        print()

    color = {c: plt.get_cmap("tab10").colors[k] for k, c in enumerate(cols)}
    title_map = {"question": "question", "answer": "answer", "final": "final token"}
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5.2 * len(TARGETS), 4.4), squeeze=False)
    lx = np.array(layers)
    for ax, t in zip(axes[0], TARGETS):
        for c in cols:
            m, e = np.array(results[t][c]), np.array(errors[t][c])
            ax.fill_between(lx, m - e, m + e, color=color[c], alpha=0.2, linewidth=0)
            ax.plot(lx, m, marker="o", color=color[c], label=c)
        ax.set_title(f"centered cos(sentence, {title_map[t]})")
        ax.set_xlabel("hidden layer"); ax.set_ylabel("cosine")
        ax.legend(title="slot col", frameon=True)
    fig.suptitle(f"{manifest.get('model_alias','model')} — N0 single sentence vs question / answer / final token")
    out = Path(args.output) if args.output else Path("plots/experiments/_legacy") / f"n0_targets_{run.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
