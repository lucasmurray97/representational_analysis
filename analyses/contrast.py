"""Contrastive two-sentence analysis.

Two runs of the prompt with sentence_1 / sentence_2:
  --fwd : sentence_1 = S4 (O1, blind), sentence_2 = P  (O2, saw S4)
  --rev : sentence_1 = P  (O1, blind), sentence_2 = S4 (O2, saw P)

Only the O2 sentence has attended to the other candidate; the O1 sentence is "blind".
Q and A precede O1 and are identical across runs. We center every vector by a shared
per-layer mean over both runs (deduped), L2-normalize, and compare centered cosine to
each of {question, answer, final-token}.

  python -m analyses.contrast --fwd outputs/Qwen2.5-0.5B-Instruct/prompt4_fwd \
                              --rev outputs/Qwen2.5-0.5B-Instruct/prompt4_rev
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.experiment_folder import ensure as ensure_exp
from core.loaders import load_contrast_quad, shared_means
from core.metrics import unit_center


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fwd", required=True)
    p.add_argument("--rev", required=True)
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--recompute-means", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fwd, rev = Path(args.fwd), Path(args.rev)
    layers = args.layers or json.loads((fwd / "manifest.json").read_text())["layers"]
    means = shared_means(fwd, rev, layers, args.recompute_means)

    series = ["S4·O2 (contrast)", "P·O2 (contrast)", "S4·O1 (blind)", "P·O1 (blind)"]
    targets = ["question", "answer", "final"]
    res = {s: {t: ([], []) for t in targets} for s in series}

    for l in layers:
        q = load_contrast_quad(fwd, rev, l)
        mu = means[l]
        Q, A = unit_center(q["q"], mu), unit_center(q["a"], mu)
        Ff, Fr = unit_center(q["f_fwd"], mu), unit_center(q["f_rev"], mu)
        vecs = {"S4·O2 (contrast)": (unit_center(q["s4_contr"], mu), Fr),
                "P·O2 (contrast)": (unit_center(q["p_contr"], mu), Ff),
                "S4·O1 (blind)": (unit_center(q["s4_blind"], mu), Ff),
                "P·O1 (blind)": (unit_center(q["p_blind"], mu), Fr)}
        for s, (v, Fown) in vecs.items():
            for t in targets:
                T = Q if t == "question" else A if t == "answer" else Fown
                cos = (v * T).sum(axis=1)
                res[s][t][0].append(float(cos.mean()))
                res[s][t][1].append(1.96 * float(cos.std(ddof=1) / np.sqrt(len(cos))))

    print(f"fwd: {fwd}\nrev: {rev}\nlayers: {layers}\n")
    for t in targets:
        s4c = np.array(res["S4·O2 (contrast)"][t][0]); pc = np.array(res["P·O2 (contrast)"][t][0])
        s4b = np.array(res["S4·O1 (blind)"][t][0]); pb = np.array(res["P·O1 (blind)"][t][0])
        print(f"cos(sentence, {t}) — S4-P gap, contrastive(O2) vs blind(O1):")
        print(f"  {'layer':>5}{'gap_O2':>9}{'gap_O1':>9}{'O2-O1':>9}")
        for i, l in enumerate(layers):
            g2, g1 = s4c[i] - pc[i], s4b[i] - pb[i]
            print(f"  {l:>5}{g2:>+9.3f}{g1:>+9.3f}{g2 - g1:>+9.3f}")
        print()

    color = {"S4": "#1f77b4", "P": "#ff7f0e"}
    style = {"O2": "-", "O1": "--"}
    title_map = {"question": "question", "answer": "answer", "final": "final token"}
    fig, axes = plt.subplots(1, len(targets), figsize=(5.3 * len(targets), 4.6), squeeze=False)
    lx = np.array(layers)
    for ax, t in zip(axes[0], targets):
        for s in series:
            c = color["S4" if s.startswith("S4") else "P"]
            ls = style["O2" if "O2" in s else "O1"]
            m, e = np.array(res[s][t][0]), np.array(res[s][t][1])
            if ls == "-":
                ax.fill_between(lx, m - e, m + e, color=c, alpha=0.15, linewidth=0)
            ax.plot(lx, m, ls, color=c, marker="o" if ls == "-" else None, markersize=4, label=s)
        ax.set_title(f"centered cos(sentence, {title_map.get(t, t)})")
        ax.set_xlabel("hidden layer"); ax.set_ylabel("cosine")
        ax.legend(fontsize=8, frameon=True)
    mf = json.loads((fwd / "manifest.json").read_text())
    alias = mf.get("model_alias", "model")
    fig.suptitle(f"{alias} — two-sentence contrastive: O2 (saw the other) vs O1 (blind)")
    out = Path(args.output) if args.output else ensure_exp(f"contrast_{alias}_{fwd.name}") / "contrast.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
