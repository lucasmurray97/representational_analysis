"""Recompute contrastive cosines in *readout space*: pass each vector through the final
RMSNorm and unembedding matrix W_U (the logit-lens transform), and compare to plain
residual space. Decomposes cos(sentence, target) into content (P-S4) and position (O2-O1),
for target in {question, answer, final}.

  python -m analyses.logit_lens --model Qwen/Qwen2.5-0.5B-Instruct \
      --fwd outputs/Qwen2.5-0.5B-Instruct/p5_fwd --rev outputs/Qwen2.5-0.5B-Instruct/p5_rev
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

from core import io
from core.experiment_folder import ensure as ensure_exp
from core.metrics import rmsnorm
from core.readout import load_unembedding

TARGETS = ["question", "answer", "final"]


def centered_unit(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    Xc = X - mu
    return Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-8, None)


def decompose(fwd: str, rev: str, layers, to_logit, wu=None, gamma=None, eps=None):
    out = {t: {"content": [], "position": []} for t in TARGETS}
    for l in layers:
        def get(run, key):
            v = io.load_layer(run, l, key)[0].astype(np.float32)
            return (rmsnorm(v, gamma, eps) @ wu.T) if to_logit else v
        s4_o1 = get(fwd, "bracket_sentence_1"); p_o2 = get(fwd, "bracket_sentence_2")
        p_o1 = get(rev, "bracket_sentence_1");  s4_o2 = get(rev, "bracket_sentence_2")
        q = get(fwd, "bracket_question"); a = get(fwd, "bracket_answer")
        Ff = get(fwd, "last"); Fr = get(rev, "last")
        mu = np.nanmean(np.concatenate([s4_o1, p_o2, p_o1, s4_o2, q, a, Ff, Fr], axis=0), axis=0)
        S4_O1, P_O2, P_O1, S4_O2 = (centered_unit(x, mu) for x in (s4_o1, p_o2, p_o1, s4_o2))
        Q, A, FF, FR = (centered_unit(x, mu) for x in (q, a, Ff, Fr))
        tgt = {"question": (Q, Q), "answer": (A, A), "final": (FF, FR)}
        for t in TARGETS:
            tf, tr = tgt[t]
            cos = {"S4_O2": np.nanmean((S4_O2 * tr).sum(1)), "P_O2": np.nanmean((P_O2 * tf).sum(1)),
                   "S4_O1": np.nanmean((S4_O1 * tf).sum(1)), "P_O1": np.nanmean((P_O1 * tr).sum(1))}
            out[t]["content"].append((cos["P_O2"] + cos["P_O1"]) / 2 - (cos["S4_O2"] + cos["S4_O1"]) / 2)
            out[t]["position"].append((cos["S4_O2"] + cos["P_O2"]) / 2 - (cos["S4_O1"] + cos["P_O1"]) / 2)
    return {t: {k: np.array(v) for k, v in d.items()} for t, d in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fwd", required=True)
    ap.add_argument("--rev", required=True)
    args = ap.parse_args()
    layers = json.loads((Path(args.fwd) / "manifest.json").read_text())["layers"]

    res = decompose(args.fwd, args.rev, layers, to_logit=False)
    wu, gamma, eps = load_unembedding(args.model)
    print(f"{args.model}: W_U {wu.shape}, eps {eps}")
    log = decompose(args.fwd, args.rev, layers, to_logit=True, wu=wu, gamma=gamma, eps=eps)

    print(f"\n{'target':>9} | {'residual: content / pos(mean) / pos(|mean|)':>44} | "
          f"{'readout: content / pos(mean) / pos(|mean|)':>44}")
    for t in TARGETS:
        rp, lp = res[t]["position"], log[t]["position"]
        rc, lc = res[t]["content"], log[t]["content"]
        print(f"{t:>9} | {rc.mean():>+11.3f} {rp.mean():>+11.3f} {np.abs(rp).mean():>+13.3f}   | "
              f"{lc.mean():>+11.3f} {lp.mean():>+11.3f} {np.abs(lp).mean():>+13.3f}")
    print("\n(content = P - S4 : >0 inference closer;  position = O2 - O1 : >0 recency)")

    alias = json.loads((Path(args.fwd) / "manifest.json").read_text()).get("model_alias", "model")
    lx = np.array(layers)
    fig, axes = plt.subplots(2, len(TARGETS), figsize=(4.6 * len(TARGETS), 7), squeeze=False, sharex=True)
    for col, t in enumerate(TARGETS):
        for row, metric in enumerate(["content", "position"]):
            ax = axes[row][col]
            ax.plot(lx, res[t][metric], "-o", ms=3, color="#1f77b4", label="residual")
            ax.plot(lx, log[t][metric], "--s", ms=3, color="#d62728", label="readout (norm·W_U)")
            ax.axhline(0, color="#888", lw=0.8)
            if row == 0:
                ax.set_title(f"{t}", fontsize=11)
            ax.set_ylabel(f"{metric}\n(P−S4)" if metric == "content" else f"{metric}\n(O2−O1)")
            if row == 1:
                ax.set_xlabel("hidden layer")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, frameon=True)
    fig.suptitle(f"{alias} — cos(sentence, target): residual vs logit-lens readout  "
                 f"(content = P−S4, position = O2−O1)", fontsize=12)
    out = ensure_exp(f"logit_lens_{alias}_{Path(args.fwd).name}") / "logit_vs_residual.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
