"""Project the answer onto the paraphrase→inference (S4→P) axis, per layer.

The axis can be built three ways, all per stimulus row, all from differences only
(so t is invariant to translation / dataset-mean centering, rotation, and global scale):

  carrier      (--run): an N0 carrier run. s4 and p are the SAME slot in two separate
                        single-sentence prompts (each sentence encoded ALONE in the Q/A context).
  blind        (--fwd/--rev): both sentences taken from O1, where neither has seen the other
                        (causal mask):  S4 = fwd.sentence_1, P = rev.sentence_1.
  contrastive  (--fwd/--rev): both sentences taken from O2, where each HAS attended to the other:
                        P = fwd.sentence_2 (saw S4),  S4 = rev.sentence_2 (saw P).

In every case the answer precedes the sentence(s), so by the causal mask `a` is identical
across the paired prompts/runs.

  python -m analyses.projection --run outputs/Llama-3.2-1B/n0_current
  python -m analyses.projection --fwd outputs/Qwen2.5-0.5B-Instruct/prompt4_fwd \
                                --rev outputs/Qwen2.5-0.5B-Instruct/prompt4_rev \
                                --run outputs/Qwen2.5-0.5B-Instruct/qwen_mensaje
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import io
from core.experiment_folder import ensure as ensure_exp
from core.metrics import axis_projection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], dest="runs",
                   help="A carrier N0 run dir. Repeatable.")
    p.add_argument("--fwd", default=None, help="Two-sentence run, O1=PU / O2=I.")
    p.add_argument("--rev", default=None, help="Two-sentence run, O1=I / O2=PU.")
    p.add_argument("--slot", default=None, help="Carrier slot name (default from manifest).")
    p.add_argument("--para", default="PU", help="Carrier column at t=0 (paraphrase).")
    p.add_argument("--infer", default="I", help="Carrier column at t=1 (inference).")
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def summarize(t: np.ndarray, resid: np.ndarray) -> dict:
    n = len(t)
    sem = float(t.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "t_mean": float(t.mean()), "t_med": float(np.median(t)), "t_ci": 1.96 * sem,
            "frac_in01": float(((t >= 0) & (t <= 1)).mean()), "resid_mean": float(resid.mean())}


def index_by(values) -> dict:
    return {v: i for i, v in enumerate(values)}


def project_carrier(run: Path, slot, para, infer, layers):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    slot = slot or manifest.get("sentence_slot")
    if not slot:
        raise SystemExit(f"{run}: not a carrier run (no sentence_slot). Pass --slot.")
    layers = layers or manifest["layers"]
    meta = pd.read_parquet(run / "metadata.parquet")
    if "sentence_col" not in meta.columns:
        raise SystemExit(f"{run}: metadata has no sentence_col; not a carrier run.")
    row_of = meta["prompt_id"].str.split(":").str[0].to_numpy()
    sent_col = meta["sentence_col"].to_numpy()
    pmap = {row_of[i]: i for i in np.where(sent_col == para)[0]}
    imap = {row_of[i]: i for i in np.where(sent_col == infer)[0]}
    rows = [r for r in pmap if r in imap]
    if not rows:
        raise SystemExit(f"{run}: no rows have both {para} and {infer} (cols {sorted(set(sent_col))}).")
    ip = np.array([pmap[r] for r in rows]); ii = np.array([imap[r] for r in rows])
    skey = f"bracket_{slot}"
    stats = {}
    for l in layers:
        S = io.load_layer(run, l, skey)[0].astype(np.float32)
        A = io.load_layer(run, l, "bracket_answer")[0].astype(np.float32)
        stats[l] = summarize(*axis_projection(A[ip], S[ip], S[ii]))
    alias = manifest.get("model_alias", run.name)
    return [{"name": f"{alias} carrier", "layers": layers,
             "n_states": manifest.get("n_hidden_states", max(layers) + 1), "stats": stats}]


def project_contrast(fwd: Path, rev: Path, layers):
    mf = json.loads((fwd / "manifest.json").read_text(encoding="utf-8"))
    layers = layers or mf["layers"]
    fmeta = pd.read_parquet(fwd / "metadata.parquet")
    rmeta = pd.read_parquet(rev / "metadata.parquet")
    fid, rid = index_by(fmeta["prompt_id"]), index_by(rmeta["prompt_id"])
    rows = [pid for pid in fmeta["prompt_id"] if pid in rid]
    fi = np.array([fid[p] for p in rows]); ri = np.array([rid[p] for p in rows])

    blind, contr = {}, {}
    for l in layers:
        s4_blind = io.load_layer(fwd, l, "bracket_sentence_1")[0].astype(np.float32)[fi]
        p_contr = io.load_layer(fwd, l, "bracket_sentence_2")[0].astype(np.float32)[fi]
        p_blind = io.load_layer(rev, l, "bracket_sentence_1")[0].astype(np.float32)[ri]
        s4_contr = io.load_layer(rev, l, "bracket_sentence_2")[0].astype(np.float32)[ri]
        a = io.load_layer(fwd, l, "bracket_answer")[0].astype(np.float32)[fi]
        blind[l] = summarize(*axis_projection(a, s4_blind, p_blind))
        contr[l] = summarize(*axis_projection(a, s4_contr, p_contr))
    alias = mf.get("model_alias", fwd.name)
    n_states = mf.get("n_hidden_states", max(layers) + 1)
    return [{"name": f"{alias} blind (O1)", "layers": layers, "n_states": n_states, "stats": blind},
            {"name": f"{alias} contrastive (O2)", "layers": layers, "n_states": n_states, "stats": contr}]


def main() -> None:
    args = parse_args()
    if not args.runs and not (args.fwd and args.rev):
        raise SystemExit("Give --run (carrier) and/or both --fwd and --rev (two-sentence).")

    series = []
    for r in args.runs:
        series += project_carrier(Path(r), args.slot, args.para, args.infer, args.layers)
    if args.fwd and args.rev:
        series += project_contrast(Path(args.fwd), Path(args.rev), args.layers)

    same_layers = all(s["layers"] == series[0]["layers"] for s in series)
    for s in series:
        ls = s["layers"]; ns = s["n_states"]
        s["x"] = np.array(ls) if same_layers else np.array([l / (ns - 1) if ns > 1 else 0.0 for l in ls])

    for s in series:
        print(f"\n{s['name']}   (axis: {args.para} t=0 -> {args.infer} t=1)")
        print(f"  {'layer':>5}{'n':>5}{'t_mean':>9}{'t_med':>8}{'95%CI':>8}{'in[0,1]':>9}{'resid':>8}")
        for l in s["layers"]:
            st = s["stats"][l]
            print(f"  {l:>5}{st['n']:>5}{st['t_mean']:>+9.3f}{st['t_med']:>+8.3f}"
                  f"{st['t_ci']:>8.3f}{st['frac_in01']*100:>8.0f}%{st['resid_mean']:>8.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    palette = plt.get_cmap("tab10").colors
    for k, s in enumerate(series):
        x = s["x"]
        tm = np.array([s["stats"][l]["t_mean"] for l in s["layers"]])
        ci = np.array([s["stats"][l]["t_ci"] for l in s["layers"]])
        rs = np.array([s["stats"][l]["resid_mean"] for l in s["layers"]])
        c = palette[k % 10]
        axes[0].fill_between(x, tm - ci, tm + ci, color=c, alpha=0.16, linewidth=0)
        axes[0].plot(x, tm, marker="o", ms=4, color=c, label=s["name"])
        axes[1].plot(x, rs, marker="o", ms=4, color=c, label=s["name"])
    xlabel = "hidden layer" if same_layers else "fractional depth"
    for ax in axes:
        ax.set_xlabel(xlabel)
    axes[0].axhline(0.0, color="#444", lw=1, ls="--"); axes[0].axhline(1.0, color="#444", lw=1, ls="--")
    axes[0].axhline(0.5, color="#bbb", lw=0.8, ls=":")
    axes[0].set_ylabel("t  (answer on paraphrase→inference axis)")
    axes[0].set_title(f"answer projected onto the {args.para}→{args.infer} axis")
    axes[1].set_ylabel("off-axis residual  ||(a-s4) - t·Δ|| / ||Δ||")
    axes[1].set_title("faithfulness of the 1-D summary")
    axes[0].legend(fontsize=8, frameon=True); axes[1].legend(fontsize=8, frameon=True)
    fig.suptitle("answer's paraphrase↔inference coordinate — carrier vs blind vs contrastive")
    if args.output:
        out = Path(args.output)
    else:
        tag = "_".join([Path(r).name for r in args.runs] + ([Path(args.fwd).name] if args.fwd else []))
        out = ensure_exp(f"projection_{tag or 'run'}") / "projection.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nWrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
