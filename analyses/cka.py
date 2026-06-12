"""K×K block CKA across brackets per layer, with three modes:

  default       single-layer K×K CKA summary + (K·N)×(K·N) supermatrix.
  --all-layers  CKA per bracket-pair across every layer (line plot).
  --layer-vs-layer  per bracket, the L×L CKA(K[i], K[j]) heatmap.

Per-bracket sample centering (linear-CKA convention).

  python -m analyses.cka --run outputs/Qwen2.5-0.5B-Instruct/qwen_mensaje --layer 12
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
from core.loaders import carrier_indices
from core.metrics import linear_cka


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Run directory (carrier or generic).")
    p.add_argument("--layer", type=int, default=None, help="Hidden layer (default: middle).")
    p.add_argument("--all-layers", action="store_true",
                   help="CKA-vs-layer trend (instead of one supermatrix).")
    p.add_argument("--layer-vs-layer", action="store_true",
                   help="Kornblith-style L×L heatmap per bracket.")
    p.add_argument("--slot", default=None, help="Sentence-slot bracket name (carrier runs).")
    p.add_argument("--bracket-pooling", choices=["mean", "last", "bertscore"], default="mean",
                   help="Span pooling: 'mean' / 'last' / 'bertscore' (IDF-weighted mean).")
    p.add_argument("--rename", nargs="+", default=[],
                   help='Rename brackets in labels, e.g. "sentence_1=P_U sentence_2=I".')
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    layers = manifest["layers"]
    L = args.layer if args.layer is not None else layers[len(layers) // 2]
    if L not in layers:
        raise SystemExit(f"Layer {L} not in manifest layers {layers}.")

    brackets_in_manifest = manifest.get("brackets") or []
    if not brackets_in_manifest:
        raise SystemExit(f"{run}: manifest has no `brackets` list.")

    meta = pd.read_parquet(run / "metadata.parquet")
    sentence_slot = args.slot or manifest.get("sentence_slot")
    carrier = ("sentence_col" in meta.columns) and (sentence_slot in brackets_in_manifest)

    from core.loaders import _bracket_suffix
    sfx = _bracket_suffix(args.bracket_pooling)

    if carrier:
        i_s4, i_p = carrier_indices(meta)
        N = len(i_s4)
        RENAME = {"question": "Q", "answer": "U"}
        KEYS = []
        for b in brackets_in_manifest:
            if b == sentence_slot:
                KEYS.append(("P_U", f"bracket_{b}", i_s4))
                KEYS.append(("I",   f"bracket_{b}", i_p))
            else:
                KEYS.append((RENAME.get(b, b), f"bracket_{b}", i_s4))
    else:
        N = len(meta)
        idx = np.arange(N)
        KEYS = [(b, f"bracket_{b}", idx) for b in brackets_in_manifest]

    user_rename = dict(s.split("=", 1) for s in args.rename) if args.rename else {}
    if user_rename:
        KEYS = [(user_rename.get(nm, nm), key, idx) for nm, key, idx in KEYS]
    K = len(KEYS); names = [n for n, _, _ in KEYS]

    def compute_at(layer: int, want_supermatrix: bool = False):
        blocks = [(nm, io.load_layer(run, layer, key + sfx)[0].astype(np.float32)[idx])
                  for nm, key, idx in KEYS]
        Xc = [V - np.nanmean(V, axis=0, keepdims=True) for _, V in blocks]
        Kmats = [X @ X.T for X in Xc]
        cka = np.array([[linear_cka(Kmats[n], Kmats[m]) for m in range(K)] for n in range(K)])
        if not want_supermatrix:
            return cka, None
        Xn = [X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None) for X in Xc]
        big = np.concatenate(Xn, axis=0)
        return cka, big @ big.T

    alias = manifest.get("model_alias", "model")
    exp_plots = ensure_exp(f"cka_{alias}_{run.name}")

    if args.layer_vs_layer:
        L_count = len(layers)
        layer_cka = {nm: np.zeros((L_count, L_count)) for nm, _, _ in KEYS}
        for nm, key, idx in KEYS:
            Kmats = []
            for l in layers:
                V = io.load_layer(run, l, key + sfx)[0].astype(np.float32)[idx]
                Vc = V - np.nanmean(V, axis=0, keepdims=True)
                Kmats.append(Vc @ Vc.T)
            for i in range(L_count):
                for j in range(L_count):
                    layer_cka[nm][i, j] = linear_cka(Kmats[i], Kmats[j])
        print(f"\nrun: {run} | layer×layer CKA per bracket | bracket pooling: {args.bracket_pooling} | N={N}")
        for nm in names:
            band = layer_cka[nm]
            print(f"  {nm}: min={band[~np.eye(L_count,dtype=bool)].min():.3f}  "
                  f"adjacent-diag mean={np.mean([band[i,i+1] for i in range(L_count-1)]):.3f}")

        out = Path(args.output) if args.output else exp_plots / f"layer_vs_layer_{args.bracket_pooling}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        for ax, nm in zip(axes.flat, names):
            im = ax.imshow(layer_cka[nm], cmap="magma", vmin=0, vmax=1, origin="lower")
            ax.set_title(nm); ax.set_xlabel("layer"); ax.set_ylabel("layer")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{alias} — layer×layer linear CKA per bracket  ({args.bracket_pooling}, N={N})")
        fig.tight_layout()
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
        plt.close(fig)
        return

    if args.all_layers:
        ckas = {l: compute_at(l, want_supermatrix=False)[0] for l in layers}
        pairs = [(n, m) for n in range(K) for m in range(n + 1, K)]
        print(f"\nrun: {run} | all layers ({len(layers)}) | bracket pooling: {args.bracket_pooling} | N={N}")
        print("layer " + " ".join(f"{names[n]}-{names[m]:>5}" for n, m in pairs))
        for l in layers:
            row = ckas[l]
            print(f"{l:>5} " + " ".join(f"{row[n, m]:>7.3f}" for n, m in pairs))

        fig, ax = plt.subplots(figsize=(9, 5))
        lx = np.array(layers)
        palette = plt.get_cmap("tab10").colors
        for k, (n, m) in enumerate(pairs):
            y = np.array([ckas[l][n, m] for l in layers])
            ax.plot(lx, y, marker="o", ms=4, color=palette[k % 10], label=f"{names[n]}–{names[m]}")
        ax.set_xlabel("hidden layer"); ax.set_ylabel("linear CKA")
        ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, ncol=2)
        ax.set_title(f"{alias} — linear CKA between bracket representations, all layers (N={N})")
        out = Path(args.output) if args.output else exp_plots / f"cka_blocks_alllayers_{args.bracket_pooling}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
        plt.close(fig)
        return

    cka, S = compute_at(L, want_supermatrix=True)
    print(f"\nrun: {run} | layer {L} | bracket pooling: {args.bracket_pooling} | N={N}")
    print("linear CKA (K x K), diagonal = 1 by construction:")
    print("        " + "".join(f"{nm:>9}" for nm in names))
    for n, nm in enumerate(names):
        print(f"  {nm:>5} " + "".join(f"{cka[n, m]:>+9.3f}" for m in range(K)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2.6, 1]})
    vmax = float(np.nanpercentile(np.abs(S), 99))
    im = axes[0].imshow(S, cmap="magma", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for k in range(1, K):
        axes[0].axhline(k * N - 0.5, color="w", lw=0.8)
        axes[0].axvline(k * N - 0.5, color="w", lw=0.8)
    axes[0].set_xticks([(k + 0.5) * N - 0.5 for k in range(K)]); axes[0].set_xticklabels(names)
    axes[0].set_yticks([(k + 0.5) * N - 0.5 for k in range(K)]); axes[0].set_yticklabels(names)
    axes[0].set_title(f"block kernel supermatrix  ({K*N}×{K*N}; per-bracket-centered cosine)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="centered cosine")

    im2 = axes[1].imshow(cka, cmap="magma", vmin=0, vmax=1)
    axes[1].set_xticks(range(K)); axes[1].set_xticklabels(names)
    axes[1].set_yticks(range(K)); axes[1].set_yticklabels(names)
    for n in range(K):
        for m in range(K):
            axes[1].text(m, n, f"{cka[n, m]:.2f}", ha="center", va="center",
                         color="white" if cka[n, m] < 0.5 else "black", fontsize=10)
    axes[1].set_title("linear CKA per bracket-pair")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"{alias} — layer {L}, bracket-pooling={args.bracket_pooling}")
    out = Path(args.output) if args.output else exp_plots / "layers" / f"cka_blocks_L{L:02d}_{args.bracket_pooling}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
