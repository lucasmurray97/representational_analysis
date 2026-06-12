"""Run a causal LM on a short sentence, take per-token hidden states at chosen
layers, PCA them into 2D, and scatter-plot the tokens.

Modes:
  (default)     one PCA over all (token x layer) vectors, coloured by layer,
                with each token's trajectory across layers connected.
  --drop-bos    exclude the BOS token before PCA (kills the attention-sink outlier).
  --normalize   L2-normalise each vector before PCA (direction, not magnitude).
  --per-layer   a separate PCA per layer (one subplot each), coloured by token.

Example:
    python plot_token_pca.py --sentence "The cat sat down" --layers 0 8 16 --drop-bos
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--sentence", default="The cat sat down")
    p.add_argument("--layers", nargs="+", type=int, default=[0, 8, 16])
    p.add_argument("--device", default="cpu")
    p.add_argument("--drop-bos", action="store_true", help="Exclude the BOS token before PCA.")
    p.add_argument("--normalize", action="store_true", help="L2-normalise each vector before PCA.")
    p.add_argument("--per-layer", action="store_true", help="One PCA per layer (separate subplots).")
    p.add_argument("--output", default=None)
    return p.parse_args()


def clean_token(s: str) -> str:
    if s.startswith("<|") and s.endswith("|>"):
        return "BOS" if "begin" in s else s.strip("<|>")
    return s.replace("Ġ", "·").replace("Ċ", "\\n").replace("▁", "·")  # BPE/SP space markers


def pca_2d(X: np.ndarray):
    """Top-2 principal components via SVD. Returns (proj [N,2], explained_var_ratio [2])."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    evr = (S**2) / (S**2).sum()
    return proj, evr[:2]


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **{dtype_kw: torch.float32}).eval()
    model.to(args.device)

    enc = tok(args.sentence, return_tensors="pt").to(args.device)  # adds BOS
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)

    token_ids = enc["input_ids"][0].tolist()
    tokens = [clean_token(t) for t in tok.convert_ids_to_tokens(token_ids)]
    n_tok = len(tokens)
    print(f"'{args.sentence}' -> {n_tok} tokens: {tokens}")

    # Collect every (layer, token) hidden-state vector.
    vecs, layer_of, tok_of = [], [], []
    for li in args.layers:
        h = out.hidden_states[li][0]  # [T, d]
        for ti in range(n_tok):
            vecs.append(h[ti].float().cpu().numpy())
            layer_of.append(li)
            tok_of.append(ti)
    X = np.stack(vecs)
    layer_of = np.array(layer_of)
    tok_of = np.array(tok_of)

    if args.drop_bos:
        bos_ids = {tok.bos_token_id}
        bos_pos = [i for i in range(n_tok) if token_ids[i] in bos_ids or tokens[i] == "BOS"]
        keep = ~np.isin(tok_of, bos_pos)
        X, layer_of, tok_of = X[keep], layer_of[keep], tok_of[keep]
        print(f"dropped BOS token(s) at position(s) {bos_pos}")
    if args.normalize:
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)

    present = sorted(set(tok_of.tolist()))
    plt.style.use("seaborn-v0_8-whitegrid")

    suffix = "".join(s for s, on in [("_nobos", args.drop_bos), ("_norm", args.normalize),
                                     ("_perlayer", args.per_layer)] if on)
    title_bits = [b for b in ["BOS dropped" if args.drop_bos else "",
                              "L2-normalised" if args.normalize else "",
                              "per-layer PCA" if args.per_layer else ""] if b]
    subtitle = ("  [" + ", ".join(title_bits) + "]") if title_bits else ""

    if args.per_layer:
        fig, axes = plt.subplots(1, len(args.layers), figsize=(5 * len(args.layers), 5), squeeze=False)
        tab = plt.get_cmap("tab10").colors
        color_by_tok = {ti: tab[k % 10] for k, ti in enumerate(present)}
        for ax, li in zip(axes[0], args.layers):
            m = layer_of == li
            proj, evr = pca_2d(X[m])
            sub_tok = tok_of[m]
            ax.scatter(proj[:, 0], proj[:, 1], s=120,
                       color=[color_by_tok[t] for t in sub_tok], edgecolor="white", linewidth=0.8, zorder=3)
            for j in range(proj.shape[0]):
                ax.annotate(tokens[sub_tok[j]], (proj[j, 0], proj[j, 1]),
                            fontsize=9, xytext=(4, 4), textcoords="offset points")
            ax.set_title(f"layer {li}   (PC1 {evr[0]*100:.0f}%, PC2 {evr[1]*100:.0f}%)", fontsize=10)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.suptitle(f"{args.model.split('/')[-1]} — per-token PCA per layer   “{args.sentence}”{subtitle}")
    else:
        proj, evr = pca_2d(X)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(args.layers)))
        color_by_layer = {li: palette[k] for k, li in enumerate(args.layers)}
        for ti in present:  # faint trajectory per token across ordered layers
            order = [np.where((tok_of == ti) & (layer_of == li))[0][0] for li in sorted(args.layers)]
            ax.plot(proj[order, 0], proj[order, 1], color="#bbbbbb", lw=0.8, zorder=1)
        for li in args.layers:
            m = layer_of == li
            ax.scatter(proj[m, 0], proj[m, 1], s=120, color=color_by_layer[li],
                       edgecolor="white", linewidth=0.8, zorder=3, label=f"layer {li}")
        for i in range(X.shape[0]):
            ax.annotate(tokens[tok_of[i]], (proj[i, 0], proj[i, 1]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points", zorder=4)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}% var)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}% var)")
        ax.set_title(f"{args.model.split('/')[-1]} — per-token hidden states\n“{args.sentence}”"
                     f"   (PCA over {X.shape[0]} vectors){subtitle}", fontsize=11)
        ax.legend(title="hidden layer", frameon=True, facecolor="white", edgecolor="#d8d8d8")

    slug = re.sub(r"\W+", "_", args.sentence.lower()).strip("_")
    out_path = Path(args.output) if args.output else Path("plots/experiments/_legacy") / f"token_pca_{slug}{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
