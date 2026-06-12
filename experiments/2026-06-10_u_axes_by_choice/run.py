"""U–P_U and U–I CKA across layers, split by what the model chose.

Pragmatic = model picks I (inference) as the more related sentence.
Literal   = model picks P_U (paraphrase) as the more related sentence.

Primary split: fwd & rev MUST AGREE on the choice (content-consistent).
    fwd answer = 1 (chose sentence_2 = I)   and   rev answer = 0 (chose sentence_1 = I)  → pragmatic
    fwd answer = 0 (chose sentence_1 = P_U) and   rev answer = 1 (chose sentence_2 = P_U) → literal
Position-flippers (disagree) are discarded — they show position bias, not preference.

Embeddings come from the fwd run (kind='p5_fwd'): U = bracket_answer, P_U = bracket_sentence_1,
I = bracket_sentence_2. Llama-3.2-1B is excluded (base model emits whitespace, no usable choice).

Output (in PLOTS_DIR):
    <alias>_curves.png         per-model: 2 panels (pragmatic | literal), U–P_U + U–I per layer
    <alias>_cka.csv            per-layer scalars, both subsets
    counts.txt                 N per subset, plus the single-direction comparison
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import io
from core.metrics import kernel, linear_cka

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    {"alias": "Qwen2.5-0.5B-Instruct",
     "fwd": Path("outputs/Qwen2.5-0.5B-Instruct/p5_fwd"),
     "rev": Path("outputs/Qwen2.5-0.5B-Instruct/p5_rev")},
]


def load_split(fwd: Path, rev: Path) -> dict:
    """Return row indices (into fwd) for the 4 content/position subsets."""
    mf = pd.read_parquet(fwd / "metadata.parquet").sort_values("row").reset_index(drop=True)
    mr = pd.read_parquet(rev / "metadata.parquet").sort_values("row").reset_index(drop=True)
    af = pd.to_numeric(mf["answer_text"], errors="coerce")
    ar = pd.to_numeric(mr["answer_text"], errors="coerce")
    valid = af.isin([0, 1]) & ar.isin([0, 1])
    # fwd: sentence_2 = I → answer=1 chose I.    rev: sentence_1 = I → answer=0 chose I.
    prag    = (valid & (af == 1) & (ar == 0)).to_numpy()  # consistent I (pragmatic)
    lit     = (valid & (af == 0) & (ar == 1)).to_numpy()  # consistent P_U (literal)
    primacy = (valid & (af == 0) & (ar == 0)).to_numpy()  # always Oración 1 (1st slot)
    recency = (valid & (af == 1) & (ar == 1)).to_numpy()  # always Oración 2 (2nd slot)
    return {
        "i_prag":    np.where(prag)[0],
        "i_lit":     np.where(lit)[0],
        "i_primacy": np.where(primacy)[0],
        "i_recency": np.where(recency)[0],
        "N_total": int(valid.sum()),
    }


def compute_kernels(fwd: Path, idx: np.ndarray):
    """Per-layer centered kernels K_U, K_PU, K_I (each N×N) for one row-subset."""
    manifest = json.loads((fwd / "manifest.json").read_text())
    layers = manifest["layers"]
    Ku_l, Kpu_l, Ki_l = [], [], []
    for l in layers:
        U  = io.load_layer(fwd, l, "bracket_answer")[0].astype(np.float32)[idx]
        PU = io.load_layer(fwd, l, "bracket_sentence_1")[0].astype(np.float32)[idx]
        I  = io.load_layer(fwd, l, "bracket_sentence_2")[0].astype(np.float32)[idx]
        Ku_l.append(kernel(U)); Kpu_l.append(kernel(PU)); Ki_l.append(kernel(I))
    return np.array(layers), Ku_l, Kpu_l, Ki_l


def curves_from_kernels(Ku_l, Kpu_l, Ki_l):
    upu = np.array([linear_cka(Ku_l[i], Kpu_l[i]) for i in range(len(Ku_l))])
    ui  = np.array([linear_cka(Ku_l[i], Ki_l[i])  for i in range(len(Ku_l))])
    return upu, ui


def matrices_from_kernels(Ku_l, Kx_l):
    """L×L cross-bracket CKA: M[i, j] = CKA(K_U[i], K_X[j])."""
    L = len(Ku_l)
    return np.array([[linear_cka(Ku_l[i], Kx_l[j]) for j in range(L)] for i in range(L)])


def plot_curves(alias: str, layers, panels) -> Path:
    """panels: list of (label, n, (upu, ui))."""
    K = len(panels)
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4.6), sharey=True)
    for ax, (label, n, (upu, ui)) in zip(axes, panels):
        ax.plot(layers, upu, "-o", ms=4, color="#1f77b4", label="U – P_U")
        ax.plot(layers, ui,  "-s", ms=4, color="#d62728", label="U – I")
        ax.set_xlabel("hidden layer"); ax.set_title(f"{label}   (N={n})")
        ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", frameon=True)
    axes[0].set_ylabel("linear CKA")
    fig.suptitle(f"{alias} — U–P_U and U–I by model choice  (p5_fwd embeddings)")
    out = PLOTS_DIR / f"{alias}_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_matrices(alias: str, panels_mats) -> Path:
    """panels_mats: list of (label, n, M_pu, M_i). Grid 2 rows (pair) × K cols (subset) — landscape."""
    K = len(panels_mats)
    fig, axes = plt.subplots(2, K, figsize=(3.6 * K, 7))
    if K == 1:
        axes = axes.reshape(2, 1)
    pair_labels = [("U – P_U", "P_U"), ("U – I", "I")]
    for col, (label, n, M_pu, M_i) in enumerate(panels_mats):
        for row, (M, (pair, xname)) in enumerate(zip([M_pu, M_i], pair_labels)):
            ax = axes[row, col]
            if np.isnan(M).all():
                ax.text(0.5, 0.5, "N too small", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, origin="lower")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(f"{label.replace(chr(10), ' ')}  (N={n})", fontsize=10)
            ax.set_xlabel(f"layer ({xname})", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{pair}\nlayer (U)", fontsize=9)
            else:
                ax.set_ylabel("layer (U)", fontsize=9)
    fig.suptitle(f"{alias} — layer×layer CKA(K_U[i], K_X[j]) by model choice  (p5_fwd embeddings)",
                 fontsize=11)
    out = PLOTS_DIR / f"{alias}_layer_vs_layer.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


GROUPS = [
    ("pragmatic\n(both chose I)",       "i_prag"),
    ("literal\n(both chose P_U)",       "i_lit"),
    ("primacy\n(always Oración 1)",     "i_primacy"),
    ("recency\n(always Oración 2)",     "i_recency"),
]


def main() -> None:
    counts_lines = []
    for r in RUNS:
        alias, fwd, rev = r["alias"], r["fwd"], r["rev"]
        split = load_split(fwd, rev)

        ns = {key: len(split[key]) for _, key in GROUPS}
        N = split["N_total"]
        print(f"\n{alias}  (N_total = {N})")
        for label, key in GROUPS:
            print(f"  {label.replace(chr(10), ' '):<40}: {ns[key]}")
        print(f"  sum                                     : {sum(ns.values())}  (matches {N})")

        counts_lines += [f"=== {alias} ===", f"total: {N}"]
        for label, key in GROUPS:
            counts_lines.append(f"{label.replace(chr(10), ' '):<40}: {ns[key]}")
        counts_lines.append("")

        # Compute kernels once per group; derive both curves and L×L matrices from them.
        df_cols = {"layer": None}
        panels_curves, panels_mats = [], []
        layers = None
        for label, key in GROUPS:
            idx = split[key]
            n = len(idx)
            tag = label.replace(chr(10), " ").split("(")[0].strip()
            if n < 2:
                print(f"  ! {tag:<12} N={n} too small for CKA; panel will be blank.")
                panels_curves.append((label, n, (None, None)))
                panels_mats.append((label, n, np.full((1, 1), np.nan), np.full((1, 1), np.nan)))
                continue
            layers, Ku_l, Kpu_l, Ki_l = compute_kernels(fwd, idx)
            upu, ui = curves_from_kernels(Ku_l, Kpu_l, Ki_l)
            M_pu = matrices_from_kernels(Ku_l, Kpu_l)
            M_i  = matrices_from_kernels(Ku_l, Ki_l)
            panels_curves.append((label, n, (upu, ui)))
            panels_mats.append((label, n, M_pu, M_i))
            df_cols[f"U_PU_{tag}"] = upu
            df_cols[f"U_I_{tag}"]  = ui
            print(f"  {tag:<12}  L0/Lf  U-P_U {upu[0]:.3f}→{upu[-1]:.3f}   U-I {ui[0]:.3f}→{ui[-1]:.3f}")

        if layers is None:
            print(f"  ! no group large enough for curves; skipping {alias}.")
            continue

        # Fill nan-stubs to the right length for the curves plot.
        for i, (label, n, uv) in enumerate(panels_curves):
            if uv[0] is None:
                panels_curves[i] = (label, n, (np.full(len(layers), np.nan), np.full(len(layers), np.nan)))

        df_cols["layer"] = layers
        df = pd.DataFrame(df_cols)
        csv = PLOTS_DIR / f"{alias}_cka.csv"
        df.to_csv(csv, index=False)

        png_curves = plot_curves(alias, layers, panels_curves)
        png_mats   = plot_matrices(alias, panels_mats)
        print(f"  wrote {png_curves.relative_to(PLOTS_DIR.parents[2])}")
        print(f"        {png_mats.relative_to(PLOTS_DIR.parents[2])}")
        print(f"        {csv.relative_to(PLOTS_DIR.parents[2])}")

    (PLOTS_DIR / "counts.txt").write_text("\n".join(counts_lines))


if __name__ == "__main__":
    main()
