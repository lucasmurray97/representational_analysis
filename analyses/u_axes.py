"""U vs P_U and U vs I — linear CKA across hidden layers.

Per run, builds per-bracket centered kernels K_U, K_PU, K_I (N×N) and reports
CKA(U, P_U) and CKA(U, I). Supports carrier and p5_fwd/p5_rev runs.

Per run, writes into plots/u_axes/<alias>_<run.name>/:
    u_axes_pPU_pI.png           per-run summary curve
    u_axes_layer_vs_layer.png   L×L heatmaps: CKA(K_U[i], K_PU[j]) and CKA(K_U[i], K_I[j])
    u_axes_cka.csv              per-layer scalars
    u_axes_kernels.npz          K_U[l], K_PU[l], K_I[l] (N×N float32)

  python -m analyses.u_axes
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.experiment_folder import ensure as ensure_exp
from core.loaders import load_triplet
from core.metrics import kernel, linear_cka


def compute(run: Path, kind: str, pooling: str = "mean"):
    manifest = json.loads((run / "manifest.json").read_text())
    layers = manifest["layers"]
    u_pu, u_i, N_seen = [], [], None
    kernels = {}
    for l in layers:
        U, PU, I, N = load_triplet(run, l, kind, pooling)
        N_seen = N
        Ku, Kpu, Ki = kernel(U), kernel(PU), kernel(I)
        u_pu.append(linear_cka(Ku, Kpu))
        u_i.append(linear_cka(Ku, Ki))
        kernels[f"K_U_layer{l:02d}"] = Ku.astype(np.float32)
        kernels[f"K_PU_layer{l:02d}"] = Kpu.astype(np.float32)
        kernels[f"K_I_layer{l:02d}"] = Ki.astype(np.float32)
    return {
        "alias": manifest.get("model_alias", run.name),
        "run_name": run.name,
        "kind": kind,
        "layers": np.array(layers),
        "u_pu": np.array(u_pu),
        "u_i": np.array(u_i),
        "N": int(N_seen),
        "kernels": kernels,
    }


def plot_curve(s: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(s["layers"], s["u_pu"], "-o", ms=4, color="#1f77b4", label="U – P_U")
    ax.plot(s["layers"], s["u_i"],  "-s", ms=4, color="#d62728", label="U – I")
    ax.set_xlabel("hidden layer"); ax.set_ylabel("linear CKA")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    ax.set_title(f"{s['alias']} / {s['run_name']}  (kind={s['kind']}, N={s['N']})")
    ax.legend(fontsize=9, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_layer_vs_layer(s: dict, out: Path) -> None:
    L = len(s["layers"])
    Ku = [s["kernels"][f"K_U_layer{l:02d}"] for l in s["layers"]]
    Kpu = [s["kernels"][f"K_PU_layer{l:02d}"] for l in s["layers"]]
    Ki = [s["kernels"][f"K_I_layer{l:02d}"] for l in s["layers"]]
    M_pu = np.array([[linear_cka(Ku[i], Kpu[j]) for j in range(L)] for i in range(L)])
    M_i = np.array([[linear_cka(Ku[i], Ki[j]) for j in range(L)] for i in range(L)])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, M, title, xname in zip(axes, [M_pu, M_i], ["U – P_U", "U – I"], ["P_U", "I"]):
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, origin="lower")
        ax.set_title(title)
        ax.set_xlabel(f"layer ({xname})")
        ax.set_ylabel("layer (U)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{s['alias']} / {s['run_name']} — layer×layer CKA(K_U[i], K_X[j])  (N={s['N']})")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_per_run(s: dict, plots_root: Path) -> None:
    folder = plots_root / f"{s['alias']}_{s['run_name']}"
    folder.mkdir(parents=True, exist_ok=True)
    plot_curve(s, folder / "u_axes_pPU_pI.png")
    plot_layer_vs_layer(s, folder / "u_axes_layer_vs_layer.png")
    pd.DataFrame({"layer": s["layers"], "U_PU": s["u_pu"], "U_I": s["u_i"],
                  "gap_PU_minus_I": s["u_pu"] - s["u_i"]}).to_csv(folder / "u_axes_cka.csv", index=False)
    np.savez_compressed(folder / "u_axes_kernels.npz", **s["kernels"])
    print(f"  wrote {folder}/ (u_axes_pPU_pI.png, u_axes_layer_vs_layer.png, "
          f"u_axes_cka.csv, u_axes_kernels.npz)")


RUNS = [
    ("outputs/Qwen2.5-0.5B-Instruct/qwen_mensaje", "carrier"),
    ("outputs/Llama-3.2-1B/n0_current",            "carrier"),
    ("outputs/Qwen2.5-0.5B-Instruct/p5_fwd",       "p5_fwd"),
    ("outputs/Qwen2.5-0.5B-Instruct/p5_rev",       "p5_rev"),
    ("outputs/Llama-3.2-1B/p5_fwd",                "p5_fwd"),
    ("outputs/Llama-3.2-1B/p5_rev",                "p5_rev"),
]


def main() -> None:
    plots_root = ensure_exp("u_axes")
    for path, kind in RUNS:
        run = Path(path)
        if not (run / "manifest.json").exists():
            print(f"[skip] {run}: no manifest")
            continue
        s = compute(run, kind)
        layers, upu, ui = s["layers"], s["u_pu"], s["u_i"]
        gap = upu - ui
        print(f"\n{s['alias']} / {s['run_name']}  (kind={kind}, N={s['N']})")
        print(f"  layer-0 / final U-P_U: {upu[0]:.3f} → {upu[-1]:.3f}   "
              f"U-I: {ui[0]:.3f} → {ui[-1]:.3f}   gap: {gap[0]:+.3f} → {gap[-1]:+.3f}")
        print(f"  U-P_U peak: layer {int(layers[np.argmax(upu)])} = {upu.max():.3f}   "
              f"U-I peak: layer {int(layers[np.argmax(ui)])} = {ui.max():.3f}")
        save_per_run(s, plots_root)


if __name__ == "__main__":
    main()
