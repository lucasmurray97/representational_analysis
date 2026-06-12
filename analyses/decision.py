"""Decision-token decomposition across the forward / swapped pair, decomposed into
content vs position.

Likert (default, 1..5):
    forward a (Oración1=S4, Oración2=P) -> 1=S4 .. 5=P
    swapped b (Oración1=P,  Oración2=S4) -> realign via 6-b
    content c = (a + (6-b))/2     position p = (a + b - 6)/2

Binary (--binary, 0/1):
    forward a (Oración1=S4, Oración2=P) -> 0=S4, 1=P
    swapped b (Oración1=P,  Oración2=S4) -> 0=P,  1=S4
    content (chose P) = (a + (1-b))/2     position (chose 2nd) = (a + b)/2

  python -m analyses.decision --fwd outputs/.../prompt4_fwd --rev outputs/.../prompt4_rev
  python -m analyses.decision --fwd outputs/.../p5_fwd      --rev outputs/.../p5_rev --binary
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

from core.experiment_folder import ensure as ensure_exp


def likert(fwd: Path, rev: Path, output: Path | None) -> None:
    mf = pd.read_parquet(fwd / "metadata.parquet")
    mr = pd.read_parquet(rev / "metadata.parquet")
    a = pd.to_numeric(mf["answer_text"], errors="coerce")
    b = pd.to_numeric(mr["answer_text"], errors="coerce")
    ok = a.between(1, 5) & b.between(1, 5)
    a, b = a[ok].to_numpy(float), b[ok].to_numpy(float)
    c = (a + (6 - b)) / 2
    pos = (a + b - 6) / 2

    n = len(a)
    print(f"paired valid rows: {n}/{len(mf)}")
    print(f"  mean forward a (1=S4..5=P)      : {a.mean():.2f}")
    print(f"  mean swapped b (1=P..5=S4)      : {b.mean():.2f}")
    print(f"  content   c = (a+(6-b))/2       : {c.mean():.2f}   (3 = no S4/P preference)")
    print(f"  position  p = (a+b-6)/2         : {pos.mean():+.2f}   (<0 primacy / Oración 1)")
    print(f"  answers favouring Oración 1 (<=2): {(np.concatenate([a,b])<=2).mean()*100:.0f}%"
          f"   Oración 2 (>=4): {(np.concatenate([a,b])>=4).mean()*100:.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    g = np.arange(1, 6)
    axes[0].hist([a, b], bins=np.arange(0.5, 6.5), label=["forward a", "swapped b"], rwidth=0.9)
    axes[0].set_xticks(g); axes[0].set_xlabel("Likert digit emitted"); axes[0].set_title("decision-token answer")
    axes[0].legend()
    axes[1].hist(pos, bins=np.arange(-2.25, 2.5, 0.5), color="#9467bd", rwidth=0.9)
    axes[1].axvline(0, color="k", lw=1, ls="--")
    axes[1].axvline(pos.mean(), color="red", lw=2, label=f"mean p = {pos.mean():+.2f}")
    axes[1].set_xlabel("position bias  p=(a+b-6)/2"); axes[1].set_title("primacy (<0) / recency (>0)")
    axes[1].legend()
    mname = json.loads((fwd / "manifest.json").read_text()).get("model_alias", "model")
    fig.suptitle(f"{mname} — decision-token Likert (forward vs swapped)")
    out = output if output else ensure_exp(f"decision_{mname}_{fwd.name}") / "decision_likert.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    plt.close(fig)


def binary(fwd: Path, rev: Path, output: Path | None) -> None:
    mf = pd.read_parquet(fwd / "metadata.parquet")
    mr = pd.read_parquet(rev / "metadata.parquet")
    a = pd.to_numeric(mf["answer_text"], errors="coerce")
    b = pd.to_numeric(mr["answer_text"], errors="coerce")
    ok = a.isin([0, 1]) & b.isin([0, 1])
    a, b = a[ok].to_numpy(float), b[ok].to_numpy(float)
    n = len(a)
    content = (a + (1 - b)) / 2
    position = (a + b) / 2

    print(f"paired valid rows: {n}/{len(mf)}")
    print(f"  forward: chose Oración 2 (=P)  : {a.mean()*100:.0f}%")
    print(f"  swapped: chose Oración 2 (=S4) : {b.mean()*100:.0f}%")
    print(f"  content  (chose P / inference) : {content.mean():.3f}  (0.5 = no S4/P preference)")
    print(f"  position (chose 2nd option)    : {position.mean():.3f}  (<0.5 primacy, >0.5 recency)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(2)
    w = 0.38
    fc = [int((a == 0).sum()), int((a == 1).sum())]
    rc = [int((b == 0).sum()), int((b == 1).sum())]
    axes[0].bar(x - w / 2, fc, w, label="forward (0=S4, 1=P)", color="#1f77b4")
    axes[0].bar(x + w / 2, rc, w, label="swapped (0=P, 1=S4)", color="#ff7f0e")
    axes[0].set_xticks(x); axes[0].set_xticklabels(["0 (Oración 1)", "1 (Oración 2)"])
    axes[0].set_ylabel("count"); axes[0].set_title("emitted answer (decision token)")
    axes[0].legend(fontsize=8)
    vals = [content.mean(), position.mean()]
    cols = ["#2ca02c", "#9467bd"]
    bars = axes[1].bar(["content\n(0=S4 … 1=P)", "position\n(0=primacy … 1=recency)"], vals, color=cols, width=0.55)
    axes[1].axhline(0.5, color="k", lw=1, ls="--")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("mean")
    axes[1].set_title("content vs position decomposition")
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    mname = json.loads((fwd / "manifest.json").read_text()).get("model_alias", "model")
    fig.suptitle(f"{mname} — binary decision token (forward vs swapped)")
    out = output if output else ensure_exp(f"decision_{mname}_{fwd.name}") / "decision_binary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fwd", required=True)
    p.add_argument("--rev", required=True)
    p.add_argument("--binary", action="store_true", help="Use 0/1 prompt instead of 1..5 Likert.")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    out = Path(args.output) if args.output else None
    (binary if args.binary else likert)(Path(args.fwd), Path(args.rev), out)


if __name__ == "__main__":
    main()
