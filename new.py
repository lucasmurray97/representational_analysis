"""Scaffold and summarize experiment folders.

Usage:
  python new.py experiment <slug>
      → experiments/<YYYY-MM-DD>_<slug>/{run.py, notes.md, plots/}
      Plots live INSIDE the experiment folder so each experiment is self-contained.
      Run with:  python experiments/<YYYY-MM-DD>_<slug>/run.py

  python new.py summarize <slug-or-substring>
      → reads run.json + plots/ in the matching experiment folder, writes:
         notes.md (auto-summary section: invocations + artifact list, preserves your
                   handwritten Question/Method/Findings/Verdict)
         latex_snippet.tex (beamer frames, one per PNG, paths relative to project root)

  python new.py list
      Print where the canonical analyses live and what kinds exist.

Slug resolution for `summarize`: exact match wins; otherwise a unique substring match
is accepted; otherwise the candidates are listed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANONICAL = ["cka", "u_axes", "contrast", "decision", "logit_lens", "projection"]


# ---------- experiment scaffold ----------

def slugify(s: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    if not out:
        raise SystemExit(f"slug {s!r} reduces to empty string after sanitizing.")
    return out


def cmd_experiment(slug: str) -> None:
    today = dt.date.today().isoformat()
    name = f"{today}_{slugify(slug)}"
    exp_dir = ROOT / "experiments" / name
    if exp_dir.exists():
        raise SystemExit(f"{exp_dir} already exists — pick a different slug or work inside it.")
    exp_dir.mkdir(parents=True)
    (exp_dir / "plots").mkdir()
    (exp_dir / "__init__.py").write_text("")
    (exp_dir / "run.py").write_text(_RUN_PY_TEMPLATE.format(name=name))
    (exp_dir / "notes.md").write_text(_NOTES_TEMPLATE.format(name=name, today=today))
    _append_to_log(name)

    print(f"Created {exp_dir.relative_to(ROOT)}/")
    print(f"        {(exp_dir / 'plots').relative_to(ROOT)}/")
    print()
    print(f"Run with:  python experiments/{name}/run.py")
    print(f"Notes:     experiments/{name}/notes.md")
    print(f"Plot dir:  experiments/{name}/plots/   (PLOTS_DIR inside run.py)")
    print()
    print(f"!! Before running any compute: fill the **Guardrail gate** in notes.md")
    print(f"   (Hypothesis / Expected outcomes + meaning / Soundness) and confirm with")
    print(f"   the human. The gate exists to catch the two most expensive mistakes —")
    print(f"   running the wrong experiment, and running one whose outcomes you can't")
    print(f"   interpret — BEFORE you spend any compute.")


def _append_to_log(name: str) -> None:
    """Append a one-line entry for this experiment to experiments/LOG.md (create if missing)."""
    log = ROOT / "experiments" / "LOG.md"
    if not log.exists():
        log.write_text(
            "# Experiments log\n\n"
            "One line per experiment. Status: ⚪ scaffolded · 🟡 running · ✅ done · "
            "❌ retracted/dead. Replace the line ending with a one-sentence conclusion "
            "as the experiment progresses.\n\n"
        )
    with log.open("a", encoding="utf-8") as f:
        f.write(f"- ⚪ {name} — scaffolded\n")


def cmd_list() -> None:
    print("canonical analyses (live in analyses/, drop into experiments/<date>_<kind>_*/plots/):")
    for k in CANONICAL:
        script = ROOT / "analyses" / f"{k}.py"
        marker = " " if script.exists() else " (missing!) "
        print(f"  {k:<12}{marker}python -m analyses.{k}")
    print()
    print("scaffold a new one-off experiment:  python new.py experiment <slug>")
    print("summarize an experiment folder:     python new.py summarize <slug>")


_RUN_PY_TEMPLATE = '''\
"""Experiment {name}.

Auto-scaffolded by new.py. Edit freely. Output PNGs/CSVs to PLOTS_DIR.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import io, metrics, loaders  # noqa: F401 — pull what you need

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"hello from {name}; write plots to {{PLOTS_DIR}}")


if __name__ == "__main__":
    main()
'''

_NOTES_TEMPLATE = '''\
# {name}

Started: {today}
Status: scaffolded

---

## Guardrail gate  (FILL IN BEFORE LAUNCHING ANY COMPUTE)

The two most expensive mistakes are running the wrong experiment and running one whose
outcomes you can't interpret. The three sections below catch both. **Confirm them with
the collaborator (human or assistant) before launching.** Full rationale and rules in
[experiments/METHODOLOGY.md](../METHODOLOGY.md).

### Hypothesis
The exact thing being tested, stated so it could be wrong. (One or two sentences. Confirm
this is the question worth answering — not a near-miss variant.)

### Expected outcomes and what each would mean
- **If X** → conclude …
- **If null (no effect)** → conclude …
- **Trap case**: if W (result that *looks* like signal but isn't) → …

If you can't say what a result would mean, the experiment isn't designed yet.

### Soundness
Either propose concrete improvements (controls, seeds, confounds to kill, metrics) or
explicitly mark the setup ready. Apply the discipline checklist below.

---

## Method
What you actually ran.

- **Cells (conditions)**: one property per cell, plus a floor/baseline cell to show the
  metric's dynamic range. Any two cells differing in >1 way are uninterpretable.
- **Stressor swept**: the axis along which "does it matter" becomes "WHEN does it matter".
  (Less data, more noise, smaller model, harder distribution, deeper layer, etc.)
- **Seeds**: per-cell n; same seeds across cells if at all possible. Split the RNG into
  independent axes (init / setup / data) if more than one source of randomness exists —
  otherwise "low variance" can be an artifact of design, not a property of the condition.

## Discipline checklist
- [ ] Cells differ in exactly one property; floor cell included
- [ ] Stressor sweep (not one operating point) if "when" is the question
- [ ] Same seeds across cells; report PAIRED per-seed diffs, not raw means
- [ ] Spread-vs-σ check before claiming any ranking
      (avg within-cell σ vs between-cell spread of means)
- [ ] Pure-noise σ (deterministic cell) < smallest gap of interest — else declare indistinguishable
- [ ] RNG axes split if multiple sources of randomness
- [ ] Mechanism check: one actual-output figure (reconstructions / heatmaps / samples) alongside every metric figure
- [ ] If running on a cluster: manifest.txt generated by code, chunk runner respects
      scheduler caps, completion marker written LAST, analysis purges zero-byte artifacts
      and reports drops

---

## Findings
Numbers, plots (under `plots/`), per-seed diffs not just means. Pair every metric with an
actual-output figure when possible.

## Caveats
What you know is fragile about these numbers (sample size, scale of effect vs σ, single
seed feeding multiple randomness sources, etc.). Be explicit about the noise floor.

## Verdict
Keep going / dead end / promote to `analyses/` / **indistinguishable at this N** /
**retracted**.

Update this section honestly as the experiment progresses. **Retract overclaims in place** —
edit the section above and the LOG.md one-liner; don't leave the overclaim standing. A
retracted overclaim is a better record than a standing one.
'''


# ---------- summarize ----------

# Filename-pattern → (description, computation, interpretation). {0}, {1} are regex groups.
# Description is plain text (used in notes.md). Computation/interpretation may use $...$ math
# (used only in the LaTeX snippet, NOT escaped).
ARTIFACT_INFO: list[tuple[str, str, str, str]] = [
    (
        r"^u_axes_pPU_pI\.png$",
        "U–P_U and U–I across layers",
        r"For each hidden layer $L$, build centered Gram matrices "
        r"$K_X = X_c X_c^{\!\top}$ ($X_c = X - \bar X$) from the per-row bracket embeddings "
        r"$X \in \{U, P_U, I\}$ over the $N$ stimuli. Plot linear CKA "
        r"$\langle K_U, K_X\rangle_F / (\|K_U\|_F\,\|K_X\|_F)$ vs $L$ for each pair.",
        r"How aligned the answer's geometry is with the paraphrase ($P_U$) and the inference ($I$) "
        r"at each depth — higher means the two brackets reorder stimuli the same way. Where the two "
        r"curves cross and how the gap evolves reveal when the model commits to a semantic vs "
        r"pragmatic binding.",
    ),
    (
        r"^u_axes_layer_vs_layer\.png$",
        "Layer×layer CKA: U vs P_U and U vs I",
        r"Left: $M_{P_U}[i,j] = \mathrm{CKA}(K_U[i], K_{P_U}[j])$. Right: same with $K_I$. "
        r"Diagonal cells reproduce the within-layer curves; off-diagonals are cross-depth alignment.",
        r"Off-diagonal brightness flags delayed binding: $U$'s geometry at layer $i$ already matches "
        r"the other bracket's geometry at a different layer $j$. Diagonal block structure suggests "
        r"depth stages over which the alignment stabilizes.",
    ),
    (
        r"^cka_blocks_alllayers_(.+)\.png$",
        "Bracket-pair CKA across all layers (pooling: {0})",
        r"For every layer, every pair of bracket representations is compared via linear CKA "
        r"after per-bracket sample centering. One curve per unordered bracket pair, plotted vs "
        r"layer index.",
        r"Reads as a depth-wise map of which brackets bind together. Rising curves = emerging "
        r"alignment; flat curves = the model keeps the two brackets representationally separate.",
    ),
    (
        r"^layer_vs_layer_(.+)\.png$",
        "Layer×layer CKA per bracket (pooling: {0})",
        r"For each bracket separately, $\mathrm{CKA}(K[i], K[j])$ over all layer pairs — a "
        r"Kornblith-style layer-similarity matrix.",
        r"Diagonal trivially 1. Off-diagonal block structure exposes representational stages within "
        r"a single bracket: depths where the geometry is preserved vs depths where it reorganizes.",
    ),
    (
        r"^cka_blocks_L(\d+)_(.+)\.png$",
        "Block CKA supermatrix at layer {0} (pooling: {1})",
        r"At one layer, build the $(K\!\cdot\!N)\times(K\!\cdot\!N)$ centered-cosine supermatrix "
        r"for all bracket pairs. Side panel: the $K\times K$ linear-CKA summary.",
        r"Single-layer snapshot. Diagonal blocks are within-bracket, off-diagonals are cross-bracket. "
        r"Useful for visually validating an interesting depth picked from the all-layers plot.",
    ),
    (
        r"^contrast\.png$",
        "Two-sentence contrastive cos(sentence, target)",
        r"From paired fwd/rev runs, center each bracket vector by a shared per-layer mean, "
        r"L2-normalize, and compute $\cos(\text{sentence}, \text{target})$ for target "
        r"$\in \{Q, A, \text{final}\}$. Solid = $O_2$ (contrastive — saw the other sentence), "
        r"dashed = $O_1$ (blind).",
        r"A widening $O_2$ vs $O_1$ gap means contextualization is sharpening the S4/P "
        r"discrimination — attending to the alternative changes the representation. A closed gap "
        r"means the causal mask is doing all the work and there is no extra contrastive signal.",
    ),
    (
        r"^decision_binary\.png$",
        "Binary 0/1 decision-token distribution",
        r"For paired 0/1 fwd/rev runs, count emitted answers per direction. Decompose into "
        r"$\text{content} = (a + (1-b))/2$ (1 leans P) and "
        r"$\text{position} = (a+b)/2$ ($>0.5$ recency, $<0.5$ primacy).",
        r"Content far from $0.5$ = real S4/P preference; position far from $0.5$ = positional bias "
        r"(the model picks slot 1 or slot 2 regardless of content). Large position bias with small "
        r"content signal indicates behavioural noise, not preference.",
    ),
    (
        r"^decision_likert\.png$",
        "Likert decision-token distribution",
        r"Paired fwd/rev Likert (1..5) answers, decomposed into $c = (a + (6-b))/2$ and "
        r"$p = (a + b - 6)/2$.",
        r"$c \approx 3$ = no S4/P preference; $p < 0$ = primacy (Oración 1 favoured), $p > 0$ = "
        r"recency. Same logic as the binary version, with finer-grained answers.",
    ),
    (
        r"^projection\.png$",
        "Answer projected onto S4→P axis (per layer)",
        r"Per row and per layer, "
        r"$t = \langle a - s_4,\, p - s_4 \rangle / \|p - s_4\|^2$, plus the off-axis residual "
        r"$\|(a - s_4) - t (p - s_4)\| / \|p - s_4\|$. Invariant to translation, rotation, scale.",
        r"$t = 0$ at paraphrase ($S_4$), $t = 1$ at inference ($P$). Trajectory across layers shows "
        r"where the answer lives between the two anchors. Large residual = the 1D summary is "
        r"misleading.",
    ),
    (
        r"^logit_vs_residual\.png$",
        "Logit-lens vs residual-space cos",
        r"Decompose $\cos(\text{sentence}, \text{target})$ into content ($P - S_4$) and position "
        r"($O_2 - O_1$) in two spaces: plain residual stream (solid) and after the final RMSNorm "
        r"composed with $W_U$ (logit-lens readout, dashed).",
        r"If readout space amplifies position relative to residual space, the model's output "
        r"behaviour is more positional than its internal representation. Compare the gap to the "
        r"actual emitted decision token to corroborate.",
    ),
    (
        r"^.+_curves\.png$",
        "Per-layer scalar curves",
        r"Generic per-layer scalar series (cosine, CKA, projection coordinate, ...) plotted vs layer "
        r"index. See the experiment's run.py for the exact metric.",
        r"Compare curves directly. Layer-wise trends reveal where in depth the measured signal "
        r"develops or saturates.",
    ),
    (
        r"^.+_layer_vs_layer\.png$",
        "Layer×layer heatmaps",
        r"Generic $L \times L$ similarity heatmap (CKA or centered cosine). See the experiment's "
        r"run.py for which brackets / metric.",
        r"Diagonal trivially 1; off-diagonal structure reveals cross-depth alignment between the "
        r"brackets being compared.",
    ),
]


def _subst(template: str, groups: tuple) -> str:
    """Replace `{0}`, `{1}`, ... with the regex groups. Plain str.format() can't be used
    because the templates contain LaTeX braces that would clash with named-field parsing."""
    out = template
    for i, g in enumerate(groups):
        out = out.replace("{" + str(i) + "}", g)
    return out


def _load_overrides(folder: Path) -> dict:
    """Per-experiment descriptions.json: keys are filenames relative to <folder>/plots/."""
    p = folder / "descriptions.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"  ! could not parse {p.relative_to(ROOT)}: {e}", file=sys.stderr)
        return {}


def artifact_info(path: Path, folder: Path | None = None) -> dict:
    """Return {description, method, interpretation} for a PNG. Resolution order:
       1. folder/descriptions.json (per-experiment override)
       2. ARTIFACT_INFO regex templates
       3. generic placeholder
    """
    if folder is not None:
        overrides = _load_overrides(folder)
        plots_dir = folder / "plots"
        try:
            rel = path.relative_to(plots_dir).as_posix()
        except ValueError:
            rel = None
        for key in (rel, path.name):
            if key and key in overrides:
                o = overrides[key]
                return {
                    "title":          o.get("title"),
                    "description":    o.get("description", path.stem.replace("_", " ")),
                    "method":         o.get("method", ""),
                    "interpretation": o.get("interpretation", ""),
                }
    for pat, desc, method, interp in ARTIFACT_INFO:
        m = re.match(pat, path.name)
        if m:
            g = m.groups()
            return {
                "title":          None,
                "description":    _subst(desc, g),
                "method":         _subst(method, g),
                "interpretation": _subst(interp, g),
            }
    return {
        "title":          None,
        "description":    path.stem.replace("_", " "),
        "method":         "(no template registered for this filename — add it to descriptions.json "
                          "in this experiment folder, or to ARTIFACT_INFO in new.py.)",
        "interpretation": "(add a one-liner here once you know what it means.)",
    }


def describe_artifact(path: Path, folder: Path | None = None) -> str:
    return artifact_info(path, folder)["description"]


def _resolve_folder(slug: str) -> Path:
    """Accept exact folder name or a unique substring. Skips _-prefixed and __ entries."""
    candidates = sorted(p for p in (ROOT / "experiments").glob("*")
                        if p.is_dir() and not p.name.startswith("__"))
    exact = [c for c in candidates if c.name == slug]
    if exact:
        return exact[0]
    partial = [c for c in candidates if slug in c.name]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        listing = "\n  ".join(c.name for c in candidates if not c.name.startswith("_"))
        raise SystemExit(f"no experiment folder matches {slug!r}. Candidates:\n  {listing}")
    listing = "\n  ".join(c.name for c in partial)
    raise SystemExit(f"{slug!r} matches multiple folders:\n  {listing}\nuse the full name.")


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _build_summary_md(folder: Path, runs: list[dict], pngs: list[Path],
                      csvs: list[Path], other: list[Path]) -> str:
    rel = folder.relative_to(ROOT)
    lines = [
        f"_Auto-generated by `python new.py summarize` on {dt.datetime.now().isoformat(timespec='seconds')}._",
        "",
        f"**Folder:** `{rel}/`",
        f"**Plots:** {len(pngs)} PNG · **CSVs:** {len(csvs)} · **Other:** {len(other)} · **Invocations logged:** {len(runs)}",
        "",
    ]
    if runs:
        lines.append("### Invocations")
        for r in runs:
            argv = " ".join(r.get("argv", []))
            lines.append(f"- `{r.get('ts','?')}` — `{argv}`")
        lines.append("")
    if pngs:
        lines.append("### Plots")
        for p in pngs:
            rel_in_folder = p.relative_to(folder)
            size = _fmt_size(p.stat().st_size)
            lines.append(f"- `{rel_in_folder}` — {describe_artifact(p, folder)} ({size})")
        lines.append("")
    if csvs:
        lines.append("### CSVs")
        for c in csvs:
            rel_in_folder = c.relative_to(folder)
            size = _fmt_size(c.stat().st_size)
            lines.append(f"- `{rel_in_folder}` ({size})")
        lines.append("")
    if other:
        lines.append("### Other files")
        for o in other:
            rel_in_folder = o.relative_to(folder)
            size = _fmt_size(o.stat().st_size)
            lines.append(f"- `{rel_in_folder}` ({size})")
        lines.append("")
    return "\n".join(lines).rstrip()


def _escape_tex(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&") \
            .replace("%", r"\%").replace("#", r"\#").replace("$", r"\$").replace("{", r"\{") \
            .replace("}", r"\}")


def _title_from_folder(folder: Path) -> str:
    """Drop the YYYY-MM-DD_ prefix from the folder name and prettify for slide titles."""
    name = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", folder.name)
    return name.replace("_", " ").strip()


def _build_latex(folder: Path, pngs: list[Path]) -> str:
    """One pair of beamer frames per PNG (image + text). Paths are relative to project root."""
    folder_short = _title_from_folder(folder)
    head = (
        f"% Auto-generated by `python new.py summarize` on "
        f"{dt.datetime.now().isoformat(timespec='seconds')}.\n"
        f"% Folder: {folder.relative_to(ROOT)}/\n"
        f"% Paste into a beamer document. Adjust the graphicspath or paths below if "
        f"compiling from a different working directory.\n"
        f"%\n"
        f"% Requires: \\usepackage{{graphicx}}\n"
        f"\n"
    )
    if not pngs:
        return head + f"% (no PNGs found in {folder.relative_to(ROOT)}/)\n"

    plots_dir = folder / "plots"
    frames = []
    for p in pngs:
        rel = p.relative_to(ROOT)
        info = artifact_info(p, folder)
        # Title precedence: descriptions.json override -> description.
        # Prepend the bucket name for sweep analyses where the PNG sits in a subfolder.
        raw_title = info["title"] or info["description"]
        title_has_math = "$" in raw_title
        title_text = raw_title if title_has_math else _escape_tex(raw_title)
        rel_in_plots = p.relative_to(plots_dir)
        if len(rel_in_plots.parts) > 1:
            title_text = f"{_escape_tex(rel_in_plots.parts[0])} --- {title_text}"
        frames.append(
            "\\begin{frame}{" + title_text + "}\n"
            "  \\begin{center}\n"
            f"    \\includegraphics[width=0.95\\textwidth,height=0.82\\textheight,keepaspectratio]{{{rel.as_posix()}}}\n"
            "  \\end{center}\n"
            "\\end{frame}\n"
            "\n"
            "\\begin{frame}{" + title_text + "}\n"
            "  \\small\n"
            f"  \\textbf{{Computation.}} {info['method']}\n"
            "  \\par\\vspace{0.8em}\n"
            f"  \\textbf{{Interpretation.}} {info['interpretation']}\n"
            "\\end{frame}\n"
        )
    return head + "\n".join(frames)


_BEGIN = "<!-- BEGIN AUTO-SUMMARY -->"
_END = "<!-- END AUTO-SUMMARY -->"


def _upsert_summary(path: Path, slug: str, today: str, summary_md: str) -> None:
    """Insert / replace the auto-summary block, preserving handwritten content above & below."""
    block = f"{_BEGIN}\n## Auto-summary\n\n{summary_md}\n{_END}"
    if path.exists():
        text = path.read_text()
        if _BEGIN in text and _END in text:
            pre = text.split(_BEGIN, 1)[0].rstrip()
            post = text.split(_END, 1)[1].lstrip()
            new_text = (pre + "\n\n" + block + (("\n\n" + post) if post else "\n")).strip() + "\n"
        else:
            new_text = text.rstrip() + "\n\n" + block + "\n"
    else:
        # Folder didn't have notes yet (auto-folder from a canonical analysis).
        # Create the template scaffold + the auto-summary section.
        new_text = _NOTES_TEMPLATE.format(name=slug, today=today).rstrip() + "\n\n" + block + "\n"
    path.write_text(new_text)


def cmd_summarize(slug: str) -> None:
    folder = _resolve_folder(slug)
    plots_dir = folder / "plots"
    runs = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else []
    if plots_dir.exists():
        pngs = sorted(plots_dir.rglob("*.png"))
        csvs = sorted(plots_dir.rglob("*.csv"))
        other = sorted(p for p in plots_dir.rglob("*")
                       if p.is_file() and p.suffix not in {".png", ".csv"})
    else:
        pngs = csvs = other = []

    summary_md = _build_summary_md(folder, runs, pngs, csvs, other)
    latex = _build_latex(folder, pngs)
    today = dt.date.today().isoformat()
    _upsert_summary(folder / "notes.md", folder.name, today, summary_md)
    (folder / "latex_snippet.tex").write_text(latex)

    print(f"Folder: experiments/{folder.name}/")
    print(f"  invocations logged : {len(runs)}")
    print(f"  PNGs               : {len(pngs)}")
    print(f"  CSVs               : {len(csvs)}")
    print(f"  Updated  {(folder / 'notes.md').relative_to(ROOT)}")
    print(f"  Wrote    {(folder / 'latex_snippet.tex').relative_to(ROOT)}")


# ---------- entry point ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("experiment"); sp.add_argument("slug")
    sp = sub.add_parser("summarize"); sp.add_argument("slug")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "experiment":
        cmd_experiment(args.slug)
    elif args.cmd == "summarize":
        cmd_summarize(args.slug)


if __name__ == "__main__":
    main()
