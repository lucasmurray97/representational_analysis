# Refactoring instructions: split a script-graveyard repo into core / analyses / experiments

Paste this whole document into a fresh Claude session in the target repo.

You will reorganize a flat directory of accumulated Python scripts into three layers:
**library**, **canonical analyses**, and **one-off experiments** — plus a scaffolder so
future experiments land in a structured slot instead of becoming dangling top-level files.

Adapt freely to the target repo's conventions. The structure below is the goal; the names
("core", "analyses", "experiments") are good defaults but rename them if the project
already uses something equivalent. Don't force the pattern onto repos that don't fit:
if it's a library, a service, or a notebook-heavy project, the three-bucket split is
probably wrong.

---

## Step 1 — Survey before touching anything

List every `.py` file at the root. For each one, classify it into one of three buckets:

- **Library code** — no `if __name__ == "__main__"`, no CLI, no side-effecting I/O at
  import time. Pure helpers other scripts import.
- **Canonical analysis** — has a `main()`, the user re-runs it regularly (multiple
  times in the recent transcript / git history), and it's *currently in rotation*.
- **One-off experiment** — has a `main()`, but it's been superseded, is exploratory,
  or hasn't been run in a while. The user probably forgot it existed.

Ask the user to confirm the bucketing before moving anything. Don't guess — show them
the list and your proposed bucket for each. If the repo has 30+ files, ask which ones
they consider "core workflow" and which are dead.

Also list what's in the output directory (`plots/`, `figures/`, `results/`, whatever).
You'll be reorganizing that too.

---

## Step 2 — Find duplication across the canonical scripts

Before designing `core/`, grep the canonical scripts for repeated patterns:

- Any function that appears (verbatim or near-verbatim) in 2+ scripts → core candidate.
- Any I/O helper (load this artifact, parse this metadata, cache this result) → core.
- Any small math utility (a normalization, a similarity metric, a projection) → core.
- Don't pull domain-specific plotting into core — plotting belongs in analyses/.

Typical wins: shared dataset loaders, kernel/metric functions, model-checkpoint readers,
manifest parsers, anything that's been copy-pasted because "it's only 5 lines".

The library should be **small and useful**, not a junk drawer. If a helper has only
one caller, leave it in that caller's file.

---

## Step 3 — Build the layout

Create three sibling directories at the project root (or wherever the scripts live):

```
core/         pure library, no main(), no CLI. Imported by everything below.
analyses/     one .py per canonical analysis. Slim — heavy lifting is in core/.
experiments/  one folder per one-off, date-prefixed (YYYY-MM-DD_<slug>/).
              Each folder has run.py + notes.md. Allowed to be ugly.
```

Each directory gets an empty `__init__.py` so it's a proper package.

Sub-modules to consider inside `core/` (only create what you actually need):

- `io.py` — save/load helpers, file format conventions.
- `loaders.py` — domain-specific data loading (the part that knows about your
  manifest format, your conventions, your row layout).
- `metrics.py` — pure math.
- `<domain>.py` — anything else cross-cutting (model readers, registry, etc.).

Don't preemptively create empty modules. Add a file to core/ only when there's
duplication to remove or a clear shared concept.

---

## Step 4 — Bootstrap pattern for analyses scripts

So that `python -m analyses.foo` AND `python analyses/foo.py` both work, put this at
the top of every analyses script and every experiment `run.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import io, metrics, loaders  # whatever you need
```

For experiments two levels deep (`experiments/2026-06-08_foo/run.py`), use
`parents[2]` instead of `parent.parent`.

The reason: invoked as `-m`, Python adds the project root to sys.path automatically.
Invoked as a file, it adds the script's directory. The two-line bootstrap normalizes
both cases. It's ugly but it's worth keeping the muscle memory of either form.

---

## Step 5 — Rewrite the canonical analyses

For each canonical analysis script:

1. Move it from `<root>/script_name.py` to `analyses/<short_name>.py`.
2. Add the bootstrap (Step 4).
3. Replace duplicated logic with calls into `core/`.
4. Keep the CLI / argparse setup, the orchestration, and the plotting in the
   analysis file. Those are not library code.
5. Update the output path to follow the new plot layout (Step 7).

Aim for analyses scripts in the 80–250 LOC range. If one is much larger, ask whether
something inside it should be promoted to core/, or whether it's actually two analyses
glued together.

When two analyses overlap heavily (e.g. one is the "binary" version of another), merge
them with a flag (`--binary`), don't keep two copies.

---

## Step 6 — Move one-off experiments

Move the one-offs to `experiments/`. Either flatten or use date-prefixed folders;
the latter is better if experiments tend to grow plot/notes companions:

```
experiments/2026-06-08_some_exploration/
    __init__.py
    run.py
    notes.md
```

Fix their imports (bootstrap pattern, `from core import …`) just enough that they
still run. Don't refactor them — they're frozen exploratory work.

If the user has experiments that are clearly dead (the function they tested has been
inlined into a canonical analysis, the question was answered, etc.), ask before deleting.
Default to keeping them: dated folders sort chronologically and `notes.md` records the
verdict, so dead experiments don't hurt.

---

## Step 7 — Reorganize the output / plots directory

Adopt one path convention everywhere:

```
plots/<kind>/<run_or_target>/<artifact>.png
```

Where `<kind>` matches the analysis name (one folder per analysis: `plots/cka/`,
`plots/contrast/`, ...). `<run_or_target>` is something that uniquely identifies the
input (model alias + run name, or a tag the analysis builds from its arguments).
`<artifact>.png` is a fixed filename per artifact type so re-runs overwrite cleanly.

For experiments: `plots/experiments/<same-dated-slug>/`. The slug matches the
experiment folder so they're easy to pair up.

**Migration**: scan the existing flat plots/ dir, bucket files by filename prefix
(`contrast_*` → `plots/contrast/`, etc.), and move them. If the original filenames
encode the target/run mangled together, just bulk-move into `plots/<kind>/` as flat
files; don't reverse-engineer a folder hierarchy you don't have ground truth for.
Anything orphaned goes into `plots/experiments/_legacy/`.

Then update every analyses script's output path to write to the new structured location.

---

## Step 8 — Build a `new.py` scaffolder at the project root

A single-file CLI with this shape:

```
python new.py list                       # show kinds + how to invoke each analysis
python new.py experiment <slug>          # creates experiments/<date>_<slug>/{run.py, notes.md}
                                         # and plots/experiments/<date>_<slug>/
python new.py <canonical_kind> <slug>    # just mkdir -p plots/<kind>/<slug>/
```

The experiment scaffold should write a `run.py` with the bootstrap, the `core` imports
the user is likely to want, and a `PLOTS_DIR` pointing at the matching plot folder.
The `notes.md` should have a tiny template (Question / Method / Findings / Verdict)
the user can fill in.

Use `datetime.date.today().isoformat()` for the prefix. Slugify the user's input
(`[^a-z0-9]+` → `_`). If the target folder already exists, error out — don't overwrite.

Date-prefixed folder names aren't valid Python module names (hyphens, leading digit),
so the scaffolder must print the **direct file** invocation as the run command, not
`-m experiments.<name>.run`.

---

## Step 9 — Verify

Smoke-test each canonical analysis once after the move:

- `python -m analyses.<name> <typical-args>` works.
- `python analyses/<name>.py <typical-args>` also works (bootstrap test).
- The output lands in the new structured plot location.
- One legacy experiment still runs after the import fixes.
- The scaffolder creates a working `run.py` that prints "hello" without errors.

For analyses that take real arguments, prefer running with a small/cheap input
(`--layers 0 12 24` instead of all layers) — the smoke test is structural, not
performance.

---

## Anti-patterns to avoid

- **Don't create core/ modules speculatively**. One caller → leave it in the caller.
  Wait for the duplication.
- **Don't refactor experiments**. They're frozen. Just fix imports.
- **Don't add a `utils.py` or `common.py`** to core. Name modules by what they hold.
- **Don't move scripts the user is actively running** without telling them first.
  Especially long-running training/extract scripts where rerunning is expensive.
- **Don't invent a pyproject.toml / setup.py** just to make imports work. The
  two-line sys.path bootstrap is uglier but has zero install ceremony.
- **Don't squash the date prefix from experiment folder names**. It's the
  cheapest sort order you'll ever get.

---

## When to stop

You'll know the refactor is done when:

- The project root has nothing executable except `new.py` and config files.
- Every `.py` is in one of the three buckets.
- A `grep -r 'def linear_cka\|def <other-suspected-duplicate>' analyses/` returns nothing.
- The plots directory has subfolders, not 40 sibling PNGs.
- The user can describe each canonical analysis in one sentence.

If the user disagrees with any bucketing along the way, follow their judgment. They
know which scripts they re-run; you don't.
