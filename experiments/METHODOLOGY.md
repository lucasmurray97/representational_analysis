# Experiment methodology

The discipline that keeps experiment conclusions in this project honest. Two halves:
**A. Scaffolding** (the mechanics — how each experiment is laid out and run) and
**B. Design logic** (the statistical/methodological rules that keep findings interpretable).

Read this before launching any non-trivial experiment. The notes.md template emitted by
`python new.py experiment <slug>` puts the guardrail gate at the top of every new
experiment for exactly the reasons spelled out below.

---

## A. Scaffolding — one self-contained directory per experiment

Each experiment is a folder under `experiments/<YYYY-MM-DD>_<slug>/`. The canonical
contents (some optional, depending on the experiment shape):

```
experiments/<YYYY-MM-DD>_<slug>/
  notes.md            # the experiment's brain — guardrail gate, method, findings, verdict
  run.py              # the analysis: load every finished run, score it, emit plots + CSV
  plots/              # output figures
  descriptions.json   # per-PNG title/method/interpretation overrides for new.py summarize
  latex_snippet.tex   # beamer frames, written by new.py summarize

  # Cluster-execution mechanics (only if running a manifest-driven sweep):
  make_manifest.py    # generate manifest.txt as the full cross-product of cells × levels × seeds
  manifest.txt        # checked in so the exact run set is reproducible
  chunk_runner.sh     # pulls lines, self-loops, respects scheduler caps
```

Plus two cross-experiment files at `experiments/`:
- **`LOG.md`** — running one-line-per-experiment index (status emoji + one-sentence conclusion).
  Updated automatically by `python new.py experiment` on scaffold; edit the line as the
  experiment progresses. This is what you read first to recall the project's arc.
- **`METHODOLOGY.md`** (this file) — the discipline rules.

The canonical analyses pipeline (`analyses/u_axes.py`, `analyses/cka.py`, etc.) also writes
into `experiments/<date>_<kind>_<discriminator>/plots/` so canonical runs are journalled
the same way as one-offs. See `core/experiment_folder.py::ensure` for the convention.

### The guardrail gate (top of every notes.md) — DO NOT SKIP

Before ANY run is launched, the notes.md must contain — and you must confirm with the
collaborator — three sections:

1. **Hypothesis** — the exact thing being tested, stated so it could be wrong. Confirm with
   the human that it's the question worth answering, not a near-miss variant.
2. **Expected outcomes and their meaning** — enumerate every possible result and what each
   would let you conclude. Explicitly include the null and the **trap case** (the result
   that looks like signal but isn't). If you can't say what a result would mean, the
   experiment isn't designed yet.
3. **Soundness** — propose concrete improvements (controls, seeds, confounds to kill,
   metrics) or explicitly mark the setup ready for launch.

**Rationale**: this catches the two most expensive mistakes — running the wrong experiment,
and running an experiment whose outcomes you can't interpret — BEFORE spending compute.
The gate also forces the confounds discussion (see B) up front instead of after the fact.

### Manifest + chunk-runner pattern (cluster execution)

If running on a scheduler-managed cluster:

- `make_manifest.py` emits the **full cross-product** as text: one run per line, each line
  the complete CLI args + a unique output tag (`<COND>_<level>_<seed>`). Generating it in
  code (not by hand) means the run set is exact, reviewable, and regenerable.
- A small **chunk runner** pulls lines from the manifest and runs them, self-looping to
  respect the scheduler's per-user job/GPU caps. Submit N chunks; each grabs the next undone
  line until the manifest is exhausted. The analysis job is chained with a dependency so it
  fires when the runs finish.
- **Idempotency via a completion marker.** Each run writes a sentinel artifact on success
  (final output file / metrics dump). The runner skips any tag whose marker already exists,
  so re-launching after a partial failure only does the missing work.
- **Make the marker the LAST thing written.** Caveat learned the hard way: if a run can
  write a *partial* marker (empty file from disk-full / killed job), idempotency skips the
  broken run forever. The analysis step should also purge/ignore zero-byte or unreadable
  artifacts and report how many it dropped — silent truncation reads as "I covered
  everything" when you didn't.

### Results live in notes.md, not chat

Every finding, metric table, caveat, and final **Verdict** goes into the experiment's own
`notes.md` (and a one-liner into `LOG.md`). Chat is ephemeral; the notes are the durable
record the next person (or the next you) reads. **Retract overclaims in place** — edit
the notes, don't leave the overclaim standing. A retracted overclaim is a better record
than a standing one.

### Compute hygiene

If working on a shared cluster:
- All non-trivial compute through the scheduler — never the login/head node. Even a
  one-off "import the framework" counts and can get you kicked.
- Outputs to the scratch / workspace tier — never the source tree or `$HOME`.
- Raw shared datasets are read-only from their shared location; derived data stays in your
  workspace.

Port the specifics to your environment; the discipline (light ops interactive, real work
batched, outputs off the source tree) is what matters.

---

## B. Design logic — the rules that keep the conclusions honest

This is the part to transplant most carefully. The scaffolding just runs jobs; this is how
you avoid fooling yourself with results.

### 1. Ablation by isolation — one property per condition, everything else matched

To test whether property P matters, build a set of **conditions** ("cells") that differ in
P **and only P**, with everything else identical. Add cells that each break a *different*
single property so you can attribute any effect to a specific cause. Typical shape:

- a clean reference condition (all properties intact),
- one cell per property you suspect, each breaking exactly that property,
- a **floor / baseline cell** that breaks the property whose necessity you're most sure of —
  this gives you the metric's dynamic range and tells you whether the regime is too hard
  for the metric to discriminate anything.

The discipline is **matching**: if two cells differ in two ways, a difference between them
is uninterpretable. Most of the design effort goes into making the cells differ in exactly
one thing. This is harder than it sounds — see the seed-confound trap in §5.

### 2. "Does it matter?" → "WHEN does it matter?" (sweep a stressor)

A tie at one operating point is not the end. The interesting question is whether the tie
**survives stress**. Pick a stressor (less data, more noise, smaller model, harder
distribution, deeper layer, ...) and sweep it. The hypothesis becomes: *the conditions stay
glued until the stressor crosses some threshold, then diverge — and the threshold + which
one wins tells you what kind of structure actually buys you something, and where.*

- Draw the floor cell at every stressor level as a reference line. If everyone hits the
  floor, the regime is just too hard and it's not a result about your property.
- Use the **same seeds across cells at each level** so shared per-level difficulty cancels
  in paired comparisons (see §4).

### 3. Scale the rigor to the claim; retract overclaims in writing

Small n is fine for a first look, but state conclusions only at the resolution the data
supports. When a result is "means differ by 0.05 with per-seed σ of 0.08," the honest
finding is **"indistinguishable at this N"** — not a ranking.

If you've claimed a ranking that turns out to be at-noise, retract it in the notes and the
LOG. The earlier U-axes-by-choice writeup in this project (`experiments/2026-06-10_u_axes_by_choice/`)
read N=15 vs N=31 differences as a content/position story — the honest finding there was
"indistinguishable at this N." The retraction belongs in that experiment's notes.

### 4. Use the paired test; check spread-vs-σ before believing any ranking

Two cheap checks before you believe any between-cell difference:

- **Between-cell spread vs within-cell σ.** Compute the mean spread across cells and the
  average within-cell seed σ. If spread ≈ σ, the cells don't separate — full stop.
- **Paired-by-seed differences.** Because the same seeds run across cells, compare cells
  **per seed** (cell A seed s − cell B seed s). This cancels shared per-seed difficulty and
  is far more sensitive than comparing independent means. **Inspect the per-seed diffs**,
  not just their mean: a "+0.09 mean, same-sign" can turn out to be one big seed and two
  near-ties — one lucky seed, not a population effect. "Same-sign across all seeds" is the
  minimum bar; "same-sign and comparable magnitude" is the real one.

### 5. The seed-as-confound trap — make sure "seed" means the same thing in every cell

This is the subtle one and the most transplantable. If your run seed feeds **more than
one** source of randomness, conditions can have non-comparable variance and your paired
test quietly breaks.

Concretely: in earlier work, `--seed` fed BOTH the network initialization AND the
generation of the condition itself (some cells' geometry was drawn from the seed). So:

- conditions with a **deterministic** setup used the *identical* setup across all seeds —
  their seed-variance measured only init jitter around one fixed point (→ tiny σ);
- conditions with a **seed-drawn** setup got a different setup each seed — their
  seed-variance conflated init jitter WITH sampling a whole family of setups (→ large σ).

Consequences, all bad:

- Comparing σ across cells is apples-to-oranges (one is "noise around a point," the other
  is "spread of a distribution"). A suspiciously LOW variance in the deterministic cell is
  an artifact of the design, not a property of that condition. That's exactly the tell
  that surfaces this trap — "why is this cell's variance so low?"
- The paired test is compromised: pairing "det-cell seed s" with "drawn-cell seed s" shares
  only the init RNG; the drawn cell's seed-s instance is just one random draw. The pairing
  cancels init noise but not the draw — so an apparent win is partly "one fixed point vs a
  cloud."

**Fix: split the RNG axes.** Give every source of randomness its own seed (one for init,
one for setup/condition generation, one for data sampling). Then you can hold the setup
fixed and vary only init (every cell now measures the same quantity — noise around a
point), OR average a cell over many setup draws × fixed init (measures "is this family
worse on average"). These are *different questions*; decide which you're asking and seed
accordingly. A single seed feeding everything silently mixes them.

### 6. Don't chase an effect below the noise floor

Before spending 6–8× the seeds to resolve a small gap, check whether the gap is even
**resolvable**. The decisive test: one condition was deterministic (its variance is *pure*
init noise), and that pure-init σ was larger than every between-condition mean gap. If
irreducible noise on a single fixed condition already exceeds the differences between
conditions, those differences are below the experiment's resolution **no matter how you
allocate seeds** — more seeds tighten the bars but can't separate means that sit inside
the init-noise band.

At that point the honest move is "indistinguishable; not a usable lever," and you stop.
Reserve the expensive high-n rerun for *defensive rigor* (a reviewer demands it), not
because you expect the conclusion to move — and say so explicitly.

### 7. Mechanism check alongside the metric

A metric tie/divergence is more believable when you can say *why*. Keep a mechanistic
story ("once the input is locally ordered, the model gets the one thing it needs; global
shape is a detail it absorbs") and check it against the data. The story tells you where an
effect *could* hide (e.g. only at the most extreme stressor) and keeps you from inventing
effects the mechanism can't produce.

Pair every metric figure with an actual-output figure (reconstructions, samples,
heatmaps). Numbers hide failure modes that the eye catches instantly. The
`u_axes_layer_vs_layer.png` cross-bracket heatmaps in this project are a good example —
they often reveal "this binding starts at layer 12, not 18" structure that the within-layer
scalar curve can't.

---

## Checklist to port this to a new project

- [ ] `experiments/<date>_<slug>/` per experiment; shared eval module; `LOG.md` index.
- [ ] notes.md starts with the **guardrail gate** (hypothesis / outcomes+meaning /
      soundness), confirmed with a human before launch.
- [ ] `make_manifest.py` emits the full cross-product; chunk runner respects scheduler
      caps; completion marker written LAST; analysis purges partial/empty artifacts and
      reports drops.
- [ ] Ablation cells differ in ONE property each; include a floor cell.
- [ ] Sweep a stressor to find *when* the property matters, not just *if*.
- [ ] Same seeds across cells; report paired per-seed diffs (inspect diffs, not the mean).
- [ ] Check between-cell spread vs within-cell σ before claiming a ranking.
- [ ] Split the RNG axes (init / setup / data) so "seed" means the same thing everywhere.
- [ ] If pure-noise σ ≥ the gap, declare **indistinguishable** and don't chase it.
- [ ] Findings + caveats + Verdict written into notes.md; one-liner into LOG.md; retract
      overclaims in place.
- [ ] Compute off the head node; outputs off the source tree; shared data read-only.
