# Server handoff — representational_analysis

Paste this whole document into the new Claude session on the server as the opening
message. It sets up the repo, orients you on the codebase, and points at recent work.

## What this is

A Spanish-language semantics-vs-pragmatics LLM study. We compare how small open-weight
models (Qwen2.5-0.5B-Instruct, Llama-3.2-1B) represent **paraphrases vs. inferences**
in their hidden states. Concretely: for each stimulus we extract per-bracket span
embeddings (Q, U, P_U, I) and use **linear CKA** and related metrics across layers to
study where the model binds the answer (U) more tightly to the paraphrase (P_U) than to
the inference (I). The behavioural complement (which sentence the model emits as its
answer) lives in a sibling repo (qualitative_analysis, not in this one).

## 1 — Setup

```bash
git clone git@github.com:lucasmurray97/representational_analysis.git
cd representational_analysis

# Python 3.11 is what the lockfile expects (anything 3.10–3.12 should work).
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Hugging Face auth.** Llama-3.2-1B is gated. Before any extraction with that model:

```bash
huggingface-cli login   # paste a read token from huggingface.co/settings/tokens
```

Qwen2.5-0.5B-Instruct is ungated; no auth needed.

**GPU.** Default device is `auto` (device_map). On a CUDA box, extraction is fast
(~minutes for 272 stimuli, all layers). On CPU it's 10–20× slower but works.

## 2 — Sanity check

```bash
# Read-only — should print the canonical analyses.
python new.py list

# Render notes + LaTeX for an existing experiment. Should print
# "Updated experiments/.../notes.md".
python new.py summarize 2026-06-10_u_axes_by_choice
```

If both succeed, the environment is wired up.

## 3 — Codebase shape

Three-layer split. Don't mix them.

```
core/         Pure library. Imported by everything below; no main, no CLI.
              io.py           save/load helpers for embedding runs
              stimuli.py      prompt template + bracket-span tracker
              loaders.py      bracket-aware loaders (carrier vs p5_fwd/p5_rev triplet)
              metrics.py      linear_cka, kernel, unit_center, axis_projection, rmsnorm
              readout.py      load_unembedding + RMSNorm (logit-lens helpers)
              experiment_folder.py
                              ensure(slug) → experiments/<date>_<slug>/plots/, stamps run.json

analyses/     Canonical, re-run regularly. Slim — most logic is in core/.
              extract.py      run a HF causal LM, save per-layer hidden states + bracket pools
              cka.py          K×K block CKA across brackets (single-layer / all-layers / L×L)
              u_axes.py       U–P_U and U–I CKA curves + L×L cross matrices (sweep over runs)
              contrast.py     fwd/rev contrastive cos(sentence, target) decomposition
              decision.py     decision-token decomposition (Likert or --binary)
              logit_lens.py   readout-space cos via norm·W_U
              projection.py   answer projected on S4→P axis (carrier / blind / contrastive)

experiments/  One folder per one-off, dated. Each canonical analyses run also lands here
              as experiments/<date>_<kind>_<disc>/. Format:
                  run.json         appended log of every invocation (argv + ISO ts)
                  plots/           PNGs / CSVs / npz; npz is gitignored
                  notes.md         handwritten + auto-summary block
                  latex_snippet.tex  beamer frames for each PNG (image + text frame)
                  descriptions.json  (optional) per-PNG override for the LaTeX text
              _legacy/          pre-refactor one-off scripts (don't reuse, just refer to)
              _canonical_archive/ pre-migration plot snapshots (gitignored)
```

**Invocation** (from the project root, with .venv active):

```bash
python -m analyses.<kind> <args>      # canonical
python experiments/<folder>/run.py    # one-off; date-prefixed names aren't valid -m modules
python new.py experiment <slug>       # scaffold a new one-off folder
python new.py summarize <slug>        # write notes.md + latex_snippet.tex from artifacts
```

## 4 — What needs to be regenerated on first use

The repo ships **code + stimuli + plots + small CSVs**. It does NOT ship:

- `outputs/` — extraction safetensors. Gitignored. Regenerate with `analyses.extract`.
- `*.npz` files inside `experiments/*/plots/` — large kernel matrices. Gitignored.
  `u_axes.py` writes these as a cache.
- `.venv/` — rebuild from `requirements.txt`.
- `experiments/_canonical_archive/` — pre-migration archive, not shipped.

### Stimuli

Default workbook is `data/estimulos/items_completo.xlsx`. Columns: `QUD`, `QUD_polar`,
`U`, `PU`, `I`, `CT`, `CT2`, `CT3`, `CT4`, `NR`. Prompt placeholders should use column
names directly (`{QUD}`, `{U}`, `{PU}`, `{I}`, `{CT}`, ...). The legacy
`{question}/{answer}/{sentence_1}/{sentence_2}` placeholders still resolve via the
default field map in `core/stimuli.py` for backwards compatibility.

`data/estimulos/estimulos_completo_backup.xlsx` is the OLD stimuli (`QUD`, `S2`, `S4`,
`P`, ...). Pre-refactor extractions reference it; new work should use items_completo.

### Counterbalanced two-sentence runs

The new pattern uses `--counterbalance A B` to run both directions in a single
model load. The flag swaps `{A}` and `{B}` in the template for the second pass and
writes to `<set_name>_fwd` and `<set_name>_rev`.

```bash
# Two-sentence prompt 5, both directions (one model load).
python -m analyses.extract --model Qwen/Qwen2.5-0.5B-Instruct \
    --template data/prompts/prompt_5.txt --brackets all \
    --bracket-pooling mean bertscore --pooling last \
    --layers all --counterbalance I PU --set-name p5

# Carrier / N0: feed each candidate sentence column into the {sentence} slot,
# one prompt per row × sentence. Brackets in metadata get prompt_id "<row>:<col>"
# so carrier_indices() can recover them.
python -m analyses.extract --model Qwen/Qwen2.5-0.5B-Instruct \
    --template data/prompts/prompt_var_N0.txt \
    --sentences PU I --sentence-slot sentence_1 \
    --brackets all --bracket-pooling mean bertscore \
    --layers all --set-name qwen_mensaje

# Repeat with --model meta-llama/Llama-3.2-1B (after huggingface-cli login).
```

### Bracket naming notes

- With direct-column placeholders (`{QUD}/{U}/{PU}/{I}`): brackets are
  `bracket_QUD/bracket_U/bracket_PU/bracket_I`. Load with
  `load_triplet(kind="direct", answer_bracket="U", para_col="PU", infer_col="I")`.
- With legacy generic placeholders (`{question}/{answer}/{sentence_1}/{sentence_2}`):
  brackets are `bracket_question/.../bracket_sentence_2`. Load with `kind="p5_fwd"` /
  `kind="p5_rev"`.

Run names matter: `analyses/u_axes.py` has a hardcoded `RUNS` list pointing at
specific run folders. Either match the names or edit the list.

After extraction, run the analyses to regenerate plots:

```bash
python -m analyses.u_axes
python -m analyses.cka --run outputs/Qwen2.5-0.5B-Instruct/qwen_mensaje --all-layers
python -m analyses.cka --run outputs/Qwen2.5-0.5B-Instruct/qwen_mensaje --layer-vs-layer
python -m analyses.decision --fwd outputs/.../p5_fwd --rev outputs/.../p5_rev --binary
```

## 5 — Gotchas

- **Llama-3.2-1B is a base model.** On the binary 0/1 prompt (`prompt_5.txt`), its
  `answer_text` is whitespace for every stimulus — no usable choice signal. For
  decision-style analyses use Qwen-Instruct, or switch to a Llama-Instruct.
- **Qwen2.5-0.5B has severe primacy bias.** On `prompt_5`, 79% of fwd answers are "0"
  (slot 1) regardless of content. Only ~46 of 272 stimuli show a content-consistent
  choice across fwd∩rev. See `experiments/2026-06-10_u_axes_by_choice/` for the split.
- **Tied embeddings.** Both target models have `tie_word_embeddings: true` — the
  checkpoint has `model.embed_tokens.weight` but no `lm_head.weight`. `core/readout.py`
  handles the fallback.
- **CPU-only.** If running on CPU, pass `--dtype fp32 --device cpu` to extract.py.
- **Path discipline.** Canonical analyses write to
  `experiments/<YYYY-MM-DD>_<kind>_<discriminator>/plots/`. Same (date, slug) → same
  folder; re-runs append to `run.json` and overwrite plots.

## 6 — Where I left off

The most recent strand of work was about **pooling alternatives** because mean-pooled
cosine is anisotropy-biased (Mitra & Kumar 2026, "Mean-Pooled Cosine Similarity is Not
Length-Invariant"). The existing CKA analyses are already length-invariant, but the
user asked about BERTScore.

What's currently implemented in `analyses/extract.py`:

- `--bracket-pooling bertscore` adds **IDF-weighted mean pooling**. IDF is computed
  in a single tokenizer pre-pass over the bracket spans of the run, per bracket field:
  `idf(t) = ln((N+1) / (df(t)+1)) + 1`. Saved as
  `pooled_bracket_<field>_bertscore.safetensors`. Loaded via
  `core.loaders._bracket_suffix("bertscore")` → `"_bertscore"` suffix. cka.py supports
  `--bracket-pooling bertscore`.

Important nuance — what I added is NOT BERTScore proper:

- BERTScore (Zhang et al. 2019) is a pairwise F1 similarity between two token sets
  via greedy max-matching. There is no "pool one span into one vector" step.
- IDF-weighted mean is what BERTScore uses to weight per-token contributions; I lifted
  that into a pooling. It's a half-measure but plug-compatible with existing CKA/cosine
  analyses.

If the user comes back asking for "real BERTScore", the missing pieces are:

1. A bracket-pooling mode that saves per-token vectors per span (ragged shape; can use
   the existing `--pooling all` machinery as a base).
2. A `core/bertscore.py` helper: `bertscore_f1(toks_A, toks_B) → (P, R, F1)` using
   greedy max-cosine matching.
3. A new analysis (e.g. `analyses/u_axes_bertscore.py`) that computes per-stimulus,
   per-layer F1 between U and {P_U, I}, then aggregates.

The third design alternative — what the paper actually recommends — is **CKA on
shared-position token matrices** (length-invariant without going via similarity scores).
The user hasn't asked for that yet; flag it if they revisit pooling.

## 7 — Useful background reads in the repo

- `README.md` — high-level overview.
- `REFACTOR_INSTRUCTIONS.md` — the recipe used to split this repo into core/analyses/
  experiments. Same pattern can be applied to sibling repos.
- The most recent experiment with hand-written method/interpretation text:
  `experiments/2026-06-10_u_axes_by_choice/descriptions.json` — good template for
  authoring future per-experiment LaTeX overrides.

## 8 — Workflow for the user's typical session

1. `python new.py experiment <slug>` to start something exploratory; edit `run.py`,
   put plots in `PLOTS_DIR`, jot notes in `notes.md`.
2. When useful, `python new.py summarize <slug>` to regenerate the auto-summary block
   in notes.md and emit a beamer-ready `latex_snippet.tex`.
3. For per-PNG title / method / interpretation control, create
   `experiments/<slug>/descriptions.json` keyed by filename relative to `plots/`.
4. If a one-off matures, promote: copy shared bits to `core/`, write a thin
   `analyses/<kind>.py`, retire the experiment folder.

The user prefers terse, substance-first communication. Skip recaps and adornment.
