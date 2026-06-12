"""Each invocation of a canonical analysis creates (or reuses) an experiment folder
under ``experiments/<YYYY-MM-DD>_<slug>/`` that holds its plots and a ``run.json``
log of how it was invoked. This makes experiments/ a chronological journal of work,
not just a place for hand-scaffolded one-offs.

Reuse policy: same (date, slug) → same folder. Re-runs append a new entry to
``run.json`` and overwrite any plots with the same filename. Cross-day reruns get
a new dated folder.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure(slug: str) -> Path:
    """Return ``experiments/<YYYY-MM-DD>_<slug>/plots/`` (creating it). Stamps run.json.

    The slug is used verbatim in the folder name — the caller should construct it from
    the analysis kind and any per-invocation discriminator, e.g.::

        ensure(f"cka_{alias}_{run.name}")
        ensure("u_axes")  # sweep — per-run subfolders go INSIDE plots/
    """
    date = dt.date.today().isoformat()
    folder = ROOT / "experiments" / f"{date}_{slug}"
    plots = folder / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    stamp = folder / "run.json"
    runs = []
    if stamp.exists():
        try:
            runs = json.loads(stamp.read_text())
        except Exception:
            runs = []
    runs.append({"argv": sys.argv, "ts": dt.datetime.now().isoformat(timespec="seconds")})
    stamp.write_text(json.dumps(runs, indent=2))
    return plots
