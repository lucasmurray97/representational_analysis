"""Fill a prompt template from a stimulus workbook, tracking each bracket's span.

A template is plain text with ``{placeholder}`` fields, e.g. the N0 prompt with
``{question} {answer} {sentence_1} {sentence_2}``. Placeholders are discovered
dynamically, so a template may use any subset / superset of brackets. Each
placeholder is resolved to a workbook column (built-in semantic map, else a
case-insensitive column-name match, else a user --field-map override). For every
row we return the filled text plus the character span of each field, which the
extractor maps to token spans for per-bracket pooling.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]\w*)\}")

# Semantic placeholder -> stimulus column (forward order: S4 = Oración 1, P = Oración 2).
DEFAULT_FIELD_MAP = {
    "question": "QUD",
    "answer": "S2",
    "sentence_1": "S4",
    "sentence_2": "P",
}

# Candidate reference sentences for the carrier slot when --sentences all is given
# (the answer S2 and the question QUD are excluded; they are the fixed context).
DEFAULT_SENTENCE_POOL = ["S1", "S3", "S4", "P"]


def find_placeholders(template: str) -> list[str]:
    """Unique placeholder names, in first-appearance order."""
    seen: list[str] = []
    for m in PLACEHOLDER_RE.finditer(template):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def resolve_columns(fields: list[str], columns: list[str], field_map: dict[str, str]):
    """Map each placeholder to a column. Returns (resolved {field: column}, missing [field])."""
    cols_lower = {c.lower(): c for c in columns}
    resolved, missing = {}, []
    for f in fields:
        col = None
        if f in field_map and field_map[f] in columns:      # explicit / semantic map
            col = field_map[f]
        elif f in columns:                                    # exact column name
            col = f
        elif f.lower() in cols_lower:                         # case-insensitive name
            col = cols_lower[f.lower()]
        if col is None:
            missing.append(f)
        else:
            resolved[f] = col
    return resolved, missing


def fill_with_spans(template: str, values: dict[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Substitute placeholders, returning (text, {field: (char_start, char_end)})."""
    result, spans, last = "", {}, 0
    for m in PLACEHOLDER_RE.finditer(template):
        result += template[last:m.start()]
        val = values.get(m.group(1), "")
        start = len(result)
        result += val
        spans[m.group(1)] = (start, start + len(val))
        last = m.end()
    result += template[last:]
    return result, spans


def build_items(template_path: str, xlsx_path: str,
                field_map: dict[str, str] | None = None, limit: int | None = None):
    """Return (items, fields). Each item: {id, text, field_spans}. fields: resolved placeholder list."""
    template = Path(template_path).read_text(encoding="utf-8")
    fields = find_placeholders(template)
    if not fields:
        raise SystemExit(f"No {{placeholders}} found in {template_path}.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style.*")
        df = pd.read_excel(xlsx_path)

    fmap = dict(DEFAULT_FIELD_MAP)
    if field_map:
        fmap.update(field_map)
    resolved, missing = resolve_columns(fields, list(df.columns), fmap)
    if missing:
        raise SystemExit(
            f"Template fields {missing} map to no column in {xlsx_path}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Pass --field-map '{{\"{missing[0]}\": \"<column>\"}}' to map them."
        )

    rows = df.head(limit) if limit else df
    items = []
    for i, (_, row) in enumerate(rows.iterrows()):
        values = {f: ("" if pd.isna(row[col]) else str(row[col])) for f, col in resolved.items()}
        text, spans = fill_with_spans(template, values)
        items.append({"id": str(i), "text": text, "field_spans": spans})
    print(f"Built {len(items)} prompts from {Path(template_path).name} x {Path(xlsx_path).name} | "
          f"fields: " + ", ".join(f"{f}->{c}" for f, c in resolved.items()))
    return items, list(resolved)


def resolve_sentence_columns(names: list[str], columns: list[str]) -> list[str]:
    """Map a --sentences selection ('all' or column names, case-insensitive) to real columns."""
    cols_lower = {c.lower(): c for c in columns}
    out: list[str] = []
    for n in names:
        if n.lower() == "all":
            for c in DEFAULT_SENTENCE_POOL:
                if c in columns and c not in out:
                    out.append(c)
        elif n in columns and n not in out:
            out.append(n)
        elif n.lower() in cols_lower and cols_lower[n.lower()] not in out:
            out.append(cols_lower[n.lower()])
        elif n.lower() != "all":
            raise SystemExit(f"--sentences '{n}' matches no column. Available: {columns}")
    if not out:
        raise SystemExit(f"--sentences resolved to nothing. Default 'all' pool is {DEFAULT_SENTENCE_POOL}.")
    return out


def build_carrier_items(template_path: str, xlsx_path: str, sentence_selection: list[str],
                        sentence_slot: str = "sentence", field_map: dict[str, str] | None = None,
                        limit: int | None = None):
    """Carrier mode: a template with a single {sentence_slot}; emit one prompt per row x
    per selected sentence column, with the other placeholders (question/answer) fixed.

    Returns (items, fields). Each item also carries 'sentence_col' (which column filled the slot).
    """
    template = Path(template_path).read_text(encoding="utf-8")
    fields = find_placeholders(template)
    if sentence_slot not in fields:
        raise SystemExit(f"Carrier template must contain {{{sentence_slot}}}; found {fields}.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style.*")
        df = pd.read_excel(xlsx_path)

    fmap = dict(DEFAULT_FIELD_MAP)
    if field_map:
        fmap.update(field_map)
    fixed_fields = [f for f in fields if f != sentence_slot]
    resolved, missing = resolve_columns(fixed_fields, list(df.columns), fmap)
    if missing:
        raise SystemExit(f"Carrier fixed fields {missing} map to no column. Columns: {list(df.columns)}")
    scols = resolve_sentence_columns(sentence_selection, list(df.columns))

    rows = df.head(limit) if limit else df
    items = []
    for i, (_, row) in enumerate(rows.iterrows()):
        base = {f: ("" if pd.isna(row[col]) else str(row[col])) for f, col in resolved.items()}
        for scol in scols:
            values = {**base, sentence_slot: ("" if pd.isna(row[scol]) else str(row[scol]))}
            text, spans = fill_with_spans(template, values)
            items.append({"id": f"{i}:{scol}", "text": text, "field_spans": spans, "sentence_col": scol})
    fixed_desc = ", ".join(f"{f}->{c}" for f, c in resolved.items())
    print(f"Built {len(items)} carrier prompts ({len(rows)} rows x sentences {scols}) | "
          f"fixed: {fixed_desc} | slot {{{sentence_slot}}}")
    return items, fields
