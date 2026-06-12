"""Extract hidden-state embeddings from an arbitrary causal LM for arbitrary prompts.

Prompt sources:
  --prompts FILE   plain .txt (one per line) or .jsonl ({"id":..,"text":..})
  --template FILE  a prompt template with {placeholders}; values are pulled per row
                   from --stimuli (an .xlsx), tracking each bracket's char span.

Pooling (per layer):
  --pooling whole      mean over all prompt tokens
            last       the last (answer-position) token, padding-safe
            all        every token (ragged -> flat [sum_tokens, d] + token index)
  --brackets all|F...  per-bracket span pooling (mean over each field's tokens);
                       requires --template and a fast tokenizer.

For each prompt it also records the model's next-token answer (argmax of the last
-position logits). Results are saved via core.io.save_run.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import time

import numpy as np

from core import io as embeddings_io

POOLINGS = ("whole", "last", "all")
# Bracket pooling modes:
#   mean        - uniform mean over the span tokens (default).
#   last        - the final token of the span.
#   bertscore   - IDF-weighted mean over the span tokens. IDF is computed per bracket
#                 field as document-frequency over the bracket spans of THIS run:
#                 idf(t) = ln((N+1)/(df(t)+1)) + 1 with smoothing. Down-weights tokens
#                 that appear in many stimuli (function words, repeated frame tokens),
#                 up-weights distinctive content tokens. Matches the IDF weighting
#                 introduced by BERTScore (Zhang et al. 2019), applied as a pooling.
BRACKET_POOLINGS = ("mean", "last", "bertscore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Alias in models_local.json, or a raw HF repo id.")
    p.add_argument("--models-config", default="models_local.json")
    p.add_argument("--prompts", default=None, help=".txt (one per line) or .jsonl ({id,text}).")
    p.add_argument("--prompt", action="append", default=[], help="Inline prompt; repeatable.")
    p.add_argument("--template", default=None, help="Template with {placeholders} (enables --brackets).")
    p.add_argument("--stimuli", default="data/estimulos/estimulos_completo.xlsx",
                   help="Workbook supplying values for the template placeholders.")
    p.add_argument("--field-map", default=None, help="JSON mapping placeholder->column, e.g. '{\"answer\":\"S2\"}'.")
    p.add_argument("--sentences", nargs="+", default=None,
                   help="Carrier mode: which column(s) fill the {sentence} slot, one prompt each: "
                        "'all' or names like s4 p s1 (case-insensitive). Requires --template with the slot.")
    p.add_argument("--sentence-slot", default="sentence", help="Placeholder name of the carrier slot.")
    p.add_argument("--layers", nargs="+", default=["-1"], help="hidden_states indices; negatives / 'all' ok.")
    p.add_argument("--pooling", nargs="+", default=["whole"], choices=POOLINGS)
    p.add_argument("--brackets", nargs="+", default=None,
                   help="'all' or specific placeholder names to pool per-bracket (needs --template).")
    p.add_argument("--bracket-pooling", nargs="+", default=["mean"], choices=BRACKET_POOLINGS,
                   help="How to pool each bracket span. 'mean' = mean over the span; "
                        "'last' = the final token of the span. Pass both to save both.")
    p.add_argument("--raw", action="store_true", help="Skip the chat template (tokenize the prompt as-is).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="auto", help="'auto' (device_map) or e.g. 'cuda:0' / 'cpu'.")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-answer", action="store_true")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--set-name", default=None)
    return p.parse_args()


def resolve_model(spec: str, config_path: str) -> tuple[str, str, bool]:
    path = Path(config_path)
    registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if spec in registry:
        e = registry[spec]
        return spec, e["hf_id"], bool(e.get("chat", True))
    return spec.split("/")[-1], spec, True


def load_plain_prompts(args) -> list[dict]:
    items: list[dict] = []
    if args.prompts:
        path = Path(args.prompts)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for i, line in enumerate(text.splitlines()):
                if line.strip():
                    obj = json.loads(line)
                    items.append({"id": str(obj.get("id", i)), "text": obj["text"], "field_spans": {}})
        else:
            for i, line in enumerate(text.splitlines()):
                if line.strip():
                    items.append({"id": str(i), "text": line, "field_spans": {}})
    for j, t in enumerate(args.prompt):
        items.append({"id": f"inline-{j}", "text": t, "field_spans": {}})
    if not items:
        raise SystemExit("No prompts. Use --prompts/--prompt, or --template with --stimuli.")
    return items


def resolve_layers(spec: list[str], n_states: int) -> list[int]:
    if len(spec) == 1 and spec[0].lower() == "all":
        return list(range(n_states))
    out = set()
    for tok in spec:
        i = int(tok)
        i = i + n_states if i < 0 else i
        if not 0 <= i < n_states:
            raise SystemExit(f"Layer {tok} out of range; model has {n_states} hidden states (0..{n_states-1}).")
        out.add(i)
    return sorted(out)


def last_token_indices(attention_mask):
    """Last attended token per row, robust to left/right padding."""
    seq_len = attention_mask.shape[1]
    return seq_len - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)


def main() -> None:
    args = parse_args()
    poolings = list(dict.fromkeys(args.pooling))

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd

    alias, hf_id, chat_default = resolve_model(args.model, args.models_config)
    use_chat = chat_default and not args.raw
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"Loading {hf_id} ({args.dtype}, device={args.device}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if use_chat and tok.chat_template is None:
        print("  (no chat template on this model -> falling back to --raw)")
        use_chat = False

    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    load_kwargs = {dtype_kw: dtype}
    if args.device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    if args.device != "auto":
        model.to(args.device)
    model.eval()
    device = next(model.parameters()).device

    template_fields: list[str] = []
    carrier_mode = bool(args.sentences)
    if carrier_mode and not args.template:
        raise SystemExit("--sentences requires --template (a carrier with a {sentence} slot).")
    if args.template:
        from core import stimuli
        field_map = json.loads(args.field_map) if args.field_map else None
        if carrier_mode:
            items, template_fields = stimuli.build_carrier_items(
                args.template, args.stimuli, args.sentences, args.sentence_slot, field_map, args.limit)
        else:
            items, template_fields = stimuli.build_items(args.template, args.stimuli, field_map, args.limit)
    else:
        items = load_plain_prompts(args)
        if args.limit is not None:
            items = items[: args.limit]

    bracket_fields: list[str] = []
    if args.brackets:
        if not args.template:
            raise SystemExit("--brackets requires --template.")
        bracket_fields = template_fields if "all" in args.brackets else args.brackets
        unknown = [f for f in bracket_fields if f not in template_fields]
        if unknown:
            raise SystemExit(f"--brackets {unknown} not in template fields {template_fields}.")
    elif carrier_mode:
        bracket_fields = template_fields
    bracket_on = bool(bracket_fields)
    if bracket_on and not tok.is_fast:
        raise SystemExit("--brackets needs a fast tokenizer (offset mapping); this model has a slow one.")

    n_states = model.config.num_hidden_layers + 1
    layers = resolve_layers(args.layers, n_states)
    print(f"{len(items)} prompts | layers {layers} | pooling {poolings} | "
          f"brackets {bracket_fields or '-'} | chat_template={use_chat}", flush=True)

    def render(item):
        raw, spans = item["text"], item.get("field_spans", {})
        if use_chat:
            s = tok.apply_chat_template([{"role": "user", "content": raw}],
                                        add_generation_prompt=True, tokenize=False)
            base = s.find(raw)
            if base >= 0:
                spans = {f: (a + base, b + base) for f, (a, b) in spans.items()}
            return s, spans
        return raw, spans

    finals = [render(it) for it in items]

    pooled = {p: {l: [] for l in layers} for p in poolings if p in ("whole", "last")}
    tokens_all = {l: [] for l in layers} if "all" in poolings else {}
    bracket_modes = list(dict.fromkeys(args.bracket_pooling))
    bracket_acc = {m: {f: {l: [] for l in layers} for f in bracket_fields} for m in bracket_modes}
    bracket_cnt = {f: [] for f in bracket_fields}
    tok_index_rows, meta_rows = [], []

    # BERTScore pooling needs IDF per (bracket field, token id), computed over the bracket
    # spans of the entire run. We tokenize each prompt once, collect the unique token ids
    # falling inside each bracket span, then form a per-vocab lookup tensor we can index by
    # input_ids in the main loop.
    idf_lookup: dict[str, "torch.Tensor"] = {}
    if "bertscore" in bracket_modes and bracket_on:
        import math
        print("Computing IDF for bertscore pooling (one tokenizer pass) ...", flush=True)
        N = len(items)
        df_per_field: dict[str, dict[int, int]] = {f: {} for f in bracket_fields}
        for (text, sp) in finals:
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=not use_chat)
            ids_i = enc["input_ids"]
            offs_i = enc["offset_mapping"]
            for f in bracket_fields:
                s, e = sp.get(f, (0, 0))
                if e <= s:
                    continue
                seen = set()
                for k, (a, b) in enumerate(offs_i):
                    if a == 0 and b == 0:
                        continue
                    if b > s and a < e:
                        seen.add(int(ids_i[k]))
                for tid in seen:
                    df_per_field[f][tid] = df_per_field[f].get(tid, 0) + 1
        vocab = model.config.vocab_size
        idf_dtype = torch.float32  # weights are cheap; keep in fp32 for stable normalization
        for f in bracket_fields:
            t = torch.zeros(vocab, dtype=idf_dtype, device=device)
            for tid, c in df_per_field[f].items():
                t[tid] = math.log((N + 1) / (c + 1)) + 1.0
            idf_lookup[f] = t
            seen_n = len(df_per_field[f])
            pos = t[t > 0]
            print(f"  IDF[{f}]: {seen_n} unique token ids, "
                  f"min/median/max IDF = {pos.min().item():.2f} / "
                  f"{pos.median().item():.2f} / {pos.max().item():.2f}", flush=True)

    def to_np(t):
        return t.detach().to(torch.float16).cpu().numpy()

    t0 = time.time()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        texts = [finals[start + k][0] for k in range(len(chunk))]
        spans = [finals[start + k][1] for k in range(len(chunk))]
        enc = tok(texts, padding=True, truncation=bool(args.max_length), max_length=args.max_length,
                  return_tensors="pt", add_special_tokens=not use_chat,
                  return_offsets_mapping=bracket_on)
        offsets = enc.pop("offset_mapping").to(device) if bracket_on else None
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        B, T = input_ids.shape

        position_ids = attn.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attn == 0, 1)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
                        output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hs = out.hidden_states
        rows = torch.arange(B, device=device)
        last_idx = last_token_indices(attn)
        mask_f = attn.unsqueeze(-1).to(hs[layers[0]].dtype)
        denom = mask_f.sum(dim=1).clamp(min=1)

        if "all" in poolings:
            flat_mask = attn.reshape(-1).bool()
            sel = flat_mask.nonzero(as_tuple=False).squeeze(-1)
            sel_row = (sel // T).tolist()
            within = (attn.cumsum(dim=1) - 1).reshape(-1)[sel].tolist()
            sel_ids = input_ids.reshape(-1)[sel].tolist()
            sel_str = tok.convert_ids_to_tokens(sel_ids)
            for r, pos, tid, tstr in zip(sel_row, within, sel_ids, sel_str):
                tok_index_rows.append({"row": start + r, "prompt_id": chunk[r]["id"],
                                       "token_pos": int(pos), "token_id": int(tid), "token_str": tstr})

        for l in layers:
            h = hs[l]
            if "last" in pooled:
                pooled["last"][l].append(to_np(h[rows, last_idx]))
            if "whole" in pooled:
                pooled["whole"][l].append(to_np((h * mask_f).sum(dim=1) / denom))
            if "all" in poolings:
                tokens_all[l].append(to_np(h.reshape(B * T, h.shape[-1])[sel]))

        if bracket_on:
            nonspecial = ~((offsets[:, :, 0] == 0) & (offsets[:, :, 1] == 0))
            for field in bracket_fields:
                fmask = torch.zeros(B, T, dtype=torch.bool, device=device)
                for b in range(B):
                    s, e = spans[b].get(field, (0, 0))
                    if e > s:
                        o = offsets[b]
                        fmask[b] = (o[:, 1] > s) & (o[:, 0] < e)
                fmask &= attn.bool() & nonspecial
                cnt = fmask.sum(dim=1)
                hs_dtype = hs[layers[0]].dtype
                if "mean" in bracket_modes:
                    fm = fmask.unsqueeze(-1).to(hs_dtype)
                    fden = fm.sum(dim=1).clamp(min=1)
                if "last" in bracket_modes:
                    last_in_span = T - 1 - fmask.flip(dims=[1]).int().argmax(dim=1)
                if "bertscore" in bracket_modes:
                    # Per-token IDF weights for this field, zeroed outside the span.
                    idf_w = idf_lookup[field][input_ids] * fmask.to(idf_lookup[field].dtype)
                    idf_den = idf_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    # If a span has zero total IDF (all tokens unseen in IDF table), fall
                    # back to uniform within the span to avoid NaN. Marked via cnt==0 later.
                    fallback = (idf_w.sum(dim=1) == 0) & (cnt > 0)
                    if fallback.any():
                        uniform = fmask.to(idf_lookup[field].dtype)
                        idf_w = torch.where(fallback.unsqueeze(1), uniform, idf_w)
                        idf_den = idf_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    idf_w_norm = (idf_w / idf_den).unsqueeze(-1).to(hs_dtype)
                for l in layers:
                    if "mean" in bracket_modes:
                        v = (hs[l] * fm).sum(dim=1) / fden
                        v[cnt == 0] = float("nan")
                        bracket_acc["mean"][field][l].append(to_np(v))
                    if "last" in bracket_modes:
                        vL = hs[l][rows, last_in_span].clone()
                        vL[cnt == 0] = float("nan")
                        bracket_acc["last"][field][l].append(to_np(vL))
                    if "bertscore" in bracket_modes:
                        vB = (hs[l] * idf_w_norm).sum(dim=1)
                        vB[cnt == 0] = float("nan")
                        bracket_acc["bertscore"][field][l].append(to_np(vB))
                bracket_cnt[field].extend(cnt.tolist())

        if not args.no_answer:
            logits_last = out.logits[:, -1, :]
            ans_id = logits_last.argmax(dim=-1)
            ans_prob = torch.softmax(logits_last.float(), dim=-1)[rows, ans_id]
            ans_text = tok.batch_decode(ans_id.unsqueeze(-1))
            ans_id, ans_prob = ans_id.tolist(), ans_prob.tolist()
        n_tokens = attn.sum(dim=1).tolist()

        for k, item in enumerate(chunk):
            row = {"row": start + k, "prompt_id": item["id"],
                   "prompt_sha1": hashlib.sha1(item["text"].encode("utf-8")).hexdigest()[:12],
                   "n_tokens": int(n_tokens[k])}
            if "sentence_col" in item:
                row["sentence_col"] = item["sentence_col"]
            if not args.no_answer:
                row.update(answer_token_id=int(ans_id[k]), answer_text=ans_text[k], answer_prob=float(ans_prob[k]))
            meta_rows.append(row)
        print(f"  {min(start + args.batch_size, len(items))}/{len(items)} prompts ({time.time() - t0:.1f}s)", flush=True)

    pooled_final = {name: {l: np.concatenate(c, 0) for l, c in d.items()} for name, d in pooled.items()}
    for field in bracket_fields:
        if "mean" in bracket_modes:
            pooled_final[f"bracket_{field}"] = {l: np.concatenate(c, 0) for l, c in bracket_acc["mean"][field].items()}
        if "last" in bracket_modes:
            pooled_final[f"bracket_{field}_last"] = {l: np.concatenate(c, 0) for l, c in bracket_acc["last"][field].items()}
        if "bertscore" in bracket_modes:
            pooled_final[f"bracket_{field}_bertscore"] = {l: np.concatenate(c, 0) for l, c in bracket_acc["bertscore"][field].items()}
    tokens_all_final = ({l: np.concatenate(c, 0) for l, c in tokens_all.items()} if "all" in poolings else None)
    tokens_index_df = pd.DataFrame(tok_index_rows) if "all" in poolings else None

    metadata_df = pd.DataFrame(meta_rows)
    for field in bracket_fields:
        metadata_df[f"n_{field}"] = bracket_cnt[field]

    set_name = args.set_name or (Path(args.template).stem if args.template
                                 else Path(args.prompts).stem if args.prompts
                                 else time.strftime("run_%Y%m%d_%H%M%S"))
    run_dir = Path(args.output_dir) / alias / set_name
    manifest = {
        "model_alias": alias, "hf_id": hf_id, "dtype": args.dtype, "device": str(device),
        "chat_template": use_chat, "max_length": args.max_length,
        "layers": layers, "n_hidden_states": n_states, "pooling": poolings,
        "brackets": bracket_fields, "bracket_pooling": bracket_modes,
        "template": args.template, "stimuli": args.stimuli if args.template else None,
        "sentences": args.sentences, "sentence_slot": args.sentence_slot if carrier_mode else None,
        "n_prompts": len(items), "record_answer": not args.no_answer,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    embeddings_io.save_run(run_dir, pooled=pooled_final, metadata=metadata_df, manifest=manifest,
                           tokens_all=tokens_all_final, tokens_index=tokens_index_df)

    print(f"\nWrote {run_dir}")
    for name, d in pooled_final.items():
        any_l = next(iter(d.values()))
        print(f"  pooled_{name}.safetensors: {len(d)} layers x {any_l.shape} (fp16)")
    if tokens_all_final is not None:
        any_l = next(iter(tokens_all_final.values()))
        print(f"  tokens_all.safetensors:   {len(tokens_all_final)} layers x {any_l.shape}")


if __name__ == "__main__":
    main()
