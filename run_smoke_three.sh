#!/usr/bin/env bash
set -u
PY=".venv/bin/python"
declare -A SET=( [qwen-2.5-3b]=smoke_qwen3b [gemma-2-2b]=smoke_gemma2b [llama-3.2-3b]=smoke_llama3b )
for ALIAS in qwen-2.5-3b gemma-2-2b llama-3.2-3b; do
  S=${SET[$ALIAS]}
  echo "===================== $ALIAS -> $S ====================="
  $PY extract_embeddings.py --model "$ALIAS" \
      --template data/prompts/prompt_var_N0.txt \
      --stimuli data/estimulos/estimulos_completo.xlsx \
      --sentences s4 p --sentence-slot sentence_1 \
      --brackets question answer sentence_1 --pooling last \
      --layers all --batch-size 4 --limit 50 --set-name "$S"     && $PY analyze_n0_targets.py --run "outputs/$ALIAS/$S" --output "plots/n0_targets_$S.png"     || echo "!!!!! $ALIAS FAILED (see above) !!!!!"
  echo
done
echo "ALL DONE"
