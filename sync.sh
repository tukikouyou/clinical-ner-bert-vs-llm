#!/bin/bash
# 作業ディレクトリ(../../shared/code, ../../shared/llm)から、このGitリポジトリへ
# 「コードと集計指標のみ」を同期する。データ/モデル/患者情報は同期しない。
#
# 使い方:  ./sync.sh  &&  git add -A  &&  git commit -m "update"  &&  git push
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
WORK="$(cd "$REPO/../../shared" && pwd)"   # /home/user/yagahara/shared

echo "== BERT スクリプト =="
mkdir -p "$REPO/bert/lib"
cp "$WORK/code/"{NER_training.py,build_conll.py,extract_structured.py,predict_reports.py,\
evaluate_ner.py,summarize_results.py,preprocess_text.py,tokenization_mod.py} "$REPO/bert/" 2>/dev/null || true
cp -r "$WORK/code/lib/"{loop,models,util} "$REPO/bert/lib/" 2>/dev/null || true
find "$REPO/bert/lib" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo "== LLM スクリプト =="
cp "$WORK/llm/"{llm_extract.py,llm_extract_vllm.py,llm_extract_ollama.py,\
testset_eval.py,qlora_train.py,build_sft_data.py,merge_lora.py} "$REPO/llm/" 2>/dev/null || true

echo "== 集計指標(数値のみ・データ無し) =="
cp "$WORK/llm/results/ehr/testset_scores.json" "$REPO/results/testset_scores.json" 2>/dev/null || true
for M in UTH NICT; do
  mkdir -p "$REPO/results/bert_$M"
  cp "$WORK/code/results/NER/$M/NER/metrics_0.json"          "$REPO/results/bert_$M/metrics.json" 2>/dev/null || true
  cp "$WORK/code/results/NER/$M/NER/NER_strict_RESULT_0.json" "$REPO/results/bert_$M/strict_per_label.json" 2>/dev/null || true
  cp "$WORK/code/results/NER/$M/NER/NER_soft_RESULT_0.json"   "$REPO/results/bert_$M/soft_per_label.json" 2>/dev/null || true
done

echo "== 期刊版 (journal/): コード・集計指標・run メタのみ (preds/raw はコーパス断片を含むため同期しない) =="
J="$REPO/../journal"
mkdir -p "$REPO/journal/code" "$REPO/journal/stats" "$REPO/journal/runs" "$REPO/journal/paper/figs"
cp "$J/code/"{eval_v2.py,parse_v2.py,reparse.py,metrics_v2.py,qlora_train_v2.py,extra_stats.py,make_tables.py,annotate_tables.py,plots.py,convert_gemma_text.py,stage.sh,run_queue2.sh,run_train_queue2.sh,chain_next.sh,chain_k4.sh,run_all_metrics.sh} "$REPO/journal/code/" 2>/dev/null || true
cp "$J/stats/"*.json "$J/stats/"*.md "$REPO/journal/stats/" 2>/dev/null || true
for d in "$J"/runs/*/; do n=$(basename "$d"); [ -f "$d/meta.json" ] && { mkdir -p "$REPO/journal/runs/$n"; cp "$d/meta.json" "$REPO/journal/runs/$n/"; }; done
cp "$J/paper/figs/"*.png "$J/paper/figs/"*.pdf "$REPO/journal/paper/figs/" 2>/dev/null || true
cp "$J/README.md" "$REPO/journal/README.md" 2>/dev/null || true

echo "== 安全チェック(データ/モデルが混入していないか) =="
BAD=$(find "$REPO" -not -path '*/.git/*' \( -name '*.csv' -o -name '*.bin' -o -name '*.pt' \
  -o -name '*.safetensors' -o -name '*.dic' -o -name '*.gguf' \) | wc -l)
if [ "$BAD" -ne 0 ]; then echo "⚠️ データ/モデルらしきファイルを検出! commit前に確認:"; \
  find "$REPO" -not -path '*/.git/*' \( -name '*.csv' -o -name '*.bin' -o -name '*.pt' \
  -o -name '*.safetensors' -o -name '*.dic' -o -name '*.gguf' \); exit 1; fi
echo "OK: データ混入なし。'git add -A && git commit && git push' で更新できます。"
