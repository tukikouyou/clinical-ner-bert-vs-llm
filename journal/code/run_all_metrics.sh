#!/bin/bash
# 全 run の stats を再計算 (metrics_v2 の変更後に実行)。pairwise も再生成。
cd /home/user/yagahara/LAU16_paper/journal
runs=$(ls runs | while read r; do [ -f runs/$r/preds.json ] && echo $r; done)
python3 code/metrics_v2.py --runs $runs --bert UTH NICT > logs/metrics_all.log 2>&1
for r in llm-jp-4-32b-zs sip-jmed-13b-zs llm-jp-1.8b-zs medgemma-27b-zs sip-jmed-13b-ft llm-jp-1.8b-ft medgemma-27b-ft; do python3 code/metrics_v2.py --runs $r --parser v2 >> logs/metrics_all.log 2>&1; done
P="python3 code/metrics_v2.py --pair --runs"
$P med42-70b-ft llama-3.3-70b-ft >> logs/metrics_all.log 2>&1
$P med42-70b-s1-ft llama-3.3-70b-s1-ft >> logs/metrics_all.log 2>&1
$P med42-70b-s2-ft llama-3.3-70b-s2-ft >> logs/metrics_all.log 2>&1
$P medgemma-27b-ft gemma-3-27b-s42-ft >> logs/metrics_all.log 2>&1
$P medgemma-27b-s1-ft gemma-3-27b-s1-ft >> logs/metrics_all.log 2>&1
$P medgemma-27b-s2-ft gemma-3-27b-s2-ft >> logs/metrics_all.log 2>&1
$P sip-jmed-13b-ft llmjp-3.1-13b-ft >> logs/metrics_all.log 2>&1
$P sip-jmed-13b-s1-ft llmjp-3.1-13b-s1-ft >> logs/metrics_all.log 2>&1
$P sip-jmed-13b-s2-ft llmjp-3.1-13b-s2-ft >> logs/metrics_all.log 2>&1
python3 code/metrics_v2.py --pair --runs med42-70b-ft llama-3.3-70b-ft medgemma-27b-ft sip-jmed-13b-ft qwen-ft llm-jp-4-32b-ft llm-jp-1.8b-ft --bert UTH NICT >> logs/metrics_all.log 2>&1
rm -f stats/pairwise_gemma-3-27b-s42-ft_medgemma-27b-ft.json stats/pairwise_BERT-UTH_llmjp-3.1-13b-ft_sip-jmed-13b-ft.json stats/pairwise_BERT-UTH_BERT-NICT.json
python3 code/make_tables.py > /dev/null
echo "ALL METRICS DONE $(date '+%H:%M')" >> logs/metrics_all.log
