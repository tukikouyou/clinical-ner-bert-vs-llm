#!/bin/bash
# 生成予算アブレーション: queue K4 → 採点 → 表
cd /home/user/yagahara/LAU16_paper/journal
LOG=logs/chain_k4.log; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
# 看護テキスト評価 (eval_nursing_v2.py) が GPU を使っている間は待つ (ユーザーの別ジョブを優先)
while ps -eo cmd | grep -v grep | grep -q 'eval_nursing_v2.py'; do log "waiting: eval_nursing_v2 running"; sleep 300; done
log "K4 start"; bash code/run_queue2.sh K4; log "K4 queue ended"
for r in llm-jp-1.8b-ft-4k sip-jmed-13b-ft-4k llmjp-3.1-13b-ft-4k medgemma-27b-ft-4k gemma-3-27b-ft-4k qwen-ft-4k llm-jp-4-32b-ft-4k med42-70b-ft-4k llama-3.3-70b-ft-4k; do
  [ -f runs/$r/meta.json ] || { log "[$r] missing"; continue; }
  python3 code/reparse.py $r >> $LOG 2>&1; python3 code/metrics_v2.py --runs $r 2>&1 | grep 'relaxed F1' >> $LOG
done
log "########## chain_k4 END ##########"
