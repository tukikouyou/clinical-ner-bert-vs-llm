#!/bin/bash
# 大予算 seed1/2 → 採点
cd /home/user/yagahara/LAU16_paper/journal
LOG=logs/chain_k4s.log; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
while ps -eo cmd | grep -v grep | grep -q 'eval_nursing_v2.py'; do log "waiting: eval_nursing_v2 running"; sleep 300; done
log "K4S start"; bash code/run_queue2.sh K4S; log "K4S queue ended"
for m in llm-jp-1.8b llmjp-3.1-13b sip-jmed-13b medgemma-27b gemma-3-27b qwen llm-jp-4-32b llama-3.3-70b med42-70b; do for sd in 1 2; do r=$m-s$sd-ft-4k
  [ -f runs/$r/meta.json ] || { log "[$r] missing"; continue; }
  python3 code/reparse.py $r >> $LOG 2>&1; python3 code/metrics_v2.py --runs $r 2>&1 | grep 'relaxed F1' >> $LOG
done; done
log "########## chain_k4s END ##########"
