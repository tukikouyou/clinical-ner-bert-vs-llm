#!/bin/bash
# ZS+C 同予算再実行: queue ZG → reparse → 採点
cd /home/user/yagahara/LAU16_paper/journal
LOG=logs/chain_zg.log; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
while ps -eo cmd | grep -v grep | grep -q 'eval_nursing_v2.py'; do log "waiting: eval_nursing_v2 running"; sleep 300; done
log "ZG start"; bash code/run_queue2.sh ZG; log "ZG queue ended"
for r in qwen-zs-guided llm-jp-4-32b-zs-guided sip-jmed-13b-zs-guided; do
  [ -f runs/$r/meta.json ] || { log "[$r] missing"; continue; }
  python3 code/reparse.py $r >> $LOG 2>&1; python3 code/metrics_v2.py --runs $r 2>&1 | grep 'relaxed F1' >> $LOG
done
log "########## chain_zg END ##########"
