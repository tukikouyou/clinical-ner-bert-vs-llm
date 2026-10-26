#!/bin/bash
# 生成予算曲線: BC1 (GPU0) ∥ BC1B (GPU1) ∥ BC2 (GPU2,3) → BC70 (4 GPU) → reparse → 採点 → 提及レベル採点
cd /home/user/yagahara/LAU16_paper/journal
LOG=logs/chain_bc.log; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
while ps -eo cmd | grep -v grep | grep -q 'eval_nursing_v2.py'; do log "waiting: eval_nursing_v2 running"; sleep 300; done
log "BC workers start"
bash code/run_queue2.sh BC1 & P1=$!; bash code/run_queue2.sh BC1B & P2=$!; bash code/run_queue2.sh BC2 & P3=$!
wait $P1; log "BC1 done"; wait $P2; log "BC1B done"; wait $P3; log "BC2 done"
bash code/run_queue2.sh BC70; log "BC70 done"
for r in runs/*-ft-512 runs/*-ft-1k runs/*-ft-2k; do n=$(basename $r)
  [ -f $r/meta.json ] || { log "[$n] missing"; continue; }
  python3 code/reparse.py $n >> $LOG 2>&1; python3 code/metrics_v2.py --runs $n 2>&1 | grep 'relaxed F1' >> $LOG
done
python3 code/metrics_mention.py >> $LOG 2>&1
log "########## chain_bc END ##########"
