#!/bin/bash
# BERT 再学習 (v2): <GPU> <B> seeds... ; 各 seed 終了後に採点
GPU=$1; B=$2; shift 2
cd /home/user/yagahara/shared/code
for sd in "$@"; do
  echo "===== $B seed $sd start $(date '+%m-%d %H:%M') =====" >> /home/user/yagahara/LAU16_paper/journal/logs/chain_bert_v2_$B.log
  CUDA_VISIBLE_DEVICES=$GPU python3 /home/user/yagahara/LAU16_paper/journal/code/bert_train_v2.py --bert_type $B --seed $sd > /home/user/yagahara/LAU16_paper/journal/logs/bert_v2_${B}_s$sd.log 2>&1 \
    || echo "!!!!! $B seed $sd FAILED" >> /home/user/yagahara/LAU16_paper/journal/logs/chain_bert_v2_$B.log
  (cd /home/user/yagahara/LAU16_paper/journal && python3 code/metrics_v2.py --bert $B-v2-s$sd 2>&1 | grep 'relaxed F1' >> logs/chain_bert_v2_$B.log)
  grep DONE /home/user/yagahara/LAU16_paper/journal/logs/bert_v2_${B}_s$sd.log >> /home/user/yagahara/LAU16_paper/journal/logs/chain_bert_v2_$B.log
done
echo "########## $B END $(date '+%m-%d %H:%M') ##########" >> /home/user/yagahara/LAU16_paper/journal/logs/chain_bert_v2_$B.log
