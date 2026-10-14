#!/bin/bash
# 期刊版 QLoRA 学習キャンペーン。 usage: run_train_queue.sh <GPU> <job>...
#   job: gemma-s42|gemma-s1|gemma-s2 | llmjp13-s1|llmjp13-s2 | sip-s1|sip-s2 | medgemma-s1|medgemma-s2 | llmjp18-s1|llmjp18-s2
#        qwen-s1|qwen-s2 | llmjp4-s1|llmjp4-s2 (MoE, t451 venv) | med42-s1|med42-s2 | llama-s1|llama-s2 (70B: GPU を "2,3" と 2 枚指定)
# 出力 /opt/llm/ft-local/<name>/ (train_meta.json があればスキップ)。設定は LAU 論文と同一 (3ep, lr2e-4, r16, seed のみ変更)。
cd /home/user/yagahara/LAU16_paper/journal
# 実行中の編集で offset がずれないよう自身のスナップショットを exec する
if [ -z "$RQ_SNAPSHOT" ]; then snap=$(mktemp /tmp/rtq2.XXXXXX.sh); cp "$0" "$snap"; RQ_SNAPSHOT=1 exec bash "$snap" "$@"; fi
GPU=$1; shift; LOG=logs/train_queue2_gpu$GPU.log
PY=/opt/llm/ft-venv/bin/python; T451=/opt/llm/ft-venv-t451/bin/python; SFT=/home/user/yagahara/shared/llm/sft_data
OUT=/media/user/Elements/ft-journal   # 新規 adapter は根分区(残 48GB)に置かず Elements へ
EXT=/media/user/Elements/hf-cache; OPT=$EXT
train() { # name model hfhome seed [py] [extra args...]
  local name=$1 model=$2 hf=$3 seed=$4; local py=${5:-$PY}; shift 4; [ $# -gt 0 ] && shift
  local dm=single; case "$GPU" in *,*) dm=auto;; esac   # 2 GPU 指定時はモデル並列 (70B)
  [ -f $OUT/$name/train_meta.json ] && { echo "[$name] SKIP done" >> $LOG; return; }
  mkdir "$OUT/$name.lock" 2>/dev/null || { echo "[$name] SKIP in progress (lock)" >> $LOG; return; }   # 複数ワーカーが同じジョブを取らないよう claim
  echo "===== [$name] start $(date '+%m-%d %H:%M') gpu=$GPU seed=$seed =====" >> $LOG
  local staged; if staged=$(bash code/stage.sh "$model" "$hf" 2>>logs/stage.log); then model=$staged
  else echo "[$name] stage failed (空き不足など) → Elements から直接読込" >> $LOG; fi   # 学習は読込 1 回だけなので HDD 直読でも可 (~25 min)
  HF_HOME=$hf CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    timeout -k 60 ${JOB_TIMEOUT:-0} $py -u code/qlora_train_v2.py --model "$model" --data_dir $SFT --out_dir $OUT/$name --epochs 3 --lr 2e-4 --seed $seed --device_map $dm "$@" \
    > logs/train_$name.log 2>&1 || echo "!!!!! [$name] FAILED rc=$? !!!!!" >> $LOG
  echo "----- [$name] end $(date '+%m-%d %H:%M') $(grep -c 'LoRA adapter saved' logs/train_$name.log) =====" >> $LOG; rmdir "$OUT/$name.lock" 2>/dev/null; }
for j in "$@"; do case $j in
  gemma-s42)   train gemma-3-27b-s42  /media/user/Elements/models/gemma-3-27b-text-it $OPT 42;;
  gemma-s1)    train gemma-3-27b-s1   /media/user/Elements/models/gemma-3-27b-text-it $OPT 1;;
  gemma-s2)    train gemma-3-27b-s2   /media/user/Elements/models/gemma-3-27b-text-it $OPT 2;;
  llmjp13-s1)  train llmjp-3.1-13b-s1 llm-jp/llm-jp-3.1-13b-instruct4 $OPT 1;;
  llmjp13-s2)  train llmjp-3.1-13b-s2 llm-jp/llm-jp-3.1-13b-instruct4 $OPT 2;;
  medgemma-s1) train medgemma-27b-s1  /media/user/Elements/models/medgemma-27b-text $OPT 1;;
  medgemma-s2) train medgemma-27b-s2  /media/user/Elements/models/medgemma-27b-text $OPT 2;;
  sip-s1)      train sip-jmed-13b-s1  SIP-med-LLM/SIP-jmed-llm-3-13b-OP-32k-R0.1 $EXT 1;;
  sip-s2)      train sip-jmed-13b-s2  SIP-med-LLM/SIP-jmed-llm-3-13b-OP-32k-R0.1 $EXT 2;;
  llmjp18-s1)  train llmjp-1.8b-s1    llm-jp/llm-jp-3.1-1.8b-instruct4 $OPT 1;;
  llmjp18-s2)  train llmjp-1.8b-s2    llm-jp/llm-jp-3.1-1.8b-instruct4 $OPT 2;;
  qwen-s1)     train qwen3-30b-a3b-s1 Qwen/Qwen3-30B-A3B-Instruct-2507 $EXT 1 $T451;;
  qwen-s2)     train qwen3-30b-a3b-s2 Qwen/Qwen3-30B-A3B-Instruct-2507 $EXT 2 $T451;;
  llmjp4-s1)   train llmjp-4-32b-s1   llm-jp/llm-jp-4-32b-a3b-thinking $EXT 1 $T451;;
  llmjp4-s2)   train llmjp-4-32b-s2   llm-jp/llm-jp-4-32b-a3b-thinking $EXT 2 $T451;;
  med42-s1)    train med42-70b-s1     /media/user/Elements/models/med42-70b $OPT 1 $PY --save_steps 50;;      # GPU は "2,3" のように 2 枚指定
  med42-s2)    train med42-70b-s2     /media/user/Elements/models/med42-70b $OPT 2 $PY --save_steps 50;;
  llama-s1)    train llama-3.3-70b-s1 /media/user/Elements/models/Llama-3.3-70B-Instruct $EXT 1 $PY --save_steps 50;;
  llama-s2)    train llama-3.3-70b-s2 /media/user/Elements/models/Llama-3.3-70B-Instruct $EXT 2 $PY --save_steps 50;;
  *) echo "unknown job $j" >> $LOG;;
esac; done
echo "########## train queue gpu$GPU end $(date '+%m-%d %H:%M') ##########" >> $LOG
