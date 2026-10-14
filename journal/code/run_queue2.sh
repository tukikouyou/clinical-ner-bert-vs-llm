#!/bin/bash
# 期刊版 推論キャンペーン。 usage: run_queue.sh <A|B|C>
#   A: tp=1 (GPU0)        B: tp=2 (GPU2,3)        C: tp=4 (GPU0-3, 学習終了後)
# 各ジョブは runs/<name>/meta.json があればスキップ、モデル/LoRA パスが無ければ(HDD未接続) スキップして後で再実行。
cd /home/user/yagahara/LAU16_paper/journal
# 実行中にこのファイルを編集すると bash がずれた offset を読んで暴走するため、自身のスナップショットを exec する
if [ -z "$RQ_SNAPSHOT" ]; then snap=$(mktemp /tmp/rq2.XXXXXX.sh); cp "$0" "$snap"; RQ_SNAPSHOT=1 exec bash "$snap" "$@"; fi
Q=$1; LOG=logs/queue2_$Q.log
VLLM=/opt/llm/vllm-venv/bin/python
T451=/opt/llm/ft-venv-t451/bin/python
EXT=/media/user/Elements/hf-cache; OPT=$EXT   # 全モデルを Elements に集約(2026-10-05)
FT=/media/user/Elements/ft; FTL=/media/user/Elements/ft; MG=/opt/llm/stage   # merge 出力(60GB)は SSD のステージ領域に置き、推論後に削除
LLAMA=/media/user/Elements/models/Llama-3.3-70B-Instruct
SIP=SIP-med-LLM/SIP-jmed-llm-3-13b-OP-32k-R0.1; QWEN=Qwen/Qwen3-30B-A3B-Instruct-2507
LJ4=llm-jp/llm-jp-4-32b-a3b-thinking; LJ18=llm-jp/llm-jp-3.1-1.8b-instruct4

hf_has() { # <hfhome> <repo id>  → 0 if snapshot exists
  local d="$1/hub/models--${2//\//--}/snapshots"; [ -d "$d" ] && [ -n "$(ls -A $d 2>/dev/null)" ]; }
avail() { # model-or-path hfhome ; lora path
  local m=$1 hf=$2 lora=$3
  if [[ "$m" == /* ]]; then [ -e "$m" ] || return 1; else hf_has "$hf" "$m" || return 1; fi
  if [ -n "$lora" ]; then   # adapter は学習完了(adapter_model.safetensors; ft-journal は train_meta.json も)を要求
    [ -f "$lora/adapter_model.safetensors" ] || return 1
    case "$lora" in */ft-journal/*) [ -f "$lora/train_meta.json" ] || return 1;; esac
  fi; return 0; }
run() { # name model hfhome tp gpus maxtok maxlen lora extra...
  local name=$1 model=$2 hf=$3 tp=$4 gpus=$5 maxtok=$6 maxlen=$7 lora=$8; shift 8
  [ -f runs/$name/meta.json ] && { echo "[$name] SKIP done" >> $LOG; return; }
  avail "$model" "$hf" "$lora" || { echo "[$name] SKIP (model/LoRA not available: $model $lora)" >> $LOG; return; }
  echo "===== [$name] start $(date '+%m-%d %H:%M') tp=$tp gpus=$gpus =====" >> $LOG
  local staged; staged=$(bash code/stage.sh "$model" "$hf" 2>>logs/stage.log) || { echo "!!!!! [$name] STAGE FAILED" >> $LOG; return; }
  model=$staged
  local extra=""; [ -n "$lora" ] && extra="--lora $lora"
  local v1=1; [ "$tp" -ge 2 ] && v1=0
  # Triton/inductor キャッシュはモデル単位で共有(初回で温め、2回目以降は再コンパイル無し → tp>=2 の FileExistsError 競合を避ける)
  local fam=${name%-ft}; fam=${fam%-zs-guided}; fam=${fam%-zs}
  # 温め済みキャッシュを系列ごとに共有(新規ディレクトリは tp>=2 で初回コンパイル競合 FileExistsError を起こす)
  local tc=/opt/llm/.tj_$fam
  case "$name" in
    med42-70b*) tc=/opt/llm/.tmed42;; llama-3.3-70b*) tc=/opt/llm/.tllamaft;;
    medgemma-27b*|gemma-3-27b*) tc=/opt/llm/.tj_medgemma-27b-ft;;
    sip-jmed-13b*) tc=/opt/llm/.tj_sip-jmed-13b-ft;; llmjp-3.1-13b*) tc=/opt/llm/.tj_llmjp-3.1-13b;;
    llm-jp-1.8b*) tc=/opt/llm/.tj_llm-jp-1.8b-ft;; qwen*) tc=/opt/llm/.tj_qwen;; llm-jp-4-32b*) tc=/opt/llm/.tj_llm-jp-4-32b;;
  esac
  local try rc
  for try in 1 2; do
    HF_HOME=$hf CUDA_VISIBLE_DEVICES=$gpus VLLM_USE_V1=$v1 TRITON_CACHE_DIR=$tc PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      timeout -k 60 4h $VLLM -u code/eval_v2.py --name $name --model "$model" --tp $tp --max_new_tokens $maxtok --max_model_len $maxlen \
      --enforce_eager $extra "$@" > logs/run_${name}_try$try.log 2>&1; rc=$?
    cp logs/run_${name}_try$try.log logs/run_$name.log
    [ $rc -eq 0 ] && [ -f runs/$name/meta.json ] && break
    echo "!!!!! [$name] try$try FAILED rc=$rc (124=timeout) !!!!!" >> $LOG
    [ -z "$(ls -A $tc 2>/dev/null)" ] || true
  done
  echo "----- [$name] $(grep -E '^\[.*\] done' logs/run_$name.log | tail -1)" >> $LOG; }
merge() { # name base lora hfhome → $MG/name
  local name=$1 base=$2 lora=$3 hf=$4
  [ -d $MG/$name ] && [ -f $MG/$name/config.json ] && return 0
  avail "$base" "$hf" "$lora" || return 1
  echo "===== [$name] merge $(date '+%m-%d %H:%M') =====" >> $LOG; rm -rf $MG/$name
  base=$(bash code/stage.sh "$base" "$hf" 2>>logs/stage.log) || return 1
  bash code/stage.sh --reserve 70 2>>logs/stage.log   # merge 出力(最大 65GB)の空きを確保
  HF_HOME=$hf CUDA_VISIBLE_DEVICES=2,3 $T451 -u /home/user/yagahara/shared/llm/merge_lora.py --base "$base" --lora "$lora" --out $MG/$name >> logs/merge_$name.log 2>&1 \
   || { rm -rf $MG/$name; HF_HOME=$hf CUDA_VISIBLE_DEVICES= $T451 -u /home/user/yagahara/shared/llm/merge_lora.py --base "$base" --lora "$lora" --out $MG/$name --device cpu >> logs/merge_$name.log 2>&1; } \
   || { echo "!!!!! [$name] MERGE FAILED" >> $LOG; return 1; }; }

echo "########## queue $Q start $(date '+%m-%d %H:%M') ##########" >> $LOG
case $Q in
A) # GPU0, tp=1
  run llm-jp-1.8b-zs-guided $LJ18 $OPT 1 0 900 4096 "" --guided
  run llm-jp-1.8b-ft        $LJ18 $OPT 1 0 900 4096 $FT/llmjp-1.8b-3ep
  run sip-jmed-13b-ft       $SIP  $EXT 1 0 900 4096 $FT/sip-jmed-13b
  run sip-jmed-13b-zs-guided $SIP $EXT 1 0 900 4096 "" --guided
  run sip-jmed-13b-zs       $SIP  $EXT 1 0 3072 8192 ""
  ;;
B) # GPU2,3, tp=2
  run medgemma-27b-ft        /media/user/Elements/models/medgemma-27b-text $OPT 2 2,3 900 4096 $FTL/medgemma-27b --max_num_seqs 48
  run medgemma-27b-zs-guided /media/user/Elements/models/medgemma-27b-text $OPT 2 2,3 900 4096 "" --guided --max_num_seqs 48
  run medgemma-27b-zs        /media/user/Elements/models/medgemma-27b-text $OPT 2 2,3 900 4096 "" --max_num_seqs 48
  run llm-jp-4-32b-zs-guided $LJ4 $EXT 2 2,3 900 4096 "" --guided
  run llm-jp-4-32b-zs        $LJ4 $EXT 2 2,3 3072 8192 ""
  run qwen-zs-guided         $QWEN $EXT 2 2,3 900 4096 "" --guided
  run qwen-zs                $QWEN $EXT 2 2,3 3072 8192 ""
  if merge llm-jp-4-32b-ft $LJ4 $FT/llmjp-4-32b $EXT; then run llm-jp-4-32b-ft $MG/llm-jp-4-32b-ft $EXT 2 2,3 900 4096 ""; rm -rf $MG/llm-jp-4-32b-ft; fi
  if merge qwen-ft $QWEN $FT/qwen3-30b-a3b $EXT; then run qwen-ft $MG/qwen-ft $EXT 2 2,3 900 4096 ""; rm -rf $MG/qwen-ft; fi
  ;;
A2) # GPU0 tp=1: 制約デコード再実行(disable-any-whitespace)
  run llm-jp-1.8b-zs-guided $LJ18 $EXT 1 0 900 4096 "" --guided
  run sip-jmed-13b-zs-guided $SIP $EXT 1 0 900 4096 "" --guided
  ;;
B2) # GPU2,3 tp=2: 制約デコード再実行
  run medgemma-27b-zs-guided /media/user/Elements/models/medgemma-27b-text $EXT 2 2,3 900 4096 "" --guided --max_num_seqs 48
  run llm-jp-4-32b-zs-guided $LJ4 $EXT 2 2,3 900 4096 "" --guided
  # 2 段階「自由思考 + 制約回答」: prompt-only ZS の思考をプロンプトに戻し final だけ制約 (max_model_len は思考分を含めるため 8192)
  run llm-jp-4-32b-zs-think-guided $LJ4 $EXT 2 2,3 900 8192 "" --guided --think_from llm-jp-4-32b-zs
  run sip-jmed-13b-zs-think-guided $SIP $EXT 1 2   900 8192 "" --guided --think_from sip-jmed-13b-zs
  ;;
F1) # GPU0 tp=1: 新 adapter (多 seed / 基盤) の FT 推論。adapter 未完成なら SKIP → 後で再実行
  FJ=/media/user/Elements/ft-journal
  run llm-jp-1.8b-s1-ft     $LJ18 $EXT 1 0 900 4096 $FJ/llmjp-1.8b-s1
  run llm-jp-1.8b-s2-ft     $LJ18 $EXT 1 0 900 4096 $FJ/llmjp-1.8b-s2
  run llmjp-3.1-13b-s1-ft   llm-jp/llm-jp-3.1-13b-instruct4 $EXT 1 0 900 4096 $FJ/llmjp-3.1-13b-s1
  run llmjp-3.1-13b-s2-ft   llm-jp/llm-jp-3.1-13b-instruct4 $EXT 1 0 900 4096 $FJ/llmjp-3.1-13b-s2
  run sip-jmed-13b-s1-ft    $SIP $EXT 1 0 900 4096 $FJ/sip-jmed-13b-s1
  run sip-jmed-13b-s2-ft    $SIP $EXT 1 0 900 4096 $FJ/sip-jmed-13b-s2
  ;;
F2) # GPU2,3 tp=2: 27B 系の新 adapter と MoE seed (merge)
  FJ=/media/user/Elements/ft-journal; GT=/media/user/Elements/models/gemma-3-27b-text-it; MGT=/media/user/Elements/models/medgemma-27b-text
  run gemma-3-27b-s42-ft    $GT  $EXT 2 2,3 900 4096 $FJ/gemma-3-27b-s42 --max_num_seqs 48
  run gemma-3-27b-s1-ft     $GT  $EXT 2 2,3 900 4096 $FJ/gemma-3-27b-s1  --max_num_seqs 48
  run gemma-3-27b-s2-ft     $GT  $EXT 2 2,3 900 4096 $FJ/gemma-3-27b-s2  --max_num_seqs 48
  run medgemma-27b-s1-ft    $MGT $EXT 2 2,3 900 4096 $FJ/medgemma-27b-s1 --max_num_seqs 48
  run medgemma-27b-s2-ft    $MGT $EXT 2 2,3 900 4096 $FJ/medgemma-27b-s2 --max_num_seqs 48
  for sd in 1 2; do
    if merge qwen-s$sd-ft $QWEN $FJ/qwen3-30b-a3b-s$sd $EXT; then run qwen-s$sd-ft $MG/qwen-s$sd-ft $EXT 2 2,3 900 4096 ""; rm -rf $MG/qwen-s$sd-ft; fi
    if merge llm-jp-4-32b-s$sd-ft $LJ4 $FJ/llmjp-4-32b-s$sd $EXT; then run llm-jp-4-32b-s$sd-ft $MG/llm-jp-4-32b-s$sd-ft $EXT 2 2,3 900 4096 ""; rm -rf $MG/llm-jp-4-32b-s$sd-ft; fi
  done
  ;;
F4) # GPU0-3 tp=4: 70B seed
  FJ=/media/user/Elements/ft-journal
  for sd in 1 2; do run llama-3.3-70b-s$sd-ft $LLAMA $EXT 4 0,1,2,3 900 4096 $FJ/llama-3.3-70b-s$sd; done   # 同一モデルを連続 (ステージ再コピー 132GB を節約; Llama が先にステージ済)
  for sd in 1 2; do run med42-70b-s$sd-ft     /media/user/Elements/models/med42-70b $EXT 4 0,1,2,3 900 4096 $FJ/med42-70b-s$sd; done
  ;;
C) # GPU0-3, tp=4
  run med42-70b-ft          /media/user/Elements/models/med42-70b $OPT 4 0,1,2,3 900 4096 $FTL/med42-70b
  run med42-70b-zs-guided   /media/user/Elements/models/med42-70b $OPT 4 0,1,2,3 900 4096 "" --guided
  run med42-70b-zs          /media/user/Elements/models/med42-70b $OPT 4 0,1,2,3 900 4096 ""
  run llama-3.3-70b-ft      $LLAMA $EXT 4 0,1,2,3 900 4096 $FT/llama-3.3-70b
  run llama-3.3-70b-zs-guided $LLAMA $EXT 4 0,1,2,3 900 4096 "" --guided
  run llama-3.3-70b-zs      $LLAMA $EXT 4 0,1,2,3 900 4096 ""
  ;;
K4) # 生成予算アブレーション: seed42 の FT 推論を max_new_tokens 4096 / max_model_len 8192 で再実行 (runs/<name>-4k)。tp1→GPU0, tp2→GPU2,3, 70B tp4 は最後
  FJ=/media/user/Elements/ft-journal; GT=/media/user/Elements/models/gemma-3-27b-text-it; MGT=/media/user/Elements/models/medgemma-27b-text
  run llm-jp-1.8b-ft-4k     $LJ18 $OPT 1 0 2600 4096 $FT/llmjp-1.8b-3ep   # LLM-jp-3.1 は max_position_embeddings 4096 → 予算 2600
  run sip-jmed-13b-ft-4k    $SIP  $EXT 1 0 4096 8192 $FT/sip-jmed-13b
  run llmjp-3.1-13b-ft-4k   llm-jp/llm-jp-3.1-13b-instruct4 $EXT 1 0 2600 4096 $FJ/llmjp-3.1-13b-s42
  run medgemma-27b-ft-4k    $MGT $OPT 2 2,3 4096 8192 $FTL/medgemma-27b --max_num_seqs 32
  run gemma-3-27b-ft-4k     $GT  $EXT 2 2,3 4096 8192 $FJ/gemma-3-27b-s42 --max_num_seqs 32
  if merge qwen-ft $QWEN $FT/qwen3-30b-a3b $EXT; then run qwen-ft-4k $MG/qwen-ft $EXT 2 2,3 4096 8192 ""; rm -rf $MG/qwen-ft; fi
  if merge llm-jp-4-32b-ft $LJ4 $FT/llmjp-4-32b $EXT; then run llm-jp-4-32b-ft-4k $MG/llm-jp-4-32b-ft $EXT 2 2,3 4096 8192 ""; rm -rf $MG/llm-jp-4-32b-ft; fi
  # 70B は 4 GPU 全部が要る: GPU1 の他ジョブ (<2GB なら可) を待つ
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -gt 2000 ]; do echo "[K4] waiting for GPU1 (busy)" >> $LOG; sleep 600; done
  run med42-70b-ft-4k       /media/user/Elements/models/med42-70b $OPT 4 0,1,2,3 4096 8192 $FTL/med42-70b
  run llama-3.3-70b-ft-4k   $LLAMA $EXT 4 0,1,2,3 4096 8192 $FT/llama-3.3-70b
  ;;
R|RQ|RG) # 参考モデル (ollama)。RQ=qwen3.6 のみ(GPU 1 枚で可)、RG=gpt-oss:120b のみ(GPU 2 枚以上)、R=両方
  specs=()
  [ "$Q" != "RG" ] && specs+=("qwen3.6-35b-zs qwen3.6:35b" "qwen3.6-35b-zs-guided qwen3.6:35b --guided" "qwen3.6-35b-zs-think-guided qwen3.6:35b --guided --ollama_think")
  [ "$Q" != "RQ" ] && specs+=("gpt-oss-120b-zs gpt-oss:120b" "gpt-oss-120b-zs-guided gpt-oss:120b --guided" "gpt-oss-120b-zs-think-guided gpt-oss:120b --guided --ollama_think")
  for spec in "${specs[@]}"; do
    set -- $spec; name=$1; model=$2; shift 2
    [ -f runs/$name/meta.json ] && { echo "[$name] SKIP done" >> $LOG; continue; }
    echo "===== [$name] start $(date '+%m-%d %H:%M') ollama =====" >> $LOG
    python3 -u code/eval_v2.py --engine ollama --name $name --model $model --max_new_tokens 3072 --max_model_len 8192 "$@" \
      > logs/run_$name.log 2>&1 || echo "!!!!! [$name] FAILED rc=$? !!!!!" >> $LOG
    echo "----- [$name] $(grep -E '^\[.*\] done' logs/run_$name.log | tail -1)" >> $LOG
  done
  ;;
esac
echo "########## queue $Q end $(date '+%m-%d %H:%M') ##########" >> $LOG
