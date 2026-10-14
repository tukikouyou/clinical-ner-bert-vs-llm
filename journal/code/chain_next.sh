#!/bin/bash
# 2026-10-06 承認済み計画の自動チェーン: C 終了 → 27B seed 学習(GPU2/3) + GPT-OSS 参考(ollama, GPU0/1)
#   → 27B 推論(F2) → MoE seed 学習(4 GPU 並列) → 70B seed 学習(2,3 と 0,1 の 2 ワーカー, 最後) → F2(MoE) + F4(70B 推論)
cd /home/user/yagahara/LAU16_paper/journal
if [ -z "$RQ_SNAPSHOT" ]; then snap=$(mktemp /tmp/chain.XXXXXX.sh); cp "$0" "$snap"; RQ_SNAPSHOT=1 exec bash "$snap" "$@"; fi
LOG=logs/chain_next.log; log(){ echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
score(){ for r in "$@"; do python3 code/reparse.py $r >> $LOG 2>&1; python3 code/metrics_v2.py --runs $r 2>&1 | grep -E 'relaxed F1' >> $LOG; done; }
OMODELS=/media/user/Elements/ollama-models; OHOST=127.0.0.1:11435; LOCK=/opt/llm/.elements_load.lock
# 再実行可能にするため、学習ワーカーは pid ファイルで管理 (生きていれば再起動しない; wait は kill -0 のポーリング)
launch_train(){ local tag=$1; shift; if [ -f logs/chain_$tag.pid ] && kill -0 $(cat logs/chain_$tag.pid) 2>/dev/null; then log "train $tag already running pid=$(cat logs/chain_$tag.pid)"; return; fi
  nohup bash code/run_train_queue2.sh "$@" >> logs/chain_$tag.log 2>&1 & echo $! > logs/chain_$tag.pid; log "train $tag started pid=$! ($*)"; }
wait_train(){ for tag in "$@"; do while kill -0 $(cat logs/chain_$tag.pid 2>/dev/null) 2>/dev/null; do sleep 60; done; done; }

# ---- 1. queue C 終了待ち → Llama-3.3-70B の zs 2 条件を採点
log "waiting for queue C"
while ! grep -q 'queue C end' logs/queue2_C.log; do sleep 60; done
log "C ended"; score llama-3.3-70b-zs-guided llama-3.3-70b-zs

if [ -f runs/gpt-oss-120b-zs/meta.json ] && [ -f runs/gpt-oss-120b-zs-think-guided/meta.json ]; then GPTOSS_SKIP=1; log "GPT-OSS runs already done, skip ollama"; fi
# ---- 2. ollama 私設インスタンス(GPU0,1)で gpt-oss:120b を Elements から先に読み込む(ロック下、学習のステージングと直列化)
#   ollama 0.30 は本機の driver 535 では CUDA を使えず Vulkan 経由 → CUDA_VISIBLE_DEVICES は無視されるので GGML_VK_VISIBLE_DEVICES で制限
#   (Vulkan0=PCI 02:00.0=nvidia idx1, Vulkan1=PCI 01:00.0=nvidia idx0)
if [ -z "$GPTOSS_SKIP" ]; then
sudo systemctl stop ollama; sleep 2
for p in $(ps -eo pid,cmd | grep '[o]llama serve' | grep -v shell-snapshots | awk '{print $1}'); do kill $p; done; sleep 3
CUDA_VISIBLE_DEVICES=0,1 GGML_VK_VISIBLE_DEVICES=0,1 OLLAMA_HOST=$OHOST OLLAMA_MODELS=$OMODELS OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=2h OLLAMA_LOAD_TIMEOUT=45m \
  nohup ollama serve > logs/ollama_11435.log 2>&1 &
OPID=$!; log "private ollama pid=$OPID"
for i in $(seq 60); do curl -s -m 3 http://$OHOST/api/tags >/dev/null && break; sleep 2; done
log "ollama devices: $(grep -c 'inference compute' logs/ollama_11435.log)"
probe=$(flock $LOCK curl -s -m 3000 http://$OHOST/api/chat -d '{"model":"gpt-oss:120b","messages":[{"role":"user","content":"hi"}],"stream":false,"think":false,"options":{"num_predict":8,"num_ctx":8192}}')
log "probe think=false: $(echo "$probe" | head -c 300)"
log "placement: $(grep -E 'offloaded|model buffer size' logs/ollama_11435.log | tail -4 | tr '\n' ';')"
if echo "$probe" | grep -q 'think'; then ZS_THINK="--ollama_think"; log "gpt-oss cannot disable thinking → zs も think=true"
elif echo "$probe" | grep -q '"error"'; then log "!!!!! ollama load failed, abort GPT-OSS"; GPTOSS_SKIP=1; else ZS_THINK=""; fi

fi
# ---- 3. 27B seed 学習 (GPU2: gemma-s1→s2, GPU3: medgemma-s2) を並行起動
launch_train gpu2 2 gemma-s1 gemma-s2
launch_train gpu3 3 medgemma-s2

# ---- 4. GPT-OSS-120B 参考 2 条件 (zs-guided は think=false だと format が無視されるため省略)
for spec in "gpt-oss-120b-zs $ZS_THINK" "gpt-oss-120b-zs-think-guided --guided --ollama_think"; do
  set -- $spec; name=$1; shift; [ -n "$GPTOSS_SKIP" ] && break
  [ -f runs/$name/meta.json ] && { log "[$name] SKIP done"; continue; }
  log "===== [$name] start ollama $*"
  python3 -u code/eval_v2.py --engine ollama --hosts http://$OHOST --name $name --model gpt-oss:120b --max_new_tokens 3072 --max_model_len 8192 "$@" \
    > logs/run_$name.log 2>&1 || log "!!!!! [$name] FAILED rc=$?"
  log "----- [$name] $(grep -E '^\[.*\] done' logs/run_$name.log | tail -1)"; score $name
done
if [ -z "$GPTOSS_SKIP" ]; then
curl -s -m 60 http://$OHOST/api/chat -d '{"model":"gpt-oss:120b","messages":[],"keep_alive":0}' >/dev/null; sleep 5
kill $OPID; sleep 5; sudo systemctl start ollama; log "private ollama stopped, system ollama restarted"

fi
# ---- 5. 27B 学習終了 → F2 推論(tp2, GPU2,3)
wait_train gpu2 gpu3; log "27B trainings ended"
bash code/run_queue2.sh F2; log "F2 (27B) ended"
score gemma-3-27b-s1-ft gemma-3-27b-s2-ft medgemma-27b-s2-ft

# ---- 6. MoE seed 学習 4 本並列 (GPU0 の大負荷はここから = 以降のジョブは GPU0 無しでも完走できる構成)
launch_train moe1 1 qwen-s1
launch_train moe2 2 qwen-s2
launch_train moe3 3 llmjp4-s1
JOB_TIMEOUT=60h launch_train moe0 0 llmjp4-s2
wait_train moe1 moe2 moe3; log "MoE trainings (gpu1-3) ended"

# ---- 7. 70B seed 学習 (最後): GPU2,3 ワーカーは順方向、GPU0,1 ワーカーは逆方向に同じリストを消化 (ジョブは .lock で排他)。
#   GPU0 が死んでも 2,3 ワーカーが残りを全部やる。GPU0,1 側は 1 本 30h で打ち切り。
wait_train moe0; log "MoE gpu0 ended"
launch_train gpu23 2,3 med42-s1 llama-s1 med42-s2 llama-s2
JOB_TIMEOUT=30h launch_train gpu01 0,1 llama-s2 med42-s2 llama-s1 med42-s1
wait_train gpu23 gpu01; log "70B trainings ended"
# 取りこぼし掃き (失敗/打ち切り分を GPU1 / GPU2,3 で再実行; 完了済みは SKIP)
bash code/run_train_queue2.sh 2,3 med42-s1 llama-s1 med42-s2 llama-s2 >> logs/chain_gpu23.log 2>&1
bash code/run_train_queue2.sh 1 qwen-s1 qwen-s2 llmjp4-s1 llmjp4-s2 >> logs/chain_moe1.log 2>&1; log "sweep ended"

# ---- 8. 残り推論: F2 (MoE merge + 推論, tp2) → F4 (70B seed, tp4)
bash code/run_queue2.sh F2; log "F2 (MoE) ended"; score qwen-s1-ft qwen-s2-ft llm-jp-4-32b-s1-ft llm-jp-4-32b-s2-ft
bash code/run_queue2.sh F4; log "F4 ended"; score med42-70b-s1-ft med42-70b-s2-ft llama-3.3-70b-s1-ft llama-3.3-70b-s2-ft
python3 code/make_tables.py >> $LOG 2>&1; log "########## chain_next END ##########"
