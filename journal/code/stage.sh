#!/bin/bash
# Elements(USB HDD) 上のモデルを根分区 SSD の /opt/llm/stage に順次コピーしてから読む(mmap のランダム読みは HDD で 10-50MB/s に落ちるため)。
# usage: stage.sh <model-path-or-hf-repo-id> <HF_HOME>   → 標準出力にステージ済みパス
#        stage.sh --reserve <GB>                           → 空き確保のみ
# コピーは /opt/llm/.elements_load.lock で直列化。空き容量が足りなければ未使用の古いステージから削除。
set -e
SRC=$1; HF=$2; STAGE=/opt/llm/stage; mkdir -p $STAGE; LOCK=/opt/llm/.elements_load.lock
evict_until() { # <GB needed>: 空きが need+40G になるまで、古い順に未使用のステージを削除(使用中はスキップして次へ)
  local need=$1 cand avail
  for cand in $(ls -tr $STAGE 2>/dev/null); do
    avail=$(df --output=avail -BG /opt | tail -1 | tr -d 'G '); [ "$avail" -ge $((need + 40)) ] && return 0
    if pgrep -af "stage/$cand" | grep -v -e shell-snapshots -e stage.sh >/dev/null; then echo "stage: $cand in use, skip" >&2; continue; fi
    echo "stage: evict $cand" >&2; rm -rf "$STAGE/$cand"
  done
  avail=$(df --output=avail -BG /opt | tail -1 | tr -d 'G ')
  [ "$avail" -ge $((need + 40)) ] || echo "stage: WARNING only ${avail}G free (need $((need + 40))G)" >&2
  return 0; }
if [ "$SRC" = "--reserve" ]; then evict_until "$HF"; exit 0; fi
if [[ "$SRC" == /* ]]; then src="$SRC"; name=$(basename "$SRC")
else d="$HF/hub/models--${SRC//\//--}/snapshots"; src="$d/$(ls -t $d | head -1)"; name=${SRC//\//--}; fi
[[ "$src" == /media/user/Elements/* ]] || { echo "$src"; exit 0; }   # Elements 以外はそのまま
dst=$STAGE/$name
if [ -f "$dst/.staged" ]; then touch "$dst/.staged"; echo "$dst"; exit 0; fi
need=$(du -sL --block-size=1G "$src" | cut -f1); need=$((need + 5))
exec 9>$LOCK; flock 9
# ロック取得中に別プロセスが同じモデルを置き終えていたらそれを使う (再コピーすると使用中の dst を消してしまう: 2026-10-07 qwen-s2 失敗の原因)
if [ -f "$dst/.staged" ]; then touch "$dst/.staged"; flock -u 9; echo "$dst"; exit 0; fi
evict_until $need
avail=$(df --output=avail -BG /opt | tail -1 | tr -d 'G ')
if [ "$avail" -lt $((need + 10)) ]; then   # 入り切らないならコピーせず失敗 (根分区を満杯にしない; 呼び出し側は Elements 直読にフォールバック)
  echo "stage: SKIP $name: only ${avail}G free, need $((need + 10))G" >&2; flock -u 9; exit 1; fi
rm -rf "$dst"; mkdir -p "$dst"
echo "stage: copying $src → $dst ($need G) $(date '+%H:%M')" >&2
cp -rL "$src/." "$dst/"
touch "$dst/.staged"; echo "stage: done $(date '+%H:%M')" >&2
flock -u 9
echo "$dst"
