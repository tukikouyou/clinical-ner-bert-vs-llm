# -*- coding: utf-8 -*-
"""gemma-3-27b-it (Gemma3ForConditionalGeneration, 画像塔付き) から言語部分だけを抜き出し、
medgemma-27b-text と同一レイアウト (Gemma3ForCausalLM, model.*) のチェックポイントを作る。
QLoRA の target_modules が vision tower に当たらないようにし、MedGemma と完全に同じコードパスで学習・推論するため。"""
import os, json, sys, shutil
from safetensors import safe_open
from safetensors.torch import save_file
SRC = sys.argv[1]; DST = sys.argv[2]; REF = "/opt/llm/models/medgemma-27b-text"
os.makedirs(DST, exist_ok=True)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
ref = set(json.load(open(f"{REF}/model.safetensors.index.json"))["weight_map"])
new_map, kept, dropped = {}, 0, 0
for shard in sorted(set(idx.values())):
    out = {}
    with safe_open(f"{SRC}/{shard}", "pt") as f:
        for k in f.keys():
            if k.startswith("language_model."):
                nk = k[len("language_model."):]
                out[nk] = f.get_tensor(k); kept += 1
            else:
                dropped += 1
    if out:
        save_file(out, f"{DST}/{shard}", metadata={"format": "pt"})
        for nk in out:
            new_map[nk] = shard
    print(shard, "kept", len(out), flush=True)
json.dump({"metadata": {}, "weight_map": new_map}, open(f"{DST}/model.safetensors.index.json", "w"), indent=1)
names = set(new_map)
print("kept", kept, "dropped(vision/projector)", dropped)
print("missing vs medgemma:", sorted(ref - names)[:5], "extra:", sorted(names - ref)[:5])
# config: text_config を Gemma3ForCausalLM 形式に
src = json.load(open(f"{SRC}/config.json")); refc = json.load(open(f"{REF}/config.json"))
tc = src["text_config"]
cfg = dict(refc)
for k, v in tc.items():
    cfg[k] = v
cfg["architectures"] = ["Gemma3ForCausalLM"]; cfg["model_type"] = "gemma3_text"
# gemma-3-27b-it の埋め込みは 262208 行(画像関連の特殊トークン 64 個を含む)。medgemma-27b-text は 262144 行。
# vocab_size を埋め込みの行数に合わせないと損失計算で view エラーになる(2026-10-05 に判明)。
from safetensors import safe_open as _so
_emb = json.load(open(f"{DST}/model.safetensors.index.json"))["weight_map"]["model.embed_tokens.weight"]
cfg["vocab_size"] = _so(f"{DST}/{_emb}", "pt").get_slice("model.embed_tokens.weight").get_shape()[0]
for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
    if k in src: cfg[k] = src[k]
diff = {k: (refc.get(k), cfg.get(k)) for k in set(refc) | set(cfg) if refc.get(k) != cfg.get(k)}
print("config diff vs medgemma:", diff)
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
for fn in ("generation_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
           "special_tokens_map.json", "added_tokens.json", "chat_template.json"):
    if os.path.exists(f"{SRC}/{fn}"):
        shutil.copy(f"{SRC}/{fn}", f"{DST}/{fn}")
print("DONE", DST)
