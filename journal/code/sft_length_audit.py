# -*- coding: utf-8 -*-
"""学習データ (sft_train.jsonl) の入力+目標のトークン長を各モデルの tokenizer で数え、max_seq_len 4096 で切り詰められる例と
その中の正解エンティティの割合を集計 → stats/sft_length_audit.json"""
import os, json, sys
from transformers import AutoTokenizer
SFT = "/home/user/yagahara/shared/llm/sft_data/sft_train.jsonl"
HF = "/media/user/Elements/hf-cache"
MODELS = {"LLM-jp-1.8B": "llm-jp/llm-jp-3.1-1.8b-instruct4", "LLM-jp-3.1-13B": "llm-jp/llm-jp-3.1-13b-instruct4",
          "SIP-jmed-13B": "SIP-med-LLM/SIP-jmed-llm-3-13b-OP-32k-R0.1", "Gemma-3-27B": "/media/user/Elements/models/gemma-3-27b-text-it",
          "MedGemma-27B": "/media/user/Elements/models/medgemma-27b-text", "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B-Instruct-2507",
          "LLM-jp-4-32B": "llm-jp/llm-jp-4-32b-a3b-thinking", "Llama-3.3-70B": "/media/user/Elements/models/Llama-3.3-70B-Instruct",
          "Med42-70B": "/media/user/Elements/models/med42-70b"}
rows = [json.loads(l) for l in open(SFT, encoding="utf-8")]
out = {"n_examples": len(rows)}
total_ent = sum(len(json.loads(r["messages"][-1]["content"])) for r in rows)
out["n_entities"] = total_ent
for name, path in MODELS.items():
    try:
        tok = AutoTokenizer.from_pretrained(path, cache_dir=os.path.join(HF, "hub") if not path.startswith("/") else None, trust_remote_code=True)
    except Exception as e:
        out[name] = {"error": str(e)[:200]}; print(name, "ERR", str(e)[:120]); continue
    lens, over, ent_lost, ent_in_over = [], 0, 0, 0
    for r in rows:
        try:
            text = tok.apply_chat_template(r["messages"], tokenize=False)
        except Exception:
            text = r["messages"][0]["content"] + "\n" + r["messages"][-1]["content"]
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        n_prompt = len(tok(tok.apply_chat_template(r["messages"][:1], tokenize=False, add_generation_prompt=True) if hasattr(tok, "chat_template") and tok.chat_template else r["messages"][0]["content"], add_special_tokens=False)["input_ids"])
        ents = json.loads(r["messages"][-1]["content"]); ne = len(ents)
        lens.append(n)
        if n > 4096:
            over += 1; ent_in_over += ne
            # 目標部分のうち収まる割合 ≈ (4096 - prompt) / (n - prompt)
            keep = max(0.0, min(1.0, (4096 - n_prompt) / max(1, n - n_prompt)))
            ent_lost += int(round(ne * (1 - keep)))
    out[name] = {"max_tokens": max(lens), "mean_tokens": round(sum(lens) / len(lens)), "n_over_4096": over, "share_over": round(over / len(rows), 4),
                 "entities_in_over": ent_in_over, "entities_lost_est": ent_lost, "entity_share_lost_est": round(ent_lost / total_ent, 4)}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../stats/sft_length_audit.json"), "w"), ensure_ascii=False, indent=1)
