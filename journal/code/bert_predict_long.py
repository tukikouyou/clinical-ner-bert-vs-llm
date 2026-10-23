# -*- coding: utf-8 -*-
"""BERT-CRF の予測 CSV (results/NER/<B>/NER/0.csv) に欠けているテスト文 (load_data_cv が 510 文字超として除外した長文) を、
学習済みチェックポイントで ≤510 サブワードの窓に分けて予測し、全 433 文を含む CSV を journal/bert_preds/<B>_pred_IOB_fold0_full.csv に書く。
原本 (shared/code/results) は変更しない。  usage: python3 bert_predict_long.py UTH|NICT   (shared/code で実行, system python)"""
import os, sys, json
import pandas as pd, torch
sys.path.insert(0, "/home/user/yagahara/shared/code")
from lib.util import load_tokenizer  # noqa
from lib.models import BERT_CRF  # noqa
B = sys.argv[1]
CODE = "/home/user/yagahara/shared/code"; OUT = "/home/user/yagahara/LAU16_paper/journal/bert_preds"
BERT_PATH = {"UTH": "/home/user/yagahara/shared/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K", "NICT": "/home/user/yagahara/shared/NICT_BERT-base_JapaneseWikipedia_32K_BPE"}[B]
class A: pass
args = A(); args.bert_path = BERT_PATH
split = json.load(open(f"{CODE}/data/index/train_test_index_1fold.json"))
df = pd.read_csv(f"{CODE}/data/csv/icorpus_{B}.csv"); df["name"] = df["name"].astype(str); df["rel_type"] = df["rel_type"].fillna("None").astype(str)
test = df[df["name"].isin(split["test_name"])]
pred = pd.read_csv(f"{CODE}/results/NER/{B}/NER/0.csv"); have = set(pred["unique_no"].astype(str))
missing = [u for u in test["unique_no"].astype(str).unique() if u not in have]
print(B, "test sentences", test["unique_no"].nunique(), "already predicted", len(have), "missing", len(missing))
tag2idx = json.load(open(f"{CODE}/results/NER/{B}/NER/NER_tag2idx_0.json")); idx2tag = {v: k for k, v in tag2idx.items()}
tok = load_tokenizer(args)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT_CRF(args, tag2idx).to(dev); model.load_state_dict(torch.load(f"{CODE}/models/NER/{B}/NER/ner_model_0.pt", map_location=dev)); model.eval()
MAXW = 510; BREAKS = {"。", "．", "，", "、", ",", ".", "；", ";"}

def windows(words):
    """≤MAXW サブワードの窓に分割。切れ目は窓内の最後の句読点の直後を優先。"""
    out, i = [], 0
    while i < len(words):
        j = min(len(words), i + MAXW)
        if j < len(words):
            cut = max((k + 1 for k in range(i, j) if words[k] in BREAKS), default=j)
            if cut <= i:
                cut = j
            j = cut
        out.append((i, j)); i = j
    return out

rows, audit = [], {}
with torch.inference_mode():
    for u in missing:
        g = test[test["unique_no"].astype(str) == u]
        words = list(g["word"].astype(str)); tags = []
        wins = windows(words)
        for a, b in wins:
            ids = tok.convert_tokens_to_ids(["[CLS]"] + words[a:b] + ["[SEP]"])
            logits = model(torch.tensor([ids]).to(dev))
            dec = model.crf.decode(logits)[0]
            tags.extend(idx2tag[t] for t in dec[:b - a])
        assert len(tags) == len(words), (u, len(tags), len(words))
        g = g.copy(); g["pred_IOB"] = tags; rows.append(g); audit[u] = {"n_subwords": len(words), "n_windows": len(wins)}
new = pd.concat(rows) if rows else pd.DataFrame(columns=pred.columns)
full = pd.concat([pred, new[pred.columns]], ignore_index=True)
os.makedirs(OUT, exist_ok=True)
full.to_csv(f"{OUT}/{B}_pred_IOB_fold0_full.csv", index=False)
json.dump({"bert_type": B, "missing_sentences": missing, "audit": audit}, open(f"{OUT}/{B}_long_sentence_audit.json", "w"), ensure_ascii=False, indent=1)
print(B, "wrote", f"{OUT}/{B}_pred_IOB_fold0_full.csv", "rows", len(full), "sentences", full["unique_no"].nunique(), "windows:", sum(v["n_windows"] for v in audit.values()))
