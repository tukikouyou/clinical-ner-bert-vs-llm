# -*- coding: utf-8 -*-
"""BERT-CRF の再学習 (期刊版 v2)。shared/code の BERT_CRF と同じ構成・同じハイパーパラメータで、次の 3 点だけ変える:
  1. 510 文字超の文を除外しない: 510 サブワード超の文は句読点で ≤510 の窓に分割して学習・予測 (窓先頭の I- は B- に直す)
  2. 検証集を LLM の QLoRA と同一にする: 学習 146 文書の文を build_sft_data と同じ順序で並べ、末尾 88 文 (5%) を検証、残り 1,672 文を学習
  3. CRF の損失マスクは全トークン有効 (元コードの mask=(tag!=0) はタグ index 0 のトークンを損失から外していた; batch=1 なので padding は無い)
出力: journal/bert_v2/<B>_s<seed>/{model.pt,train_log.json,metrics.json}, journal/bert_preds/<B>-v2-s<seed>_pred_IOB_full.csv
usage (shared/code を cwd にして system python3): python3 bert_train_v2.py --bert_type UTH --seed 42 [--max_epoch 250 --patience 15]
"""
import os, sys, json, glob, time, random, argparse, copy
import numpy as np, pandas as pd, torch, transformers
from torch.nn.utils.rnn import pad_sequence
CODE = "/home/user/yagahara/shared/code"; sys.path.insert(0, CODE)
from lib.models import BERT_CRF  # noqa
from lib.util import load_tokenizer  # noqa
from lib.util.eval_ner import eval_ner_strict  # noqa
J = "/home/user/yagahara/LAU16_paper/journal"
CORPUS = "/home/user/yagahara/shared/corpus/icorpus_20220531/data/json"
BERT_PATH = {"UTH": "/home/user/yagahara/shared/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K", "NICT": "/home/user/yagahara/shared/NICT_BERT-base_JapaneseWikipedia_32K_BPE"}
MAXW = 510; BREAKS = {"。", "．", "，", "、", ",", ".", "；", ";"}

p = argparse.ArgumentParser()
p.add_argument("--bert_type", required=True); p.add_argument("--seed", type=int, default=42)
p.add_argument("--max_epoch", type=int, default=250); p.add_argument("--patience", type=int, default=15)
p.add_argument("--n_val", type=int, default=88)
args = p.parse_args(); args.bert_path = BERT_PATH[args.bert_type]; args.batch_size = 1
B = args.bert_type; OUT = os.path.join(J, "bert_v2", f"{B}_s{args.seed}"); os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(J, "bert_preds"), exist_ok=True)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
dev = torch.device("cuda")

# ---- データ: 長さフィルタ無し
df = pd.read_csv(f"{CODE}/data/csv/icorpus_{B}.csv"); df["name"] = df["name"].astype(str); df["unique_no"] = df["unique_no"].astype(str)
df["rel_type"] = df["rel_type"].fillna("None").astype(str); df = df.sort_values("serial")
split = json.load(open(f"{CODE}/data/index/train_test_index_1fold.json"))
train_docs, test_docs = set(split["train_name"]), set(split["test_name"])
# タグ辞書は全データ (学習+検証+テスト, 長文含む) の IOB から作る (元の NER_tag2idx_0.json は長文除外後の 203 種で、長文にしか無いタグが欠ける)
tag2idx = {t: i for i, t in enumerate(sorted(df["IOB"].astype(str).unique()))}; idx2tag = {v: k for k, v in tag2idx.items()}
json.dump(tag2idx, open(os.path.join(OUT, "tag2idx.json"), "w"), ensure_ascii=False, indent=1)
tok = load_tokenizer(args)
# 検証集: build_sft_data.py と同じ順序 (sorted glob の文書順 × レコード順) の末尾 n_val 文
order = []
for f in sorted(glob.glob(os.path.join(CORPUS, "*.json"))):
    stem = os.path.basename(f).replace(".json", "")
    if f"{stem}.json" not in train_docs:
        continue
    for rec in json.load(open(f, encoding="utf-8")):
        order.append(f"{stem}_{rec['sentence_id']}")
val_keys = set(order[-args.n_val:]); train_keys = set(order[:-args.n_val])
groups = {u: g for u, g in df.groupby("unique_no", sort=False)}
missing_in_csv = [k for k in order if k not in groups]
print(B, f"seed {args.seed}: corpus train sentences {len(order)}, in CoNLL {len(order) - len(missing_in_csv)}, val {len(val_keys)}, train {len(train_keys)}; missing from CoNLL: {len(missing_in_csv)}", flush=True)


def windows(words):
    out, i = [], 0
    while i < len(words):
        j = min(len(words), i + MAXW)
        if j < len(words):
            cut = max((k + 1 for k in range(i, j) if words[k] in BREAKS), default=j)
            j = cut if cut > i else j
        out.append((i, j)); i = j
    return out


def sequences(keys):
    """[(key, a, b, ids, tag_ids)] 窓ごと"""
    seqs = []
    for k in keys:
        if k not in groups:
            continue
        g = groups[k]; words = list(g["word"].astype(str)); tags = list(g["IOB"].astype(str))
        for a, b in windows(words):
            t = tags[a:b]
            if t and t[0].startswith("I-"):
                t = ["B-" + t[0][2:]] + t[1:]
            ids = tok.convert_tokens_to_ids(["[CLS]"] + words[a:b] + ["[SEP]"])
            seqs.append((k, a, b, ids, [tag2idx[x] for x in t]))
    return seqs


train_seqs = sequences([k for k in order if k in train_keys]); val_seqs = sequences([k for k in order if k in val_keys])
test_keys = list(df[df["name"].isin(test_docs)]["unique_no"].unique()); test_seqs = sequences(test_keys)
n_long_train = sum(1 for k in train_keys if k in groups and len(groups[k]) > MAXW)
print(f"windows: train {len(train_seqs)} (from {len(train_keys)} sentences, {n_long_train} split), val {len(val_seqs)}, test {len(test_seqs)} (from {len(test_keys)} sentences)", flush=True)


def predict(model, seqs):
    """窓ごとの予測を文に連結 → {key: [tags]}"""
    out = {}
    model.eval()
    with torch.inference_mode():
        for k, a, b, ids, _ in seqs:
            logits = model(torch.tensor([ids]).to(dev))
            dec = model.crf.decode(logits)[0][:b - a]
            out.setdefault(k, []).extend(idx2tag[t] for t in dec)
    return out


def to_df(keys, pred):
    rows = []
    for k in keys:
        if k not in groups:
            continue
        g = groups[k].copy(); g["pred_IOB"] = pred.get(k, ["O"] * len(g)); rows.append(g)
    return pd.concat(rows)


# ---- 学習 (元の train_val_loop_ner と同じ設定: AdamW 2e-5, 線形 warmup 10%, batch 1, 早停は検証 strict micro F1)
model = BERT_CRF(args, tag2idx).to(dev)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
steps_per_epoch = len(train_seqs)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.max_epoch * steps_per_epoch * 0.1), num_training_steps=steps_per_epoch * args.max_epoch)
best_f, best_epoch, no_improve, log = -1.0, -1, 0, {"epoch": [], "train_loss": [], "val_F": [], "epoch_sec": []}
t0 = time.time()
for epoch in range(args.max_epoch):
    te = time.time(); model.train(); idx = list(range(len(train_seqs))); random.shuffle(idx); running = 0.0
    for i in idx:
        _, _, _, ids, tg = train_seqs[i]
        optimizer.zero_grad()
        sent = torch.tensor([ids]).to(dev); tag = torch.tensor([tg]).to(dev)
        logits = model(sent)
        loss = -model.crf(logits, tag, mask=torch.ones_like(tag, dtype=torch.bool), reduction="sum")
        loss.backward(); optimizer.step(); scheduler.step(); running += loss.item()
    pv = predict(model, val_seqs); vdf = to_df([k for k in order if k in val_keys], pv)
    val_f = eval_ner_strict(vdf)["micro avg"]["f1"]
    log["epoch"].append(epoch); log["train_loss"].append(round(running, 1)); log["val_F"].append(round(val_f, 4)); log["epoch_sec"].append(round(time.time() - te, 1))
    print(f"epoch {epoch} loss {running:.0f} val_F {val_f:.4f} ({time.time()-te:.0f}s)", flush=True)
    if val_f > best_f:
        best_f, best_epoch, no_improve = val_f, epoch, 0; torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    else:
        no_improve += 1
        if args.patience and no_improve >= args.patience:
            print(f"early stop at epoch {epoch} (best {best_f:.4f} @ {best_epoch})", flush=True); break
train_sec = round(time.time() - t0, 1)
json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"), indent=1)

# ---- テスト予測 (ベストモデル)
model.load_state_dict(torch.load(os.path.join(OUT, "model.pt"), map_location=dev))
pt = predict(model, test_seqs); tdf = to_df(test_keys, pt)
csv = os.path.join(J, "bert_preds", f"{B}-v2-s{args.seed}_pred_IOB_full.csv"); tdf.to_csv(csv, index=False)
metrics = {"bert_type": B, "seed": args.seed, "best_epoch": best_epoch, "best_val_F": best_f, "train_sec": train_sec, "epochs_run": len(log["epoch"]),
           "n_train_sent": len(train_keys), "n_train_windows": len(train_seqs), "n_long_train_split": n_long_train, "n_val_sent": len(val_keys), "n_test_sent": tdf["unique_no"].nunique(),
           "n_test_windows": len(test_seqs), "val_keys_tail": sorted(val_keys)[:3], "pred_csv": csv, "test_strict_micro_tag_level": eval_ner_strict(tdf)["micro avg"]}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), ensure_ascii=False, indent=1)
print("DONE", json.dumps({k: metrics[k] for k in ("best_epoch", "best_val_F", "train_sec", "n_train_windows", "n_test_sent")}), flush=True)
