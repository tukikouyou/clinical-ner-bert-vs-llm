# -*- coding: utf-8 -*-
"""
iCorpus の学習分割から LLM ファインチューニング用 SFT データ(JSONL)を作る。

各文について:
  user     : NERプロンプト(ラベル一覧 + 臨床文)  ← 評価時と同一
  assistant: gold の JSON配列 [{"label","text"}]  ← 文字レベルentityから生成

BERTと同じ train/test 分割(train_test_index_1fold.json)を使い、
BERTの学習データと同一のソースにする。zero-width entityは表層が無いので除外。

出力: sft_train.jsonl (学習), sft_val.jsonl(検証, train内から少量)
    形式: {"messages":[{"role":"user",...},{"role":"assistant",...}]}
"""
import os
import re
import json
import glob
import argparse

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "corpus/icorpus_20220531/data/json")
SPLIT = os.path.join(ROOT, "code/data/index/train_test_index_1fold.json")
TAG2IDX = os.path.join(ROOT, "code/results/NER/UTH/NER/NER_tag2idx_0.json")

PROMPT_TEMPLATE = """あなたは日本語の臨床テキストからの固有表現抽出(NER)を行う専門家です。
以下の臨床テキストから、指定されたラベルに該当する表現をすべて抽出してください。

# ラベル一覧
{labels}

# 抽出ルール
- テキスト中に出現した表現をそのまま抜き出すこと(言い換え・要約をしない)
- 文や節ではなく、最小の固有表現(単語〜短い句)を抜き出すこと
- テキスト中に出現したラベルだけを出力すること(空文字や「なし」は出力しない)
- 該当が無ければ空配列 [] を返すこと
- 出力は必ず次形式のJSON配列のみ。説明文を付けない:
[{{"label": "ラベル名", "text": "抽出した表現"}}, ...]

# 臨床テキスト
{text}

# 出力(JSON配列のみ)"""


def load_labels():
    t = json.load(open(TAG2IDX, encoding="utf-8"))
    return sorted({x[2:] for x in t if x.startswith(("B-", "I-"))})


def build(split_key, label_block):
    docs = set(json.load(open(SPLIT))[split_key])
    examples = []
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.json"))):
        name = os.path.basename(f)
        if name not in docs:
            continue
        for rec in json.load(open(f, encoding="utf-8")):
            chars = rec["chars"]
            text = "".join(chars)
            ents = []
            for e in rec["entities"]:
                if e["end"] > e["start"]:  # 表層のあるentityのみ
                    ents.append({"label": e["type"],
                                 "text": "".join(chars[e["start"]:e["end"]])})
            gold = json.dumps(ents, ensure_ascii=False)
            examples.append({"messages": [
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    labels=label_block, text=text)},
                {"role": "assistant", "content": gold}]})
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="./sft_data")
    p.add_argument("--val_frac", type=float, default=0.05)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    label_block = "\n".join(f"- {x}" for x in load_labels())

    train = build("train_name", label_block)
    # train内から末尾を検証に(順序固定・再現性のため乱数を使わない)
    n_val = max(1, int(len(train) * args.val_frac))
    val = train[-n_val:]
    train = train[:-n_val]

    for name, data in [("sft_train.jsonl", train), ("sft_val.jsonl", val)]:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    # 平均出力長の目安
    import statistics
    olens = [len(json.loads(ex["messages"][1]["content"])) for ex in train]
    print(f"train {len(train)} / val {len(val)} 例 → {args.out_dir}")
    print(f"1例あたり平均 {statistics.mean(olens):.1f} entities, 最大 {max(olens)}")


if __name__ == "__main__":
    main()
