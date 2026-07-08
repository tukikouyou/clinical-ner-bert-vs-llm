# -*- coding: utf-8 -*-
"""
vLLM による電子カルテ(EHR)からの LLM-NER 抽出(大量バッチ処理)。

BERT側 predict_reports.py と同じ入出力方針:
    cp932 の複数列CSV(【所見】等) → 1テキスト=1プロンプト → vLLM一括生成
    → JSONパース → entities.csv + stats.json (doc_id/year/column/label/text)

vLLM の continuous batching により、逐次(transformers batch=1)の
~8s/件 が数十〜百倍高速化される。レポートを1プロンプトに詰め込むと
どのレポート由来かの取り違えが起きやすいので、1レポート=1プロンプトのまま
エンジン側でバッチ化する。

使い方(venv: /opt/llm/vllm-venv):
    /opt/llm/vllm-venv/bin/python llm_extract_vllm.py \
        --model llm-jp/llm-jp-3.1-13b-instruct4 \
        --data_dir "../predicted data" \
        --out_dir ./results/ehr/llm-jp-13b \
        [--limit 2000]   # 動作確認・サンプル用に件数制限
"""
import os
import re
import json
import glob
import time
import argparse
from collections import Counter

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_TAG2IDX = os.path.join(os.path.dirname(__file__),
                               "../code/results/NER/UTH/NER/NER_tag2idx_0.json")

PROMPT_TEMPLATE = """あなたは日本語の臨床テキストからの固有表現抽出(NER)を行う専門家です。
以下の臨床テキストから、指定されたラベルに該当する表現をすべて抽出してください。

# ラベル一覧
{labels}

# 抽出ルール
- テキスト中に出現した表現をそのまま抜き出すこと(言い換え・要約をしない)
- テキスト中に出現したラベルだけを出力すること(空文字や「なし」は出力しない)
- 同じラベルが複数回出現したら、その回数だけ出力すること
- 該当が無ければ空配列 [] を返すこと
- 出力は必ず次形式のJSON配列のみ。説明文を付けない:
[{{"label": "ラベル名", "text": "抽出した表現"}}, ...]

# 臨床テキスト
{text}

# 出力(JSON配列のみ)
"""


def load_labels(tag2idx_path):
    tag2idx = json.load(open(tag2idx_path, encoding="utf-8"))
    return sorted({t[2:] for t in tag2idx if t.startswith(("B-", "I-"))})


def load_ehr(data_dir, columns, limit=None):
    """(doc_id, year, column, text) を返す"""
    rows = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        year = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f, encoding="cp932")
        for i, r in df.iterrows():
            for col in columns:
                if col not in df.columns:
                    continue
                t = r[col]
                if isinstance(t, str) and t.strip():
                    rows.append((f"{year}#{i}", year, col, t.strip()))
            if limit and len(rows) >= limit:
                return rows[:limit]
    return rows


def _repair(cand):
    pos = cand.rfind("}")
    return cand[:pos + 1] + "]" if pos != -1 else None


def parse_entities(raw):
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    cands = [m.group(1)] if m else []
    m2 = re.search(r"\[.*\]", raw, re.DOTALL)
    if m2:
        cands.append(m2.group(0))
    if not cands and "[" in raw:
        cands.append(raw[raw.index("["):])
    for c in cands:
        for a in (c, _repair(c)):
            if a is None:
                continue
            try:
                obj = json.loads(a)
                if isinstance(obj, list):
                    return [x for x in obj if isinstance(x, dict)
                            and "label" in x and "text" in x]
            except json.JSONDecodeError:
                continue
    return None  # パース失敗


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="../predicted data")
    p.add_argument("--columns", nargs="+", default=["【所見】", "【診断/所見のまとめ】"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag2idx_path", default=DEFAULT_TAG2IDX)
    p.add_argument("--limit", type=int, default=None, help="件数制限(サンプル用)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size(GPU数)")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_mem", type=float, default=0.90)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels = load_labels(args.tag2idx_path)
    label_block = "\n".join(f"- {x}" for x in labels)
    reports = load_ehr(args.data_dir, args.columns, args.limit)
    print(f"{len(reports)} テキスト / {len(labels)} ラベル / model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for _, _, _, text in reports:
        msg = [{"role": "user",
                "content": PROMPT_TEMPLATE.format(labels=label_block, text=text)}]
        prompts.append(tok.apply_chat_template(msg, add_generation_prompt=True,
                                               tokenize=False))

    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    gen_sec = time.time() - t0

    all_ents, n_fail = [], 0
    for (doc_id, year, col, _text), out in zip(reports, outputs):
        raw = out.outputs[0].text
        ents = parse_entities(raw)
        if ents is None:
            n_fail += 1
            continue
        for e in ents:
            tv = str(e["text"]).strip()
            if tv in ("", "なし", "特になし", "N/A", "None", "null"):
                continue
            all_ents.append({"doc_id": doc_id, "year": year, "column": col,
                             "label": str(e["label"]), "text": tv})

    ent_df = pd.DataFrame(all_ents)
    ent_df.to_csv(os.path.join(args.out_dir, "entities.csv"), index=False)
    stats = {
        "model": args.model, "engine": "vllm",
        "n_texts": len(reports), "n_entities": len(all_ents),
        "entities_per_text": round(len(all_ents) / max(len(reports), 1), 2),
        "n_parse_fail": n_fail,
        "gen_sec": round(gen_sec, 1),
        "texts_per_sec": round(len(reports) / max(gen_sec, 1e-9), 2),
        "per_label": dict(Counter(e["label"] for e in all_ents).most_common()),
        "per_column": dict(Counter(e["column"] for e in all_ents)),
        "per_year_entities": dict(Counter(e["year"] for e in all_ents)),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n{len(all_ents)} entities / {len(reports)} texts "
          f"({stats['texts_per_sec']} text/s, parse失敗 {n_fail}) → {args.out_dir}")


if __name__ == "__main__":
    main()
