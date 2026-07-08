# -*- coding: utf-8 -*-
"""
ollama 経由の LLM-NER 抽出(EHR)。vLLM版 llm_extract_vllm.py と同じ入出力。
ollama に既にあるモデル(qwen3.6:35b, llama3.3:70b, gpt-oss:120b)用。

並列リクエストでスループットを稼ぐ(ThreadPoolExecutor)。

使い方:
    python3 llm_extract_ollama.py --model qwen3.6:35b \
        --data_dir <sample_dir> --out_dir ./results/ehr/qwen35b [--concurrency 6]
"""
import os
import re
import json
import glob
import time
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

DEFAULT_TAG2IDX = os.path.join(os.path.dirname(__file__),
                               "../code/results/NER/UTH/NER/NER_tag2idx_0.json")

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

# 出力(JSON配列のみ)
"""


def load_labels(path):
    tag2idx = json.load(open(path, encoding="utf-8"))
    return sorted({t[2:] for t in tag2idx if t.startswith(("B-", "I-"))})


def load_ehr(data_dir, columns, limit=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        year = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f, encoding="cp932")
        for i, r in df.iterrows():
            for col in columns:
                if col in df.columns and isinstance(r[col], str) and r[col].strip():
                    rows.append((f"{year}#{i}", year, col, r[col].strip()))
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
    return None


def call_ollama(host, model, prompt, max_tokens):
    # think=False: 推論モデル(qwen3.6等)の思考出力を抑制し、直接JSONを得る
    payload = {"model": model, "stream": False, "think": False,
               "messages": [{"role": "user", "content": prompt}],
               "options": {"temperature": 0.0, "num_predict": max_tokens,
                           "num_ctx": 4096}}
    r = requests.post(host.rstrip("/") + "/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"].get("content", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="../predicted data")
    p.add_argument("--columns", nargs="+", default=["【所見】", "【診断/所見のまとめ】"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag2idx_path", default=DEFAULT_TAG2IDX)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--hosts", default="http://127.0.0.1:11434",
                   help="カンマ区切りの ollama ベースURL(GPU毎の複数インスタンス)")
    p.add_argument("--per_host", type=int, default=2,
                   help="1インスタンスあたりの同時リクエスト数")
    args = p.parse_args()
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    labels = load_labels(args.tag2idx_path)
    label_block = "\n".join(f"- {x}" for x in labels)
    reports = load_ehr(args.data_dir, args.columns, args.limit)
    concurrency = len(hosts) * args.per_host
    print(f"{len(reports)} テキスト / {len(labels)} ラベル / model={args.model} / "
          f"{len(hosts)}インスタンス×{args.per_host}={concurrency}並列")

    results = [None] * len(reports)

    def work(i):
        _, _, _, text = reports[i]
        prompt = PROMPT_TEMPLATE.format(labels=label_block, text=text)
        host = hosts[i % len(hosts)]  # ラウンドロビンで4GPUに分散
        try:
            return i, call_ollama(host, args.model, prompt, args.max_new_tokens)
        except Exception as e:
            return i, f"__ERROR__ {e}"

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(work, i) for i in range(len(reports))]
        for fu in as_completed(futs):
            i, raw = fu.result()
            results[i] = raw
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(reports)}  {done/el:.2f} text/s")
    gen_sec = time.time() - t0

    all_ents, n_fail = [], 0
    for (doc_id, year, col, _t), raw in zip(reports, results):
        ents = parse_entities(raw) if raw and not raw.startswith("__ERROR__") else None
        if ents is None:
            n_fail += 1
            continue
        for e in ents:
            tv = str(e["text"]).strip()
            if tv in ("", "なし", "特になし", "N/A", "None", "null"):
                continue
            all_ents.append({"doc_id": doc_id, "year": year, "column": col,
                             "label": str(e["label"]), "text": tv})

    pd.DataFrame(all_ents).to_csv(os.path.join(args.out_dir, "entities.csv"), index=False)
    stats = {
        "model": args.model, "engine": "ollama",
        "n_texts": len(reports), "n_entities": len(all_ents),
        "entities_per_text": round(len(all_ents) / max(len(reports), 1), 2),
        "n_parse_fail": n_fail, "gen_sec": round(gen_sec, 1),
        "texts_per_sec": round(len(reports) / max(gen_sec, 1e-9), 2),
        "per_label": dict(Counter(e["label"] for e in all_ents).most_common()),
        "per_column": dict(Counter(e["column"] for e in all_ents)),
        "per_year_entities": dict(Counter(e["year"] for e in all_ents)),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n{len(all_ents)} entities / {len(reports)} texts "
          f"({stats['texts_per_sec']} text/s, fail {n_fail}) → {args.out_dir}")


if __name__ == "__main__":
    main()
