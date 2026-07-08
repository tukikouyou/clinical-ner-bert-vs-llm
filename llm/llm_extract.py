# -*- coding: utf-8 -*-
"""
LLMによる構造化抽出パイプライン(BERT側 extract_structured.py と同じ出力形式)

    生テキスト → プロンプト(ラベル定義付き) → LLM → JSONパース → entities.json / entities.csv

対応モデル: HuggingFace形式のCausalLM全般
    例: llm-jp/llm-jp-3.1-13b-instruct4, openai/gpt-oss-120b,
        sip-med-llm/SIP-jmed-llm-2-8x13b-OP-instruct, ~/models/llama4-hf など

使い方(venv: /opt/llm/venv):
    /opt/llm/venv/bin/python llm_extract.py \
        --model openai/gpt-oss-120b \
        --input sample_reports.txt \
        --output_dir ./results/gpt-oss-120b

    # ラベルセットはデフォルトでBERT側の tag2idx から自動生成(B-/I-を剥がして統合)
"""
import os
import re
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TAG2IDX = os.path.join(os.path.dirname(__file__),
                               "../code/results/NER/UTH/NER/NER_tag2idx_0.json")

PROMPT_TEMPLATE = """あなたは日本語の臨床テキストからの固有表現抽出(NER)を行う専門家です。
以下の臨床テキストから、指定されたラベルに該当する表現をすべて抽出してください。

# ラベル一覧
{labels}

# 抽出ルール
- テキスト中に出現した表現をそのまま抜き出すこと(言い換え・要約をしない)
- テキスト中に出現したラベルだけを出力すること。出現しなかったラベルは出力に
  含めない(空文字や「なし」を出力しない)
- 同じラベルが複数回出現したら、その回数だけ出力すること
- 該当する表現が1つもなければ空の配列 [] を返すこと
- 出力は必ず次の形式のJSON配列のみとし、説明文を付けないこと:
[{{"label": "ラベル名", "text": "抽出した表現"}}, ...]

# 臨床テキスト
{text}

# 出力(JSON配列のみ)
"""


def load_labels(tag2idx_path, labels_path=None):
    """ラベル一覧を返す。labels_path(JSON: {label: 説明} または [label,...])優先。"""
    if labels_path:
        obj = json.load(open(labels_path, encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
        return {x: "" for x in obj}
    tag2idx = json.load(open(tag2idx_path, encoding="utf-8"))
    labels = sorted({t[2:] for t in tag2idx if t.startswith(("B-", "I-"))})
    return {x: "" for x in labels}


def load_reports(input_path):
    reports = []
    if os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(input_path, fname), encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                reports.append((os.path.splitext(fname)[0], text))
    else:
        with open(input_path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    reports.append((f"doc{i:04d}", line))
    return reports


def parse_json_entities(raw_text):
    """LLM出力からJSON配列を頑健に取り出す"""
    # コードブロックを優先
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_text, re.DOTALL)
    candidates = [m.group(1)] if m else []
    # 最後に現れる [ ... ] ブロック(貪欲)も候補に
    m2 = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if m2:
        candidates.append(m2.group(0))
    # 閉じ括弧が無い(途中で切れた)場合: 最初の [ から末尾までを候補に
    if not candidates and "[" in raw_text:
        candidates.append(raw_text[raw_text.index("["):])
    for cand in candidates:
        for attempt in (cand, _repair_truncated_array(cand)):
            if attempt is None:
                continue
            try:
                obj = json.loads(attempt)
                if isinstance(obj, list):
                    return [x for x in obj
                            if isinstance(x, dict) and "label" in x and "text" in x], True
            except json.JSONDecodeError:
                continue
    return [], False


def _repair_truncated_array(cand):
    """max_new_tokensで途中終了したJSON配列を、最後の完全な要素まで切って閉じる"""
    pos = cand.rfind("}")
    if pos == -1:
        return None
    return cand[:pos + 1] + "]"


def build_messages(labels, text):
    label_lines = "\n".join(f"- {k}" + (f": {v}" if v else "") for k, v in labels.items())
    return [{"role": "user",
             "content": PROMPT_TEMPLATE.format(labels=label_lines, text=text)}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help="HFモデルID またはローカルパス")
    parser.add_argument('--input', type=str, required=True,
                        help="1行=1レポートのtxt、または.txtディレクトリ")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tag2idx_path', type=str, default=DEFAULT_TAG2IDX)
    parser.add_argument('--labels_path', type=str, default=None,
                        help="ラベル定義JSON({label:説明}or[label,...])。省略時はtag2idxから生成")
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--dtype', type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    labels = load_labels(args.tag2idx_path, args.labels_path)
    reports = load_reports(args.input)
    print(f"{len(reports)} 件のレポート / {len(labels)} ラベル")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=args.dtype, device_map="auto")
    model.eval()

    has_chat_template = tokenizer.chat_template is not None

    entities, raw_outputs = [], {}
    for doc_id, text in reports:
        messages = build_messages(labels, text)
        if has_chat_template:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt", return_dict=True).to(model.device)
        else:
            # ベースモデル(chat template無し)は素のプロンプトで実行
            inputs = tokenizer(messages[0]["content"],
                               return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        raw_outputs[doc_id] = raw
        ents, ok = parse_json_entities(raw)
        if not ok:
            print(f"[WARN] {doc_id}: JSONパース失敗(raw_outputs.jsonを確認)")
        for e in ents:
            text_val = str(e["text"]).strip()
            if text_val in ("", "なし", "特になし", "N/A", "null", "None"):
                continue
            entities.append({"doc": doc_id, "label": str(e["label"]),
                             "text": text_val})
        print(f"{doc_id}: {len(ents)} エンティティ")

    with open(os.path.join(args.output_dir, "entities.json"), "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "raw_outputs.json"), "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, ensure_ascii=False, indent=2)
    try:
        import pandas as pd
        pd.DataFrame(entities).to_csv(os.path.join(args.output_dir, "entities.csv"),
                                      index=False)
    except ImportError:
        pass
    print(f"抽出エンティティ数: {len(entities)} → {args.output_dir}/entities.json")


if __name__ == "__main__":
    main()
