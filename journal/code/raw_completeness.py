# -*- coding: utf-8 -*-
"""FT run ごとの出力完全性 → stats/raw_completeness.json
各文を reparse.py と同じ parser v3 (+LAU v1 fallback) で読み直し、排他的に分類する:
  parse_fail: 配列を取り出せず (preds.json には [] として保存されている → 空回答とは区別する)
  clean_json: 生出力 (LLM-jp-4 は Harmony チャネル記号を除去) がそのまま JSON 配列
  continued_after_array: 完全な配列の後に生成が続いた
  repaired_after_cutoff: 完全な配列が無く、途中で切れた配列を最後の完全な要素で閉じて読めた
  empty_answer: 読めた配列が空 (正当な [] 回答)
stop は終端トークンで終わった文数 (分類とは独立)。saved_mismatch は preds.json との不一致数 (0 であること)。"""
import os, sys, json, re, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "../runs"); STATS = os.path.join(HERE, "../stats")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../../../shared/llm"))
from testset_eval import parse_entities  # noqa: E402
from parse_v2 import parse_entities_v3  # noqa: E402
_dec = json.JSONDecoder()


def strip_channel(raw):
    return re.sub(r"<\|channel\|>\s*final\s*<\|message\|>", "", raw).split("<|")[0].strip() if "<|channel|>" in raw else raw.strip()


def complete_array(raw):
    """最初に完全に読める配列の (start, end)。無ければ None"""
    for m in re.finditer(r"\[", raw):
        try:
            v, end = _dec.raw_decode(raw[m.start():])
        except ValueError:
            continue
        if isinstance(v, list):
            return m.start(), m.start() + end
    return None


def norm(ents):
    seen, out = set(), []
    for e in ents or []:
        k = (str(e["label"]), unicodedata.normalize("NFKC", str(e["text"])).strip())
        if k[1] and k[1] != "なし" and k not in seen:
            seen.add(k); out.append(list(k))
    return out


out = {}
for r in sorted(os.listdir(RUNS)):
    raw_p = os.path.join(RUNS, r, "raw.jsonl")
    if "-ft" not in r or not os.path.exists(raw_p):
        continue
    rows = [json.loads(l) for l in open(raw_p, encoding="utf-8")]
    preds = json.load(open(os.path.join(RUNS, r, "preds.json")))
    c = {"n": len(rows), "stop": 0, "clean_json": 0, "repaired_after_cutoff": 0, "continued_after_array": 0, "empty_answer": 0, "parse_fail": 0, "saved_mismatch": 0, "parse_fail_keys": []}
    for x in rows:
        raw = x["raw"]; raw2 = strip_channel(raw)
        if x["finish"] == "stop": c["stop"] += 1
        ents = parse_entities_v3(raw, fallback=parse_entities)
        if ents is None:
            c["parse_fail"] += 1; c["parse_fail_keys"].append(x["key"])
        else:
            try:
                clean = isinstance(json.loads(raw2), list)
            except Exception:
                clean = False
            ca = complete_array(raw2)
            if clean: c["clean_json"] += 1
            elif ca is not None and raw2[ca[1]:].strip(): c["continued_after_array"] += 1
            elif ca is None: c["repaired_after_cutoff"] += 1
            else: c["clean_json"] += 1   # 配列の前に余分な文字があるが配列は完全 (稀)
            if len(ents) == 0: c["empty_answer"] += 1
        if norm(ents) != preds.get(x["key"]): c["saved_mismatch"] += 1
    out[r] = c
json.dump(out, open(os.path.join(STATS, "raw_completeness.json"), "w"), ensure_ascii=False, indent=1)
print("runs", len(out), "parse_fail total", sum(c["parse_fail"] for c in out.values()), "saved_mismatch total", sum(c["saved_mismatch"] for c in out.values()))
for r, c in out.items():
    if r.endswith("-ft-4k") or r.endswith("-ft") or r.endswith("-2k6"):
        print(f"{r:26s}", {k: v for k, v in c.items() if k != "parse_fail_keys"})
