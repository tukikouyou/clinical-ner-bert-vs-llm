# -*- coding: utf-8 -*-
"""FT run ごとの出力完全性: finish=stop / 原文そのまま JSON として読める / 打ち切り後に修復 / 空回答 / parse 失敗 / 配列後に続きを生成 → stats/raw_completeness.json"""
import os, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "../runs"); STATS = os.path.join(HERE, "../stats")
_dec = json.JSONDecoder()
out = {}
for r in sorted(os.listdir(RUNS)):
    raw_p = os.path.join(RUNS, r, "raw.jsonl")
    if "-ft" not in r or not os.path.exists(raw_p):
        continue
    rows = [json.loads(l) for l in open(raw_p, encoding="utf-8")]
    preds = json.load(open(os.path.join(RUNS, r, "preds.json")))
    c = {"n": len(rows), "stop": 0, "clean_json": 0, "repaired_after_cutoff": 0, "empty_answer": 0, "parse_fail": 0, "continued_after_array": 0}
    for x in rows:
        raw = x["raw"].strip(); raw2 = re.sub(r"<\|channel\|>\s*final\s*<\|message\|>", "", raw).split("<|")[0].strip() if "<|channel|>" in raw else raw
        if x["finish"] == "stop": c["stop"] += 1
        try:
            o = json.loads(raw2); ok = isinstance(o, list)
        except Exception:
            ok = False
        if ok: c["clean_json"] += 1
        else:
            i = raw2.find("[")
            if i >= 0:
                try:
                    o, end = _dec.raw_decode(raw2[i:])
                    if raw2[i + end:].strip(): c["continued_after_array"] += 1
                except Exception:
                    if x["finish"] == "length": c["repaired_after_cutoff"] += 1
        p = preds.get(x["key"])
        if p is None: c["parse_fail"] += 1
        elif len(p) == 0: c["empty_answer"] += 1
    out[r] = c
json.dump(out, open(os.path.join(STATS, "raw_completeness.json"), "w"), indent=1)
for r, c in out.items():
    if r.endswith("-ft-4k") or r.endswith("-ft"):
        print(f"{r:24s}", c)
