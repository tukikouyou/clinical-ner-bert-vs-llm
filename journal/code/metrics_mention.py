# -*- coding: utf-8 -*-
"""提及レベル (重複を潰さない) 採点 → stats/mention_level.json
通常の採点は文内の (型, 表層) を 1 個に集約する (テスト 15,119 提及 → 12,919 組; 14.6% が折り畳まれる)。
ここでは gold を出現ごとに数え、予測も出力中の出現回数ごとに数えて同じ貪欲 1:1 対応で relaxed / strict を出す。
位置情報は使わない: 同じ表層が gold に n 回、予測に m 回あれば min(n, m) 回まで対応しうる。
LLM: raw.jsonl を FT は v3 (+v1 fallback)、ZS は v1 で読み直し (preds.json と同じ規則、重複除去だけ外す)。BERT: pred_IOB CSV から出現ごと。"""
import os, sys, json, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../../../shared/llm"))
from testset_eval import parse_entities, nfkc, CORPUS, SPLIT  # noqa: E402
from parse_v2 import parse_entities_v3  # noqa: E402
from metrics_v2 import match, prf, doc_of, bootstrap, pct  # noqa: E402
import numpy as np
RUNS = os.path.join(HERE, "../runs"); STATS = os.path.join(HERE, "../stats"); BP = os.path.join(HERE, "../bert_preds")


def gold_mentions():
    test_docs = set(json.load(open(SPLIT))["test_name"]); gold, keys = {}, []
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.json"))):
        stem = os.path.basename(f).replace(".json", "")
        if f"{stem}.json" not in test_docs:
            continue
        for rec in json.load(open(f)):
            ch = rec["chars"]; key = f"{stem}_{rec['sentence_id']}"
            ents = sorted((e["start"], -(e["end"] - e["start"]), e["type"], nfkc("".join(ch[e["start"]:e["end"] + 1]))) for e in rec["entities"] if e["end"] >= e["start"])
            gold[key] = [(t, s) for _, _, t, s in ents]; keys.append(key)
    return gold, keys


def llm_mentions(run):
    fn = (lambda raw: parse_entities_v3(raw, fallback=parse_entities)) if "-ft" in run else parse_entities
    out = {}
    for l in open(os.path.join(RUNS, run, "raw.jsonl"), encoding="utf-8"):
        r = json.loads(l); ents = fn(r["raw"]) or []
        lst = []
        for e in ents:
            if not isinstance(e, dict) or "label" not in e or "text" not in e:
                continue
            t = (str(e["label"]), nfkc(str(e["text"])).strip())
            if t[1] in ("", "なし"):
                continue
            lst.append(t)
        out[r["key"]] = lst
    return out


def bert_mentions(name):
    cands = [os.path.join(BP, f"{name}_pred_IOB_full.csv"), os.path.join(BP, f"{name}_pred_IOB_fold0_full.csv")]
    df = pd.read_csv(next(c for c in cands if os.path.exists(c))); out = {}
    for uno, g in df.groupby("unique_no", sort=False):
        lst, cur, buf = [], None, ""
        for w, lab in zip(g["word"].astype(str), g["pred_IOB"]):
            w = w[2:] if w.startswith("##") else w
            if lab.startswith("B-"):
                if cur: lst.append((cur, nfkc(buf)))
                cur, buf = lab[2:], w
            elif lab.startswith("I-") and cur == lab[2:]:
                buf += w
            else:
                if cur: lst.append((cur, nfkc(buf)))
                cur, buf = None, ""
        if cur: lst.append((cur, nfkc(buf)))
        out[str(uno)] = lst
    return out


def score(preds, gold, keys, mode):
    tp = fp = fn = 0; per_doc = {}
    for k in keys:
        a, b, c, _, _ = match(preds.get(k, []), gold[k], mode)
        tp += a; fp += b; fn += c; d = per_doc.setdefault(doc_of(k), [0, 0, 0]); d[0] += a; d[1] += b; d[2] += c
    P, R, F = prf(tp, fp, fn)
    return {"P": P, "R": R, "F1": F, "tp": tp, "fp": fp, "fn": fn}, per_doc


if __name__ == "__main__":
    gold, keys = gold_mentions(); n_gold = sum(len(v) for v in gold.values())
    docs = sorted({doc_of(k) for k in keys}); rng = np.random.default_rng(0); idx_sets = [rng.integers(0, len(docs), len(docs)) for _ in range(2000)]
    out = {"n_gold_mentions": n_gold, "n_gold_pairs_dedup": sum(len(set(v)) for v in gold.values()), "runs": {}}
    names = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(RUNS, "*"))) if os.path.exists(os.path.join(d, "raw.jsonl"))]
    berts = ["UTH-v2-s42", "UTH-v2-s1", "UTH-v2-s2", "NICT-v2-s42", "NICT-v2-s1", "NICT-v2-s2", "UTH", "NICT"]
    for n in names + berts:
        try:
            preds = bert_mentions(n) if n in berts else llm_mentions(n)
        except Exception as e:
            print("skip", n, e); continue
        rec = {"n_pred_mentions": sum(len(v) for v in preds.values())}
        for mode in ("relaxed", "strict"):
            s, per_doc = score(preds, gold, keys, mode); rec[mode] = s
            if n.endswith("-ft-4k") or n.endswith("-2k6") or n in ("UTH-v2-s42", "NICT-v2-s42") or n.endswith("-ft"):
                fs = bootstrap(per_doc, docs, idx_sets); rec[mode + "_ci95"] = [pct(fs, 0.025), pct(fs, 0.975)]
        out["runs"]["BERT-" + n if n in berts else n] = rec
        print(f"{n:34s} mention relaxed F1={rec['relaxed']['F1']:.4f} P={rec['relaxed']['P']:.3f} R={rec['relaxed']['R']:.3f} | strict F1={rec['strict']['F1']:.4f} | n_pred={rec['n_pred_mentions']}")
    json.dump(out, open(os.path.join(STATS, "mention_level.json"), "w"), ensure_ascii=False, indent=1)
