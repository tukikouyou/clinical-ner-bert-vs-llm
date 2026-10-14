# -*- coding: utf-8 -*-
"""論文用の追加統計 → stats/extra.json
  1. 截断分析: FT 各モデル(seed42) で finish=length の文と stop の文に分けた P/R/F1 (relaxed)、截断文に含まれる正解数
  2. Med42-70B-ft のラベル別 (relaxed/strict P,R,F1) と UTH-BERT の対応値、頻度–F1 の Spearman
  3. -patient/-others 統合で増える TP 数 (= 主体取り違えの件数の近似)
  4. ZS: 予測数・正解数 (網羅性)
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from metrics_v2 import load_gold_ordered, bert_pred_ordered, evaluate, match, prf, STATS, RUNS  # noqa
from scipy.stats import spearmanr

gold, keys = load_gold_ordered()
n_gold = sum(len(v) for v in gold.values())
out = {"n_gold": n_gold, "n_sent": len(keys)}

def load_preds(run, v2=False):
    p = json.load(open(os.path.join(RUNS, run, "preds_v2.json" if v2 else "preds.json")))
    return {k: [tuple(x) for x in v] for k, v in p.items()}

def finish_map(run):
    return {json.loads(l)["key"]: json.loads(l)["finish"] for l in open(os.path.join(RUNS, run, "raw.jsonl"), encoding="utf-8")}

# 1. 截断分析
FT = {"Med42-70B": "med42-70b-ft", "Llama-3.3-70B": "llama-3.3-70b-ft", "MedGemma-27B": "medgemma-27b-ft", "Gemma-3-27B": "gemma-3-27b-s42-ft",
      "SIP-jmed-13B": "sip-jmed-13b-ft", "LLM-jp-3.1-13B": "llmjp-3.1-13b-ft", "Qwen3-30B-A3B": "qwen-ft", "LLM-jp-4-32B": "llm-jp-4-32b-ft", "LLM-jp-1.8B": "llm-jp-1.8b-ft"}
trunc = {}
for disp, run in FT.items():
    preds, fin = load_preds(run), finish_map(run)
    kt = [k for k in keys if fin[k] == "length"]; ks = [k for k in keys if fin[k] != "length"]
    et, es, ea = evaluate(preds, gold, kt, "relaxed"), evaluate(preds, gold, ks, "relaxed"), evaluate(preds, gold, keys, "relaxed")
    gt = sum(len(gold[k]) for k in kt)
    trunc[disp] = {"n_trunc_sent": len(kt), "gold_in_trunc": gt, "gold_share_in_trunc": round(gt / n_gold, 3),
                   "trunc": {m: et[m] for m in ("P", "R", "F1", "tp", "fp", "fn")}, "rest": {m: es[m] for m in ("P", "R", "F1", "tp", "fp", "fn")},
                   "all": {m: ea[m] for m in ("P", "R", "F1")},
                   "fn_share_from_trunc": round(et["fn"] / (et["fn"] + es["fn"]), 3),
                   "mean_len_trunc": round(sum(len(gold[k]) for k in kt) / max(1, len(kt)), 1), "mean_len_rest": round(sum(len(gold[k]) for k in ks) / max(1, len(ks)), 1)}
out["truncation"] = trunc
# UTH-BERT: same sentence split as Med42 (for reference)
bert = bert_pred_ordered("UTH"); fin = finish_map("med42-70b-ft")
kt = [k for k in keys if fin[k] == "length"]; ks = [k for k in keys if fin[k] != "length"]
out["truncation"]["UTH-BERT (Med42 の截断文で分割)"] = {"trunc": {m: evaluate(bert, gold, kt, "relaxed")[m] for m in ("P", "R", "F1")}, "rest": {m: evaluate(bert, gold, ks, "relaxed")[m] for m in ("P", "R", "F1")}}

# 2. ラベル別 (Med42 seed42) + UTH + 頻度相関
m42 = load_preds("med42-70b-ft")
er, es_ = evaluate(m42, gold, keys, "relaxed"), evaluate(m42, gold, keys, "strict")
ur, us = evaluate(bert, gold, keys, "relaxed"), evaluate(bert, gold, keys, "strict")
gold_n = {}
for k in keys:
    for t, _ in gold[k]:
        gold_n[t] = gold_n.get(t, 0) + 1
per = {}
for t, n in gold_n.items():
    per[t] = {"n": n, "relaxed": prf(*er["per_label"].get(t, [0, 0, n])), "strict": prf(*es_["per_label"].get(t, [0, 0, n])),
              "uth_relaxed": prf(*ur["per_label"].get(t, [0, 0, n])), "uth_strict": prf(*us["per_label"].get(t, [0, 0, n]))}
out["per_label_med42"] = per
xs = [(n, per[t]["relaxed"][2]) for t, n in gold_n.items()]
rho, pv = spearmanr([x for x, _ in xs], [y for _, y in xs])
out["spearman_freq_f1_med42"] = {"rho": round(rho, 3), "p": float(f"{pv:.2e}"), "n_labels": len(xs)}
others = [(t, n) for t, n in gold_n.items() if t.endswith("-others")]
out["others_labels"] = {"n_types": len(others), "n_entities": sum(n for _, n in others), "share": round(sum(n for _, n in others) / n_gold, 3),
                        "mean_f1_med42": round(sum(per[t]["relaxed"][2] for t, _ in others) / len(others), 3)}

# 3. 統合で増える TP (全 FT モデル + BERT)
conf = {}
for disp, run in FT.items():
    p = load_preds(run); a, b = evaluate(p, gold, keys, "relaxed"), evaluate(p, gold, keys, "relaxed", merge=True)
    conf[disp] = {"tp_gain": b["tp"] - a["tp"], "F1": a["F1"], "F1_merged": b["F1"]}
for b_ in ("UTH", "NICT"):
    p = bert_pred_ordered(b_); a, b = evaluate(p, gold, keys, "relaxed"), evaluate(p, gold, keys, "relaxed", merge=True)
    conf[f"{b_}-BERT"] = {"tp_gain": b["tp"] - a["tp"], "F1": a["F1"], "F1_merged": b["F1"]}
out["merge_gain"] = conf

# 4. ZS の予測数
zs = {}
for disp, run in {"MedGemma-27B": "medgemma-27b-zs", "Llama-3.3-70B": "llama-3.3-70b-zs", "Qwen3-30B-A3B": "qwen-zs", "Med42-70B": "med42-70b-zs", "LLM-jp-4-32B": "llm-jp-4-32b-zs", "LLM-jp-1.8B": "llm-jp-1.8b-zs", "SIP-jmed-13B": "sip-jmed-13b-zs"}.items():
    p = load_preds(run, v2=True); zs[disp] = {"n_pred": sum(len(v) for v in p.values()), "per_sent": round(sum(len(v) for v in p.values()) / len(keys), 1)}
zs["gold"] = {"n_pred": n_gold, "per_sent": round(n_gold / len(keys), 1)}
for disp, run in FT.items():
    p = load_preds(run); zs[disp + " (FT)"] = {"n_pred": sum(len(v) for v in p.values()), "per_sent": round(sum(len(v) for v in p.values()) / len(keys), 1)}
out["n_pred"] = zs
json.dump(out, open(os.path.join(STATS, "extra.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("n_gold", "spearman_freq_f1_med42", "others_labels", "merge_gain")}, ensure_ascii=False, indent=1))
for d, v in trunc.items():
    if "n_trunc_sent" not in v: continue
    print(f"{d:16s} trunc {v['n_trunc_sent']:3d} sent, gold share {v['gold_share_in_trunc']:.2f}, R trunc {v['trunc']['R']:.3f} rest {v['rest']['R']:.3f}, FN share from trunc {v['fn_share_from_trunc']:.2f}, mean gold/sent trunc {v['mean_len_trunc']} rest {v['mean_len_rest']}")
print(out["truncation"]["UTH-BERT (Med42 の截断文で分割)"])
print({k: v["per_sent"] for k, v in zs.items()})
