# -*- coding: utf-8 -*-
"""
期刊版 指標計算 (推論不要)。runs/<name>/preds.json または BERT の pred CSV から
  relaxed(部分一致) / strict(完全一致) / min2 / min3 / -others統合 / マクロF1 /
  文書単位ブートストラップ 95%CI / 部分一致の内訳 / ラベル別
を計算し stats/<name>.json に保存する。--pair で対応ありブートストラップの差も出す。

  python3 metrics_v2.py --runs med42-70b-ft llama-3.3-70b-ft ... [--bert UTH NICT] [--pair]
"""
import os, sys, json, argparse, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../shared/llm"))
from testset_eval import load_alldocs, load_labels, nfkc, ROOT, CORPUS, SPLIT  # noqa: E402
import glob
import pandas as pd


def load_gold_ordered():
    """gold を出現位置順(start, 長い順)のリストで返す。同一(型,表層)は1個に集約(testset_eval と同じ集合定義)。"""
    test_docs = set(json.load(open(SPLIT))["test_name"])
    gold, keys = {}, []
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.json"))):
        stem = os.path.basename(f).replace(".json", "")
        if f"{stem}.json" not in test_docs:
            continue
        for rec in json.load(open(f)):
            chars = rec["chars"]; key = f"{stem}_{rec['sentence_id']}"
            ents = sorted((e["start"], -(e["end"] - e["start"]), e["type"],
                           nfkc("".join(chars[e["start"]:e["end"] + 1])))
                          for e in rec["entities"] if e["end"] >= e["start"])
            seen, lst = set(), []
            for _, _, t, s in ents:
                if (t, s) in seen:
                    continue
                seen.add((t, s)); lst.append((t, s))
            gold[key] = lst; keys.append(key)
    return gold, keys


def bert_pred_ordered(bert_type):
    """BERT の pred_IOB CSV から文内出現順の (型,表層) リスト(重複は最初のみ)。"""
    df = pd.read_csv(os.path.join(ROOT, f"code/results/NER/{bert_type}/NER/0.csv"))
    out = {}
    for uno, g in df.groupby("unique_no", sort=False):
        lst, seen, cur, buf = [], set(), None, ""
        def flush():
            if cur:
                t = (cur, nfkc(buf))
                if t not in seen:
                    seen.add(t); lst.append(t)
        for w, lab in zip(g["word"].astype(str), g["pred_IOB"]):
            w = w[2:] if w.startswith("##") else w
            if lab.startswith("B-"):
                flush(); cur, buf = lab[2:], w
            elif lab.startswith("I-") and cur == lab[2:]:
                buf += w
            else:
                flush(); cur, buf = None, ""
        flush()
        out[uno] = lst
    return out

RUNS = os.path.join(HERE, "../runs")
STATS = os.path.join(HERE, "../stats")
LABELS = set(load_labels())


def ok(pt, ps, gt, gs, mode):
    if pt != gt:
        return False
    if mode == "strict":
        return ps == gs
    if ps == gs:
        return True
    sub = gs in ps or ps in gs
    if not sub:
        return False
    if mode == "relaxed":
        return True
    k = int(mode[3:])            # min2 / min3: 短い方が k 文字未満なら完全一致のみ
    return min(len(gs), len(ps)) >= k


def merge_lab(t):
    """-patient / -others の区別を統合 (両方の接尾辞を落とす; 2026-10-14 修正: 以前は -others しか落とさず統合になっていなかった)"""
    return t[:-7] if t.endswith("-others") else (t[:-8] if t.endswith("-patient") else t)


def match(preds, golds, mode, merge=False):
    """貪欲 1対1 対応 (testset_eval._match_sentence と同じ走査順)。
    戻り値: tp, fp, fn, per-label dict(label→[tp,fp,fn]), 内訳 Counter"""
    f = merge_lab if merge else (lambda t: t)
    golds = [(f(t), s) for t, s in golds]
    preds = [(f(t), s) for t, s in preds]
    used = [False] * len(golds)
    per = defaultdict(lambda: [0, 0, 0])
    kind = defaultdict(int)
    tp = 0
    for pt, ps in preds:
        hit = False
        # 予測は列挙順、正解は出現順に走査。完全一致する未対応の正解があればそれを優先し、
        # 無ければ条件を満たす最初の未対応の正解と対応づける(順序依存を減らす決定的規則)。
        cand = None
        for j, (gt, gs) in enumerate(golds):
            if used[j] or not ok(pt, ps, gt, gs, mode):
                continue
            if ps == gs:
                cand = j; break
            if cand is None:
                cand = j
        if cand is not None:
            j = cand; gt, gs = golds[j]
            used[j] = True; tp += 1; hit = True; per[gt][0] += 1
            kind["exact" if ps == gs else ("pred_in_gold" if ps in gs else "gold_in_pred")] += 1
        if not hit:
            per[pt][1] += 1
    for j, (gt, gs) in enumerate(golds):
        if not used[j]:
            per[gt][2] += 1
    return tp, len(preds) - tp, len(golds) - tp, per, kind


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return round(P, 4), round(R, 4), round(F, 4)


def doc_of(k):
    return k.rpartition("_")[0]


def evaluate(preds, gold, keys, mode, merge=False):
    tp = fp = fn = 0
    per_doc = defaultdict(lambda: [0, 0, 0])
    per_lab = defaultdict(lambda: [0, 0, 0])
    kinds = defaultdict(int)
    for k in keys:
        a, b, c, per, kind = match(preds.get(k, []), gold[k], mode, merge)
        tp += a; fp += b; fn += c
        d = per_doc[doc_of(k)]; d[0] += a; d[1] += b; d[2] += c
        for lab, v in per.items():
            for i in range(3):
                per_lab[lab][i] += v[i]
        for kk, v in kind.items():
            kinds[kk] += v
    P, R, F = prf(tp, fp, fn)
    return {"P": P, "R": R, "F1": F, "tp": tp, "fp": fp, "fn": fn,
            "per_doc": dict(per_doc), "per_label": dict(per_lab), "kinds": dict(kinds)}


def bootstrap(per_doc, docs, idx_sets):
    fs = []
    for idx in idx_sets:
        tp = fp = fn = 0
        for i in idx:
            d = per_doc.get(docs[i], [0, 0, 0]); tp += d[0]; fp += d[1]; fn += d[2]
        fs.append(prf(tp, fp, fn)[2])
    return fs


def pct(xs, q):
    xs = sorted(xs); i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--bert", nargs="*", default=[])
    p.add_argument("--pair", action="store_true")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--macro_min", type=int, default=20)
    p.add_argument("--parser", default="v1", choices=["v1", "v2"], help="v2: preds_v2.json (最後の JSON 配列を採用)")
    args = p.parse_args()
    os.makedirs(STATS, exist_ok=True)

    gold, keys = load_gold_ordered()
    docs = sorted({doc_of(k) for k in keys})
    gold_lab = defaultdict(int)
    for k in keys:
        for t, s in gold[k]:
            gold_lab[t] += 1
    macro_labs = [t for t, n in gold_lab.items() if n >= args.macro_min]
    rng = random.Random(12345)
    idx_sets = [[rng.randrange(len(docs)) for _ in range(len(docs))] for _ in range(args.n_boot)]

    systems = {}
    for b in args.bert:
        systems[f"BERT-{b}"] = bert_pred_ordered(b)
    for r in args.runs:
        pr = json.load(open(os.path.join(RUNS, r, "preds.json" if args.parser == "v1" else "preds_v2.json")))
        systems[r + ("" if args.parser == "v1" else "@v2")] = {k: [tuple(x) for x in v] for k, v in pr.items()}

    boot_cache = {}
    summary = {}
    for name, preds in systems.items():
        res = {}
        for mode in ("relaxed", "strict", "min2", "min3"):
            res[mode] = evaluate(preds, gold, keys, mode)
        res["relaxed_merged"] = evaluate(preds, gold, keys, "relaxed", merge=True)
        res["strict_merged"] = evaluate(preds, gold, keys, "strict", merge=True)
        rl = res["relaxed"]
        macro = [prf(*rl["per_label"][t])[2] if t in rl["per_label"] else 0.0 for t in macro_labs]
        unk = sum(1 for k in keys for t, s in preds.get(k, []) if t not in LABELS)
        fs = bootstrap(rl["per_doc"], docs, idx_sets)
        boot_cache[name] = fs
        fs_s = bootstrap(res["strict"]["per_doc"], docs, idx_sets)
        out = {"name": name,
               "relaxed": {k: rl[k] for k in ("P", "R", "F1", "tp", "fp", "fn")},
               "relaxed_ci95": [pct(fs, .025), pct(fs, .975)],
               "strict": {k: res["strict"][k] for k in ("P", "R", "F1", "tp", "fp", "fn")},
               "strict_ci95": [pct(fs_s, .025), pct(fs_s, .975)],
               "min2_F1": res["min2"]["F1"], "min3_F1": res["min3"]["F1"],
               "relaxed_merged_F1": res["relaxed_merged"]["F1"],
               "strict_merged_F1": res["strict_merged"]["F1"],
               "macro_F1": round(sum(macro) / len(macro), 4), "n_macro_labels": len(macro_labs),
               "kinds": rl["kinds"], "exact_share_of_tp": round(rl["kinds"].get("exact", 0) / max(rl["tp"], 1), 4),
               "n_pred": sum(len(v) for v in preds.values()), "n_unknown_label_pred": unk,
               "per_label_relaxed": {t: {"n": gold_lab.get(t, 0), "P": prf(*v)[0], "R": prf(*v)[1], "F1": prf(*v)[2]}
                                     for t, v in sorted(rl["per_label"].items())},
               "per_label_strict": {t: {"F1": prf(*v)[2]} for t, v in sorted(res["strict"]["per_label"].items())},
               "per_doc_relaxed": rl["per_doc"]}
        json.dump(out, open(os.path.join(STATS, f"{name}.json"), "w"), ensure_ascii=False, indent=1)
        summary[name] = out
        print(f"{name:22s} relaxed F1={rl['F1']:.4f} [{out['relaxed_ci95'][0]:.3f},{out['relaxed_ci95'][1]:.3f}] "
              f"P={rl['P']:.3f} R={rl['R']:.3f} | strict F1={res['strict']['F1']:.4f} "
              f"min2={res['min2']['F1']:.4f} min3={res['min3']['F1']:.4f} | merged={res['relaxed_merged']['F1']:.4f} "
              f"macro={out['macro_F1']:.4f} | exact/TP={out['exact_share_of_tp']:.3f} unk={unk}")

    if args.pair and len(systems) > 1:
        names = list(systems)
        pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                diffs = [x - y for x, y in zip(boot_cache[a], boot_cache[b])]
                d0 = round(summary[a]["relaxed"]["F1"] - summary[b]["relaxed"]["F1"], 4)
                pairs[f"{a} - {b}"] = {"diff": d0, "ci95": [pct(diffs, .025), pct(diffs, .975)]}
                print(f"  {a} - {b}: {d0:+.4f} [{pct(diffs,.025):+.4f}, {pct(diffs,.975):+.4f}]")
        json.dump(pairs, open(os.path.join(STATS, "pairwise_" + "_".join(names)[:80] + ".json"), "w"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
