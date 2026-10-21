# -*- coding: utf-8 -*-
"""stats/<name>.json と runs/<name>/meta.json から論文用の表(Markdown + LaTeX)を生成 → stats/tables.md, stats/tables.tex
主条件 = 生成予算 4096 (runs/*-ft-4k; LLM-jp-3.1 系は文脈 4096 のため 2600)、LAU 設定 = 900 (runs/*-ft)。"""
import os, json, glob, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "../stats"); RUNS = os.path.join(HERE, "../runs")

# (表示名, 種別, size, ft-run(seed42, 予算900), zs-run, zs-guided-run) ; 4k は ft-run + "-4k"、seed k は ft-run の "-ft" を "-s{k}-ft" に
MODELS = [
    ("Med42-70B", "医療", "70B", "med42-70b-ft", "med42-70b-zs", "med42-70b-zs-guided"),
    ("Llama-3.3-70B", "汎用", "70B", "llama-3.3-70b-ft", "llama-3.3-70b-zs", "llama-3.3-70b-zs-guided"),
    ("MedGemma-27B", "医療", "27B", "medgemma-27b-ft", "medgemma-27b-zs", "medgemma-27b-zs-guided"),
    ("Gemma-3-27B (MedGemma base)", "汎用", "27B", "gemma-3-27b-s42-ft", None, None),
    ("SIP-jmed-13B", "医療", "13B", "sip-jmed-13b-ft", "sip-jmed-13b-zs", "sip-jmed-13b-zs-guided"),
    ("LLM-jp-3.1-13B (SIP base)", "汎用", "13B", "llmjp-3.1-13b-ft", None, None),
    ("Qwen3-30B-A3B", "汎用", "30B", "qwen-ft", "qwen-zs", "qwen-zs-guided"),
    ("LLM-jp-4-32B", "汎用", "32B", "llm-jp-4-32b-ft", "llm-jp-4-32b-zs", "llm-jp-4-32b-zs-guided"),
    ("LLM-jp-1.8B", "汎用", "1.8B", "llm-jp-1.8b-ft", "llm-jp-1.8b-zs", "llm-jp-1.8b-zs-guided"),
]
BERTS = [("UTH-BERT", "医療", "BERT-UTH"), ("NICT-BERT", "汎用", "BERT-NICT")]


def seed_runs(ft, budget):
    """seed 42/1/2 の run 名。budget: '' (900) or '-4k'"""
    base = ft.replace("-s42-ft", "-ft")
    stem = base[:-3]  # strip '-ft'
    first = ft if budget == "" else base + budget   # gemma の seed42/900 だけ '-s42-ft' という名前
    return [first, f"{stem}-s1-ft{budget}", f"{stem}-s2-ft{budget}"]


def S(name):
    f = os.path.join(STATS, f"{name}.json")
    return json.load(open(f)) if name and os.path.exists(f) else None


def M(name):
    f = os.path.join(RUNS, name, "meta.json") if name else None
    return json.load(open(f)) if f and os.path.exists(f) else {}


def ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


def pairdiff(a, b):
    for f in glob.glob(os.path.join(STATS, "pairwise_*.json")):
        d = json.load(open(f))
        if f"{a} - {b}" in d:
            v = d[f"{a} - {b}"]; return f"{v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
        if f"{b} - {a}" in d:
            v = d[f"{b} - {a}"]; return f"{-v['diff']:+.3f} [{-v['ci95'][1]:+.3f}, {-v['ci95'][0]:+.3f}]"
    return "-"


md, tex = [], []


def ft_table(budget, title):
    md.append(f"\n## {title}\n")
    md.append("| モデル | 種別 | 予算 | relaxed F1 | 95% CI | P | R | strict F1 | 95% CI | min2 | min3 | Macro | -others統合 | exact/TP | 截断/433 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tex.append(f"% FT table {title}\n\\begin{{tabular}}{{llcccccccccc}}\\toprule\nModel & Type & Budget & Relaxed F1 & 95\\% CI & P & R & Strict F1 & 95\\% CI & Macro & Merged & Exact/TP & Trunc.\\\\\\midrule")
    for disp, typ, size, ft, zs, zg in MODELS:
        name = seed_runs(ft, budget)[0]; s = S(name)
        if not s:
            continue
        m = M(name); r, t = s["relaxed"], s["strict"]; tr = m.get("truncated", "-"); bud = m.get("max_new_tokens", "-")
        md.append(f"| {disp} | {typ} | {bud} | {r['F1']:.3f} | {ci(s['relaxed_ci95'])} | {r['P']:.3f} | {r['R']:.3f} | {t['F1']:.3f} | {ci(s['strict_ci95'])} | {s['min2_F1']:.3f} | {s['min3_F1']:.3f} | {s['macro_F1']:.3f} | {s['relaxed_merged_F1']:.3f} | {s['exact_share_of_tp']:.3f} | {tr} |")
        tex.append(f"{disp} & {typ} & {bud} & {r['F1']:.3f} & {ci(s['relaxed_ci95'])} & {r['P']:.3f} & {r['R']:.3f} & {t['F1']:.3f} & {ci(s['strict_ci95'])} & {s['macro_F1']:.3f} & {s['relaxed_merged_F1']:.3f} & {s['exact_share_of_tp']:.3f} & {tr}\\\\")
    for disp, typ, b in BERTS:
        s = S(b); r, t = s["relaxed"], s["strict"]
        md.append(f"| {disp} | {typ} | – | {r['F1']:.3f} | {ci(s['relaxed_ci95'])} | {r['P']:.3f} | {r['R']:.3f} | {t['F1']:.3f} | {ci(s['strict_ci95'])} | {s['min2_F1']:.3f} | {s['min3_F1']:.3f} | {s['macro_F1']:.3f} | {s['relaxed_merged_F1']:.3f} | {s['exact_share_of_tp']:.3f} | – |")
        tex.append(f"{disp} & {typ} & -- & {r['F1']:.3f} & {ci(s['relaxed_ci95'])} & {r['P']:.3f} & {r['R']:.3f} & {t['F1']:.3f} & {ci(s['strict_ci95'])} & {s['macro_F1']:.3f} & {s['relaxed_merged_F1']:.3f} & {s['exact_share_of_tp']:.3f} & --\\\\")
    tex.append("\\bottomrule\\end{tabular}\n")


def seed_table(budget, title):
    md.append(f"\n## {title}\n")
    md.append("| モデル | 種別 | relaxed s42 | s1 | s2 | 平均 ± SD | strict s42 | s1 | s2 | 平均 ± SD | macro 平均 | 截断 平均 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    tex.append(f"% seed table {title}\n\\begin{{tabular}}{{llccccccccc}}\\toprule\nModel & Type & Relaxed s42 & s1 & s2 & Mean $\\pm$ SD & Strict s42 & s1 & s2 & Mean $\\pm$ SD & Macro\\\\\\midrule")
    for disp, typ, size, ft, zs, zg in MODELS:
        runs = seed_runs(ft, budget); ss = [S(r) for r in runs]
        if not any(ss):
            continue
        r = [x["relaxed"]["F1"] if x else None for x in ss]; t = [x["strict"]["F1"] if x else None for x in ss]
        rv = [v for v in r if v is not None]; tv = [v for v in t if v is not None]
        mc = [x["macro_F1"] for x in ss if x]; trn = [M(n).get("truncated") for n, x in zip(runs, ss) if x]
        f3 = lambda v: "-" if v is None else f"{v:.3f}"
        msd = lambda v: f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) >= 2 else (f"{v[0]:.3f} (1 seed)" if v else "-")
        row = [disp, typ] + [f3(v) for v in r] + [msd(rv)] + [f3(v) for v in t] + [msd(tv), f"{st.mean(mc):.3f}", f"{st.mean([x for x in trn if x is not None]):.0f}"]
        md.append("| " + " | ".join(row) + " |")
        tex.append(" & ".join(x.replace("±", "$\\pm$") for x in row[:-1]) + "\\\\")
    tex.append("\\bottomrule\\end{tabular}\n")


ft_table("-4k", "FT 後・主条件 (生成予算 4096; LLM-jp-3.1 系は 2600) seed 42 (test36; 95% CI = 文書ブートストラップ 2000 回)")
ft_table("", "FT 後・LAU 設定 (生成予算 900) seed 42")
seed_table("-4k", "QLoRA 多 seed (42 / 1 / 2)・主条件 (予算 4096)")
seed_table("", "QLoRA 多 seed (42 / 1 / 2)・LAU 設定 (予算 900)")

# ---- 表: 予算アブレーション (900 vs 4k) ----
md.append("\n## 生成予算アブレーション: 900 → 4096 (seed 42; 3 seed 平均は両方そろう場合)\n")
md.append("| モデル | 予算 | relaxed 900 → 大 | Δ | strict 900 → 大 | R 900 → 大 | P 900 → 大 | 截断 900 → 大 | 3seed relaxed 900 → 大 |")
md.append("|---|---|---|---|---|---|---|---|---|")
for disp, typ, size, ft, zs, zg in MODELS:
    a, b = S(ft), S(seed_runs(ft, "-4k")[0])
    if not (a and b):
        continue
    ma, mb = M(ft), M(seed_runs(ft, "-4k")[0])
    s9 = [S(r) for r in seed_runs(ft, "")]; s4 = [S(r) for r in seed_runs(ft, "-4k")]
    m9 = st.mean([x["relaxed"]["F1"] for x in s9]) if all(s9) else None; m4 = st.mean([x["relaxed"]["F1"] for x in s4]) if all(s4) else None
    md.append(f"| {disp} | {mb.get('max_new_tokens')} | {a['relaxed']['F1']:.3f} → {b['relaxed']['F1']:.3f} | {b['relaxed']['F1']-a['relaxed']['F1']:+.3f} | {a['strict']['F1']:.3f} → {b['strict']['F1']:.3f} | {a['relaxed']['R']:.3f} → {b['relaxed']['R']:.3f} | {a['relaxed']['P']:.3f} → {b['relaxed']['P']:.3f} | {ma.get('truncated')} → {mb.get('truncated')} | " + (f"{m9:.3f} → {m4:.3f}" if m9 and m4 else "-") + " |")

# ---- 表: ゼロショット prompt-only vs constrained ----
md.append("\n## ゼロショット: prompt のみ vs JSON Schema 制約デコード (test36; 予算 900 / 思考型 3072)\n")
md.append("| モデル | 種別 | ZS F1 | ZS strict | parse失敗 | 截断 | 未知ラベル | ZS F1 (parser v2) | parse失敗 v2 | 制約 F1 | 制約 strict | 制約 P | 制約 R | 截断 | FT F1 (4k) |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
tex.append("% ZS table\n\\begin{tabular}{llcccccccccc}\\toprule\nModel & Type & ZS F1 & Strict & Parse fail & Trunc. & Constr. F1 & Strict & P & R & Trunc. & FT F1\\\\\\midrule")
for disp, typ, size, ft, zs, zg in MODELS:
    a, b, c = S(zs), S(zg), S(seed_runs(ft, "-4k")[0])
    if not (a or b):
        continue
    ma, mb = M(zs), M(zg)
    a2 = S(zs + "@v2") if zs else None
    row = [disp, typ,
           f"{a['relaxed']['F1']:.3f}" if a else "-", f"{a['strict']['F1']:.3f}" if a else "-",
           str(ma.get("parse_fail", "-")), str(ma.get("truncated", "-")), str(a["n_unknown_label_pred"]) if a else "-",
           f"{a2['relaxed']['F1']:.3f}" if a2 else "-", str(ma.get("parse_fail_v2", "-")),
           f"{b['relaxed']['F1']:.3f}" if b else "-", f"{b['strict']['F1']:.3f}" if b else "-",
           f"{b['relaxed']['P']:.3f}" if b else "-", f"{b['relaxed']['R']:.3f}" if b else "-", str(mb.get("truncated", "-")),
           f"{c['relaxed']['F1']:.3f}" if c else "-"]
    md.append("| " + " | ".join(row) + " |")
    tex.append(" & ".join(row[:6] + row[9:]) + "\\\\")
tex.append("\\bottomrule\\end{tabular}\n")

# ---- 表: 思考型の ZS 4 条件 + 参考モデル ----
md.append("\n## 思考型モデルの ZS 条件 (prompt-only 旧パーサ / 新パーサ / 整段制約 / 2 段階 自由思考+制約回答)\n")
md.append("| モデル | prompt-only (v1) | parse失敗 | prompt-only (v2) | parse失敗 v2 | 整段制約 | 2段階 (think→constrain) | strict (2段階) | FT (4k) |")
md.append("|---|---|---|---|---|---|---|---|---|")
for disp, zs, zg, tg, ft in [("LLM-jp-4-32B", "llm-jp-4-32b-zs", "llm-jp-4-32b-zs-guided", "llm-jp-4-32b-zs-think-guided", "llm-jp-4-32b-ft-4k"),
                              ("SIP-jmed-13B", "sip-jmed-13b-zs", "sip-jmed-13b-zs-guided", "sip-jmed-13b-zs-think-guided", "sip-jmed-13b-ft-4k")]:
    a, a2, b, c, d = S(zs), S(zs + "@v2"), S(zg), S(tg), S(ft); m = M(zs)
    md.append(f"| {disp} | {a['relaxed']['F1']:.3f} | {m.get('parse_fail','-')} | {a2['relaxed']['F1']:.3f} | {m.get('parse_fail_v2','-')} | {b['relaxed']['F1']:.3f} | {c['relaxed']['F1']:.3f} | {c['strict']['F1']:.3f} | {d['relaxed']['F1']:.3f} |")
md.append("\n## 参考モデル (ollama, 学習不可; think=true は思考を許可)\n")
md.append("| モデル | 条件 | relaxed F1 | strict F1 | P | R | parse失敗 | 截断 |")
md.append("|---|---|---|---|---|---|---|---|")
for disp, cond, run in [("Qwen3.6-35B", "ZS prompt-only (think=false)", "qwen3.6-35b-zs"), ("Qwen3.6-35B", "think=true + 制約 (3072 tok)", "qwen3.6-35b-zs-think-guided"),
                        ("GPT-OSS-120B", "ZS prompt-only (think=true; ollama で無効化不可)", "gpt-oss-120b-zs"), ("GPT-OSS-120B", "think=true + 制約", "gpt-oss-120b-zs-think-guided")]:
    a = S(run); m = M(run)
    if a:
        md.append(f"| {disp} | {cond} | {a['relaxed']['F1']:.3f} | {a['strict']['F1']:.3f} | {a['relaxed']['P']:.3f} | {a['relaxed']['R']:.3f} | {m.get('parse_fail','-')} | {m.get('truncated','-')} |")

# ---- 表: 基盤対齐 (両予算) ----
PAIRS = [("MedGemma-27B − Gemma-3-27B", "medgemma-27b-ft", "gemma-3-27b-s42-ft"),
         ("SIP-jmed-13B − LLM-jp-3.1-13B", "sip-jmed-13b-ft", "llmjp-3.1-13b-ft"),
         ("Med42-70B − Llama-3.3-70B", "med42-70b-ft", "llama-3.3-70b-ft")]
for budget, title in [("-4k", "主条件 (予算 4096)"), ("", "LAU 設定 (予算 900)")]:
    md.append(f"\n## 基盤を揃えた対比較・{title} (医療 − 基盤, 同 seed 対応ありブートストラップ, relaxed F1)\n")
    md.append("| 対 | seed 42 | seed 1 | seed 2 |")
    md.append("|---|---|---|---|")
    for name, a, b in PAIRS:
        ra, rb = seed_runs(a, budget), seed_runs(b, budget)
        md.append(f"| {name} | " + " | ".join(pairdiff(x, y) for x, y in zip(ra, rb)) + " |")

# ---- 表: BERT との対差 (両予算) ----
for budget, title in [("-4k", "主条件 (予算 4096)"), ("", "LAU 設定 (予算 900)")]:
    md.append(f"\n## FT 後 LLM − BERT・{title} (seed 42, 対応ありブートストラップ, relaxed F1)\n")
    md.append("| LLM | vs UTH-BERT | vs NICT-BERT |")
    md.append("|---|---|---|")
    for disp, typ, size, ft, zs, zg in MODELS:
        md.append(f"| {disp} | {pairdiff(seed_runs(ft, budget)[0], 'BERT-UTH')} | {pairdiff(seed_runs(ft, budget)[0], 'BERT-NICT')} |")

open(os.path.join(STATS, "tables.md"), "w").write("\n".join(md) + "\n")
open(os.path.join(STATS, "tables.tex"), "w").write("\n".join(tex) + "\n")
print("\n".join(md))
