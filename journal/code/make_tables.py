# -*- coding: utf-8 -*-
"""stats/<name>.json と runs/<name>/meta.json から論文用の表(Markdown + LaTeX)を生成 → stats/tables.md, stats/tables.tex"""
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "../stats"); RUNS = os.path.join(HERE, "../runs")

MODELS = [  # (表示名, 種別, size, ft-run, zs-run, zs-guided-run)
    ("Med42-70B", "医療", "70B", "med42-70b-ft", "med42-70b-zs", "med42-70b-zs-guided"),
    ("Llama-3.3-70B", "汎用", "70B", "llama-3.3-70b-ft", "llama-3.3-70b-zs", "llama-3.3-70b-zs-guided"),
    ("MedGemma-27B", "医療", "27B", "medgemma-27b-ft", "medgemma-27b-zs", "medgemma-27b-zs-guided"),
    ("SIP-jmed-13B", "医療", "13B", "sip-jmed-13b-ft", "sip-jmed-13b-zs", "sip-jmed-13b-zs-guided"),
    ("Qwen3-30B-A3B", "汎用", "30B", "qwen-ft", "qwen-zs", "qwen-zs-guided"),
    ("LLM-jp-4-32B", "汎用", "32B", "llm-jp-4-32b-ft", "llm-jp-4-32b-zs", "llm-jp-4-32b-zs-guided"),
    ("LLM-jp-1.8B", "汎用", "1.8B", "llm-jp-1.8b-ft", "llm-jp-1.8b-zs", "llm-jp-1.8b-zs-guided"),
    ("LLM-jp-3.1-13B (SIP base)", "汎用", "13B", "llmjp-3.1-13b-ft", None, None),
    ("Gemma-3-27B (MedGemma base)", "汎用", "27B", "gemma-3-27b-s42-ft", None, None),
    ("UTH-BERT", "医療", "110M", "BERT-UTH", None, None),
    ("NICT-BERT", "汎用", "110M", "BERT-NICT", None, None),
]


def S(name):
    f = os.path.join(STATS, f"{name}.json")
    return json.load(open(f)) if name and os.path.exists(f) else None


def M(name):
    f = os.path.join(RUNS, name, "meta.json") if name else None
    return json.load(open(f)) if f and os.path.exists(f) else {}


def ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


md, tex = [], []
# ---- 表: ファインチューニング後 ----
md.append("## FT 後 (test36). relaxed / strict, 95% CI = 文書ブートストラップ 2000 回\n")
md.append("| モデル | 種別 | relaxed F1 | 95% CI | P | R | strict F1 | 95% CI | min2 | min3 | Macro | -others統合 | exact/TP | 截断/433 |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
tex.append("% FT table\n\\begin{tabular}{llcccccccccc}\\toprule\nModel & Type & Relaxed F1 & 95\\% CI & P & R & Strict F1 & 95\\% CI & Macro & Merged & Exact/TP & Trunc.\\\\\\midrule")
for disp, typ, size, ft, zs, zg in MODELS:
    s = S(ft)
    if not s:
        continue
    m = M(ft); r, st = s["relaxed"], s["strict"]
    tr = m.get("truncated", "-")
    md.append(f"| {disp} | {typ} | {r['F1']:.3f} | {ci(s['relaxed_ci95'])} | {r['P']:.3f} | {r['R']:.3f} | {st['F1']:.3f} | {ci(s['strict_ci95'])} | {s['min2_F1']:.3f} | {s['min3_F1']:.3f} | {s['macro_F1']:.3f} | {s['relaxed_merged_F1']:.3f} | {s['exact_share_of_tp']:.3f} | {tr} |")
    tex.append(f"{disp} & {typ} & {r['F1']:.3f} & {ci(s['relaxed_ci95'])} & {r['P']:.3f} & {r['R']:.3f} & {st['F1']:.3f} & {ci(s['strict_ci95'])} & {s['macro_F1']:.3f} & {s['relaxed_merged_F1']:.3f} & {s['exact_share_of_tp']:.3f} & {tr}\\\\")
tex.append("\\bottomrule\\end{tabular}\n")
# ---- 表: ゼロショット prompt-only vs constrained ----
md.append("\n## ゼロショット: prompt のみ vs JSON Schema 制約デコード (test36)\n")
md.append("| モデル | 種別 | ZS F1 | ZS strict | parse失敗 | 截断 | 未知ラベル | ZS F1 (parser v2) | parse失敗 v2 | 制約 F1 | 制約 strict | 制約 P | 制約 R | 截断 | FT F1 |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
tex.append("% ZS table\n\\begin{tabular}{llcccccccccc}\\toprule\nModel & Type & ZS F1 & Strict & Parse fail & Trunc. & Constr. F1 & Strict & P & R & Trunc. & FT F1\\\\\\midrule")
for disp, typ, size, ft, zs, zg in MODELS:
    a, b, c = S(zs), S(zg), S(ft)
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
# ---- 表: 多 seed (42/1/2) ----
import statistics as st
SEEDS = [  # (表示名, 種別, [seed42, seed1, seed2])
    ("Med42-70B", "医療", ["med42-70b-ft", "med42-70b-s1-ft", "med42-70b-s2-ft"]),
    ("Llama-3.3-70B", "汎用", ["llama-3.3-70b-ft", "llama-3.3-70b-s1-ft", "llama-3.3-70b-s2-ft"]),
    ("MedGemma-27B", "医療", ["medgemma-27b-ft", "medgemma-27b-s1-ft", "medgemma-27b-s2-ft"]),
    ("Gemma-3-27B (MedGemma base)", "汎用", ["gemma-3-27b-s42-ft", "gemma-3-27b-s1-ft", "gemma-3-27b-s2-ft"]),
    ("SIP-jmed-13B", "医療", ["sip-jmed-13b-ft", "sip-jmed-13b-s1-ft", "sip-jmed-13b-s2-ft"]),
    ("LLM-jp-3.1-13B (SIP base)", "汎用", ["llmjp-3.1-13b-ft", "llmjp-3.1-13b-s1-ft", "llmjp-3.1-13b-s2-ft"]),
    ("Qwen3-30B-A3B", "汎用", ["qwen-ft", "qwen-s1-ft", "qwen-s2-ft"]),
    ("LLM-jp-4-32B", "汎用", ["llm-jp-4-32b-ft", "llm-jp-4-32b-s1-ft", "llm-jp-4-32b-s2-ft"]),
    ("LLM-jp-1.8B", "汎用", ["llm-jp-1.8b-ft", "llm-jp-1.8b-s1-ft", "llm-jp-1.8b-s2-ft"]),
]
md.append("\n## QLoRA 多 seed (42 / 1 / 2): relaxed F1 と strict F1 の平均 ± SD\n")
md.append("| モデル | 種別 | relaxed s42 | s1 | s2 | 平均 ± SD | strict s42 | s1 | s2 | 平均 ± SD | macro 平均 |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
tex.append("% seed table\n\\begin{tabular}{llccccccccc}\\toprule\nModel & Type & Relaxed s42 & s1 & s2 & Mean $\\pm$ SD & Strict s42 & s1 & s2 & Mean $\\pm$ SD & Macro\\\\\\midrule")
for disp, typ, runs in SEEDS:
    ss = [S(r) for r in runs]
    if not all(ss):
        continue
    r = [x["relaxed"]["F1"] for x in ss]; t = [x["strict"]["F1"] for x in ss]; mc = [x["macro_F1"] for x in ss]
    row = [disp, typ] + [f"{v:.3f}" for v in r] + [f"{st.mean(r):.3f} ± {st.stdev(r):.3f}"] + [f"{v:.3f}" for v in t] + [f"{st.mean(t):.3f} ± {st.stdev(t):.3f}", f"{st.mean(mc):.3f}"]
    md.append("| " + " | ".join(row) + " |")
    tex.append(" & ".join(x.replace("±", "$\\pm$") for x in row) + "\\\\")
tex.append("\\bottomrule\\end{tabular}\n")
# ---- 表: 思考型の ZS 4 条件 + 参考モデル ----
md.append("\n## 思考型モデルの ZS 条件 (prompt-only 旧パーサ / 新パーサ / 整段制約 / 2 段階 自由思考+制約回答)\n")
md.append("| モデル | prompt-only (v1) | parse失敗 | prompt-only (v2) | parse失敗 v2 | 整段制約 | 2段階 (think→constrain) | strict (2段階) | FT |")
md.append("|---|---|---|---|---|---|---|---|---|")
for disp, zs, zg, tg, ft in [("LLM-jp-4-32B", "llm-jp-4-32b-zs", "llm-jp-4-32b-zs-guided", "llm-jp-4-32b-zs-think-guided", "llm-jp-4-32b-ft"),
                              ("SIP-jmed-13B", "sip-jmed-13b-zs", "sip-jmed-13b-zs-guided", "sip-jmed-13b-zs-think-guided", "sip-jmed-13b-ft")]:
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
# ---- 表: 基盤対齐 (seed ごとの対差) ----
md.append("\n## 基盤を揃えた対比較 (医療 − 基盤, 同 seed 対応ありブートストラップ)\n")
md.append("| 対 | seed 42 | seed 1 | seed 2 |")
md.append("|---|---|---|---|")
def pairdiff(a, b):
    for f in glob.glob(os.path.join(STATS, "pairwise_*.json")):
        d = json.load(open(f))
        for k, v in d.items():
            if k == f"{a} - {b}":
                return f"{v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
            if k == f"{b} - {a}":
                return f"{-v['diff']:+.3f} [{-v['ci95'][1]:+.3f}, {-v['ci95'][0]:+.3f}]"
    return "-"
for name, pairs in [("MedGemma-27B − Gemma-3-27B", [("medgemma-27b-ft", "gemma-3-27b-s42-ft"), ("medgemma-27b-s1-ft", "gemma-3-27b-s1-ft"), ("medgemma-27b-s2-ft", "gemma-3-27b-s2-ft")]),
                    ("SIP-jmed-13B − LLM-jp-3.1-13B", [("sip-jmed-13b-ft", "llmjp-3.1-13b-ft"), ("sip-jmed-13b-s1-ft", "llmjp-3.1-13b-s1-ft"), ("sip-jmed-13b-s2-ft", "llmjp-3.1-13b-s2-ft")]),
                    ("Med42-70B − Llama-3.3-70B", [("med42-70b-ft", "llama-3.3-70b-ft"), ("med42-70b-s1-ft", "llama-3.3-70b-s1-ft"), ("med42-70b-s2-ft", "llama-3.3-70b-s2-ft")])]:
    md.append(f"| {name} | " + " | ".join(pairdiff(a, b) for a, b in pairs) + " |")
# ---- pairwise ----
for f in sorted(glob.glob(os.path.join(STATS, "pairwise_*.json"))):
    md.append(f"\n## 対差ブートストラップ ({os.path.basename(f)})\n\n| 対 | 差 | 95% CI |\n|---|---|---|")
    for k, v in json.load(open(f)).items():
        md.append(f"| {k} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |")
open(os.path.join(STATS, "tables.md"), "w").write("\n".join(md) + "\n")
open(os.path.join(STATS, "tables.tex"), "w").write("\n".join(tex) + "\n")
print("\n".join(md))
