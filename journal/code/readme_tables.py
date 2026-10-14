# -*- coding: utf-8 -*-
"""英語版の全対照表 (Markdown) を stats/ から生成 → stdout。README 用。"""
import os, sys, json, glob, statistics as st
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
TR = {
"ja": {
 "#### Table A. QLoRA fine-tuned models, seed 42 (test-36, 433 sentences, 12,919 gold entities)": "#### 表 A. QLoRA ファインチューニング後 (seed 42; test-36, 433 文, 正解 12,919 個)",
 "95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the 900-token budget.": "95% CI: 文書単位ブートストラップ (36 文書, 2,000 回)。min2/min3: 部分文字列側に最小長 2/3 文字を課した relaxed F1。Macro: テスト集合に 20 件以上あるラベル 48 種の F1 の単純平均。Merged: `-patient`/`-others` を統合した relaxed F1。Exact/TP: relaxed の TP のうち表層が完全一致した割合。Trunc.: 900 トークン上限で打ち切られた文数。",
 "| Model | Type | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro | Merged | Exact/TP | Trunc./433 |": "| モデル | 種別 | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro | 統合 | Exact/TP | 打ち切り/433 |",
 "#### Table B. Three QLoRA seeds (42 / 1 / 2)": "#### 表 B. QLoRA 3 seed (42 / 1 / 2)",
 "| Model | Type | Relaxed 42 / 1 / 2 | Mean ± SD | Strict 42 / 1 / 2 | Mean ± SD | Macro (mean) |": "| モデル | 種別 | Relaxed 42 / 1 / 2 | 平均 ± SD | Strict 42 / 1 / 2 | 平均 ± SD | Macro (平均) |",
 "#### Table C. Zero-shot: prompt only vs. JSON-schema constrained decoding (relaxed / strict F1)": "#### 表 C. ゼロショット: プロンプトのみ vs. JSON Schema 制約デコード (relaxed / strict F1)",
 "Parser v1 = the LAU-version parser (first `[` … last `]`); parser v2 = last complete JSON array (reasoning channel discarded). Fail: sentences without a parseable array. Unk.: predictions with a label outside the 109-label list. Trunc.: sentences cut off by the generation budget (900 tokens; 3,072 for the two reasoning models).": "パーサ v1 = LAU 版のパーサ (最初の `[` … 最後の `]`)、v2 = 最後の完全な JSON 配列 (思考チャネルは除外)。Fail: 配列を取り出せなかった文数。Unk.: 109 ラベル外のラベルを持つ予測数。Trunc.: 生成上限 (900 トークン、思考型 2 モデルは 3,072) で打ち切られた文数。",
 "| Model | ZS F1 (v1) | Fail (v1) | Unk. (v1) | ZS F1 (v2) rel. / str. | Fail (v2) | Trunc. | ZS+constrained rel. / str. | P / R | Trunc. | FT F1 (s42) |": "| モデル | ZS F1 (v1) | Fail (v1) | Unk. (v1) | ZS F1 (v2) rel. / str. | Fail (v2) | Trunc. | ZS+制約 rel. / str. | P / R | Trunc. | FT F1 (s42) |",
 "#### Table D. Reasoning models: four zero-shot conditions (relaxed F1)": "#### 表 D. 思考型モデルの 4 条件 (relaxed F1)",
 "| Model | Prompt only, parser v1 | Prompt only, parser v2 | Whole-output constraint | Think, then constrain (rel. / str.) | Fine-tuned |": "| モデル | プロンプトのみ (v1) | プロンプトのみ (v2) | 整段制約 | 自由思考→制約回答 (rel. / str.) | FT 後 |",
 "#### Table E. Reference models (ollama; cannot be fine-tuned locally)": "#### 表 E. 参考モデル (ollama; ローカルでは学習不可)",
 "| Model | Condition | Relaxed F1 | Strict F1 | P / R | Fail | Trunc. |": "| モデル | 条件 | Relaxed F1 | Strict F1 | P / R | Fail | Trunc. |",
 "prompt only, reasoning off": "プロンプトのみ、思考オフ", "reasoning on + schema on answer (3,072-token budget)": "思考オン + 回答に schema (予算 3,072 トークン)",
 "prompt only (reasoning cannot be disabled)": "プロンプトのみ (思考は無効化不可)", "reasoning + schema on answer": "思考 + 回答に schema",
 "#### Table F. Medical model minus its general-purpose base, same seed (paired bootstrap, relaxed F1)": "#### 表 F. 医療モデル − その基盤モデル、同 seed (対応ありブートストラップ, relaxed F1)",
 "| Pair | Seed 42 | Seed 1 | Seed 2 |": "| 対 | Seed 42 | Seed 1 | Seed 2 |",
 "#### Table G. Fine-tuned LLM minus BERT, seed 42 (paired bootstrap, relaxed F1)": "#### 表 G. FT 後 LLM − BERT、seed 42 (対応ありブートストラップ, relaxed F1)",
 "| LLM | vs. UTH-BERT | vs. NICT-BERT |": "| LLM | vs. UTH-BERT | vs. NICT-BERT |",
 "#### Table H. Effect of the 900-token generation budget (fine-tuned, seed 42, relaxed)": "#### 表 H. 生成上限 900 トークンの影響 (FT 後, seed 42, relaxed)",
 "Trunc.: sentences whose output was cut off; Gold: share of all gold entities in those sentences; R: recall on truncated / other sentences; FN: share of the model's false negatives in truncated sentences. UTH-BERT is scored on the sentences truncated by Med42-70B.": "Trunc.: 出力が打ち切られた文数。Gold: それらの文に含まれる正解の割合。R: 打ち切り文 / その他の文の再現率。FN: モデルの FN のうち打ち切り文にある割合。UTH-BERT は Med42-70B が打ち切られた文で分割して評価。",
 "| Model | Trunc. | Gold share | R (trunc.) | R (rest) | P (trunc.) | P (rest) | FN in trunc. |": "| モデル | Trunc. | 正解の割合 | R (打ち切り) | R (他) | P (打ち切り) | P (他) | FN の割合 |",
 "UTH-BERT (same split as Med42)": "UTH-BERT (Med42 と同じ分割)",
 "#### Table I. Per-label scores of fine-tuned Med42-70B (seed 42) for nine labels; UTH = UTH-BERT relaxed / strict F1": "#### 表 I. FT 後 Med42-70B (seed 42) のラベル別 (9 種); UTH = UTH-BERT の relaxed / strict F1",
 "| Label | n | P | R | F1 (relaxed) | F1 (strict) | UTH (rel. / str.) |": "| ラベル | n | P | R | F1 (relaxed) | F1 (strict) | UTH (rel. / str.) |",
 "medical": "医療", "general": "汎用",
},
"zh": {
 "#### Table A. QLoRA fine-tuned models, seed 42 (test-36, 433 sentences, 12,919 gold entities)": "#### 表 A. QLoRA 微调后 (seed 42; test-36, 433 句, 正解 12,919 个)",
 "95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the 900-token budget.": "95% CI：文档级 bootstrap（36 篇，2,000 次）。min2/min3：子串一侧至少 2/3 字的 relaxed F1。Macro：测试集出现 ≥20 次的 48 个标签的 F1 平均。Merged：合并 `-patient`/`-others` 后的 relaxed F1。Exact/TP：relaxed 的 TP 中表层完全一致的比例。Trunc.：在 900 token 上限被截断的句数。",
 "| Model | Type | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro | Merged | Exact/TP | Trunc./433 |": "| 模型 | 类型 | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro | 合并 | Exact/TP | 截断/433 |",
 "#### Table B. Three QLoRA seeds (42 / 1 / 2)": "#### 表 B. QLoRA 三个 seed (42 / 1 / 2)",
 "| Model | Type | Relaxed 42 / 1 / 2 | Mean ± SD | Strict 42 / 1 / 2 | Mean ± SD | Macro (mean) |": "| 模型 | 类型 | Relaxed 42 / 1 / 2 | 均值 ± SD | Strict 42 / 1 / 2 | 均值 ± SD | Macro (均值) |",
 "#### Table C. Zero-shot: prompt only vs. JSON-schema constrained decoding (relaxed / strict F1)": "#### 表 C. 零样本：仅提示 vs. JSON Schema 约束解码 (relaxed / strict F1)",
 "Parser v1 = the LAU-version parser (first `[` … last `]`); parser v2 = last complete JSON array (reasoning channel discarded). Fail: sentences without a parseable array. Unk.: predictions with a label outside the 109-label list. Trunc.: sentences cut off by the generation budget (900 tokens; 3,072 for the two reasoning models).": "解析器 v1 = LAU 版解析器（第一个 `[` … 最后一个 `]`），v2 = 最后一个完整 JSON 数组（丢弃思考通道）。Fail：无法取出数组的句数。Unk.：标签不在 109 个之内的预测数。Trunc.：被生成上限（900 token，两个思考型为 3,072）截断的句数。",
 "| Model | ZS F1 (v1) | Fail (v1) | Unk. (v1) | ZS F1 (v2) rel. / str. | Fail (v2) | Trunc. | ZS+constrained rel. / str. | P / R | Trunc. | FT F1 (s42) |": "| 模型 | ZS F1 (v1) | Fail (v1) | Unk. (v1) | ZS F1 (v2) rel. / str. | Fail (v2) | Trunc. | ZS+约束 rel. / str. | P / R | Trunc. | FT F1 (s42) |",
 "#### Table D. Reasoning models: four zero-shot conditions (relaxed F1)": "#### 表 D. 思考型模型的四种零样本条件 (relaxed F1)",
 "| Model | Prompt only, parser v1 | Prompt only, parser v2 | Whole-output constraint | Think, then constrain (rel. / str.) | Fine-tuned |": "| 模型 | 仅提示 (v1) | 仅提示 (v2) | 整段约束 | 自由思考→约束答案 (rel. / str.) | 微调后 |",
 "#### Table E. Reference models (ollama; cannot be fine-tuned locally)": "#### 表 E. 参考模型 (ollama；无法本地微调)",
 "| Model | Condition | Relaxed F1 | Strict F1 | P / R | Fail | Trunc. |": "| 模型 | 条件 | Relaxed F1 | Strict F1 | P / R | Fail | Trunc. |",
 "prompt only, reasoning off": "仅提示，关闭思考", "reasoning on + schema on answer (3,072-token budget)": "开启思考 + 答案加 schema（预算 3,072 token）",
 "prompt only (reasoning cannot be disabled)": "仅提示（思考无法关闭）", "reasoning + schema on answer": "思考 + 答案加 schema",
 "#### Table F. Medical model minus its general-purpose base, same seed (paired bootstrap, relaxed F1)": "#### 表 F. 医学模型 − 其通用基座，同 seed (配对 bootstrap, relaxed F1)",
 "| Pair | Seed 42 | Seed 1 | Seed 2 |": "| 配对 | Seed 42 | Seed 1 | Seed 2 |",
 "#### Table G. Fine-tuned LLM minus BERT, seed 42 (paired bootstrap, relaxed F1)": "#### 表 G. 微调 LLM − BERT，seed 42 (配对 bootstrap, relaxed F1)",
 "| LLM | vs. UTH-BERT | vs. NICT-BERT |": "| LLM | vs. UTH-BERT | vs. NICT-BERT |",
 "#### Table H. Effect of the 900-token generation budget (fine-tuned, seed 42, relaxed)": "#### 表 H. 900 token 生成上限的影响 (微调后, seed 42, relaxed)",
 "Trunc.: sentences whose output was cut off; Gold: share of all gold entities in those sentences; R: recall on truncated / other sentences; FN: share of the model's false negatives in truncated sentences. UTH-BERT is scored on the sentences truncated by Med42-70B.": "Trunc.：输出被截断的句数。Gold：这些句子所含正解的比例。R：截断句 / 其余句的召回。FN：该模型的 FN 落在截断句中的比例。UTH-BERT 按 Med42-70B 的截断句划分评价。",
 "| Model | Trunc. | Gold share | R (trunc.) | R (rest) | P (trunc.) | P (rest) | FN in trunc. |": "| 模型 | Trunc. | 正解比例 | R (截断) | R (其余) | P (截断) | P (其余) | FN 比例 |",
 "UTH-BERT (same split as Med42)": "UTH-BERT（与 Med42 相同划分）",
 "#### Table I. Per-label scores of fine-tuned Med42-70B (seed 42) for nine labels; UTH = UTH-BERT relaxed / strict F1": "#### 表 I. 微调后 Med42-70B (seed 42) 九个标签的逐标签成绩；UTH = UTH-BERT 的 relaxed / strict F1",
 "| Label | n | P | R | F1 (relaxed) | F1 (strict) | UTH (rel. / str.) |": "| 标签 | n | P | R | F1 (relaxed) | F1 (strict) | UTH (rel. / str.) |",
 "medical": "医学", "general": "通用",
}}
def tr(x):
    core = x.rstrip("\n"); return TR.get(LANG, {}).get(core, core) + x[len(core):]
HERE = os.path.dirname(os.path.abspath(__file__)); STATS = os.path.join(HERE, "../stats"); RUNS = os.path.join(HERE, "../runs")
def S(n):
    f = os.path.join(STATS, f"{n}.json"); return json.load(open(f)) if n and os.path.exists(f) else None
def M(n):
    f = os.path.join(RUNS, n, "meta.json") if n else None; return json.load(open(f)) if f and os.path.exists(f) else {}
ci = lambda x: f"[{x[0]:.3f}, {x[1]:.3f}]"
L = []
FT = [("Med42-70B", "medical", "med42-70b-ft", ["med42-70b-s1-ft", "med42-70b-s2-ft"], "med42-70b-zs", "med42-70b-zs-guided"),
      ("Llama-3.3-70B", "general", "llama-3.3-70b-ft", ["llama-3.3-70b-s1-ft", "llama-3.3-70b-s2-ft"], "llama-3.3-70b-zs", "llama-3.3-70b-zs-guided"),
      ("MedGemma-27B", "medical", "medgemma-27b-ft", ["medgemma-27b-s1-ft", "medgemma-27b-s2-ft"], "medgemma-27b-zs", "medgemma-27b-zs-guided"),
      ("Gemma-3-27B (base of MedGemma)", "general", "gemma-3-27b-s42-ft", ["gemma-3-27b-s1-ft", "gemma-3-27b-s2-ft"], None, None),
      ("SIP-jmed-13B", "medical", "sip-jmed-13b-ft", ["sip-jmed-13b-s1-ft", "sip-jmed-13b-s2-ft"], "sip-jmed-13b-zs", "sip-jmed-13b-zs-guided"),
      ("LLM-jp-3.1-13B (base of SIP-jmed)", "general", "llmjp-3.1-13b-ft", ["llmjp-3.1-13b-s1-ft", "llmjp-3.1-13b-s2-ft"], None, None),
      ("Qwen3-30B-A3B (MoE)", "general", "qwen-ft", ["qwen-s1-ft", "qwen-s2-ft"], "qwen-zs", "qwen-zs-guided"),
      ("LLM-jp-4-32B (MoE, reasoning)", "general", "llm-jp-4-32b-ft", ["llm-jp-4-32b-s1-ft", "llm-jp-4-32b-s2-ft"], "llm-jp-4-32b-zs", "llm-jp-4-32b-zs-guided"),
      ("LLM-jp-1.8B", "general", "llm-jp-1.8b-ft", ["llm-jp-1.8b-s1-ft", "llm-jp-1.8b-s2-ft"], "llm-jp-1.8b-zs", "llm-jp-1.8b-zs-guided")]
L.append(tr("#### Table A. QLoRA fine-tuned models, seed 42 (test-36, 433 sentences, 12,919 gold entities)\n"))
L.append(tr("95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the 900-token budget.\n"))
L.append(tr("| Model | Type | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro | Merged | Exact/TP | Trunc./433 |"))
L.append("|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for disp, typ, ft, _, _, _ in FT:
    s = S(ft); m = M(ft); r, t = s["relaxed"], s["strict"]
    L.append(f"| {disp} | {tr(typ)} | {r['F1']:.3f} | {ci(s['relaxed_ci95'])} | {r['P']:.3f} | {r['R']:.3f} | {t['F1']:.3f} | {ci(s['strict_ci95'])} | {s['min2_F1']:.3f} | {s['min3_F1']:.3f} | {s['macro_F1']:.3f} | {s['relaxed_merged_F1']:.3f} | {s['exact_share_of_tp']:.3f} | {m.get('truncated','-')} |")
for disp, typ, b in [("UTH-BERT (fine-tuned)", "medical", "BERT-UTH"), ("NICT-BERT (fine-tuned)", "general", "BERT-NICT")]:
    s = S(b); r, t = s["relaxed"], s["strict"]
    L.append(f"| {disp} | {tr(typ)} | {r['F1']:.3f} | {ci(s['relaxed_ci95'])} | {r['P']:.3f} | {r['R']:.3f} | {t['F1']:.3f} | {ci(s['strict_ci95'])} | {s['min2_F1']:.3f} | {s['min3_F1']:.3f} | {s['macro_F1']:.3f} | {s['relaxed_merged_F1']:.3f} | {s['exact_share_of_tp']:.3f} | – |")
L.append("\n" + tr("#### Table B. Three QLoRA seeds (42 / 1 / 2)\n"))
L.append(tr("| Model | Type | Relaxed 42 / 1 / 2 | Mean ± SD | Strict 42 / 1 / 2 | Mean ± SD | Macro (mean) |"))
L.append("|---|---|:-:|:-:|:-:|:-:|:-:|")
for disp, typ, ft, others, _, _ in FT:
    ss = [S(ft)] + [S(o) for o in others]; r = [x["relaxed"]["F1"] for x in ss]; t = [x["strict"]["F1"] for x in ss]; mc = [x["macro_F1"] for x in ss]
    L.append(f"| {disp} | {tr(typ)} | {' / '.join(f'{v:.3f}' for v in r)} | {st.mean(r):.3f} ± {st.stdev(r):.3f} | {' / '.join(f'{v:.3f}' for v in t)} | {st.mean(t):.3f} ± {st.stdev(t):.3f} | {st.mean(mc):.3f} |")
L.append("\n" + tr("#### Table C. Zero-shot: prompt only vs. JSON-schema constrained decoding (relaxed / strict F1)\n"))
L.append(tr("Parser v1 = the LAU-version parser (first `[` … last `]`); parser v2 = last complete JSON array (reasoning channel discarded). Fail: sentences without a parseable array. Unk.: predictions with a label outside the 109-label list. Trunc.: sentences cut off by the generation budget (900 tokens; 3,072 for the two reasoning models).\n"))
L.append(tr("| Model | ZS F1 (v1) | Fail (v1) | Unk. (v1) | ZS F1 (v2) rel. / str. | Fail (v2) | Trunc. | ZS+constrained rel. / str. | P / R | Trunc. | FT F1 (s42) |"))
L.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for disp, typ, ft, _, zs, zg in FT:
    if not zs: continue
    a, b, c = S(zs), S(zg), S(ft); a2 = S(zs + "@v2") or a; ma, mb = M(zs), M(zg)
    L.append(f"| {disp} | {a['relaxed']['F1']:.3f} | {ma.get('parse_fail','-')} | {a['n_unknown_label_pred']} | {a2['relaxed']['F1']:.3f} / {a2['strict']['F1']:.3f} | {ma.get('parse_fail_v2', ma.get('parse_fail','-'))} | {ma.get('truncated','-')} | {b['relaxed']['F1']:.3f} / {b['strict']['F1']:.3f} | {b['relaxed']['P']:.3f} / {b['relaxed']['R']:.3f} | {mb.get('truncated','-')} | {c['relaxed']['F1']:.3f} |")
L.append("\n" + tr("#### Table D. Reasoning models: four zero-shot conditions (relaxed F1)\n"))
L.append(tr("| Model | Prompt only, parser v1 | Prompt only, parser v2 | Whole-output constraint | Think, then constrain (rel. / str.) | Fine-tuned |"))
L.append("|---|:-:|:-:|:-:|:-:|:-:|")
for disp, zs, zg, tg, ft in [("LLM-jp-4-32B", "llm-jp-4-32b-zs", "llm-jp-4-32b-zs-guided", "llm-jp-4-32b-zs-think-guided", "llm-jp-4-32b-ft"), ("SIP-jmed-13B", "sip-jmed-13b-zs", "sip-jmed-13b-zs-guided", "sip-jmed-13b-zs-think-guided", "sip-jmed-13b-ft")]:
    a, a2, b, c, d = S(zs), S(zs + "@v2"), S(zg), S(tg), S(ft); m = M(zs)
    L.append(f"| {disp} | {a['relaxed']['F1']:.3f} (fail {m.get('parse_fail')}) | {a2['relaxed']['F1']:.3f} (fail {m.get('parse_fail_v2')}) | {b['relaxed']['F1']:.3f} | {c['relaxed']['F1']:.3f} / {c['strict']['F1']:.3f} | {d['relaxed']['F1']:.3f} |")
L.append("\n" + tr("#### Table E. Reference models (ollama; cannot be fine-tuned locally)\n"))
L.append(tr("| Model | Condition | Relaxed F1 | Strict F1 | P / R | Fail | Trunc. |"))
L.append("|---|---|:-:|:-:|:-:|:-:|:-:|")
for disp, cond, run in [("Qwen3.6-35B", "prompt only, reasoning off", "qwen3.6-35b-zs"), ("Qwen3.6-35B", "reasoning on + schema on answer (3,072-token budget)", "qwen3.6-35b-zs-think-guided"),
                        ("GPT-OSS-120B", "prompt only (reasoning cannot be disabled)", "gpt-oss-120b-zs"), ("GPT-OSS-120B", "reasoning + schema on answer", "gpt-oss-120b-zs-think-guided")]:
    a = S(run); m = M(run)
    L.append(f"| {disp} | {tr(cond)} | {a['relaxed']['F1']:.3f} | {a['strict']['F1']:.3f} | {a['relaxed']['P']:.3f} / {a['relaxed']['R']:.3f} | {m.get('parse_fail','-')} | {m.get('truncated','-')} |")
def pairdiff(a, b):
    for f in glob.glob(os.path.join(STATS, "pairwise_*.json")):
        d = json.load(open(f))
        if f"{a} - {b}" in d: v = d[f"{a} - {b}"]; return f"{v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
        if f"{b} - {a}" in d: v = d[f"{b} - {a}"]; return f"{-v['diff']:+.3f} [{-v['ci95'][1]:+.3f}, {-v['ci95'][0]:+.3f}]"
    return "-"
L.append("\n" + tr("#### Table F. Medical model minus its general-purpose base, same seed (paired bootstrap, relaxed F1)\n"))
L.append(tr("| Pair | Seed 42 | Seed 1 | Seed 2 |")); L.append("|---|:-:|:-:|:-:|")
for name, pairs in [("SIP-jmed-13B − LLM-jp-3.1-13B", [("sip-jmed-13b-ft", "llmjp-3.1-13b-ft"), ("sip-jmed-13b-s1-ft", "llmjp-3.1-13b-s1-ft"), ("sip-jmed-13b-s2-ft", "llmjp-3.1-13b-s2-ft")]),
                    ("MedGemma-27B − Gemma-3-27B", [("medgemma-27b-ft", "gemma-3-27b-s42-ft"), ("medgemma-27b-s1-ft", "gemma-3-27b-s1-ft"), ("medgemma-27b-s2-ft", "gemma-3-27b-s2-ft")]),
                    ("Med42-70B − Llama-3.3-70B", [("med42-70b-ft", "llama-3.3-70b-ft"), ("med42-70b-s1-ft", "llama-3.3-70b-s1-ft"), ("med42-70b-s2-ft", "llama-3.3-70b-s2-ft")])]:
    L.append(f"| {name} | " + " | ".join(pairdiff(a, b) for a, b in pairs) + " |")
L.append("\n" + tr("#### Table G. Fine-tuned LLM minus BERT, seed 42 (paired bootstrap, relaxed F1)\n"))
L.append(tr("| LLM | vs. UTH-BERT | vs. NICT-BERT |")); L.append("|---|:-:|:-:|")
for disp, typ, ft, _, _, _ in FT:
    if "base" in disp: continue
    L.append(f"| {disp} | {pairdiff(ft, 'BERT-UTH')} | {pairdiff(ft, 'BERT-NICT')} |")
e = json.load(open(os.path.join(STATS, "extra.json")))
L.append("\n" + tr("#### Table H. Effect of the 900-token generation budget (fine-tuned, seed 42, relaxed)\n"))
L.append(tr("Trunc.: sentences whose output was cut off; Gold: share of all gold entities in those sentences; R: recall on truncated / other sentences; FN: share of the model's false negatives in truncated sentences. UTH-BERT is scored on the sentences truncated by Med42-70B.\n"))
L.append(tr("| Model | Trunc. | Gold share | R (trunc.) | R (rest) | P (trunc.) | P (rest) | FN in trunc. |")); L.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
for k, v in e["truncation"].items():
    if "n_trunc_sent" in v:
        L.append(f"| {k} | {v['n_trunc_sent']} | {v['gold_share_in_trunc']:.2f} | {v['trunc']['R']:.3f} | {v['rest']['R']:.3f} | {v['trunc']['P']:.3f} | {v['rest']['P']:.3f} | {v['fn_share_from_trunc']:.2f} |")
    else:
        L.append(f"| {tr('UTH-BERT (same split as Med42)')} | (87) | 0.53 | {v['trunc']['R']:.3f} | {v['rest']['R']:.3f} | {v['trunc']['P']:.3f} | {v['rest']['P']:.3f} | – |")
L.append("\n" + tr("#### Table I. Per-label scores of fine-tuned Med42-70B (seed 42) for nine labels; UTH = UTH-BERT relaxed / strict F1\n"))
L.append(tr("| Label | n | P | R | F1 (relaxed) | F1 (strict) | UTH (rel. / str.) |")); L.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|")
for t in ["state-patient", "clinical_test", "treatment", "drug", "PN-Negative-patient", "state-others", "device", "time_span", "value-others"]:
    v = e["per_label_med42"][t]
    L.append(f"| {t} | {v['n']} | {v['relaxed'][0]:.2f} | {v['relaxed'][1]:.2f} | {v['relaxed'][2]:.2f} | {v['strict'][2]:.2f} | {v['uth_relaxed'][2]:.2f} / {v['uth_strict'][2]:.2f} |")
sp = e["spearman_freq_f1_med42"]
SP = {"ja": ("ラベル頻度と relaxed F1 の Spearman 相関", "`-others` 27 種の平均 F1"), "zh": ("标签频率与 relaxed F1 的 Spearman 相关", "27 个 `-others` 标签的平均 F1"), "en": ("Spearman correlation between label frequency and relaxed F1", "The 27 `-others` labels average F1")}[LANG if LANG in ("ja", "zh") else "en"]
L.append(f"\n{SP[0]} (Med42-70B, {sp['n_labels']} labels): ρ = {sp['rho']:.2f} (p = {sp['p']:.0e}). {SP[1]} {e['others_labels']['mean_f1_med42']:.2f}.")
print("\n".join(L))
