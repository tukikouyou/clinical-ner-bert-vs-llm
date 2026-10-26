# -*- coding: utf-8 -*-
"""英/日/中の全対照表 (Markdown) を stats/ から生成 → stdout。 usage: readme_tables.py [en|ja|zh]
主条件 = 生成予算 4096 (runs/*-ft-4k; LLM-jp-3.1 系は 2600), LAU 設定 = 900 (runs/*-ft)."""
import os, sys, json, glob, statistics as st
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
HERE = os.path.dirname(os.path.abspath(__file__)); STATS = os.path.join(HERE, "../stats"); RUNS = os.path.join(HERE, "../runs")
def S(n):
    f = os.path.join(STATS, f"{n}.json"); return json.load(open(f)) if n and os.path.exists(f) else None
def M(n):
    f = os.path.join(RUNS, n, "meta.json") if n else None; return json.load(open(f)) if f and os.path.exists(f) else {}
ci = lambda x: f"[{x[0]:.3f}, {x[1]:.3f}]"
TR = {"ja": {
 "medical": "医療", "general": "汎用",
 "Model": "モデル", "Type": "種別", "Budget": "予算", "Pair": "対", "Label": "ラベル", "Condition": "条件", "Mean ± SD": "平均 ± SD", "Macro (mean)": "Macro (平均)",
 "Trunc./433": "打ち切り/433", "Merged": "統合", "Gold share": "正解の割合", "R (trunc.)": "R (打ち切り)", "R (rest)": "R (他)", "P (trunc.)": "P (打ち切り)", "P (rest)": "P (他)", "FN in trunc.": "FN の割合",
 "Fine-tuned": "FT 後", "Prompt only, parser v1": "プロンプトのみ (v1)", "Prompt only, parser v2": "プロンプトのみ (v2)", "Whole-output constraint": "整段制約", "Think, then constrain (rel. / str.)": "自由思考→制約回答 (rel. / str.)",
 "ZS+constrained rel. / str.": "ZS+制約 rel. / str.", "Budget": "予算", "Relaxed (main)": "relaxed (主)", "#### Table M. Generation-budget curve (seed 42): relaxed F1 / sentences cut off / mean output tokens per budget (4,096 column is 2,600 for the LLM-jp-3.1 models)": "#### 表 M. 生成予算曲線 (seed 42): 予算ごとの relaxed F1 / 打ち切り文数 / 平均出力トークン (4,096 列は LLM-jp-3.1 系では 2,600)", "Relaxed (mention)": "relaxed (提及)", "Strict (main)": "strict (主)", "Strict (mention)": "strict (提及)", "Pred. mentions (s42)": "予測提及数 (s42)", "#### Table L. Mention-level scoring (repeated mentions within a sentence not collapsed): main vs. mention-level F1, three-seed means (4,096-token budget; 2,600 for LLM-jp-3.1 models)": "#### 表 L. 提及レベル採点 (文内の重複提及を潰さない): 主採点 vs. 提及レベル F1, 3 seed 平均 (予算 4,096; LLM-jp-3.1 系は 2,600)", "The test set has 15,119 gold mentions by character offsets; the main scores collapse them to 12,919 (type, surface) pairs (14.6% collapsed, 248 sentences). Mention-level: gold counted per occurrence, predictions per occurrence in the output (per BIO segment for BERT), same greedy matching.": "テスト集合の正解提及は文字オフセットで 15,119; 主採点は文内の同一 (型, 表層) を 1 個に集約し 12,919 組 (14.6% が折り畳み, 248 文)。提及レベル: 正解は出現ごと、予測は出力中の出現ごと (BERT は BIO 片ごと) に数え、同じ貪欲対応。", "Superseded constrained runs at 900 output / 4,096 context (prompt-only run of the same model used 3,072 / 8,192; kept for reference):": "差し替え前の制約デコード (900 出力 / 4,096 文脈; 同モデルの prompt-only は 3,072 / 8,192; 参考):",
 "prompt only, reasoning off": "プロンプトのみ、思考オフ", "reasoning on + schema on answer (3,072-token budget)": "思考オン + 回答に schema (予算 3,072)", "prompt only (reasoning cannot be disabled)": "プロンプトのみ (思考は無効化不可)", "reasoning + schema on answer": "思考 + 回答に schema",
 "UTH-BERT (same split as Med42)": "UTH-BERT (Med42 と同じ分割)",
 "#### Table A. Fine-tuned models, main condition (generation budget 4,096 tokens; 2,600 for the LLM-jp-3.1 family), seed 42": "#### 表 A. ファインチューニング後・主条件 (生成予算 4,096 トークン; LLM-jp-3.1 系は 2,600)、seed 42",
 "#### Table A2. Fine-tuned models, LAU setting (generation budget 900 tokens), seed 42": "#### 表 A2. ファインチューニング後・LAU 設定 (生成予算 900 トークン)、seed 42",
 "95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the generation budget.": "95% CI: 文書単位ブートストラップ (36 文書, 2,000 回)。min2/min3: 部分文字列側に最小長 2/3 文字を課した relaxed F1。Macro: テスト集合に 20 件以上あるラベル 48 種の F1 の単純平均。Merged: `-patient`/`-others` を統合した relaxed F1。Exact/TP: relaxed の TP のうち表層が完全一致した割合。Trunc.: 生成上限で打ち切られた文数。",
 "#### Table B. Three QLoRA seeds (42 / 1 / 2), main condition (4,096-token budget)": "#### 表 B. QLoRA 3 seed (42 / 1 / 2)・主条件 (予算 4,096)",
 "#### Table B2. Three QLoRA seeds (42 / 1 / 2), LAU setting (900-token budget)": "#### 表 B2. QLoRA 3 seed (42 / 1 / 2)・LAU 設定 (予算 900)",
 "#### Table C. Generation-budget ablation: 900 → 4,096 tokens (seed 42; last column: three-seed means)": "#### 表 C. 生成予算アブレーション: 900 → 4,096 トークン (seed 42; 最終列は 3 seed 平均)",
 "#### Table D. Zero-shot: prompt only vs. JSON-schema constrained decoding (relaxed / strict F1; both conditions of a model use the same budget: 900 output / 4,096 context, or 3,072 / 8,192)": "#### 表 D. ゼロショット: プロンプトのみ vs. JSON Schema 制約デコード (同一モデルの 2 条件は同じ予算: 900 出力 / 4,096 文脈 または 3,072 / 8,192; relaxed / strict F1; 予算 900、思考型は 3,072)",
 "Parser v1 = the LAU-version parser (first `[` … last `]`); parser v2 = last complete JSON array (reasoning channel discarded). Fail: sentences without a parseable array. Unk.: predictions with a label outside the 109-label list. Trunc.: sentences cut off by the generation budget.": "パーサ v1 = LAU 版のパーサ (最初の `[` … 最後の `]`)、v2 = 最後の完全な JSON 配列 (思考チャネルは除外)。Fail: 配列を取り出せなかった文数。Unk.: 109 ラベル外のラベルを持つ予測数。Trunc.: 生成上限で打ち切られた文数。",
 "#### Table E. Reasoning models: four zero-shot conditions (relaxed F1)": "#### 表 E. 思考型モデルの 4 条件 (relaxed F1)",
 "#### Table F. Reference models (ollama; cannot be fine-tuned locally)": "#### 表 F. 参考モデル (ollama; ローカルでは学習不可)",
 "#### Table G. Medical model minus its general-purpose base, same seed (paired bootstrap, relaxed F1), main condition (4,096-token budget)": "#### 表 G. 医療モデル − その基盤モデル、同 seed (対応ありブートストラップ, relaxed F1)・主条件 (予算 4,096)",
 "#### Table G2. Same, LAU setting (900-token budget)": "#### 表 G2. 同・LAU 設定 (予算 900)",
 "#### Table H. Fine-tuned LLM minus BERT, seed 42 (paired bootstrap, relaxed F1): main condition / LAU setting": "#### 表 H. FT 後 LLM − BERT、seed 42 (対応ありブートストラップ, relaxed F1): 主条件 / LAU 設定",
 "#### Table I. Effect of the 900-token generation budget (fine-tuned, seed 42, relaxed)": "#### 表 I. 生成上限 900 トークンの影響 (FT 後, seed 42, relaxed)",
 "Trunc.: sentences whose output was cut off; Gold: share of all gold entities in those sentences; R: recall on truncated / other sentences; FN: share of the model's false negatives in truncated sentences. UTH-BERT is scored on the sentences truncated by Med42-70B.": "Trunc.: 出力が打ち切られた文数。Gold: それらの文に含まれる正解の割合。R: 打ち切り文 / その他の文の再現率。FN: モデルの FN のうち打ち切り文にある割合。UTH-BERT は Med42-70B が打ち切られた文で分割して評価。",
 "#### Table J. Per-label scores of fine-tuned Med42-70B (seed 42, 4,096-token budget) for nine labels; UTH = UTH-BERT relaxed / strict F1": "#### 表 J. FT 後 Med42-70B (seed 42, 予算 4,096) のラベル別 (9 種); UTH = UTH-BERT の relaxed / strict F1",

 "#### Table K. Output completeness of fine-tuned models (seed 42): 900-token budget / main condition": "#### 表 K. FT 後モデルの出力完全性 (seed 42): 予算 900 / 主条件",
 "Stop: sentences where generation ended with the end-of-sequence token; Clean: raw output is itself a valid JSON array (Harmony channel markers stripped for LLM-jp-4); Repaired: array cut off by the budget and closed at the last complete element; Continued: output continued after a complete array (parsed as the first array); Empty: parsed answer has no entities; Fail: no array could be parsed.": "Stop: 終端トークンで生成が終わった文数。Clean: 生出力がそのまま正しい JSON 配列 (LLM-jp-4 は Harmony のチャネル記号を除去)。Repaired: 上限で途切れた配列を最後の完全な要素で閉じた文数。Continued: 完全な配列の後に生成が続いた文数 (最初の配列を採用)。Empty: 解析結果が空。Fail: 配列を取り出せず。",
 "Stop (900 / main)": "Stop (900 / 主)", "Clean (900 / main)": "Clean (900 / 主)", "Repaired (900 / main)": "Repaired (900 / 主)", "Continued (900 / main)": "Continued (900 / 主)", "Empty (900 / main)": "Empty (900 / 主)", "Fail (900 / main)": "Fail (900 / 主)",
 "UTH-BERT v2 (retrained, seed 42)": "UTH-BERT v2 (再学習, seed 42)", "NICT-BERT v2 (retrained, seed 42)": "NICT-BERT v2 (再学習, seed 42)",
 "UTH-BERT (original training, coverage completed)": "UTH-BERT (元の学習, 欠損文を補完予測)", "NICT-BERT (original training, coverage completed)": "NICT-BERT (元の学習, 欠損文を補完予測)",
}, "zh": {
 "medical": "医学", "general": "通用",
 "Model": "模型", "Type": "类型", "Budget": "预算", "Pair": "配对", "Label": "标签", "Condition": "条件", "Mean ± SD": "均值 ± SD", "Macro (mean)": "Macro (均值)",
 "Trunc./433": "截断/433", "Merged": "合并", "Gold share": "正解比例", "R (trunc.)": "R (截断)", "R (rest)": "R (其余)", "P (trunc.)": "P (截断)", "P (rest)": "P (其余)", "FN in trunc.": "FN 比例",
 "Fine-tuned": "微调后", "Prompt only, parser v1": "仅提示 (v1)", "Prompt only, parser v2": "仅提示 (v2)", "Whole-output constraint": "整段约束", "Think, then constrain (rel. / str.)": "自由思考→约束答案 (rel. / str.)",
 "ZS+constrained rel. / str.": "ZS+约束 rel. / str.", "Budget": "预算", "Relaxed (main)": "relaxed（主）", "#### Table M. Generation-budget curve (seed 42): relaxed F1 / sentences cut off / mean output tokens per budget (4,096 column is 2,600 for the LLM-jp-3.1 models)": "#### 表 M. 生成预算曲线（seed 42）：各预算下的 relaxed F1 / 截断句数 / 平均输出 token（4,096 列对 LLM-jp-3.1 系为 2,600）", "Relaxed (mention)": "relaxed（提及级）", "Strict (main)": "strict（主）", "Strict (mention)": "strict（提及级）", "Pred. mentions (s42)": "预测提及数 (s42)", "#### Table L. Mention-level scoring (repeated mentions within a sentence not collapsed): main vs. mention-level F1, three-seed means (4,096-token budget; 2,600 for LLM-jp-3.1 models)": "#### 表 L. 提及级评价（不折叠句内重复提及）：主评价 vs. 提及级 F1，三 seed 均值（预算 4,096；LLM-jp-3.1 系为 2,600）", "The test set has 15,119 gold mentions by character offsets; the main scores collapse them to 12,919 (type, surface) pairs (14.6% collapsed, 248 sentences). Mention-level: gold counted per occurrence, predictions per occurrence in the output (per BIO segment for BERT), same greedy matching.": "测试集按字符偏移有 15,119 个正解提及；主评价将句内相同（类型, 表层）折叠为 12,919 对（14.6% 被折叠，涉及 248 句）。提及级：正解按每次出现计数，预测按输出中的出现次数计数（BERT 按 BIO 片段），同一贪心匹配。", "Superseded constrained runs at 900 output / 4,096 context (prompt-only run of the same model used 3,072 / 8,192; kept for reference):": "替换前的约束解码结果（900 输出 / 4,096 上下文；同模型 prompt-only 为 3,072 / 8,192；仅供参考）：",
 "prompt only, reasoning off": "仅提示，关闭思考", "reasoning on + schema on answer (3,072-token budget)": "开启思考 + 答案加 schema（预算 3,072）", "prompt only (reasoning cannot be disabled)": "仅提示（思考无法关闭）", "reasoning + schema on answer": "思考 + 答案加 schema",
 "UTH-BERT (same split as Med42)": "UTH-BERT（与 Med42 相同划分）",
 "#### Table A. Fine-tuned models, main condition (generation budget 4,096 tokens; 2,600 for the LLM-jp-3.1 family), seed 42": "#### 表 A. 微调后·主条件（生成预算 4,096 token；LLM-jp-3.1 系为 2,600），seed 42",
 "#### Table A2. Fine-tuned models, LAU setting (generation budget 900 tokens), seed 42": "#### 表 A2. 微调后·LAU 设置（生成预算 900 token），seed 42",
 "95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the generation budget.": "95% CI：文档级 bootstrap（36 篇，2,000 次）。min2/min3：子串一侧至少 2/3 字的 relaxed F1。Macro：测试集出现 ≥20 次的 48 个标签的 F1 平均。Merged：合并 `-patient`/`-others` 后的 relaxed F1。Exact/TP：relaxed 的 TP 中表层完全一致的比例。Trunc.：在生成上限被截断的句数。",
 "#### Table B. Three QLoRA seeds (42 / 1 / 2), main condition (4,096-token budget)": "#### 表 B. QLoRA 三个 seed (42 / 1 / 2)·主条件（预算 4,096）",
 "#### Table B2. Three QLoRA seeds (42 / 1 / 2), LAU setting (900-token budget)": "#### 表 B2. QLoRA 三个 seed (42 / 1 / 2)·LAU 设置（预算 900）",
 "#### Table C. Generation-budget ablation: 900 → 4,096 tokens (seed 42; last column: three-seed means)": "#### 表 C. 生成预算消融：900 → 4,096 token（seed 42；末列为三 seed 均值）",
 "#### Table D. Zero-shot: prompt only vs. JSON-schema constrained decoding (relaxed / strict F1; both conditions of a model use the same budget: 900 output / 4,096 context, or 3,072 / 8,192)": "#### 表 D. 零样本：仅提示 vs. JSON Schema 约束解码（同一模型两条件预算相同：900 输出 / 4,096 上下文 或 3,072 / 8,192；relaxed / strict F1；预算 900，思考型 3,072）",
 "Parser v1 = the LAU-version parser (first `[` … last `]`); parser v2 = last complete JSON array (reasoning channel discarded). Fail: sentences without a parseable array. Unk.: predictions with a label outside the 109-label list. Trunc.: sentences cut off by the generation budget.": "解析器 v1 = LAU 版解析器（第一个 `[` … 最后一个 `]`），v2 = 最后一个完整 JSON 数组（丢弃思考通道）。Fail：无法取出数组的句数。Unk.：标签不在 109 个之内的预测数。Trunc.：被生成上限截断的句数。",
 "#### Table E. Reasoning models: four zero-shot conditions (relaxed F1)": "#### 表 E. 思考型模型的四种零样本条件（relaxed F1）",
 "#### Table F. Reference models (ollama; cannot be fine-tuned locally)": "#### 表 F. 参考模型（ollama；无法本地微调）",
 "#### Table G. Medical model minus its general-purpose base, same seed (paired bootstrap, relaxed F1), main condition (4,096-token budget)": "#### 表 G. 医学模型 − 其通用基座，同 seed（配对 bootstrap，relaxed F1）·主条件（预算 4,096）",
 "#### Table G2. Same, LAU setting (900-token budget)": "#### 表 G2. 同上·LAU 设置（预算 900）",
 "#### Table H. Fine-tuned LLM minus BERT, seed 42 (paired bootstrap, relaxed F1): main condition / LAU setting": "#### 表 H. 微调 LLM − BERT，seed 42（配对 bootstrap，relaxed F1）：主条件 / LAU 设置",
 "#### Table I. Effect of the 900-token generation budget (fine-tuned, seed 42, relaxed)": "#### 表 I. 900 token 生成上限的影响（微调后，seed 42，relaxed）",
 "Trunc.: sentences whose output was cut off; Gold: share of all gold entities in those sentences; R: recall on truncated / other sentences; FN: share of the model's false negatives in truncated sentences. UTH-BERT is scored on the sentences truncated by Med42-70B.": "Trunc.：输出被截断的句数。Gold：这些句子所含正解的比例。R：截断句 / 其余句的召回。FN：该模型的 FN 落在截断句中的比例。UTH-BERT 按 Med42-70B 的截断句划分评价。",
 "#### Table J. Per-label scores of fine-tuned Med42-70B (seed 42, 4,096-token budget) for nine labels; UTH = UTH-BERT relaxed / strict F1": "#### 表 J. 微调后 Med42-70B（seed 42，预算 4,096）九个标签的逐标签成绩；UTH = UTH-BERT 的 relaxed / strict F1",

 "#### Table K. Output completeness of fine-tuned models (seed 42): 900-token budget / main condition": "#### 表 K. 微调模型的输出完整性 (seed 42)：预算 900 / 主条件",
 "Stop: sentences where generation ended with the end-of-sequence token; Clean: raw output is itself a valid JSON array (Harmony channel markers stripped for LLM-jp-4); Repaired: array cut off by the budget and closed at the last complete element; Continued: output continued after a complete array (parsed as the first array); Empty: parsed answer has no entities; Fail: no array could be parsed.": "Stop：以结束符正常结束的句数。Clean：原始输出本身就是合法 JSON 数组（LLM-jp-4 去掉 Harmony 通道标记后）。Repaired：数组在上限处被截断、按最后一个完整元素闭合。Continued：完整数组之后仍继续生成（取第一个数组）。Empty：解析结果为空。Fail：无法取出数组。",
 "Stop (900 / main)": "Stop (900 / 主)", "Clean (900 / main)": "Clean (900 / 主)", "Repaired (900 / main)": "Repaired (900 / 主)", "Continued (900 / main)": "Continued (900 / 主)", "Empty (900 / main)": "Empty (900 / 主)", "Fail (900 / main)": "Fail (900 / 主)",
 "UTH-BERT v2 (retrained, seed 42)": "UTH-BERT v2（重训，seed 42）", "NICT-BERT v2 (retrained, seed 42)": "NICT-BERT v2（重训，seed 42）",
 "UTH-BERT (original training, coverage completed)": "UTH-BERT（原训练，补齐缺失句预测）", "NICT-BERT (original training, coverage completed)": "NICT-BERT（原训练，补齐缺失句预测）",
}}
def tr(x):
    core = x.rstrip("\n"); return TR.get(LANG, {}).get(core, core) + x[len(core):]
def H(cols):  # header row + separator
    return "| " + " | ".join(tr(c) for c in cols) + " |\n|" + "|".join("---" if i == 0 else ":-:" for i in range(len(cols))) + "|"
L = []
# (display, type, stem) ; runs: <stem>-ft (seed42, 900; gemma: gemma-3-27b-s42-ft), <stem>-s1-ft, <stem>-s2-ft ; 4k: + "-4k"
FT = [("Med42-70B", "medical", "med42-70b", "med42-70b-zs", "med42-70b-zs-guided"),
      ("Llama-3.3-70B", "general", "llama-3.3-70b", "llama-3.3-70b-zs", "llama-3.3-70b-zs-guided"),
      ("MedGemma-27B", "medical", "medgemma-27b", "medgemma-27b-zs", "medgemma-27b-zs-guided"),
      ("Gemma-3-27B (base of MedGemma)", "general", "gemma-3-27b", None, None),
      ("SIP-jmed-13B", "medical", "sip-jmed-13b", "sip-jmed-13b-zs", "sip-jmed-13b-zs-guided"),
      ("LLM-jp-3.1-13B (base of SIP-jmed)", "general", "llmjp-3.1-13b", None, None),
      ("Qwen3-30B-A3B (MoE)", "general", "qwen", "qwen-zs", "qwen-zs-guided"),
      ("LLM-jp-4-32B (MoE, reasoning)", "general", "llm-jp-4-32b", "llm-jp-4-32b-zs", "llm-jp-4-32b-zs-guided"),
      ("LLM-jp-1.8B", "general", "llm-jp-1.8b", "llm-jp-1.8b-zs", "llm-jp-1.8b-zs-guided")]
GOLD_N = json.load(open(os.path.join(STATS, "extra.json")))["per_label_med42"] if os.path.exists(os.path.join(STATS, "extra.json")) else {}
BERT_ROWS = [("UTH-BERT v2 (retrained, seed 42)", "medical", "BERT-UTH-v2-s42"), ("NICT-BERT v2 (retrained, seed 42)", "general", "BERT-NICT-v2-s42"),
             ("UTH-BERT (original training, coverage completed)", "medical", "BERT-UTH"), ("NICT-BERT (original training, coverage completed)", "general", "BERT-NICT")]
def runs(stem, budget):
    first = f"{stem}-ft{budget}"
    if stem == "gemma-3-27b" and budget == "":
        first = "gemma-3-27b-s42-ft"
    return [first, f"{stem}-s1-ft{budget}", f"{stem}-s2-ft{budget}"]
def pairdiff(a, b):
    for f in glob.glob(os.path.join(STATS, "pairwise_*.json")):
        d = json.load(open(f))
        if f"{a} - {b}" in d: v = d[f"{a} - {b}"]; return f"{v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
        if f"{b} - {a}" in d: v = d[f"{b} - {a}"]; return f"{-v['diff']:+.3f} [{-v['ci95'][1]:+.3f}, {-v['ci95'][0]:+.3f}]"
    return "–"
GLOSS = "95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the generation budget."

def ft_table(budget, title):
    L.append(tr(title) + "\n"); L.append(tr(GLOSS) + "\n")
    L.append(H(["Model", "Type", "Budget", "Relaxed F1", "95% CI", "P", "R", "Strict F1", "95% CI", "min2", "min3", "Macro48", "Macro98", "Merged", "Exact/TP", "Trunc./433"]))
    def macro_all(s):
        pl = s["per_label_relaxed"]; v = [pl[t]["F1"] for t in GOLD_N if t in pl]; return sum(v) / len(v)
    for disp, typ, stem, _, _ in FT:
        n = runs(stem, budget)[0]; s = S(n)
        if not s: continue
        m = M(n); r, t = s["relaxed"], s["strict"]
        L.append(f"| {disp} | {tr(typ)} | {m.get('max_new_tokens','–')} | {r['F1']:.3f} | {ci(s['relaxed_ci95'])} | {r['P']:.3f} | {r['R']:.3f} | {t['F1']:.3f} | {ci(s['strict_ci95'])} | {s['min2_F1']:.3f} | {s['min3_F1']:.3f} | {s['macro_F1']:.3f} | {macro_all(s):.3f} | {s['relaxed_merged_F1']:.3f} | {s['exact_share_of_tp']:.3f} | {m.get('truncated','–')} |")
    for disp, typ, b in BERT_ROWS:
        s = S(b)
        if not s: continue
        r, t = s["relaxed"], s["strict"]
        L.append(f"| {disp} | {tr(typ)} | – | {r['F1']:.3f} | {ci(s['relaxed_ci95'])} | {r['P']:.3f} | {r['R']:.3f} | {t['F1']:.3f} | {ci(s['strict_ci95'])} | {s['min2_F1']:.3f} | {s['min3_F1']:.3f} | {s['macro_F1']:.3f} | {macro_all(s):.3f} | {s['relaxed_merged_F1']:.3f} | {s['exact_share_of_tp']:.3f} | – |")
    L.append("")

def seed_table(budget, title):
    L.append(tr(title) + "\n")
    L.append(H(["Model", "Type", "Relaxed 42 / 1 / 2", "Mean ± SD", "Strict 42 / 1 / 2", "Mean ± SD", "Macro (mean)"]))
    for disp, typ, stem, _, _ in FT:
        ss = [S(r) for r in runs(stem, budget)]
        if not any(ss): continue
        f3 = lambda x: "–" if x is None else f"{x:.3f}"
        r = [x["relaxed"]["F1"] if x else None for x in ss]; t = [x["strict"]["F1"] if x else None for x in ss]
        rv = [v for v in r if v is not None]; tv = [v for v in t if v is not None]; mc = [x["macro_F1"] for x in ss if x]
        msd = lambda v: (f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
        L.append(f"| {disp} | {tr(typ)} | {' / '.join(f3(v) for v in r)} | {msd(rv)} | {' / '.join(f3(v) for v in t)} | {msd(tv)} | {st.mean(mc):.3f} |")
    if budget == "-4k":
        for disp, typ, b in [("UTH-BERT v2 (retrained)", "medical", "BERT-UTH-v2"), ("NICT-BERT v2 (retrained)", "general", "BERT-NICT-v2")]:
            ss = [S(f"{b}-s{k}") for k in (42, 1, 2)]
            if not any(ss): continue
            r = [x["relaxed"]["F1"] if x else None for x in ss]; t = [x["strict"]["F1"] if x else None for x in ss]
            rv = [v for v in r if v is not None]; tv = [v for v in t if v is not None]; mc = [x["macro_F1"] for x in ss if x]
            L.append(f"| {disp} | {tr(typ)} | {' / '.join(f3(v) for v in r)} | {msd(rv)} | {' / '.join(f3(v) for v in t)} | {msd(tv)} | {st.mean(mc):.3f} |")
    L.append("")

ft_table("-4k", "#### Table A. Fine-tuned models, main condition (generation budget 4,096 tokens; 2,600 for the LLM-jp-3.1 family), seed 42")
ft_table("", "#### Table A2. Fine-tuned models, LAU setting (generation budget 900 tokens), seed 42")
seed_table("-4k", "#### Table B. Three QLoRA seeds (42 / 1 / 2), main condition (4,096-token budget)")
seed_table("", "#### Table B2. Three QLoRA seeds (42 / 1 / 2), LAU setting (900-token budget)")
# C ablation
L.append(tr("#### Table C. Generation-budget ablation: 900 → 4,096 tokens (seed 42; last column: three-seed means)") + "\n")
L.append(H(["Model", "Budget", "Relaxed F1 900 → large", "Δ", "Strict F1 900 → large", "R 900 → large", "P 900 → large", "Trunc. 900 → large", "3-seed relaxed 900 → large"]))
for disp, typ, stem, _, _ in FT:
    a, b = S(runs(stem, "")[0]), S(runs(stem, "-4k")[0])
    if not (a and b): continue
    ma, mb = M(runs(stem, "")[0]), M(runs(stem, "-4k")[0])
    s9 = [S(r) for r in runs(stem, "")]; s4 = [S(r) for r in runs(stem, "-4k")]
    m9 = f"{st.mean([x['relaxed']['F1'] for x in s9]):.3f}" if all(s9) else "–"; m4 = f"{st.mean([x['relaxed']['F1'] for x in s4]):.3f}" if all(s4) else "–"
    L.append(f"| {disp} | {mb.get('max_new_tokens')} | {a['relaxed']['F1']:.3f} → {b['relaxed']['F1']:.3f} | {b['relaxed']['F1']-a['relaxed']['F1']:+.3f} | {a['strict']['F1']:.3f} → {b['strict']['F1']:.3f} | {a['relaxed']['R']:.3f} → {b['relaxed']['R']:.3f} | {a['relaxed']['P']:.3f} → {b['relaxed']['P']:.3f} | {ma.get('truncated')} → {mb.get('truncated')} | {m9} → {m4} |")
L.append("")
# D zero-shot
L.append(tr("#### Table D. Zero-shot: prompt only vs. JSON-schema constrained decoding (relaxed / strict F1; both conditions of a model use the same budget: 900 output / 4,096 context, or 3,072 / 8,192)") + "\n")
L.append(tr("Parser v1 = the LAU-version parser (first `[` … last `]`); parser v2 = last complete JSON array (reasoning channel discarded). Fail: sentences without a parseable array. Unk.: predictions with a label outside the 109-label list. Trunc.: sentences cut off by the generation budget.") + "\n")
L.append(H(["Model", "Budget", "ZS F1 (v1)", "Fail (v1)", "Unk. (v1)", "ZS F1 (v2) rel. / str.", "Fail (v2)", "Trunc.", "ZS+constrained rel. / str.", "P / R", "Trunc.", "FT F1 (main, s42)"]))
for disp, typ, stem, zs, zg in FT:
    if not zs: continue
    a, b, c = S(zs), S(zg), S(runs(stem, "-4k")[0]); a2 = S(zs + "@v2") or a; ma, mb = M(zs), M(zg)
    L.append(f"| {disp} | {ma.get('max_new_tokens')} / {ma.get('max_model_len')} | {a['relaxed']['F1']:.3f} | {ma.get('parse_fail','–')} | {a['n_unknown_label_pred']} | {a2['relaxed']['F1']:.3f} / {a2['strict']['F1']:.3f} | {ma.get('parse_fail_v2', ma.get('parse_fail','–'))} | {ma.get('truncated','–')} | {b['relaxed']['F1']:.3f} / {b['strict']['F1']:.3f} | {b['relaxed']['P']:.3f} / {b['relaxed']['R']:.3f} | {mb.get('truncated','–')} | {c['relaxed']['F1']:.3f} |")
old = [(disp, S(zg + "-900"), M(zg + "-900")) for disp, typ, stem, zs, zg in FT if zg and S(zg + "-900")]
if old:
    L.append("\n" + tr("Superseded constrained runs at 900 output / 4,096 context (prompt-only run of the same model used 3,072 / 8,192; kept for reference):") + " " + "; ".join(f"{d} {o['relaxed']['F1']:.3f} / {o['strict']['F1']:.3f} (trunc. {m.get('truncated')})" for d, o, m in old))
L.append("")
# E reasoning
L.append(tr("#### Table E. Reasoning models: four zero-shot conditions (relaxed F1)") + "\n")
L.append(H(["Model", "Prompt only, parser v1", "Prompt only, parser v2", "Whole-output constraint", "Think, then constrain (rel. / str.)", "Fine-tuned"]))
for disp, zs, zg, tg, ft in [("LLM-jp-4-32B", "llm-jp-4-32b-zs", "llm-jp-4-32b-zs-guided", "llm-jp-4-32b-zs-think-guided", "llm-jp-4-32b-ft-4k"), ("SIP-jmed-13B", "sip-jmed-13b-zs", "sip-jmed-13b-zs-guided", "sip-jmed-13b-zs-think-guided", "sip-jmed-13b-ft-4k")]:
    a, a2, b, c, d = S(zs), S(zs + "@v2"), S(zg), S(tg), S(ft); m = M(zs)
    L.append(f"| {disp} | {a['relaxed']['F1']:.3f} (fail {m.get('parse_fail')}) | {a2['relaxed']['F1']:.3f} (fail {m.get('parse_fail_v2')}) | {b['relaxed']['F1']:.3f} | {c['relaxed']['F1']:.3f} / {c['strict']['F1']:.3f} | {d['relaxed']['F1']:.3f} |")
L.append("")
# F reference
L.append(tr("#### Table F. Reference models (ollama; cannot be fine-tuned locally)") + "\n")
L.append(H(["Model", "Condition", "Relaxed F1", "Strict F1", "P / R", "Fail", "Trunc."]))
for disp, cond, run in [("Qwen3.6-35B", "prompt only, reasoning off", "qwen3.6-35b-zs"), ("Qwen3.6-35B", "reasoning on + schema on answer (3,072-token budget)", "qwen3.6-35b-zs-think-guided"),
                        ("GPT-OSS-120B", "prompt only (reasoning cannot be disabled)", "gpt-oss-120b-zs"), ("GPT-OSS-120B", "reasoning + schema on answer", "gpt-oss-120b-zs-think-guided")]:
    a = S(run); m = M(run)
    L.append(f"| {disp} | {tr(cond)} | {a['relaxed']['F1']:.3f} | {a['strict']['F1']:.3f} | {a['relaxed']['P']:.3f} / {a['relaxed']['R']:.3f} | {m.get('parse_fail','–')} | {m.get('truncated','–')} |")
L.append("")
# G pairs
PAIRS = [("SIP-jmed-13B − LLM-jp-3.1-13B", "sip-jmed-13b", "llmjp-3.1-13b"), ("MedGemma-27B − Gemma-3-27B", "medgemma-27b", "gemma-3-27b"), ("Med42-70B − Llama-3.3-70B", "med42-70b", "llama-3.3-70b")]
def runs2k6(stem):
    return [f"{stem}-ft-2k6", f"{stem}-s1-ft-2k6", f"{stem}-s2-ft-2k6"]
for budget, title in [("-4k", "#### Table G. Medical model minus its general-purpose base, same seed and same generation budget (paired bootstrap, relaxed F1), main condition (4,096 tokens; the 13B pair at 2,600 tokens, the context limit of the base model)"), ("", "#### Table G2. Same, 900-token budget")]:
    L.append(tr(title) + "\n"); L.append(H(["Pair", "Seed 42", "Seed 1", "Seed 2"]))
    for name, a, b in PAIRS:
        ra = runs2k6(a) if (budget == "-4k" and a == "sip-jmed-13b" and S(runs2k6(a)[0])) else runs(a, budget)
        L.append(f"| {name} | " + " | ".join(pairdiff(x, y) for x, y in zip(ra, runs(b, budget))) + " |")
    L.append("")
# H vs BERT
L.append(tr("#### Table H. Fine-tuned LLM minus BERT, seed 42 (paired bootstrap, relaxed F1): main condition / LAU setting") + "\n")
BU, BN = ("BERT-UTH-v2-s42", "BERT-NICT-v2-s42") if S("BERT-UTH-v2-s42") else ("BERT-UTH", "BERT-NICT")
L.append(H(["LLM", "vs. UTH-BERT (4,096)", "vs. NICT-BERT (4,096)", "vs. UTH-BERT (900)", "vs. NICT-BERT (900)"]))
for disp, typ, stem, _, _ in FT:
    L.append(f"| {disp} | {pairdiff(runs(stem, '-4k')[0], BU)} | {pairdiff(runs(stem, '-4k')[0], BN)} | {pairdiff(runs(stem, '')[0], BU)} | {pairdiff(runs(stem, '')[0], BN)} |")
L.append("")
# I truncation (900)
e = json.load(open(os.path.join(STATS, "extra.json")))
L.append(tr("#### Table I. Effect of the 900-token generation budget (fine-tuned, seed 42, relaxed)") + "\n")
L.append(tr("Trunc.: sentences whose output was cut off; Gold: share of all gold entities in those sentences; R: recall on truncated / other sentences; FN: share of the model's false negatives in truncated sentences. UTH-BERT is scored on the sentences truncated by Med42-70B.") + "\n")
L.append(H(["Model", "Trunc.", "Gold share", "R (trunc.)", "R (rest)", "P (trunc.)", "P (rest)", "FN in trunc."]))
for k, v in e["truncation"].items():
    if "n_trunc_sent" in v:
        L.append(f"| {k} | {v['n_trunc_sent']} | {v['gold_share_in_trunc']:.2f} | {v['trunc']['R']:.3f} | {v['rest']['R']:.3f} | {v['trunc']['P']:.3f} | {v['rest']['P']:.3f} | {v['fn_share_from_trunc']:.2f} |")
    else:
        L.append(f"| {tr('UTH-BERT (same split as Med42)')} | (87) | 0.53 | {v['trunc']['R']:.3f} | {v['rest']['R']:.3f} | {v['trunc']['P']:.3f} | {v['rest']['P']:.3f} | – |")
L.append("")
# J per-label (4k)
per = e.get("per_label_med42_4k", e["per_label_med42"]); sp = e.get("spearman_freq_f1_med42_4k", e["spearman_freq_f1_med42"]); oth = e.get("others_labels_4k", e["others_labels"])
L.append(tr("#### Table J. Per-label scores of fine-tuned Med42-70B (seed 42, 4,096-token budget) for nine labels; UTH = UTH-BERT relaxed / strict F1") + "\n")
L.append(H(["Label", "n", "P", "R", "F1 (relaxed)", "F1 (strict)", "UTH (rel. / str.)"]))
for t in ["state-patient", "clinical_test", "treatment", "drug", "PN-Negative-patient", "state-others", "device", "time_span", "value-others"]:
    v = per[t]
    L.append(f"| {t} | {v['n']} | {v['relaxed'][0]:.2f} | {v['relaxed'][1]:.2f} | {v['relaxed'][2]:.2f} | {v['strict'][2]:.2f} | {v['uth_relaxed'][2]:.2f} / {v['uth_strict'][2]:.2f} |")
# L mention-level
ML = json.load(open(os.path.join(STATS, "mention_level.json")))["runs"] if os.path.exists(os.path.join(STATS, "mention_level.json")) else {}
if ML:
    L.append(tr("#### Table L. Mention-level scoring (repeated mentions within a sentence not collapsed): main vs. mention-level F1, three-seed means (4,096-token budget; 2,600 for LLM-jp-3.1 models)") + "\n")
    L.append(tr("The test set has 15,119 gold mentions by character offsets; the main scores collapse them to 12,919 (type, surface) pairs (14.6% collapsed, 248 sentences). Mention-level: gold counted per occurrence, predictions per occurrence in the output (per BIO segment for BERT), same greedy matching.") + "\n")
    L.append(H(["Model", "Relaxed (main)", "Relaxed (mention)", "Strict (main)", "Strict (mention)", "Pred. mentions (s42)"]))
    for disp, typ, stem, zs, zg in FT:
        rs = runs(stem, "-4k"); a = [S(r) for r in rs]; b = [ML.get(r) for r in rs]
        if not all(a) or not all(b): continue
        L.append(f"| {disp} | {st.mean(x['relaxed']['F1'] for x in a):.3f} | {st.mean(x['relaxed']['F1'] for x in b):.3f} | {st.mean(x['strict']['F1'] for x in a):.3f} | {st.mean(x['strict']['F1'] for x in b):.3f} | {b[0]['n_pred_mentions']} |")
    for disp, stem in [("UTH-BERT v2 (retrained, seed 42)", "UTH-v2"), ("NICT-BERT v2 (retrained, seed 42)", "NICT-v2")]:
        rs = [f"{stem}-s42", f"{stem}-s1", f"{stem}-s2"]; a = [S("BERT-" + r) for r in rs]; b = [ML.get("BERT-" + r) for r in rs]
        if not all(a) or not all(b): continue
        L.append(f"| {tr(disp)} | {st.mean(x['relaxed']['F1'] for x in a):.3f} | {st.mean(x['relaxed']['F1'] for x in b):.3f} | {st.mean(x['strict']['F1'] for x in a):.3f} | {st.mean(x['strict']['F1'] for x in b):.3f} | {b[0]['n_pred_mentions']} |")
    L.append("")
# M budget curve
BCJ = json.load(open(os.path.join(STATS, "budget_curve.json"))) if os.path.exists(os.path.join(STATS, "budget_curve.json")) else {}
if BCJ:
    L.append(tr("#### Table M. Generation-budget curve (seed 42): relaxed F1 / sentences cut off / mean output tokens per budget (4,096 column is 2,600 for the LLM-jp-3.1 models)") + "\n")
    L.append(H(["Model", "512", "900", "1,024", "2,048", "4,096"]))
    for d, row in BCJ.items():
        L.append(f"| {d} | " + " | ".join(f"{row[k]['F1']:.3f} / {row[k]['trunc']} / {row[k]['mean_out_tokens']}" for k in ("512", "900", "1k", "2k", "4k") if k in row) + " |")
    L.append("")
# K completeness
rc = json.load(open(os.path.join(STATS, "raw_completeness.json"))) if os.path.exists(os.path.join(STATS, "raw_completeness.json")) else {}
if rc:
    L.append("\n" + tr("#### Table K. Output completeness of fine-tuned models (seed 42): 900-token budget / main condition") + "\n")
    L.append(tr("Stop: sentences where generation ended with the end-of-sequence token; Clean: raw output is itself a valid JSON array (Harmony channel markers stripped for LLM-jp-4); Repaired: array cut off by the budget and closed at the last complete element; Continued: output continued after a complete array (parsed as the first array); Empty: parsed answer has no entities; Fail: no array could be parsed.") + "\n")
    L.append(H(["Model", "Stop (900 / main)", "Clean (900 / main)", "Repaired (900 / main)", "Continued (900 / main)", "Empty (900 / main)", "Fail (900 / main)"]))
    for disp, typ, stem, _, _ in FT:
        a, b = rc.get(runs(stem, "")[0]), rc.get(runs(stem, "-4k")[0])
        if not (a and b): continue
        f = lambda k: f"{a[k]} / {b[k]}"
        L.append(f"| {disp} | {f('stop')} | {f('clean_json')} | {f('repaired_after_cutoff')} | {f('continued_after_array')} | {f('empty_answer')} | {f('parse_fail')} |")
SP = {"ja": ("ラベル頻度と relaxed F1 の Spearman 相関", "`-others` 27 種の平均 F1"), "zh": ("标签频率与 relaxed F1 的 Spearman 相关", "27 个 `-others` 标签的平均 F1"), "en": ("Spearman correlation between label frequency and relaxed F1", "The 27 `-others` labels average F1")}[LANG if LANG in ("ja", "zh") else "en"]
L.append(f"\n{SP[0]} (Med42-70B, {sp['n_labels']} labels): ρ = {sp['rho']:.2f} (p = {sp['p']:.0e}). {SP[1]} {oth['mean_f1_med42']:.2f}.")
print("\n".join(L))
