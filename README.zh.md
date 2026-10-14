# 日语临床 NER：微调 BERT vs. 零样本/微调 LLM

[English](README.md) | [日本語](README.ja.md) | **中文**

对日语临床文本做命名实体识别（NER），在同一份 gold 标注语料上、用统一评价流程，
比较**微调后的 BERT 编码器**与**大语言模型（LLM）**（零样本及 QLoRA 微调后）。

本项目有三个目标：

1. **精度对比** — 在公开标注语料上测量 BERT（UTH-BERT、NICT-BERT）与 9 个 LLM
   （LLM-jp、SIP-jmed、Gemma/MedGemma、Qwen、Llama/Med42）的 NER 精度。
2. **真实数据抽取** — 用同一套管线抽取真实临床报告。
3. **FHIR 抽取**（计划中） — 从临床报告中抽取可映射到 FHIR 的项目。

> ⚠️ **本仓库不包含任何临床数据。** 训练/评测语料（iCorpus / 症例報告コーパス）
> 采用仅限研究、禁止再分发的许可，真实临床报告含患者数据。本仓库**只有代码和
> 聚合指标（数字与标签名，无任何文本）**。见 [数据](#数据)。

---

## 结果（期刊版，2026 年 10 月）

设置：iCorpus（113 种实体类型），训练 146 篇 / 测试 36 篇（433 句，正解 12,919 个）；所有 LLM 使用同一提示和同一
QLoRA 超参数；确定性一对一配对，报告 **relaxed**（标签相同且一方表层是另一方的子串）和 **strict**（标签与 NFKC
规范化后的表层完全一致）F1；文档级 bootstrap 置信区间；每个模型 **3 个微调 seed**；**JSON Schema 约束解码**和
思考型模型的两段式"自由思考→约束答案"；SIP-jmed-13B 与 MedGemma-27B 的**通用基座模型**。代码、run 元数据、
统计和图在 [`journal/`](journal/)。LAU-16 的单 seed 结果由下表取代。

主要结论：

- 零样本下所有 LLM 远低于微调 BERT（relaxed F1 ≤ 0.32 vs. 0.75），而且这**不是**输出格式问题：约束解码消除了全部
  parse 失败和体系外标签，但 F1 最多变化 0.02；思考型模型的最好成绩（自由思考＋约束答案）是 0.17。思考型模型的
  "parse 失败"大部分是旧解析器（第一个 `[` … 最后一个 `]`）的缺陷。
- QLoRA 微调后所有 LLM 超过 BERT；≥13B 的模型彼此等价（relaxed 0.815–0.827，strict 0.779–0.798；seed 间 SD ≤ 0.006，
  大于模型间差异）。LLM 对 BERT 的优势在 strict 下（+0.10）大于 relaxed（+0.07）。
- 固定基座后**检测不到医学预训练的效果**（3 对 × 3 seed，配对差全部在 ±0.005 内且区间含 0）。
- 剩余漏检集中在 JSON 输出超过 900 token 上限的约 20% 句子（含 52% 的正解；召回 0.58 vs. 其余 0.92）和主体不是
  患者的标签（`-others`）。

### 对照表

以下数字全部由 `journal/code/readme_tables.py zh` 从 `journal/stats/` 生成（与
[`journal/stats/tables_zh.md`](journal/stats/tables_zh.md) 相同）。relaxed = 标签相同且一方表层是另一方的子串，
strict = 标签与 NFKC 规范化后的表层完全一致；句内一对一贪心配对。图见 [`journal/paper/figs/`](journal/paper/figs/)。

<details open>
<summary>表 A–I</summary>

#### 表 A. QLoRA 微调后 (seed 42; test-36, 433 句, 正解 12,919 个)

95% CI：文档级 bootstrap（36 篇，2,000 次）。min2/min3：子串一侧至少 2/3 字的 relaxed F1。Macro：测试集出现 ≥20 次的 48 个标签的 F1 平均。Merged：合并 `-patient`/`-others` 后的 relaxed F1。Exact/TP：relaxed 的 TP 中表层完全一致的比例。Trunc.：在 900 token 上限被截断的句数。

| 模型 | 类型 | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro | 合并 | Exact/TP | 截断/433 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 医学 | 0.825 | [0.787, 0.858] | 0.924 | 0.745 | 0.793 | [0.756, 0.826] | 0.820 | 0.807 | 0.770 | 0.837 | 0.958 | 87 |
| Llama-3.3-70B | 通用 | 0.823 | [0.786, 0.854] | 0.921 | 0.744 | 0.791 | [0.754, 0.823] | 0.818 | 0.806 | 0.777 | 0.834 | 0.958 | 86 |
| MedGemma-27B | 医学 | 0.824 | [0.786, 0.856] | 0.918 | 0.747 | 0.790 | [0.752, 0.823] | 0.818 | 0.804 | 0.775 | 0.836 | 0.956 | 81 |
| Gemma-3-27B (base of MedGemma) | 通用 | 0.828 | [0.792, 0.860] | 0.923 | 0.751 | 0.796 | [0.759, 0.829] | 0.823 | 0.809 | 0.779 | 0.838 | 0.958 | 84 |
| SIP-jmed-13B | 医学 | 0.820 | [0.782, 0.853] | 0.919 | 0.741 | 0.787 | [0.749, 0.820] | 0.816 | 0.802 | 0.772 | 0.830 | 0.956 | 84 |
| LLM-jp-3.1-13B (base of SIP-jmed) | 通用 | 0.816 | [0.779, 0.848] | 0.913 | 0.737 | 0.777 | [0.740, 0.810] | 0.811 | 0.795 | 0.763 | 0.826 | 0.948 | 82 |
| Qwen3-30B-A3B (MoE) | 通用 | 0.812 | [0.774, 0.845] | 0.912 | 0.732 | 0.774 | [0.736, 0.807] | 0.805 | 0.790 | 0.760 | 0.822 | 0.951 | 85 |
| LLM-jp-4-32B (MoE, reasoning) | 通用 | 0.813 | [0.776, 0.843] | 0.904 | 0.738 | 0.772 | [0.735, 0.803] | 0.807 | 0.790 | 0.758 | 0.823 | 0.946 | 79 |
| LLM-jp-1.8B | 通用 | 0.777 | [0.739, 0.810] | 0.880 | 0.695 | 0.730 | [0.691, 0.765] | 0.770 | 0.752 | 0.716 | 0.789 | 0.935 | 81 |
| UTH-BERT (fine-tuned) | 医学 | 0.751 | [0.686, 0.807] | 0.871 | 0.661 | 0.688 | [0.628, 0.739] | 0.738 | 0.707 | 0.732 | 0.760 | 0.908 | – |
| NICT-BERT (fine-tuned) | 通用 | 0.741 | [0.675, 0.796] | 0.860 | 0.651 | 0.710 | [0.645, 0.765] | 0.738 | 0.724 | 0.707 | 0.750 | 0.954 | – |

#### 表 B. QLoRA 三个 seed (42 / 1 / 2)

| 模型 | 类型 | Relaxed 42 / 1 / 2 | 均值 ± SD | Strict 42 / 1 / 2 | 均值 ± SD | Macro (均值) |
|---|---|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 医学 | 0.825 / 0.826 / 0.829 | 0.827 ± 0.002 | 0.793 / 0.798 / 0.802 | 0.798 ± 0.004 | 0.773 |
| Llama-3.3-70B | 通用 | 0.823 / 0.828 / 0.830 | 0.827 ± 0.003 | 0.791 / 0.801 / 0.801 | 0.798 ± 0.006 | 0.778 |
| MedGemma-27B | 医学 | 0.824 / 0.829 / 0.830 | 0.827 ± 0.003 | 0.790 / 0.797 / 0.798 | 0.795 ± 0.004 | 0.780 |
| Gemma-3-27B (base of MedGemma) | 通用 | 0.828 / 0.825 / 0.828 | 0.827 ± 0.002 | 0.796 / 0.793 / 0.798 | 0.796 ± 0.002 | 0.777 |
| SIP-jmed-13B | 医学 | 0.820 / 0.819 / 0.817 | 0.819 ± 0.002 | 0.787 / 0.785 / 0.783 | 0.785 ± 0.002 | 0.771 |
| LLM-jp-3.1-13B (base of SIP-jmed) | 通用 | 0.816 / 0.817 / 0.817 | 0.817 ± 0.001 | 0.777 / 0.781 / 0.781 | 0.780 ± 0.002 | 0.765 |
| Qwen3-30B-A3B (MoE) | 通用 | 0.812 / 0.815 / 0.817 | 0.815 ± 0.002 | 0.774 / 0.781 / 0.782 | 0.779 ± 0.004 | 0.763 |
| LLM-jp-4-32B (MoE, reasoning) | 通用 | 0.813 / 0.820 / 0.819 | 0.817 ± 0.004 | 0.772 / 0.783 / 0.783 | 0.779 ± 0.006 | 0.765 |
| LLM-jp-1.8B | 通用 | 0.777 / 0.778 / 0.780 | 0.778 ± 0.002 | 0.730 / 0.730 / 0.733 | 0.731 ± 0.002 | 0.724 |

#### 表 C. 零样本：仅提示 vs. JSON Schema 约束解码 (relaxed / strict F1)

解析器 v1 = LAU 版解析器（第一个 `[` … 最后一个 `]`），v2 = 最后一个完整 JSON 数组（丢弃思考通道）。Fail：无法取出数组的句数。Unk.：标签不在 109 个之内的预测数。Trunc.：被生成上限（900 token，两个思考型为 3,072）截断的句数。

| 模型 | ZS F1 (v1) | Fail (v1) | Unk. (v1) | ZS F1 (v2) rel. / str. | Fail (v2) | Trunc. | ZS+约束 rel. / str. | P / R | Trunc. | FT F1 (s42) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 0.136 | 10 | 581 | 0.136 / 0.044 | 6 | 1 | 0.146 / 0.044 | 0.345 / 0.093 | 0 | 0.825 |
| Llama-3.3-70B | 0.276 | 0 | 188 | 0.276 / 0.160 | 0 | 2 | 0.265 / 0.157 | 0.508 / 0.179 | 3 | 0.823 |
| MedGemma-27B | 0.320 | 0 | 197 | 0.320 / 0.209 | 0 | 40 | 0.316 / 0.205 | 0.452 / 0.243 | 28 | 0.824 |
| SIP-jmed-13B | 0.010 | 322 | 47 | 0.025 / 0.014 | 92 | 147 | 0.015 / 0.003 | 0.043 / 0.009 | 108 | 0.820 |
| Qwen3-30B-A3B (MoE) | 0.235 | 3 | 236 | 0.235 / 0.139 | 3 | 24 | 0.224 / 0.133 | 0.332 / 0.169 | 46 | 0.812 |
| LLM-jp-4-32B (MoE, reasoning) | 0.092 | 261 | 95 | 0.150 / 0.068 | 17 | 35 | 0.000 / 0.000 | 0.000 / 0.000 | 1 | 0.813 |
| LLM-jp-1.8B | 0.027 | 3 | 195 | 0.027 / 0.001 | 3 | 427 | 0.019 / 0.001 | 0.044 / 0.013 | 343 | 0.777 |

#### 表 D. 思考型模型的四种零样本条件 (relaxed F1)

| 模型 | 仅提示 (v1) | 仅提示 (v2) | 整段约束 | 自由思考→约束答案 (rel. / str.) | 微调后 |
|---|:-:|:-:|:-:|:-:|:-:|
| LLM-jp-4-32B | 0.092 (fail 261) | 0.150 (fail 17) | 0.000 | 0.172 / 0.081 | 0.813 |
| SIP-jmed-13B | 0.010 (fail 322) | 0.025 (fail 92) | 0.015 | 0.048 / 0.023 | 0.820 |

#### 表 E. 参考模型 (ollama；无法本地微调)

| 模型 | 条件 | Relaxed F1 | Strict F1 | P / R | Fail | Trunc. |
|---|---|:-:|:-:|:-:|:-:|:-:|
| Qwen3.6-35B | 仅提示，关闭思考 | 0.300 | 0.153 | 0.439 / 0.228 | 0 | 3 |
| Qwen3.6-35B | 开启思考 + 答案加 schema（预算 3,072 token） | 0.022 | 0.017 | 0.561 / 0.011 | 310 | 310 |
| GPT-OSS-120B | 仅提示（思考无法关闭） | 0.265 | 0.131 | 0.571 / 0.173 | 8 | 16 |
| GPT-OSS-120B | 思考 + 答案加 schema | 0.263 | 0.135 | 0.562 / 0.172 | 11 | 11 |

#### 表 F. 医学模型 − 其通用基座，同 seed (配对 bootstrap, relaxed F1)

| 配对 | Seed 42 | Seed 1 | Seed 2 |
|---|:-:|:-:|:-:|
| SIP-jmed-13B − LLM-jp-3.1-13B | +0.004 [+0.000, +0.009] | +0.002 [-0.001, +0.006] | +0.000 [-0.007, +0.006] |
| MedGemma-27B − Gemma-3-27B | -0.005 [-0.010, +0.000] | +0.004 [-0.002, +0.010] | +0.001 [-0.002, +0.005] |
| Med42-70B − Llama-3.3-70B | +0.002 [-0.002, +0.006] | -0.001 [-0.005, +0.003] | -0.001 [-0.005, +0.003] |

#### 表 G. 微调 LLM − BERT，seed 42 (配对 bootstrap, relaxed F1)

| LLM | vs. UTH-BERT | vs. NICT-BERT |
|---|:-:|:-:|
| Med42-70B | +0.074 [+0.045, +0.108] | +0.084 [+0.056, +0.118] |
| Llama-3.3-70B | +0.072 [+0.041, +0.108] | +0.082 [+0.052, +0.118] |
| MedGemma-27B | +0.072 [+0.042, +0.109] | +0.083 [+0.054, +0.118] |
| SIP-jmed-13B | +0.069 [+0.040, +0.103] | +0.080 [+0.051, +0.113] |
| Qwen3-30B-A3B (MoE) | +0.061 [+0.029, +0.097] | +0.071 [+0.041, +0.107] |
| LLM-jp-4-32B (MoE, reasoning) | +0.061 [+0.030, +0.098] | +0.072 [+0.041, +0.107] |
| LLM-jp-1.8B | +0.026 [-0.004, +0.061] | +0.036 [+0.007, +0.070] |

#### 表 H. 900 token 生成上限的影响 (微调后, seed 42, relaxed)

Trunc.：输出被截断的句数。Gold：这些句子所含正解的比例。R：截断句 / 其余句的召回。FN：该模型的 FN 落在截断句中的比例。UTH-BERT 按 Med42-70B 的截断句划分评价。

| 模型 | Trunc. | 正解比例 | R (截断) | R (其余) | P (截断) | P (其余) | FN 比例 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 87 | 0.53 | 0.583 | 0.927 | 0.908 | 0.937 | 0.87 |
| Llama-3.3-70B | 86 | 0.53 | 0.580 | 0.925 | 0.909 | 0.930 | 0.86 |
| MedGemma-27B | 81 | 0.51 | 0.577 | 0.920 | 0.901 | 0.930 | 0.84 |
| Gemma-3-27B | 84 | 0.52 | 0.587 | 0.927 | 0.905 | 0.936 | 0.86 |
| SIP-jmed-13B | 84 | 0.52 | 0.578 | 0.916 | 0.904 | 0.929 | 0.84 |
| LLM-jp-3.1-13B | 82 | 0.51 | 0.572 | 0.909 | 0.896 | 0.925 | 0.83 |
| Qwen3-30B-A3B | 85 | 0.52 | 0.571 | 0.907 | 0.892 | 0.926 | 0.83 |
| LLM-jp-4-32B | 79 | 0.50 | 0.571 | 0.903 | 0.886 | 0.916 | 0.81 |
| LLM-jp-1.8B | 81 | 0.50 | 0.525 | 0.867 | 0.852 | 0.898 | 0.78 |
| UTH-BERT（与 Med42 相同划分） | (87) | 0.53 | 0.486 | 0.856 | 0.858 | 0.879 | – |

#### 表 I. 微调后 Med42-70B (seed 42) 九个标签的逐标签成绩；UTH = UTH-BERT 的 relaxed / strict F1

| 标签 | n | P | R | F1 (relaxed) | F1 (strict) | UTH (rel. / str.) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| state-patient | 2318 | 0.92 | 0.80 | 0.85 | 0.80 | 0.72 / 0.65 |
| clinical_test | 587 | 0.95 | 0.77 | 0.85 | 0.80 | 0.71 / 0.64 |
| treatment | 614 | 0.93 | 0.67 | 0.78 | 0.73 | 0.71 / 0.66 |
| drug | 283 | 0.96 | 0.68 | 0.80 | 0.79 | 0.70 / 0.67 |
| PN-Negative-patient | 276 | 0.96 | 0.67 | 0.79 | 0.76 | 0.86 / 0.84 |
| state-others | 155 | 0.89 | 0.33 | 0.48 | 0.44 | 0.41 / 0.37 |
| device | 75 | 0.73 | 0.59 | 0.65 | 0.53 | 0.46 / 0.46 |
| time_span | 59 | 0.74 | 0.47 | 0.58 | 0.58 | 0.54 / 0.49 |
| value-others | 63 | 0.32 | 0.10 | 0.15 | 0.15 | 0.07 / 0.07 |

标签频率与 relaxed F1 的 Spearman 相关 (Med42-70B, 98 labels): ρ = 0.59 (p = 2e-10). 27 个 `-others` 标签的平均 F1 0.11.


</details>

---

## 仓库结构

```
bert/                     BERT-CRF NER 管线（系统 Python：torch 2.0, transformers 4.46）
  NER_training.py           用 CoNLL 数据微调 BERT-CRF（早停 + 指标）
  build_conll.py            iCorpus 字符级 JSON -> CoNLL CSV（各模型分词器）
  extract_structured.py     原文 -> MeCab -> BERT-CRF -> 结构化实体
  predict_reports.py        真实报告批量预测（cp932 CSV）
  evaluate_ner.py           从预测 CSV 算 strict/soft 的 span 级 P/R/F1
  summarize_results.py      各配置对比
  preprocess_text.py        文本归一化（neologdn / NFKC / 全角化）
  tokenization_mod.py       MeCab + WordPiece 分词器（UTH-BERT 官方）
  lib/                      模型（BERT_CRF、CRF）、训练循环、utils、评测
llm/                      LLM 抽取与评测（venv：torch 2.6+cu124）
  testset_eval.py           gold 语料上的 BERT vs LLM 评测（宽松指标）
  llm_extract_vllm.py       vLLM 批量抽取（HF 模型：LLM-jp, SIP-jmed）
  llm_extract_ollama.py     ollama 抽取，多 GPU 数据并行（Qwen, Llama, GPT-OSS）
  llm_extract.py            transformers 单条基线
results/                  仅聚合指标（无文本、无患者数据）
  testset_scores.json       BERT vs LLM 对比（本仓库核心结果）
  bert_UTH/ bert_NICT/      各标签 strict/soft P/R/F1 + 训练指标
docs/WORKFLOW.md          详细操作记录
docs/RESULTS.md           LAU-16（单 seed）旧结果（存档，已被 journal/ 取代）
journal/                  期刊版：代码、run 元数据、统计、图（见 journal/README.md）
```

---

## 方法

### BERT（微调）

`BERT-CRF` = 预训练 BERT 编码器 + 线性头 + CRF。两个编码器：

- **UTH-BERT** — 日语临床文本预训练（词表 25k；MeCab ipadic-neologd + 万病辞書
  (J-Medic) + 数字拆分 + WordPiece）。
- **NICT-BERT** — 日语维基百科预训练（词表 32k；MeCab Juman 词典 + WordPiece）。

用 `difflib` 把字符 span 对齐到子词，将字符级实体标注转成 CoNLL
（[`bert/build_conll.py`](bert/build_conll.py)）；实体覆盖率 95.7%(UTH) /
99.7%(NICT)。微调用 AdamW、按验证 F1 早停。

> **CRF 的坑：** CRF 对非法 BIO 转移（`O→I-x`、`B-x→I-y` …）施加 −1e7 惩罚。喂进去
> 的 IOB 必须是严格合法的 BIO，否则损失会爆炸（到约 1e8）而 F1 看起来正常。
> `build_conll.py` 里含 BIO 修复步骤。

### LLM（零样本与微调）

把每句放进一个列出 109 个标签、要求输出 JSON 数组 `[{"label","text"}]` 的提示
（`journal/code/eval_v2.py`，vLLM，贪心解码）。条件：仅提示；JSON Schema 约束解码（`--guided`，XGrammar，
标签限制在体系内）；思考型模型两段式"自由思考→约束答案"（`--think_from`）；QLoRA 微调
（`journal/code/qlora_train_v2.py`：NF4、LoRA r=16、3 epoch、seed 42/1/2）后以 adapter 附加或合并推理。
两个参考模型（Qwen3.6-35B、GPT-OSS-120B）用 ollama 运行。

### 评测

BERT 预测和 LLM 输出都转成每句的 `(类型, 表层)`，由 `journal/code/metrics_v2.py` 打分：确定性一对一贪心配对；
**relaxed**（类型相同且一方表层是另一方子串，NFKC）与 **strict**（表层完全一致）F1，以及最小长度、主体合并、
宏平均变体；文档级 bootstrap 置信区间与配对差。`llm/testset_eval.py` 是 LAU-16 版打分器（仅 relaxed，依赖集合遍历顺序）。

---

## 数据

**不包含。** 复现需要：

- **iCorpus（症例報告コーパス）** — 标注的症例报告语料（183 篇，字符级实体）。
  仅限研究许可；从东京大学（医療AI開発学講座）获取，放在
  `corpus/icorpus_.../data/json/`。
- **UTH-BERT** — https://ai-health.m.u-tokyo.ac.jp/home/research/uth-bert
- **NICT-BERT** — https://alaginrc.nict.go.jp/nict-bert/
- **万病辞書**（J-Medic） — UTH 分词用。
- 抽取步骤用的真实临床报告（你自己的患者数据）。

---

## 环境

```bash
# BERT 管线（系统 Python）
pip install torch==2.0.1 transformers==4.46.3 seqeval pandas mecab-python3 jaconv neologdn
# + MeCab 词典：mecab-ipadic-neologd(UTH), mecab-jumandic-utf8(NICT)

# LLM 评测（独立 venv）
pip install "vllm==0.8.5" "transformers==4.51.3" pandas requests
# + ollama（用于 Qwen/Llama/GPT-OSS）
```

### 运行

```bash
# 1) 从语料构建训练数据
python bert/build_conll.py

# 2) 微调（按编码器）
python bert/NER_training.py --bert_type UTH  --bert_path <UTH-BERT>  --data_path data/csv/icorpus_UTH.csv  --patience 15
python bert/NER_training.py --bert_type NICT --bert_path <NICT-BERT> --data_path data/csv/icorpus_NICT.csv --patience 15

# 3) BERT vs LLM 精度对比
python llm/testset_eval.py                                   # BERT 基线
python llm/testset_eval.py --model qwen3.6:35b --engine ollama --hosts <urls>
python llm/testset_eval.py --model <hf-model> --engine vllm --tp 2

# 4) 从真实报告抽取
python bert/predict_reports.py --bert_type UTH --model_path <model.pt> --data_dir <reports> --out_dir <out>
```

完整操作细节（多 GPU ollama 搭建、vLLM/推理模型处理）见
[`docs/WORKFLOW.md`](docs/WORKFLOW.md)。

---

## 许可

代码：MIT。**模型、词典、语料各有自己的许可，不包含在本仓库内。** 临床/患者数据
按设计排除。
