# 日语临床 NER：微调 BERT vs. 零样本 LLM

[English](README.md) | [日本語](README.ja.md) | **中文**

对日语临床文本做命名实体识别（NER），在同一份 gold 标注语料上、用统一指标，
比较**微调后的 BERT 编码器**与**零样本大语言模型（LLM）**。

本项目有三个目标：

1. **精度对比** — 在公开标注语料上测量 BERT（UTH-BERT、NICT-BERT）与 LLM
   （LLM-jp、GPT-OSS、SIP-jmed-LLM、Qwen、Llama）的 NER 精度。
2. **真实数据抽取** — 用同一套管线抽取真实临床报告。
3. **FHIR 抽取**（计划中） — 从临床报告中抽取可映射到 FHIR 的项目。

> ⚠️ **本仓库不包含任何临床数据。** 训练/评测语料（iCorpus / 症例報告コーパス）
> 采用仅限研究、禁止再分发的许可，真实临床报告含患者数据。本仓库**只有代码和
> 聚合指标（数字与标签名，无任何文本）**。见 [数据](#数据)。

---

## 期刊版（2026 年 10 月）：追加实验

下面的 LAU-16 研究已扩展为期刊投稿（MDPI *Machine Learning and Knowledge Extraction*，特刊 "Language
Acquisition and Understanding"）。扩展部分全部在 [`journal/`](journal/)（代码、run 元数据、全部统计、图），
完整对照表见 [`journal/stats/tables.md`](journal/stats/tables.md)，带术语说明和考察的版本见
[`journal/stats/tables_annotated.md`](journal/stats/tables_annotated.md)。

新增内容：确定性配对规则（旧贪心法依赖 Python 的 hash seed）、在 relaxed 之外报告 **strict（完全一致）F1**、
文档级 bootstrap 置信区间与配对检验、**每个模型 3 个 QLoRA seed**、SIP-jmed-13B 和 MedGemma-27B 的**基座模型**
（LLM-jp-3.1-13B、Gemma-3-27B）、**JSON Schema 约束解码**和思考型模型的两段式"自由思考→约束答案"、修正后的
JSON 解析器、两个参考模型（Qwen3.6-35B、GPT-OSS-120B）。

主要结果（test-36，3 seed 均值）：13B 以上的所有模型收敛到 relaxed F1 0.815–0.827 / strict 0.779–0.798，
全部高于 BERT（0.751 / 0.688）。零样本在约束解码下 F1 几乎不变（如 MedGemma 0.320→0.316），思考型模型
"66–74% parse 失败"大部分是旧解析器的缺陷。三对基座对齐比较（13B/27B/70B）× 3 seed 检测不到医学预训练的
效果（配对差都在 ±0.005 内，区间含 0），LAU 版"医学预训练可替代规模"的推测予以撤回。微调后的漏检集中在被
900 token 上限截断的约 20% 句子（含 52% 的正解实体）。

### 期刊版：完整对照表

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

## 主要结果

在语料测试集（36 篇留出文档）上的实体级 NER 精度，用**宽松匹配**（同一实体类型
+ 表层字符串包含，NFKC 归一化）打分。所有方法用完全相同的指标，因此 BERT 和 LLM
可直接对比。微调 BERT 只在留出测试集上评测（无数据泄漏）；零样本 LLM 在全部
183 篇上运行，取其中测试集 36 篇子集做正面对比。

> ⚠️ 数字为**修正 gold（`end+1`）后**的结果（详见 [docs/RESULTS.md](docs/RESULTS.md)）。

| 方法 | 类型 | 测试36篇 F1 | Precision | Recall |
|------|------|:----------:|:---------:|:------:|
| **UTH-BERT**（微调） | 编码器，临床预训练 | **0.752** | 0.871 | 0.661 |
| **NICT-BERT**（微调） | 编码器，维基百科预训练 | **0.742** | 0.861 | 0.651 |
| Qwen3.6 35B | LLM，零样本 | 0.288 | 0.446 | 0.212 |
| Llama3.3 70B | LLM，零样本 | 0.261 | 0.522 | 0.174 |
| Qwen3-30B-A3B | LLM，零样本 | 0.244 | 0.316 | 0.199 |
| LLM-jp-4 32B | LLM，零样本（推理型） | 0.068 | 0.324 | 0.038 |
| GPT-OSS 120B | LLM，零样本（推理型） | 0.040 | 0.606 | 0.021 |
| SIP-jmed-LLM-3 13B | LLM，零样本（推理型·**医学**） | 0.012 | 0.119 | 0.006 |

指标：宽松匹配（类型 + 表层包含，NFKC）。全部 test36、同一修正 gold。LLM 零样本
（推理模型开启思考、给大 token 预算 — 见[说明](#关于-llm-数字的说明)）。
完整数字：[`results/testset_scores.json`](results/testset_scores.json)。

### 微调 LLM（QLoRA）——排名彻底反转

把同样这些 LLM 用**和 BERT 相同的 146 篇训练集**做 QLoRA 微调（SFT：prompt→gold
JSON），再用同一指标评测：

| 方法 | 零样本 F1 | **微调后 F1** | Δ |
|------|:--------:|:------------:|:--:|
| **Llama-3.3 70B** | 0.261 | **0.822** *(最高)* | +0.561 |
| **SIP-jmed-LLM-3 13B**（医学·仅13B）| 0.012 *(零样本最差)* | **0.819** | **+0.807** |
| **Qwen3-30B-A3B** | 0.244 | **0.812** | +0.568 |
| **LLM-jp-4 32B** | 0.068 | **0.811** | +0.743 |
| **LLM-jp-3.1 1.8B** | ≈0 | **0.777** | +0.78 |
| UTH-BERT（微调编码器）| — | 0.752 | — |
| NICT-BERT（微调编码器）| — | 0.742 | — |

**微调让整个排名反转。** 零样本时 BERT（0.752）超过所有 LLM、医学模型最差（0.012）；
QLoRA 微调后**全部 5 个 LLM（0.777–0.822）反超 BERT**。哪怕 1.8B 的小模型（零样本≈0）也到
0.777。零样本的差距是**任务/格式适配问题，不是能力问题**——一旦模型学会产出结构化输出，
LLM 更丰富的表示（加上医学模型的领域知识）就超过了 BERT。零样本时是*负担*的领域预训练
（吐不出 JSON 的推理模型），微调后变成决定性*优势*：**13B 的医学 SIP-jmed（0.819）几乎
追平 70B 的通用 Llama-3.3（0.822）**——领域预训练让 13B 抵得上 70B。

方法：QLoRA（4-bit NF4 + LoRA r=16，attention+MLP），3 epoch，单卡。
见 [`llm/qlora_train.py`](llm/qlora_train.py)、[`llm/build_sft_data.py`](llm/build_sft_data.py)。

### 结论（零样本）

- **微调 BERT 远超所有零样本 LLM**（约 0.75 vs. ≤0.29，2.5 倍以上差距）。在这种
  细粒度、领域专用的 NER 任务上，任务能力几乎完全来自对 100 多类标注体系的微调；
  通用 LLM 零样本做不到。
- **LLM 召回很低。** 语料标注极其详尽（"網羅的"），而 LLM 抽取没那么密，也难以
  套用 100 多个陌生标签名（最好的 LLM 也只有 R≈0.19–0.23，远低于 BERT 的约 0.66）。
- **医学专用模型反而最差。** SIP-jmed-LLM-3（13B，日语临床）只有 0.012，是所有 LLM
  里最低的——因为它是*推理调优*模型，会无止境地推理，**约 74% 的输入吐不出可解析
  JSON**。模型无法产出结构化输出时，领域知识帮不上忙。（但微调后它一跃成为并列最好。）
- **「思考」型 LLM 不适合结构化抽取**，全部聚在底部（SIP-jmed 0.012、GPT-OSS 0.040、
  LLM-jp-4 0.068）。强制立即输出 JSON（guided decoding）会丢掉推理、使输出退化
  （如 LLM-jp-4 降到 0.049）；让它们自由推理又会在 JSON 前超出 token 上限。直接
  给答案的模型（Qwen/Llama 关掉思考）好得多（0.27–0.29）。
- **UTH-BERT（临床预训练） > NICT-BERT（维基百科预训练）**，符合预期。NICT 的词表
  缺很多临床汉字（会变成 `[UNK]`），是它用于临床文本的实际短板。

### 关于 LLM 数字的说明

- Qwen / Llama / GPT-OSS 通过 **ollama 关闭思考**（`think:false`）运行。
- LLM-jp-4 和 SIP-jmed 是推理模型、没有可靠的关思考开关，用 **vLLM 自然推理模式**、
  大 token 预算（最多 12k）运行。SIP-jmed 仍有 75% 输入被截断，加大预算也不改变分数
  （6k 是 0.018 ≈ 12k 也是 0.018）。
- LLM-jp-4 的 12k token 运行中途遇到 vLLM detokenizer 崩溃；报告值 0.094 是它完整
  跑完的 6k token 自然推理版。

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

### LLM（零样本）

把每个临床句放进一个列出全部标签的 prompt，要求输出 JSON 数组
`[{"label","text"}]`。两个后端：

- **ollama**（Qwen3.6-35B、Llama3.3-70B、GPT-OSS-120B） — 用 `think:false` 关思考；
  为提升吞吐做**4-GPU 数据并行**（每张卡一个实例、请求轮询分发）；上下文限制到
  4096 以省 KV 缓存。
- **vLLM**（LLM-jp-4-32B、SIP-jmed-13B） — HF 模型；对推理模型用 guided JSON
  解码或大 token 预算的自然推理。

### 统一评测

BERT 的预测和 LLM 的输出都对 gold 字符级标注用**同一个宽松指标**打分
（`llm/testset_eval.py`）。实体 = `(类型, 表层)`；类型相同且一方表层包含另一方
（NFKC 归一化）即算命中。这对基于 token 的 BERT 和自由文本的 LLM 都公平，也能容忍
语料中碎片式的 gold span。

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
