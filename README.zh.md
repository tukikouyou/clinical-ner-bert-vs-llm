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
