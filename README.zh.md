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

| 方法 | 类型 | 测试36篇 F1 | Precision | Recall |
|------|------|:----------:|:---------:|:------:|
| **UTH-BERT**（微调） | 编码器，临床预训练 | **0.726** | 0.796 | 0.669 |
| **NICT-BERT**（微调） | 编码器，维基百科预训练 | **0.703** | 0.763 | 0.653 |
| Qwen3.6 35B | LLM，零样本 | 0.292 | 0.415 | 0.225 |
| Llama3.3 70B | LLM，零样本 | 0.271 | 0.493 | 0.187 |
| LLM-jp-4 32B（自然推理） | LLM，零样本 | 0.094 | 0.351 | 0.055 |
| LLM-jp-4 32B（强制 JSON） | LLM，零样本 | 0.049 | 0.140 | 0.030 |
| SIP-jmed-LLM-3 13B | LLM，零样本 | *(汇总中)* | | |
| GPT-OSS 120B | LLM，零样本 | *(汇总中)* | | |

> 快照 — 写本 README 时，SIP-jmed-LLM、GPT-OSS 以及 LLM-jp-4 的加长 token 版本仍在
> 运行；最新数字见 [`results/testset_scores.json`](results/testset_scores.json)。

### 结论

- **微调 BERT 远超零样本 LLM**（约 0.73 vs. ≤0.29）。在这种细粒度、领域专用的 NER
  任务上，任务能力几乎完全来自对 100 多类标注体系的微调；通用 LLM 零样本做不到。
- **LLM 精度尚可但召回很低**（P≈0.4–0.5，R≈0.19–0.23）。语料标注极其详尽
  （"網羅的"），而 LLM 抽取没那么密，也难以套用 100 多个陌生的标签名。
- **UTH-BERT（临床预训练） > NICT-BERT（维基百科预训练）**，符合预期。NICT 的词表
  缺很多临床汉字（会变成 `[UNK]`），这是它用于临床文本的实际短板。
- **「思考」型 LLM 不适合结构化抽取。** 强制立即输出 JSON（guided decoding）会丢掉
  它们的推理优势、使输出退化；让它们自由推理又很慢、常在吐出可解析 JSON 前就
  超出 token 上限。关掉思考的普通输出在本任务上反而更好。

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
