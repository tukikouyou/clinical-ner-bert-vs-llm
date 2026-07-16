# 详细结果与实验参数

**中文** | [日本語](RESULTS.ja.md)

日语临床 NER：微调 BERT vs. 零样本/微调 LLM。截至 2026-07-16 的完整记录。
（部分微调模型仍在训练，见表内标注。）

---

## 1. 任务设置

| 项 | 值 |
|----|----|
| 语料 | iCorpus 症例報告コーパス（20220531 版），183 篇难病症例报告 |
| 训练/测试划分 | `train_test_index_1fold.json`：146 篇训练 / 36 篇测试（BERT 从训练里再切验证） |
| 实体标注 | 字符级，98 个有表层的实体类型（含 section/modality 等共约 109 类标签） |
| 统一评测指标 | **宽松实体匹配 F1**：`(类型一致) 且 (一方表层是另一方子串, NFKC 归一化)`，贪婪 1-1 匹配 |
| BERT 评测范围 | 36 篇留出测试集（无泄漏） |
| LLM 评测范围 | 全 183 篇（零样本对任何篇都是未见）；取 36 篇测试子集与 BERT 正面对比 |

> 为什么用宽松匹配：corpus gold 是碎片式标注（如 `慢性閉塞性肺疾患`→`慢`/`閉塞`/`疾`），
> BERT（token 边界）、LLM（自由文本）粒度都不同，宽松包含匹配对三方公平、且分词无关。

---

## 2. 各方法详细参数

### 2.1 BERT（微调）— `bert/`

**架构**：预训练 BERT 编码器 + Dropout(0.5) + Linear(768→标签数) + CRF。
CRF 对非法 BIO 转移施加 −1e7 硬惩罚（`torchcrf_mod.py`）。

| | UTH-BERT | NICT-BERT |
|---|---|---|
| 预训练 | 临床文本 | 日语维基百科 |
| 词表 | 25,000 | 32,016 |
| 分词 | preprocess(neologdn/NFKC/全角化) + MeCab(ipadic-neologd + 万病辞書) + **数字逐位拆分** + WordPiece | h2z(全角化) + MeCab(juman 词典) + WordPiece |
| 标签数 | 193 | 188 |
| best epoch | 64 | 47 |
| 训练耗时 | 8322 s | 6591 s |

**训练超参**（`NER_training.py`，两模型相同）：
- 优化器 AdamW，学习率 **2e-5**，linear warmup（总步数的 10%）
- batch_size **1**（模型 `embed[:,1:-1]` 去 CLS/SEP 只支持 bs=1）
- max_epoch 250，**早停 patience 15**（按验证集 strict micro-F1）
- CRF 损失 reduction="sum"
- 数据转换：`build_conll.py`（difflib 字符偏移对齐 + BIO 修复），实体覆盖 UTH 95.7% / NICT 99.7%

**原生 strict span F1**（论文常报的指标，非宽松匹配）：UTH 0.895 / NICT 0.888；soft：0.930 / 0.925。

### 2.2 零样本 LLM — `llm/`

**Prompt**（`testset_eval.py` 内 `PROMPT_TEMPLATE`）：列出全部标签 + 临床句，要求只输出
JSON 数组 `[{"label":…,"text":…}]`。temperature=0（贪婪）。

| 引擎 | 模型 | 关键设置 |
|------|------|---------|
| ollama | qwen3.6:35b, llama3.3:70b, gpt-oss:120b | `think:false`（关思考）, `num_ctx=4096`, `num_predict=1024`；qwen/llama 用 **4-GPU 数据并行**（每卡一个 ollama 实例，端口 11434-11437，请求轮询），gpt-oss 单实例 4 卡 |
| vLLM | llm-jp-4-32b, SIP-jmed-13b | 推理模型无可靠关思考开关；试了 ①guided JSON（schema+minLength）②预填 `<think></think>` ③自然推理（max_tokens 最大 12000）。报告值取最好的自然推理版 |

**踩坑**：qwen3.6/llm-jp-4/gpt-oss/SIP-jmed 都是「思考型」模型；关不掉思考时会一直推理、
在 max_tokens 内吐不出 JSON（parse 失败率见结果表）。

### 2.3 微调 LLM（QLoRA）— `llm/qlora_train.py` + `llm/build_sft_data.py`

**SFT 数据**：146 篇训练文档 → **1672 训练 / 88 验证** 对（`prompt → gold JSON`）。
gold 用字符级原始表层，和 BERT 训练同源。平均每例 29.4 个实体。

**QLoRA 配置**：
- 量化：4-bit **NF4**，double quant，compute dtype bf16
- LoRA：**r=16, alpha=32, dropout=0.05**，target=`[q,k,v,o,gate,up,down]_proj`，bias=none
- 优化：AdamW（SFTTrainer），学习率 **2e-4**，cosine 调度，warmup 0.03，**3 epochs**
- gradient checkpointing 开，bf16，单卡 `device_map={"":0}`
- 训练中验证关闭（epoch 边界会 OOM），改测试集单独评测
- batch / grad_accum：1.8b & SIP-jmed = **1 / 8**；llm-jp-4(MoE) = **2 / 4**（大 batch 摊薄 MoE 权重加载）
- max_seq_len：4096（1.8b, SIP-jmed）/ 2048（llm-jp-4）

**环境**：
- `/opt/llm/ft-venv`：transformers 5.13 + trl 0.14 + peft 0.19 + bnb 0.49（标准模型：1.8b, SIP-jmed）
- `/opt/llm/ft-venv-t451`：transformers 4.51 + trl 0.13 + peft 0.14（自定义 MoE：llm-jp-4，5.x 会崩权重转换）

**评测**：base + LoRA 经 vLLM（`testset_eval.py --lora <adapter>`），同一宽松指标。

**踩坑**：①2 卡 device_map="auto" 训练会 CUDA launch failure → 单卡；②32B MoE 4-bit 极慢
（内存带宽瓶颈，batch=1≈31h，batch=2≈16h）；③epoch 边界的训练内验证 OOM → 关掉。

---

## 3. 结果对比表（测试集 36 篇，宽松匹配 F1）

### 零样本

| 方法 | F1 | Precision | Recall | parse失败 | 耗时 |
|------|:--:|:---------:|:------:|:--------:|:----:|
| **UTH-BERT**（微调编码器） | **0.726** | 0.795 | 0.668 | — | — |
| **NICT-BERT**（微调编码器） | **0.702** | 0.763 | 0.650 | — | — |
| Qwen3.6-35B | 0.292 | 0.415 | 0.225 | ~0 | 69 min |
| Llama3.3-70B | 0.271 | 0.493 | 0.187 | ~0 | 187 min |
| LLM-jp-4-32B（自然推理） | 0.094 | 0.351 | 0.055 | 63% | — |
| GPT-OSS-120B | 0.052 | 0.584 | 0.027 | 54% | — |
| SIP-jmed-13B（自然推理） | 0.018 | 0.066 | 0.011 | 75% | — |

### QLoRA 微调后

| 方法 | 零样本 F1 | **微调 F1** | P | R | parse失败 |
|------|:--------:|:----------:|:---:|:---:|:--------:|
| **SIP-jmed-13B**（医学） | 0.018（最差） | **0.857（最好）** | 0.881 | 0.834 | 0.2% |
| **LLM-jp-1.8B**（3 epoch） | ≈0 | **0.792** | 0.840 | 0.748 | 1.4% |
| LLM-jp-1.8B（1 epoch，对照） | ≈0 | 0.625 | 0.687 | 0.573 | 1.3% |
| UTH-BERT | — | 0.726 | — | — | — |
| NICT-BERT | — | 0.702 | — | — | — |
| LLM-jp-4-32B | 0.094 | *训练中* | | | |
| Qwen3.6 / GPT-OSS / Llama | — | *待训练* | | | |

**核心结论**：微调让排名完全反转。零样本 BERT ≫ LLM、医学模型最差；QLoRA 后 LLM ＞ BERT、
医学模型最好（0.857 vs 0.726）。零样本差距是「格式/任务适配」问题，不是「能力」问题。

---

## 4. 结果存储位置

| 内容 | 路径 |
|------|------|
| **全部宽松匹配分数**（BERT+全部LLM+微调） | `llm/results/ehr/testset_scores.json`（仓库 `results/testset_scores.json` 同步） |
| BERT 各标签 strict/soft P/R/F1 + 训练指标 | `code/results/NER/{UTH,NICT}/NER/{metrics,NER_strict_RESULT,NER_soft_RESULT}_0.json` |
| BERT loss 曲线 | `code/results/NER/{UTH,NICT}/loss_NER_0.json` |
| BERT 模型权重 | `code/models/NER/{UTH,NICT}/NER/ner_model_0.pt` |
| **LoRA 适配器**（微调产物） | `llm/ft/{llmjp-1.8b-3ep, sip-jmed-13b, llmjp-4-32b}/adapter_model.safetensors` |
| SFT 训练数据 | `llm/sft_data/{sft_train,sft_val}.jsonl` |
| 真实病历抽取（BERT） | `code/results/predict/{UTH,NICT}/entities.csv` + `stats.json` |
| 真实病历抽取（LLM，全183篇 gold 评测的产物） | `llm/results/ehr/{qwen35b,llama70b}/entities.csv` |
| 训练日志 | `scratchpad/ft_*.log`（微调）、`code/logs/train_*_run.log`（BERT） |
| 评测日志 | `scratchpad/eval_*.log` |

> ⚠️ `entities.csv` 和 corpus 含患者/受限数据，**不上传 GitHub**；仓库只放 `testset_scores.json`
> 和 BERT 指标 JSON（纯数字/标签名）。

---

## 5. 复现命令速查

```bash
# BERT: 建数据→训练→评测
python bert/build_conll.py
python bert/NER_training.py --bert_type UTH --bert_path <UTH-BERT> --data_path data/csv/icorpus_UTH.csv --patience 15

# 零样本 LLM 评测
python llm/testset_eval.py --model qwen3.6:35b --engine ollama --hosts <urls>
python llm/testset_eval.py --model <hf> --engine vllm --tp 1        # BERT 基线每次自动重算

# QLoRA 微调 + 评测
python llm/build_sft_data.py --out_dir ./sft_data
python llm/qlora_train.py --model <hf> --out_dir ./ft/<name> --epochs 3 --lr 2e-4
python llm/testset_eval.py --model <base> --lora ./ft/<name> --engine vllm --tp 1
```
