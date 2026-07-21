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
| 实体标注 | 字符级，**113 类**实体类型（`end+1` 修正后，含之前被丢弃的单字实体类型）；总实体 **73797** |
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

**评测基础设施**（`testset_eval.py`，均 test36、同一宽松指标）：
- **dense 模型**（SIP-jmed-13B、1.8B）：vLLM tp=1 + `--lora`（Llama 架构支持 LoRA）
- **MoE 模型**（Qwen3-30B、LLM-jp-4-32B，架构=Qwen3-MoE）：vLLM **不支持 MoE 的 LoRA**，
  故先把 LoRA **合并进 base**（`merge_lora.py`）再用 **vLLM V0 引擎 tp=2** 服务
- **Llama-3.3-70B-ft**：dense，vLLM **V0 tp=4** + `--lora`（4 卡放 bf16 70B）
- **零样本**：ollama（qwen3.6/llama3.3/gpt-oss，`think:false`）、vLLM V0 tp=2（MoE）
- 关键坑：vLLM **V1 引擎 tp≥2 有 triton 缓存竞争**（`FileExistsError`）→ 用 **V0 引擎**（`VLLM_USE_V1=0`）+ `enforce_eager` 规避

**踩坑**：①2 卡 device_map="auto" 训练会 CUDA launch failure → 单卡；②32B MoE 4-bit 极慢
（内存带宽瓶颈，batch=1≈31h，batch=2≈16h）；③epoch 边界的训练内验证 OOM → 关掉。

---

## 3. 结果对比表（测试集 36 篇 = 433 句，宽松匹配 F1，修正 gold）

> ⚠️ **重要修正（2025 更新）**：发现 iCorpus 的实体 `end` 偏移是**闭区间**，原管线误当开区间，
> 导致①每个实体少最后一字（症例→症）②`end==start` 的 8466 个单字实体（占 11.5%）被全部丢弃。
> 已全量修复（`end+1`）→ 重建数据（实体 65331→**73797**，标签 109→**113 类**）→ 重训 BERT +
> 重跑全部微调 + 重评。下表**全部为修正 gold、同一 test36** 的结果。修正对宽松匹配分数影响很小
> （BERT ±0.01），但抽取产物、单字实体、训练信号都更正确。详见 [WORKFLOW.md](WORKFLOW.md)。

### 零样本（全部很低——推理模型吐不出 JSON）

| 方法 | F1 | Precision | Recall | parse失败 |
|------|:--:|:---------:|:------:|:--------:|
| **UTH-BERT**（微调编码器） | **0.752** | 0.871 | 0.661 | — |
| **NICT-BERT**（微调编码器） | **0.742** | 0.861 | 0.651 | — |
| Qwen3.6-35B (ollama) | 0.288 | 0.446 | 0.212 | 0 |
| Llama3.3-70B (ollama) | 0.261 | 0.522 | 0.174 | 0 |
| Qwen3-30B-A3B (vLLM) | 0.244 | 0.316 | 0.199 | 1 |
| LLM-jp-4-32B | 0.068 | 0.324 | 0.038 | 66% |
| GPT-OSS-120B | 0.040 | 0.606 | 0.021 | 59% |
| SIP-jmed-13B（医学） | 0.012 | 0.119 | 0.006 | 74% |

### QLoRA 微调后（全部反超 BERT）

| 方法 | 零样本 F1 | **微调 F1** | P | R | parse失败 |
|------|:--------:|:----------:|:---:|:---:|:--------:|
| **Llama-3.3-70B** | 0.261 | **0.822（最高）** | 0.921 | 0.743 | 0 |
| **SIP-jmed-13B**（医学·仅13B） | 0.012（零样本最差） | **0.819** | 0.918 | 0.740 | 0 |
| **Qwen3-30B-A3B** | 0.244 | **0.812** | 0.910 | 0.733 | 0 |
| **LLM-jp-4-32B** | 0.068 | **0.811** | 0.902 | 0.737 | 0 |
| **LLM-jp-1.8B**（3 epoch） | ≈0 | **0.777** | 0.880 | 0.695 | 2 |
| UTH-BERT | — | 0.752 | — | — | — |
| NICT-BERT | — | 0.742 | — | — | — |

**核心结论（修正数据上依然成立）**：微调让排名完全反转。零样本 BERT（0.752）≫ 所有 LLM、
医学模型最差（0.012）；QLoRA 后**全部 5 个微调 LLM（0.777–0.822）＞ BERT（0.752）**。
零样本差距是「格式/任务适配」问题，不是「能力」问题。
**最亮眼**：13B 的医学 SIP-jmed（0.819）几乎追平 70B 通用 Llama-3.3（0.822）——
领域预训练让 13B 抵得上 70B。

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
