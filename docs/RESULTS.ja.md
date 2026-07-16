# 詳細結果と実験パラメータ

[中文](RESULTS.md) | **日本語**

日本語臨床NER：ファインチューニングBERT vs. ゼロショット/ファインチューニングLLM。
2026-07-16 時点の完全な記録。（一部のFTモデルは学習中。表内に注記。）

---

## 1. タスク設定

| 項目 | 値 |
|------|----|
| コーパス | iCorpus 症例報告コーパス(20220531版)、183文書の希少疾患症例報告 |
| 学習/テスト分割 | `train_test_index_1fold.json`：学習146文書 / テスト36文書（BERTは学習から検証をさらに分離） |
| アノテーション | 文字レベル、表層のある98エンティティ型（section/modality等を含め約109ラベル） |
| 統一評価指標 | **緩和エンティティ一致 F1**：`(型が一致) かつ (一方の表層が他方の部分文字列, NFKC正規化)`、貪欲1-1マッチ |
| BERT評価範囲 | ホールドアウトのテスト36文書（漏洩なし） |
| LLM評価範囲 | 全183文書（ゼロショットはどれも未見）；テスト36サブセットをBERTとの直接比較に使用 |

> 緩和一致を用いる理由：corpus goldは断片的アノテーション(例 `慢性閉塞性肺疾患`→`慢`/`閉塞`/`疾`)で、
> BERT(トークン境界)・LLM(自由文)の粒度が異なる。包含による緩和一致は三者に公平でトークナイザ非依存。

---

## 2. 各手法の詳細パラメータ

### 2.1 BERT（ファインチューニング）— `bert/`

**アーキテクチャ**：事前学習BERTエンコーダ + Dropout(0.5) + Linear(768→ラベル数) + CRF。
CRFは不正なBIO遷移に −1e7 のペナルティを与える（`torchcrf_mod.py`）。

| | UTH-BERT | NICT-BERT |
|---|---|---|
| 事前学習 | 臨床テキスト | 日本語Wikipedia |
| 語彙 | 25,000 | 32,016 |
| 分かち書き | preprocess(neologdn/NFKC/全角化) + MeCab(ipadic-neologd + 万病辞書) + **数字1文字分割** + WordPiece | h2z(全角化) + MeCab(juman辞書) + WordPiece |
| ラベル数 | 193 | 188 |
| best epoch | 64 | 47 |
| 学習時間 | 8322 s | 6591 s |

**学習ハイパーパラメータ**（`NER_training.py`、両モデル共通）：
- 最適化 AdamW、学習率 **2e-5**、linear warmup（総ステップの10%）
- batch_size **1**（モデルの `embed[:,1:-1]` によるCLS/SEP除去はbs=1前提）
- max_epoch 250、**早期終了 patience 15**（検証strict micro-F1基準）
- CRF損失 reduction="sum"
- データ変換：`build_conll.py`（difflibで文字オフセット対応 + BIO修復）、entity被覆率 UTH 95.7% / NICT 99.7%

**素のstrict span F1**（論文で一般的な指標、緩和一致ではない）：UTH 0.895 / NICT 0.888；soft：0.930 / 0.925。

### 2.2 ゼロショットLLM — `llm/`

**プロンプト**（`testset_eval.py` の `PROMPT_TEMPLATE`）：全ラベルを列挙 + 臨床文、
JSON配列 `[{"label":…,"text":…}]` のみを要求。temperature=0（貪欲）。

| エンジン | モデル | 主要設定 |
|---------|--------|---------|
| ollama | qwen3.6:35b, llama3.3:70b, gpt-oss:120b | `think:false`（思考無効）, `num_ctx=4096`, `num_predict=1024`；qwen/llamaは **4-GPUデータ並列**（GPU毎に1インスタンス、port 11434-11437、ラウンドロビン）、gpt-ossは単一インスタンス4GPU |
| vLLM | llm-jp-4-32b, SIP-jmed-13b | 推論モデルには思考を確実に切る手段が無く、①guided JSON(schema+minLength) ②`<think></think>`プリフィル ③自然推論(max_tokens最大12000) を試行。報告値は最良の自然推論版 |

**落とし穴**：qwen3.6/llm-jp-4/gpt-oss/SIP-jmedはいずれも「思考型」モデル。思考を切れない場合、
延々と推論しmax_tokens内にJSONを出せない（パース失敗率は結果表を参照）。

### 2.3 ファインチューニングLLM（QLoRA）— `llm/qlora_train.py` + `llm/build_sft_data.py`

**SFTデータ**：学習146文書 → **1672 学習 / 88 検証** ペア（`プロンプト → gold JSON`）。
goldは文字レベルの素の表層で、BERT学習と同一ソース。1例あたり平均29.4エンティティ。

**QLoRA設定**：
- 量子化：4-bit **NF4**、double quant、compute dtype bf16
- LoRA：**r=16, alpha=32, dropout=0.05**、target=`[q,k,v,o,gate,up,down]_proj`、bias=none
- 最適化：AdamW（SFTTrainer）、学習率 **2e-4**、cosineスケジューラ、warmup 0.03、**3 epochs**
- gradient checkpointing 有効、bf16、単一GPU `device_map={"":0}`
- 学習中の評価は無効化（epoch境界でOOMするため）、評価はテスト集で別途実施
- batch / grad_accum：1.8b & SIP-jmed = **1 / 8**；llm-jp-4(MoE) = **2 / 4**（大batchでMoE重み読込を償却）
- max_seq_len：4096（1.8b, SIP-jmed）/ 2048（llm-jp-4）

**環境**：
- `/opt/llm/ft-venv`：transformers 5.13 + trl 0.14 + peft 0.19 + bnb 0.49（標準モデル：1.8b, SIP-jmed）
- `/opt/llm/ft-venv-t451`：transformers 4.51 + trl 0.13 + peft 0.14（自作MoE：llm-jp-4。5.xは重み変換で失敗）

**評価**：base + LoRA を vLLM 経由（`testset_eval.py --lora <adapter>`）、同一の緩和指標。

**落とし穴**：①2GPU device_map="auto" の学習は CUDA launch failure → 単一GPU；②32B MoE 4-bitは非常に遅い
（メモリ帯域律速、batch=1≈31h、batch=2≈16h）；③epoch境界の学習内評価がOOM → 無効化。

---

## 3. 結果比較表（テスト36文書、緩和一致 F1）

### ゼロショット

| 手法 | F1 | Precision | Recall | パース失敗 | 時間 |
|------|:--:|:---------:|:------:|:---------:|:----:|
| **UTH-BERT**（FT済みエンコーダ） | **0.726** | 0.795 | 0.668 | — | — |
| **NICT-BERT**（FT済みエンコーダ） | **0.702** | 0.763 | 0.650 | — | — |
| Qwen3.6-35B | 0.292 | 0.415 | 0.225 | ~0 | 69 min |
| Llama3.3-70B | 0.271 | 0.493 | 0.187 | ~0 | 187 min |
| LLM-jp-4-32B（自然推論） | 0.094 | 0.351 | 0.055 | 63% | — |
| GPT-OSS-120B | 0.052 | 0.584 | 0.027 | 54% | — |
| SIP-jmed-13B（自然推論） | 0.018 | 0.066 | 0.011 | 75% | — |

### QLoRAファインチューニング後

| 手法 | ゼロショット F1 | **FT後 F1** | P | R | パース失敗 |
|------|:-----------:|:----------:|:---:|:---:|:--------:|
| **SIP-jmed-13B**（医療） | 0.018（最下位） | **0.857（最高）** | 0.881 | 0.834 | 0.2% |
| **LLM-jp-1.8B**（3 epoch） | ≈0 | **0.792** | 0.840 | 0.748 | 1.4% |
| LLM-jp-1.8B（1 epoch、対照） | ≈0 | 0.625 | 0.687 | 0.573 | 1.3% |
| UTH-BERT | — | 0.726 | — | — | — |
| NICT-BERT | — | 0.702 | — | — | — |
| LLM-jp-4-32B | 0.094 | *学習中* | | | |
| Qwen3.6 / GPT-OSS / Llama | — | *学習予定* | | | |

**主要な結論**：ファインチューニングで順位が完全に逆転する。ゼロショットではBERT ≫ LLMで
医療モデルが最下位；QLoRA後はLLM ＞ BERTで医療モデルが最高（0.857 vs 0.726）。
ゼロショットの差は「形式/タスク適応」の問題であり、「能力」の問題ではない。

---

## 4. 結果の保存場所

| 内容 | パス |
|------|------|
| **全緩和一致スコア**（BERT+全LLM+FT） | `llm/results/ehr/testset_scores.json`（リポジトリ `results/testset_scores.json` に同期） |
| BERT ラベル別 strict/soft P/R/F1 + 学習指標 | `code/results/NER/{UTH,NICT}/NER/{metrics,NER_strict_RESULT,NER_soft_RESULT}_0.json` |
| BERT loss曲線 | `code/results/NER/{UTH,NICT}/loss_NER_0.json` |
| BERT モデル重み | `code/models/NER/{UTH,NICT}/NER/ner_model_0.pt` |
| **LoRAアダプタ**（FT成果物） | `llm/ft/{llmjp-1.8b-3ep, sip-jmed-13b, llmjp-4-32b}/adapter_model.safetensors` |
| SFT学習データ | `llm/sft_data/{sft_train,sft_val}.jsonl` |
| 実カルテ抽出（BERT） | `code/results/predict/{UTH,NICT}/entities.csv` + `stats.json` |
| 実カルテ抽出（LLM、全183文書gold評価の成果物） | `llm/results/ehr/{qwen35b,llama70b}/entities.csv` |
| 学習ログ | `scratchpad/ft_*.log`（FT）、`code/logs/train_*_run.log`（BERT） |
| 評価ログ | `scratchpad/eval_*.log` |

> ⚠️ `entities.csv` とcorpusは患者/制限データを含むため **GitHubには上げない**。
> リポジトリには `testset_scores.json` とBERT指標JSON（数値/ラベル名のみ）だけを置く。

---

## 5. 再現コマンド早見表

```bash
# BERT: データ作成→学習→評価
python bert/build_conll.py
python bert/NER_training.py --bert_type UTH --bert_path <UTH-BERT> --data_path data/csv/icorpus_UTH.csv --patience 15

# ゼロショットLLM評価
python llm/testset_eval.py --model qwen3.6:35b --engine ollama --hosts <urls>
python llm/testset_eval.py --model <hf> --engine vllm --tp 1        # BERTベースラインは毎回自動再計算

# QLoRAファインチューニング + 評価
python llm/build_sft_data.py --out_dir ./sft_data
python llm/qlora_train.py --model <hf> --out_dir ./ft/<name> --epochs 3 --lr 2e-4
python llm/testset_eval.py --model <base> --lora ./ft/<name> --engine vllm --tp 1
```
