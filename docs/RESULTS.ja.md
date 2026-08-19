# 詳細結果と実験パラメータ

[中文](RESULTS.md) | **日本語**

日本語臨床NER：ファインチューニングBERT vs. ゼロショット/ファインチューニングLLM。
2026-08-18 時点の完全な記録。（新規医療モデル Med42-70B / MedGemma-27B のゼロショット + ファインチューニング結果を含む。）

> **先行研究と本プロジェクトの位置づけ**：臨床NERにおける「BERT vs LLM」「zero-shot vs fine-tuning」の
> 比較には既に先例があり、「初の比較」と主張すべきではない。
> - **Shimizu et al. 2025**（JMIR Med Inform 13:e76773）：日本語臨床の**疾患名**認識で fine-tuned Llama-3.1 > BERT。
>   ただし比較対象は汎用日本語BERTのみで、著者自身が医療ドメイン事前学習BERTを含めていないことを limitation として明記。
> - **Lu et al. 2024**（AMIA）：RareDis 上で汎用/医療 LLM（Meditron/Llama2-MedTuned/ChatGPT）と BioClinicalBERT を比較、
>   zero/few-shot・RAG・instruction-FT を網羅。
> - **Chen et al. 2023**（arXiv:2305.16326）：BioNLP benchmark、GPT/LLaMA/PMC-LLaMA × zero/few/FT vs domain BERT/BART、NER を含む。
>
> 本プロジェクトの位置づけ（上記より網羅的）：**同一 iCorpus + 113 細粒度タグ + 同一評価指標**の下で、
> **汎用/医療の BERT（UTH/NICT）と LLM（1.8B–70B、汎用+医療）× zero-shot/QLoRA** を体系的に比較し、
> **タグ別誤り分析**まで行う。

---

## 1. タスク設定

| 項目 | 値 |
|------|----|
| コーパス | iCorpus 症例報告コーパス(20220531版)、183文書の希少疾患症例報告 |
| 学習/テスト分割 | `train_test_index_1fold.json`：学習146文書 / テスト36文書（BERTは学習から検証をさらに分離） |
| アノテーション | 文字レベル、**113種**のエンティティ型（`end+1`修正後、以前脱落していた単字実体型を含む）；総実体 **73797** |
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
| vLLM | llm-jp-4-32b, SIP-jmed-13b, **Med42-70B**, **MedGemma-27B** | 推論モデルには思考を確実に切る手段が無く、①guided JSON(schema+minLength) ②`<think></think>`プリフィル ③自然推論(max_tokens最大12000) を試行。報告値は最良の自然推論版。Med42/MedGemmaの重みは移動HDDからローカルNVMe(`/opt/llm/models/`)へコピーしてからロード(FUSE/ntfsはvLLMマルチワーカーの並列読込に耐えない) |

**落とし穴**：qwen3.6/llm-jp-4/gpt-oss/SIP-jmedはいずれも「思考型」モデル。思考を切れない場合、
延々と推論しmax_tokens内にJSONを出せない（パース失敗率は結果表を参照）。

### 2.3 ファインチューニングLLM（QLoRA）— `llm/qlora_train.py` + `llm/build_sft_data.py`

**SFTデータ**：学習146文書 → **1672 学習 / 88 検証** ペア（`プロンプト → gold JSON`）。
goldは文字レベルの素の表層で、BERT学習と同一ソース。1例あたり平均33.2エンティティ（`end+1`修正後）。

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

**評価インフラ**（`testset_eval.py`、全て test36・同一の緩和指標）：
- **denseモデル**（SIP-jmed-13B、1.8B）：vLLM tp=1 + `--lora`（LlamaアーキはLoRA対応）
- **MoEモデル**（Qwen3-30B、LLM-jp-4-32B＝Qwen3-MoE）：vLLMは**MoEのLoRA未対応**のため、
  LoRAを**baseにマージ**（`merge_lora.py`）してから **vLLM V0エンジン tp=2** で配信
- **Llama-3.3-70B-ft / Med42-70B-ft**：dense、vLLM **V0 tp=4** + `--lora`（4GPUにbf16 70B）
- **MedGemma-27B-ft**：dense（Gemmaアーキ、vLLMはLoRA対応）、vLLM **V0 tp=2** + `--lora`
- **Med42-70B ファインチューニング**：70Bは単一GPUで `logits.float()` がOOM → 2GPUモデル並列（`--device_map auto`）
- **ゼロショット**：ollama（qwen3.6/llama3.3/gpt-oss、`think:false`）、vLLM V0 tp=2（MoE）
- 落とし穴：vLLM **V1エンジンのtp≥2はtritonキャッシュ競合**（`FileExistsError`）→ **V0エンジン**（`VLLM_USE_V1=0`）+ `enforce_eager` で回避

**落とし穴**：①2GPU device_map="auto" の学習は CUDA launch failure → 単一GPU；②32B MoE 4-bitは非常に遅い
（メモリ帯域律速、batch=1≈31h、batch=2≈16h）；③epoch境界の学習内評価がOOM → 無効化。

---

## 3. 結果比較表（テスト36文書 = 433文、緩和一致 F1、修正gold）

> ⚠️ **重要な修正（2025更新）**：iCorpusの実体 `end` オフセットが**閉区間**であることが判明。
> 元パイプラインは開区間として扱っており、①各実体の最後の1文字が欠落（症例→症）②`end==start`の
> 単字実体8466個（全体の11.5%）が全て脱落、していた。全面修正（`end+1`）→データ再構築
> （実体 65331→**73797**、ラベル 109→**113種**）→ BERT再学習 + 全FT再実行 + 再評価。
> 下表は**全て修正gold・同一test36**。緩和一致スコアへの影響は小さい（BERT ±0.01）が、
> 抽出結果・単字実体・学習信号はすべて正しくなった。詳細は [WORKFLOW.md](WORKFLOW.md)。

### ゼロショット（いずれも低い——推論モデルはJSONを出せない）

| 手法 | F1 | Precision | Recall | パース失敗 |
|------|:--:|:---------:|:------:|:---------:|
| **UTH-BERT**（FT済みエンコーダ） | **0.752** | 0.871 | 0.661 | — |
| **NICT-BERT**（FT済みエンコーダ） | **0.742** | 0.861 | 0.651 | — |
| MedGemma-27B（医療） | 0.319 | 0.457 | 0.245 | 0 |
| Qwen3.6-35B (ollama) | 0.288 | 0.446 | 0.212 | 0 |
| Llama3.3-70B (ollama) | 0.261 | 0.522 | 0.174 | 0 |
| Qwen3-30B-A3B (vLLM) | 0.244 | 0.316 | 0.199 | 1 |
| Med42-70B（医療） | 0.136 | 0.300 | 0.088 | 11 |
| LLM-jp-4-32B | 0.068 | 0.324 | 0.038 | 66% |
| GPT-OSS-120B | 0.040 | 0.606 | 0.021 | 59% |
| LLM-jp-1.8B | 0.026 | 0.054 | 0.017 | 3 |
| SIP-jmed-13B（医療） | 0.012 | 0.119 | 0.006 | 74% |

> 医療モデルの二面性：instruct型の **MedGemma-27B がゼロショット最高（0.319）**、
> しかし同じ医療の **SIP-jmed はゼロショット最下位（0.012）**——差は指定形式で出力できるか否かであり、ドメイン知識そのものではない。

### QLoRAファインチューニング後（全てBERTを逆転）

| 手法 | ゼロショット F1 | **FT後 F1** | P | R | パース失敗 |
|------|:-----------:|:----------:|:---:|:---:|:--------:|
| **Med42-70B**（医療・70B） | 0.136 | **0.825（最高）** | 0.925 | 0.745 | 0 |
| **Llama-3.3-70B** | 0.261 | **0.822** | 0.921 | 0.743 | 0 |
| **MedGemma-27B**（医療・27B） | 0.319 | **0.822** | 0.917 | 0.746 | 0 |
| **SIP-jmed-13B**（医療・13Bのみ） | 0.012（ゼロショット最下位） | **0.819** | 0.918 | 0.740 | 0 |
| **Qwen3-30B-A3B** | 0.244 | **0.812** | 0.910 | 0.733 | 0 |
| **LLM-jp-4-32B** | 0.068 | **0.811** | 0.902 | 0.737 | 0 |
| **LLM-jp-1.8B**（3 epoch） | 0.026 | **0.777** | 0.880 | 0.695 | 2 |
| UTH-BERT | — | 0.752 | — | — | — |
| NICT-BERT | — | 0.742 | — | — | — |

**主要な結論（修正データでも成立）**：ファインチューニングで順位が完全に逆転する。ゼロショットでは
BERT（0.752）≫ 全LLM；QLoRA後は**7つのFT-LLM全て（0.777–0.825）＞ BERT（0.752）**。
ゼロショットの差は「形式/タスク適応」の問題であり、「能力」の問題ではない。
**最も注目**：ファインチューニング後の **Top-4のうち3つが医療モデル**——13BのSIP-jmed（0.819）、
27BのMedGemma（0.822）が70B汎用のLlama-3.3（0.822）にほぼ並び、最高は70B医療のMed42（0.825）。
ドメイン事前学習により中小モデルが大モデルに匹敵する。GPT-OSS-120BはMXFP4量子化 + Ampereアーキ非互換のため**ファインチューニング未実施**（詳細は [WORKFLOW.md](WORKFLOW.md)）。

### 3.5 タグ別分析

**7つのファインチューニングモデル + 2つのBERT すべて**についてタグ別分析を行った（各モデル個別CSV：`scratchpad/per_tag/per_tag_<model>.csv`；
クロスモデル F1 行列：`scratchpad/per_tag_allmodels.csv`）。下表は最良の **med42-70b-ft** を代表として示す
（全102タグの明細：`scratchpad/per_tag_med42ft.csv`）。データ源：`compare_csv`（1実体ごとに gold/予測/判定：
正解/ラベル誤り/未抽出/誤抽出）→ P/R/F1 を集計。

**クロスモデル一貫性（重要）**：苦手タグのパターンは全モデルで高度に一致する——各モデルの top-5 苦手は同一の
「-others」主体系であり、最大の混同はいずれも「-others → -patient」。macro-F1（support≥20 の等重平均）：
7つのFTモデルは **0.72–0.77**（相互に≤0.02）、BERT-UTH 0.73、NICT 0.71。
> 注：等重 macro では LLM-jp-1.8b（0.72）が BERT-UTH（0.73）をわずかに**下回る**——小モデルは稀少タグで不利。
> ただし論文の主指標は**頻度加重の全体 F1** であり、そこでは 1.8b（0.777）＞ BERT（0.752）。両者は矛盾しない。

**全体**：精度は総じて高い（P 0.88–0.98）が、**ボトルネックは再現率**（R 0.67–0.86）——過剰抽出は極めて少なく、主に「取りこぼし」傾向。

**① 頻出タグ（vs BERT）——向上は臨床上重要なカテゴリに集中**
| タグ | 件数 | med42-ft F1 | BERT F1 | Δ |
|---|--:|--:|--:|--:|
| state-patient（症状・所見） | 2318 | 0.85 | 0.72 | +0.13 |
| clinical_test（検査） | 587 | 0.85 | 0.71 | +0.14 |
| treatment（治療） | 614 | 0.78 | 0.71 | +0.07 |
| drug（薬剤） | 283 | 0.80 | 0.70 | +0.10 |
| PN-Negative-patient（否定） | 276 | 0.79 | 0.86 | **−0.07**（唯一 BERT 優位） |

**② 苦手タグ（support≥20, F1 低）**
| タグ | 件数 | F1 | BERT | 主因 |
|---|--:|--:|--:|---|
| value-others | 63 | 0.12 | 0.07 | 取りこぼし(R=0.08) |
| state-others | 155 | 0.48 | 0.41 | 取りこぼし(FN=104) |
| device | 75 | 0.65 | 0.46 | — |
| time_span | 59 | 0.58 | 0.54 | — |

（変化類 quantity/quality_progress も弱く、F1 0.60–0.74）

**③ 誤りのメカニズム**：最大の誤りは **「-others → -patient」の主体取り違え**（state-others→patient 40件、value-others→patient 31件…）
——モデルは「患者 vs 家族/他者」を区別できず、一律に患者と判定する。次いで意味的近傍の混同（state↔item、tissue↔body、clinical_test↔item）。

**④ 臨床的意義**：BERT とファインチューニング LLM の約7ポイント差は**ほぼ全て再現率に由来**し、症状/検査/薬剤など臨床上重要なカテゴリの取りこぼし低減として現れる；
両者とも精度は既に高い。苦手タグ（-others 主体系/変化類/否定）は BERT でも同様に低く、**本質的な難タグ**である。

---

## 4. 結果の保存場所

| 内容 | パス |
|------|------|
| **全緩和一致スコア**（BERT+全LLM+FT） | `llm/results/ehr/testset_scores.json`（リポジトリ `results/testset_scores.json` に同期） |
| BERT ラベル別 strict/soft P/R/F1 + 学習指標 | `code/results/NER/{UTH,NICT}/NER/{metrics,NER_strict_RESULT,NER_soft_RESULT}_0.json` |
| BERT loss曲線 | `code/results/NER/{UTH,NICT}/loss_NER_0.json` |
| BERT モデル重み | `code/models/NER/{UTH,NICT}/NER/ner_model_0.pt` |
| **LoRAアダプタ**（FT成果物） | `llm/ft/{llmjp-1.8b-3ep, sip-jmed-13b, llmjp-4-32b, qwen3-30b-a3b, llama-3.3-70b}/`；新規医療モデルは `/opt/llm/ft-local/{med42-70b, medgemma-27b}/adapter_model.safetensors`（いずれも627ステップ = 3 epoch） |
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
