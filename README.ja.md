# 日本語臨床NER：ファインチューニングBERT vs. ゼロショットLLM

[English](README.md) | **日本語** | [中文](README.zh.md)

日本語臨床テキストに対する固有表現抽出(NER)。**ファインチューニングした
BERTエンコーダ**と**ゼロショットの大規模言語モデル(LLM)**を、同一のgold
アノテーション付きコーパス上で、統一した指標で比較する。

本プロジェクトの目標は3つ:

1. **精度比較** — BERT(UTH-BERT, NICT-BERT)とLLM(LLM-jp, GPT-OSS, SIP-jmed-LLM,
   Qwen, Llama)のNER精度を、公開アノテーション済みコーパスで測定する。
2. **実データ抽出** — 同じパイプラインを実臨床レポートに適用する。
3. **FHIR抽出**（予定） — 臨床レポートからFHIRにマッピング可能な項目を抽出する。

> ⚠️ **本リポジトリに臨床データは一切含まれません。** 学習・評価コーパス
> (iCorpus / 症例報告コーパス)は研究目的限定のライセンスで再配布不可であり、
> 実臨床レポートは患者データを含みます。本リポジトリには**コードと集計指標
> (数値とラベル名のみ・本文なし)**だけを収録しています。[データ](#データ)を参照。

---

## 主な結果

コーパスのテスト分割(ホールドアウト36文書)での固有表現抽出精度。**緩和マッチ**
(同一エンティティ型 + 表層文字列の包含、NFKC正規化)で採点。全手法を同一指標で
採点するため、BERTとLLMを直接比較できる。ファインチューニングBERTはホールド
アウトのテスト分割のみで評価(データ漏洩なし)、ゼロショットLLMは全183文書で実行し
テスト36サブセットを head-to-head として報告する。

| 手法 | 種別 | Test-36 F1 | Precision | Recall |
|------|------|:----------:|:---------:|:------:|
| **UTH-BERT**（ファインチューニング） | encoder, 臨床事前学習 | **0.726** | 0.796 | 0.669 |
| **NICT-BERT**（ファインチューニング） | encoder, Wikipedia事前学習 | **0.703** | 0.763 | 0.653 |
| Qwen3.6 35B | LLM, ゼロショット | 0.292 | 0.415 | 0.225 |
| Llama3.3 70B | LLM, ゼロショット | 0.271 | 0.493 | 0.187 |
| LLM-jp-4 32B（自然推論） | LLM, ゼロショット | 0.094 | 0.351 | 0.055 |
| LLM-jp-4 32B（JSON強制） | LLM, ゼロショット | 0.049 | 0.140 | 0.030 |
| SIP-jmed-LLM-3 13B | LLM, ゼロショット | *(集計中)* | | |
| GPT-OSS 120B | LLM, ゼロショット | *(集計中)* | | |

> スナップショット — 本README作成時点で SIP-jmed-LLM・GPT-OSS・LLM-jp-4のトークン
> 拡張版は実行中でした。最新値は [`results/testset_scores.json`](results/testset_scores.json) を参照。

### 要点

- **ファインチューニングBERTはゼロショットLLMを大きく上回る**（〜0.73 vs. ≤0.29）。
  この細粒度・ドメイン特化のNERにおいて、タスク遂行能力はほぼ全て
  100超のラベル体系へのファインチューニングから来る。汎用LLMはゼロショットでは
  太刀打ちできない。
- **LLMは精度はそこそこ、再現率が低い**（P≈0.4〜0.5, R≈0.19〜0.23）。コーパスの
  アノテーションは網羅的だが、LLMはそこまで密には抽出せず、100超の不慣れな
  ラベル名の適用にも苦戦する。
- **UTH-BERT(臨床事前学習) > NICT-BERT(Wikipedia事前学習)**。想定通り。NICTの
  語彙は臨床漢字を多く欠き(`[UNK]`になる)、臨床テキストでの実質的な弱点となる。
- **「思考」型LLMは構造化抽出に不向き。** 即座にJSON出力を強制(guided decoding)
  すると推論の利点を失い出力が劣化する。一方で自由に推論させると遅く、
  パース可能なJSONを出す前にトークン上限を超えることが多い。思考を無効化した
  通常出力の方が本タスクでは良い傾向。

---

## リポジトリ構成

```
bert/                     BERT-CRF NERパイプライン(システムPython: torch 2.0, transformers 4.46)
  NER_training.py           CoNLLデータでBERT-CRFをファインチューニング(早期終了+指標)
  build_conll.py            iCorpus文字レベルJSON -> CoNLL CSV(モデル別トーカナイザ)
  extract_structured.py     生テキスト -> MeCab -> BERT-CRF -> 構造化エンティティ
  predict_reports.py        実レポートの一括予測(cp932 CSV)
  evaluate_ner.py           予測CSVから strict/soft のspan-level P/R/F1
  summarize_results.py      設定間の比較
  preprocess_text.py        テキスト正規化(neologdn / NFKC / 全角化)
  tokenization_mod.py       MeCab + WordPiece トーカナイザ(UTH-BERT公式)
  lib/                      モデル(BERT_CRF, CRF)・学習ループ・utils・評価
llm/                      LLM抽出・評価(venv: torch 2.6+cu124)
  testset_eval.py           gold corpusでのBERT vs LLM評価(緩和指標)
  llm_extract_vllm.py       vLLM一括抽出(HFモデル: LLM-jp, SIP-jmed)
  llm_extract_ollama.py     ollama抽出、マルチGPUデータ並列(Qwen, Llama, GPT-OSS)
  llm_extract.py            transformers単発ベースライン
results/                  集計指標のみ(本文なし・患者データなし)
  testset_scores.json       BERT vs LLM比較(本リポジトリの主要結果)
  bert_UTH/ bert_NICT/      ラベル別 strict/soft P/R/F1 + 学習指標
docs/WORKFLOW.md          詳細な運用メモ
```

---

## 手法

### BERT（ファインチューニング）

`BERT-CRF` = 事前学習BERTエンコーダ + 線形ヘッド + CRF。2つのエンコーダ:

- **UTH-BERT** — 日本語臨床テキストで事前学習(語彙25k; MeCab ipadic-neologd +
  万病辞書(J-Medic) + 数字分割 + WordPiece)。
- **NICT-BERT** — 日本語Wikipediaで事前学習(語彙32k; MeCab Juman辞書 + WordPiece)。

文字レベルのエンティティアノテーションを、`difflib`で文字スパンをサブワードに
対応付けてCoNLLへ変換する([`bert/build_conll.py`](bert/build_conll.py))。
エンティティ被覆率は 95.7%(UTH) / 99.7%(NICT)。ファインチューニングはAdamW、
検証F1での早期終了。

> **CRFの落とし穴:** CRFは不正なBIO遷移(`O→I-x`, `B-x→I-y` …)に−1e7のペナルティを
> 与える。投入するIOBは厳密に正しいBIOでなければ、F1は正常なまま損失が爆発
> (〜1e8)する。`build_conll.py`にBIO修復処理を含む。

### LLM（ゼロショット）

各臨床文を、全ラベルを列挙したプロンプトに入れ、JSON配列
`[{"label","text"}]` を要求する。2つのバックエンド:

- **ollama**(Qwen3.6-35B, Llama3.3-70B, GPT-OSS-120B) — `think:false`で思考を抑制、
  スループットのため**4-GPUデータ並列**(GPU毎に1インスタンス、ラウンドロビン)、
  KVキャッシュ節約のためcontextを4096に制限。
- **vLLM**(LLM-jp-4-32B, SIP-jmed-13B) — HFモデル。推論モデルにはguided JSON
  decoding か、大きなトークン予算での自然推論のいずれか。

### 統一評価

BERTの予測とLLMの出力を、共にgoldの文字レベルアノテーションに対して**同一の
緩和指標**で採点する(`llm/testset_eval.py`)。エンティティ = `(型, 表層)`。
型が一致し、かつ一方の表層が他方を包含(NFKC正規化)すれば一致とみなす。
トークンベース(BERT)と自由文(LLM)の両方に公平で、コーパスの断片的なgoldスパンにも
頑健。

---

## データ

**含まれません。** 再現には以下が必要:

- **iCorpus（症例報告コーパス）** — アノテーション付き症例報告コーパス
  (183文書、文字レベルエンティティ)。研究目的限定ライセンス。東京大学
  (医療AI開発学講座)から入手し、`corpus/icorpus_.../data/json/`に配置。
- **UTH-BERT** — https://ai-health.m.u-tokyo.ac.jp/home/research/uth-bert
- **NICT-BERT** — https://alaginrc.nict.go.jp/nict-bert/
- **万病辞書**(J-Medic) — UTHトークナイズ用。
- 抽出ステップ用の実臨床レポート(各自の患者データ)。

---

## セットアップ

```bash
# BERTパイプライン(システムPython)
pip install torch==2.0.1 transformers==4.46.3 seqeval pandas mecab-python3 jaconv neologdn
# + MeCab辞書: mecab-ipadic-neologd(UTH), mecab-jumandic-utf8(NICT)

# LLM評価(別venv)
pip install "vllm==0.8.5" "transformers==4.51.3" pandas requests
# + ollama (Qwen/Llama/GPT-OSS用)
```

### 実行

```bash
# 1) コーパスから学習データを作成
python bert/build_conll.py

# 2) ファインチューニング(エンコーダ別)
python bert/NER_training.py --bert_type UTH  --bert_path <UTH-BERT>  --data_path data/csv/icorpus_UTH.csv  --patience 15
python bert/NER_training.py --bert_type NICT --bert_path <NICT-BERT> --data_path data/csv/icorpus_NICT.csv --patience 15

# 3) BERT vs LLM 精度比較
python llm/testset_eval.py                                   # BERTベースライン
python llm/testset_eval.py --model qwen3.6:35b --engine ollama --hosts <urls>
python llm/testset_eval.py --model <hf-model> --engine vllm --tp 2

# 4) 実レポートから抽出
python bert/predict_reports.py --bert_type UTH --model_path <model.pt> --data_dir <reports> --out_dir <out>
```

詳細な運用手順(マルチGPU ollama構成、vLLM/推論モデルの扱いを含む)は
[`docs/WORKFLOW.md`](docs/WORKFLOW.md)を参照。

---

## ライセンス

コード: MIT。**モデル・辞書・コーパスは各自のライセンスに従い、本リポジトリには
含まれません。** 臨床・患者データは設計上除外しています。
