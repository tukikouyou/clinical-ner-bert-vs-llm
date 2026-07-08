# 臨床テキストNER・構造化抽出プロジェクト 作業手順

更新日: 2026-07-03 (コード整理後。旧predict.py/process.py/RE関連は削除、
バックアップは /opt/llm/backups/code_before_cleanup_20260703.tar.gz)

## 全体像

| 目標 | 内容 | 状態 |
|------|------|------|
| 1 | 公開データで BERT (UTH/NICT) と LLM (LLM-jp, GPT-OSS, SIP-jmed-llm) の抽出精度を比較 | BERT側パイプライン稼働済み / LLM側環境構築中 |
| 2 | 同じ手法で実臨床レポートから抽出 | パイプラインは入力を差し替えるだけ |
| 3 | 実臨床レポートから FHIR 項目を抽出 | 未着手(エンティティ→FHIRマッピング設計が必要) |

## 1. 既存モデルで構造化抽出(すぐ使える)

学習済み UTH-BERT + CRF (`code/models/NER/UTH/NER/ner_model_0.pt`、203タグ)を使用。

```bash
cd ~/yagahara/code
python3 extract_structured.py \
    --input <入力>       # 1行=1レポートのtxt、または .txt を含むディレクトリ
    --output_dir ./results/extraction/<実験名>
```

出力(`output_dir` 内):
- `entities.json` — 構造化エンティティ `[{doc, chunk, label, text, start_token, end_token}]`
- `entities.csv` — 同内容のフラットテーブル
- `token_predictions.csv` — トークン単位のIOB予測(デバッグ・評価用)

特徴: 510トークン超の文書も文境界(。)で自動分割して全文処理する(削除済みの旧
`predict.py` は長文を捨てていた)。

評価(gold付き予測CSVがある場合):

```bash
python3 evaluate_ner.py --pred_csv ./results/NER/UTH/NER/0.csv
# 既存モデルのベースライン: strict micro F1=0.9215, soft micro F1=0.9455
```

## 1.5 iCorpus での BERT ファインチューニング & 実データ予測 (2026-07-06 追加)

大タスク: NICT-BERT と UTH-BERT を iCorpus(症例報告コーパス)で fine-tuning し、
実臨床レポート(`predicted data/`)を予測する。

### モデル(確認済み)
- `NICT_BERT-base_JapaneseWikipedia_32K_BPE`: 日本語Wikipedia事前学習, vocab32016,
  分かち書き = MeCab(juman辞書) + WordPiece。生テキスト予測には mecab-jumandic が必要
  (`/var/lib/mecab/dic/juman-utf8`, `apt install mecab-jumandic-utf8` 済)。
- `UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K`: 臨床事前学習, vocab25000,
  分かち書き = preprocess + MeCab(ipadic-neologd+万病辞書) + 数字1文字分割 + WordPiece。
  (`model/` と同一実体)

### データ生成: iCorpus JSON → CoNLL CSV
```bash
python3 build_conll.py   # → data/csv/icorpus_UTH.csv, icorpus_NICT.csv
```
- 文字レベルの entity アノテーションを、各モデルのトークナイザに合わせた
  サブワード列へ `lib/align_tokenize.py`(difflib で文字オフセット復元)で対応付け、
  IOB を付与。学習/予測で**同一トークナイザ**を使うため両者の整合が取れる。
- entity カバレッジ: UTH 95.7% / NICT 99.7%(残りは辞書粒度による細粒度実体の衝突。
  train/predict 両方で同じ制約なので不整合にはならない)。
- 旧 `data/csv/UTH_CR_conll_format_arbitrary_*.csv` は同 corpus(20220531)の別辞書版で、
  再現不可のため未使用。train/test 分割は `train_test_index_1fold.json`(146/36)を流用。

### 学習(早停あり)
```bash
CUDA_VISIBLE_DEVICES=0 python3 NER_training.py --bert_type UTH \
  --bert_path ../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K \
  --data_path ./data/csv/icorpus_UTH.csv --max_epoch 250 --patience 15
CUDA_VISIBLE_DEVICES=1 python3 NER_training.py --bert_type NICT \
  --bert_path ../NICT_BERT-base_JapaneseWikipedia_32K_BPE \
  --data_path ./data/csv/icorpus_NICT.csv --max_epoch 250 --patience 15
```
出力: `models/NER/<type>/NER/ner_model_0.pt`,
`results/NER/<type>/NER/{metrics,NER_strict_RESULT,NER_soft_RESULT,NER_tag2idx}_0.json`,
`results/NER/<type>/loss_NER_0.json`(epoch別 loss/val_F/時間/best_epoch)。
注: モデルの `embed[:,1:-1]` は CLS/SEP 除去のため **batch_size=1 前提**。

### 実データ予測(cp932, 【所見】+【診断まとめ】)
```bash
python3 predict_reports.py --bert_type UTH \
  --bert_path ../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K \
  --model_path ./models/NER/UTH/NER/ner_model_0.pt \
  --tag2idx ./results/NER/UTH/NER/NER_tag2idx_0.json \
  --data_dir "../predicted data" --out_dir ./results/predict/UTH
```
予測は batch 対応(CRF に padding mask を渡す)。出力: `entities.csv` + `stats.json`
(ラベル別/列別/年別件数, 空予測率, 処理時間)。`summarize_results.py` で UTH/NICT 比較。

## 2. 新データでの再学習(fine-tuning)

### 2.1 必要なデータ形式

`code/data/csv/UTH_CR_conll_format_arbitrary_UTH.csv` と同じCoNLL風CSV:

| 列 | 意味 |
|----|------|
| word | トークン(UTH-BERT語彙のWordPiece。extract_structured.pyと同じ分かち書き) |
| IOB | 正解タグ (B-xxx / I-xxx / O) |
| name | 文書ID |
| unique_no | チャンクID(文書内の510トークン以下の単位) |
| serial, index, brad_id, rel_type, rel_tail | 関係抽出用(NERだけなら未使用。ダミー可) |

+ `code/data/index/` に訓練/テスト文書名の分割JSON (`{"train_name": [...], "test_name": [...]}`)

**新データが別形式(brat/XML/JSON等)で来たら変換スクリプトを書く。**
生テキストのトークン化は `extract_structured.py` の `tokenize_reports()` がそのまま使える。

### 2.2 学習の実行

```bash
cd ~/yagahara/code
python3 NER_training.py \
    --bert_path ../model \                # UTH-BERT本体
    --bert_type UTH \
    --data_path ./data/csv/<新データ>.csv \
    --max_epoch 250 --batch_size 1
```

- モデル → `code/models/NER/UTH/NER/ner_model_0.pt`(上書きされるので旧モデルは退避)
- タグ辞書 → `code/results/NER/UTH/NER/NER_tag2idx_0.json`
- 評価(strict span F1) → `code/results/NER/UTH/NER/NER_strict_RESULT_0.json`

### 2.3 再学習済みモデルで抽出

```bash
python3 extract_structured.py --input <入力> --output_dir <出力> \
    --model_path ./models/NER/UTH/NER/ner_model_0.pt \
    --tag2idx_path ./results/NER/UTH/NER/NER_tag2idx_0.json
```

### 2.4 NICT-BERT で学習する場合

- データはNICT分かち書き版が必要(公開データは `UTH_CR_conll_format_arbitrary_NICT.csv` が既にある)
- NICT-BERT本体(NICT_BERT_BASE_100K)をダウンロードして `--bert_path` に指定、`--bert_type NICT`

## 3. LLMによる抽出

### 3.1 環境

- venv: `/opt/llm/venv`(transformers>=4.55, torch cu124。ルートパーティション側、
  `/home` が96%満杯のため)
- HFキャッシュ: `/opt/llm/hf-cache`(モデルのダウンロード先。
  使う時は `export HF_HOME=/opt/llm/hf-cache`)

### 3.2 実行

```bash
export HF_HOME=/opt/llm/hf-cache
/opt/llm/venv/bin/python ~/yagahara/llm/llm_extract.py \
    --model openai/gpt-oss-120b \        # HF ID または ローカルパス
    --input <入力txt/ディレクトリ> \
    --output_dir ./results/gpt-oss-120b
```

- ラベルセットはデフォルトでBERT側 `NER_tag2idx_0.json` から自動生成(101ラベル)
- `--labels_path` で `{ラベル: 説明}` 形式のJSONを渡すと精度向上が期待できる
- 出力は BERT側と同じ `entities.json`(+ `raw_outputs.json` でLLMの生出力を保存)

### 3.3 モデルごとの注意(4x RTX A6000 = VRAM 192GB)

| モデル | サイズ | 備考 |
|--------|--------|------|
| openai/gpt-oss-120b | MXFP4 63GB | A6000(Ampere)はMXFP4カーネル非対応→transformersはbf16に展開(~240GB)しCPUオフロード必要で低速。**実用はollama推奨**: `ollama pull gpt-oss:120b`(ルート側に62GB) |
| llm-jp/llm-jp-3.1-13b-instruct4 等 | bf16 ~26GB | 1GPUで動く。LLM-jp-4系は公開名を要確認 |
| SIP-jmed-llm-2-8x13b-OP-instruct | bf16 ~150GB | 4GPUでギリギリ載る。**ディスク残量注意**(ルート241GB空きからgpt-oss分を引いた残りで管理) |
| ~/models/llama4-hf | 44GB | ダウンロード済み |

## 4. 精度比較(目標1)の進め方

1. 公開データセットを決めて入手(→ CoNLL CSV へ変換)
2. BERT側: `NER_training.py` で学習→テスト分割の strict F1 は自動出力
3. LLM側: テスト分割の生テキストに `llm_extract.py` → 評価スクリプトで gold と照合
   (エンティティ(label, text)単位のP/R/F1。offset無しのlenient評価から始めるのが現実的)
4. 表にまとめる

## 5. FHIR抽出(目標3)の方針メモ

- NERエンティティ → FHIRリソース(Condition, MedicationStatement, Observation,
  Procedure, Patient...)へのマッピング表を設計
- LLM直接生成(テキスト→FHIR JSON)との2方式を比較すると論文的に面白い
- 用語コード(ICD-10, YJコード等)への正規化が必要になる段階で万病辞書・MEDIS辞書を活用

## コード整理の記録(2026-07-03)

削除(バックアップ: `/opt/llm/backups/code_before_cleanup_20260703.tar.gz`):
- `main_ner.py` / `main_re.py` / `main_joint.py` — 旧実験エントリ(RE・joint含む)
- `example_main.py` — UTH-BERT公式のトーカナイズ例
- `predict.py` / `process.py` — `extract_structured.py` に置き換え
- `lib/loop/train_test_loop.py`, `lib/loop/pl_train_test_loop_for_re.py`,
  `lib/models/attention.py`, `lib/preprocess/` — RE/joint専用の死コード
- `lib/models/joint.py` → NERで使う `BERT_CRF` だけ `bert_crf.py` に分離
- `lib/util/utils.py` — 29関数中17個(RE専用)を削除、664行→177行
- `lib/util/eval_ner_re.py` → RE評価を除き `eval_ner.py` に改名
- `使い方.txt` → `code/README.md` に置き換え
- MANBYO辞書 → `code/resources/` へ移動

追加: `evaluate_ner.py`(strict/soft F1評価の独立スクリプト)

## 既知の問題・環境メモ

- `/home` パーティションは96%使用(残20GB)。大きいのは `~/Desktop`(124GB)、
  `~/Downloads`(65GB)、`~/models`(44GB)。モデル類は `/opt/llm`(ルート側)に置く運用
- GPT-OSSは「ダウンロード済み」ではなかった(キャッシュに4KBの空メタデータのみ)。
  2026-07-03 に `/opt/llm/hf-cache` へダウンロード完了(63.8GB, 14/14シャード)。
  ollama側には gpt-oss は存在しない(llama3.3:70b と qwen3.6:35b のみ)
- 既存venv(axoenv等)はtransformers<=4.52でGPT-OSS非対応。`/opt/llm/venv` を使う
