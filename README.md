# Japanese Clinical NER: Fine-tuned BERT vs. Zero-shot LLMs

**English** | [日本語](README.ja.md) | [中文](README.zh.md)

Named-entity recognition (NER) on Japanese clinical text, comparing **fine-tuned
BERT encoders** against **zero-shot large language models (LLMs)** on the same
gold-annotated corpus, under one unified evaluation metric.

The project has three goals:

1. **Accuracy comparison** — measure NER accuracy of BERT (UTH-BERT, NICT-BERT)
   and LLMs (LLM-jp, GPT-OSS, SIP-jmed-LLM, Qwen, Llama) on a public annotated corpus.
2. **Real-data extraction** — apply the same pipelines to real clinical reports.
3. **FHIR extraction** *(planned)* — extract FHIR-mappable items from clinical reports.

> ⚠️ **No clinical data is included in this repository.** The training/evaluation
> corpus (iCorpus / 症例報告コーパス) is distributed under a restrictive
> research-only license and may not be redistributed, and the real clinical
> reports contain patient data. This repo contains **only code and aggregate
> metrics (numbers and label names — no text)**. See [Data](#data).

---

## Key results

Entity-level NER accuracy on the corpus test split (36 held-out documents),
scored with a **relaxed match** (same entity type + containment of surface
strings, NFKC-normalized). All methods are scored identically so BERT and LLMs
are directly comparable. Fine-tuned BERT is evaluated only on the held-out test
split (no data leakage); zero-shot LLMs are run on all 183 documents and the
test-36 subset is reported for the head-to-head.

| Method | Type | Test-36 F1 | Precision | Recall |
|--------|------|:----------:|:---------:|:------:|
| **UTH-BERT** (fine-tuned) | encoder, clinical pretraining | **0.725** | 0.794 | 0.667 |
| **NICT-BERT** (fine-tuned) | encoder, Wikipedia pretraining | **0.703** | 0.763 | 0.651 |
| Qwen3.6 35B | LLM, zero-shot | 0.292 | 0.415 | 0.225 |
| Llama3.3 70B | LLM, zero-shot | 0.271 | 0.493 | 0.187 |
| LLM-jp-4 32B | LLM, zero-shot (reasoning) | 0.094 | 0.351 | 0.054 |
| GPT-OSS 120B | LLM, zero-shot (reasoning) | 0.052 | 0.584 | 0.027 |
| SIP-jmed-LLM-3 13B | LLM, zero-shot (reasoning, **medical**) | 0.018 | 0.066 | 0.011 |

Metric: relaxed entity match (type + surface containment, NFKC). BERT on the
held-out 36-doc test split; LLMs run zero-shot (reasoning models with thinking
enabled and a large token budget — see [Notes](#notes-on-the-llm-numbers)).
Full numbers: [`results/testset_scores.json`](results/testset_scores.json).

### Fine-tuning the LLMs (QLoRA) — the ranking reverses

The same LLMs, QLoRA-fine-tuned on the **same 146-document training split** as
BERT (SFT: prompt → gold-JSON) and evaluated identically:

| Method | Zero-shot F1 | **Fine-tuned F1** | Δ |
|--------|:-----------:|:-----------------:|:--:|
| **SIP-jmed-LLM-3 13B** (medical) | 0.018 *(last)* | **0.857** *(best)* | **+0.839** |
| **LLM-jp-3.1 1.8B** | ≈0 | **0.792** | +0.79 |
| UTH-BERT (fine-tuned encoder) | — | 0.725 | — |
| NICT-BERT (fine-tuned encoder) | — | 0.703 | — |
| LLM-jp-4 32B | 0.094 | *(training)* | |
| Qwen / Llama / GPT-OSS | 0.27–0.29 / 0.05 | *(pending)* | |

**Fine-tuning reverses the entire ranking.** Zero-shot, BERT beats every LLM and
the medical model is *worst*; after QLoRA, LLMs beat BERT and the medical model
is *best* (0.857 vs. BERT's 0.725). Even a tiny 1.8B model (≈0 zero-shot) reaches
0.792. The zero-shot gap was about **task/format adaptation, not capability** —
once the model learns to emit the structured output, the LLM's richer
representation (and the medical model's domain knowledge) surpasses BERT. Domain
pretraining that was a *liability* zero-shot (a reasoning model that could not
produce JSON) becomes the decisive *advantage* after fine-tuning.

Method: QLoRA (4-bit NF4 + LoRA r=16 on attention+MLP), 3 epochs, single-GPU.
See [`llm/qlora_train.py`](llm/qlora_train.py) and [`llm/build_sft_data.py`](llm/build_sft_data.py).

### Takeaways (zero-shot)

- **Fine-tuned BERT vastly outperforms every zero-shot LLM** on this fine-grained,
  domain-specific NER task (~0.71 vs. ≤0.29 — a 2.5×+ gap). The task capability
  comes almost entirely from fine-tuning on the 100+-label annotation scheme;
  general LLMs cannot match it zero-shot.
- **LLMs are recall-poor.** The corpus annotation is exhaustive ("網羅的");
  LLMs do not extract that densely and struggle to apply 100+ unfamiliar label
  names (R ≈ 0.19–0.23 for the best LLMs, far below BERT's ~0.66).
- **The medical-specialized model scored *worst***. SIP-jmed-LLM-3 (13B, Japanese
  clinical) landed at 0.018 — last among all LLMs — because it is a
  *reasoning-tuned* model that rambles indefinitely and fails to emit parseable
  JSON for **75% of inputs even with a 12k-token budget**. Domain knowledge does
  not help when the model cannot produce structured output.
- **"Thinking"/reasoning LLMs are a poor fit for structured extraction** and
  cluster at the bottom (SIP-jmed 0.018, GPT-OSS 0.052, LLM-jp-4 0.094).
  Forcing immediate JSON (guided decoding) discards their reasoning and degrades
  output (e.g. LLM-jp-4 drops to 0.049); letting them reason freely overruns the
  token budget before the JSON. Models that emit an answer directly (Qwen/Llama
  with thinking disabled) do much better (0.27–0.29).
- **UTH-BERT (clinical pretraining) > NICT-BERT (Wikipedia pretraining)**,
  as expected. NICT's vocabulary also lacks many clinical kanji (they become
  `[UNK]`), a real limitation on clinical text.

### Notes on the LLM numbers

- Qwen / Llama / GPT-OSS run via **ollama with reasoning disabled** (`think:false`).
- LLM-jp-4 and SIP-jmed are reasoning models with no reliable "thinking-off"
  switch; they were run via **vLLM in natural-reasoning mode** with a large token
  budget (up to 12k). SIP-jmed still truncated on 75% of inputs; a longer budget
  did not change its score (0.018 at 6k ≈ 0.018 at 12k).
- The LLM-jp-4 12k-token run hit a vLLM detokenizer crash mid-way; the reported
  0.094 is its completed 6k-token natural run.

---

## Repository layout

```
bert/                     BERT-CRF NER pipeline (system Python: torch 2.0, transformers 4.46)
  NER_training.py           fine-tune BERT-CRF on CoNLL data (early stopping + metrics)
  build_conll.py            iCorpus char-level JSON -> CoNLL CSV (per-model tokenizer)
  extract_structured.py     raw text -> MeCab -> BERT-CRF -> structured entities
  predict_reports.py        batch prediction over real reports (cp932 CSVs)
  evaluate_ner.py           strict/soft span-level P/R/F1 from a prediction CSV
  summarize_results.py      compare configurations
  preprocess_text.py        text normalization (neologdn / NFKC / full-width)
  tokenization_mod.py       MeCab + WordPiece tokenizer (UTH-BERT official)
  lib/                      model (BERT_CRF, CRF), training loop, utils, evaluation
llm/                      LLM extraction & evaluation (venv: torch 2.6+cu124)
  testset_eval.py           BERT-vs-LLM eval on the gold corpus (relaxed metric)
  llm_extract_vllm.py       vLLM batch extraction (HF models: LLM-jp, SIP-jmed)
  llm_extract_ollama.py     ollama extraction, multi-GPU data-parallel (Qwen, Llama, GPT-OSS)
  llm_extract.py            single-request transformers baseline
results/                  aggregate metrics only (no text / no patient data)
  testset_scores.json       BERT vs LLM comparison (this repo's headline result)
  bert_UTH/ bert_NICT/      per-label strict/soft P/R/F1 + training metrics
docs/WORKFLOW.md          detailed operational notes
```

---

## Methods

### BERT (fine-tuned)

`BERT-CRF` = pretrained BERT encoder + linear head + CRF. Two encoders:

- **UTH-BERT** — pretrained on Japanese clinical text (vocab 25k; MeCab
  ipadic-neologd + J-Medic dictionary + digit splitting + WordPiece).
- **NICT-BERT** — pretrained on Japanese Wikipedia (vocab 32k; MeCab Juman
  dictionary + WordPiece).

The char-level entity annotations are converted to CoNLL (per-model tokenizer)
by aligning character spans to sub-word tokens via `difflib`
([`bert/build_conll.py`](bert/build_conll.py)); entity coverage is 95.7% (UTH) /
99.7% (NICT). Fine-tuning uses AdamW, early stopping on validation F1.

> **CRF gotcha:** the CRF assigns a −1e7 penalty to invalid BIO transitions
> (`O→I-x`, `B-x→I-y`, …). The IOB fed to it must be strictly valid BIO or the
> loss explodes (to ~1e8) while F1 stays fine. `build_conll.py` includes a BIO
> repair pass.

### LLMs (zero-shot)

Each clinical sentence is put into a prompt listing all entity labels, asking for
a JSON array `[{"label","text"}]`. Two backends:

- **ollama** (Qwen3.6-35B, Llama3.3-70B, GPT-OSS-120B) — reasoning suppressed
  via `think:false`; **4-GPU data parallelism** (one ollama instance per GPU,
  round-robin requests) for throughput; context capped at 4096 to free KV cache.
- **vLLM** (LLM-jp-4-32B, SIP-jmed-13B) — HF models; for reasoning models,
  either guided JSON decoding or natural reasoning with a large token budget.

### Unified evaluation

BERT predictions and LLM outputs are both scored against the gold char-level
annotations with the **same relaxed metric** (`bert/../llm/testset_eval.py`):
entity = `(type, surface)`; a prediction matches a gold entity if the type is
equal and one surface contains the other (NFKC-normalized). This is fair to both
token-based (BERT) and free-text (LLM) extractors and tolerant of the corpus's
fragmentary gold spans.

---

## Data

**Not included.** To reproduce you need:

- **iCorpus (症例報告コーパス)** — the annotated case-report corpus (183 docs,
  char-level entities). Research-only license; obtain from the University of
  Tokyo (医療AI開発学講座). Place under `corpus/icorpus_.../data/json/`.
- **UTH-BERT** — https://ai-health.m.u-tokyo.ac.jp/home/research/uth-bert
- **NICT-BERT** — https://alaginrc.nict.go.jp/nict-bert/
- **MANBYO (万病) dictionary** (J-Medic) for UTH tokenization.
- Real clinical reports for the extraction step (your own; patient data).

---

## Setup

```bash
# BERT pipeline (system Python)
pip install torch==2.0.1 transformers==4.46.3 seqeval pandas mecab-python3 jaconv neologdn
# + MeCab dictionaries: mecab-ipadic-neologd (UTH), mecab-jumandic-utf8 (NICT)

# LLM eval (separate venv)
pip install "vllm==0.8.5" "transformers==4.51.3" pandas requests
# + ollama (for Qwen/Llama/GPT-OSS)
```

### Run

```bash
# 1) Build training data from the corpus
python bert/build_conll.py

# 2) Fine-tune (per encoder)
python bert/NER_training.py --bert_type UTH  --bert_path <UTH-BERT>  --data_path data/csv/icorpus_UTH.csv  --patience 15
python bert/NER_training.py --bert_type NICT --bert_path <NICT-BERT> --data_path data/csv/icorpus_NICT.csv --patience 15

# 3) BERT vs LLM accuracy comparison
python llm/testset_eval.py                                   # BERT baseline
python llm/testset_eval.py --model qwen3.6:35b --engine ollama --hosts <urls>
python llm/testset_eval.py --model <hf-model> --engine vllm --tp 2

# 4) Extract from real reports
python bert/predict_reports.py --bert_type UTH --model_path <model.pt> --data_dir <reports> --out_dir <out>
```

See [`docs/WORKFLOW.md`](docs/WORKFLOW.md) for full operational detail, including
the multi-GPU ollama setup and vLLM/reasoning-model handling.

---

## License

Code: MIT (see repository). **Models, dictionaries, and corpus have their own
licenses and are not included.** Clinical/patient data is excluded by design.
