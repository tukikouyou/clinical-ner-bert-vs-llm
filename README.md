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

## Journal version (October 2026): extended experiments

The LAU-16 study below was extended for a journal submission (MDPI *Machine Learning and Knowledge
Extraction*, Special Issue "Language Acquisition and Understanding"). Everything for the extension is
under [`journal/`](journal/) (code, per-run metadata, all statistics, figures); the full comparison
tables are in [`journal/stats/tables.md`](journal/stats/tables.md) and, with a glossary and
commentary, [`journal/stats/tables_annotated.md`](journal/stats/tables_annotated.md).

What was added: a deterministic matching procedure (the earlier greedy matching depended on
Python's hash seed), **strict (exact-span) F1** next to the relaxed metric, document-level bootstrap
confidence intervals and paired tests, **three QLoRA seeds per model**, the two general-purpose
**base models** of SIP-jmed-13B (LLM-jp-3.1-13B) and MedGemma-27B (Gemma-3-27B), **JSON-schema
constrained decoding** and a two-pass "think, then constrain" procedure for the reasoning models, a
corrected JSON parser, and two reference models (Qwen3.6-35B, GPT-OSS-120B).

**Fine-tuned (QLoRA), test-36, mean ± SD over seeds 42/1/2**

| Model | Type | Relaxed F1 | Strict F1 |
|---|---|:---:|:---:|
| Med42-70B | medical | 0.827 ± 0.002 | 0.798 ± 0.004 |
| Llama-3.3-70B | general | 0.827 ± 0.003 | 0.798 ± 0.006 |
| MedGemma-27B | medical | 0.827 ± 0.003 | 0.795 ± 0.004 |
| Gemma-3-27B (base of MedGemma) | general | 0.827 ± 0.002 | 0.796 ± 0.002 |
| SIP-jmed-13B | medical | 0.819 ± 0.002 | 0.785 ± 0.002 |
| LLM-jp-3.1-13B (base of SIP-jmed) | general | 0.817 ± 0.001 | 0.780 ± 0.002 |
| LLM-jp-4-32B (MoE) | general | 0.817 ± 0.004 | 0.779 ± 0.006 |
| Qwen3-30B-A3B (MoE) | general | 0.815 ± 0.002 | 0.779 ± 0.004 |
| LLM-jp-1.8B | general | 0.778 ± 0.002 | 0.731 ± 0.002 |
| UTH-BERT (fine-tuned) | medical | 0.751 | 0.688 |
| NICT-BERT (fine-tuned) | general | 0.741 | 0.710 |

**Zero-shot, relaxed F1 (test-36)**: prompt only → with JSON-schema constrained decoding:
MedGemma-27B 0.320 → 0.316, Llama-3.3-70B 0.276 → 0.265, Qwen3-30B-A3B 0.235 → 0.224,
Med42-70B 0.136 → 0.146, LLM-jp-1.8B 0.027 → 0.019; reasoning models LLM-jp-4-32B 0.092 (old parser)
/ 0.150 (fixed parser) / 0.000 (whole-output constraint) / 0.172 (think, then constrain),
SIP-jmed-13B 0.010 / 0.025 / 0.015 / 0.048. Reference: Qwen3.6-35B 0.300, GPT-OSS-120B 0.265.

**Medical vs. its own base, same seed (paired bootstrap, relaxed F1, medical − base)**:
13B +0.004 / +0.002 / +0.000, 27B −0.005 / +0.004 / +0.001, 70B +0.002 / −0.001 / −0.001;
every 95% interval includes zero.

Corrected conclusions (they supersede the "Takeaways" of the LAU version further down):

- The zero-shot gap is **not** an output-format problem: constrained decoding removes all parse
  failures and out-of-schema labels but changes F1 by at most 0.02, and the best zero-shot score of a
  reasoning model with free reasoning and a constrained answer is 0.17. The 66–74 % "parse failures"
  of the reasoning models were mostly an artifact of the old parser (first `[` … last `]`).
- After fine-tuning, every model with ≥ 13B parameters is equivalent (seed SD ≤ 0.006, larger than
  the differences between models); the single-seed ranking of the LAU version was noise.
- **Medical-domain pretraining has no detectable effect** once the base model is held fixed and the
  models are fine-tuned on the same data (three pairs, three seeds each). The earlier suggestion that
  medical pretraining lets 13B match 70B is withdrawn.
- The LLM–BERT gap is larger under strict matching (+0.10) than under relaxed matching (+0.07).
- The remaining recall deficit of fine-tuned LLMs sits in the ~20 % of sentences whose JSON output
  exceeds the 900-token budget (they hold 52 % of the gold entities; recall 0.58 there vs. 0.92
  elsewhere) and in labels with a non-patient subject (`-others`).

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
| **UTH-BERT** (fine-tuned) | encoder, clinical pretraining | **0.752** | 0.871 | 0.661 |
| **NICT-BERT** (fine-tuned) | encoder, Wikipedia pretraining | **0.742** | 0.861 | 0.651 |
| Qwen3.6 35B | LLM, zero-shot | 0.288 | 0.446 | 0.212 |
| Llama3.3 70B | LLM, zero-shot | 0.261 | 0.522 | 0.174 |
| Qwen3-30B-A3B | LLM, zero-shot | 0.244 | 0.316 | 0.199 |
| LLM-jp-4 32B | LLM, zero-shot (reasoning) | 0.068 | 0.324 | 0.038 |
| GPT-OSS 120B | LLM, zero-shot (reasoning) | 0.040 | 0.606 | 0.021 |
| SIP-jmed-LLM-3 13B | LLM, zero-shot (reasoning, **medical**) | 0.012 | 0.119 | 0.006 |

Metric: relaxed entity match (type + surface containment, NFKC). BERT on the
held-out 36-doc test split; LLMs run zero-shot (reasoning models with thinking
enabled and a large token budget — see [Notes](#notes-on-the-llm-numbers)).
Full numbers: [`results/testset_scores.json`](results/testset_scores.json).

### Fine-tuning the LLMs (QLoRA) — the ranking reverses

The same LLMs, QLoRA-fine-tuned on the **same 146-document training split** as
BERT (SFT: prompt → gold-JSON) and evaluated identically:

| Method | Zero-shot F1 | **Fine-tuned F1** | Δ |
|--------|:-----------:|:-----------------:|:--:|
| **Llama-3.3 70B** | 0.261 | **0.822** *(best)* | +0.561 |
| **SIP-jmed-LLM-3 13B** (medical, 13B only) | 0.012 *(zero-shot last)* | **0.819** | **+0.807** |
| **Qwen3-30B-A3B** | 0.244 | **0.812** | +0.568 |
| **LLM-jp-4 32B** | 0.068 | **0.811** | +0.743 |
| **LLM-jp-3.1 1.8B** | ≈0 | **0.777** | +0.78 |
| UTH-BERT (fine-tuned encoder) | — | 0.752 | — |
| NICT-BERT (fine-tuned encoder) | — | 0.742 | — |

**Fine-tuning reverses the entire ranking.** Zero-shot, BERT (0.752) beats every LLM and
the medical model is *worst* (0.012); after QLoRA, **all 5 LLMs (0.777–0.822) beat BERT**.
Even a tiny 1.8B model (≈0 zero-shot) reaches 0.777. Most striking: **the 13B medical
SIP-jmed (0.819) essentially ties the 70B general Llama-3.3 (0.822)** — domain pretraining
lets 13B match 70B. The zero-shot gap was about **task/format adaptation, not capability** —
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
- **The medical-specialized model scored *worst* zero-shot***. SIP-jmed-LLM-3 (13B,
  Japanese clinical) landed at 0.012 — last among all LLMs — because it is a
  *reasoning-tuned* model that rambles indefinitely and fails to emit parseable
  JSON for **~74% of inputs**. Domain knowledge does not help when the model cannot
  produce structured output. (Yet after fine-tuning it becomes tied-for-best.)
- **"Thinking"/reasoning LLMs are a poor fit for structured extraction** and
  cluster at the bottom (SIP-jmed 0.012, GPT-OSS 0.040, LLM-jp-4 0.068).
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
