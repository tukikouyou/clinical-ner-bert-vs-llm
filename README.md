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

### Journal version: full comparison tables

All numbers below are generated from `journal/stats/` by `journal/code/readme_tables.py`
(also saved as [`journal/stats/tables_en.md`](journal/stats/tables_en.md)). Relaxed = type equal and
one surface a substring of the other; strict = type and NFKC-normalised surface identical; matching is
one-to-one greedy within each sentence. Figures: [`journal/paper/figs/`](journal/paper/figs/).

<details open>
<summary>Tables A–I</summary>

#### Table A. QLoRA fine-tuned models, seed 42 (test-36, 433 sentences, 12,919 gold entities)

95% CI: document-level bootstrap (36 documents, 2,000 resamples). min2/min3: relaxed F1 with a minimum substring length of 2/3 characters. Macro: mean F1 over the 48 labels with ≥ 20 test entities. Merged: relaxed F1 after merging the `-patient`/`-others` suffixes. Exact/TP: share of relaxed true positives whose surface is an exact match. Trunc.: sentences cut off at the 900-token budget.

| Model | Type | Relaxed F1 | 95% CI | P | R | Strict F1 | 95% CI | min2 | min3 | Macro | Merged | Exact/TP | Trunc./433 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | medical | 0.825 | [0.787, 0.858] | 0.924 | 0.745 | 0.793 | [0.756, 0.826] | 0.820 | 0.807 | 0.770 | 0.837 | 0.958 | 87 |
| Llama-3.3-70B | general | 0.823 | [0.786, 0.854] | 0.921 | 0.744 | 0.791 | [0.754, 0.823] | 0.818 | 0.806 | 0.777 | 0.834 | 0.958 | 86 |
| MedGemma-27B | medical | 0.824 | [0.786, 0.856] | 0.918 | 0.747 | 0.790 | [0.752, 0.823] | 0.818 | 0.804 | 0.775 | 0.836 | 0.956 | 81 |
| Gemma-3-27B (base of MedGemma) | general | 0.828 | [0.792, 0.860] | 0.923 | 0.751 | 0.796 | [0.759, 0.829] | 0.823 | 0.809 | 0.779 | 0.838 | 0.958 | 84 |
| SIP-jmed-13B | medical | 0.820 | [0.782, 0.853] | 0.919 | 0.741 | 0.787 | [0.749, 0.820] | 0.816 | 0.802 | 0.772 | 0.830 | 0.956 | 84 |
| LLM-jp-3.1-13B (base of SIP-jmed) | general | 0.816 | [0.779, 0.848] | 0.913 | 0.737 | 0.777 | [0.740, 0.810] | 0.811 | 0.795 | 0.763 | 0.826 | 0.948 | 82 |
| Qwen3-30B-A3B (MoE) | general | 0.812 | [0.774, 0.845] | 0.912 | 0.732 | 0.774 | [0.736, 0.807] | 0.805 | 0.790 | 0.760 | 0.822 | 0.951 | 85 |
| LLM-jp-4-32B (MoE, reasoning) | general | 0.813 | [0.776, 0.843] | 0.904 | 0.738 | 0.772 | [0.735, 0.803] | 0.807 | 0.790 | 0.758 | 0.823 | 0.946 | 79 |
| LLM-jp-1.8B | general | 0.777 | [0.739, 0.810] | 0.880 | 0.695 | 0.730 | [0.691, 0.765] | 0.770 | 0.752 | 0.716 | 0.789 | 0.935 | 81 |
| UTH-BERT (fine-tuned) | medical | 0.751 | [0.686, 0.807] | 0.871 | 0.661 | 0.688 | [0.628, 0.739] | 0.738 | 0.707 | 0.732 | 0.760 | 0.908 | – |
| NICT-BERT (fine-tuned) | general | 0.741 | [0.675, 0.796] | 0.860 | 0.651 | 0.710 | [0.645, 0.765] | 0.738 | 0.724 | 0.707 | 0.750 | 0.954 | – |

#### Table B. Three QLoRA seeds (42 / 1 / 2)

| Model | Type | Relaxed 42 / 1 / 2 | Mean ± SD | Strict 42 / 1 / 2 | Mean ± SD | Macro (mean) |
|---|---|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | medical | 0.825 / 0.826 / 0.829 | 0.827 ± 0.002 | 0.793 / 0.798 / 0.802 | 0.798 ± 0.004 | 0.773 |
| Llama-3.3-70B | general | 0.823 / 0.828 / 0.830 | 0.827 ± 0.003 | 0.791 / 0.801 / 0.801 | 0.798 ± 0.006 | 0.778 |
| MedGemma-27B | medical | 0.824 / 0.829 / 0.830 | 0.827 ± 0.003 | 0.790 / 0.797 / 0.798 | 0.795 ± 0.004 | 0.780 |
| Gemma-3-27B (base of MedGemma) | general | 0.828 / 0.825 / 0.828 | 0.827 ± 0.002 | 0.796 / 0.793 / 0.798 | 0.796 ± 0.002 | 0.777 |
| SIP-jmed-13B | medical | 0.820 / 0.819 / 0.817 | 0.819 ± 0.002 | 0.787 / 0.785 / 0.783 | 0.785 ± 0.002 | 0.771 |
| LLM-jp-3.1-13B (base of SIP-jmed) | general | 0.816 / 0.817 / 0.817 | 0.817 ± 0.001 | 0.777 / 0.781 / 0.781 | 0.780 ± 0.002 | 0.765 |
| Qwen3-30B-A3B (MoE) | general | 0.812 / 0.815 / 0.817 | 0.815 ± 0.002 | 0.774 / 0.781 / 0.782 | 0.779 ± 0.004 | 0.763 |
| LLM-jp-4-32B (MoE, reasoning) | general | 0.813 / 0.820 / 0.819 | 0.817 ± 0.004 | 0.772 / 0.783 / 0.783 | 0.779 ± 0.006 | 0.765 |
| LLM-jp-1.8B | general | 0.777 / 0.778 / 0.780 | 0.778 ± 0.002 | 0.730 / 0.730 / 0.733 | 0.731 ± 0.002 | 0.724 |

#### Table C. Zero-shot: prompt only vs. JSON-schema constrained decoding (relaxed / strict F1)

Parser v1 = the LAU-version parser (first `[` … last `]`); parser v2 = last complete JSON array (reasoning channel discarded). Fail: sentences without a parseable array. Unk.: predictions with a label outside the 109-label list. Trunc.: sentences cut off by the generation budget (900 tokens; 3,072 for the two reasoning models).

| Model | ZS F1 (v1) | Fail (v1) | Unk. (v1) | ZS F1 (v2) rel. / str. | Fail (v2) | Trunc. | ZS+constrained rel. / str. | P / R | Trunc. | FT F1 (s42) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 0.136 | 10 | 581 | 0.136 / 0.044 | 6 | 1 | 0.146 / 0.044 | 0.345 / 0.093 | 0 | 0.825 |
| Llama-3.3-70B | 0.276 | 0 | 188 | 0.276 / 0.160 | 0 | 2 | 0.265 / 0.157 | 0.508 / 0.179 | 3 | 0.823 |
| MedGemma-27B | 0.320 | 0 | 197 | 0.320 / 0.209 | 0 | 40 | 0.316 / 0.205 | 0.452 / 0.243 | 28 | 0.824 |
| SIP-jmed-13B | 0.010 | 322 | 47 | 0.025 / 0.014 | 92 | 147 | 0.015 / 0.003 | 0.043 / 0.009 | 108 | 0.820 |
| Qwen3-30B-A3B (MoE) | 0.235 | 3 | 236 | 0.235 / 0.139 | 3 | 24 | 0.224 / 0.133 | 0.332 / 0.169 | 46 | 0.812 |
| LLM-jp-4-32B (MoE, reasoning) | 0.092 | 261 | 95 | 0.150 / 0.068 | 17 | 35 | 0.000 / 0.000 | 0.000 / 0.000 | 1 | 0.813 |
| LLM-jp-1.8B | 0.027 | 3 | 195 | 0.027 / 0.001 | 3 | 427 | 0.019 / 0.001 | 0.044 / 0.013 | 343 | 0.777 |

#### Table D. Reasoning models: four zero-shot conditions (relaxed F1)

| Model | Prompt only, parser v1 | Prompt only, parser v2 | Whole-output constraint | Think, then constrain (rel. / str.) | Fine-tuned |
|---|:-:|:-:|:-:|:-:|:-:|
| LLM-jp-4-32B | 0.092 (fail 261) | 0.150 (fail 17) | 0.000 | 0.172 / 0.081 | 0.813 |
| SIP-jmed-13B | 0.010 (fail 322) | 0.025 (fail 92) | 0.015 | 0.048 / 0.023 | 0.820 |

#### Table E. Reference models (ollama; cannot be fine-tuned locally)

| Model | Condition | Relaxed F1 | Strict F1 | P / R | Fail | Trunc. |
|---|---|:-:|:-:|:-:|:-:|:-:|
| Qwen3.6-35B | prompt only, reasoning off | 0.300 | 0.153 | 0.439 / 0.228 | 0 | 3 |
| Qwen3.6-35B | reasoning on + schema on answer (3,072-token budget) | 0.022 | 0.017 | 0.561 / 0.011 | 310 | 310 |
| GPT-OSS-120B | prompt only (reasoning cannot be disabled) | 0.265 | 0.131 | 0.571 / 0.173 | 8 | 16 |
| GPT-OSS-120B | reasoning + schema on answer | 0.263 | 0.135 | 0.562 / 0.172 | 11 | 11 |

#### Table F. Medical model minus its general-purpose base, same seed (paired bootstrap, relaxed F1)

| Pair | Seed 42 | Seed 1 | Seed 2 |
|---|:-:|:-:|:-:|
| SIP-jmed-13B − LLM-jp-3.1-13B | +0.004 [+0.000, +0.009] | +0.002 [-0.001, +0.006] | +0.000 [-0.007, +0.006] |
| MedGemma-27B − Gemma-3-27B | -0.005 [-0.010, +0.000] | +0.004 [-0.002, +0.010] | +0.001 [-0.002, +0.005] |
| Med42-70B − Llama-3.3-70B | +0.002 [-0.002, +0.006] | -0.001 [-0.005, +0.003] | -0.001 [-0.005, +0.003] |

#### Table G. Fine-tuned LLM minus BERT, seed 42 (paired bootstrap, relaxed F1)

| LLM | vs. UTH-BERT | vs. NICT-BERT |
|---|:-:|:-:|
| Med42-70B | +0.074 [+0.045, +0.108] | +0.084 [+0.056, +0.118] |
| Llama-3.3-70B | +0.072 [+0.041, +0.108] | +0.082 [+0.052, +0.118] |
| MedGemma-27B | +0.072 [+0.042, +0.109] | +0.083 [+0.054, +0.118] |
| SIP-jmed-13B | +0.069 [+0.040, +0.103] | +0.080 [+0.051, +0.113] |
| Qwen3-30B-A3B (MoE) | +0.061 [+0.029, +0.097] | +0.071 [+0.041, +0.107] |
| LLM-jp-4-32B (MoE, reasoning) | +0.061 [+0.030, +0.098] | +0.072 [+0.041, +0.107] |
| LLM-jp-1.8B | +0.026 [-0.004, +0.061] | +0.036 [+0.007, +0.070] |

#### Table H. Effect of the 900-token generation budget (fine-tuned, seed 42, relaxed)

Trunc.: sentences whose output was cut off; Gold: share of all gold entities in those sentences; R: recall on truncated / other sentences; FN: share of the model's false negatives in truncated sentences. UTH-BERT is scored on the sentences truncated by Med42-70B.

| Model | Trunc. | Gold share | R (trunc.) | R (rest) | P (trunc.) | P (rest) | FN in trunc. |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Med42-70B | 87 | 0.53 | 0.583 | 0.927 | 0.908 | 0.937 | 0.87 |
| Llama-3.3-70B | 86 | 0.53 | 0.580 | 0.925 | 0.909 | 0.930 | 0.86 |
| MedGemma-27B | 81 | 0.51 | 0.577 | 0.920 | 0.901 | 0.930 | 0.84 |
| Gemma-3-27B | 84 | 0.52 | 0.587 | 0.927 | 0.905 | 0.936 | 0.86 |
| SIP-jmed-13B | 84 | 0.52 | 0.578 | 0.916 | 0.904 | 0.929 | 0.84 |
| LLM-jp-3.1-13B | 82 | 0.51 | 0.572 | 0.909 | 0.896 | 0.925 | 0.83 |
| Qwen3-30B-A3B | 85 | 0.52 | 0.571 | 0.907 | 0.892 | 0.926 | 0.83 |
| LLM-jp-4-32B | 79 | 0.50 | 0.571 | 0.903 | 0.886 | 0.916 | 0.81 |
| LLM-jp-1.8B | 81 | 0.50 | 0.525 | 0.867 | 0.852 | 0.898 | 0.78 |
| UTH-BERT (same split as Med42) | (87) | 0.53 | 0.486 | 0.856 | 0.858 | 0.879 | – |

#### Table I. Per-label scores of fine-tuned Med42-70B (seed 42) for nine labels; UTH = UTH-BERT relaxed / strict F1

| Label | n | P | R | F1 (relaxed) | F1 (strict) | UTH (rel. / str.) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| state-patient | 2318 | 0.92 | 0.80 | 0.85 | 0.80 | 0.72 / 0.65 |
| clinical_test | 587 | 0.95 | 0.77 | 0.85 | 0.80 | 0.71 / 0.64 |
| treatment | 614 | 0.93 | 0.67 | 0.78 | 0.73 | 0.71 / 0.66 |
| drug | 283 | 0.96 | 0.68 | 0.80 | 0.79 | 0.70 / 0.67 |
| PN-Negative-patient | 276 | 0.96 | 0.67 | 0.79 | 0.76 | 0.86 / 0.84 |
| state-others | 155 | 0.89 | 0.33 | 0.48 | 0.44 | 0.41 / 0.37 |
| device | 75 | 0.73 | 0.59 | 0.65 | 0.53 | 0.46 / 0.46 |
| time_span | 59 | 0.74 | 0.47 | 0.58 | 0.58 | 0.54 / 0.49 |
| value-others | 63 | 0.32 | 0.10 | 0.15 | 0.15 | 0.07 / 0.07 |

Spearman correlation between label frequency and relaxed F1 (Med42-70B, 98 labels): ρ = 0.59 (p = 2e-10). The 27 `-others` labels average F1 0.11.


</details>

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
