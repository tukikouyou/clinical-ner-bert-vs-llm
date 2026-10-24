# Zero-shot and fine-tuned LLMs for fine-grained Japanese clinical NER (journal version)

Code, statistics and model outputs for the journal extension of our LAU-16 paper
(Han, Yagahara, Hirata, Yoshida, Kudo). Corpus: iCorpus (University of Tokyo), fold 0,
146 training / 36 test documents (433 sentences, 113 entity types). The corpus itself is
distributed for research use by its authors and is **not** included here.

## Layout

| Path | Content |
|---|---|
| `code/eval_v2.py` | vLLM / ollama inference on the test set (zero-shot, constrained decoding `--guided`, two-pass `--think_from`, fine-tuned `--lora`) |
| `code/parse_v2.py` | final-array JSON parser (parser v2); `code/reparse.py` re-scores raw outputs |
| `code/metrics_v2.py` | deterministic greedy matching, relaxed / strict / min-length / merged-subject / macro F1, document-level bootstrap CIs, paired differences |
| `code/qlora_train_v2.py` | QLoRA fine-tuning (identical hyperparameters, `--seed`, step checkpoints) |
| `code/extra_stats.py` | truncation analysis, per-label scores, frequency correlation |
| `code/make_tables.py`, `code/annotate_tables.py`, `code/readme_tables.py`, `code/plots.py` | tables (`stats/tables.md`; with glossary and commentary `stats/tables_annotated.md`; English `stats/tables_en.md`) and figures (`paper/figs/`) |
| `code/run_queue2.sh`, `code/run_train_queue2.sh`, `code/chain_next.sh`, `code/stage.sh` | job queues used on the 4×A6000 workstation |
| `runs/<run>/` | per-run `meta.json` (settings, parse failures, truncation), `preds.json` (parsed (label, text) pairs per sentence), `raw.jsonl` (raw model output per sentence) |
| `stats/<run>.json` | all metrics for one run; `stats/pairwise_*.json` paired bootstrap differences; `stats/extra.json` |
| `paper/` | manuscript (`main.tex`, lualatex) |
| `bert_preds/` | token-level predictions of the two fine-tuned BERT models on the fold-0 test set (source of `stats/BERT-*.json`) |
| `lau_original/` | copies of the LAU-16 materials: Japanese paper, reviewer reports and responses, LAU-phase scores |

Run names: `<model>-zs` (prompt only), `<model>-zs-guided` (constrained decoding, same budget as `-zs`; `-zs-guided-900` = superseded 900-token runs of three models),
`<model>-zs-think-guided` (two-pass), `<model>-ft` (QLoRA seed 42), `<model>-s1-ft` / `-s2-ft`
(seeds 1, 2) under the 900-token budget of the LAU paper; `<model>-ft-4k`, `-s1-ft-4k`, `-s2-ft-4k`
(**main condition**: 4,096-token generation budget; 2,600 for the two LLM-jp-3.1 models); `sip-jmed-13b*-ft-2k6` (SIP-jmed at the 2,600-token budget of its base model, for the budget-matched pair); `*-guided-ws`
(constrained decoding with whitespace allowed; supplementary). BERT: `stats/BERT-<UTH|NICT>-v2-s<seed>.json` are the
retrained encoders used in the paper (`code/bert_train_v2.py`: windowed long sentences, same 1,672/88 split as the LLMs). Fine-tuned outputs are parsed as the first
complete JSON array (`preds.json`; the LAU parser's result is kept in `preds_v1.json`), zero-shot outputs
with the LAU parser (`preds.json`) and the final-array parser (`preds_v2.json`).

Note: `raw.jsonl` and `preds.json` contain entity surfaces, i.e. short substrings of the test
sentences, but not the sentences themselves.
