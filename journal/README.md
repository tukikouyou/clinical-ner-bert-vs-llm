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
| `code/make_tables.py`, `code/annotate_tables.py`, `code/plots.py` | tables (`stats/tables.md`, `stats/tables_annotated.md`) and figures (`paper/figs/`) |
| `code/run_queue2.sh`, `code/run_train_queue2.sh`, `code/chain_next.sh`, `code/stage.sh` | job queues used on the 4×A6000 workstation |
| `runs/<run>/` | per-run `meta.json` (settings, parse failures, truncation), `preds.json` (parsed (label, text) pairs per sentence), `raw.jsonl` (raw model output per sentence) |
| `stats/<run>.json` | all metrics for one run; `stats/pairwise_*.json` paired bootstrap differences; `stats/extra.json` |
| `paper/` | manuscript (`main.tex`, lualatex) |

Run names: `<model>-zs` (prompt only), `<model>-zs-guided` (constrained decoding),
`<model>-zs-think-guided` (two-pass), `<model>-ft` (QLoRA seed 42), `<model>-s1-ft` / `-s2-ft`
(seeds 1, 2), `<model>-ft-4k` (4,096-token generation budget), `*-guided-ws` (constrained decoding
with whitespace allowed; supplementary).

Note: in this public repository `runs/<run>/` contains only `meta.json`; `preds.json` and
`raw.jsonl` (model outputs, which contain entity surfaces from the corpus) are withheld under the
corpus licence and are available from the authors on request.
