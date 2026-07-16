# -*- coding: utf-8 -*-
"""
iCorpus テスト分割(gold有)での BERT vs LLM 精度比較。

全手法を同一の緩和指標で評価: エンティティ (type, 表層) の集合一致。
表層は NFKC 正規化して全角/半角差を吸収。gold は corpus JSON の真値。

- BERT(UTH/NICT): 予測 CSV(results/NER/<t>/NER/0.csv) の pred_IOB から (type,表層) を復元
- LLM: 各テスト文をプロンプト→ollama(4GPU分散)→(label,表層) をパース

使い方:
    # BERTのみ(LLM無し)で gold/BERT の F1 を表示
    python3 testset_eval.py
    # LLM を1つ評価して結果を追記
    python3 testset_eval.py --model qwen3.6:35b \
        --hosts http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436,http://127.0.0.1:11437
"""
import os
import re
import json
import glob
import time
import argparse
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(ROOT, "corpus/icorpus_20220531/data/json")
SPLIT = os.path.join(ROOT, "code/data/index/train_test_index_1fold.json")
TAG2IDX = os.path.join(ROOT, "code/results/NER/UTH/NER/NER_tag2idx_0.json")
RESULT_JSON = os.path.join(os.path.dirname(__file__), "results/ehr/testset_scores.json")

PROMPT_TEMPLATE = """あなたは日本語の臨床テキストからの固有表現抽出(NER)を行う専門家です。
以下の臨床テキストから、指定されたラベルに該当する表現をすべて抽出してください。

# ラベル一覧
{labels}

# 抽出ルール
- テキスト中に出現した表現をそのまま抜き出すこと(言い換え・要約をしない)
- 文や節ではなく、最小の固有表現(単語〜短い句)を抜き出すこと
- テキスト中に出現したラベルだけを出力すること(空文字や「なし」は出力しない)
- 該当が無ければ空配列 [] を返すこと
- 出力は必ず次形式のJSON配列のみ。説明文を付けない:
[{{"label": "ラベル名", "text": "抽出した表現"}}, ...]

# 臨床テキスト
{text}

# 出力(JSON配列のみ)
"""


def nfkc(s):
    return unicodedata.normalize("NFKC", str(s)).strip()


def load_labels():
    t = json.load(open(TAG2IDX, encoding="utf-8"))
    return sorted({x[2:] for x in t if x.startswith(("B-", "I-"))})


def load_alldocs():
    """全183文書を読み込む。
    戻り値: (data, test_keys)
      data: {sent_key: {"text":..., "gold": set((type, nfkc(surface)))}}
      test_keys: BERTのテスト分割(36文書)に属する sent_key の集合
    """
    test_docs = set(json.load(open(SPLIT))["test_name"])
    data, test_keys = {}, set()
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.json"))):
        stem = os.path.basename(f).replace(".json", "")
        is_test = f"{stem}.json" in test_docs
        for rec in json.load(open(f)):
            chars = rec["chars"]
            key = f"{stem}_{rec['sentence_id']}"
            gold = set()
            for e in rec["entities"]:
                if e["end"] > e["start"]:
                    gold.add((e["type"], nfkc("".join(chars[e["start"]:e["end"]]))))
            data[key] = {"text": "".join(chars), "gold": gold}
            if is_test:
                test_keys.add(key)
    return data, test_keys


def bert_pred(bert_type):
    """BERTの pred_IOB CSV から {sent_key: set((type, nfkc(surface)))}"""
    csv = os.path.join(ROOT, f"code/results/NER/{bert_type}/NER/0.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    out = {}
    for uno, g in df.groupby("unique_no"):
        ents, cur, buf = set(), None, ""
        words, labs = list(g["word"].astype(str)), list(g["pred_IOB"])
        def flush():
            if cur:
                ents.add((cur, nfkc(buf)))
        for w, lab in zip(words, labs):
            w = w[2:] if w.startswith("##") else w
            if lab.startswith("B-"):
                flush(); cur, buf = lab[2:], w
            elif lab.startswith("I-") and cur == lab[2:]:
                buf += w
            else:
                flush(); cur, buf = None, ""
        flush()
        out[uno] = ents
    return out


def _match_sentence(preds, golds):
    """同一文内で貪欲に含意マッチ(同type かつ 一方が他方の部分文字列)。
    corpus goldは断片的なので、境界の緩い部分一致で公平に評価する。
    戻り値: (tp, fp, fn)"""
    golds = list(golds)
    used = [False] * len(golds)
    tp = 0
    for pt, ps in preds:
        for j, (gt, gs) in enumerate(golds):
            if not used[j] and pt == gt and (gs in ps or ps in gs):
                used[j] = True
                tp += 1
                break
    return tp, len(preds) - tp, len(golds) - tp


def prf(pred_by_sent, gold_by_sent, keys=None):
    tp = fp = fn = 0
    for key in (keys if keys is not None else gold_by_sent):
        g = gold_by_sent[key]
        p = pred_by_sent.get(key, set())
        a, b, c = _match_sentence(p, g)
        tp += a; fp += b; fn += c
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
            "tp": tp, "fp": fp, "fn": fn}


def parse_entities(raw):
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    cands = [m.group(1)] if m else []
    m2 = re.search(r"\[.*\]", raw, re.DOTALL)
    if m2:
        cands.append(m2.group(0))
    if not cands and "[" in raw:
        cands.append(raw[raw.index("["):])
    for c in cands:
        for a in (c, (c[:c.rfind('}') + 1] + "]") if "}" in c else None):
            if a is None:
                continue
            try:
                o = json.loads(a)
                if isinstance(o, list):
                    return [x for x in o if isinstance(x, dict)
                            and "label" in x and "text" in x]
            except json.JSONDecodeError:
                continue
    return None


def run_llm(model, hosts, per_host, data, label_block, max_tokens):
    keys = list(data)
    preds = {}
    fails = 0

    def work(idx):
        key = keys[idx]
        prompt = PROMPT_TEMPLATE.format(labels=label_block, text=data[key]["text"])
        host = hosts[idx % len(hosts)]
        body = {"model": model, "stream": False, "think": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.0, "num_predict": max_tokens, "num_ctx": 4096}}
        try:
            r = requests.post(host.rstrip("/") + "/api/chat", json=body, timeout=600)
            return idx, r.json()["message"].get("content", "")
        except Exception as e:
            return idx, f"__ERROR__ {e}"

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=len(hosts) * per_host) as ex:
        futs = [ex.submit(work, i) for i in range(len(keys))]
        for fu in as_completed(futs):
            idx, raw = fu.result()
            ents = parse_entities(raw) if raw and not raw.startswith("__ERROR__") else None
            if ents is None:
                fails += 1
                preds[keys[idx]] = set()
            else:
                preds[keys[idx]] = {(str(e["label"]), nfkc(e["text"])) for e in ents
                                    if nfkc(e["text"]) not in ("", "なし")}
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(keys)}  {done/(time.time()-t0):.2f} sent/s")
    return preds, fails, round(time.time() - t0, 1)


def run_llm_vllm(model_path, tp, data, label_block, max_tokens, max_model_len,
                 guided=False, prefill=False, enforce_eager=False, lora=None):
    """vLLM で一括生成(HF形式モデル: llm-jp-4, SIP-jmed 等)。
    guided=True: JSONスキーマで制約(パース失敗を防ぐが思考を殺す)。
    prefill=True: 空<think></think>で思考スキップ。
    lora=<path>: QLoRAアダプタを付けて評価(ファインチューニング済みモデル)。"""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    keys = list(data)
    # tokenizerは常にbaseから(QLoRAは分詞器を変えない。アダプタ側はFT環境の
    # 新しいtransformersで保存されており評価環境で読めないことがある)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = []
    for k in keys:
        msg = [{"role": "user",
                "content": PROMPT_TEMPLATE.format(labels=label_block, text=data[k]["text"])}]
        try:  # 推論モデル(Qwen3系)は enable_thinking=False で思考を抑制
            s = tok.apply_chat_template(msg, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        if prefill:  # 空の<think></think>をプリフィルして思考をスキップさせる
            s += "<think>\n\n</think>\n\n"
        prompts.append(s)
    # enforce_eager: torch.compile/cudagraph を無効化(複数vLLM同時実行時のtriton
    # キャッシュ競合回避用。単独実行なら不要でcudagraphの方が高速)
    llm = LLM(model=model_path, dtype="bfloat16", tensor_parallel_size=tp,
              gpu_memory_utilization=0.90, max_model_len=max_model_len,
              trust_remote_code=True, enforce_eager=enforce_eager,
              enable_lora=bool(lora), max_lora_rank=64)
    lora_req = None
    if lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("ft", 1, lora)
    gd = None
    if guided:
        # text は minLength=1 で空文字を禁止(思考モデルの「全ラベル空文字列挙」退化を防ぐ)
        schema = {"type": "array", "items": {
            "type": "object",
            "properties": {"label": {"type": "string"},
                           "text": {"type": "string", "minLength": 1}},
            "required": ["label", "text"]}}
        gd = GuidedDecodingParams(json=schema)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_decoding=gd)
    t0 = time.time()
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    sec = round(time.time() - t0, 1)
    preds, fails = {}, 0
    for k, o in zip(keys, outs):
        ents = parse_entities(o.outputs[0].text)
        if ents is None:
            fails += 1
            preds[k] = set()
        else:
            preds[k] = {(str(e["label"]), nfkc(e["text"])) for e in ents
                        if nfkc(e["text"]) not in ("", "なし")}
    return preds, fails, sec


def save_score(name, score, extra=None):
    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    allsc = json.load(open(RESULT_JSON)) if os.path.exists(RESULT_JSON) else {}
    allsc[name] = {**score, **(extra or {})}
    json.dump(allsc, open(RESULT_JSON, "w"), ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="LLMモデル(ollama名 or HFパス)。省略時はBERTのみ")
    p.add_argument("--name", default=None, help="結果保存名(省略時は--model)")
    p.add_argument("--engine", default="ollama", choices=["ollama", "vllm"])
    p.add_argument("--hosts", default="http://127.0.0.1:11434")
    p.add_argument("--per_host", type=int, default=3)
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor_parallel_size")
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_new_tokens", type=int, default=900)
    p.add_argument("--guided", action="store_true", help="vLLM: JSONスキーマ強制")
    p.add_argument("--prefill", action="store_true", help="vLLM: 空<think>で思考スキップ")
    p.add_argument("--enforce_eager", action="store_true",
                   help="vLLM: cudagraph無効(複数vLLM同時実行時のみ推奨)")
    p.add_argument("--lora", default=None, help="vLLM: QLoRAアダプタのパス(FT評価)")
    args = p.parse_args()

    data, test_keys = load_alldocs()
    gold = {k: v["gold"] for k, v in data.items()}
    print(f"全{len(data)}文 (gold実体{sum(len(g) for g in gold.values())}) / "
          f"うちBERTテスト分割 {len(test_keys)}文")

    # BERT(テスト分割36文書のみ = 公平な held-out)
    for bt in ("UTH", "NICT"):
        bp = bert_pred(bt)
        if bp:
            sc = prf(bp, gold, test_keys)  # テスト分割で評価
            save_score(f"BERT-{bt}", sc,
                       {"scope": "test36", "note": "finetuned held-out, relaxed(type+surface)"})
            print(f"[BERT-{bt}] test36 relaxed F1={sc['f1']} "
                  f"(P={sc['precision']} R={sc['recall']})")

    # LLM(指定時): 全183文で実行し、全183とテスト36の両方で集計
    if args.model:
        name = args.name or args.model
        label_block = "\n".join(f"- {x}" for x in load_labels())
        if args.engine == "vllm":
            mode = "guided" if args.guided else ("prefill" if args.prefill else "natural")
            print(f"LLM(vLLM/{mode}) {args.model} tp={args.tp} を全{len(data)}文で実行...")
            preds, fails, sec = run_llm_vllm(args.model, args.tp, data, label_block,
                                             args.max_new_tokens, args.max_model_len,
                                             guided=args.guided, prefill=args.prefill,
                                             enforce_eager=args.enforce_eager,
                                             lora=args.lora)
        else:
            hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
            print(f"LLM(ollama) {args.model} を全{len(data)}文 / {len(hosts)}×{args.per_host}並列...")
            preds, fails, sec = run_llm(args.model, hosts, args.per_host, data,
                                        label_block, args.max_new_tokens)
        sc_all = prf(preds, gold)                 # 全183
        sc_test = prf(preds, gold, test_keys)     # テスト36(BERTと同一データ)
        save_score(name, sc_all,
                   {"scope": "all183", "engine": args.engine, "parse_fail": fails,
                    "sec": sec, "test36_f1": sc_test["f1"],
                    "test36": sc_test, "note": "zero-shot, relaxed(type+surface)"})
        print(f"[{name}] all183 F1={sc_all['f1']} (P={sc_all['precision']} R={sc_all['recall']}) | "
              f"test36 F1={sc_test['f1']} (P={sc_test['precision']} R={sc_test['recall']}) | "
              f"parse失敗{fails} {sec}s")


if __name__ == "__main__":
    main()
