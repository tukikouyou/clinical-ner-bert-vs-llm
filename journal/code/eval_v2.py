# -*- coding: utf-8 -*-
"""
期刊版 評価スクリプト v2 (推論のみ)。vLLM で test36 (433文) を推論し、
生出力(raw)・予測表層(preds)・メタ情報を runs/<name>/ に保存する。
指標計算は metrics_v2.py で行う(保存した preds から何度でも再計算可能)。

プロンプト・パース・データ読込は shared/llm/testset_eval.py からそのまま import し、
LAU 論文と完全に同一にする。

  /opt/llm/vllm-venv/bin/python eval_v2.py --name llm-jp-1.8b-zs \
      --model llm-jp/llm-jp-3.1-1.8b-instruct4 --tp 1 [--lora PATH] [--guided] \
      [--max_new_tokens 900] [--max_model_len 4096] [--enforce_eager]

--guided : JSON Schema による制約付きデコード。label は 109 ラベルの enum に限定。
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_LLM = os.path.join(HERE, "../../../shared/llm")
sys.path.insert(0, SHARED_LLM)
from testset_eval import (PROMPT_TEMPLATE, load_alldocs, load_labels,
                          parse_entities, nfkc)  # noqa: E402

RUNS = os.path.join(HERE, "../runs")
LOAD_LOCK = "/opt/llm/.elements_load.lock"   # Elements(HDD) からの読込を全ジョブで直列化(2本同時 mmap 読みで 10MB/s に落ちる)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--lora", default=None)
    p.add_argument("--guided", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=900)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--gpu_mem", type=float, default=0.90)
    p.add_argument("--max_num_seqs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="動作確認用: 先頭N文のみ")
    p.add_argument("--think_from", default=None,
                   help="2 段階方式: 指定 run (prompt-only ZS) の raw 出力から思考部分を取り出してプロンプト末尾に付け、"
                        "最終回答のみを --guided で制約する(思考型モデル用「自由思考 + 制約回答」)")
    p.add_argument("--engine", default="vllm", choices=["vllm", "ollama"],
                   help="ollama: 参考モデル(gpt-oss:120b, qwen3.6:35b)用。--guided は format=JSON Schema で制約")
    p.add_argument("--hosts", default="http://127.0.0.1:11434")
    p.add_argument("--per_host", type=int, default=4)
    p.add_argument("--ollama_think", action="store_true",
                   help="ollama: think=true (思考を許可; --guided と併用で「自由思考 + 制約回答」。既定 think=false は LAU 論文と同一)")
    args = p.parse_args()
    if args.engine == "ollama":
        return run_ollama(args)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    data, test_keys = load_alldocs()
    keys = sorted(test_keys, key=lambda k: (k.rpartition("_")[0], int(k.rpartition("_")[2])))
    if args.limit:
        keys = keys[:args.limit]
    labels = load_labels()
    label_block = "\n".join(f"- {x}" for x in labels)
    out_dir = os.path.join(RUNS, args.name)
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    for k in keys:
        msg = [{"role": "user",
                "content": PROMPT_TEMPLATE.format(labels=label_block, text=data[k]["text"])}]
        try:
            s = tok.apply_chat_template(msg, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        prompts.append(s)
    think_meta = None
    if args.think_from:
        import re as _re
        rows = {json.loads(l)["key"]: json.loads(l) for l in open(os.path.join(RUNS, args.think_from, "raw.jsonl"), encoding="utf-8")}
        n_h = 0
        for i, k in enumerate(keys):
            raw = rows[k]["raw"]
            m = _re.search(r"<\|channel\|>\s*analysis\s*<\|message\|>(.*?)(<\|end\|>|$)", raw, _re.S)
            if m:   # Harmony (LLM-jp-4): analysis チャネルを再現し final チャネルを開く
                prompts[i] += "<|channel|>analysis<|message|>" + m.group(1).strip() + "<|end|><|start|>assistant<|channel|>final<|message|>"
                n_h += 1
            else:   # 自由文の思考 (SIP-jmed 等): 出力をそのまま assistant 発話として続け、JSON を促す
                prompts[i] += raw.rstrip() + "\n\n# 出力(JSON配列のみ)\n"
        think_meta = {"think_from": args.think_from, "harmony": n_h, "free_text": len(keys) - n_h}
        print(f"think_from={args.think_from}: harmony {n_h}, free-text {len(keys)-n_h}", flush=True)
    plen = [len(tok(s)["input_ids"]) for s in prompts]
    print(f"{len(keys)} 文 / prompt tokens mean {sum(plen)/len(plen):.0f} max {max(plen)}", flush=True)

    kw = {}
    if args.max_num_seqs:
        kw["max_num_seqs"] = args.max_num_seqs
    if args.guided:
        kw["guided_decoding_backend"] = "xgrammar:disable-any-whitespace"
    import fcntl
    lock = open(LOAD_LOCK, "w"); print("waiting for load lock...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX); t_lock = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
              trust_remote_code=True, enforce_eager=args.enforce_eager,
              enable_lora=bool(args.lora), max_lora_rank=64, **kw)
    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    print(f"model loaded in {time.time()-t_lock:.0f}s (lock released)", flush=True)
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("ft", 1, args.lora)
    gd = None
    schema = None
    if args.guided:
        schema = {"type": "array", "items": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": labels},
                           "text": {"type": "string"}},   # minLength は xgrammar 非対応(outlines に退化し極端に遅い)ため付けない。空文字は事後に除外
            "required": ["label", "text"], "additionalProperties": False}}
        # disable-any-whitespace: 思考型モデルが JSON の任意空白を延々と出力して上限に達する退化(LLM-jp-4 で 334/433 句)を防ぐ
        gd = GuidedDecodingParams(json=schema)   # backend はエンジン初期化時に指定(V1 はリクエスト単位の指定不可)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, guided_decoding=gd)

    t0 = time.time()
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    sec = round(time.time() - t0, 1)

    preds, fails, n_trunc = {}, 0, 0
    with open(os.path.join(out_dir, "raw.jsonl"), "w", encoding="utf-8") as f:
        for k, o, pl in zip(keys, outs, plen):
            c = o.outputs[0]
            raw = c.text
            ents = parse_entities(raw)
            if ents is None:
                fails += 1
                preds[k] = []
            else:
                seen, lst = set(), []   # 列挙順を保持し重複は最初の1個のみ
                for e in ents:
                    t = (str(e["label"]), nfkc(e["text"]))
                    if t[1] in ("", "なし") or t in seen:
                        continue
                    seen.add(t); lst.append(t)
                preds[k] = lst
            if c.finish_reason == "length":
                n_trunc += 1
            f.write(json.dumps({"key": k, "prompt_tokens": pl, "out_tokens": len(c.token_ids),
                                "finish": c.finish_reason, "parse_ok": ents is not None,
                                "raw": raw}, ensure_ascii=False) + "\n")
    json.dump(preds, open(os.path.join(out_dir, "preds.json"), "w"),
              ensure_ascii=False, indent=0)
    meta = {"name": args.name, "model": args.model, "lora": args.lora, "tp": args.tp,
            "guided": args.guided, "schema": schema, "guided_backend": "xgrammar:disable-any-whitespace" if args.guided else None, "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len, "n_sent": len(keys), "parse_fail": fails, "think": think_meta,
            "truncated": n_trunc, "sec": sec, "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[{args.name}] done: parse_fail={fails} truncated={n_trunc} {sec}s → {out_dir}", flush=True)


def run_ollama(args):
    """ollama /api/chat で test36 を推論。think=false, temperature 0 は LAU 論文の run_llm と同一。
    --guided 時は format に JSON Schema(label enum 109) を渡す(ollama は思考部を除く最終回答のみ制約)。"""
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    data, test_keys = load_alldocs()
    keys = sorted(test_keys, key=lambda k: (k.rpartition("_")[0], int(k.rpartition("_")[2])))
    if args.limit:
        keys = keys[:args.limit]
    labels = load_labels()
    label_block = "\n".join(f"- {x}" for x in labels)
    out_dir = os.path.join(RUNS, args.name); os.makedirs(out_dir, exist_ok=True)
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    schema = None
    if args.guided:
        schema = {"type": "array", "items": {"type": "object",
                  "properties": {"label": {"type": "string", "enum": labels}, "text": {"type": "string"}},
                  "required": ["label", "text"], "additionalProperties": False}}

    def work(i):
        k = keys[i]
        body = {"model": args.model, "stream": False, "think": bool(args.ollama_think),
                "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(labels=label_block, text=data[k]["text"])}],
                "options": {"temperature": 0.0, "num_predict": args.max_new_tokens, "num_ctx": args.max_model_len}}
        if schema:
            body["format"] = schema
        try:
            r = requests.post(hosts[i % len(hosts)].rstrip("/") + "/api/chat", json=body, timeout=3600).json()
            if "error" in r:   # 例: think=false 非対応モデル → 黙って空出力にせず記録
                return i, f"__ERROR__ {r['error']}", None, "error", ""
            m = r.get("message", {})
            return i, m.get("content", ""), r.get("eval_count"), r.get("done_reason"), m.get("thinking", "")
        except Exception as e:
            return i, f"__ERROR__ {e}", None, "error", ""

    t0 = time.time(); preds, fails, n_trunc = {}, 0, 0; rows = {}
    part = os.path.join(out_dir, "raw.partial.jsonl")   # 停電対策: 1 句ずつ追記し、再実行時は済んだ句を飛ばす

    def ingest(i, raw, ntok, reason, thinking):
        nonlocal fails, n_trunc
        ents = parse_entities(raw) if not raw.startswith("__ERROR__") else None
        if ents is None:
            fails += 1; lst = []
        else:
            seen, lst = set(), []
            for e in ents:
                t = (str(e["label"]), nfkc(e["text"]))
                if t[1] in ("", "なし") or t in seen:
                    continue
                seen.add(t); lst.append(t)
        if reason == "length":
            n_trunc += 1
        preds[keys[i]] = lst
        rows[keys[i]] = {"key": keys[i], "out_tokens": ntok, "finish": reason, "parse_ok": ents is not None,
                         "raw": raw, "thinking": thinking}

    kidx = {k: i for i, k in enumerate(keys)}
    if os.path.exists(part):
        for l in open(part, encoding="utf-8"):
            r = json.loads(l)
            if r["key"] in kidx and not r["raw"].startswith("__ERROR__"):
                ingest(kidx[r["key"]], r["raw"], r["out_tokens"], r["finish"], r.get("thinking", ""))
        print(f"resume: {len(rows)} rows from {part}", flush=True)
    pf = open(part, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=len(hosts) * args.per_host) as ex:
        futs = [ex.submit(work, i) for i in range(len(keys)) if keys[i] not in rows]
        for n, fu in enumerate(as_completed(futs), 1):
            i, raw, ntok, reason, thinking = fu.result()
            ingest(i, raw, ntok, reason, thinking)
            pf.write(json.dumps(rows[keys[i]], ensure_ascii=False) + "\n"); pf.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(futs)} {time.time()-t0:.0f}s", flush=True)
    pf.close()
    sec = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "raw.jsonl"), "w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    json.dump(preds, open(os.path.join(out_dir, "preds.json"), "w"), ensure_ascii=False, indent=0)
    meta = {"name": args.name, "model": args.model, "engine": "ollama", "guided": args.guided, "schema": schema, "think": bool(args.ollama_think),
            "max_new_tokens": args.max_new_tokens, "num_ctx": args.max_model_len, "n_sent": len(keys),
            "parse_fail": fails, "truncated": n_trunc, "sec": sec, "date": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), ensure_ascii=False, indent=2)
    os.remove(part)
    print(f"[{args.name}] done: parse_fail={fails} truncated={n_trunc} {sec}s → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
