# -*- coding: utf-8 -*-
"""runs/<name>/raw.jsonl から preds.json を再生成(列挙順保持・重複除去)。推論不要。"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../shared/llm"))
from testset_eval import parse_entities, nfkc  # noqa: E402
from parse_v2 import parse_entities_v2  # noqa: E402
for name in sys.argv[1:]:
    d = os.path.join(HERE, "../runs", name)
    rows = [json.loads(l) for l in open(os.path.join(d, "raw.jsonl"), encoding="utf-8")]
    for ver, fn in (("v1", parse_entities), ("v2", lambda raw: parse_entities_v2(raw, fallback=parse_entities))):
        preds, fails = {}, 0
        for r in rows:
            ents = fn(r["raw"])
            if ents is None:
                fails += 1; preds[r["key"]] = []; continue
            seen, lst = set(), []
            for e in ents:
                t = (str(e["label"]), nfkc(e["text"]))
                if t[1] in ("", "なし") or t in seen:
                    continue
                seen.add(t); lst.append(t)
            preds[r["key"]] = lst
        out = "preds.json" if ver == "v1" else "preds_v2.json"
        json.dump(preds, open(os.path.join(d, out), "w"), ensure_ascii=False, indent=0)
        meta_p = os.path.join(d, "meta.json")
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p)); meta["parse_fail_" + ver] = fails
            json.dump(meta, open(meta_p, "w"), ensure_ascii=False, indent=2)
        print(name, ver, "parse_fail", fails)
