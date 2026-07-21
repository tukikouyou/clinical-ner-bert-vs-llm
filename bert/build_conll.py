# -*- coding: utf-8 -*-
"""
iCorpus(症例報告コーパス, 文字レベルJSON) → CoNLL形式CSV へ変換。

UTH / NICT それぞれのトークナイザで別々のCSVを生成する。
- UTH : preprocess(neologdn等) + MeCab(ipadic-neologd + 万病辞書) + 数字1文字分割 + WordPiece
- NICT: 全角化 + MeCab(juman辞書) + WordPiece

出力列: serial, word, IOB, name, unique_no, rel_type
  (NER学習に必要なのは word/IOB/name/unique_no/rel_type。rel_typeは"None"固定)

使い方:
    python3 build_conll.py --corpus ../corpus/icorpus_20220531 \
        --out_uth data/csv/icorpus_UTH.csv --out_nict data/csv/icorpus_NICT.csv
"""
import os
import json
import glob
import argparse
import pandas as pd

from lib.align_tokenize import OffsetTokenizer, entities_to_iob


def convert(corpus_dir, bert_type, vocab_file, out_csv, **kw):
    tok = OffsetTokenizer(bert_type, vocab_file, **kw)
    rows = []
    serial = 0
    covered = total = 0
    for f in sorted(glob.glob(os.path.join(corpus_dir, "data/json/*.json"))):
        name = os.path.basename(f)                 # 例: 001_2.json
        stem = name[:-5] if name.endswith(".json") else name
        for rec in json.load(open(f, encoding="utf-8")):
            sid = rec["sentence_id"]
            uno = f"{stem}_{sid}"
            two = tok.tokenize_with_offsets("".join(rec["chars"]))
            if not two:
                continue
            # iCorpusのendは閉区間(最後の文字位置)。ここで end+1 に正規化して
            # 以降の [start,end) 前提の処理を正しくする(単字entityも幅1として含む)。
            ents = [{**e, "end": e["end"] + 1} for e in rec["entities"]]
            labels = entities_to_iob(two, ents)
            for (word, _s, _e), iob in zip(two, labels):
                rows.append((serial, word, iob, name, uno, "None"))
                serial += 1
            # カバレッジ集計
            spans = _decode(two, labels)
            for e in ents:
                total += 1
                if any(p[0] == e["type"] and p[1] < max(e["end"], e["start"] + 1)
                       and p[2] > e["start"] for p in spans):
                    covered += 1
    df = pd.DataFrame(rows, columns=["serial", "word", "IOB", "name", "unique_no", "rel_type"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[{bert_type}] {out_csv}: {len(df)} tokens, {df['name'].nunique()} docs, "
          f"{df['unique_no'].nunique()} sentences, {df['IOB'].nunique()} tags")
    print(f"[{bert_type}] entity coverage: {covered}/{total} = {100*covered/max(total,1):.2f}%")
    return df


def _decode(two, labels):
    spans, cur, s, e = [], None, None, None
    for (t, ts, te), lab in zip(two, labels):
        if lab.startswith("B-"):
            if cur:
                spans.append((cur, s, e))
            cur, s, e = lab[2:], ts, te
        elif lab.startswith("I-") and cur == lab[2:]:
            e = te
        else:
            if cur:
                spans.append((cur, s, e))
            cur = None
    if cur:
        spans.append((cur, s, e))
    return spans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="../corpus/icorpus_20220531")
    p.add_argument("--out_uth", default="data/csv/icorpus_UTH.csv")
    p.add_argument("--out_nict", default="data/csv/icorpus_NICT.csv")
    p.add_argument("--uth_vocab", default="../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K/vocab.txt")
    p.add_argument("--nict_vocab", default="../NICT_BERT-base_JapaneseWikipedia_32K_BPE/vocab.txt")
    p.add_argument("--neologd_dic", default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    p.add_argument("--manbyo_dic", default="./resources/MANBYO_201907_Dic-utf8.dic")
    p.add_argument("--juman_dic", default="/var/lib/mecab/dic/juman-utf8")
    p.add_argument("--only", choices=["UTH", "NICT"], default=None)
    args = p.parse_args()

    if args.only in (None, "UTH"):
        convert(args.corpus, "UTH", args.uth_vocab, args.out_uth,
                neologd_dic=args.neologd_dic, manbyo_dic=args.manbyo_dic)
    if args.only in (None, "NICT"):
        convert(args.corpus, "NICT", args.nict_vocab, args.out_nict,
                juman_dic=args.juman_dic)


if __name__ == "__main__":
    main()
