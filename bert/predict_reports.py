# -*- coding: utf-8 -*-
"""
学習済み BERT-CRF で実臨床レポート(predicted data/*.csv)から固有表現を抽出。

学習と同一の OffsetTokenizer を使うため、train/predict のトークン化が一致する。
UTH / NICT どちらのモデルにも対応。cp932 の複数列CSVを読み、指定列を対象にする。

出力(--out_dir):
    entities.csv     : 抽出エンティティ [doc_id, year, column, label, text, char_start, char_end]
    token_pred.parquet: トークン単位の予測(任意, --save_tokens 指定時)
    stats.json       : 抽出統計(ラベル別件数, 列別/年別件数, 空予測率, 処理時間 等)

使い方:
    python3 predict_reports.py --bert_type UTH \
        --bert_path ../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K \
        --model_path ./models/NER/UTH/NER/ner_model_0.pt \
        --tag2idx ./results/NER/UTH/NER/NER_tag2idx_0.json \
        --data_dir "../predicted data" \
        --columns "【所見】" "【診断/所見のまとめ】" \
        --out_dir ./results/predict/UTH
"""
import os
import re
import json
import glob
import time
import argparse
from collections import Counter, defaultdict

import pandas as pd
import jaconv
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from lib.models import BERT_CRF
from lib.align_tokenize import OffsetTokenizer

# 半角/全角のASCII英数字(漢字・かなは含めない)
_ASCII_ALNUM = re.compile(r"^[0-9A-Za-z０-９Ａ-Ｚａ-ｚ]+$")

MAX_TOKENS = 510


class _Args:  # BERT_CRF は hyper.bert_path を参照する
    pass


def load_model(bert_path, model_path, tag2idx, device, no_finetune=False, seed=1478754):
    a = _Args()
    a.bert_path = bert_path
    if no_finetune:
        # 事前学習済みBERT本体 + 未学習(ランダム初期化)のNERヘッド。
        # ファインチューニングの効果を示すための下限ベースライン。
        torch.manual_seed(seed)
        model = BERT_CRF(a, tag2idx).to(device)
    else:
        model = BERT_CRF(a, tag2idx).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = torch.nn.DataParallel(model)
    model.eval()
    return model


def chunk_tokens(two, max_tokens=MAX_TOKENS):
    """(token,start,end) 列を max_tokens 以下のチャンクに分割"""
    for i in range(0, len(two), max_tokens):
        yield two[i:i + max_tokens]


def decode_entities(chunk, tags, doc_id, year, column, text):
    """IOB → エンティティレコード(元テキストの文字オフセット付き)。
    text は元の列テキスト(オフセットの参照元)。"""
    # 1) B/I 連続からエンティティのトークン索引区間を得る
    spans = []  # (label, i0, i1)  i1は排他的
    cur, i0 = None, None
    for i, lab in enumerate(tags):
        if lab.startswith("B-"):
            if cur is not None:
                spans.append((cur, i0, i))
            cur, i0 = lab[2:], i
        elif lab.startswith("I-") and cur == lab[2:]:
            continue
        else:
            if cur is not None:
                spans.append((cur, i0, i))
            cur, i0 = None, None
    if cur is not None:
        spans.append((cur, i0, len(tags)))

    n = len(chunk)

    def orig(idx):
        return text[chunk[idx][1]:chunk[idx][2]]

    def is_hash(idx):
        return 0 <= idx < n and chunk[idx][0].startswith("##")

    def is_alnum(idx):
        return 0 <= idx < n and bool(_ASCII_ALNUM.match(orig(idx)))

    def touches(a, b):  # a,b が隣接し文字ギャップが無いか(同一語)
        return chunk[a][2] == chunk[b][1]

    def surface(idx):
        tok = chunk[idx][0]
        if tok == "[UNK]":  # 語彙外(NICTの医療漢字/記号等)は元文字を全角化して復元
            return jaconv.h2z(orig(idx), kana=True, digit=True, ascii=True)
        return tok[2:] if tok.startswith("##") else tok

    # 2) 単語境界スナップ: エンティティ境界が語の途中で切れている場合、
    #    O ラベルの継続片(## もしくは 隣接ASCII英数字)を語末/語頭まで取り込む。
    #    漢字・かな・空白・記号は取り込まない(過剰拡張を防ぐ)。
    def extendable_fwd(j):   # j-1 の末尾から j へ伸ばせるか
        return tags[j] == "O" and (is_hash(j) or
               (touches(j - 1, j) and is_alnum(j) and is_alnum(j - 1)))

    def extendable_bwd(j):   # j の先頭から j-1 へ遡れるか
        return tags[j - 1] == "O" and (is_hash(j) or
               (touches(j - 1, j) and is_alnum(j) and is_alnum(j - 1)))

    ents = []
    for label, i0, i1 in spans:
        while i1 < n and extendable_fwd(i1):
            i1 += 1
        while i0 - 1 >= 0 and extendable_bwd(i0):
            i0 -= 1
        toks = [surface(j) for j in range(i0, i1)]
        ents.append({"doc_id": doc_id, "year": year, "column": column,
                     "label": label, "text": "".join(toks),
                     "char_start": chunk[i0][1], "char_end": chunk[i1 - 1][2]})
    return ents


def predict_file(path, tok, model, tag2idx, columns, device, batch_size):
    idx2tag = {v: k for k, v in tag2idx.items()}
    year = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, encoding="cp932")
    all_ents, n_docs, n_empty = [], 0, 0
    # まず全チャンクをトークン化(doc/col対応を保持)
    chunks, meta = [], []
    for row_i, row in df.iterrows():
        doc_id = f"{year}#{row_i}"
        for col in columns:
            if col not in df.columns:
                continue
            text = row[col]
            if not isinstance(text, str) or not text.strip():
                continue
            n_docs += 1
            two = tok.tokenize_with_offsets(text)
            for ck in chunk_tokens(two):
                if ck:
                    chunks.append(ck)
                    meta.append((doc_id, col, text))
    # バッチ推論
    with torch.inference_mode():
        for ofs in tqdm(range(0, len(chunks), batch_size),
                        desc=f"{year}", leave=False):
            batch = chunks[ofs:ofs + batch_size]
            seqs = []
            for ck in batch:
                ids = tok.vocab.get("[CLS]", 2)
                sep = tok.vocab.get("[SEP]", 3)
                wp = [tok.vocab.get(t, tok.vocab.get("[UNK]", 1)) for t, _, _ in ck]
                seqs.append(torch.tensor([ids] + wp + [sep]))
            sentence = pad_sequence(seqs, padding_value=0, batch_first=True).to(device)
            logits = model(sentence)  # [B, L-2, tags] (先頭CLS+末尾1列を除去)
            # emissions長 = L_max-2。各系列の有効長は len(ck)(実トークン数)。
            Lm2 = logits.shape[1]
            mask = torch.zeros((len(batch), Lm2), dtype=torch.bool, device=device)
            for j, ck in enumerate(batch):
                mask[j, :len(ck)] = 1
            preds = model.module.crf.decode(logits, mask=mask)
            for j, ck in enumerate(batch):
                tags = [idx2tag[t] for t in preds[j][:len(ck)]]
                doc_id, col, ctext = meta[ofs + j]
                ents = decode_entities(ck, tags, doc_id, year, col, ctext)
                if not ents:
                    n_empty += 1
                all_ents.extend(ents)
    return all_ents, n_docs, n_empty


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bert_type", required=True, choices=["UTH", "NICT"])
    p.add_argument("--bert_path", required=True)
    p.add_argument("--model_path", default=None,
                   help="ファインチューニング済みモデル(.pt)。--no_finetune 時は不要")
    p.add_argument("--tag2idx", required=True)
    p.add_argument("--no_finetune", action="store_true",
                   help="学習済みヘッドを読まず、ランダム初期化ヘッドで予測(下限ベースライン)")
    p.add_argument("--data_dir", default="../predicted data")
    p.add_argument("--columns", nargs="+", default=["【所見】", "【診断/所見のまとめ】"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--neologd_dic", default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    p.add_argument("--manbyo_dic", default="./resources/MANBYO_201907_Dic-utf8.dic")
    p.add_argument("--juman_dic", default="/var/lib/mecab/dic/juman-utf8")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tag2idx = json.load(open(args.tag2idx))
    if args.bert_type == "UTH":
        tok = OffsetTokenizer("UTH", os.path.join(args.bert_path, "vocab.txt"),
                              neologd_dic=args.neologd_dic, manbyo_dic=args.manbyo_dic)
    else:
        tok = OffsetTokenizer("NICT", os.path.join(args.bert_path, "vocab.txt"),
                              juman_dic=args.juman_dic)
    model = load_model(args.bert_path, args.model_path, tag2idx, device,
                       no_finetune=args.no_finetune)

    t0 = time.time()
    all_ents, tot_docs, tot_empty = [], 0, 0
    per_year_docs = {}
    for f in sorted(glob.glob(os.path.join(args.data_dir, "*.csv"))):
        ents, nd, ne = predict_file(f, tok, model, tag2idx, args.columns,
                                    device, args.batch_size)
        all_ents.extend(ents)
        tot_docs += nd
        tot_empty += ne
        per_year_docs[os.path.splitext(os.path.basename(f))[0]] = nd
        print(f"{os.path.basename(f)}: {nd} texts, {len(ents)} entities")

    ent_df = pd.DataFrame(all_ents)
    ent_df.to_csv(os.path.join(args.out_dir, "entities.csv"), index=False)

    # 統計
    stats = {
        "bert_type": args.bert_type,
        "no_finetune": args.no_finetune,
        "columns": args.columns,
        "n_texts": tot_docs,
        "n_entities": len(all_ents),
        "entities_per_text": round(len(all_ents) / max(tot_docs, 1), 2),
        "n_empty_chunks": tot_empty,
        "elapsed_sec": round(time.time() - t0, 1),
        "per_label": dict(Counter(e["label"] for e in all_ents).most_common()),
        "per_column": dict(Counter(e["column"] for e in all_ents)),
        "per_year_texts": per_year_docs,
        "per_year_entities": dict(Counter(e["year"] for e in all_ents)),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[{args.bert_type}] {len(all_ents)} entities from {tot_docs} texts "
          f"in {stats['elapsed_sec']}s → {args.out_dir}")


if __name__ == "__main__":
    main()
