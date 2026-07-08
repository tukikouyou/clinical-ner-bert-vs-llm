# -*- coding: utf-8 -*-
"""
端到端の構造化抽出パイプライン:
    生テキスト → 前処理 → MeCab+WordPiece → BERT-CRF NER → 構造化エンティティ(JSON/CSV)

使い方:
    python3 extract_structured.py --input 入力.txt --output_dir ./results/extraction/run1
    # 入力は「1行=1レポート」のtxt、または .txt を含むディレクトリ(1ファイル=1レポート)

特徴:
    - 510トークン超の文書を捨てずに、文境界(。)で自動分割して全文を処理する
    - IOBタグ列をエンティティ単位(文書名/タグ/表層/トークン位置)にマージして
      entities.json / entities.csv を出力する
"""
import os
import sys
import json
import argparse
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from preprocess_text import preprocess
from tokenization_mod import MecabTokenizer, FullTokenizerForMecab
from lib.models import BERT_CRF
from lib.util import no_tag_make_test_vecs_pipeline, decode_ner_pipeline

MAX_TOKENS = 510  # [CLS]/[SEP] を除く1チャンクの上限


def build_tokenizer(vocab_file, neologd_dic, manbyo_dic):
    sub_tokenizer = MecabTokenizer(mecab_ipadic_neologd=neologd_dic,
                                   mecab_J_medic=manbyo_dic,
                                   name_token="＠＠Ｎ")
    return FullTokenizerForMecab(sub_tokenizer=sub_tokenizer,
                                 vocab_file=vocab_file,
                                 do_lower_case=False)


def load_reports(input_path):
    """(doc_id, 生テキスト) のリストを返す"""
    reports = []
    if os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(input_path, fname), encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                reports.append((os.path.splitext(fname)[0], text))
    else:
        with open(input_path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    reports.append((f"doc{i:04d}", line))
    return reports


def split_sentences(text):
    """。で文分割(。は前の文に残す)"""
    sents, buf = [], ""
    for ch in text:
        buf += ch
        if ch == "。":
            sents.append(buf)
            buf = ""
    if buf:
        sents.append(buf)
    return sents


def tokenize_reports(reports, tokenizer):
    """トークン化し、MAX_TOKENS以下のチャンクへ分割した DataFrame を返す"""
    rows = []
    for doc_id, raw in tqdm(reports, desc="Tokenizing"):
        text = preprocess(raw.replace("\n", ""))
        chunk_idx, chunk_tokens = 0, []
        for sent in split_sentences(text):
            tokens = tokenizer.tokenize(sent)
            if not tokens:
                continue
            # 1文がそれ自体で上限を超える場合は強制分割
            while len(tokens) > MAX_TOKENS:
                if chunk_tokens:
                    rows += [(doc_id, f"{doc_id}_{chunk_idx}", t) for t in chunk_tokens]
                    chunk_idx, chunk_tokens = chunk_idx + 1, []
                rows += [(doc_id, f"{doc_id}_{chunk_idx}", t) for t in tokens[:MAX_TOKENS]]
                chunk_idx += 1
                tokens = tokens[MAX_TOKENS:]
            if len(chunk_tokens) + len(tokens) > MAX_TOKENS:
                rows += [(doc_id, f"{doc_id}_{chunk_idx}", t) for t in chunk_tokens]
                chunk_idx, chunk_tokens = chunk_idx + 1, []
            chunk_tokens += tokens
        if chunk_tokens:
            rows += [(doc_id, f"{doc_id}_{chunk_idx}", t) for t in chunk_tokens]
    return pd.DataFrame(rows, columns=["name", "unique_no", "word"])


def predict(df, tokenizer, tag2idx, args, device):
    test_vecs = no_tag_make_test_vecs_pipeline(df, tokenizer)
    with torch.inference_mode():
        model = BERT_CRF(args, tag2idx).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = torch.nn.DataParallel(model)
        model.eval()
        ner_preds = []
        for ofs in tqdm(range(0, len(test_vecs), args.batch_size), desc="Predicting NER"):
            batch_X = test_vecs[ofs: ofs + args.batch_size]
            sentence = pad_sequence([torch.tensor(x) for x in batch_X],
                                    padding_value=0, batch_first=True).to(device)
            ner_preds.append(model(sentence))
    pred_tags = decode_ner_pipeline(model, ner_preds, tag2idx, test_vecs)
    list_df = []
    for i, no in enumerate(df["unique_no"].unique()):
        tmp = df[df["unique_no"] == no].copy()
        tmp["pred_IOB"] = pred_tags[i]
        list_df.append(tmp)
    return pd.concat(list_df)


def join_tokens(tokens):
    """WordPiece(##)を復元して表層文字列にする"""
    out = ""
    for t in tokens:
        out += t[2:] if t.startswith("##") else t
    return out


def iob_to_entities(res_df):
    """トークン単位のIOB列 → エンティティ単位のレコード"""
    entities = []
    for no in res_df["unique_no"].unique():
        tmp = res_df[res_df["unique_no"] == no]
        doc = tmp["name"].iloc[0]
        words = list(tmp["word"])
        labels = list(tmp["pred_IOB"])
        cur_label, cur_start = None, None
        for i, lab in enumerate(labels + ["O"]):  # 番兵で末尾エンティティを確定
            if cur_label is not None and not (lab == f"I-{cur_label}"):
                entities.append({
                    "doc": doc,
                    "chunk": no,
                    "label": cur_label,
                    "text": join_tokens(words[cur_start:i]),
                    "start_token": cur_start,
                    "end_token": i - 1,
                })
                cur_label, cur_start = None, None
            if lab.startswith("B-"):
                cur_label, cur_start = lab[2:], i
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help="1行=1レポートのtxt、または.txtディレクトリ")
    parser.add_argument('--output_dir', type=str, default="./results/extraction/run1")
    parser.add_argument('--bert_path', type=str, default='../model',
                        help="UTH-BERTのディレクトリ(config.json/pytorch_model.bin/vocab.txt)")
    parser.add_argument('--model_path', type=str, default="./models/NER/UTH/NER/ner_model_0.pt")
    parser.add_argument('--tag2idx_path', type=str, default="./results/NER/UTH/NER/NER_tag2idx_0.json")
    parser.add_argument('--neologd_dic', type=str,
                        default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    parser.add_argument('--manbyo_dic', type=str, default='./resources/MANBYO_201907_Dic-utf8.dic')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    vocab_file = os.path.join(args.bert_path, "vocab.txt")
    tokenizer = build_tokenizer(vocab_file, args.neologd_dic, args.manbyo_dic)
    tag2idx = json.load(open(args.tag2idx_path))

    reports = load_reports(args.input)
    print(f"{len(reports)} 件のレポートを読み込みました")
    df = tokenize_reports(reports, tokenizer)
    print(f"{df['unique_no'].nunique()} チャンク / {len(df)} トークン")

    res_df = predict(df, tokenizer, tag2idx, args, device)
    entities = iob_to_entities(res_df)

    res_df.to_csv(os.path.join(args.output_dir, "token_predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "entities.json"), "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    pd.DataFrame(entities).to_csv(os.path.join(args.output_dir, "entities.csv"), index=False)
    print(f"抽出エンティティ数: {len(entities)}")
    print(f"保存先: {args.output_dir}/entities.json, entities.csv, token_predictions.csv")


if __name__ == "__main__":
    main()
