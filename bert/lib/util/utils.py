import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import json, os
import collections
from transformers import BertModel, BertTokenizer
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as ner_eval
from tqdm import tqdm


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def load_data_cv(args):
    df = pd.read_csv(args.data_path)
    df["rel_type"] = df["rel_type"].fillna("None").astype(str)

    # 按 unique_no 计算每个文本的长度
    text_lengths = df.groupby("unique_no")["word"].apply(lambda x: x.str.len().sum())

    # 过滤掉长度超过 510 的文本
    valid_unique_nos = text_lengths[text_lengths <= 510].index
    df = df[df["unique_no"].isin(valid_unique_nos)]

    return df

    #all_unique_nos = json.load(open('./data/index/all_unique_nos.json', 'r'))["unique_no"]
    #return df.query('unique_no in @all_unique_nos')


def load_tokenizer(args):
    return BertTokenizer(Path(args.bert_path) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)


def train_val_split_doc(X_train, args):
    # 訓練データのid一覧
    ids = list(sorted(X_train["name"].unique()))
    # シャッフル
    random.seed(args.seed)
    random.shuffle(ids)
    # 分割
    train_ids, val_ids = ids[:int(len(ids) * 0.8)], ids[int(len(ids) * 0.8):]
    # 確認
    assert len(set(ids[:int(len(ids) * 0.8)]) & set((ids[int(len(ids) * 0.8):])))==0, "trainとvalが重複してますよ"
    # 分割
    X_train, X_val = X_train[X_train["name"].isin(train_ids)].copy(), X_train[X_train["name"].isin(val_ids)].copy()
    return X_train, X_val


def make_idx(df, args):
    if args.idx_flag == "T":
        # タグ
        tag_vocab = list(sorted(set([x for x in df["IOB"]])))
        tag2idx = {x: i + 2 for i, x in enumerate(tag_vocab)}
        tag2idx["PAD"] = 0
        tag2idx["UNK"] = 1
        # 関係
        df["rel_type"] = df["rel_type"].fillna("None").astype(str)  # NaNを置き換えて文字列化
        rel_vocab = list(sorted(set([y for x in df["rel_type"] for y in x.split(",")])))
        rel2idx = {x: i + 2 for i, x in enumerate(rel_vocab)}
        rel2idx["PAD"] = 0
        rel2idx["UNK"] = 1
    else:
        tag_vocab = list(sorted(set([x for x in df["IOB"]])))
        tag2idx = {x: i + 1 for i, x in enumerate(tag_vocab)}
        tag2idx["PAD"] = 0
        df["rel_type"] = df["rel_type"].fillna("None").astype(str)  # NaNを置き換えて文字列化
        rel_vocab = list(sorted(set([y for x in df["rel_type"] for y in x.split(",")])))
        rel2idx = {}
        last_value = 1
        for _, x in enumerate(rel_vocab):
            if x == "None":
                pass
            else:
                rel2idx["R-" + x] = last_value
                rel2idx["L-" + x] = last_value + 1
                last_value +=2
        rel2idx["PAD"] = 0
        rel2idx["None"] = max(list(rel2idx.values())) + 1
    return tag2idx, rel2idx


def no_tag_save_ner_result(args, fold, tag2idx):
    with open('./results/{0}/{1}/NER/NER_tag2idx_{2}.json'.format(args.task, args.bert_type, fold), 'w') as f:
        json.dump(tag2idx, f, indent=4, cls=NpEncoder)


def result2df_for_ner(X_test, ner_preds_decode):
    list_df = []
    for i, idx in enumerate(X_test["unique_no"].unique()):
        # DataFrame
        tmp_df = X_test[X_test["unique_no"]==idx].copy()
        # NERを代入
        tmp_df["pred_IOB"] = ner_preds_decode[i]
        list_df.append(tmp_df)
    return pd.concat(list_df)


def make_train_vecs_pipeline(df, tokenizer, tag2idx):
    vecs1, vecs2 = [], []
    # テキスト
    for no in df["unique_no"].unique():
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        # NER
        ner = [tag2idx[x] for x in list(tmp_df["IOB"])]
        # REL
        # ADD
        vecs1.append(ids)
        vecs2.append(ner)
    return vecs1, vecs2


def make_test_vecs_pipeline(df, tokenizer, tag2idx, exp_type):
    vecs1, vecs2 = [], []
    # テキスト
    for no in df["unique_no"].unique():
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        # NER
        if exp_type == "NER":
            ner = [tag2idx[x] if x in tag2idx else tag2idx["UNK"] for x in list(tmp_df["IOB"])]
        else:
            ner = [tag2idx[x] if x in tag2idx else tag2idx["UNK"] for x in list(tmp_df["pred_IOB"])]
        # REL
        # ADD
        vecs1.append(ids)
        vecs2.append(ner)
    return vecs1, vecs2

def no_tag_make_test_vecs_pipeline(df, tokenizer):
    vecs1= []
    # テキスト
    print("Converting test data to vectors...")
    for no in tqdm(df["unique_no"].unique(), desc="Tokenizing"):
        # 取得
        tmp_df = df[df["unique_no"] == no]
        # 単語ベクトル
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + list(tmp_df["word"]) + ["[SEP]"])
        vecs1.append(ids)
    return vecs1

def decode_ner_pipeline(model, preds, tag2idx, vecs):
    idx2tag = {v: k for k, v in tag2idx.items()}
    pred_tags = []
    index = 0
    for predx in preds:
        predx = model.module.crf.decode(predx)
        for pred in predx:
            pred_tags.append([idx2tag[pred] for pred in pred[:len(vecs[index])-2]])
            index += 1
    return pred_tags


def save_csv_pipeline(fold_res_df, args, fold, name):
    fold_res_df.to_csv('./results/{0}/{1}/{3}/{2}.csv'.format(args.task, args.bert_type, fold, name), index=False)
