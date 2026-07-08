import os, sys, random, logging
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.utils.data

from lib.util import (load_data_cv, load_tokenizer, train_val_split_doc, make_idx,
                      make_train_vecs_pipeline, make_test_vecs_pipeline,
                      save_csv_pipeline, no_tag_make_test_vecs_pipeline,
                      no_tag_save_ner_result)
from lib.util import eval_ner_strict, eval_ner_soft
from lib.util.utils import NpEncoder
from lib.loop import train_val_loop_ner, no_tag_test_loop_ner


def main():
    logger.info("----------{0}-NERの実験を開始----------".format(args.bert_type))
    # 実験データ
    df = load_data_cv(args)
    # 交差検証用の分割ファイル
    train_test_indxe_dct = json.load(open(args.split_file, "r"))
    fold = 0
    logger.info("データのロード開始")
    train_index = train_test_indxe_dct["train_name"]
    test_index = train_test_indxe_dct["test_name"]
    X_train = df.query('name in @train_index')
    X_test = df.query('name in @test_index')
    # トーカナイザー
    bert_tokenizer = load_tokenizer(args)
    # 検証データを訓練から分離
    X_train, X_val = train_val_split_doc(X_train, args)
    # タグ辞書
    tag2idx, _ = make_idx(pd.concat([X_train, X_val, X_test]), args)
    # ベクトル化
    train_vecs, ner_train_labels = make_train_vecs_pipeline(X_train, bert_tokenizer, tag2idx)
    val_vecs, ner_val_labels = make_test_vecs_pipeline(X_val, bert_tokenizer, tag2idx, "NER")
    test_vecs = no_tag_make_test_vecs_pipeline(X_test, bert_tokenizer)
    logger.info("train: {0}, val: {1}, test: {2} (sentences)".format(
        len(train_vecs), len(val_vecs), len(test_vecs)))
    logger.info("train docs: {0}, test docs: {1}, tags: {2}".format(
        X_train["name"].nunique() + X_val["name"].nunique(),
        X_test["name"].nunique(), len(tag2idx)))

    # 学習(早停あり)
    loss_dct = train_val_loop_ner(train_vecs, ner_train_labels,
                                  X_val, val_vecs, ner_val_labels,
                                  tag2idx, fold, args, device, logger)

    # テスト予測(ベストモデルを読み込んで予測)
    t0 = time.time()
    res_df = no_tag_test_loop_ner(X_test, test_vecs, fold, tag2idx, args, device)
    predict_sec = round(time.time() - t0, 1)

    # 保存(tag2idx / 予測CSV)
    no_tag_save_ner_result(args, fold, tag2idx)
    save_csv_pipeline(res_df, args, fold, "NER")

    # テスト評価(strict / soft, タグ別 + micro/macro)
    strict = eval_ner_strict(res_df)
    soft = eval_ner_soft(res_df)
    metrics = {
        "bert_type": args.bert_type,
        "data_path": args.data_path,
        "best_epoch": loss_dct.get("best_epoch"),
        "best_val_F": loss_dct.get("best_val_F"),
        "stopped_early": loss_dct.get("stopped_early"),
        "train_total_sec": loss_dct.get("total_sec"),
        "predict_sec": predict_sec,
        "n_train_sent": len(train_vecs), "n_val_sent": len(val_vecs),
        "n_test_sent": len(test_vecs), "n_tags": len(tag2idx),
        "test_strict_micro": strict["micro avg"],
        "test_strict_macro": strict["macro avg"],
        "test_soft_micro": soft["micro avg"],
        "test_soft_macro": soft["macro avg"],
    }
    out_dir = './results/{0}/{1}/NER'.format(args.task, args.bert_type)
    with open(os.path.join(out_dir, 'NER_strict_RESULT_{0}.json'.format(fold)), 'w') as f:
        json.dump(strict, f, indent=4, ensure_ascii=False, cls=NpEncoder)
    with open(os.path.join(out_dir, 'NER_soft_RESULT_{0}.json'.format(fold)), 'w') as f:
        json.dump(soft, f, indent=4, ensure_ascii=False, cls=NpEncoder)
    with open(os.path.join(out_dir, 'metrics_{0}.json'.format(fold)), 'w') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False, cls=NpEncoder)

    logger.info("=== テスト結果 ({0}) ===".format(args.bert_type))
    logger.info("strict micro F1: {0:.4f} / soft micro F1: {1:.4f}".format(
        strict["micro avg"]["f1"], soft["micro avg"]["f1"]))
    logger.info("best epoch: {0}, 学習時間: {1}s, 予測時間: {2}s".format(
        loss_dct.get("best_epoch"), loss_dct.get("total_sec"), predict_sec))
    logger.info("完成")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='../UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K')
    parser.add_argument('--bert_type', type=str, default='UTH')
    parser.add_argument('--data_path', type=str, default="./data/csv/icorpus_UTH.csv")
    parser.add_argument('--split_file', type=str, default="./data/index/train_test_index_1fold.json")
    parser.add_argument('--max_epoch', type=int, default=250)
    parser.add_argument('--patience', type=int, default=10, help="早停: 検証F1がこの回数改善しなければ停止(0で無効)")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_words', type=int, default=512)
    parser.add_argument('--task', type=str, default="NER")
    parser.add_argument('--idx_flag', type=str, default="F")
    parser.add_argument('--seed', type=int, default=1478754)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('use cuda device')
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        print('use cpu')
        device = torch.device('cpu')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    for SAMPLE_DIR in ["./models/{0}/{1}/NER".format(args.task, args.bert_type),
                       "./results/{0}/{1}/NER".format(args.task, args.bert_type),
                       "./logs"]:
        os.makedirs(SAMPLE_DIR, exist_ok=True)
    logger = logging.getLogger('LoggingTest')
    logger.setLevel(10)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    fh = logging.FileHandler('./logs/Pipeline_NER_{0}.log'.format(args.bert_type), "w")
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    main()
